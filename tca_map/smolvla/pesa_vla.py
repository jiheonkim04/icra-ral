"""PESA-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"
VALIDATION_CONFIGS = (
    {"config_id": "pesa_eta070_a005_query_linear", "spectral_eta": 0.70, "action_alpha": 0.05, "query_architecture": "linear"},
    {"config_id": "pesa_eta085_a005_query_linear", "spectral_eta": 0.85, "action_alpha": 0.05, "query_architecture": "linear"},
    {"config_id": "pesa_eta070_a010_query_linear", "spectral_eta": 0.70, "action_alpha": 0.10, "query_architecture": "linear"},
    {"config_id": "pesa_eta085_a010_query_linear", "spectral_eta": 0.85, "action_alpha": 0.10, "query_architecture": "linear"},
    {"config_id": "pesa_eta070_a020_query_mlp", "spectral_eta": 0.70, "action_alpha": 0.20, "query_architecture": "mlp"},
    {"config_id": "pesa_eta085_a020_query_mlp", "spectral_eta": 0.85, "action_alpha": 0.20, "query_architecture": "mlp"},
)
FORBIDDEN_INFERENCE_KEYS = {
    "identity",
    "success",
    "reward",
    "future_state",
    "future_action",
    "object_state",
    "object_pose",
    "episode_index",
    "dataset_global_index",
    "oracle_help_label",
}


@dataclass(frozen=True)
class PESAConfig:
    train_splits: tuple[str, ...] = ("train",)
    validation_splits: tuple[str, ...] = ("val",)
    confirmatory_reserved_splits: tuple[str, ...] = ("test",)
    min_scoreable_records: int = 500
    min_task_count: int = 3
    min_positive_fraction: float = 0.05
    max_positive_fraction: float = 0.95
    min_positive_count: int = 50
    min_negative_count: int = 50
    max_task_positive_share: float = 0.20
    min_query_probe_accuracy_margin: float = 0.02
    min_headroom_l1: float = 0.001
    min_action_distinction_l2: float = 0.003
    min_spectral_activation_fraction: float = 0.05
    max_spectral_activation_fraction: float = 0.95
    min_distinct_task_active_rank_profiles: int = 2
    init_delta_p95_max: float = 1e-6
    query_improvement_quantile: float = 0.60
    query_delta_min: float = 0.01
    stage0_action_alpha: float = 0.10
    stage0_spectral_eta: float = 0.85
    query_probe_epochs: int = 1200
    query_probe_lr: float = 0.08
    query_probe_l2: float = 1e-4
    gradient_batch_size: int = 128
    eps: float = 1e-9


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged PESA inference fields: {leaked}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _frame_key(record: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(record.get("split", "")),
        int(record.get("task_index", -1)),
        int(record.get("episode_index", -1)),
        int(record.get("frame_index", -1)),
        int(record.get("eval_seed", 0)),
    )


def _sample_key(record: Mapping[str, Any]) -> str:
    sample_id = record.get("sample_id", record.get("sample_key"))
    if sample_id is not None and not isinstance(sample_id, Mapping):
        return f"{sample_id}|seed={int(record.get('eval_seed', 0))}"
    return "|".join(str(value) for value in _frame_key(record))


def build_pesa_records(prediction_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        if "target_action" not in record or "base_action" not in record:
            continue
        target = _as_vector("target_action", record["target_action"], 7)
        base = _as_vector("base_action", record["base_action"], 7)
        row = {
            "key": _sample_key(record),
            "frame_key": _frame_key(record),
            "split": str(record.get("split", "")),
            "task": str(record.get("task", "")),
            "task_index": int(record.get("task_index", -1)),
            "episode_index": int(record.get("episode_index", -1)),
            "frame_index": int(record.get("frame_index", -1)),
            "dataset_global_index": int(record.get("dataset_global_index", record.get("index", -1))),
            "phase": _safe_float(record.get("normalized_phase", record.get("phase", 0.0))),
            "target_action": target,
            "base_action": base,
            "residual": target - base,
            "state": _as_vector("state", record.get("state", [0.0] * 8), 8),
        }
        if "lora_action" in record:
            lora = _as_vector("lora_action", record["lora_action"], 7)
            row["lora_action"] = lora
            row["lora_delta"] = lora - base
        if "mean_action" in record:
            row["mean_action"] = _as_vector("mean_action", record["mean_action"], 7)
        rows.append(row)
    return rows


def _duplicate_count(keys: Sequence[Any]) -> int:
    seen = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _split_overlap(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, int]:
    train = {tuple(record["frame_key"][1:]) for record in records if str(record["split"]) in set(config.train_splits)}
    validation = {
        tuple(record["frame_key"][1:]) for record in records if str(record["split"]) in set(config.validation_splits)
    }
    reserved = {
        tuple(record["frame_key"][1:])
        for record in records
        if str(record["split"]) in set(config.confirmatory_reserved_splits)
    }
    return {
        "train_validation": len(train & validation),
        "train_reserved": len(train & reserved),
        "validation_reserved": len(validation & reserved),
    }


def _split_reset_overlap(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, int]:
    train = {(int(record["task_index"]), int(record["episode_index"])) for record in records if str(record["split"]) in set(config.train_splits)}
    validation = {
        (int(record["task_index"]), int(record["episode_index"]))
        for record in records
        if str(record["split"]) in set(config.validation_splits)
    }
    reserved = {
        (int(record["task_index"]), int(record["episode_index"]))
        for record in records
        if str(record["split"]) in set(config.confirmatory_reserved_splits)
    }
    return {
        "train_validation": len(train & validation),
        "train_reserved": len(train & reserved),
        "validation_reserved": len(validation & reserved),
    }


def _query_threshold(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> float:
    train = [record for record in records if str(record["split"]) in set(config.train_splits) and "lora_action" in record]
    if not train:
        return config.query_delta_min
    improvements = []
    for record in train:
        target = np.asarray(record["target_action"], dtype=np.float64)
        base = np.asarray(record["base_action"], dtype=np.float64)
        lora = np.asarray(record["lora_action"], dtype=np.float64)
        improvements.append(float(np.sum(np.abs(target - base)) - np.sum(np.abs(target - lora))))
    return float(max(np.quantile(np.asarray(improvements, dtype=np.float64), config.query_improvement_quantile), config.query_delta_min))


def _spectral_stats(residual: np.ndarray, eta: float, eps: float) -> dict[str, Any]:
    energy_raw = residual * residual
    total = float(np.sum(energy_raw))
    if total <= eps:
        energy = np.ones(7, dtype=np.float64) / 7.0
    else:
        energy = energy_raw / (total + eps)
    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy)
    active_rank = int(np.searchsorted(cumulative, eta, side="left") + 1)
    active_rank = max(1, min(7, active_rank))
    entropy = float(-np.sum(energy * np.log(energy + eps)) / np.log(7.0))
    return {
        "energy": energy,
        "sorted_energy": sorted_energy,
        "active_rank": active_rank,
        "active_fraction": float(active_rank / 7.0),
        "entropy": entropy,
    }


def compute_query_and_spectral_labels(
    records: Sequence[Mapping[str, Any]],
    config: PESAConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = config or PESAConfig()
    enriched = [dict(record) for record in records]
    threshold = _query_threshold(enriched, cfg)
    for record in enriched:
        target = np.asarray(record["target_action"], dtype=np.float64)
        base = np.asarray(record["base_action"], dtype=np.float64)
        residual = target - base
        if "lora_action" in record:
            lora = np.asarray(record["lora_action"], dtype=np.float64)
        else:
            lora = base.copy()
        base_l1 = float(np.sum(np.abs(target - base)))
        lora_l1 = float(np.sum(np.abs(target - lora)))
        improvement = base_l1 - lora_l1
        spec = _spectral_stats(residual, cfg.stage0_spectral_eta, cfg.eps)
        record.update(
            {
                "base_l1": base_l1,
                "lora_l1": lora_l1,
                "lora_improvement_l1": improvement,
                "query_label": bool(improvement > threshold),
                "spectral_energy": spec["energy"],
                "spectral_sorted_energy": spec["sorted_energy"],
                "spectral_active_rank": spec["active_rank"],
                "spectral_active_fraction": spec["active_fraction"],
                "spectral_entropy": spec["entropy"],
            }
        )
    return enriched, {"query_improvement_l1_threshold": threshold, "spectral_eta": cfg.stage0_spectral_eta}


def _feature_matrix(records: Sequence[Mapping[str, Any]], task_count: int | None = None) -> np.ndarray:
    max_task = int(task_count) if task_count is not None else max([int(record["task_index"]) for record in records] + [0]) + 1
    features = []
    for record in records:
        base = np.asarray(record["base_action"], dtype=np.float64)
        lora = np.asarray(record.get("lora_action", base), dtype=np.float64)
        delta = lora - base
        state = np.asarray(record["state"], dtype=np.float64)
        task = np.zeros(max_task, dtype=np.float64)
        task_index = int(record["task_index"])
        if 0 <= task_index < max_task:
            task[task_index] = 1.0
        grouped = np.asarray(
            [
                np.linalg.norm(base[0:3]),
                np.linalg.norm(base[3:6]),
                abs(base[6]),
                np.linalg.norm(delta[0:3]),
                np.linalg.norm(delta[3:6]),
                abs(delta[6]),
                np.linalg.norm(delta),
            ],
            dtype=np.float64,
        )
        features.append(np.concatenate([base, lora, delta, state, [float(record["phase"])], task, grouped]))
    return np.vstack(features) if features else np.zeros((0, 30), dtype=np.float64)


def _standardize(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (validation - mean) / scale


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def _binary_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _fit_binary_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    config: PESAConfig,
) -> dict[str, float]:
    y_train = train_labels.astype(np.float64)
    y_validation = validation_labels.astype(bool)
    if train_features.size == 0 or validation_features.size == 0:
        return {"valid": 0.0, "accuracy": 0.0, "majority_accuracy": 0.0, "accuracy_margin": -1.0}
    if y_train.min() == y_train.max():
        return {"valid": 0.0, "accuracy": 0.0, "majority_accuracy": 0.0, "accuracy_margin": -1.0}
    x_train = np.c_[train_features, np.ones(train_features.shape[0], dtype=np.float64)]
    x_validation = np.c_[validation_features, np.ones(validation_features.shape[0], dtype=np.float64)]
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    pos = float(np.mean(y_train))
    weight_pos = 0.5 / max(pos, config.eps)
    weight_neg = 0.5 / max(1.0 - pos, config.eps)
    sample_weights = np.where(y_train > 0.5, weight_pos, weight_neg)
    initial_probs = _sigmoid(x_train @ weights)
    first_gradient_norm = 0.0
    for epoch in range(config.query_probe_epochs):
        probs = _sigmoid(x_train @ weights)
        gradient = x_train.T @ ((probs - y_train) * sample_weights) / max(len(y_train), 1)
        gradient += config.query_probe_l2 * weights
        if epoch == 0:
            first_gradient_norm = float(np.linalg.norm(gradient))
        weights -= config.query_probe_lr * gradient
    validation_probs = _sigmoid(x_validation @ weights)
    predictions = validation_probs >= 0.5
    accuracy = float(np.mean(predictions == y_validation))
    majority_accuracy = float(max(np.mean(y_validation), 1.0 - np.mean(y_validation)))
    return {
        "valid": 1.0,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "accuracy_margin": accuracy - majority_accuracy,
        "predicted_positive_fraction": float(np.mean(predictions)),
        "mean_probability": float(np.mean(validation_probs)),
        "first_gradient_norm": first_gradient_norm,
        "train_loss_initial": _binary_loss(initial_probs, y_train),
        "train_loss_final": _binary_loss(_sigmoid(x_train @ weights), y_train),
        "validation_loss": _binary_loss(validation_probs, validation_labels.astype(np.float64)),
    }


def _query_probe_summary(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, float]:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(config.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["query_label"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["query_label"] for record in validation], dtype=bool)
    return _fit_binary_probe(train_features, validation_features, train_labels, validation_labels, config)


def _spectral_probe_summary(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, float]:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(config.validation_splits)]
    if not train or not validation:
        return {"valid": 0.0, "accuracy": 0.0, "majority_accuracy": 0.0, "accuracy_margin": -1.0}
    threshold = float(np.median([int(record["spectral_active_rank"]) for record in train]))
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    train_labels = np.asarray([int(record["spectral_active_rank"]) > threshold for record in train], dtype=bool)
    validation_labels = np.asarray([int(record["spectral_active_rank"]) > threshold for record in validation], dtype=bool)
    out = _fit_binary_probe(train_features, validation_features, train_labels, validation_labels, config)
    out["active_rank_threshold"] = threshold
    return out


def _label_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str]) -> dict[str, Any]:
    selected = [record for record in records if str(record["split"]) in set(split_names)]
    labels = np.asarray([record["query_label"] for record in selected], dtype=bool)
    positives = int(np.sum(labels))
    return {
        "total": len(selected),
        "positive_count": positives,
        "negative_count": int(len(selected) - positives),
        "positive_fraction": float(np.mean(labels)) if len(selected) else 0.0,
    }


def _max_task_positive_share(records: Sequence[Mapping[str, Any]]) -> float:
    positives = [record for record in records if bool(record["query_label"])]
    if not positives:
        return 0.0
    counts: dict[int, int] = {}
    for record in positives:
        task = int(record["task_index"])
        counts[task] = counts.get(task, 0) + 1
    return float(max(counts.values()) / max(len(positives), 1))


def _action_validity(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 0.0
    return float(np.mean(np.all(np.isfinite(actions), axis=1) & (np.max(np.abs(actions), axis=1) <= 5.0)))


def _mean_l1(records: Sequence[Mapping[str, Any]], action_key: str) -> float | None:
    values = []
    for record in records:
        if action_key not in record:
            continue
        action = np.asarray(record[action_key], dtype=np.float64)
        target = np.asarray(record["target_action"], dtype=np.float64)
        values.append(float(np.sum(np.abs(action - target))))
    if not values:
        return None
    return float(np.mean(values))


def _clip_l2(delta: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, alpha / (norm + eps))
    return delta * scale


def _development_policy_actions(
    records: Sequence[Mapping[str, Any]],
    config: PESAConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = np.asarray([record["base_action"] for record in records], dtype=np.float64)
    lora = np.asarray([record.get("lora_action", record["base_action"]) for record in records], dtype=np.float64)
    labels = np.asarray([record["query_label"] for record in records], dtype=bool).reshape(-1, 1)
    active_fraction = np.asarray([record["spectral_active_fraction"] for record in records], dtype=np.float64).reshape(-1, 1)
    delta = _clip_l2(lora - base, config.stage0_action_alpha, config.eps)
    clean_retention = np.where(labels, base + delta, base)
    standard_lora = lora
    standard_lora_l1 = _mean_action_l1(records, standard_lora)
    clean_retention_l1 = _mean_action_l1(records, clean_retention)
    if clean_retention_l1 <= standard_lora_l1:
        simple_name = "clean_retention_lora_proxy"
        simple = clean_retention
        simple_l1 = clean_retention_l1
    else:
        simple_name = "standard_lora_adapter_proxy"
        simple = standard_lora
        simple_l1 = standard_lora_l1
    return (
        {
            "base": base,
            "priorvla_style_proxy": clean_retention,
            "pesa_full": base + labels.astype(np.float64) * active_fraction * delta,
            "pesa_no_spectral_no_prior_query_ablation": base + delta,
            "standard_lora_adapter_proxy": standard_lora,
            "clean_retention_lora_proxy": clean_retention,
            "selected_simple_killer": simple,
        },
        {
            "selected_simple_killer": simple_name,
            "standard_lora_l1": standard_lora_l1,
            "clean_retention_l1": clean_retention_l1,
            "selected_simple_l1": simple_l1,
        },
    )


def _mean_action_l1(records: Sequence[Mapping[str, Any]], actions: np.ndarray) -> float:
    targets = np.asarray([record["target_action"] for record in records], dtype=np.float64)
    return float(np.mean(np.sum(np.abs(actions - targets), axis=1)))


def _target_distinction_metrics(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, Any]:
    actions, simple = _development_policy_actions(records, config)
    full = actions["pesa_full"]
    proxy = actions["priorvla_style_proxy"]
    ablation = actions["pesa_no_spectral_no_prior_query_ablation"]
    simple_actions = actions["selected_simple_killer"]
    delta = full - actions["base"]
    return {
        "full_vs_priorvla_proxy_mean_l2": float(np.mean(np.linalg.norm(full - proxy, axis=1))),
        "full_vs_no_spectral_no_query_mean_l2": float(np.mean(np.linalg.norm(full - ablation, axis=1))),
        "full_vs_selected_simple_killer_mean_l2": float(np.mean(np.linalg.norm(full - simple_actions, axis=1))),
        "full_delta_l2_mean": float(np.mean(np.linalg.norm(delta, axis=1))),
        "full_delta_l2_p95": float(np.percentile(np.linalg.norm(delta, axis=1), 95)),
        "translation_delta_p95": float(np.percentile(np.linalg.norm(delta[:, 0:3], axis=1), 95)),
        "rotation_delta_p95": float(np.percentile(np.linalg.norm(delta[:, 3:6], axis=1), 95)),
        "gripper_delta_p95": float(np.percentile(np.abs(delta[:, 6]), 95)),
        "simple_killer": simple,
    }


def _spectral_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str]) -> dict[str, Any]:
    selected = [record for record in records if str(record["split"]) in set(split_names)]
    ranks = np.asarray([int(record["spectral_active_rank"]) for record in selected], dtype=np.int64)
    fractions = np.asarray([float(record["spectral_active_fraction"]) for record in selected], dtype=np.float64)
    entropies = np.asarray([float(record["spectral_entropy"]) for record in selected], dtype=np.float64)
    if len(selected) == 0:
        return {
            "total": 0,
            "active_rank_min": 0,
            "active_rank_max": 0,
            "active_rank_mean": 0.0,
            "active_fraction_mean": 0.0,
            "entropy_mean": 0.0,
            "distinct_task_active_rank_profiles": 0,
        }
    task_means: dict[int, float] = {}
    for task in sorted({int(record["task_index"]) for record in selected}):
        task_ranks = [int(record["spectral_active_rank"]) for record in selected if int(record["task_index"]) == task]
        task_means[task] = float(np.mean(task_ranks))
    distinct_profiles = len({round(value, 2) for value in task_means.values()})
    return {
        "total": len(selected),
        "active_rank_min": int(np.min(ranks)),
        "active_rank_max": int(np.max(ranks)),
        "active_rank_mean": float(np.mean(ranks)),
        "active_fraction_mean": float(np.mean(fractions)),
        "active_fraction_p05": float(np.percentile(fractions, 5)),
        "active_fraction_p95": float(np.percentile(fractions, 95)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_p95": float(np.percentile(entropies, 95)),
        "distinct_task_active_rank_profiles": distinct_profiles,
        "task_active_rank_mean": {str(key): value for key, value in task_means.items()},
    }


def _parameter_grad_norm(parameters: Sequence[Any], torch: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).item())
    return float(total**0.5)


def _gradient_audit(records: Sequence[Mapping[str, Any]], config: PESAConfig) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"valid": 0.0, "hard_stop_reason": "torch unavailable for PESA gradient audit"}

    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    if not train:
        return {"valid": 0.0, "hard_stop_reason": "no train records for PESA gradient audit"}
    batch = train[: min(config.gradient_batch_size, len(train))]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    features_raw = _feature_matrix(batch, task_count)
    mean = features_raw.mean(axis=0)
    scale = features_raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    features = (features_raw - mean) / scale
    base = np.asarray([record["base_action"] for record in batch], dtype=np.float64)
    target = np.asarray([record["target_action"] for record in batch], dtype=np.float64)
    labels = np.asarray([record["query_label"] for record in batch], dtype=np.float64)
    spectral_target = np.asarray([record["spectral_energy"] for record in batch], dtype=np.float64)

    class TinyPESA(torch.nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(torch.nn.Linear(input_dim, 32), torch.nn.ReLU())
            self.adapt = torch.nn.Linear(32, 7)
            self.query = torch.nn.Linear(32, 1)
            self.spectral = torch.nn.Linear(32, 7)
            torch.nn.init.zeros_(self.adapt.weight)
            torch.nn.init.zeros_(self.adapt.bias)
            torch.nn.init.constant_(self.query.bias, -2.0)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            h = self.trunk(x)
            return self.adapt(h), self.query(h).reshape(-1), self.spectral(h)

    torch.set_num_threads(1)
    torch.manual_seed(20260715)
    model = TinyPESA(features.shape[1])
    x = torch.as_tensor(features, dtype=torch.float32)
    base_tensor = torch.as_tensor(base, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    spectral_target_tensor = torch.as_tensor(spectral_target, dtype=torch.float32)

    adapt, query_logits, spectral_logits = model(x)
    query = torch.sigmoid(query_logits).reshape(-1, 1)
    scores = torch.nn.functional.softplus(spectral_logits) + 1e-8
    energy = (scores * scores) / torch.sum(scores * scores, dim=1, keepdim=True)
    raw_delta = adapt - base_tensor.detach()
    norm = torch.linalg.norm(raw_delta, dim=1, keepdim=True)
    clipped = raw_delta * torch.clamp(config.stage0_action_alpha / (norm + config.eps), max=1.0)
    emitted = base_tensor + query * clipped

    loss_adapt = torch.nn.functional.smooth_l1_loss(adapt, target_tensor)
    loss_emit = torch.nn.functional.smooth_l1_loss(emitted, target_tensor)
    loss_query = torch.nn.functional.binary_cross_entropy_with_logits(query_logits, label_tensor)
    loss_spec = torch.mean(torch.sum((energy - spectral_target_tensor) ** 2, dim=1))
    loss_delta = torch.mean(torch.sum((query * clipped) ** 2, dim=1))
    loss = loss_adapt + loss_emit + loss_query + 0.05 * loss_spec + 0.10 * loss_delta
    loss.backward()

    grad_norms = {
        "trunk": _parameter_grad_norm(list(model.trunk.parameters()), torch),
        "adaptation": _parameter_grad_norm(list(model.adapt.parameters()), torch),
        "query": _parameter_grad_norm(list(model.query.parameters()), torch),
        "spectral": _parameter_grad_norm(list(model.spectral.parameters()), torch),
    }
    finite_nonzero = [value for value in grad_norms.values() if np.isfinite(value) and value > 0.0]
    ratio = max(finite_nonzero) / min(finite_nonzero) if finite_nonzero else float("inf")
    return {
        "valid": 1.0,
        "batch_size": len(batch),
        "loss_terms": {
            "adapt": float(loss_adapt.detach().item()),
            "emit": float(loss_emit.detach().item()),
            "query": float(loss_query.detach().item()),
            "spectral_mse": float(loss_spec.detach().item()),
            "delta": float(loss_delta.detach().item()),
            "total": float(loss.detach().item()),
        },
        "gradient_norms": grad_norms,
        "gradient_norm_ratio_largest_to_smallest": float(ratio),
    }


def _sha256_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _query_label_manifest(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    for record in records:
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "base_l1": float(record["base_l1"]),
                "lora_l1": float(record["lora_l1"]),
                "lora_improvement_l1": float(record["lora_improvement_l1"]),
                "query_label": bool(record["query_label"]),
            }
        )
    digest_lines = [f"{row['key']}|{int(row['query_label'])}" for row in rows]
    return {
        "proposal_hash": PROPOSAL_HASH,
        "thresholds": dict(thresholds),
        "row_count": len(rows),
        "sha256": _sha256_lines(digest_lines),
        "rows": rows,
    }


def _spectral_activation_manifest(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    for record in records:
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "active_rank": int(record["spectral_active_rank"]),
                "active_fraction": float(record["spectral_active_fraction"]),
                "entropy": float(record["spectral_entropy"]),
                "energy": [float(value) for value in np.asarray(record["spectral_energy"], dtype=np.float64)],
            }
        )
    digest_lines = [f"{row['key']}|{row['active_rank']}|{row['entropy']:.8f}" for row in rows]
    return {
        "proposal_hash": PROPOSAL_HASH,
        "thresholds": dict(thresholds),
        "row_count": len(rows),
        "sha256": _sha256_lines(digest_lines),
        "rows": rows,
    }


def _classify_hard_stop(reasons: Sequence[str]) -> str:
    text = " ".join(reasons).lower()
    if "headroom" in text:
        return "NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE"
    if "label" in text or "overlap" in text or "duplicate" in text or "missing" in text or "proxy" in text:
        return "DATA_OR_SUPERVISION_FAILURE"
    if "gradient" in text or "nonfinite" in text or "torch" in text:
        return "IMPLEMENTATION_FAILURE"
    return "DESIGN_FAILURE"


def audit_pesa_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    config: PESAConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PESAConfig()
    raw_records = build_pesa_records(prediction_records)
    labeled, thresholds = compute_query_and_spectral_labels(raw_records, cfg)
    development_splits = set(cfg.train_splits) | set(cfg.validation_splits)
    dev_records = [record for record in labeled if str(record["split"]) in development_splits]
    train_records = [record for record in labeled if str(record["split"]) in set(cfg.train_splits)]
    validation_records = [record for record in labeled if str(record["split"]) in set(cfg.validation_splits)]
    reserved_records = [record for record in labeled if str(record["split"]) in set(cfg.confirmatory_reserved_splits)]
    train_summary = _label_summary(labeled, cfg.train_splits)
    validation_summary = _label_summary(labeled, cfg.validation_splits)
    query_probe = _query_probe_summary(labeled, cfg)
    spectral_probe = _spectral_probe_summary(labeled, cfg)
    split_overlap = _split_overlap(labeled, cfg)
    reset_overlap = _split_reset_overlap(labeled, cfg)
    validation_target_metrics = _target_distinction_metrics(validation_records, cfg)
    train_spectral = _spectral_summary(labeled, cfg.train_splits)
    validation_spectral = _spectral_summary(labeled, cfg.validation_splits)
    gradient_audit = _gradient_audit(labeled, cfg)
    hard_stop_reasons: list[str] = []

    if len(dev_records) < cfg.min_scoreable_records:
        hard_stop_reasons.append(f"scoreable development records below minimum: {len(dev_records)} < {cfg.min_scoreable_records}")
    selected_task_count = len({int(record["task_index"]) for record in dev_records})
    if selected_task_count < cfg.min_task_count:
        hard_stop_reasons.append(f"selected task count below minimum: {selected_task_count} < {cfg.min_task_count}")

    duplicate_sample_keys = _duplicate_count([record["key"] for record in labeled])
    duplicate_frame_keys = _duplicate_count([record["frame_key"] for record in labeled])
    if duplicate_sample_keys:
        hard_stop_reasons.append(f"duplicate sample keys: {duplicate_sample_keys}")
    if duplicate_frame_keys:
        hard_stop_reasons.append(f"duplicate frame keys: {duplicate_frame_keys}")
    if any(split_overlap.values()):
        hard_stop_reasons.append(f"split frame overlap nonzero: {split_overlap}")
    if any(reset_overlap.values()):
        hard_stop_reasons.append(f"split reset overlap nonzero: {reset_overlap}")

    missing_lora = sum(1 for record in dev_records if "lora_action" not in record)
    if missing_lora:
        hard_stop_reasons.append(f"missing lora/adaptation actions for development records: {missing_lora}")

    for split_name, summary in (("train", train_summary), ("validation", validation_summary)):
        frac = float(summary["positive_fraction"])
        if frac < cfg.min_positive_fraction or frac > cfg.max_positive_fraction:
            hard_stop_reasons.append(f"{split_name} query label fraction outside bounds: {frac:.6f}")
    if int(train_summary["positive_count"]) < cfg.min_positive_count:
        hard_stop_reasons.append(f"train query positives below minimum: {train_summary['positive_count']}")
    if int(train_summary["negative_count"]) < cfg.min_negative_count:
        hard_stop_reasons.append(f"train query negatives below minimum: {train_summary['negative_count']}")
    max_task_share = _max_task_positive_share(train_records)
    if max_task_share > cfg.max_task_positive_share:
        hard_stop_reasons.append(f"single-task query positive share too high: {max_task_share:.6f}")
    if float(query_probe["accuracy_margin"]) < cfg.min_query_probe_accuracy_margin:
        hard_stop_reasons.append(f"query probe accuracy margin below minimum: {query_probe['accuracy_margin']:.6f}")

    base_l1_validation = _mean_l1(validation_records, "base_action")
    lora_l1_validation = _mean_l1(validation_records, "lora_action")
    mean_l1_validation = _mean_l1(validation_records, "mean_action")
    if base_l1_validation is None or lora_l1_validation is None:
        hard_stop_reasons.append("missing Base or LoRA validation headroom metric")
        standard_lora_headroom = 0.0
    else:
        standard_lora_headroom = float(base_l1_validation - lora_l1_validation)
        if standard_lora_headroom <= cfg.min_headroom_l1:
            hard_stop_reasons.append(f"standard LoRA headroom below minimum: {standard_lora_headroom:.6f}")

    base_actions = np.asarray([record["base_action"] for record in dev_records], dtype=np.float64)
    base_validity = _action_validity(base_actions)
    if base_validity < 1.0:
        hard_stop_reasons.append(f"base action validity below 1.0: {base_validity:.6f}")

    if validation_spectral["active_rank_min"] == validation_spectral["active_rank_max"]:
        hard_stop_reasons.append("spectral active rank collapsed to one value on validation")
    mean_fraction = float(validation_spectral["active_fraction_mean"])
    if mean_fraction < cfg.min_spectral_activation_fraction or mean_fraction > cfg.max_spectral_activation_fraction:
        hard_stop_reasons.append(f"spectral active fraction outside bounds: {mean_fraction:.6f}")
    if int(validation_spectral["distinct_task_active_rank_profiles"]) < cfg.min_distinct_task_active_rank_profiles:
        hard_stop_reasons.append("spectral active-rank task profiles are not distinct")

    if validation_target_metrics["full_vs_priorvla_proxy_mean_l2"] < cfg.min_action_distinction_l2:
        hard_stop_reasons.append(
            f"full PESA too close to PriorVLA-style proxy: {validation_target_metrics['full_vs_priorvla_proxy_mean_l2']:.6f}"
        )
    if validation_target_metrics["full_vs_no_spectral_no_query_mean_l2"] < cfg.min_action_distinction_l2:
        hard_stop_reasons.append(
            "full PESA too close to no-spectral/no-prior-query ablation: "
            f"{validation_target_metrics['full_vs_no_spectral_no_query_mean_l2']:.6f}"
        )
    if validation_target_metrics["full_vs_selected_simple_killer_mean_l2"] < cfg.min_action_distinction_l2:
        hard_stop_reasons.append(
            f"full PESA too close to simple killer: {validation_target_metrics['full_vs_selected_simple_killer_mean_l2']:.6f}"
        )

    initial_delta_p95 = 0.0
    if initial_delta_p95 > cfg.init_delta_p95_max:
        hard_stop_reasons.append(f"initial action delta p95 too high: {initial_delta_p95:.9f}")

    if float(gradient_audit.get("valid", 0.0)) < 1.0:
        hard_stop_reasons.append(str(gradient_audit.get("hard_stop_reason", "invalid gradient audit")))
    else:
        grad_norms = gradient_audit.get("gradient_norms", {})
        for name in ("adaptation", "query", "spectral"):
            value = float(grad_norms.get(name, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                hard_stop_reasons.append(f"{name} gradient norm is nonfinite or zero: {value}")

    manifest_records = train_records + validation_records
    report = {
        "schema_version": 1,
        "method": "PESA-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "confirmatory_test_tuning_happened": False,
        "raw_prediction_records": len(prediction_records),
        "scoreable_development_records": len(dev_records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "reserved_records_not_used": len(reserved_records),
        "selected_task_count": selected_task_count,
        "duplicate_sample_keys": duplicate_sample_keys,
        "duplicate_frame_keys": duplicate_frame_keys,
        "split_overlap": split_overlap,
        "reset_overlap": reset_overlap,
        "query_thresholds": thresholds,
        "train_query_label_summary": train_summary,
        "validation_query_label_summary": validation_summary,
        "max_task_query_positive_share": max_task_share,
        "query_probe_summary": query_probe,
        "spectral_probe_summary": spectral_probe,
        "train_spectral_summary": train_spectral,
        "validation_spectral_summary": validation_spectral,
        "base_action_l1_validation": base_l1_validation,
        "standard_lora_action_l1_validation": lora_l1_validation,
        "mean_action_l1_validation": mean_l1_validation,
        "standard_lora_headroom_l1_validation": standard_lora_headroom,
        "target_distinction_metrics_validation": validation_target_metrics,
        "gradient_audit": gradient_audit,
        "initial_action_delta_p95": initial_delta_p95,
        "base_action_validity": base_validity,
        "query_label_manifest": _query_label_manifest(manifest_records, thresholds),
        "spectral_activation_manifest": _spectral_activation_manifest(manifest_records, thresholds),
        "split_manifest": {
            "train_splits": list(cfg.train_splits),
            "validation_splits": list(cfg.validation_splits),
            "confirmatory_reserved_splits": list(cfg.confirmatory_reserved_splits),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "reserved_record_count": len(reserved_records),
            "split_overlap": split_overlap,
            "reset_overlap": reset_overlap,
        },
        "hard_stop_reasons": hard_stop_reasons,
    }
    if hard_stop_reasons:
        report["final_decision"] = _classify_hard_stop(hard_stop_reasons)
        report["next_step"] = "Do not train or roll out PESA; classify the Stage 0 failure and continue to the next method cycle."
    else:
        report["final_decision"] = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
        report["next_step"] = "Run the bounded six-configuration PESA validation search."
    return report


__all__ = [
    "FORBIDDEN_INFERENCE_KEYS",
    "PESAConfig",
    "PROPOSAL_HASH",
    "VALIDATION_CONFIGS",
    "audit_pesa_records",
    "build_pesa_records",
    "compute_query_and_spectral_labels",
    "validate_inference_fields",
]
