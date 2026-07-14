"""MARC-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A"
VALIDATION_CONFIGS = (
    {"config_id": "marc_a005_gate_linear", "correction_alpha": 0.05, "gate_architecture": "linear"},
    {"config_id": "marc_a010_gate_linear", "correction_alpha": 0.10, "gate_architecture": "linear"},
    {"config_id": "marc_a020_gate_linear", "correction_alpha": 0.20, "gate_architecture": "linear"},
    {"config_id": "marc_a005_gate_mlp", "correction_alpha": 0.05, "gate_architecture": "mlp"},
    {"config_id": "marc_a010_gate_mlp", "correction_alpha": 0.10, "gate_architecture": "mlp"},
    {"config_id": "marc_a020_gate_mlp", "correction_alpha": 0.20, "gate_architecture": "mlp"},
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
class MARCConfig:
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
    min_gate_probe_accuracy_margin: float = 0.02
    min_full_vs_proxy_target_l2: float = 0.003
    min_full_vs_no_gate_target_l2: float = 0.003
    min_full_vs_static_target_l2: float = 0.003
    init_delta_p95_max: float = 1e-6
    disagreement_quantile: float = 0.60
    gate_probe_epochs: int = 1200
    gate_probe_lr: float = 0.08
    gate_probe_l2: float = 1e-4
    validation_epochs: int = 220
    validation_lr: float = 1e-3
    validation_seed: int = 20260715
    mlp_hidden_dim: int = 32
    eps: float = 1e-9


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged MARC inference fields: {leaked}")


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
    if sample_id is not None:
        return f"{sample_id}|seed={int(record.get('eval_seed', 0))}"
    return "|".join(str(value) for value in _frame_key(record))


def build_marc_records(prediction_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
            row["lora_action"] = _as_vector("lora_action", record["lora_action"], 7)
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


def _split_overlap(records: Sequence[Mapping[str, Any]], config: MARCConfig) -> dict[str, int]:
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


def _disagreement_threshold(records: Sequence[Mapping[str, Any]], config: MARCConfig) -> float:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    if not train:
        return 0.0
    values = np.asarray([np.linalg.norm(record["residual"]) for record in train], dtype=np.float64)
    return float(np.quantile(values, config.disagreement_quantile))


def compute_disagreement_labels(
    records: Sequence[Mapping[str, Any]],
    config: MARCConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = config or MARCConfig()
    enriched = [dict(record) for record in records]
    threshold = _disagreement_threshold(enriched, cfg)
    for record in enriched:
        residual = np.asarray(record["residual"], dtype=np.float64)
        disagreement_l2 = float(np.linalg.norm(residual))
        record.update(
            {
                "disagreement_l2": disagreement_l2,
                "disagreement_label": bool(disagreement_l2 > threshold),
            }
        )
    return enriched, {"disagreement_l2_quantile_0_60": threshold}


def _feature_matrix(records: Sequence[Mapping[str, Any]], task_count: int | None = None) -> np.ndarray:
    max_task = int(task_count) if task_count is not None else max([int(record["task_index"]) for record in records] + [0]) + 1
    features = []
    for record in records:
        base = np.asarray(record["base_action"], dtype=np.float64)
        state = np.asarray(record["state"], dtype=np.float64)
        task = np.zeros(max_task, dtype=np.float64)
        task_index = int(record["task_index"])
        if 0 <= task_index < max_task:
            task[task_index] = 1.0
        base_norms = np.asarray(
            [np.linalg.norm(base[0:3]), np.linalg.norm(base[3:6]), abs(base[6])],
            dtype=np.float64,
        )
        features.append(np.concatenate([base, state, [float(record["phase"])], task, base_norms]))
    return np.vstack(features) if features else np.zeros((0, 19), dtype=np.float64)


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


def _fit_gate_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    config: MARCConfig,
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
    for epoch in range(config.gate_probe_epochs):
        probs = _sigmoid(x_train @ weights)
        gradient = x_train.T @ ((probs - y_train) * sample_weights) / max(len(y_train), 1)
        gradient += config.gate_probe_l2 * weights
        if epoch == 0:
            first_gradient_norm = float(np.linalg.norm(gradient))
        weights -= config.gate_probe_lr * gradient
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


def _gate_probe_summary(records: Sequence[Mapping[str, Any]], config: MARCConfig) -> dict[str, Any]:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(config.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["disagreement_label"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["disagreement_label"] for record in validation], dtype=bool)
    return _fit_gate_probe(train_features, validation_features, train_labels, validation_labels, config)


def _label_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str]) -> dict[str, Any]:
    selected = [record for record in records if str(record["split"]) in set(split_names)]
    labels = np.asarray([record["disagreement_label"] for record in selected], dtype=bool)
    positives = int(np.sum(labels))
    return {
        "total": len(selected),
        "positive_count": positives,
        "negative_count": int(len(selected) - positives),
        "positive_fraction": float(np.mean(labels)) if len(selected) else 0.0,
    }


def _max_task_positive_share(records: Sequence[Mapping[str, Any]]) -> float:
    positives = [record for record in records if bool(record["disagreement_label"])]
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


def _mean_l2(records: Sequence[Mapping[str, Any]], action_key: str) -> float | None:
    values = []
    for record in records:
        if action_key not in record:
            continue
        action = np.asarray(record[action_key], dtype=np.float64)
        target = np.asarray(record["target_action"], dtype=np.float64)
        values.append(float(np.linalg.norm(action - target)))
    if not values:
        return None
    return float(np.mean(values))


def _target_distinction_metrics(records: Sequence[Mapping[str, Any]], train_positive_fraction: float) -> dict[str, float]:
    if not records:
        return {
            "full_vs_l1_proxy_target_mean_l2": 0.0,
            "full_vs_no_gate_target_mean_l2": 0.0,
            "full_vs_static_target_mean_l2": 0.0,
            "full_target_delta_mean_l2": 0.0,
        }
    full_targets = []
    l1_targets = []
    no_gate_targets = []
    static_targets = []
    for record in records:
        base = np.asarray(record["base_action"], dtype=np.float64)
        target = np.asarray(record["target_action"], dtype=np.float64)
        residual = target - base
        label = float(bool(record["disagreement_label"]))
        full = base + label * residual
        l1 = target
        no_gate = target
        static = base + train_positive_fraction * residual
        full_targets.append(full)
        l1_targets.append(l1)
        no_gate_targets.append(no_gate)
        static_targets.append(static)
    full_array = np.asarray(full_targets, dtype=np.float64)
    return {
        "full_vs_l1_proxy_target_mean_l2": float(np.mean(np.linalg.norm(full_array - np.asarray(l1_targets), axis=1))),
        "full_vs_no_gate_target_mean_l2": float(np.mean(np.linalg.norm(full_array - np.asarray(no_gate_targets), axis=1))),
        "full_vs_static_target_mean_l2": float(np.mean(np.linalg.norm(full_array - np.asarray(static_targets), axis=1))),
        "full_target_delta_mean_l2": float(np.mean(np.linalg.norm(full_array - np.asarray([r["base_action"] for r in records]), axis=1))),
    }


def _sha256_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _disagreement_label_manifest(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    for record in records:
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "disagreement_l2": float(record["disagreement_l2"]),
                "disagreement_label": bool(record["disagreement_label"]),
            }
        )
    digest_lines = [f"{row['key']}|{int(row['disagreement_label'])}" for row in rows]
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
    if "label" in text or "overlap" in text or "duplicate" in text or "positive" in text or "gate" in text:
        return "DATA_OR_SUPERVISION_FAILURE"
    if "gradient" in text or "nonfinite" in text:
        return "IMPLEMENTATION_FAILURE"
    return "DESIGN_FAILURE"


def audit_marc_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    config: MARCConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MARCConfig()
    raw_records = build_marc_records(prediction_records)
    labeled, thresholds = compute_disagreement_labels(raw_records, cfg)
    development_splits = set(cfg.train_splits) | set(cfg.validation_splits)
    dev_records = [record for record in labeled if str(record["split"]) in development_splits]
    train_records = [record for record in labeled if str(record["split"]) in set(cfg.train_splits)]
    validation_records = [record for record in labeled if str(record["split"]) in set(cfg.validation_splits)]
    reserved_records = [record for record in labeled if str(record["split"]) in set(cfg.confirmatory_reserved_splits)]
    train_summary = _label_summary(labeled, cfg.train_splits)
    validation_summary = _label_summary(labeled, cfg.validation_splits)
    probe_summary = _gate_probe_summary(labeled, cfg)
    split_overlap = _split_overlap(labeled, cfg)
    train_positive_fraction = float(train_summary["positive_fraction"])
    target_metrics = _target_distinction_metrics(validation_records, train_positive_fraction)
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

    for split_name, summary in (("train", train_summary), ("validation", validation_summary)):
        frac = float(summary["positive_fraction"])
        if frac < cfg.min_positive_fraction or frac > cfg.max_positive_fraction:
            hard_stop_reasons.append(f"{split_name} disagreement label fraction outside bounds: {frac:.6f}")
    if int(train_summary["positive_count"]) < cfg.min_positive_count:
        hard_stop_reasons.append(f"train disagreement positives below minimum: {train_summary['positive_count']}")
    if int(train_summary["negative_count"]) < cfg.min_negative_count:
        hard_stop_reasons.append(f"train disagreement negatives below minimum: {train_summary['negative_count']}")

    max_task_share = _max_task_positive_share(train_records)
    if max_task_share > cfg.max_task_positive_share:
        hard_stop_reasons.append(f"single-task positive share too high: {max_task_share:.6f}")
    if float(probe_summary["accuracy_margin"]) < cfg.min_gate_probe_accuracy_margin:
        hard_stop_reasons.append(f"gate probe accuracy margin below minimum: {probe_summary['accuracy_margin']:.6f}")

    if target_metrics["full_vs_l1_proxy_target_mean_l2"] < cfg.min_full_vs_proxy_target_l2:
        hard_stop_reasons.append("MARC target too close to L1 proxy target")
    if target_metrics["full_vs_no_gate_target_mean_l2"] < cfg.min_full_vs_no_gate_target_l2:
        hard_stop_reasons.append("MARC target too close to no-gate target")
    if target_metrics["full_vs_static_target_mean_l2"] < cfg.min_full_vs_static_target_l2:
        hard_stop_reasons.append("MARC target too close to static mixture target")

    base_actions = np.asarray([record["base_action"] for record in dev_records], dtype=np.float64)
    base_validity = _action_validity(base_actions)
    if base_validity < 1.0:
        hard_stop_reasons.append(f"base action validity below 1.0: {base_validity:.6f}")
    initial_delta_p95 = 0.0
    if initial_delta_p95 > cfg.init_delta_p95_max:
        hard_stop_reasons.append(f"initial MARC action delta p95 too high: {initial_delta_p95:.9f}")

    mean_action_l2_validation = _mean_l2(validation_records, "mean_action")
    lora_action_l2_validation = _mean_l2(validation_records, "lora_action")
    base_action_l2_validation = _mean_l2(validation_records, "base_action")

    manifest_records = train_records + validation_records
    report = {
        "schema_version": 1,
        "method": "MARC-VLA",
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
        "disagreement_thresholds": thresholds,
        "train_disagreement_label_summary": train_summary,
        "validation_disagreement_label_summary": validation_summary,
        "max_task_positive_share": max_task_share,
        "gate_probe_summary": probe_summary,
        "target_distinction_metrics_validation": target_metrics,
        "base_action_l2_validation": base_action_l2_validation,
        "mean_action_l2_validation": mean_action_l2_validation,
        "lora_action_l2_validation": lora_action_l2_validation,
        "initial_action_delta_p95": initial_delta_p95,
        "base_action_validity": base_validity,
        "disagreement_label_manifest": _disagreement_label_manifest(manifest_records, thresholds),
        "split_manifest": {
            "train_splits": list(cfg.train_splits),
            "validation_splits": list(cfg.validation_splits),
            "confirmatory_reserved_splits": list(cfg.confirmatory_reserved_splits),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "reserved_record_count": len(reserved_records),
            "split_overlap": split_overlap,
        },
        "hard_stop_reasons": hard_stop_reasons,
    }
    if hard_stop_reasons:
        report["final_decision"] = _classify_hard_stop(hard_stop_reasons)
        report["next_step"] = "Do not train or roll out MARC; classify the Stage 0 failure and continue to the next method cycle."
    else:
        report["final_decision"] = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
        report["next_step"] = "Run the bounded six-configuration MARC validation search."
    return report


def _parameter_grad_norm(parameters: Sequence[Any], torch: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).item())
    return float(total**0.5)


def _run_one_validation_config(
    *,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    base_train_actions: np.ndarray,
    base_validation_actions: np.ndarray,
    target_train_actions: np.ndarray,
    target_validation_actions: np.ndarray,
    config_item: Mapping[str, Any],
    output_dir: Path,
    seed_offset: int,
    config: MARCConfig,
) -> dict[str, Any]:
    import torch

    class MARCHead(torch.nn.Module):
        def __init__(self, input_dim: int, architecture: str) -> None:
            super().__init__()
            if architecture == "mlp":
                self.trunk = torch.nn.Sequential(torch.nn.Linear(input_dim, config.mlp_hidden_dim), torch.nn.ReLU())
                hidden_dim = config.mlp_hidden_dim
            else:
                self.trunk = torch.nn.Identity()
                hidden_dim = input_dim
            self.gate = torch.nn.Linear(hidden_dim, 1)
            self.residual = torch.nn.Linear(hidden_dim, 7)
            torch.nn.init.constant_(self.gate.bias, -6.0)
            torch.nn.init.zeros_(self.residual.weight)
            torch.nn.init.zeros_(self.residual.bias)

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.trunk(x)
            return self.gate(h), self.residual(h)

    def clipped(delta: Any, alpha: float) -> Any:
        norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        scale = torch.clamp(float(alpha) / (norm + config.eps), max=1.0)
        return delta * scale

    def loss_terms(model: Any, x: Any, labels: Any, residuals: Any, base_actions: Any, target_actions: Any) -> dict[str, Any]:
        logits, predicted_residual = model(x)
        anchor = base_actions + predicted_residual
        anchor_loss = torch.nn.functional.smooth_l1_loss(anchor, target_actions)
        gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.reshape(-1))
        gate = torch.sigmoid(logits)
        correction = clipped(predicted_residual, float(config_item["correction_alpha"])) * gate
        delta_loss = torch.mean(torch.sum(correction * correction, dim=1))
        clean_loss = torch.mean((1.0 - labels.reshape(-1)) * torch.sum(correction * correction, dim=1))
        total = anchor_loss + gate_loss + 0.10 * delta_loss + 0.10 * clean_loss
        return {"total": total, "anchor": anchor_loss, "gate": gate_loss, "delta": delta_loss, "clean": clean_loss}

    torch.set_num_threads(1)
    torch.manual_seed(config.validation_seed + seed_offset)
    x_train = torch.as_tensor(train_features, dtype=torch.float32)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels.astype(np.float32), dtype=torch.float32)
    y_validation = torch.as_tensor(validation_labels.astype(np.float32), dtype=torch.float32)
    r_train = torch.as_tensor(train_residuals.astype(np.float32), dtype=torch.float32)
    r_validation = torch.as_tensor(validation_residuals.astype(np.float32), dtype=torch.float32)
    base_train = torch.as_tensor(base_train_actions.astype(np.float32), dtype=torch.float32)
    base_validation = torch.as_tensor(base_validation_actions.astype(np.float32), dtype=torch.float32)
    target_train = torch.as_tensor(target_train_actions.astype(np.float32), dtype=torch.float32)
    target_validation = torch.as_tensor(target_validation_actions.astype(np.float32), dtype=torch.float32)

    model = MARCHead(x_train.shape[1], str(config_item["gate_architecture"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.validation_lr)
    with torch.no_grad():
        initial_terms = loss_terms(model, x_train, y_train, r_train, base_train, target_train)
        init_logits, init_residual = model(x_validation[:128])
        init_delta = clipped(init_residual, float(config_item["correction_alpha"])) * torch.sigmoid(init_logits)
        initial_delta_p95 = float(np.percentile(torch.linalg.norm(init_delta, dim=1).cpu().numpy(), 95))

    first_grad_norms: dict[str, float] | None = None
    for epoch in range(config.validation_epochs):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, x_train, y_train, r_train, base_train, target_train)
        terms["total"].backward()
        if epoch == 0:
            first_grad_norms = {
                "trunk": _parameter_grad_norm(list(model.trunk.parameters()), torch)
                if hasattr(model.trunk, "parameters")
                else 0.0,
                "gate": _parameter_grad_norm(list(model.gate.parameters()), torch),
                "anchor_residual": _parameter_grad_norm(list(model.residual.parameters()), torch),
            }
        optimizer.step()

    with torch.no_grad():
        final_train_terms = loss_terms(model, x_train, y_train, r_train, base_train, target_train)
        validation_terms = loss_terms(model, x_validation, y_validation, r_validation, base_validation, target_validation)
        logits, predicted_residual = model(x_validation)
        gate = torch.sigmoid(logits)
        alpha = float(config_item["correction_alpha"])
        clipped_residual = clipped(predicted_residual, alpha)
        full_delta = clipped_residual * gate
        no_gate_delta = clipped_residual
        beta = float(np.mean(train_labels))
        static_delta = clipped_residual * beta
        full_actions = base_validation + full_delta
        l1_proxy_actions = base_validation + predicted_residual
        no_gate_actions = base_validation + no_gate_delta
        static_actions = base_validation + static_delta

    full_np = full_actions.cpu().numpy()
    l1_np = l1_proxy_actions.cpu().numpy()
    no_gate_np = no_gate_actions.cpu().numpy()
    static_np = static_actions.cpu().numpy()
    full_delta_np = full_delta.cpu().numpy()
    gate_np = gate.cpu().numpy().reshape(-1)
    validation_labels_bool = validation_labels.astype(bool)
    predictions = gate_np >= 0.5
    accuracy = float(np.mean(predictions == validation_labels_bool))
    majority = float(max(np.mean(validation_labels_bool), 1.0 - np.mean(validation_labels_bool)))
    delta_l2 = np.linalg.norm(full_delta_np, axis=1)
    clean_rows = ~validation_labels_bool
    clean_delta_p95 = float(np.percentile(delta_l2[clean_rows], 95)) if bool(np.any(clean_rows)) else 0.0
    action_validity = _action_validity(full_np)
    full_vs_proxy = float(np.mean(np.linalg.norm(full_np - l1_np, axis=1)))
    full_vs_no_gate = float(np.mean(np.linalg.norm(full_np - no_gate_np, axis=1)))
    full_vs_static = float(np.mean(np.linalg.norm(full_np - static_np, axis=1)))
    l1_action_l2 = float(np.mean(np.linalg.norm(l1_np - target_validation_actions, axis=1)))
    full_action_l2 = float(np.mean(np.linalg.norm(full_np - target_validation_actions, axis=1)))

    gate_predictability = float(np.clip(max(0.0, accuracy - majority) / 0.10, 0.0, 1.0))
    clean_retention = float(np.clip(1.0 - clean_delta_p95 / 0.20, 0.0, 1.0))
    bounded_delta = float(np.clip(1.0 - float(np.percentile(delta_l2, 95)) / 0.35, 0.0, 1.0))
    distinction = float(np.clip(min(full_vs_proxy, full_vs_no_gate, full_vs_static) / 0.010, 0.0, 1.0))
    compute_overhead = 1.0 if config_item["gate_architecture"] == "linear" else 0.95
    l1_proxy_validity = 1.0 if np.all(np.isfinite(l1_np)) else 0.0
    total_score = (
        0.25 * distinction
        + 0.25 * gate_predictability
        + 0.20 * clean_retention
        + 0.15 * distinction
        + 0.10 * action_validity
        + 0.05 * compute_overhead
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{config_item['config_id']}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config_item),
            "proposal_hash": PROPOSAL_HASH,
            "feature_count": int(x_train.shape[1]),
        },
        checkpoint_path,
    )
    reloaded = MARCHead(x_train.shape[1], str(config_item["gate_architecture"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        old_logits, old_residual = model(x_validation[:32])
        new_logits, new_residual = reloaded(x_validation[:32])
        reload_max_abs_diff = max(
            float(torch.max(torch.abs(old_logits - new_logits)).item()),
            float(torch.max(torch.abs(old_residual - new_residual)).item()),
        )

    hard_stop_reasons = []
    if reload_max_abs_diff > 1e-6:
        hard_stop_reasons.append("checkpoint reload mismatch")
    if float((first_grad_norms or {}).get("gate", 0.0)) <= 0.0:
        hard_stop_reasons.append("gate first gradient is zero")
    if float((first_grad_norms or {}).get("anchor_residual", 0.0)) <= 0.0:
        hard_stop_reasons.append("anchor residual first gradient is zero")
    if action_validity < 1.0:
        hard_stop_reasons.append("invalid MARC validation action")
    if initial_delta_p95 > config.init_delta_p95_max:
        hard_stop_reasons.append("initial MARC action is not base-passthrough")
    if accuracy - majority < config.min_gate_probe_accuracy_margin:
        hard_stop_reasons.append(f"validation gate accuracy margin below minimum: {accuracy - majority:.6f}")
    predicted_positive_fraction = float(np.mean(predictions))
    if predicted_positive_fraction < config.min_positive_fraction or predicted_positive_fraction > config.max_positive_fraction:
        hard_stop_reasons.append(f"validation gate activation collapsed: {predicted_positive_fraction:.6f}")

    return {
        **dict(config_item),
        "proposal_hash": PROPOSAL_HASH,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_max_abs_diff": reload_max_abs_diff,
        "initial_delta_p95": initial_delta_p95,
        "first_gradient_norms": first_grad_norms or {},
        "loss_initial": {key: float(value.detach().item()) for key, value in initial_terms.items()},
        "loss_final_train": {key: float(value.detach().item()) for key, value in final_train_terms.items()},
        "loss_validation": {key: float(value.detach().item()) for key, value in validation_terms.items()},
        "gate_metrics": {
            "accuracy": accuracy,
            "majority_accuracy": majority,
            "accuracy_margin": accuracy - majority,
            "predicted_positive_fraction": predicted_positive_fraction,
            "mean_probability": float(np.mean(gate_np)),
        },
        "validation_metrics": {
            "delta_l2_mean": float(np.mean(delta_l2)),
            "delta_l2_p95": float(np.percentile(delta_l2, 95)),
            "clean_delta_l2_p95": clean_delta_p95,
            "full_vs_l1_proxy_mean_l2": full_vs_proxy,
            "full_vs_no_gate_mean_l2": full_vs_no_gate,
            "full_vs_static_mean_l2": full_vs_static,
            "l1_proxy_action_l2": l1_action_l2,
            "marc_full_action_l2": full_action_l2,
            "action_validity": action_validity,
            "l1_proxy_validity": l1_proxy_validity,
        },
        "score_terms": {
            "l1_proxy_validity_and_full_proxy_distinction": 0.5 * l1_proxy_validity + 0.5 * distinction,
            "gate_predictability": gate_predictability,
            "clean_retention_and_bounded_delta": clean_retention,
            "full_ablation_static_distinction": distinction,
            "action_validity": action_validity,
            "compute_overhead": compute_overhead,
            "total": float(total_score),
        },
        "hard_stop_reasons": hard_stop_reasons,
        "final_decision": "VALIDATION_CONFIG_PASS" if not hard_stop_reasons else "VALIDATION_CONFIG_STOP",
    }


def run_validation_search(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path = "reports/marc_vla/validation_checkpoints",
    config: MARCConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MARCConfig()
    audit = audit_marc_records(prediction_records, config=cfg)
    if audit["final_decision"] != "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH":
        return {
            "schema_version": 1,
            "method": "MARC-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "closed_loop_experiment_happened": False,
            "training_happened": False,
            "confirmatory_test_tuning_happened": False,
            "audit_final_decision": audit["final_decision"],
            "final_decision": "VALIDATION_SEARCH_BLOCKED_BY_STAGE_0",
            "tried_config_count": 0,
            "tried_configs": [],
            "selected_config": None,
            "next_step": "Archive the Stage 0 failure and continue to the next method cycle.",
        }

    records, _thresholds = compute_disagreement_labels(build_marc_records(prediction_records), cfg)
    train = [record for record in records if str(record["split"]) in set(cfg.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(cfg.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["disagreement_label"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["disagreement_label"] for record in validation], dtype=bool)
    train_residuals = np.asarray([record["residual"] for record in train], dtype=np.float64)
    validation_residuals = np.asarray([record["residual"] for record in validation], dtype=np.float64)
    base_train_actions = np.asarray([record["base_action"] for record in train], dtype=np.float64)
    base_validation_actions = np.asarray([record["base_action"] for record in validation], dtype=np.float64)
    target_train_actions = np.asarray([record["target_action"] for record in train], dtype=np.float64)
    target_validation_actions = np.asarray([record["target_action"] for record in validation], dtype=np.float64)

    tried = []
    for index, config_item in enumerate(VALIDATION_CONFIGS):
        tried.append(
            _run_one_validation_config(
                train_features=train_features,
                validation_features=validation_features,
                train_labels=train_labels,
                validation_labels=validation_labels,
                train_residuals=train_residuals,
                validation_residuals=validation_residuals,
                base_train_actions=base_train_actions,
                base_validation_actions=base_validation_actions,
                target_train_actions=target_train_actions,
                target_validation_actions=target_validation_actions,
                config_item=config_item,
                output_dir=Path(output_dir),
                seed_offset=index,
                config=cfg,
            )
        )

    passing = [item for item in tried if item["final_decision"] == "VALIDATION_CONFIG_PASS"]
    selected = max(passing, key=lambda item: float((item.get("score_terms") or {}).get("total", -1.0))) if passing else None
    final_decision = "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING" if selected else "VALIDATION_SEARCH_NO_PASSING_CONFIG"
    return {
        "schema_version": 1,
        "method": "MARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": True,
        "confirmatory_test_tuning_happened": False,
        "audit_final_decision": audit["final_decision"],
        "search_budget": "6 configs: correction alpha in {0.05, 0.10, 0.20} x gate architecture in {linear, mlp}",
        "score_weights": {
            "l1_proxy_validity_and_full_proxy_distinction": 0.25,
            "gate_predictability_above_majority": 0.25,
            "clean_action_retention_and_bounded_deltas": 0.20,
            "full_versus_no_gate_and_static_distinction": 0.15,
            "action_validity": 0.10,
            "compute_overhead": 0.05,
        },
        "tried_config_count": len(tried),
        "tried_configs": tried,
        "selected_config": selected,
        "final_decision": final_decision,
        "next_step": (
            "Freeze the selected MARC config and train disk-reloadable policy identities for the five-policy comparison before Stage A."
            if selected
            else "Archive MARC validation-search failure and continue to the next method cycle."
        ),
    }


__all__ = [
    "FORBIDDEN_INFERENCE_KEYS",
    "MARCConfig",
    "PROPOSAL_HASH",
    "VALIDATION_CONFIGS",
    "audit_marc_records",
    "build_marc_records",
    "compute_disagreement_labels",
    "run_validation_search",
    "validate_inference_fields",
]
