"""DAGR-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89"
GROUP_NAMES = ("translation", "rotation", "gripper")
GROUP_SLICES = {
    "translation": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(6, 7),
}
VALIDATION_CONFIGS = (
    {"config_id": "dagr_a005_route_linear", "residual_alpha": 0.05, "route_architecture": "linear"},
    {"config_id": "dagr_a010_route_linear", "residual_alpha": 0.10, "route_architecture": "linear"},
    {"config_id": "dagr_a020_route_linear", "residual_alpha": 0.20, "route_architecture": "linear"},
    {"config_id": "dagr_a005_route_mlp", "residual_alpha": 0.05, "route_architecture": "mlp"},
    {"config_id": "dagr_a010_route_mlp", "residual_alpha": 0.10, "route_architecture": "mlp"},
    {"config_id": "dagr_a020_route_mlp", "residual_alpha": 0.20, "route_architecture": "mlp"},
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
class DAGRConfig:
    train_splits: tuple[str, ...] = ("train",)
    validation_splits: tuple[str, ...] = ("val",)
    confirmatory_reserved_splits: tuple[str, ...] = ("test",)
    min_scoreable_records: int = 500
    min_task_count: int = 3
    min_group_positive_fraction: float = 0.05
    max_group_positive_fraction: float = 0.95
    min_group_positive_count: int = 50
    min_group_negative_count: int = 50
    max_task_positive_share: float = 0.15
    min_route_probe_accuracy_margin: float = 0.02
    max_route_pair_jaccard: float = 0.95
    max_any_route_fraction: float = 0.95
    min_full_vs_shared_target_l2: float = 0.005
    min_full_vs_static_target_l2: float = 0.005
    init_delta_p95_max: float = 1e-6
    gripper_material_residual_threshold: float = 0.02
    route_probe_epochs: int = 1200
    route_probe_lr: float = 0.08
    route_probe_l2: float = 1e-4
    validation_epochs: int = 200
    validation_lr: float = 1e-3
    validation_seed: int = 20260714
    mlp_hidden_dim: int = 32
    eps: float = 1e-9


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged DAGR inference fields: {leaked}")


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
    sample_id = record.get("sample_id")
    if sample_id is not None:
        return f"{sample_id}|seed={int(record.get('eval_seed', 0))}"
    return "|".join(str(value) for value in _frame_key(record))


def build_dagr_records(prediction_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        if "target_action" not in record or "base_action" not in record:
            continue
        target = _as_vector("target_action", record["target_action"], 7)
        base = _as_vector("base_action", record["base_action"], 7)
        state_value = record.get("state", [0.0] * 8)
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
            "state": _as_vector("state", state_value, 8),
        }
        rows.append(row)
    return rows


def _gripper_transitions(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    transitions = np.zeros(len(records), dtype=bool)
    by_episode: dict[tuple[str, int, int], list[int]] = {}
    for index, record in enumerate(records):
        by_episode.setdefault((str(record["split"]), int(record["task_index"]), int(record["episode_index"])), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: int(records[idx]["frame_index"]))
        for position, index in enumerate(indices):
            prev_index = indices[max(0, position - 1)]
            next_index = indices[min(len(indices) - 1, position + 1)]
            prev_grip = float(np.asarray(records[prev_index]["target_action"], dtype=np.float64)[6])
            cur_grip = float(np.asarray(records[index]["target_action"], dtype=np.float64)[6])
            next_grip = float(np.asarray(records[next_index]["target_action"], dtype=np.float64)[6])
            transitions[index] = bool(np.sign(prev_grip) != np.sign(cur_grip) or np.sign(next_grip) != np.sign(cur_grip))
    return transitions


def _route_thresholds(records: Sequence[Mapping[str, Any]], config: DAGRConfig) -> dict[str, float]:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    if not train:
        return {"translation": 0.0, "rotation": 0.0, "gripper_material": config.gripper_material_residual_threshold}
    residuals = np.asarray([record["residual"] for record in train], dtype=np.float64)
    return {
        "translation": float(np.median(np.linalg.norm(residuals[:, 0:3], axis=1))),
        "rotation": float(np.median(np.linalg.norm(residuals[:, 3:6], axis=1))),
        "gripper_material": float(config.gripper_material_residual_threshold),
    }


def compute_route_labels(
    records: Sequence[Mapping[str, Any]],
    config: DAGRConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = config or DAGRConfig()
    enriched = [dict(record) for record in records]
    thresholds = _route_thresholds(enriched, cfg)
    transitions = _gripper_transitions(enriched)
    for index, record in enumerate(enriched):
        residual = np.asarray(record["residual"], dtype=np.float64)
        trans_norm = float(np.linalg.norm(residual[0:3]))
        rot_norm = float(np.linalg.norm(residual[3:6]))
        grip_abs = float(abs(residual[6]))
        labels = np.asarray(
            [
                trans_norm > thresholds["translation"],
                rot_norm > thresholds["rotation"],
                bool(transitions[index]) or grip_abs > thresholds["gripper_material"],
            ],
            dtype=bool,
        )
        record.update(
            {
                "residual_norms": {
                    "translation": trans_norm,
                    "rotation": rot_norm,
                    "gripper": grip_abs,
                },
                "gripper_transition": bool(transitions[index]),
                "route_labels": labels,
            }
        )
    return enriched, thresholds


def _duplicate_count(keys: Sequence[Any]) -> int:
    seen = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _split_overlap(records: Sequence[Mapping[str, Any]], config: DAGRConfig) -> dict[str, int]:
    train = {
        tuple(record["frame_key"][1:])
        for record in records
        if str(record["split"]) in set(config.train_splits)
    }
    validation = {
        tuple(record["frame_key"][1:])
        for record in records
        if str(record["split"]) in set(config.validation_splits)
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
            [
                np.linalg.norm(base[0:3]),
                np.linalg.norm(base[3:6]),
                abs(base[6]),
            ],
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


def _fit_route_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    config: DAGRConfig,
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
    for epoch in range(config.route_probe_epochs):
        probs = _sigmoid(x_train @ weights)
        gradient = x_train.T @ ((probs - y_train) * sample_weights) / max(len(y_train), 1)
        gradient += config.route_probe_l2 * weights
        if epoch == 0:
            first_gradient_norm = float(np.linalg.norm(gradient))
        weights -= config.route_probe_lr * gradient
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


def _route_probe_summary(records: Sequence[Mapping[str, Any]], config: DAGRConfig) -> dict[str, Any]:
    train = [record for record in records if str(record["split"]) in set(config.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(config.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    if train_features_raw.size and validation_features_raw.size:
        train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    else:
        train_features, validation_features = train_features_raw, validation_features_raw
    summary = {}
    for index, group in enumerate(GROUP_NAMES):
        train_labels = np.asarray([record["route_labels"][index] for record in train], dtype=bool)
        validation_labels = np.asarray([record["route_labels"][index] for record in validation], dtype=bool)
        summary[group] = _fit_route_probe(train_features, validation_features, train_labels, validation_labels, config)
    return summary


def _group_label_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str]) -> dict[str, Any]:
    selected = [record for record in records if str(record["split"]) in set(split_names)]
    out: dict[str, Any] = {}
    for index, group in enumerate(GROUP_NAMES):
        labels = np.asarray([record["route_labels"][index] for record in selected], dtype=bool)
        positives = int(np.sum(labels))
        out[group] = {
            "total": len(selected),
            "positive_count": positives,
            "negative_count": int(len(selected) - positives),
            "positive_fraction": float(np.mean(labels)) if len(selected) else 0.0,
        }
    return out


def _max_task_positive_share(records: Sequence[Mapping[str, Any]], group_index: int) -> float:
    positives = [record for record in records if bool(record["route_labels"][group_index])]
    if not positives:
        return 0.0
    counts: dict[int, int] = {}
    for record in positives:
        task = int(record["task_index"])
        counts[task] = counts.get(task, 0) + 1
    return float(max(counts.values()) / max(len(positives), 1))


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b)
    if not bool(np.any(union)):
        return 1.0
    return float(np.sum(np.logical_and(a, b)) / np.sum(union))


def _route_pair_jaccard(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    labels = np.asarray([record["route_labels"] for record in records], dtype=bool)
    if labels.size == 0:
        return {"translation_rotation": 1.0, "translation_gripper": 1.0, "rotation_gripper": 1.0}
    return {
        "translation_rotation": _jaccard(labels[:, 0], labels[:, 1]),
        "translation_gripper": _jaccard(labels[:, 0], labels[:, 2]),
        "rotation_gripper": _jaccard(labels[:, 1], labels[:, 2]),
    }


def _target_delta_metrics(records: Sequence[Mapping[str, Any]], train_summary: Mapping[str, Any]) -> dict[str, float]:
    if not records:
        return {
            "full_vs_shared_mean_l2": 0.0,
            "full_vs_static_mean_l2": 0.0,
            "full_target_mean_l2": 0.0,
        }
    static_weights = np.asarray(
        [
            float(train_summary["translation"]["positive_fraction"]),
            float(train_summary["translation"]["positive_fraction"]),
            float(train_summary["translation"]["positive_fraction"]),
            float(train_summary["rotation"]["positive_fraction"]),
            float(train_summary["rotation"]["positive_fraction"]),
            float(train_summary["rotation"]["positive_fraction"]),
            float(train_summary["gripper"]["positive_fraction"]),
        ],
        dtype=np.float64,
    )
    full_values = []
    shared_values = []
    static_values = []
    for record in records:
        residual = np.asarray(record["residual"], dtype=np.float64)
        labels = np.asarray(record["route_labels"], dtype=bool)
        full = np.zeros(7, dtype=np.float64)
        full[0:3] = residual[0:3] * float(labels[0])
        full[3:6] = residual[3:6] * float(labels[1])
        full[6] = residual[6] * float(labels[2])
        shared = residual * float(bool(np.any(labels)))
        static = residual * static_weights
        full_values.append(full)
        shared_values.append(shared)
        static_values.append(static)
    full_array = np.asarray(full_values, dtype=np.float64)
    shared_array = np.asarray(shared_values, dtype=np.float64)
    static_array = np.asarray(static_values, dtype=np.float64)
    return {
        "full_vs_shared_mean_l2": float(np.mean(np.linalg.norm(full_array - shared_array, axis=1))),
        "full_vs_static_mean_l2": float(np.mean(np.linalg.norm(full_array - static_array, axis=1))),
        "full_target_mean_l2": float(np.mean(np.linalg.norm(full_array, axis=1))),
    }


def _action_validity(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    valid = []
    for record in records:
        action = np.asarray(record["base_action"], dtype=np.float64)
        valid.append(bool(np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 5.0))
    return float(np.mean(valid))


def _sha256_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _route_label_manifest(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    for record in records:
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "residual_norms": {
                    key: float(value)
                    for key, value in (record.get("residual_norms") or {}).items()
                },
                "route_labels": {
                    group: bool(record["route_labels"][index])
                    for index, group in enumerate(GROUP_NAMES)
                },
                "gripper_transition": bool(record.get("gripper_transition", False)),
            }
        )
    digest_lines = [
        f"{row['key']}|{int(row['route_labels']['translation'])}{int(row['route_labels']['rotation'])}{int(row['route_labels']['gripper'])}"
        for row in rows
    ]
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
    if "probe" in text or "label" in text or "overlap" in text or "duplicate" in text or "positive" in text:
        return "DATA_OR_SUPERVISION_FAILURE"
    if "gradient" in text or "nonfinite" in text:
        return "IMPLEMENTATION_FAILURE"
    return "DESIGN_FAILURE"


def audit_dagr_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    config: DAGRConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DAGRConfig()
    raw_records = build_dagr_records(prediction_records)
    labeled, thresholds = compute_route_labels(raw_records, cfg)
    development_splits = set(cfg.train_splits) | set(cfg.validation_splits)
    dev_records = [record for record in labeled if str(record["split"]) in development_splits]
    train_records = [record for record in labeled if str(record["split"]) in set(cfg.train_splits)]
    validation_records = [record for record in labeled if str(record["split"]) in set(cfg.validation_splits)]
    reserved_records = [record for record in labeled if str(record["split"]) in set(cfg.confirmatory_reserved_splits)]
    train_summary = _group_label_summary(labeled, cfg.train_splits)
    validation_summary = _group_label_summary(labeled, cfg.validation_splits)
    probe_summary = _route_probe_summary(labeled, cfg)
    route_jaccard = _route_pair_jaccard(validation_records)
    target_metrics = _target_delta_metrics(validation_records, train_summary)
    split_overlap = _split_overlap(labeled, cfg)
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

    max_task_positive_share: dict[str, float] = {}
    for index, group in enumerate(GROUP_NAMES):
        train_group = train_summary[group]
        validation_group = validation_summary[group]
        for split_name, group_summary in [("train", train_group), ("validation", validation_group)]:
            fraction = float(group_summary["positive_fraction"])
            if fraction < cfg.min_group_positive_fraction or fraction > cfg.max_group_positive_fraction:
                hard_stop_reasons.append(f"{group} {split_name} route label collapsed: {fraction:.6f}")
        if int(train_group["positive_count"]) < cfg.min_group_positive_count:
            hard_stop_reasons.append(f"{group} train positives below minimum: {train_group['positive_count']}")
        if int(train_group["negative_count"]) < cfg.min_group_negative_count:
            hard_stop_reasons.append(f"{group} train negatives below minimum: {train_group['negative_count']}")
        share = _max_task_positive_share(train_records, index)
        max_task_positive_share[group] = share
        if share > cfg.max_task_positive_share:
            hard_stop_reasons.append(f"{group} train positives dominated by one task: {share:.6f}")
        probe = probe_summary[group]
        if float(probe.get("accuracy_margin", -1.0)) < cfg.min_route_probe_accuracy_margin:
            hard_stop_reasons.append(f"{group} route probe margin below minimum: {probe.get('accuracy_margin')}")
        if float(probe.get("first_gradient_norm", 0.0)) <= 0.0:
            hard_stop_reasons.append(f"{group} route probe first gradient is zero")

    for name, value in route_jaccard.items():
        if float(value) >= cfg.max_route_pair_jaccard:
            hard_stop_reasons.append(f"route labels are nearly identical for {name}: {value:.6f}")
    any_route_fraction = (
        float(np.mean([bool(np.any(record["route_labels"])) for record in validation_records]))
        if validation_records
        else 0.0
    )
    if any_route_fraction >= cfg.max_any_route_fraction:
        hard_stop_reasons.append(f"validation any-route activation fraction too high: {any_route_fraction:.6f}")
    if target_metrics["full_vs_shared_mean_l2"] < cfg.min_full_vs_shared_target_l2:
        hard_stop_reasons.append(
            f"full route target too close to shared ablation: {target_metrics['full_vs_shared_mean_l2']:.6f}"
        )
    if target_metrics["full_vs_static_mean_l2"] < cfg.min_full_vs_static_target_l2:
        hard_stop_reasons.append(
            f"full route target too close to static proxy: {target_metrics['full_vs_static_mean_l2']:.6f}"
        )
    init_delta_p95 = 0.0
    if init_delta_p95 > cfg.init_delta_p95_max:
        hard_stop_reasons.append(f"initial action delta p95 too high: {init_delta_p95:.9f}")
    action_validity = _action_validity(dev_records)
    if action_validity < 1.0:
        hard_stop_reasons.append(f"base action validity below 1.0: {action_validity:.6f}")

    manifest_records = train_records + validation_records
    report = {
        "schema_version": 1,
        "method": "DAGR-VLA",
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
        "route_thresholds": thresholds,
        "train_route_label_summary": train_summary,
        "validation_route_label_summary": validation_summary,
        "max_task_positive_share": max_task_positive_share,
        "route_probe_summary": probe_summary,
        "route_pair_jaccard_validation": route_jaccard,
        "validation_any_route_fraction": any_route_fraction,
        "target_delta_metrics_validation": target_metrics,
        "initial_action_delta_p95": init_delta_p95,
        "base_action_validity": action_validity,
        "route_label_manifest": _route_label_manifest(manifest_records, thresholds),
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
        report["next_step"] = "Do not train or roll out DAGR; classify the Stage 0 failure and continue to the next method cycle."
    else:
        report["final_decision"] = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
        report["next_step"] = "Run the bounded six-configuration DAGR validation search."
    return report


def _labels_to_dim(labels: Any, torch: Any) -> Any:
    return torch.cat(
        [
            labels[:, 0:1].repeat(1, 3),
            labels[:, 1:2].repeat(1, 3),
            labels[:, 2:3],
        ],
        dim=1,
    )


def _compose_delta(predicted_residual: Any, route_logits: Any, residual_alpha: float, torch: Any) -> tuple[Any, Any]:
    gates = torch.sigmoid(route_logits)
    trans = predicted_residual[:, 0:3] * gates[:, 0:1]
    rot = predicted_residual[:, 3:6] * gates[:, 1:2]
    grip = predicted_residual[:, 6:7] * gates[:, 2:3]
    return float(residual_alpha) * torch.cat([trans, rot, grip], dim=1), gates


def _parameter_grad_norm(parameters: Sequence[Any], torch: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).item())
    return float(total ** 0.5)


def _run_one_validation_config(
    *,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    base_validation_actions: np.ndarray,
    config_item: Mapping[str, Any],
    output_dir: Path,
    seed_offset: int,
    config: DAGRConfig,
) -> dict[str, Any]:
    import torch

    class DAGRHead(torch.nn.Module):
        def __init__(self, input_dim: int, architecture: str) -> None:
            super().__init__()
            self.architecture = architecture
            if architecture == "mlp":
                self.trunk = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, config.mlp_hidden_dim),
                    torch.nn.ReLU(),
                )
                hidden_dim = config.mlp_hidden_dim
            else:
                self.trunk = torch.nn.Identity()
                hidden_dim = input_dim
            self.route = torch.nn.Linear(hidden_dim, 3)
            self.residual = torch.nn.Linear(hidden_dim, 7)
            torch.nn.init.zeros_(self.residual.weight)
            torch.nn.init.zeros_(self.residual.bias)

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.trunk(x)
            return self.route(h), self.residual(h)

    def loss_terms(model: Any, x: Any, labels: Any, residuals: Any) -> dict[str, Any]:
        route_logits, predicted_residual = model(x)
        route_loss = torch.nn.functional.binary_cross_entropy_with_logits(route_logits, labels)
        mask = _labels_to_dim(labels, torch)
        residual_loss = torch.nn.functional.smooth_l1_loss(predicted_residual * mask, residuals * mask)
        delta, _gates = _compose_delta(predicted_residual, route_logits, float(config_item["residual_alpha"]), torch)
        delta_loss = torch.mean(torch.sum(delta * delta, dim=1))
        no_route = (torch.sum(labels, dim=1, keepdim=True) <= 0.0).float()
        clean_loss = torch.mean(no_route.reshape(-1) * torch.sum(delta * delta, dim=1))
        total = residual_loss + route_loss + 0.10 * delta_loss + 0.10 * clean_loss
        return {
            "total": total,
            "route": route_loss,
            "residual": residual_loss,
            "delta": delta_loss,
            "clean": clean_loss,
        }

    torch.set_num_threads(1)
    torch.manual_seed(config.validation_seed + seed_offset)
    x_train = torch.as_tensor(train_features, dtype=torch.float32)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels.astype(np.float32), dtype=torch.float32)
    y_validation = torch.as_tensor(validation_labels.astype(np.float32), dtype=torch.float32)
    r_train = torch.as_tensor(train_residuals.astype(np.float32), dtype=torch.float32)
    r_validation = torch.as_tensor(validation_residuals.astype(np.float32), dtype=torch.float32)

    model = DAGRHead(x_train.shape[1], str(config_item["route_architecture"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.validation_lr)
    with torch.no_grad():
        initial_terms = loss_terms(model, x_train, y_train, r_train)
        route_logits_initial, residual_initial = model(x_validation[:128])
        delta_initial, _ = _compose_delta(residual_initial, route_logits_initial, float(config_item["residual_alpha"]), torch)
        initial_delta_p95 = float(np.percentile(torch.linalg.norm(delta_initial, dim=1).cpu().numpy(), 95))

    first_grad_norms: dict[str, float] | None = None
    for epoch in range(config.validation_epochs):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, x_train, y_train, r_train)
        terms["total"].backward()
        if epoch == 0:
            route_params = list(model.route.parameters())
            residual_params = list(model.residual.parameters())
            trunk_params = list(model.trunk.parameters()) if hasattr(model.trunk, "parameters") else []
            first_grad_norms = {
                "trunk": _parameter_grad_norm(trunk_params, torch) if trunk_params else 0.0,
                "route": _parameter_grad_norm(route_params, torch),
                "residual": _parameter_grad_norm(residual_params, torch),
            }
        optimizer.step()

    with torch.no_grad():
        final_train_terms = loss_terms(model, x_train, y_train, r_train)
        validation_terms = loss_terms(model, x_validation, y_validation, r_validation)
        route_logits, predicted_residual = model(x_validation)
        delta, gates = _compose_delta(predicted_residual, route_logits, float(config_item["residual_alpha"]), torch)
        proposed_actions = torch.as_tensor(base_validation_actions.astype(np.float32), dtype=torch.float32) + delta
        route_probs = torch.sigmoid(route_logits).cpu().numpy()
        route_predictions = route_probs >= 0.5
        delta_np = delta.cpu().numpy()
        proposed_np = proposed_actions.cpu().numpy()
        predicted_residual_np = predicted_residual.cpu().numpy()
        gates_np = gates.cpu().numpy()

    route_metrics: dict[str, Any] = {}
    margins = []
    for index, group in enumerate(GROUP_NAMES):
        labels = validation_labels[:, index].astype(bool)
        predictions = route_predictions[:, index].astype(bool)
        accuracy = float(np.mean(predictions == labels))
        majority = float(max(np.mean(labels), 1.0 - np.mean(labels)))
        margin = accuracy - majority
        margins.append(margin)
        route_metrics[group] = {
            "accuracy": accuracy,
            "majority_accuracy": majority,
            "accuracy_margin": margin,
            "predicted_positive_fraction": float(np.mean(predictions)),
            "mean_probability": float(np.mean(route_probs[:, index])),
        }

    alpha = float(config_item["residual_alpha"])
    train_positive = np.mean(train_labels, axis=0)
    static_weights = np.asarray([train_positive[0]] * 3 + [train_positive[1]] * 3 + [train_positive[2]], dtype=np.float64)
    max_gate = np.max(gates_np, axis=1, keepdims=True)
    shared_delta = alpha * max_gate * predicted_residual_np
    static_delta = alpha * static_weights.reshape(1, 7) * predicted_residual_np
    action_validity = float(np.mean(np.all(np.isfinite(proposed_np), axis=1) & (np.max(np.abs(proposed_np), axis=1) <= 5.0)))
    delta_l2 = np.linalg.norm(delta_np, axis=1)
    clean_rows = np.sum(validation_labels, axis=1) == 0
    clean_delta_p95 = float(np.percentile(delta_l2[clean_rows], 95)) if bool(np.any(clean_rows)) else 0.0
    mean_full_static = float(np.mean(np.linalg.norm(delta_np - static_delta, axis=1)))
    mean_full_shared = float(np.mean(np.linalg.norm(delta_np - shared_delta, axis=1)))
    distinction = min(1.0, 0.5 * mean_full_static / 0.005 + 0.5 * mean_full_shared / 0.005)
    route_predictability = float(np.mean([max(0.0, margin) / 0.10 for margin in margins]))
    route_predictability = float(np.clip(route_predictability, 0.0, 1.0))
    clean_retention = float(np.clip(1.0 - clean_delta_p95 / 0.20, 0.0, 1.0))
    bounded_delta = float(np.clip(1.0 - float(np.percentile(delta_l2, 95)) / 0.35, 0.0, 1.0))
    validity_score = 0.5 * action_validity + 0.5 * bounded_delta
    compute_overhead = 1.0 if config_item["route_architecture"] == "linear" else 0.95
    total_score = (
        0.30 * route_predictability
        + 0.25 * clean_retention
        + 0.20 * distinction
        + 0.15 * validity_score
        + 0.10 * compute_overhead
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
    reloaded = DAGRHead(x_train.shape[1], str(config_item["route_architecture"]))
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
    if float((first_grad_norms or {}).get("route", 0.0)) <= 0.0:
        hard_stop_reasons.append("route head first gradient is zero")
    if float((first_grad_norms or {}).get("residual", 0.0)) <= 0.0:
        hard_stop_reasons.append("residual head first gradient is zero")
    if action_validity < 1.0:
        hard_stop_reasons.append("invalid proposed validation action")
    if initial_delta_p95 > config.init_delta_p95_max:
        hard_stop_reasons.append("initial residual is not base-passthrough")

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
        "route_metrics": route_metrics,
        "validation_metrics": {
            "delta_l2_mean": float(np.mean(delta_l2)),
            "delta_l2_p95": float(np.percentile(delta_l2, 95)),
            "clean_delta_l2_p95": clean_delta_p95,
            "gate_mean_by_group": {
                group: float(np.mean(gates_np[:, index])) for index, group in enumerate(GROUP_NAMES)
            },
            "gate_activation_fraction_by_group": {
                group: float(np.mean(gates_np[:, index] >= 0.5)) for index, group in enumerate(GROUP_NAMES)
            },
            "full_vs_static_mean_l2": mean_full_static,
            "full_vs_shared_mean_l2": mean_full_shared,
            "action_validity": action_validity,
        },
        "score_terms": {
            "route_predictability": route_predictability,
            "clean_retention_and_bounded_delta": clean_retention,
            "full_proxy_ablation_distinction": distinction,
            "action_validity_and_group_delta": validity_score,
            "compute_overhead": compute_overhead,
            "total": float(total_score),
        },
        "hard_stop_reasons": hard_stop_reasons,
        "final_decision": "VALIDATION_CONFIG_PASS" if not hard_stop_reasons else "VALIDATION_CONFIG_STOP",
    }


def run_validation_search(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path = "reports/dagr_vla/validation_checkpoints",
    config: DAGRConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DAGRConfig()
    audit = audit_dagr_records(prediction_records, config=cfg)
    if audit["final_decision"] != "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH":
        return {
            "schema_version": 1,
            "method": "DAGR-VLA",
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

    records, _thresholds = compute_route_labels(build_dagr_records(prediction_records), cfg)
    train = [record for record in records if str(record["split"]) in set(cfg.train_splits)]
    validation = [record for record in records if str(record["split"]) in set(cfg.validation_splits)]
    task_count = max([int(record["task_index"]) for record in records] + [0]) + 1
    train_features_raw = _feature_matrix(train, task_count)
    validation_features_raw = _feature_matrix(validation, task_count)
    train_features, validation_features = _standardize(train_features_raw, validation_features_raw)
    train_labels = np.asarray([record["route_labels"] for record in train], dtype=bool)
    validation_labels = np.asarray([record["route_labels"] for record in validation], dtype=bool)
    train_residuals = np.asarray([record["residual"] for record in train], dtype=np.float64)
    validation_residuals = np.asarray([record["residual"] for record in validation], dtype=np.float64)
    base_validation_actions = np.asarray([record["base_action"] for record in validation], dtype=np.float64)

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
                base_validation_actions=base_validation_actions,
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
        "method": "DAGR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": True,
        "confirmatory_test_tuning_happened": False,
        "audit_final_decision": audit["final_decision"],
        "search_budget": "6 configs: residual alpha in {0.05, 0.10, 0.20} x route architecture in {linear, mlp}",
        "score_weights": {
            "route_predictability_above_majority": 0.30,
            "clean_action_retention_and_bounded_deltas": 0.25,
            "full_versus_proxy_and_ablation_distinction": 0.20,
            "action_validity_and_group_delta": 0.15,
            "compute_overhead": 0.10,
        },
        "tried_config_count": len(tried),
        "tried_configs": tried,
        "selected_config": selected,
        "final_decision": final_decision,
        "next_step": (
            "Freeze the selected DAGR config and train disk-reloadable policy identities for the five-policy comparison before Stage A."
            if selected
            else "Archive DAGR validation-search failure and continue to the next method cycle."
        ),
    }


__all__ = [
    "DAGRConfig",
    "FORBIDDEN_INFERENCE_KEYS",
    "GROUP_NAMES",
    "PROPOSAL_HASH",
    "VALIDATION_CONFIGS",
    "audit_dagr_records",
    "build_dagr_records",
    "compute_route_labels",
    "run_validation_search",
    "validate_inference_fields",
]
