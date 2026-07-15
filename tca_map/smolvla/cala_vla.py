"""CALA-VLA development-only Stage 0 audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76"
FORBIDDEN_INFERENCE_KEYS = {
    "dataset_global_index",
    "episode_index",
    "frame_index",
    "future_action",
    "future_action_segment",
    "future_observation",
    "identity",
    "latent_label",
    "object_pose",
    "object_state",
    "oracle_help_label",
    "phase",
    "placement_pose",
    "reward",
    "success",
    "target_action",
}
LEGAL_INFERENCE_FEATURES = (
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "language_or_task_instruction",
    "base_action",
)


@dataclass(frozen=True)
class CALAConfig:
    train_splits: tuple[str, ...] = ("train",)
    validation_splits: tuple[str, ...] = ("val",)
    confirmatory_reserved_splits: tuple[str, ...] = ("test",)
    min_scoreable_records: int = 500
    min_task_count: int = 3
    latent_horizon: int = 16
    horizon_health: tuple[int, ...] = (8, 16, 32)
    min_nonzero_latent_dims: int = 3
    max_latent_bin_share: float = 0.95
    max_task_latent_share: float = 0.35
    min_predictability_margin: float = 0.02
    min_oracle_action_headroom_l2: float = 0.01
    init_delta_p95_max: float = 1e-6
    ridge_l2: float = 1e-3
    eps: float = 1e-9


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


def _duplicate_count(keys: Sequence[Any]) -> int:
    seen = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _split_overlap(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> dict[str, int]:
    train = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.train_splits)}
    validation = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.validation_splits)}
    reserved = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.confirmatory_reserved_splits)}
    return {
        "train_validation": len(train & validation),
        "train_reserved": len(train & reserved),
        "validation_reserved": len(validation & reserved),
    }


def build_cala_records(prediction_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        if "state" not in record or "base_action" not in record or "target_action" not in record:
            continue
        rows.append(
            {
                "key": _sample_key(record),
                "frame_key": _frame_key(record),
                "split": str(record.get("split", "")),
                "task": str(record.get("task", "")),
                "task_index": int(record.get("task_index", -1)),
                "episode_index": int(record.get("episode_index", -1)),
                "frame_index": int(record.get("frame_index", -1)),
                "dataset_global_index": int(record.get("dataset_global_index", record.get("index", -1))),
                "phase": _safe_float(record.get("normalized_phase", 0.0)),
                "phase_label": str(record.get("phase", "")),
                "state": _as_vector("state", record["state"], 8),
                "base_action": _as_vector("base_action", record["base_action"], 7),
                "target_action": _as_vector("target_action", record["target_action"], 7),
            }
        )
    return rows


def _action_scale(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> np.ndarray:
    train_actions = np.asarray(
        [row["target_action"] for row in records if row["split"] in set(config.train_splits)],
        dtype=np.float64,
    )
    if train_actions.size == 0:
        return np.ones(7, dtype=np.float64)
    scale = np.std(train_actions, axis=0)
    scale[scale < 1e-3] = 1.0
    return scale


def _segment_summary(segment: np.ndarray, scale_a: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float64) / scale_a.reshape(1, -1)
    first = segment[0]
    last = segment[-1]
    mean = segment.mean(axis=0)
    std = segment.std(axis=0)
    diff = last - first
    return np.concatenate([mean, first, last, diff, std])


def _attach_latent_labels(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = [dict(record) for record in records]
    scale_a = _action_scale(rows, config)
    by_episode: dict[tuple[str, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        by_episode.setdefault((row["split"], int(row["task_index"]), int(row["episode_index"])), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: int(rows[idx]["frame_index"]))
        actions = [np.asarray(rows[idx]["target_action"], dtype=np.float64) for idx in indices]
        for position, index in enumerate(indices):
            end = min(len(indices), position + config.latent_horizon)
            segment = list(actions[position:end])
            if not segment:
                rows[index]["latent_valid"] = False
                rows[index]["future_action_segment_used_for_training_only"] = True
                rows[index]["latent_action"] = None
                continue
            while len(segment) < config.latent_horizon:
                segment.append(segment[-1])
            rows[index]["latent_valid"] = True
            rows[index]["future_action_segment_used_for_training_only"] = True
            rows[index]["latent_horizon"] = config.latent_horizon
            rows[index]["latent_action"] = _segment_summary(np.vstack(segment), scale_a)
    return rows, scale_a


def _latent_for_horizon(records: Sequence[Mapping[str, Any]], horizon: int, config: CALAConfig) -> np.ndarray:
    horizon_config = CALAConfig(latent_horizon=horizon)
    rows, _ = _attach_latent_labels(records, horizon_config)
    return np.asarray([row["latent_action"] for row in rows if row.get("latent_valid")], dtype=np.float64)


def _label_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str], config: CALAConfig) -> dict[str, Any]:
    subset = [row for row in records if row["split"] in set(split_names)]
    valid = [row for row in subset if bool(row.get("latent_valid"))]
    z = np.asarray([row["latent_action"] for row in valid], dtype=np.float64) if valid else np.zeros((0, 35))
    task_counts: dict[str, int] = {}
    for row in valid:
        task_counts[str(row["task_index"])] = task_counts.get(str(row["task_index"]), 0) + 1
    max_task_share = max(task_counts.values(), default=0) / max(1, len(valid))
    variance = np.var(z, axis=0) if z.size else np.zeros(35, dtype=np.float64)
    latent_norm = np.linalg.norm(z, axis=1) if z.size else np.zeros(0, dtype=np.float64)
    if latent_norm.size:
        median = float(np.median(latent_norm))
        high_fraction = float(np.mean(latent_norm >= median))
        max_bin_share = max(high_fraction, 1.0 - high_fraction)
    else:
        median = 0.0
        max_bin_share = 1.0
    return {
        "total_records": len(subset),
        "valid_latent_count": len(valid),
        "valid_latent_fraction": len(valid) / max(1, len(subset)),
        "latent_dim": int(z.shape[1]) if z.ndim == 2 else 0,
        "latent_horizon": config.latent_horizon,
        "task_count": len(task_counts),
        "max_task_latent_share": max_task_share,
        "latent_variance": [float(x) for x in variance.tolist()],
        "latent_variance_nonzero_dims": int(np.sum(variance > 1e-8)),
        "latent_norm_mean": float(np.mean(latent_norm)) if latent_norm.size else 0.0,
        "latent_norm_p95": float(np.quantile(latent_norm, 0.95)) if latent_norm.size else 0.0,
        "latent_norm_median": median,
        "max_high_low_bin_share": max_bin_share,
        "source": "future_7d_action_segment_from_official_development_records",
    }


def _feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    include_phase: bool,
    include_state: bool,
    include_base: bool,
    include_task: bool,
    task_count: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for row in records:
        chunks: list[np.ndarray] = []
        if include_state:
            chunks.append(np.asarray(row["state"], dtype=np.float64))
        if include_base:
            base = np.asarray(row["base_action"], dtype=np.float64)
            chunks.append(base)
            chunks.append(np.asarray([np.linalg.norm(base[:3]), np.linalg.norm(base[3:6]), abs(base[6])]))
        if include_phase:
            chunks.append(np.asarray([float(row["phase"])], dtype=np.float64))
        if include_task:
            task = np.zeros(task_count, dtype=np.float64)
            task_index = int(row["task_index"])
            if 0 <= task_index < task_count:
                task[task_index] = 1.0
            chunks.append(task)
        features.append(np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float64))
    return np.vstack(features) if features else np.zeros((0, 1), dtype=np.float64)


def _standardize(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (validation - mean) / scale


def _fit_ridge_predictor(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    l2: float,
) -> np.ndarray:
    x_train = np.c_[train_features, np.ones(train_features.shape[0], dtype=np.float64)]
    x_validation = np.c_[validation_features, np.ones(validation_features.shape[0], dtype=np.float64)]
    reg = np.eye(x_train.shape[1], dtype=np.float64) * l2
    reg[-1, -1] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ train_targets
    return x_validation @ weights


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0 or target.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1))))


def _predictability_probe(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> dict[str, Any]:
    train = [row for row in records if row["split"] in set(config.train_splits) and bool(row.get("latent_valid"))]
    validation = [row for row in records if row["split"] in set(config.validation_splits) and bool(row.get("latent_valid"))]
    if not train or not validation:
        return {"valid": False, "accuracy_margin": -1.0, "reason": "missing_train_or_validation_latents"}
    task_count = max([int(row["task_index"]) for row in records] + [0]) + 1
    y_train = np.asarray([row["latent_action"] for row in train], dtype=np.float64)
    y_validation = np.asarray([row["latent_action"] for row in validation], dtype=np.float64)
    label_scale = float(np.sqrt(np.mean(np.sum((y_validation - y_train.mean(axis=0)) ** 2, axis=1))))
    label_scale = max(label_scale, config.eps)

    full_train = _feature_matrix(
        train, include_phase=False, include_state=True, include_base=True, include_task=True, task_count=task_count
    )
    full_val = _feature_matrix(
        validation, include_phase=False, include_state=True, include_base=True, include_task=True, task_count=task_count
    )
    full_train, full_val = _standardize(full_train, full_val)
    full_pred = _fit_ridge_predictor(full_train, y_train, full_val, config.ridge_l2)
    full_rmse = _rmse(full_pred, y_validation)

    baselines: dict[str, float] = {}
    baselines["train_mean"] = _rmse(np.repeat(y_train.mean(axis=0, keepdims=True), len(validation), axis=0), y_validation)
    baselines["zero_latent"] = _rmse(np.zeros_like(y_validation), y_validation)
    for name, kwargs in {
        "task_only": dict(include_phase=False, include_state=False, include_base=False, include_task=True),
        "phase_only": dict(include_phase=True, include_state=False, include_base=False, include_task=False),
        "action_history_only": dict(include_phase=False, include_state=False, include_base=True, include_task=False),
        "state_only": dict(include_phase=False, include_state=True, include_base=False, include_task=False),
    }.items():
        x_train = _feature_matrix(train, task_count=task_count, **kwargs)
        x_val = _feature_matrix(validation, task_count=task_count, **kwargs)
        x_train, x_val = _standardize(x_train, x_val)
        pred = _fit_ridge_predictor(x_train, y_train, x_val, config.ridge_l2)
        baselines[name] = _rmse(pred, y_validation)

    best_name = min(baselines, key=baselines.get)
    best_rmse = baselines[best_name]
    margin = (best_rmse - full_rmse) / label_scale
    score = max(0.0, 1.0 - full_rmse / label_scale)
    return {
        "valid": True,
        "full_probe_rmse": float(full_rmse),
        "label_scale_rmse": float(label_scale),
        "full_probe_score": float(score),
        "best_trivial_baseline": best_name,
        "best_trivial_rmse": float(best_rmse),
        "accuracy_margin": float(margin),
        "baseline_rmses": {key: float(value) for key, value in baselines.items()},
        "full_probe_uses_only_deployment_observable_features": True,
        "full_probe_features": ["observation.state", "base_action", "language_or_task_instruction_proxy"],
    }


def _gradient_smoke(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> dict[str, Any]:
    train = [row for row in records if row["split"] in set(config.train_splits) and bool(row.get("latent_valid"))]
    if not train:
        return {"valid": False, "latent_probe_gradient_norm": 0.0, "adapter_surrogate_gradient_norm": 0.0}
    subset = train[: min(64, len(train))]
    task_count = max([int(row["task_index"]) for row in records] + [0]) + 1
    x = _feature_matrix(
        subset, include_phase=False, include_state=True, include_base=True, include_task=True, task_count=task_count
    )
    y = np.asarray([row["latent_action"] for row in subset], dtype=np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    error = -y
    latent_grad = x.T @ error / max(1, len(subset))
    latent_norms = np.linalg.norm(y, axis=1)
    base_norms = np.asarray([np.linalg.norm(row["base_action"]) for row in subset], dtype=np.float64)
    adapter_norm = float(np.linalg.norm(latent_norms * base_norms) / max(1, len(subset)))
    latent_norm = float(np.linalg.norm(latent_grad))
    finite = bool(np.isfinite(latent_norm) and np.isfinite(adapter_norm))
    return {
        "valid": bool(finite and latent_norm > 0.0 and adapter_norm > 0.0),
        "latent_probe_gradient_norm": latent_norm,
        "adapter_surrogate_gradient_norm": adapter_norm,
        "largest_to_smallest_nonzero_ratio": float(max(latent_norm, adapter_norm) / max(min(latent_norm, adapter_norm), config.eps)),
        "batch_size": len(subset),
    }


def _source_gate_manifest(source_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(source_metadata or {})
    features = metadata.get("features") or {}
    feature_names = sorted(str(key) for key in features)
    has_rgb = any(name.startswith("observation.images") for name in feature_names)
    has_state = "observation.state" in feature_names
    object_like = [name for name in feature_names if "object" in name.lower() or "pose" in name.lower()]
    return {
        "legal_inference_features": list(LEGAL_INFERENCE_FEATURES),
        "used_inference_features_for_stage_0_probe": [
            "observation.state",
            "base_action",
            "language_or_task_instruction_proxy",
        ],
        "forbidden_inference_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
        "dataset_feature_names": feature_names,
        "rgb_video_available_in_dataset": bool(has_rgb),
        "state_available_in_dataset": bool(has_state),
        "object_or_pose_feature_names": object_like,
        "privileged_object_pose_available_as_dataset_feature": bool(object_like),
        "future_action_segments_used_at_inference": False,
        "latent_labels_used_at_inference": False,
        "future_action_segments_used_for_training_only": True,
        "source_gate_passed": bool(has_state and not object_like),
    }


def _horizon_health(records: Sequence[Mapping[str, Any]], config: CALAConfig) -> dict[str, Any]:
    health: dict[str, Any] = {}
    base_rows = [dict(row) for row in records]
    for horizon in config.horizon_health:
        labels = _latent_for_horizon(base_rows, horizon, config)
        variance = np.var(labels, axis=0) if labels.size else np.zeros(35, dtype=np.float64)
        norm = np.linalg.norm(labels, axis=1) if labels.size else np.zeros(0, dtype=np.float64)
        if norm.size:
            high = float(np.mean(norm >= np.median(norm)))
            max_bin = max(high, 1.0 - high)
        else:
            max_bin = 1.0
        health[str(horizon)] = {
            "record_count": int(labels.shape[0]) if labels.ndim == 2 else 0,
            "latent_dim": int(labels.shape[1]) if labels.ndim == 2 else 0,
            "latent_variance_nonzero_dims": int(np.sum(variance > 1e-8)),
            "max_high_low_bin_share": float(max_bin),
        }
    return health


def audit_cala_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    source_metadata: Mapping[str, Any] | None = None,
    config: CALAConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CALAConfig()
    base_rows = build_cala_records(prediction_records)
    rows, scale_a = _attach_latent_labels(base_rows, cfg)
    train = [row for row in rows if row["split"] in set(cfg.train_splits)]
    validation = [row for row in rows if row["split"] in set(cfg.validation_splits)]
    reserved = [row for row in rows if row["split"] in set(cfg.confirmatory_reserved_splits)]
    source_gate = _source_gate_manifest(source_metadata)
    train_labels = _label_summary(rows, cfg.train_splits, cfg)
    validation_labels = _label_summary(rows, cfg.validation_splits, cfg)
    split_manifest = {
        "train_records": len(train),
        "validation_records": len(validation),
        "reserved_records_not_used": len(reserved),
        "split_overlap": _split_overlap(rows, cfg),
        "duplicate_sample_keys": _duplicate_count([row["key"] for row in rows]),
        "duplicate_frame_keys": _duplicate_count([row["frame_key"] for row in rows]),
    }
    predictability = _predictability_probe(rows, cfg)
    gradient = _gradient_smoke(rows, cfg)
    validation_residuals = np.asarray(
        [row["target_action"] - row["base_action"] for row in validation], dtype=np.float64
    )
    base_validity = (
        float(np.mean([np.all(np.isfinite(row["base_action"])) and np.asarray(row["base_action"]).size == 7 for row in rows]))
        if rows
        else 0.0
    )
    oracle_headroom = float(np.mean(np.linalg.norm(validation_residuals, axis=1))) if validation_residuals.size else 0.0
    hard_stop_reasons: list[str] = []
    stop_class = ""

    if not source_gate["source_gate_passed"]:
        hard_stop_reasons.append("source gate failed: legal state/RGB source missing or privileged object/pose feature exposed")
        stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
    if len(rows) < cfg.min_scoreable_records:
        hard_stop_reasons.append(f"scoreable development records below minimum: {len(rows)} < {cfg.min_scoreable_records}")
        stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
    if len({row["task_index"] for row in rows}) < cfg.min_task_count:
        hard_stop_reasons.append("task coverage below minimum")
        stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
    if split_manifest["duplicate_sample_keys"] != 0 or split_manifest["duplicate_frame_keys"] != 0:
        hard_stop_reasons.append("duplicate sample or frame keys present")
        stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
    if any(value != 0 for value in split_manifest["split_overlap"].values()):
        hard_stop_reasons.append("train/validation/reserved split overlap is nonzero")
        stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"

    for split_name, summary in (("train", train_labels), ("validation", validation_labels)):
        if summary["latent_variance_nonzero_dims"] < cfg.min_nonzero_latent_dims:
            hard_stop_reasons.append(f"{split_name} latent variance has too few active dimensions")
            stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
        if summary["max_task_latent_share"] > cfg.max_task_latent_share:
            hard_stop_reasons.append(f"{split_name} latent labels dominated by one task")
            stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
        if summary["max_high_low_bin_share"] > cfg.max_latent_bin_share:
            hard_stop_reasons.append(f"{split_name} latent high/low bin collapsed: {summary['max_high_low_bin_share']:.6f}")
            stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"

    if not predictability.get("valid") or predictability.get("accuracy_margin", -1.0) < cfg.min_predictability_margin:
        hard_stop_reasons.append(
            f"latent predictability margin below minimum: {predictability.get('accuracy_margin', -1.0):.6f}"
        )
        stop_class = stop_class or "DESIGN_FAILURE"
    if oracle_headroom < cfg.min_oracle_action_headroom_l2:
        hard_stop_reasons.append(f"oracle action headroom below minimum: {oracle_headroom:.6f}")
        stop_class = stop_class or "NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE"
    if not gradient.get("valid"):
        hard_stop_reasons.append("small-batch latent/adapter gradient smoke failed")
        stop_class = stop_class or "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if base_validity < 1.0:
        hard_stop_reasons.append("base action validity below 1.0")
        stop_class = stop_class or "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"

    final_decision = stop_class or "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    latent_label_manifest = {
        "label_source": "future_7d_action_segment_from_official_development_records",
        "inference_uses_future_action_segment": False,
        "latent_encoder": "oat_lite_summary_mean_first_last_diff_std",
        "latent_horizon": cfg.latent_horizon,
        "latent_dim": train_labels["latent_dim"],
        "action_scale": [float(x) for x in scale_a.tolist()],
        "horizon_health": _horizon_health(base_rows, cfg),
        "train_label_summary": train_labels,
        "validation_label_summary": validation_labels,
    }
    report = {
        "method": "CALA-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "final_decision": final_decision,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "scoreable_development_records": len(rows),
        "train_records": len(train),
        "validation_records": len(validation),
        "reserved_records_not_used": len(reserved),
        "selected_task_count": len({row["task_index"] for row in rows}),
        "duplicate_sample_keys": split_manifest["duplicate_sample_keys"],
        "duplicate_frame_keys": split_manifest["duplicate_frame_keys"],
        "split_manifest": split_manifest,
        "source_gate_manifest": source_gate,
        "latent_label_manifest": latent_label_manifest,
        "train_latent_label_summary": train_labels,
        "validation_latent_label_summary": validation_labels,
        "latent_predictability_summary": predictability,
        "gradient_audit": gradient,
        "base_action_validity": base_validity,
        "initial_action_delta_p95": 0.0,
        "oracle_action_headroom_l2_validation": oracle_headroom,
        "hard_stop_reasons": hard_stop_reasons,
        "stage_0_completed": True,
        "stage_0_passed": final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
        "stage_0_failure_class": "" if final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" else final_decision,
        "next_step": (
            "Run bounded six-config validation search."
            if final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
            else "Record pre-rollout Stage 0 stop and continue to the next method cycle without rescuing CALA."
        ),
    }
    return report
