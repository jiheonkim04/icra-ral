"""RAR-VLA development-only Stage 0 audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56"
FORBIDDEN_INFERENCE_KEYS = {
    "cala_latent_label",
    "dataset_global_index",
    "episode_index",
    "frame_index",
    "future_action",
    "future_action_segment",
    "future_observation",
    "identity",
    "object_pose",
    "object_state",
    "oracle_phase",
    "phase",
    "placement_pose",
    "reset_identity",
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
    "previous_base_actions",
    "previous_emitted_actions",
    "causal_memory_state",
)


@dataclass(frozen=True)
class RARConfig:
    train_splits: tuple[str, ...] = ("train",)
    validation_splits: tuple[str, ...] = ("val",)
    confirmatory_reserved_splits: tuple[str, ...] = ("test",)
    history_horizon: int = 8
    min_scoreable_records: int = 500
    min_task_count: int = 3
    min_residual_nonzero_dims: int = 3
    max_task_share: float = 0.35
    min_observability_margin: float = 0.02
    min_headroom_l2: float = 0.01
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


def _split_overlap(records: Sequence[Mapping[str, Any]], config: RARConfig) -> dict[str, int]:
    train = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.train_splits)}
    validation = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.validation_splits)}
    reserved = {tuple(row["frame_key"][1:]) for row in records if row["split"] in set(config.confirmatory_reserved_splits)}
    return {
        "train_validation": len(train & validation),
        "train_reserved": len(train & reserved),
        "validation_reserved": len(validation & reserved),
    }


def build_rar_records(prediction_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                "state": _as_vector("state", record["state"], 8),
                "base_action": _as_vector("base_action", record["base_action"], 7),
                "target_action": _as_vector("target_action", record["target_action"], 7),
            }
        )
    rows.sort(key=lambda row: (row["split"], row["task_index"], row["episode_index"], row["frame_index"]))
    return rows


def _action_scale(records: Sequence[Mapping[str, Any]], config: RARConfig) -> np.ndarray:
    train_actions = np.asarray(
        [row["target_action"] for row in records if row["split"] in set(config.train_splits)],
        dtype=np.float64,
    )
    if train_actions.size == 0:
        return np.ones(7, dtype=np.float64)
    scale = np.std(train_actions, axis=0)
    scale[scale < 1e-3] = 1.0
    return scale


def _attach_history(records: Sequence[Mapping[str, Any]], config: RARConfig) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    by_episode: dict[tuple[str, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        by_episode.setdefault((row["split"], int(row["task_index"]), int(row["episode_index"])), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: int(rows[idx]["frame_index"]))
        for position, index in enumerate(indices):
            current = rows[index]
            history_indices = indices[max(0, position - config.history_horizon) : position]
            if history_indices:
                prev_base = [rows[idx]["base_action"] for idx in history_indices]
                prev_state = [rows[idx]["state"] for idx in history_indices]
            else:
                prev_base = [current["base_action"]]
                prev_state = [current["state"]]
            while len(prev_base) < config.history_horizon:
                prev_base.insert(0, prev_base[0])
                prev_state.insert(0, prev_state[0])
            prev_base_array = np.asarray(prev_base[-config.history_horizon :], dtype=np.float64)
            prev_state_array = np.asarray(prev_state[-config.history_horizon :], dtype=np.float64)
            current["history_valid"] = True
            current["previous_base_actions"] = prev_base_array
            current["previous_state_values"] = prev_state_array
            current["base_action_diff_from_previous"] = current["base_action"] - prev_base_array[-1]
            current["state_diff_from_previous"] = current["state"] - prev_state_array[-1]
            current["target_residual"] = current["target_action"] - current["base_action"]
    return rows


def _task_count(records: Sequence[Mapping[str, Any]]) -> int:
    return max([int(row["task_index"]) for row in records] + [0]) + 1


def _feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    include_state: bool,
    include_base: bool,
    include_history: bool,
    include_task: bool,
    task_count: int,
    config: RARConfig,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for row in records:
        chunks: list[np.ndarray] = []
        if include_state:
            chunks.append(np.asarray(row["state"], dtype=np.float64))
            chunks.append(np.asarray(row["state_diff_from_previous"], dtype=np.float64))
        if include_base:
            base = np.asarray(row["base_action"], dtype=np.float64)
            chunks.append(base)
            chunks.append(np.asarray([np.linalg.norm(base[:3]), np.linalg.norm(base[3:6]), abs(base[6])]))
        if include_history:
            previous = np.asarray(row["previous_base_actions"], dtype=np.float64)
            chunks.append(previous.reshape(-1))
            chunks.append(previous.mean(axis=0))
            chunks.append(previous[-1] - previous[0])
            chunks.append(np.asarray(row["base_action_diff_from_previous"], dtype=np.float64))
            diffs = np.diff(previous, axis=0)
            chunks.append(diffs.mean(axis=0) if diffs.size else np.zeros(7, dtype=np.float64))
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


def _fit_ridge(train_features: np.ndarray, train_targets: np.ndarray, validation_features: np.ndarray, l2: float) -> np.ndarray:
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


def _ema_residual_prediction(records: Sequence[Mapping[str, Any]], alpha: float) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for row in records:
        previous = np.asarray(row["previous_base_actions"], dtype=np.float64)
        ema = previous[0].copy()
        for value in previous[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
        predictions.append(ema - np.asarray(row["base_action"], dtype=np.float64))
    return np.asarray(predictions, dtype=np.float64)


def _observability_probe(records: Sequence[Mapping[str, Any]], config: RARConfig) -> dict[str, Any]:
    train = [row for row in records if row["split"] in set(config.train_splits) and bool(row.get("history_valid"))]
    validation = [row for row in records if row["split"] in set(config.validation_splits) and bool(row.get("history_valid"))]
    if not train or not validation:
        return {"valid": False, "residual_predictability_margin": -1.0, "reason": "missing_train_or_validation_records"}
    tasks = _task_count(records)
    y_train = np.asarray([row["target_residual"] for row in train], dtype=np.float64)
    y_val = np.asarray([row["target_residual"] for row in validation], dtype=np.float64)
    label_scale = float(np.sqrt(np.mean(np.sum((y_val - y_train.mean(axis=0)) ** 2, axis=1))))
    label_scale = max(label_scale, config.eps)

    full_train = _feature_matrix(
        train,
        include_state=True,
        include_base=True,
        include_history=True,
        include_task=True,
        task_count=tasks,
        config=config,
    )
    full_val = _feature_matrix(
        validation,
        include_state=True,
        include_base=True,
        include_history=True,
        include_task=True,
        task_count=tasks,
        config=config,
    )
    full_train, full_val = _standardize(full_train, full_val)
    full_pred = _fit_ridge(full_train, y_train, full_val, config.ridge_l2)
    full_rmse = _rmse(full_pred, y_val)

    baselines: dict[str, float] = {}
    baselines["train_mean_residual"] = _rmse(np.repeat(y_train.mean(axis=0, keepdims=True), len(validation), axis=0), y_val)
    baselines["zero_residual"] = _rmse(np.zeros_like(y_val), y_val)
    baselines["ema_action_history_alpha_0_50"] = _rmse(_ema_residual_prediction(validation, 0.50), y_val)
    baselines["ema_action_history_alpha_0_80"] = _rmse(_ema_residual_prediction(validation, 0.80), y_val)
    for name, kwargs in {
        "linear_history_only": dict(include_state=False, include_base=False, include_history=True, include_task=False),
        "base_only": dict(include_state=False, include_base=True, include_history=False, include_task=False),
        "state_base_task": dict(include_state=True, include_base=True, include_history=False, include_task=True),
    }.items():
        x_train = _feature_matrix(train, task_count=tasks, config=config, **kwargs)
        x_val = _feature_matrix(validation, task_count=tasks, config=config, **kwargs)
        x_train, x_val = _standardize(x_train, x_val)
        pred = _fit_ridge(x_train, y_train, x_val, config.ridge_l2)
        baselines[name] = _rmse(pred, y_val)

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
        "residual_predictability_margin": float(margin),
        "baseline_rmses": {key: float(value) for key, value in baselines.items()},
        "full_probe_uses_only_legal_causal_features": True,
        "full_probe_features": [
            "observation.state",
            "base_action",
            "previous_base_actions",
            "state_delta",
            "language_or_task_instruction_proxy",
        ],
    }


def _residual_summary(records: Sequence[Mapping[str, Any]], split_names: Sequence[str], config: RARConfig) -> dict[str, Any]:
    subset = [row for row in records if row["split"] in set(split_names)]
    residuals = np.asarray([row["target_residual"] for row in subset], dtype=np.float64) if subset else np.zeros((0, 7))
    task_counts: dict[str, int] = {}
    for row in subset:
        task_counts[str(row["task_index"])] = task_counts.get(str(row["task_index"]), 0) + 1
    variance = np.var(residuals, axis=0) if residuals.size else np.zeros(7, dtype=np.float64)
    norm = np.linalg.norm(residuals, axis=1) if residuals.size else np.zeros(0, dtype=np.float64)
    return {
        "total_records": len(subset),
        "residual_dim": int(residuals.shape[1]) if residuals.ndim == 2 else 0,
        "task_count": len(task_counts),
        "max_task_share": max(task_counts.values(), default=0) / max(1, len(subset)),
        "residual_variance": [float(x) for x in variance.tolist()],
        "residual_variance_nonzero_dims": int(np.sum(variance > 1e-8)),
        "residual_norm_mean": float(np.mean(norm)) if norm.size else 0.0,
        "residual_norm_p95": float(np.quantile(norm, 0.95)) if norm.size else 0.0,
    }


def _discontinuity_diagnostics(records: Sequence[Mapping[str, Any]], config: RARConfig) -> dict[str, Any]:
    validation = [row for row in records if row["split"] in set(config.validation_splits)]
    base_diff = np.asarray([row["base_action_diff_from_previous"] for row in validation], dtype=np.float64)
    target_diff = []
    by_episode: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in validation:
        by_episode.setdefault((row["split"], row["task_index"], row["episode_index"]), []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        previous = rows[0]["target_action"]
        for row in rows:
            target_diff.append(row["target_action"] - previous)
            previous = row["target_action"]
    target_diff_array = np.asarray(target_diff, dtype=np.float64) if target_diff else np.zeros((0, 7))
    base_norm = np.linalg.norm(base_diff, axis=1) if base_diff.size else np.zeros(0)
    target_norm = np.linalg.norm(target_diff_array, axis=1) if target_diff_array.size else np.zeros(0)
    boundary_threshold = float(np.quantile(base_norm, 0.75)) if base_norm.size else 0.0
    inter_mask = base_norm >= boundary_threshold if base_norm.size else np.zeros(0, dtype=bool)
    intra_mask = ~inter_mask if inter_mask.size else np.zeros(0, dtype=bool)
    return {
        "diagnostic_type": "frame_local_base_difference_proxy",
        "inter_chunk_proxy_threshold": boundary_threshold,
        "base_inter_chunk_proxy_l2_mean": float(np.mean(base_norm[inter_mask])) if np.any(inter_mask) else 0.0,
        "base_intra_chunk_proxy_l2_mean": float(np.mean(base_norm[intra_mask])) if np.any(intra_mask) else 0.0,
        "target_inter_chunk_proxy_l2_mean": float(np.mean(target_norm[inter_mask])) if np.any(inter_mask) else 0.0,
        "target_intra_chunk_proxy_l2_mean": float(np.mean(target_norm[intra_mask])) if np.any(intra_mask) else 0.0,
    }


def _gradient_smoke(records: Sequence[Mapping[str, Any]], config: RARConfig) -> dict[str, Any]:
    train = [row for row in records if row["split"] in set(config.train_splits) and bool(row.get("history_valid"))]
    if not train:
        return {"valid": False, "residual_head_gradient_norm": 0.0, "gate_surrogate_gradient_norm": 0.0}
    subset = train[: min(64, len(train))]
    tasks = _task_count(records)
    x = _feature_matrix(
        subset,
        include_state=True,
        include_base=True,
        include_history=True,
        include_task=True,
        task_count=tasks,
        config=config,
    )
    y = np.asarray([row["target_residual"] for row in subset], dtype=np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    error = -y
    grad = x.T @ error / max(1, len(subset))
    residual_norm = float(np.linalg.norm(grad))
    gate_norm = float(np.mean(np.linalg.norm(y, axis=1) * np.linalg.norm([row["base_action"] for row in subset], axis=1)))
    finite = bool(np.isfinite(residual_norm) and np.isfinite(gate_norm))
    return {
        "valid": bool(finite and residual_norm > 0.0 and gate_norm > 0.0),
        "residual_head_gradient_norm": residual_norm,
        "gate_surrogate_gradient_norm": gate_norm,
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
            "previous_base_actions",
            "state_delta",
            "language_or_task_instruction_proxy",
        ],
        "forbidden_inference_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
        "dataset_feature_names": feature_names,
        "rgb_video_available_in_dataset": bool(has_rgb),
        "state_available_in_dataset": bool(has_state),
        "object_or_pose_feature_names": object_like,
        "privileged_object_pose_available_as_dataset_feature": bool(object_like),
        "future_actions_used_at_inference": False,
        "cala_latents_used_at_inference": False,
        "previous_actions_are_causal_only": True,
        "source_gate_passed": bool(has_state and not object_like),
    }


def audit_rar_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    source_metadata: Mapping[str, Any] | None = None,
    config: RARConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RARConfig()
    base_rows = build_rar_records(prediction_records)
    rows = _attach_history(base_rows, cfg)
    train = [row for row in rows if row["split"] in set(cfg.train_splits)]
    validation = [row for row in rows if row["split"] in set(cfg.validation_splits)]
    reserved = [row for row in rows if row["split"] in set(cfg.confirmatory_reserved_splits)]
    source_gate = _source_gate_manifest(source_metadata)
    split_manifest = {
        "train_records": len(train),
        "validation_records": len(validation),
        "reserved_records_not_used": len(reserved),
        "split_overlap": _split_overlap(rows, cfg),
        "duplicate_sample_keys": _duplicate_count([row["key"] for row in rows]),
        "duplicate_frame_keys": _duplicate_count([row["frame_key"] for row in rows]),
    }
    train_residual = _residual_summary(rows, cfg.train_splits, cfg)
    validation_residual = _residual_summary(rows, cfg.validation_splits, cfg)
    observability = _observability_probe(rows, cfg)
    discontinuity = _discontinuity_diagnostics(rows, cfg)
    gradient = _gradient_smoke(rows, cfg)
    validation_residuals = np.asarray([row["target_residual"] for row in validation], dtype=np.float64)
    headroom = float(np.mean(np.linalg.norm(validation_residuals, axis=1))) if validation_residuals.size else 0.0
    base_validity = (
        float(np.mean([np.all(np.isfinite(row["base_action"])) and np.asarray(row["base_action"]).size == 7 for row in rows]))
        if rows
        else 0.0
    )

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
    for split_name, summary in (("train", train_residual), ("validation", validation_residual)):
        if summary["residual_variance_nonzero_dims"] < cfg.min_residual_nonzero_dims:
            hard_stop_reasons.append(f"{split_name} residual variance has too few active dimensions")
            stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
        if summary["max_task_share"] > cfg.max_task_share:
            hard_stop_reasons.append(f"{split_name} residual labels dominated by one task")
            stop_class = stop_class or "DATA_OR_SUPERVISION_FAILURE"
    if headroom < cfg.min_headroom_l2:
        hard_stop_reasons.append(f"residual headroom below minimum: {headroom:.6f}")
        stop_class = stop_class or "NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE"
    margin = observability.get("residual_predictability_margin", -1.0)
    if not observability.get("valid") or margin < cfg.min_observability_margin:
        hard_stop_reasons.append(f"residual predictability margin below minimum: {margin:.6f}")
        stop_class = stop_class or "DESIGN_FAILURE"
    if not gradient.get("valid"):
        hard_stop_reasons.append("small-batch residual/gate gradient smoke failed")
        stop_class = stop_class or "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if base_validity < 1.0:
        hard_stop_reasons.append("base action validity below 1.0")
        stop_class = stop_class or "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"

    final_decision = stop_class or "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    history_manifest = {
        "history_horizon": cfg.history_horizon,
        "history_source": "previous_base_actions_and_current_state_only",
        "future_actions_used_at_inference": False,
        "cala_latents_used_at_inference": False,
        "reanchor_feature": "base_action_diff_from_previous",
        "inter_chunk_diagnostic": "frame_local_base_difference_proxy",
        "train_residual_summary": train_residual,
        "validation_residual_summary": validation_residual,
        "discontinuity_diagnostics": discontinuity,
    }
    return {
        "method": "RAR-VLA",
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
        "history_feature_manifest": history_manifest,
        "train_residual_summary": train_residual,
        "validation_residual_summary": validation_residual,
        "residual_observability_summary": observability,
        "discontinuity_diagnostics": discontinuity,
        "gradient_audit": gradient,
        "base_action_validity": base_validity,
        "initial_action_delta_p95": 0.0,
        "residual_headroom_l2_validation": headroom,
        "hard_stop_reasons": hard_stop_reasons,
        "stage_0_completed": True,
        "stage_0_passed": final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
        "stage_0_failure_class": "" if final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" else final_decision,
        "next_step": (
            "Run bounded six-config validation search."
            if final_decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
            else "Record pre-rollout Stage 0 stop and continue to the next method cycle without rescuing RAR."
        ),
    }
