"""Frozen TSC-VLA Stage 0 temporal-spatial completion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941"
ACTION_DIM = 7
PROPRIO_DIM = 8
VISUAL_FEATURE_DIM = 960
TASK_COUNT = 4
PHASE_BINS = 10
CHUNK_SIZE = 50
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0
MASK_QUANTILE = 0.80
MASK_THRESHOLD = 0.5
DIAGNOSTIC_ALPHA = 0.1
RIDGE_COEFFICIENT = 1e-4
HEADROOM_RELATIVE_GATE = 0.05
HEADROOM_ABSOLUTE_HUBER_GATE = 0.005
MAX_CHANGED_CELL_FRACTION = 0.60


def json_default(value: Any) -> Any:
    """Convert supported scientific values into strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=json_default,
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def tsc_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["proxy_variant"],
        row["policy_probe"],
    )
    return "|".join(str(value) for value in fields)


def tsc_feature_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        "tsc_current",
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [tsc_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["suite"],
            row["task_identity"],
            row["source_edge_sha256"],
            row["demo_id"],
            row["frame_index"],
            row["proxy_variant"],
            row["policy_probe"],
        )

    discovery = {split_identity(row) for row in manifest_rows if row["partition"] == "discovery"}
    validation = {split_identity(row) for row in manifest_rows if row["partition"] == "validation"}
    return {
        "manifest_row_count": len(expected),
        "partial_row_count": len(completed),
        "duplicate_manifest_key_count": len(expected) - len(expected_set),
        "duplicate_partial_key_count": len(completed) - len(completed_set),
        "missing_manifest_key_count": len(expected_set - completed_set),
        "extra_partial_key_count": len(completed_set - expected_set),
        "split_overlap_key_count": len(discovery & validation),
        "key_sets_equal": expected_set == completed_set,
    }


def action_chunk(actions: Any, frame_index: int, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    frame = int(frame_index)
    length = int(chunk_size)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTION_DIM}], received {value.shape}")
    if frame < 0 or frame + length > len(value):
        raise ValueError("frame must have a complete action chunk")
    chunk = value[frame : frame + length]
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains nonfinite values")
    return chunk


def phase_bin(phase: float, bins: int = PHASE_BINS) -> int:
    if not np.isfinite(float(phase)):
        raise ValueError("phase must be finite")
    clipped = min(max(float(phase), 0.0), 1.0)
    return min(int(np.floor(clipped * int(bins))), int(bins) - 1)


def one_hot(index: int, width: int = TASK_COUNT) -> np.ndarray:
    if int(index) < 0 or int(index) >= int(width):
        raise ValueError(f"index must be in [0,{width})")
    value = np.zeros(int(width), dtype=np.float64)
    value[int(index)] = 1.0
    return value


def raw_tsc_feature(visual: Any, proprio: Any, task_index: int, phase: float) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float64).reshape(-1)
    proprio_value = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if visual_value.shape != (VISUAL_FEATURE_DIM,):
        raise ValueError(f"visual feature must have shape [{VISUAL_FEATURE_DIM}], received {visual_value.shape}")
    if proprio_value.shape != (PROPRIO_DIM,):
        raise ValueError(f"proprio feature must have shape [{PROPRIO_DIM}], received {proprio_value.shape}")
    continuous = np.concatenate([visual_value, proprio_value, np.asarray([float(phase)], dtype=np.float64)])
    if not np.isfinite(continuous).all():
        raise ValueError("TSC feature contains nonfinite values")
    return np.concatenate([continuous, one_hot(task_index)])


def fit_discovery_zscore(features: Any) -> dict[str, Any]:
    value = _matrix(features, name="features", width=VISUAL_FEATURE_DIM + PROPRIO_DIM + 1 + TASK_COUNT)
    continuous_dim = VISUAL_FEATURE_DIM + PROPRIO_DIM + 1
    continuous = value[:, :continuous_dim]
    return {
        "continuous_dim": continuous_dim,
        "task_count": TASK_COUNT,
        "mean": continuous.mean(axis=0),
        "std": np.maximum(continuous.std(axis=0, ddof=0), STD_FLOOR),
        "discovery_row_count": int(len(value)),
    }


def apply_discovery_zscore(stats: Mapping[str, Any], features: Any) -> np.ndarray:
    value = _matrix(features, name="features", width=int(stats["continuous_dim"]) + int(stats["task_count"]))
    continuous_dim = int(stats["continuous_dim"])
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, continuous_dim)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, continuous_dim)
    continuous = (value[:, :continuous_dim] - mean) / std
    return np.concatenate([continuous, value[:, continuous_dim:]], axis=1)


def flattened_chunks(chunks: Any) -> np.ndarray:
    value = _chunk_matrix(chunks)
    return value.reshape(value.shape[0], CHUNK_SIZE * ACTION_DIM)


def mean_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    values = np.where(absolute <= delta, 0.5 * np.square(error), delta * (absolute - 0.5 * delta))
    return float(np.mean(values))


def prediction_metrics(prediction: Any, baseline: Any, target: Any) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred_mse = float(np.mean(np.square(pred - truth)))
    base_mse = float(np.mean(np.square(base - truth)))
    pred_huber = mean_huber(pred, truth)
    base_huber = mean_huber(base, truth)
    return {
        "prediction_mse": pred_mse,
        "baseline_mse": base_mse,
        "relative_mse_improvement": float((base_mse - pred_mse) / max(base_mse, 1e-12)),
        "prediction_huber": pred_huber,
        "baseline_huber": base_huber,
        "absolute_huber_improvement": float(base_huber - pred_huber),
    }


def fit_ridge(
    features: Any,
    targets: Any,
    *,
    ridge: float = RIDGE_COEFFICIENT,
    std_floor: float = STD_FLOOR,
) -> dict[str, Any]:
    x = _matrix(features, name="features")
    y = _matrix(targets, name="targets")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("features and targets require aligned rows and at least two samples")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_std = np.maximum(x.std(axis=0, ddof=0), float(std_floor))
    y_std = np.maximum(y.std(axis=0, ddof=0), float(std_floor))
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    design = np.concatenate([x_norm, np.ones((len(x_norm), 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_norm)
    return {
        "ridge_coefficient": float(ridge),
        "std_floor": float(std_floor),
        "input_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "feature_mean": x_mean,
        "feature_std": x_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "beta": beta,
        "rank": int(np.linalg.matrix_rank(beta[:-1])),
        "discovery_row_count": int(len(x)),
    }


def predict_ridge(model: Mapping[str, Any], features: Any, *, normalized: bool = False) -> np.ndarray:
    x = _matrix(features, name="features", width=int(model["input_dim"]))
    x_mean = np.asarray(model["feature_mean"], dtype=np.float64).reshape(1, int(model["input_dim"]))
    x_std = np.asarray(model["feature_std"], dtype=np.float64).reshape(1, int(model["input_dim"]))
    beta = np.asarray(model["beta"], dtype=np.float64)
    design = np.concatenate([(x - x_mean) / x_std, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    y_norm = design @ beta
    if normalized:
        return y_norm
    y_mean = np.asarray(model["target_mean"], dtype=np.float64).reshape(1, int(model["target_dim"]))
    y_std = np.asarray(model["target_std"], dtype=np.float64).reshape(1, int(model["target_dim"]))
    return y_norm * y_std + y_mean


def fit_mask_label_stats(residual_chunks: Any, valid_mask: Any | None = None) -> dict[str, Any]:
    residual = _chunk_matrix(residual_chunks)
    valid = _valid_mask(valid_mask, len(residual))
    valid_residual = residual[np.broadcast_to(valid, residual.shape)]
    if valid_residual.size == 0:
        raise ValueError("mask labels require at least one valid residual cell")
    scale = np.median(np.abs(residual)[np.broadcast_to(valid, residual.shape)].reshape(-1, ACTION_DIM), axis=0)
    scale = np.maximum(scale, 0.0) + STD_FLOOR
    normalized = np.abs(residual) / scale.reshape(1, 1, ACTION_DIM)
    tau = float(np.quantile(normalized[np.broadcast_to(valid, normalized.shape)], MASK_QUANTILE))
    labels = normalized >= tau
    labels = np.logical_and(labels, np.broadcast_to(valid, labels.shape))
    return {
        "scale_by_action_dim": scale,
        "tau": tau,
        "mask_quantile": MASK_QUANTILE,
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "positive_rate": float(labels.mean()),
    }


def make_error_mask_labels(residual_chunks: Any, stats: Mapping[str, Any], valid_mask: Any | None = None) -> np.ndarray:
    residual = _chunk_matrix(residual_chunks)
    valid = _valid_mask(valid_mask, len(residual))
    scale = np.asarray(stats["scale_by_action_dim"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    normalized = np.abs(residual) / scale
    labels = normalized >= float(stats["tau"])
    return np.logical_and(labels, np.broadcast_to(valid, labels.shape))


def fit_structured_mask_probe(features: Any, labels: Any) -> dict[str, Any]:
    label_matrix = _label_matrix(labels)
    model = fit_ridge(features, label_matrix)
    return {
        "kind": "structured_temporal_spatial_mask_probe",
        "model": model,
        "threshold": MASK_THRESHOLD,
        "label_shape": [CHUNK_SIZE, ACTION_DIM],
        "model_hash": canonical_json_sha256({"kind": "structured_temporal_spatial_mask_probe", "model": model}),
    }


def predict_structured_mask_scores(model: Mapping[str, Any], features: Any) -> np.ndarray:
    raw = predict_ridge(model["model"], features).reshape((-1, CHUNK_SIZE, ACTION_DIM))
    return np.clip(raw, 1e-6, 1.0 - 1e-6)


def hard_mask(scores: Any, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"mask scores must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], received {value.shape}")
    return value >= float(threshold)


def binary_cross_entropy(probability: Any, labels: Any) -> float:
    prob = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    truth = np.asarray(labels, dtype=np.float64)
    if prob.shape != truth.shape:
        raise ValueError(f"probability and labels must align, received {prob.shape} and {truth.shape}")
    return float(-np.mean(truth * np.log(prob) + (1.0 - truth) * np.log(1.0 - prob)))


def trivial_mask_probability(labels: Any) -> np.ndarray:
    value = _label_matrix(labels)
    return np.clip(value.mean(axis=0).reshape(1, CHUNK_SIZE, ACTION_DIM), 1e-6, 1.0 - 1e-6)


def fit_magnitude_mask_baseline(base_chunks: Any, labels: Any) -> dict[str, Any]:
    base = _chunk_matrix(base_chunks)
    truth = np.asarray(labels, dtype=bool)
    if truth.shape != base.shape:
        raise ValueError("magnitude baseline labels must align with base chunks")
    magnitude = np.abs(base)
    scale = np.maximum(np.median(magnitude.reshape(-1, ACTION_DIM), axis=0), STD_FLOOR)
    score = magnitude / scale.reshape(1, 1, ACTION_DIM)
    positive_rate = float(np.mean(truth))
    threshold = float(np.quantile(score.reshape(-1), max(0.0, min(1.0, 1.0 - positive_rate))))
    return {
        "kind": "base_action_magnitude_only_mask",
        "scale_by_action_dim": scale,
        "threshold": threshold,
        "positive_rate": positive_rate,
    }


def predict_magnitude_mask_probability(model: Mapping[str, Any], base_chunks: Any) -> np.ndarray:
    base = _chunk_matrix(base_chunks)
    scale = np.asarray(model["scale_by_action_dim"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    score = np.abs(base) / scale
    centered = score - float(model["threshold"])
    probability = 1.0 / (1.0 + np.exp(-centered))
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def fit_completion_model(features: Any, base_chunks: Any, expert_chunks: Any) -> dict[str, Any]:
    feature_matrix = _matrix(features, name="features")
    base = _chunk_matrix(base_chunks)
    expert = _chunk_matrix(expert_chunks)
    if len(feature_matrix) != len(base) or len(base) != len(expert):
        raise ValueError("completion model inputs must align")
    model_input = np.concatenate([feature_matrix, flattened_chunks(base)], axis=1)
    target = flattened_chunks(expert - base)
    model = fit_ridge(model_input, target)
    return {
        "kind": "continuous_masked_completion_ridge",
        "feature_dim": int(feature_matrix.shape[1]),
        "chunk_shape": [CHUNK_SIZE, ACTION_DIM],
        "model": model,
        "model_hash": canonical_json_sha256({"kind": "continuous_masked_completion_ridge", "model": model}),
    }


def predict_completion_residual(model: Mapping[str, Any], features: Any, base_chunks: Any) -> np.ndarray:
    feature_matrix = _matrix(features, name="features", width=int(model["feature_dim"]))
    base = _chunk_matrix(base_chunks)
    model_input = np.concatenate([feature_matrix, flattened_chunks(base)], axis=1)
    residual = predict_ridge(model["model"], model_input)
    return residual.reshape((-1, CHUNK_SIZE, ACTION_DIM))


def deterministic_random_mask(row_keys: Sequence[str], rate: float, *, salt: str = "ts_mask_proxy") -> np.ndarray:
    clipped_rate = max(0.0, min(float(rate), 1.0))
    values = np.empty((len(row_keys), CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    for row_index, key in enumerate(row_keys):
        for time_index in range(CHUNK_SIZE):
            for action_dim in range(ACTION_DIM):
                token = f"{salt}|{key}|{time_index}|{action_dim}".encode("utf-8")
                integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
                values[row_index, time_index, action_dim] = integer / float(2**64 - 1)
    return values < clipped_rate


def apply_masked_completion(
    base_chunks: Any,
    residual_prediction: Any,
    mask: Any,
    *,
    alpha: float = DIAGNOSTIC_ALPHA,
) -> np.ndarray:
    base = _chunk_matrix(base_chunks)
    residual = _chunk_matrix(residual_prediction)
    mask_value = np.asarray(mask, dtype=bool)
    if mask_value.shape != base.shape:
        raise ValueError(f"mask must have shape {base.shape}, received {mask_value.shape}")
    return base + mask_value.astype(np.float64) * float(alpha) * residual


def unselected_clamp_error(base_chunks: Any, completed_chunks: Any, mask: Any) -> float:
    base = _chunk_matrix(base_chunks)
    completed = _chunk_matrix(completed_chunks)
    mask_value = np.asarray(mask, dtype=bool)
    if mask_value.shape != base.shape:
        raise ValueError("mask and chunks must align")
    if bool(np.all(mask_value)):
        return 0.0
    return float(np.max(np.abs((completed - base)[~mask_value])))


def action_delta_summary(base_chunks: Any, completed_chunks: Any, mask: Any) -> dict[str, Any]:
    base = _chunk_matrix(base_chunks)
    completed = _chunk_matrix(completed_chunks)
    delta = completed - base
    mask_value = np.asarray(mask, dtype=bool)
    if mask_value.shape != base.shape:
        raise ValueError("mask and chunks must align")
    translation = np.linalg.norm(delta[:, :, :3], axis=2)
    rotation = np.linalg.norm(delta[:, :, 3:6], axis=2)
    gripper = np.abs(delta[:, :, 6])

    def stats(value: np.ndarray) -> dict[str, float]:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "p95": float(np.quantile(flat, 0.95)),
            "max": float(np.max(flat)),
        }

    return {
        "changed_cell_fraction": float(mask_value.mean()),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta": stats(translation),
        "rotation_delta": stats(rotation),
        "gripper_delta": stats(gripper),
        "changed_dimensions": [
            int(index) for index in range(ACTION_DIM) if float(np.max(np.abs(delta[:, :, index]))) > 1e-12
        ],
    }


@dataclass(frozen=True)
class Stage0DecisionInputs:
    proposal_hash_ok: bool
    serializer_preflight_ok: bool
    official_prior_asset_check_persisted: bool
    manifest_integrity_ok: bool
    source_alignment_ok: bool
    feature_action_proprio_finite_aligned: bool
    minimum_discovery_windows: int
    minimum_validation_windows: int
    all_tasks_reported: bool
    maximum_validation_task_fraction: float
    labels_noncollapsed_discovery: bool
    labels_noncollapsed_validation: bool
    structured_mask_beats_trivial: bool
    structured_mask_beats_magnitude: bool
    tsc_beats_prior_relative: float
    tsc_beats_prior_absolute_huber: float
    tsc_beats_ablation_relative: float
    tsc_beats_ablation_absolute_huber: float
    unselected_cell_clamp_max_error: float
    changed_cell_fraction: float
    deltas_finite_and_bounded: bool
    tsc_distinct_from_prior_and_ablation: bool
    finite_objectives_and_gradients: bool
    tsc_gradient_nonzero: bool
    gradient_ratio_at_most_100: bool
    frozen_parameter_gradient_count: int
    identity_max_error: float
    base_hash_unchanged: bool
    checkpoint_reload_ok: bool
    action_validity_ok: bool
    reward_read_count: int
    success_read_count: int
    done_read_count: int
    confirmatory_records_read: int
    exception_count: int


def classify_stage0(inputs: Stage0DecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or not inputs.official_prior_asset_check_persisted
        or float(inputs.unselected_cell_clamp_max_error) > 1e-6
        or float(inputs.identity_max_error) > 1e-6
        or not inputs.base_hash_unchanged
        or not inputs.checkpoint_reload_ok
        or not inputs.action_validity_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.tsc_gradient_nonzero
        or not inputs.gradient_ratio_at_most_100
        or int(inputs.frozen_parameter_gradient_count) != 0
        or int(inputs.reward_read_count) != 0
        or int(inputs.success_read_count) != 0
        or int(inputs.done_read_count) != 0
        or int(inputs.confirmatory_records_read) != 0
        or int(inputs.exception_count) != 0
    ):
        return "TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < 512
        or int(inputs.minimum_validation_windows) < 128
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.labels_noncollapsed_discovery
        or not inputs.labels_noncollapsed_validation
    ):
        return "TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        not inputs.structured_mask_beats_trivial
        or not inputs.structured_mask_beats_magnitude
        or not _headroom_gate(inputs.tsc_beats_prior_relative, inputs.tsc_beats_prior_absolute_huber)
        or not _headroom_gate(inputs.tsc_beats_ablation_relative, inputs.tsc_beats_ablation_absolute_huber)
    ):
        return "TSC_STAGE_0_NO_USABLE_HEADROOM"
    if (
        float(inputs.changed_cell_fraction) <= 0.0
        or float(inputs.changed_cell_fraction) >= MAX_CHANGED_CELL_FRACTION
        or not inputs.deltas_finite_and_bounded
        or not inputs.tsc_distinct_from_prior_and_ablation
    ):
        return "TSC_STAGE_0_DESIGN_FAILURE"
    return "TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _headroom_gate(relative: float, absolute_huber: float) -> bool:
    return float(relative) >= HEADROOM_RELATIVE_GATE or float(absolute_huber) >= HEADROOM_ABSOLUTE_HUBER_GATE


def _matrix(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix, received {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, received {array.shape[1]}")
    return array


def _chunk_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(array).all():
        raise ValueError(f"chunks must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], received {array.shape}")
    return array


def _label_matrix(labels: Any) -> np.ndarray:
    value = np.asarray(labels, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"labels must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], received {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("labels contain nonfinite values")
    return value.reshape(value.shape[0], CHUNK_SIZE * ACTION_DIM)


def _valid_mask(valid_mask: Any | None, row_count: int) -> np.ndarray:
    if valid_mask is None:
        return np.ones((int(row_count), CHUNK_SIZE, 1), dtype=bool)
    value = np.asarray(valid_mask, dtype=bool)
    if value.shape == (int(row_count), CHUNK_SIZE):
        value = value[:, :, None]
    if value.shape != (int(row_count), CHUNK_SIZE, 1):
        raise ValueError(f"valid mask must have shape [N,{CHUNK_SIZE},1], received {value.shape}")
    return value
