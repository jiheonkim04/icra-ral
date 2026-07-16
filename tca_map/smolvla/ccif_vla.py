"""Frozen CCIF-VLA Stage 0 continuous coarse-intent helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1"
ACTION_DIM = 7
PROPRIO_DIM = 8
VISUAL_FEATURE_DIM = 960
TASK_COUNT = 4
PHASE_BINS = 10
CHUNK_SIZE = 50
WAYPOINT_INDICES = (9, 19, 34, 49)
INTENT_DIM = 31
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0
ACTION_HUBER_DELTA = 0.05
RIDGE_COEFFICIENT = 1e-4
HEADROOM_RELATIVE_GATE = 0.05
HEADROOM_ABSOLUTE_HUBER_GATE = 0.005
GRADIENT_RATIO_MAX = 100.0
DEFAULT_TEMPLATE_BETA = 0.10
DEFAULT_RESIDUAL_CAP_QUANTILE = 0.95


def json_default(value: Any) -> Any:
    """Convert supported scientific values into strict JSON values."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
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


def ccif_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["model_or_probe"],
        row["policy_probe"],
    )
    return "|".join(str(value) for value in fields)


def ccif_feature_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        "ccif_current",
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [ccif_row_key(row) for row in manifest_rows]
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
            row["model_or_probe"],
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


def coarse_intent(action_chunks: Any) -> np.ndarray:
    chunks = _chunk_matrix(action_chunks)
    translation = chunks[:, :, 0:3]
    rotation = chunks[:, :, 3:6]
    gripper = chunks[:, :, 6]
    cumulative_translation = np.cumsum(translation, axis=1)
    cumulative_rotation = np.cumsum(rotation, axis=1)
    parts = [
        translation.mean(axis=1),
        rotation.mean(axis=1),
        gripper[:, -5:].mean(axis=1, keepdims=True),
        cumulative_translation[:, WAYPOINT_INDICES, :].reshape(chunks.shape[0], -1),
        cumulative_rotation[:, WAYPOINT_INDICES, :].reshape(chunks.shape[0], -1),
    ]
    intent = np.concatenate(parts, axis=1)
    if intent.shape[1] != INTENT_DIM:
        raise AssertionError(f"CCIF intent must have {INTENT_DIM} dimensions, got {intent.shape[1]}")
    if not np.isfinite(intent).all():
        raise ValueError("coarse intent contains nonfinite values")
    return intent


def parse_intent_raw(intent: Any) -> dict[str, np.ndarray]:
    value = _intent_matrix(intent)
    return {
        "mean_translation": value[:, 0:3],
        "mean_rotation": value[:, 3:6],
        "terminal_gripper": value[:, 6:7],
        "translation_waypoints": value[:, 7:19].reshape(value.shape[0], len(WAYPOINT_INDICES), 3),
        "rotation_waypoints": value[:, 19:31].reshape(value.shape[0], len(WAYPOINT_INDICES), 3),
    }


def assemble_intent_raw(
    mean_translation: Any,
    mean_rotation: Any,
    terminal_gripper: Any,
    translation_waypoints: Any,
    rotation_waypoints: Any,
) -> np.ndarray:
    mean_t = np.asarray(mean_translation, dtype=np.float64).reshape(-1, 3)
    mean_r = np.asarray(mean_rotation, dtype=np.float64).reshape(len(mean_t), 3)
    gripper = np.asarray(terminal_gripper, dtype=np.float64).reshape(len(mean_t), 1)
    trans_wp = np.asarray(translation_waypoints, dtype=np.float64).reshape(len(mean_t), len(WAYPOINT_INDICES), 3)
    rot_wp = np.asarray(rotation_waypoints, dtype=np.float64).reshape(len(mean_t), len(WAYPOINT_INDICES), 3)
    return _intent_matrix(
        np.concatenate([mean_t, mean_r, gripper, trans_wp.reshape(len(mean_t), -1), rot_wp.reshape(len(mean_t), -1)], axis=1)
    )


def fit_intent_normalizer(raw_intents: Any, *, eps: float = STD_FLOOR) -> dict[str, Any]:
    raw = _intent_matrix(raw_intents)
    raw_std = raw.std(axis=0, ddof=0)
    collapsed = raw_std < float(eps)
    return {
        "intent_dim": INTENT_DIM,
        "eps": float(eps),
        "mean": raw.mean(axis=0),
        "std": np.maximum(raw_std, float(eps)),
        "raw_std": raw_std,
        "collapsed_component_mask": collapsed,
        "collapsed_intent_component_count": int(collapsed.sum()),
        "discovery_row_count": int(len(raw)),
    }


def normalize_intent(stats: Mapping[str, Any], raw_intents: Any) -> np.ndarray:
    raw = _intent_matrix(raw_intents)
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, INTENT_DIM)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, INTENT_DIM)
    return (raw - mean) / std


def denormalize_intent(stats: Mapping[str, Any], normalized_intents: Any) -> np.ndarray:
    value = _intent_matrix(normalized_intents)
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, INTENT_DIM)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, INTENT_DIM)
    return value * std + mean


def endpoint_only_intent(normalized_intents: Any, stats: Mapping[str, Any]) -> np.ndarray:
    raw = denormalize_intent(stats, normalized_intents)
    parts = parse_intent_raw(raw)
    terminal_translation = parts["translation_waypoints"][:, -1, :]
    terminal_rotation = parts["rotation_waypoints"][:, -1, :]
    fractions = np.asarray([(index + 1) / CHUNK_SIZE for index in WAYPOINT_INDICES], dtype=np.float64).reshape(1, -1, 1)
    endpoint_raw = assemble_intent_raw(
        terminal_translation / CHUNK_SIZE,
        terminal_rotation / CHUNK_SIZE,
        parts["terminal_gripper"],
        fractions * terminal_translation[:, None, :],
        fractions * terminal_rotation[:, None, :],
    )
    return normalize_intent(stats, endpoint_raw)


def intent_template(normalized_intents: Any, stats: Mapping[str, Any]) -> np.ndarray:
    raw = denormalize_intent(stats, normalized_intents)
    parts = parse_intent_raw(raw)
    rows = raw.shape[0]
    template = np.zeros((rows, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    x_points = np.asarray([0, 10, 20, 35, 50], dtype=np.float64)
    sample_points = np.arange(CHUNK_SIZE + 1, dtype=np.float64)
    for row in range(rows):
        for dim in range(3):
            y = np.concatenate([[0.0], parts["translation_waypoints"][row, :, dim]])
            cumulative = np.interp(sample_points, x_points, y)
            template[row, :, dim] = np.diff(cumulative)
        for dim in range(3):
            y = np.concatenate([[0.0], parts["rotation_waypoints"][row, :, dim]])
            cumulative = np.interp(sample_points, x_points, y)
            template[row, :, 3 + dim] = np.diff(cumulative)
        template[row, -5:, 6] = float(parts["terminal_gripper"][row, 0]) / 5.0
    if not np.isfinite(template).all():
        raise ValueError("intent template contains nonfinite values")
    return template


def intent_consistency_summary(raw_intents: Any) -> dict[str, float]:
    raw = _intent_matrix(raw_intents)
    parts = parse_intent_raw(raw)
    implied_mean_translation = parts["translation_waypoints"][:, -1, :] / CHUNK_SIZE
    implied_mean_rotation = parts["rotation_waypoints"][:, -1, :] / CHUNK_SIZE
    return {
        "mean_translation_terminal_consistency_mae": float(
            np.mean(np.abs(parts["mean_translation"] - implied_mean_translation))
        ),
        "mean_rotation_terminal_consistency_mae": float(np.mean(np.abs(parts["mean_rotation"] - implied_mean_rotation))),
    }


def raw_ccif_feature(visual: Any, proprio: Any, task_index: int, phase: float, base_chunk: Any | None = None) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float64).reshape(-1)
    proprio_value = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if visual_value.shape != (VISUAL_FEATURE_DIM,):
        raise ValueError(f"visual feature must have shape [{VISUAL_FEATURE_DIM}], received {visual_value.shape}")
    if proprio_value.shape != (PROPRIO_DIM,):
        raise ValueError(f"proprio feature must have shape [{PROPRIO_DIM}], received {proprio_value.shape}")
    if base_chunk is None:
        base_intent = np.zeros(INTENT_DIM, dtype=np.float64)
    else:
        base_intent = coarse_intent(np.asarray(base_chunk, dtype=np.float64).reshape(1, CHUNK_SIZE, ACTION_DIM))[0]
    continuous = np.concatenate(
        [visual_value, proprio_value, np.asarray([float(phase)], dtype=np.float64), base_intent]
    )
    if not np.isfinite(continuous).all():
        raise ValueError("CCIF feature contains nonfinite values")
    return np.concatenate([continuous, one_hot(task_index)])


def fit_discovery_zscore(features: Any) -> dict[str, Any]:
    width = VISUAL_FEATURE_DIM + PROPRIO_DIM + 1 + INTENT_DIM + TASK_COUNT
    value = _matrix(features, name="features", width=width)
    continuous_dim = VISUAL_FEATURE_DIM + PROPRIO_DIM + 1 + INTENT_DIM
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


def prediction_metrics(prediction: Any, baseline: Any, target: Any, *, delta: float = HUBER_DELTA) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred_mse = float(np.mean(np.square(pred - truth)))
    base_mse = float(np.mean(np.square(base - truth)))
    pred_huber = mean_huber(pred, truth, delta=delta)
    base_huber = mean_huber(base, truth, delta=delta)
    return {
        "prediction_mse": pred_mse,
        "baseline_mse": base_mse,
        "relative_mse_improvement": float((base_mse - pred_mse) / max(base_mse, 1e-12)),
        "prediction_huber": pred_huber,
        "baseline_huber": base_huber,
        "relative_huber_improvement": float((base_huber - pred_huber) / max(base_huber, 1e-12)),
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


def fit_task_phase_mean_intent(task_indices: Any, phases: Any, normalized_intents: Any) -> dict[str, Any]:
    task_value = np.asarray(task_indices, dtype=np.int64).reshape(-1)
    phase_value = np.asarray(phases, dtype=np.float64).reshape(-1)
    intent = _intent_matrix(normalized_intents)
    if len(task_value) != len(intent) or len(phase_value) != len(intent):
        raise ValueError("task, phase, and intent rows must align")
    default = intent.mean(axis=0)
    groups: dict[str, Any] = {}
    for task in range(TASK_COUNT):
        for bin_index in range(PHASE_BINS):
            mask = (task_value == task) & (np.asarray([phase_bin(value) for value in phase_value]) == bin_index)
            if np.any(mask):
                groups[f"{task}:{bin_index}"] = intent[mask].mean(axis=0)
    return {
        "kind": "task_phase_mean_intent",
        "task_count": TASK_COUNT,
        "phase_bins": PHASE_BINS,
        "default": default,
        "groups": groups,
    }


def predict_task_phase_mean_intent(model: Mapping[str, Any], task_indices: Any, phases: Any) -> np.ndarray:
    task_value = np.asarray(task_indices, dtype=np.int64).reshape(-1)
    phase_value = np.asarray(phases, dtype=np.float64).reshape(-1)
    default = np.asarray(model["default"], dtype=np.float64).reshape(INTENT_DIM)
    groups = {str(key): np.asarray(value, dtype=np.float64).reshape(INTENT_DIM) for key, value in model["groups"].items()}
    rows = []
    for task, phase in zip(task_value, phase_value, strict=True):
        rows.append(groups.get(f"{int(task)}:{phase_bin(float(phase))}", default))
    return np.asarray(rows, dtype=np.float64)


def fit_intent_probe(features: Any, normalized_intents: Any) -> dict[str, Any]:
    model = fit_ridge(features, _intent_matrix(normalized_intents))
    return {
        "kind": "deployment_intent_probe_ridge",
        "model": model,
        "model_hash": canonical_json_sha256({"kind": "deployment_intent_probe_ridge", "model": model}),
    }


def predict_intent_probe(model: Mapping[str, Any], features: Any) -> np.ndarray:
    return predict_ridge(model["model"], features)


def residual_cap_from_discovery(residual_chunks: Any, *, quantile: float = DEFAULT_RESIDUAL_CAP_QUANTILE) -> float:
    residual = _chunk_matrix(residual_chunks)
    norms = np.linalg.norm(residual, axis=2).reshape(-1)
    cap = float(np.quantile(norms, float(quantile)))
    return max(cap, STD_FLOOR)


def clip_l2(residual_chunks: Any, cap: float) -> np.ndarray:
    residual = _chunk_matrix(residual_chunks)
    norms = np.linalg.norm(residual, axis=2, keepdims=True)
    scale = np.minimum(1.0, float(cap) / np.maximum(norms, STD_FLOOR))
    return residual * scale


def apply_ccif_residual(
    base_chunks: Any,
    residual_chunks: Any,
    normalized_intents: Any,
    intent_stats: Mapping[str, Any],
    *,
    gate: Any = 1.0,
    residual_cap: float,
    beta: float = DEFAULT_TEMPLATE_BETA,
) -> np.ndarray:
    base = _chunk_matrix(base_chunks)
    residual = _chunk_matrix(residual_chunks)
    if len(base) != len(residual):
        raise ValueError("base and residual rows must align")
    template = intent_template(normalized_intents, intent_stats)
    if len(template) != len(base):
        raise ValueError("intent rows must align with action chunks")
    gate_value = np.asarray(gate, dtype=np.float64)
    if gate_value.ndim == 0:
        gate_value = np.full((len(base), CHUNK_SIZE, 1), float(gate_value), dtype=np.float64)
    elif gate_value.shape == (len(base), CHUNK_SIZE):
        gate_value = gate_value[:, :, None]
    elif gate_value.shape != (len(base), CHUNK_SIZE, 1):
        raise ValueError(f"gate must be scalar or shape [N,{CHUNK_SIZE},1], received {gate_value.shape}")
    projected = clip_l2(residual, residual_cap) + float(beta) * template
    output = base + gate_value * projected
    if not np.isfinite(output).all():
        raise ValueError("CCIF action chunk contains nonfinite values")
    return output


def action_delta_summary(base_chunks: Any, prediction_chunks: Any) -> dict[str, Any]:
    base = _chunk_matrix(base_chunks)
    prediction = _chunk_matrix(prediction_chunks)
    delta = prediction - base

    def stats(value: np.ndarray) -> dict[str, float]:
        return {
            "mean_abs": float(np.mean(np.abs(value))),
            "max_abs": float(np.max(np.abs(value))),
            "l2_mean": float(np.mean(np.linalg.norm(value.reshape(value.shape[0], -1), axis=1))),
        }

    changed = np.abs(delta) > 1e-12
    return {
        "changed_cell_fraction": float(changed.mean()),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta": stats(delta[:, :, 0:3]),
        "rotation_delta": stats(delta[:, :, 3:6]),
        "gripper_delta": stats(delta[:, :, 6:7]),
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
    split_integrity_ok: bool
    minimum_discovery_windows: int
    minimum_validation_windows: int
    all_tasks_reported: bool
    maximum_validation_task_fraction: float
    labels_noncollapsed_discovery: bool
    labels_noncollapsed_validation: bool
    collapsed_intent_component_count: int
    intent_probe_beats_task_phase_mean: bool
    intent_probe_relative_improvement: float
    intent_probe_absolute_huber: float
    endpoint_only_explains_ccif: bool
    ccif_beats_prior_relative: float
    ccif_beats_prior_absolute_huber: float
    ccif_beats_ablation_relative: float
    ccif_beats_ablation_absolute_huber: float
    action_validity_ok: bool
    identity_max_abs_error: float
    checkpoint_reload_ok: bool
    finite_objectives_and_gradients: bool
    ccif_gradient_nonzero: bool
    frozen_parameter_gradient_count: int
    weighted_gradient_norm_ratio_max: float
    reward_read_count: int
    success_read_count: int
    done_read_count: int
    confirmatory_records_read: int
    closed_loop_experiment_happened: bool
    simulator_load_count: int
    training_happened: bool
    validation_search_happened: bool
    exception_count: int


def classify_stage0(inputs: Stage0DecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or not inputs.official_prior_asset_check_persisted
        or not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.split_integrity_ok
        or int(inputs.reward_read_count) != 0
        or int(inputs.success_read_count) != 0
        or int(inputs.done_read_count) != 0
        or int(inputs.confirmatory_records_read) != 0
        or bool(inputs.closed_loop_experiment_happened)
        or int(inputs.simulator_load_count) != 0
        or bool(inputs.training_happened)
        or bool(inputs.validation_search_happened)
        or int(inputs.exception_count) != 0
    ):
        return "CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < 512
        or int(inputs.minimum_validation_windows) < 128
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.labels_noncollapsed_discovery
        or not inputs.labels_noncollapsed_validation
        or int(inputs.collapsed_intent_component_count) != 0
    ):
        return "CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        not inputs.intent_probe_beats_task_phase_mean
        or not _headroom_gate(inputs.intent_probe_relative_improvement, inputs.intent_probe_absolute_huber)
    ):
        return "CCIF_STAGE_0_DESIGN_FAILURE"
    if inputs.endpoint_only_explains_ccif:
        return "CCIF_STAGE_0_DESIGN_FAILURE"
    if not _headroom_gate(inputs.ccif_beats_prior_relative, inputs.ccif_beats_prior_absolute_huber):
        return "CCIF_STAGE_0_NO_USABLE_HEADROOM"
    if not _headroom_gate(inputs.ccif_beats_ablation_relative, inputs.ccif_beats_ablation_absolute_huber):
        return "CCIF_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not inputs.action_validity_ok
        or float(inputs.identity_max_abs_error) > 1e-6
        or not inputs.checkpoint_reload_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.ccif_gradient_nonzero
        or int(inputs.frozen_parameter_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
    ):
        return "CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    return "CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


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
    if array.ndim == 2 and array.shape == (CHUNK_SIZE, ACTION_DIM):
        array = array.reshape(1, CHUNK_SIZE, ACTION_DIM)
    if array.ndim != 3 or array.shape[1:] != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(array).all():
        raise ValueError(f"chunks must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], received {array.shape}")
    return array


def _intent_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape == (INTENT_DIM,):
        array = array.reshape(1, INTENT_DIM)
    if array.ndim != 2 or array.shape[1] != INTENT_DIM or not np.isfinite(array).all():
        raise ValueError(f"intent must have shape [N,{INTENT_DIM}], received {array.shape}")
    return array
