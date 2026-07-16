"""Frozen AFID-VLA Stage 0 action-factor audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC"
HORIZON = 50
ACTION_DIM = 7
TAU_AXIS_MOTION = 0.03
TAU_DIR = 0.01
TAU_ROT = 0.02
TAU_GRIP_EVENT = 0.20
TAU_SETTLE = 0.015
TAU_RESIDUAL_MASK = 0.50
TAU_CONF = 0.60
TAU_ENTROPY = 0.75
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
MIN_DISCOVERY_WINDOWS = 512
MIN_VALIDATION_WINDOWS = 128
MAX_VALIDATION_TASK_FRACTION = 0.40
MASK_GLOBAL_MIN = 0.02
MASK_GLOBAL_MAX = 0.80
MASK_TASK_MIN = 0.01
MASK_TASK_MAX = 0.90
PREDICTOR_IMPROVEMENT_MIN = 0.05
ORACLE_REDUCTION_MIN = 0.02
FINEVLA_HEADROOM_MIN = 0.0
GATE_ACTIVATION_MIN = 0.02
GATE_ACTIVATION_MAX = 0.80
IDENTITY_RELOAD_ERROR_MAX = 1e-6
GRADIENT_RATIO_MAX = 20.0
ACTION_HUBER_DELTA = 0.05
STD_FLOOR = 1e-8


POLICY_ROWS = (
    "smolvla_base",
    "finevla_action_factor_proxy",
    "afid_full",
    "afid_no_factor_ablation",
    "standard_lora",
    "factor_conditioned_oracle_diagnostic",
    "task_phase_residual_diagnostic",
    "mask_only_residual_diagnostic",
)


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


def afid_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["split"],
        row["task_suite"],
        row["task_id"],
        row["demo_id"],
        row["window_start"],
        row["factor_key"],
        row["policy"],
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [afid_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["task_suite"],
            row["task_id"],
            row["demo_id"],
            row["window_start"],
            row["factor_key"],
            row["policy"],
        )

    discovery = {split_identity(row) for row in manifest_rows if row["split"] == "discovery"}
    validation = {split_identity(row) for row in manifest_rows if row["split"] == "validation"}
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


def chunk_matrix(value: Any, name: str = "chunks") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape == (HORIZON, ACTION_DIM):
        array = array.reshape(1, HORIZON, ACTION_DIM)
    if array.ndim != 3 or array.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"{name} must have shape [N,{HORIZON},{ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def group_clip(residual: Any) -> np.ndarray:
    value = chunk_matrix(residual, "residual").copy()
    value[:, :, 0:3] = np.clip(value[:, :, 0:3], -TRANSLATION_CAP, TRANSLATION_CAP)
    value[:, :, 3:6] = np.clip(value[:, :, 3:6], -ROTATION_CAP, ROTATION_CAP)
    value[:, :, 6:7] = np.clip(value[:, :, 6:7], -GRIPPER_CAP, GRIPPER_CAP)
    return value


def fit_residual_scale(residual_chunks: Any) -> dict[str, Any]:
    residual = chunk_matrix(residual_chunks, "residual_chunks")
    scale = np.maximum(np.percentile(np.abs(residual), 75, axis=(0, 1)), STD_FLOOR)
    return {
        "kind": "discovery_abs_p75_per_action_dim",
        "tau_residual_mask": TAU_RESIDUAL_MASK,
        "action_dimension": ACTION_DIM,
        "scale": scale,
        "discovery_row_count": int(len(residual)),
    }


def factor_mask(residual_chunks: Any, residual_scale: Mapping[str, Any]) -> np.ndarray:
    residual = chunk_matrix(residual_chunks, "residual_chunks")
    scale = np.asarray(residual_scale["scale"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    return (np.abs(residual) / np.maximum(scale, STD_FLOOR) >= TAU_RESIDUAL_MASK).astype(np.float64)


def extract_action_factors(base_chunks: Any, expert_chunks: Any) -> dict[str, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunks differ: {base.shape} vs {expert.shape}")
    residual = expert - base
    trans = residual[:, :, 0:3]
    rot = residual[:, :, 3:6]
    grip = residual[:, :, 6]
    abs_trans = np.abs(trans)
    axis = np.argmax(abs_trans, axis=2) + 1
    axis_strength = np.max(abs_trans, axis=2)
    axis = np.where(axis_strength >= TAU_AXIS_MOTION, axis, 0).astype(np.int64)
    chosen = np.take_along_axis(trans, np.maximum(axis - 1, 0)[:, :, None], axis=2)[:, :, 0]
    direction = np.where(np.abs(chosen) >= TAU_DIR, np.sign(chosen), 0).astype(np.int64)
    grip_event = (np.abs(grip) >= TAU_GRIP_EVENT).astype(np.int64)
    grip_bin = np.where(grip_event > 0, np.sign(grip), 0).astype(np.int64)
    rot_norm = np.linalg.norm(rot, axis=2)
    rot_active = (rot_norm >= TAU_ROT).astype(np.int64)
    trans_norm = np.linalg.norm(trans, axis=2)
    settle = (
        (trans_norm <= TAU_SETTLE)
        & (rot_norm <= TAU_SETTLE)
        & (np.abs(grip) <= 0.5 * TAU_GRIP_EVENT)
    ).astype(np.int64)
    return {
        "axis": axis,
        "direction": direction,
        "grip_type": grip_event,
        "grip_bin": grip_bin,
        "rotation": rot_active,
        "termination": settle,
    }


def factor_label_health(labels: Mapping[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    usable = 0
    for name, raw in labels.items():
        value = np.asarray(raw).reshape(-1)
        keys, counts = np.unique(value, return_counts=True)
        total = int(np.sum(counts))
        fractions = counts / max(total, 1)
        entropy = float(-np.sum(fractions * np.log2(np.maximum(fractions, STD_FLOOR))))
        max_entropy = float(np.log2(max(len(keys), 2)))
        normalized_entropy = entropy / max(max_entropy, STD_FLOOR)
        largest = float(np.max(fractions)) if len(fractions) else 1.0
        noncollapsed = bool(len(keys) > 1 and largest <= TAU_ENTROPY)
        usable += int(noncollapsed)
        summaries[name] = {
            "class_count": int(len(keys)),
            "counts": {str(key.item() if isinstance(key, np.generic) else key): int(count) for key, count in zip(keys, counts)},
            "largest_class_fraction": largest,
            "normalized_entropy": normalized_entropy,
            "noncollapsed": noncollapsed,
        }
    return {"usable_factor_count": usable, "factors": summaries}


def factor_keys(labels: Mapping[str, Any]) -> list[str]:
    axis = _mode_per_row(labels["axis"])
    direction = _mode_per_row(labels["direction"])
    grip = _mode_per_row(labels["grip_bin"])
    rotation = _mode_per_row(labels["rotation"])
    term = _mode_per_row(labels["termination"])
    return [
        f"axis:{int(a)}|dir:{int(d)}|grip:{int(g)}|rot:{int(r)}|term:{int(t)}"
        for a, d, g, r, t in zip(axis, direction, grip, rotation, term)
    ]


def mask_health(mask: Any, *, task_ids: Sequence[str] | None = None) -> dict[str, Any]:
    value = chunk_matrix(mask, "mask")
    global_fraction = float(np.mean(value > 0.5))
    result: dict[str, Any] = {
        "factor_mask_global_positive_fraction": global_fraction,
        "factor_mask_all_zero": bool(np.all(value <= 0.5)),
        "factor_mask_all_one": bool(np.all(value > 0.5)),
        "factor_mask_noncollapsed": bool(MASK_GLOBAL_MIN <= global_fraction <= MASK_GLOBAL_MAX),
    }
    if task_ids is not None:
        per_task: dict[str, float] = {}
        for task in sorted(set(task_ids)):
            indexes = [index for index, value_task in enumerate(task_ids) if value_task == task]
            if indexes:
                per_task[task] = float(np.mean(value[indexes] > 0.5))
        result["factor_mask_positive_fraction_by_task"] = per_task
        result["validation_task_mask_fraction_min"] = min(per_task.values()) if per_task else 0.0
        result["validation_task_mask_fraction_max"] = max(per_task.values()) if per_task else 1.0
    return result


def fit_linear_factor_predictor(base_chunks: Any, target_mask: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    target = chunk_matrix(target_mask, "target_mask")
    feature = np.abs(base)
    numerator = np.sum(feature * target, axis=(0, 1))
    denominator = np.maximum(np.sum(np.square(feature), axis=(0, 1)), STD_FLOOR)
    coef = numerator / denominator
    return {"kind": "abs_base_per_dim_linear_mask_predictor", "coef": coef}


def predict_factor_confidence(model: Mapping[str, Any], base_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    coef = np.asarray(model["coef"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    return np.clip(np.abs(base) * coef, 0.0, 1.0)


def binary_prediction_metrics(predicted: Any, target: Any) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=bool).reshape(-1)
    tgt = np.asarray(target, dtype=bool).reshape(-1)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    tp = int(np.sum(pred & tgt))
    fp = int(np.sum(pred & ~tgt))
    fn = int(np.sum(~pred & tgt))
    tn = int(np.sum(~pred & ~tgt))
    accuracy = float((tp + tn) / max(len(tgt), 1))
    positive_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    negative_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    macro_f1 = float(0.5 * (positive_f1 + negative_f1))
    majority = bool(np.mean(tgt) >= 0.5)
    majority_accuracy = float(np.mean(np.full_like(tgt, majority) == tgt))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_accuracy": majority_accuracy,
        "majority_macro_f1": 0.5,
        "accuracy_improvement_over_majority": accuracy - majority_accuracy,
        "macro_f1_improvement_over_majority": macro_f1 - 0.5,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def fit_group_mean(chunks: Any, keys: Sequence[str]) -> dict[str, Any]:
    value = chunk_matrix(chunks, "chunks")
    if len(value) != len(keys):
        raise ValueError("chunks and keys must align")
    default = value.mean(axis=0)
    groups: dict[str, Any] = {}
    for key in sorted(set(keys)):
        group = np.asarray([value[index] for index, item in enumerate(keys) if item == key], dtype=np.float64)
        groups[key] = group.mean(axis=0)
    return {"default": default, "groups": groups, "group_count": len(groups)}


def predict_group_mean(model: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    default = np.asarray(model["default"], dtype=np.float64).reshape(HORIZON, ACTION_DIM)
    groups = {str(key): np.asarray(value, dtype=np.float64).reshape(HORIZON, ACTION_DIM) for key, value in model["groups"].items()}
    return np.asarray([groups.get(str(key), default) for key in keys], dtype=np.float64)


def apply_afid_gate(base_chunks: Any, residual_head: Any, mask: Any, confidence: Any) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_head)
    mask_value = chunk_matrix(mask, "mask")
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.shape == (len(base), HORIZON):
        conf = conf[:, :, None]
    if conf.shape == (len(base), HORIZON, 1):
        conf = np.repeat(conf, ACTION_DIM, axis=2)
    if conf.shape != base.shape:
        raise ValueError(f"confidence must broadcast to [N,{HORIZON},{ACTION_DIM}], got {conf.shape}")
    gate = ((conf >= TAU_CONF).astype(np.float64) * (mask_value > 0.5).astype(np.float64))
    output = base + gate * residual
    if not np.isfinite(output).all():
        raise ValueError("AFID action chunk contains nonfinite values")
    return output, gate


def apply_finevla_proxy(base_chunks: Any, residual_head: Any, confidence: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_head)
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.shape == (len(base), HORIZON):
        conf = conf[:, :, None]
    if conf.shape == (len(base), HORIZON, 1):
        conf = np.repeat(conf, ACTION_DIM, axis=2)
    if conf.shape != base.shape:
        raise ValueError(f"confidence must broadcast to [N,{HORIZON},{ACTION_DIM}], got {conf.shape}")
    return base + np.clip(conf, 0.0, 1.0) * residual


def action_delta_summary(base_chunks: Any, prediction_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    prediction = chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction chunks differ: {base.shape} vs {prediction.shape}")
    delta = prediction - base

    def p95(value: np.ndarray) -> float:
        return float(np.percentile(np.abs(value), 95))

    return {
        "changed_cell_fraction": float(np.mean(np.abs(delta) > 1e-12)),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta_p95": p95(delta[:, :, 0:3]),
        "rotation_delta_p95": p95(delta[:, :, 3:6]),
        "gripper_delta_p95": p95(delta[:, :, 6:7]),
        "action_deltas_bounded": bool(
            p95(delta[:, :, 0:3]) <= TRANSLATION_CAP
            and p95(delta[:, :, 3:6]) <= ROTATION_CAP
            and p95(delta[:, :, 6:7]) <= GRIPPER_CAP
        ),
        "changed_dimensions": [
            int(index) for index in range(ACTION_DIM) if float(np.max(np.abs(delta[:, :, index]))) > 1e-12
        ],
    }


def clean_retention_summary(base_chunks: Any, identity_chunks: Any, inactive_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    identity = chunk_matrix(identity_chunks, "identity_chunks")
    inactive = chunk_matrix(inactive_chunks, "inactive_chunks")
    identity_error = float(np.max(np.abs(identity - base)))
    inactive_error = float(np.max(np.abs(inactive - base)))
    return {
        "identity_max_abs_error": identity_error,
        "inactive_gate_max_abs_error": inactive_error,
        "clean_retention_ok": bool(
            identity_error <= IDENTITY_RELOAD_ERROR_MAX and inactive_error <= IDENTITY_RELOAD_ERROR_MAX
        ),
    }


def huber_values(error: Any, *, delta: float = ACTION_HUBER_DELTA) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("huber input contains nonfinite values")
    threshold = float(delta)
    if threshold <= 0.0:
        raise ValueError("delta must be positive")
    absolute = np.abs(value)
    return np.where(absolute <= threshold, 0.5 * np.square(value), threshold * (absolute - 0.5 * threshold))


def mean_huber(prediction: Any, target: Any, *, delta: float = ACTION_HUBER_DELTA) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    return float(np.mean(huber_values(pred - tgt, delta=delta)))


def relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(float(baseline)) or not np.isfinite(float(candidate)):
        return 0.0
    return float((float(baseline) - float(candidate)) / max(abs(float(baseline)), 1e-12))


def gradient_smoke(base_chunks: Any, residual_head: Any, gate: Any, target_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_head)
    gate_value = chunk_matrix(gate, "gate")
    target = chunk_matrix(target_chunks, "target_chunks")
    update = gate_value * residual
    gradient = float(np.mean((base - target) * update))
    update_norm = float(np.mean(np.square(update)))
    return {
        "finite_objectives_and_gradients": bool(np.isfinite(gradient) and np.isfinite(update_norm)),
        "expected_parameter_gradient_nonzero": bool(abs(gradient) > STD_FLOOR or update_norm > STD_FLOOR),
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "scalar_gain_gradient": gradient,
        "update_mean_square": update_norm,
    }


@dataclass(frozen=True)
class Stage0DecisionInputs:
    proposal_hash_ok: bool
    serializer_preflight_ok: bool
    official_prior_asset_check_persisted: bool
    manifest_integrity_ok: bool
    source_alignment_ok: bool
    action_semantics_ok: bool
    base_chunks_valid: bool
    factor_labels_noncollapsed: bool
    usable_factor_count: int
    factor_mask_global_positive_fraction: float
    validation_task_mask_fraction_min: float
    validation_task_mask_fraction_max: float
    factor_predictor_beats_majority: bool
    factor_predictor_beats_task_phase: bool
    factor_conditioned_oracle_reduction: float
    finevla_proxy_residual_headroom: float
    afid_differs_from_base: bool
    afid_differs_from_finevla_proxy: bool
    afid_differs_from_no_factor: bool
    afid_differs_from_standard_lora: bool
    identity_max_abs_error: float
    inactive_gate_max_abs_error: float
    finite_objectives_and_gradients: bool
    expected_parameter_gradient_nonzero: bool
    frozen_base_gradient_count: int
    weighted_gradient_norm_ratio_max: float
    gate_activation_fraction: float
    action_deltas_bounded: bool
    action_validity_ok: bool
    clean_retention_ok: bool
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
        or not inputs.action_semantics_ok
        or not inputs.base_chunks_valid
        or float(inputs.identity_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or float(inputs.inactive_gate_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or not inputs.finite_objectives_and_gradients
        or not inputs.expected_parameter_gradient_nonzero
        or int(inputs.frozen_base_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
        or not inputs.action_deltas_bounded
        or not inputs.action_validity_ok
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
        return "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    if (
        not inputs.factor_labels_noncollapsed
        or int(inputs.usable_factor_count) < 1
        or not (MASK_GLOBAL_MIN <= float(inputs.factor_mask_global_positive_fraction) <= MASK_GLOBAL_MAX)
        or float(inputs.validation_task_mask_fraction_min) < MASK_TASK_MIN
        or float(inputs.validation_task_mask_fraction_max) > MASK_TASK_MAX
    ):
        return "AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        float(inputs.factor_conditioned_oracle_reduction) < ORACLE_REDUCTION_MIN
        or float(inputs.finevla_proxy_residual_headroom) <= FINEVLA_HEADROOM_MIN
    ):
        return "AFID_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not inputs.factor_predictor_beats_majority
        or not inputs.factor_predictor_beats_task_phase
        or not inputs.afid_differs_from_base
        or not inputs.afid_differs_from_finevla_proxy
        or not inputs.afid_differs_from_no_factor
        or not inputs.afid_differs_from_standard_lora
        or not (GATE_ACTIVATION_MIN <= float(inputs.gate_activation_fraction) <= GATE_ACTIVATION_MAX)
        or not inputs.clean_retention_ok
    ):
        return "AFID_STAGE_0_DESIGN_FAILURE"
    return "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _mode_per_row(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array
    modes = []
    for row in array.reshape(array.shape[0], -1):
        keys, counts = np.unique(row, return_counts=True)
        modes.append(keys[int(np.argmax(counts))])
    return np.asarray(modes)
