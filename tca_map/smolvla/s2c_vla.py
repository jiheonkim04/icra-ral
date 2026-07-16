"""Frozen S2C-VLA Stage 0 seam-supervised chunk-consistency helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3"
ACTION_DIM = 7
CHUNK_SIZE = 50
REPLAN_STRIDE = 10
OVERLAP_LENGTH = 10
HUBER_DELTA = 1.0
ACTION_HUBER_DELTA = 0.05
BOUNDARY_HEADROOM_MEAN_MIN = 0.0025
BOUNDARY_HEADROOM_P75_MIN = 0.005
CHUNKFLOW_HEADROOM_MIN_RELATIVE = 0.02
S2C_VS_CHUNKFLOW_MIN_RELATIVE = 0.02
S2C_VS_NO_MASK_MIN_RELATIVE = 0.05
MASK_POSITIVE_MIN = 0.02
MASK_POSITIVE_MAX = 0.80
IDENTITY_RELOAD_ERROR_MAX = 1e-6
GRADIENT_RATIO_MAX = 20.0
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
LAMBDA_TAIL = 1.0
LAMBDA_D1 = 0.25
LAMBDA_D2 = 0.10


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


def s2c_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["split"],
        row["task_suite"],
        row["task_id"],
        row["demo_id"],
        row["window_start"],
        row["stride"],
        row["previous_policy_source"],
        row["policy"],
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [s2c_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["task_suite"],
            row["task_id"],
            row["demo_id"],
            row["window_start"],
            row["stride"],
            row["previous_policy_source"],
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
    if array.ndim == 2 and array.shape == (CHUNK_SIZE, ACTION_DIM):
        array = array.reshape(1, CHUNK_SIZE, ACTION_DIM)
    if array.ndim != 3 or array.shape[1:] != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"{name} must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def overlap_matrix(value: Any, name: str = "overlap") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape == (OVERLAP_LENGTH, ACTION_DIM):
        array = array.reshape(1, OVERLAP_LENGTH, ACTION_DIM)
    if array.ndim != 3 or array.shape[1:] != (OVERLAP_LENGTH, ACTION_DIM):
        raise ValueError(f"{name} must have shape [N,{OVERLAP_LENGTH},{ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def previous_tail(previous_chunks: Any, *, stride: int = REPLAN_STRIDE, length: int = OVERLAP_LENGTH) -> np.ndarray:
    chunks = chunk_matrix(previous_chunks, "previous_chunks")
    start = int(stride)
    stop = start + int(length)
    if start < 0 or stop > CHUNK_SIZE:
        raise ValueError("previous tail slice must stay inside chunk")
    return chunks[:, start:stop, :]


def current_head(current_chunks: Any, *, length: int = OVERLAP_LENGTH) -> np.ndarray:
    chunks = chunk_matrix(current_chunks, "current_chunks")
    stop = int(length)
    if stop <= 0 or stop > CHUNK_SIZE:
        raise ValueError("current head length must stay inside chunk")
    return chunks[:, :stop, :]


def first_difference_matrix(length: int = OVERLAP_LENGTH) -> np.ndarray:
    if int(length) < 2:
        raise ValueError("length must be at least 2")
    matrix = np.zeros((int(length) - 1, int(length)), dtype=np.float64)
    for index in range(int(length) - 1):
        matrix[index, index] = -1.0
        matrix[index, index + 1] = 1.0
    return matrix


def second_difference_matrix(length: int = OVERLAP_LENGTH) -> np.ndarray:
    if int(length) < 3:
        raise ValueError("length must be at least 3")
    matrix = np.zeros((int(length) - 2, int(length)), dtype=np.float64)
    for index in range(int(length) - 2):
        matrix[index, index] = 1.0
        matrix[index, index + 1] = -2.0
        matrix[index, index + 2] = 1.0
    return matrix


def bridge_target(
    base_head: Any,
    tail: Any,
    *,
    lambda_tail: float = LAMBDA_TAIL,
    lambda_d1: float = LAMBDA_D1,
    lambda_d2: float = LAMBDA_D2,
) -> np.ndarray:
    base = overlap_matrix(base_head, "base_head")
    previous = overlap_matrix(tail, "tail")
    if base.shape != previous.shape:
        raise ValueError(f"base head and tail shapes differ: {base.shape} vs {previous.shape}")
    k = base.shape[1]
    identity = np.eye(k, dtype=np.float64)
    d1 = first_difference_matrix(k)
    d2 = second_difference_matrix(k)
    lhs = identity + float(lambda_tail) * identity + float(lambda_d1) * (d1.T @ d1) + float(lambda_d2) * (d2.T @ d2)
    rhs_base = base + float(lambda_tail) * previous
    rhs_d1 = float(lambda_d1) * np.einsum("ab,nbd->nad", d1.T, np.einsum("ab,nbd->nad", d1, previous))
    rhs_d2 = float(lambda_d2) * np.einsum("ab,nbd->nad", d2.T, np.einsum("ab,nbd->nad", d2, previous))
    rhs = rhs_base + rhs_d1 + rhs_d2
    solved = np.empty_like(base)
    for row in range(base.shape[0]):
        solved[row] = np.linalg.solve(lhs, rhs[row])
    return solved


def group_clip(residual: Any) -> np.ndarray:
    value = overlap_matrix(residual, "residual").copy()
    value[:, :, 0:3] = np.clip(value[:, :, 0:3], -TRANSLATION_CAP, TRANSLATION_CAP)
    value[:, :, 3:6] = np.clip(value[:, :, 3:6], -ROTATION_CAP, ROTATION_CAP)
    value[:, :, 6:7] = np.clip(value[:, :, 6:7], -GRIPPER_CAP, GRIPPER_CAP)
    return value


def effective_mask(mask_logits: Any, *, gamma: float = 0.0) -> np.ndarray:
    logits = overlap_matrix(mask_logits, "mask_logits")
    clipped = np.clip(logits, -60.0, 60.0)
    return float(gamma) / (1.0 + np.exp(-clipped))


def apply_s2c_edit(
    base_chunks: Any,
    previous_chunks: Any,
    mask_logits: Any,
    *,
    gamma: float = 0.0,
    no_previous_tail: Any | None = None,
) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    prev = chunk_matrix(previous_chunks, "previous_chunks")
    if base.shape != prev.shape:
        raise ValueError(f"base and previous shapes differ: {base.shape} vs {prev.shape}")
    head = current_head(base)
    tail = previous_tail(prev)
    target = bridge_target(head, tail)
    mask = effective_mask(mask_logits, gamma=gamma)
    if mask.shape != head.shape:
        raise ValueError(f"mask and head shapes differ: {mask.shape} vs {head.shape}")
    valid_previous = np.ones((base.shape[0], 1, 1), dtype=np.float64)
    if no_previous_tail is not None:
        missing = np.asarray(no_previous_tail, dtype=bool).reshape(base.shape[0], 1, 1)
        valid_previous = np.where(missing, 0.0, 1.0)
    output = base.copy()
    output[:, :OVERLAP_LENGTH, :] = head + valid_previous * mask * group_clip(target - head)
    if not np.array_equal(output[:, OVERLAP_LENGTH:, :], base[:, OVERLAP_LENGTH:, :]):
        raise AssertionError("future zone must remain exact Base")
    if not np.isfinite(output).all():
        raise ValueError("S2C output contains nonfinite values")
    return output


def huber_values(error: Any, *, delta: float = HUBER_DELTA) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("huber input contains nonfinite values")
    absolute = np.abs(value)
    threshold = float(delta)
    if threshold <= 0.0:
        raise ValueError("delta must be positive")
    return np.where(absolute <= threshold, 0.5 * value * value, threshold * (absolute - 0.5 * threshold))


def mean_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")
    return float(np.mean(huber_values(pred - tgt, delta=delta)))


def boundary_disagreement(base_chunks: Any, previous_chunks: Any) -> np.ndarray:
    return huber_values(current_head(base_chunks) - previous_tail(previous_chunks), delta=ACTION_HUBER_DELTA)


def boundary_headroom_summary(base_chunks: Any, previous_chunks: Any) -> dict[str, Any]:
    values = boundary_disagreement(base_chunks, previous_chunks).reshape(-1)
    mean_value = float(np.mean(values))
    p75_value = float(np.percentile(values, 75))
    return {
        "base_boundary_huber_mean": mean_value,
        "base_boundary_huber_p75": p75_value,
        "base_boundary_headroom_ok": bool(
            mean_value >= BOUNDARY_HEADROOM_MEAN_MIN or p75_value >= BOUNDARY_HEADROOM_P75_MIN
        ),
    }


def derivative_metrics(chunks: Any, previous_chunks: Any) -> dict[str, float]:
    current = current_head(chunks)
    tail = previous_tail(previous_chunks)
    d1_current = np.diff(current, axis=1)
    d1_tail = np.diff(tail, axis=1)
    d2_current = np.diff(current, n=2, axis=1)
    d2_tail = np.diff(tail, n=2, axis=1)
    return {
        "first_order_huber": mean_huber(d1_current, d1_tail, delta=ACTION_HUBER_DELTA),
        "second_order_huber": mean_huber(d2_current, d2_tail, delta=ACTION_HUBER_DELTA),
    }


def high_frequency_energy(overlap: Any) -> float:
    value = overlap_matrix(overlap, "overlap")
    if value.shape[1] < 3:
        return 0.0
    second = np.diff(value, n=2, axis=1)
    return float(np.mean(np.square(second)))


def gripper_event_mask(base_head: Any, tail: Any, *, threshold: float = 0.0) -> np.ndarray:
    base = overlap_matrix(base_head, "base_head")[:, :, 6]
    previous = overlap_matrix(tail, "tail")[:, :, 6]
    base_sign = base >= float(threshold)
    tail_sign = previous >= float(threshold)
    base_change = np.concatenate([np.zeros((base.shape[0], 1), dtype=bool), base_sign[:, 1:] != base_sign[:, :-1]], axis=1)
    tail_change = np.concatenate([np.zeros((tail_sign.shape[0], 1), dtype=bool), tail_sign[:, 1:] != tail_sign[:, :-1]], axis=1)
    return base_change | tail_change | (base_sign != tail_sign)


def gripper_event_destruction_count(base_chunks: Any, prediction_chunks: Any, previous_chunks: Any) -> int:
    base_head = current_head(base_chunks)
    pred_head = current_head(prediction_chunks)
    tail = previous_tail(previous_chunks)
    events = gripper_event_mask(base_head, tail)
    if not np.any(events):
        return 0
    changed = np.abs(pred_head[:, :, 6] - base_head[:, :, 6]) > 1e-12
    return int(np.sum(events & changed))


def action_delta_summary(base_chunks: Any, prediction_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    prediction = chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction shapes differ: {base.shape} vs {prediction.shape}")
    delta = prediction - base

    def p95(value: np.ndarray) -> float:
        return float(np.percentile(np.abs(value), 95))

    future_drift = float(np.max(np.abs(delta[:, OVERLAP_LENGTH:, :]))) if delta.shape[1] > OVERLAP_LENGTH else 0.0
    return {
        "changed_cell_fraction": float(np.mean(np.abs(delta) > 1e-12)),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta_p95": p95(delta[:, :, 0:3]),
        "rotation_delta_p95": p95(delta[:, :, 3:6]),
        "gripper_delta_p95": p95(delta[:, :, 6:7]),
        "future_zone_drift_max": future_drift,
        "action_deltas_bounded": bool(
            p95(delta[:, :, 0:3]) <= TRANSLATION_CAP
            and p95(delta[:, :, 3:6]) <= ROTATION_CAP
            and p95(delta[:, :, 6:7]) <= GRIPPER_CAP
            and future_drift == 0.0
        ),
        "changed_dimensions": [
            int(index) for index in range(ACTION_DIM) if float(np.max(np.abs(delta[:, :, index]))) > 1e-12
        ],
    }


def mask_health(mask: Any) -> dict[str, Any]:
    value = overlap_matrix(mask, "mask")
    positive = float(np.mean(value > 0.01))
    return {
        "mask_positive_fraction": positive,
        "mask_noncollapsed": bool(MASK_POSITIVE_MIN <= positive <= MASK_POSITIVE_MAX),
        "mask_all_zero": bool(np.all(value <= 0.01)),
        "mask_all_one": bool(np.all(value >= 1.0 - 1e-12)),
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
    adjacent_pair_count: int
    all_tasks_reported: bool
    maximum_validation_task_fraction: float
    label_contrast_noncollapsed: bool
    base_boundary_headroom_ok: bool
    chunkflow_residual_headroom_relative: float
    identity_max_abs_error: float
    checkpoint_reload_ok: bool
    mask_positive_fraction: float
    mask_all_zero: bool
    mask_all_one: bool
    future_zone_drift_max: float
    action_validity_ok: bool
    s2c_beats_chunkflow_relative: float
    s2c_beats_no_mask_relative: float
    standard_lora_explains: bool
    gripper_event_destruction_count: int
    finite_objectives_and_gradients: bool
    s2c_gradient_nonzero: bool
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
        return "S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.feature_action_proprio_finite_aligned
        or int(inputs.adjacent_pair_count) <= 0
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.label_contrast_noncollapsed
    ):
        return "S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if not inputs.base_boundary_headroom_ok:
        return "S2C_STAGE_0_NO_ADJACENT_BOUNDARY_HEADROOM"
    if float(inputs.chunkflow_residual_headroom_relative) < CHUNKFLOW_HEADROOM_MIN_RELATIVE:
        return "S2C_STAGE_0_NO_ADJACENT_BOUNDARY_HEADROOM"
    if (
        float(inputs.identity_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or not inputs.checkpoint_reload_ok
        or float(inputs.future_zone_drift_max) != 0.0
        or not inputs.action_validity_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.s2c_gradient_nonzero
        or int(inputs.frozen_parameter_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
    ):
        return "S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        bool(inputs.mask_all_zero)
        or bool(inputs.mask_all_one)
        or not (MASK_POSITIVE_MIN <= float(inputs.mask_positive_fraction) <= MASK_POSITIVE_MAX)
        or float(inputs.s2c_beats_chunkflow_relative) < S2C_VS_CHUNKFLOW_MIN_RELATIVE
        or float(inputs.s2c_beats_no_mask_relative) < S2C_VS_NO_MASK_MIN_RELATIVE
        or bool(inputs.standard_lora_explains)
        or int(inputs.gripper_event_destruction_count) != 0
    ):
        return "S2C_STAGE_0_DESIGN_FAILURE"
    return "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
