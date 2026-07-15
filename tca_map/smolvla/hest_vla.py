"""Frozen HEST-VLA trajectory transforms and Stage 0A decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527"
HORIZON = 50
ACTION_DIM = 7
ARM_DIM = 6
GRIPPER_DIM = 6
CURVATURE_LAMBDA = 4.0
SUPPORT_TOLERANCE = 0.01


def parse_sha256_registry(value: str) -> str:
    tokens = value.strip().split()
    if len(tokens) < 2 or tokens[0].upper() != "SHA256":
        return ""
    for token in tokens[1:]:
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def chunk_sha256(chunk: Any) -> str:
    array = _as_chunk(chunk)
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest().upper()


def _as_chunk(chunk: Any) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float64)
    if array.shape != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected action chunk [{HORIZON},{ACTION_DIM}], received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("action chunk contains nonfinite values")
    return array


@lru_cache(maxsize=4)
def second_difference_matrix(horizon: int = HORIZON) -> np.ndarray:
    if horizon < 3:
        raise ValueError("horizon must be at least three")
    matrix = np.zeros((horizon - 2, horizon), dtype=np.float64)
    rows = np.arange(horizon - 2)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=8)
def _constrained_kkt(horizon: int = HORIZON, curvature_lambda: float = CURVATURE_LAMBDA) -> np.ndarray:
    d2 = second_difference_matrix(horizon)
    system = np.eye(horizon, dtype=np.float64) + float(curvature_lambda) * (d2.T @ d2)
    selector = np.zeros((2, horizon), dtype=np.float64)
    selector[0, 0] = 1.0
    selector[1, -1] = 1.0
    kkt = np.block(
        [
            [system, selector.T],
            [selector, np.zeros((2, 2), dtype=np.float64)],
        ]
    )
    inverse = np.linalg.inv(kkt)
    inverse.setflags(write=False)
    return inverse


@lru_cache(maxsize=8)
def _unconstrained_inverse(
    horizon: int = HORIZON, curvature_lambda: float = CURVATURE_LAMBDA
) -> np.ndarray:
    d2 = second_difference_matrix(horizon)
    system = np.eye(horizon, dtype=np.float64) + float(curvature_lambda) * (d2.T @ d2)
    inverse = np.linalg.inv(system)
    inverse.setflags(write=False)
    return inverse


def smooth_cumulative_path(
    increments: Any,
    *,
    curvature_lambda: float = CURVATURE_LAMBDA,
    constrain_endpoints: bool = True,
) -> np.ndarray:
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != HORIZON:
        raise ValueError(f"increments must have shape [{HORIZON},d], received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("increments contain nonfinite values")
    cumulative = np.cumsum(values, axis=0)
    if constrain_endpoints:
        rhs = np.concatenate([cumulative, cumulative[[0, -1], :]], axis=0)
        solved = _constrained_kkt(HORIZON, float(curvature_lambda)) @ rhs
        smoothed = solved[:HORIZON]
    else:
        smoothed = _unconstrained_inverse(HORIZON, float(curvature_lambda)) @ cumulative
    decoded = np.empty_like(smoothed)
    decoded[0] = smoothed[0]
    decoded[1:] = np.diff(smoothed, axis=0)
    return decoded


def support_bounds(chunks: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    if not chunks:
        raise ValueError("at least one discovery chunk is required")
    array = np.concatenate([_as_chunk(chunk) for chunk in chunks], axis=0)
    return array.min(axis=0), array.max(axis=0)


def support_valid(
    chunk: Any,
    lower: Any,
    upper: Any,
    *,
    tolerance: float = SUPPORT_TOLERANCE,
) -> bool:
    array = _as_chunk(chunk)
    lo = np.asarray(lower, dtype=np.float64).reshape(ACTION_DIM)
    hi = np.asarray(upper, dtype=np.float64).reshape(ACTION_DIM)
    width = np.maximum(hi - lo, 1e-12)
    return bool(np.all(array >= lo - float(tolerance) * width) and np.all(array <= hi + float(tolerance) * width))


def hest_transform(
    chunk: Any,
    *,
    alpha: float = 1.0,
    lower: Any | None = None,
    upper: Any | None = None,
    curvature_lambda: float = CURVATURE_LAMBDA,
    tolerance: float = SUPPORT_TOLERANCE,
) -> tuple[np.ndarray, str | None]:
    base = _as_chunk(chunk)
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    smoothed = smooth_cumulative_path(base[:, :ARM_DIM], curvature_lambda=curvature_lambda)
    output = base.copy()
    output[:, :ARM_DIM] = (1.0 - float(alpha)) * base[:, :ARM_DIM] + float(alpha) * smoothed
    output[:, GRIPPER_DIM] = base[:, GRIPPER_DIM]
    if not np.isfinite(output).all():
        return base.copy(), "nonfinite_output"
    endpoint_error = float(np.max(np.abs(output[:, :ARM_DIM].sum(axis=0) - base[:, :ARM_DIM].sum(axis=0))))
    if endpoint_error > 1e-8:
        return base.copy(), "endpoint_error"
    if not np.array_equal(output[:, GRIPPER_DIM], base[:, GRIPPER_DIM]):
        return base.copy(), "gripper_changed"
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("lower and upper support bounds must be provided together")
        if not support_valid(output, lower, upper, tolerance=tolerance):
            return base.copy(), "support_violation"
    return output, None


def spline_proxy(chunk: Any, *, alpha: float = 1.0) -> np.ndarray:
    base = _as_chunk(chunk)
    smoothed = smooth_cumulative_path(base, constrain_endpoints=True)
    return (1.0 - float(alpha)) * base + float(alpha) * smoothed


def no_endpoint_ablation(chunk: Any, *, alpha: float = 1.0) -> np.ndarray:
    base = _as_chunk(chunk)
    smoothed = smooth_cumulative_path(base[:, :ARM_DIM], constrain_endpoints=False)
    output = base.copy()
    output[:, :ARM_DIM] = (1.0 - float(alpha)) * base[:, :ARM_DIM] + float(alpha) * smoothed
    return output


def moving_average_control(chunk: Any) -> np.ndarray:
    base = _as_chunk(chunk)
    padded = np.pad(base[:, :ARM_DIM], ((1, 1), (0, 0)), mode="edge")
    output = base.copy()
    output[:, :ARM_DIM] = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    return output


def cumulative_arm_energy(chunk: Any) -> float:
    array = _as_chunk(chunk)
    cumulative = np.cumsum(array[:, :ARM_DIM], axis=0)
    difference = second_difference_matrix(HORIZON) @ cumulative
    return float(np.mean(np.square(difference)))


def gripper_transition(chunk: Any, *, threshold: float = 1e-8) -> bool:
    array = _as_chunk(chunk)
    return bool(np.any(np.abs(np.diff(array[:, GRIPPER_DIM])) > float(threshold)))


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    manifest_keys = [str(row["window_key"]) for row in manifest_rows]
    partial_keys = [str(row["window_key"]) for row in partial_rows]
    manifest_set = set(manifest_keys)
    partial_set = set(partial_keys)
    discovery_keys = {str(row["window_key"]) for row in manifest_rows if row.get("partition") == "discovery"}
    validation_keys = {str(row["window_key"]) for row in manifest_rows if row.get("partition") == "validation"}
    return {
        "manifest_row_count": len(manifest_keys),
        "partial_row_count": len(partial_keys),
        "duplicate_manifest_key_count": len(manifest_keys) - len(manifest_set),
        "duplicate_partial_key_count": len(partial_keys) - len(partial_set),
        "missing_manifest_key_count": len(manifest_set - partial_set),
        "extra_partial_key_count": len(partial_set - manifest_set),
        "partition_overlap_count": len(discovery_keys & validation_keys),
        "key_sets_equal": manifest_set == partial_set,
    }


@dataclass(frozen=True)
class Stage0ADecisionInputs:
    proposal_hash_ok: bool
    manifest_audit_ok: bool
    source_finite_shape_ok: bool
    arm_support_noncollapsed: bool
    validation_transition_count: int
    endpoint_max_error: float
    first_action_max_error: float
    gripper_max_error: float
    all_variant_support_valid: bool
    acting_fraction: float
    median_energy_reduction: float
    comparator_distinct: bool
    roundtrip_max_error: float
    exception_count: int


def classify_stage0a(inputs: Stage0ADecisionInputs) -> str:
    if not inputs.proposal_hash_ok or not inputs.manifest_audit_ok or not inputs.source_finite_shape_ok:
        return "HEST_STAGE_0A_DATA_FAILURE"
    if not inputs.arm_support_noncollapsed or int(inputs.validation_transition_count) < 8:
        return "HEST_STAGE_0A_DATA_FAILURE"
    implementation_ok = (
        float(inputs.endpoint_max_error) <= 1e-8
        and float(inputs.first_action_max_error) <= 1e-8
        and float(inputs.gripper_max_error) == 0.0
        and bool(inputs.all_variant_support_valid)
        and float(inputs.roundtrip_max_error) <= 1e-12
        and int(inputs.exception_count) == 0
    )
    if not implementation_ok:
        return "HEST_STAGE_0A_IMPLEMENTATION_FAILURE"
    if float(inputs.acting_fraction) < 0.80 or float(inputs.median_energy_reduction) < 0.10:
        return "HEST_STAGE_0A_NO_HEADROOM"
    if not inputs.comparator_distinct:
        return "HEST_STAGE_0A_DESIGN_FAILURE_EQUIVALENT_CONTROL"
    return "HEST_STAGE_0A_PASS_STAGE_0B_ALLOWED"
