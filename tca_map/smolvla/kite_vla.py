"""Frozen KITE-VLA realization operators and Stage 0A audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8"
ACTION_DIM = 7
ARM_DIM = 6
HORIZONS = (5, 20)
RIDGE_COEFFICIENT = 1e-4
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0


def json_default(value: Any) -> Any:
    """Convert supported structured scientific values to ordinary JSON values."""
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


def realization_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["horizon"],
    )
    return "|".join(str(value) for value in fields)


def frame_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
    )
    return "|".join(str(value) for value in fields)


def cumulative_arm_command(actions: Any, frame_index: int, horizon: int) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    frame = int(frame_index)
    length = int(horizon)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTION_DIM}], received {value.shape}")
    if length not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if frame < 0 or frame + length >= len(value):
        raise ValueError("frame must have a complete future state at the requested horizon")
    if not np.isfinite(value[frame : frame + length]).all():
        raise ValueError("action window contains nonfinite values")
    return value[frame : frame + length, :ARM_DIM].sum(axis=0)


def state_displacement(states: Any, frame_index: int, horizon: int) -> np.ndarray:
    value = np.asarray(states, dtype=np.float64)
    frame = int(frame_index)
    length = int(horizon)
    if value.ndim != 2 or value.shape[1] != ARM_DIM:
        raise ValueError(f"ee_states must have shape [T,{ARM_DIM}], received {value.shape}")
    if length not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if frame < 0 or frame + length >= len(value):
        raise ValueError("frame must have a complete future state at the requested horizon")
    pair = value[[frame, frame + length]]
    if not np.isfinite(pair).all():
        raise ValueError("state endpoints contain nonfinite values")
    return pair[1] - pair[0]


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != ARM_DIM or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape [N,{ARM_DIM}], received {array.shape}")
    return array


def fit_realization_operator(
    commands: Any,
    displacements: Any,
    *,
    ridge: float = RIDGE_COEFFICIENT,
    std_floor: float = STD_FLOOR,
) -> dict[str, Any]:
    x = _matrix(commands, name="commands")
    y = _matrix(displacements, name="displacements")
    if len(x) != len(y) or len(x) < ARM_DIM:
        raise ValueError("commands and displacements require aligned rows and at least six samples")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_std = np.maximum(x.std(axis=0, ddof=0), float(std_floor))
    y_std = np.maximum(y.std(axis=0, ddof=0), float(std_floor))
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    design = np.concatenate([x_norm, np.ones((len(x_norm), 1), dtype=np.float64)], axis=1)
    penalty = np.eye(ARM_DIM + 1, dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_norm)
    coefficient = beta[:ARM_DIM]
    intercept = beta[ARM_DIM]
    singular_values = np.linalg.svd(coefficient, compute_uv=False)
    return {
        "ridge_coefficient": float(ridge),
        "std_floor": float(std_floor),
        "command_mean": x_mean,
        "command_std": x_std,
        "displacement_mean": y_mean,
        "displacement_std": y_std,
        "coefficient": coefficient,
        "intercept": intercept,
        "rank": int(np.linalg.matrix_rank(coefficient)),
        "singular_values": singular_values,
        "discovery_row_count": int(len(x)),
    }


def predict_realization(operator: Mapping[str, Any], commands: Any, *, normalized: bool = False) -> np.ndarray:
    x = _matrix(commands, name="commands")
    x_mean = np.asarray(operator["command_mean"], dtype=np.float64).reshape(ARM_DIM)
    x_std = np.asarray(operator["command_std"], dtype=np.float64).reshape(ARM_DIM)
    coefficient = np.asarray(operator["coefficient"], dtype=np.float64).reshape(ARM_DIM, ARM_DIM)
    intercept = np.asarray(operator["intercept"], dtype=np.float64).reshape(ARM_DIM)
    prediction = ((x - x_mean) / x_std) @ coefficient + intercept
    if normalized:
        return prediction
    y_mean = np.asarray(operator["displacement_mean"], dtype=np.float64).reshape(ARM_DIM)
    y_std = np.asarray(operator["displacement_std"], dtype=np.float64).reshape(ARM_DIM)
    return prediction * y_std + y_mean


def realization_metrics(operator: Mapping[str, Any], commands: Any, displacements: Any) -> dict[str, float]:
    target = _matrix(displacements, name="displacements")
    predicted = predict_realization(operator, commands)
    mean = np.asarray(operator["displacement_mean"], dtype=np.float64).reshape(1, ARM_DIM)
    std = np.asarray(operator["displacement_std"], dtype=np.float64).reshape(1, ARM_DIM)
    target_norm = (target - mean) / std
    predicted_norm = (predicted - mean) / std
    model_mse = float(np.mean(np.square(predicted_norm - target_norm)))
    baseline_mse = float(np.mean(np.square(target_norm)))
    relative_improvement = (baseline_mse - model_mse) / max(baseline_mse, 1e-12)
    return {
        "normalized_model_mse": model_mse,
        "normalized_discovery_mean_mse": baseline_mse,
        "normalized_relative_improvement": float(relative_improvement),
        "raw_model_mse": float(np.mean(np.square(predicted - target))),
        "raw_discovery_mean_mse": float(np.mean(np.square(mean - target))),
    }


def huber_loss(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    values = np.where(absolute <= delta, 0.5 * np.square(error), delta * (absolute - 0.5 * delta))
    return float(np.mean(values))


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [realization_row_key(row) for row in manifest_rows]
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
            row["horizon"],
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


def differentiable_mean_std_unnormalize(actions: Any, mean: Any, std: Any) -> Any:
    """Apply the checkpoint's affine action transform without breaking autograd."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in research environments
        torch = None
    if torch is not None and isinstance(actions, torch.Tensor):
        mu = torch.as_tensor(mean, device=actions.device, dtype=actions.dtype)
        scale = torch.as_tensor(std, device=actions.device, dtype=actions.dtype)
        return actions * scale + mu
    return np.asarray(actions) * np.asarray(std) + np.asarray(mean)


def torch_realization_normalized(operator: Mapping[str, Any], commands: Any) -> Any:
    """Apply a frozen standardized operator while preserving action gradients."""
    import torch

    x_mean = torch.as_tensor(operator["command_mean"], device=commands.device, dtype=commands.dtype)
    x_std = torch.as_tensor(operator["command_std"], device=commands.device, dtype=commands.dtype)
    coefficient = torch.as_tensor(operator["coefficient"], device=commands.device, dtype=commands.dtype)
    intercept = torch.as_tensor(operator["intercept"], device=commands.device, dtype=commands.dtype)
    return ((commands - x_mean) / x_std) @ coefficient + intercept


@dataclass(frozen=True)
class Stage0ADecisionInputs:
    proposal_hash_ok: bool
    serializer_preflight_ok: bool
    manifest_integrity_ok: bool
    source_alignment_ok: bool
    minimum_discovery_rows_per_horizon: int
    minimum_validation_rows_per_horizon: int
    command_variance_all_positive: bool
    state_variance_all_positive: bool
    maximum_sampled_task_fraction: float
    all_operator_ranks_six: bool
    minimum_operator_relative_improvement: float
    all_tasks_reported: bool
    base_headroom_passed: bool
    finite_objectives_and_gradients: bool
    kite_gradient_nonzero: bool
    gradient_ratio_at_most_100: bool
    frozen_parameter_gradient_count: int
    identity_max_error: float
    base_hash_unchanged: bool
    checkpoint_reload_ok: bool
    action_validity_ok: bool
    exception_count: int


def classify_stage0a(inputs: Stage0ADecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or float(inputs.identity_max_error) > 1e-6
        or not inputs.base_hash_unchanged
        or not inputs.checkpoint_reload_ok
        or not inputs.action_validity_ok
        or int(inputs.exception_count) != 0
    ):
        return "KITE_STAGE_0A_IMPLEMENTATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or int(inputs.minimum_discovery_rows_per_horizon) < 512
        or int(inputs.minimum_validation_rows_per_horizon) < 96
        or not inputs.command_variance_all_positive
        or not inputs.state_variance_all_positive
        or float(inputs.maximum_sampled_task_fraction) > 0.40
        or not inputs.all_operator_ranks_six
        or float(inputs.minimum_operator_relative_improvement) < 0.50
        or not inputs.all_tasks_reported
    ):
        return "KITE_STAGE_0A_DATA_FAILURE"
    if not inputs.base_headroom_passed:
        return "KITE_STAGE_0A_NO_HEADROOM"
    if (
        not inputs.finite_objectives_and_gradients
        or not inputs.kite_gradient_nonzero
        or not inputs.gradient_ratio_at_most_100
        or int(inputs.frozen_parameter_gradient_count) != 0
    ):
        return "KITE_STAGE_0A_DESIGN_FAILURE"
    return "KITE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
