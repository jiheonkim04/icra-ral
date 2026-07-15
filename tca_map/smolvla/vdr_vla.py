"""Frozen VDR-VLA Stage 0A data and objective audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72"
ACTION_DIM = 7
ARM_DIM = 6
FEATURE_DIM = 960
PROJECTION_DIM = 32
HORIZONS = (4, 12)
RIDGE_COEFFICIENT = 1e-4
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0
STATIC_RELATIVE_GATE = 0.25
RESIDUAL_RELATIVE_GATE = 0.05
RESIDUAL_ABSOLUTE_GATE = 0.02
FUTURE_PROXY_RELATIVE_GATE = 0.05
FUTURE_PROXY_ABSOLUTE_GATE = 0.02


def json_default(value: Any) -> Any:
    """Convert structured scientific values into strict JSON values."""
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


def vdr_row_key(row: Mapping[str, Any]) -> str:
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


def visual_frame_key(row: Mapping[str, Any], frame_index: int | None = None) -> str:
    frame = row["frame_index"] if frame_index is None else frame_index
    fields = (
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        frame,
    )
    return "|".join(str(value) for value in fields)


def action_summary(actions: Any, frame_index: int, horizon: int) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    frame = int(frame_index)
    length = int(horizon)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTION_DIM}], received {value.shape}")
    if length not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if frame < 0 or frame + length >= len(value):
        raise ValueError("frame must have complete action and future-feature windows")
    window = value[frame : frame + length]
    if not np.isfinite(window).all():
        raise ValueError("action window contains nonfinite values")
    return np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0, ddof=0),
            window[:, :ARM_DIM].sum(axis=0),
            window[-1:, 6],
        ]
    )


def _matrix(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix, received {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, received {array.shape}")
    return array


def _positive_std(value: np.ndarray, *, floor: float = STD_FLOOR) -> np.ndarray:
    return np.maximum(value.std(axis=0, ddof=0), float(floor))


def fit_pca_whitener(deltas: Any, *, projection_dim: int = PROJECTION_DIM) -> dict[str, Any]:
    matrix = _matrix(deltas, name="deltas", width=FEATURE_DIM)
    if len(matrix) <= projection_dim:
        raise ValueError("PCA whitening requires more rows than retained dimensions")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:projection_dim].copy()
    projected = centered @ components.T
    std = np.maximum(projected.std(axis=0, ddof=0), STD_FLOOR)
    return {
        "feature_dim": FEATURE_DIM,
        "projection_dim": int(projection_dim),
        "mean": mean,
        "components": components,
        "projected_std": std,
        "singular_values": singular_values[:projection_dim],
        "discovery_row_count": int(len(matrix)),
    }


def project_with_whitener(whitener: Mapping[str, Any], deltas: Any) -> np.ndarray:
    matrix = _matrix(deltas, name="deltas", width=int(whitener["feature_dim"]))
    mean = np.asarray(whitener["mean"], dtype=np.float64).reshape(1, int(whitener["feature_dim"]))
    components = np.asarray(whitener["components"], dtype=np.float64)
    std = np.asarray(whitener["projected_std"], dtype=np.float64).reshape(1, int(whitener["projection_dim"]))
    return ((matrix - mean) @ components.T) / std


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


def regression_metrics(model: Mapping[str, Any], features: Any, targets: Any) -> dict[str, float]:
    target = _matrix(targets, name="targets", width=int(model["target_dim"]))
    prediction = predict_ridge(model, features)
    target_mean = np.asarray(model["target_mean"], dtype=np.float64).reshape(1, int(model["target_dim"]))
    target_std = np.asarray(model["target_std"], dtype=np.float64).reshape(1, int(model["target_dim"]))
    target_norm = (target - target_mean) / target_std
    prediction_norm = (prediction - target_mean) / target_std
    model_mse = float(np.mean(np.square(prediction_norm - target_norm)))
    baseline_mse = float(np.mean(np.square(target_norm)))
    return {
        "normalized_model_mse": model_mse,
        "normalized_discovery_mean_mse": baseline_mse,
        "normalized_relative_improvement": float((baseline_mse - model_mse) / max(baseline_mse, 1e-12)),
        "raw_model_mse": float(np.mean(np.square(prediction - target))),
        "raw_discovery_mean_mse": float(np.mean(np.square(target_mean - target))),
    }


def mean_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    loss = np.where(absolute <= delta, 0.5 * np.square(error), delta * (absolute - 0.5 * delta))
    return float(np.mean(loss))


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [vdr_row_key(row) for row in manifest_rows]
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


def torch_action_summary(actions: Any, horizon: int) -> Any:
    import torch

    if int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    window = actions[:, : int(horizon), :ACTION_DIM]
    return torch.cat(
        [
            window.mean(dim=1),
            window.std(dim=1, unbiased=False),
            window[:, :, :ARM_DIM].sum(dim=1),
            window[:, -1:, 6],
        ],
        dim=1,
    )


def torch_predict_ridge(model: Mapping[str, Any], features: Any) -> Any:
    import torch

    x_mean = torch.as_tensor(model["feature_mean"], dtype=features.dtype, device=features.device)
    x_std = torch.as_tensor(model["feature_std"], dtype=features.dtype, device=features.device)
    beta = torch.as_tensor(model["beta"], dtype=features.dtype, device=features.device)
    normalized = (features - x_mean) / x_std
    design = torch.cat([normalized, torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)], dim=1)
    y_norm = design @ beta
    y_mean = torch.as_tensor(model["target_mean"], dtype=features.dtype, device=features.device)
    y_std = torch.as_tensor(model["target_std"], dtype=features.dtype, device=features.device)
    return y_norm * y_std + y_mean


def differentiable_mean_std_unnormalize(actions: Any, mean: Any, std: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(actions, torch.Tensor):
        mu = torch.as_tensor(mean, device=actions.device, dtype=actions.dtype)
        scale = torch.as_tensor(std, device=actions.device, dtype=actions.dtype)
        return actions * scale + mu
    return np.asarray(actions) * np.asarray(std) + np.asarray(mean)


@dataclass(frozen=True)
class Stage0ADecisionInputs:
    proposal_hash_ok: bool
    serializer_preflight_ok: bool
    manifest_integrity_ok: bool
    source_alignment_ok: bool
    feature_action_proprio_finite_aligned: bool
    minimum_discovery_rows_per_horizon: int
    minimum_validation_rows_per_horizon: int
    residual_variance_all_positive: bool
    maximum_validation_task_fraction: float
    all_tasks_reported: bool
    static_predictor_relative_improvement: float
    action_residual_relative_improvement: float
    action_residual_absolute_improvement: float
    future_proxy_relative_improvement: float
    future_proxy_absolute_gap: float
    finite_objectives_and_gradients: bool
    vdr_gradient_nonzero: bool
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
        return "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_rows_per_horizon) < 512
        or int(inputs.minimum_validation_rows_per_horizon) < 128
        or not inputs.residual_variance_all_positive
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.all_tasks_reported
    ):
        return "VDR_STAGE_0A_DATA_OR_SUPERVISION_FAILURE"
    residual_probe_ok = (
        float(inputs.action_residual_relative_improvement) >= RESIDUAL_RELATIVE_GATE
        or float(inputs.action_residual_absolute_improvement) >= RESIDUAL_ABSOLUTE_GATE
    )
    future_proxy_ok = (
        float(inputs.future_proxy_relative_improvement) >= FUTURE_PROXY_RELATIVE_GATE
        or float(inputs.future_proxy_absolute_gap) >= FUTURE_PROXY_ABSOLUTE_GATE
    )
    if (
        float(inputs.static_predictor_relative_improvement) < STATIC_RELATIVE_GATE
        or not residual_probe_ok
        or not future_proxy_ok
    ):
        return "VDR_STAGE_0A_NO_USABLE_HEADROOM"
    if (
        not inputs.finite_objectives_and_gradients
        or not inputs.vdr_gradient_nonzero
        or not inputs.gradient_ratio_at_most_100
        or int(inputs.frozen_parameter_gradient_count) != 0
    ):
        return "VDR_STAGE_0A_DESIGN_FAILURE"
    return "VDR_STAGE_0A_PASS_STAGE_0B_ALLOWED"
