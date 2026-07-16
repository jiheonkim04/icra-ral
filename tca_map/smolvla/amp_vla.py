"""Frozen AMP-VLA Stage 0 action-manifold audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4"
ACTION_DIM = 7
PROPRIO_DIM = 8
VISUAL_FEATURE_DIM = 960
TASK_COUNT = 4
PHASE_BINS = 10
CHUNK_SIZE = 50
LATENT_DIMS = (8, 16)
RIDGE_COEFFICIENT = 1e-4
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0
MANIFOLD_RECON_RELATIVE_GATE = 0.10
MANIFOLD_RECON_ABSOLUTE_HUBER_GATE = 0.01
COORDINATE_RELATIVE_GATE = 0.05
COORDINATE_ABSOLUTE_HUBER_GATE = 0.005
ABOT_HEADROOM_RELATIVE_GATE = 0.05
ABOT_HEADROOM_ABSOLUTE_HUBER_GATE = 0.005


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


def amp_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["latent_dim"],
        row["policy_probe"],
    )
    return "|".join(str(value) for value in fields)


def amp_feature_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        "amp_current",
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [amp_row_key(row) for row in manifest_rows]
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
            row["latent_dim"],
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


def fit_action_manifold(chunks: Any, *, latent_dim: int, std_floor: float = STD_FLOOR) -> dict[str, Any]:
    flat = flattened_chunks(chunks)
    latent = int(latent_dim)
    if latent <= 0 or latent > flat.shape[1]:
        raise ValueError("latent_dim must be positive and no larger than flattened action dimension")
    mean = flat.mean(axis=0)
    centered = flat - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:latent]
    coordinates = centered @ components.T
    coord_mean = coordinates.mean(axis=0)
    coord_std = np.maximum(coordinates.std(axis=0, ddof=0), float(std_floor))
    variance = np.square(singular) / max(len(flat) - 1, 1)
    total_variance = float(np.sum(variance))
    explained = variance[:latent] / max(total_variance, 1e-12)
    return {
        "latent_dim": latent,
        "std_floor": float(std_floor),
        "action_shape": [CHUNK_SIZE, ACTION_DIM],
        "mean": mean,
        "components": components,
        "coordinate_mean": coord_mean,
        "coordinate_std": coord_std,
        "explained_variance_ratio": explained,
        "discovery_row_count": int(len(flat)),
    }


def encode_manifold(model: Mapping[str, Any], chunks: Any, *, standardized: bool = False) -> np.ndarray:
    flat = flattened_chunks(chunks)
    mean = np.asarray(model["mean"], dtype=np.float64).reshape(1, CHUNK_SIZE * ACTION_DIM)
    components = np.asarray(model["components"], dtype=np.float64)
    coords = (flat - mean) @ components.T
    if standardized:
        coord_mean = np.asarray(model["coordinate_mean"], dtype=np.float64).reshape(1, -1)
        coord_std = np.asarray(model["coordinate_std"], dtype=np.float64).reshape(1, -1)
        coords = (coords - coord_mean) / coord_std
    return coords


def decode_manifold(model: Mapping[str, Any], coordinates: Any, *, standardized: bool = False) -> np.ndarray:
    coords = _matrix(coordinates, name="coordinates", width=int(model["latent_dim"]))
    if standardized:
        coord_mean = np.asarray(model["coordinate_mean"], dtype=np.float64).reshape(1, -1)
        coord_std = np.asarray(model["coordinate_std"], dtype=np.float64).reshape(1, -1)
        coords = coords * coord_std + coord_mean
    mean = np.asarray(model["mean"], dtype=np.float64).reshape(1, CHUNK_SIZE * ACTION_DIM)
    components = np.asarray(model["components"], dtype=np.float64)
    flat = coords @ components + mean
    return flat.reshape((-1, CHUNK_SIZE, ACTION_DIM))


def project_to_manifold(model: Mapping[str, Any], chunks: Any) -> np.ndarray:
    return decode_manifold(model, encode_manifold(model, chunks))


def manifold_consistency(model: Mapping[str, Any], chunks: Any) -> float:
    return mean_huber(chunks, project_to_manifold(model, chunks))


def task_phase_mean_chunks(
    chunks: Any,
    tasks: Sequence[Any],
    phases: Sequence[float],
    query_tasks: Sequence[Any],
    query_phases: Sequence[float],
    *,
    bins: int = PHASE_BINS,
) -> np.ndarray:
    source = _chunk_matrix(chunks)
    if len(tasks) != len(source) or len(phases) != len(source):
        raise ValueError("source tasks and phases must align with chunks")
    buckets: dict[tuple[Any, int], list[np.ndarray]] = {}
    task_values = list(tasks)
    for chunk, task, phase in zip(source, task_values, phases, strict=True):
        buckets.setdefault((task, phase_bin(float(phase), bins)), []).append(chunk)
    task_fallbacks = {
        task: source[np.asarray(task_values, dtype=object) == task].mean(axis=0)
        for task in sorted({str(value) for value in task_values})
    }
    global_fallback = source.mean(axis=0)
    predictions = []
    for task, phase in zip(query_tasks, query_phases, strict=True):
        key = (task, phase_bin(float(phase), bins))
        if key in buckets:
            predictions.append(np.asarray(buckets[key]).mean(axis=0))
        elif task in task_fallbacks:
            predictions.append(task_fallbacks[task])
        else:
            predictions.append(global_fallback)
    return np.asarray(predictions, dtype=np.float64)


def task_phase_mean_coordinates(
    coordinates: Any,
    tasks: Sequence[Any],
    phases: Sequence[float],
    query_tasks: Sequence[Any],
    query_phases: Sequence[float],
    *,
    bins: int = PHASE_BINS,
) -> np.ndarray:
    source = _matrix(coordinates, name="coordinates")
    if len(tasks) != len(source) or len(phases) != len(source):
        raise ValueError("source tasks and phases must align with coordinates")
    buckets: dict[tuple[Any, int], list[np.ndarray]] = {}
    task_values = list(tasks)
    for coord, task, phase in zip(source, task_values, phases, strict=True):
        buckets.setdefault((task, phase_bin(float(phase), bins)), []).append(coord)
    task_fallbacks = {
        task: source[np.asarray(task_values, dtype=object) == task].mean(axis=0)
        for task in sorted({str(value) for value in task_values})
    }
    global_fallback = source.mean(axis=0)
    predictions = []
    for task, phase in zip(query_tasks, query_phases, strict=True):
        key = (task, phase_bin(float(phase), bins))
        if key in buckets:
            predictions.append(np.asarray(buckets[key]).mean(axis=0))
        elif task in task_fallbacks:
            predictions.append(task_fallbacks[task])
        else:
            predictions.append(global_fallback)
    return np.asarray(predictions, dtype=np.float64)


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
    coordinate_variance_all_positive: bool
    manifold_reconstruction_relative_improvement: float
    manifold_reconstruction_absolute_huber_improvement: float
    coordinate_probe_relative_improvement: float
    coordinate_probe_absolute_huber_improvement: float
    abot_proxy_headroom_relative_improvement: float
    abot_proxy_headroom_absolute_huber_improvement: float
    clipping_explains_projection: bool
    projection_path_distinct: bool
    finite_objectives_and_gradients: bool
    amp_gradient_nonzero: bool
    gradient_ratio_at_most_100: bool
    frozen_parameter_gradient_count: int
    identity_max_error: float
    base_hash_unchanged: bool
    checkpoint_reload_ok: bool
    action_validity_ok: bool
    exception_count: int


def classify_stage0(inputs: Stage0DecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or not inputs.official_prior_asset_check_persisted
        or float(inputs.identity_max_error) > 1e-6
        or not inputs.base_hash_unchanged
        or not inputs.checkpoint_reload_ok
        or not inputs.action_validity_ok
        or int(inputs.exception_count) != 0
    ):
        return "AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < 512
        or int(inputs.minimum_validation_windows) < 128
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.coordinate_variance_all_positive
    ):
        return "AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    manifold_ok = (
        float(inputs.manifold_reconstruction_relative_improvement) >= MANIFOLD_RECON_RELATIVE_GATE
        or float(inputs.manifold_reconstruction_absolute_huber_improvement) >= MANIFOLD_RECON_ABSOLUTE_HUBER_GATE
    )
    coordinate_ok = (
        float(inputs.coordinate_probe_relative_improvement) >= COORDINATE_RELATIVE_GATE
        or float(inputs.coordinate_probe_absolute_huber_improvement) >= COORDINATE_ABSOLUTE_HUBER_GATE
    )
    headroom_ok = (
        float(inputs.abot_proxy_headroom_relative_improvement) >= ABOT_HEADROOM_RELATIVE_GATE
        or float(inputs.abot_proxy_headroom_absolute_huber_improvement) >= ABOT_HEADROOM_ABSOLUTE_HUBER_GATE
    )
    if not manifold_ok or not headroom_ok or inputs.clipping_explains_projection:
        return "AMP_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not coordinate_ok
        or not inputs.projection_path_distinct
        or not inputs.finite_objectives_and_gradients
        or not inputs.amp_gradient_nonzero
        or not inputs.gradient_ratio_at_most_100
        or int(inputs.frozen_parameter_gradient_count) != 0
    ):
        return "AMP_STAGE_0_DESIGN_FAILURE"
    return "AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


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
