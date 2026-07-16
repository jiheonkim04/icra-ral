"""Frozen RAP-VLA Stage 0 retrieval-anchor and residual audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008"
ACTION_DIM = 7
ARM_DIM = 6
PROPRIO_DIM = 8
VISUAL_FEATURE_DIM = 960
TASK_COUNT = 4
PHASE_BINS = 10
CHUNK_SIZE = 50
TOP_K = 8
RIDGE_COEFFICIENT = 1e-4
STD_FLOOR = 1e-6
HUBER_DELTA = 1.0
ANCHOR_RELATIVE_GATE = 0.10
ANCHOR_ABSOLUTE_HUBER_GATE = 0.01
RESIDUAL_RELATIVE_GATE = 0.05
RESIDUAL_ABSOLUTE_HUBER_GATE = 0.01
MIN_TOPK_UNIQUE_DEMOS = 3
MAX_TOP1_SOURCE_FRACTION = 0.25


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


def rap_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["top_k"],
        row["policy_probe"],
    )
    return "|".join(str(value) for value in fields)


def rap_feature_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        "current",
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [rap_row_key(row) for row in manifest_rows]
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
            row["top_k"],
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


def raw_retrieval_feature(visual: Any, proprio: Any, task_index: int, phase: float) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float64).reshape(-1)
    proprio_value = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if visual_value.shape != (VISUAL_FEATURE_DIM,):
        raise ValueError(f"visual feature must have shape [{VISUAL_FEATURE_DIM}], received {visual_value.shape}")
    if proprio_value.shape != (PROPRIO_DIM,):
        raise ValueError(f"proprio feature must have shape [{PROPRIO_DIM}], received {proprio_value.shape}")
    continuous = np.concatenate([visual_value, proprio_value, np.asarray([float(phase)], dtype=np.float64)])
    if not np.isfinite(continuous).all():
        raise ValueError("retrieval feature contains nonfinite values")
    return np.concatenate([continuous, one_hot(task_index)])


def fit_discovery_zscore(features: Any) -> dict[str, Any]:
    value = _matrix(features, name="features", width=VISUAL_FEATURE_DIM + PROPRIO_DIM + 1 + TASK_COUNT)
    continuous = value[:, : VISUAL_FEATURE_DIM + PROPRIO_DIM + 1]
    mean = continuous.mean(axis=0)
    std = np.maximum(continuous.std(axis=0, ddof=0), STD_FLOOR)
    return {
        "continuous_dim": VISUAL_FEATURE_DIM + PROPRIO_DIM + 1,
        "task_count": TASK_COUNT,
        "mean": mean,
        "std": std,
        "discovery_row_count": int(len(value)),
    }


def apply_discovery_zscore(stats: Mapping[str, Any], features: Any) -> np.ndarray:
    value = _matrix(features, name="features", width=int(stats["continuous_dim"]) + int(stats["task_count"]))
    continuous_dim = int(stats["continuous_dim"])
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, continuous_dim)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, continuous_dim)
    continuous = (value[:, :continuous_dim] - mean) / std
    task = value[:, continuous_dim:]
    return np.concatenate([continuous, task], axis=1)


def retrieve_topk_same_task(
    query_features: Any,
    memory_features: Any,
    query_tasks: Sequence[Any],
    memory_tasks: Sequence[Any],
    *,
    k: int = TOP_K,
    query_keys: Sequence[str] | None = None,
    memory_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    query = _matrix(query_features, name="query_features")
    memory = _matrix(memory_features, name="memory_features", width=query.shape[1])
    if len(query_tasks) != len(query) or len(memory_tasks) != len(memory):
        raise ValueError("task labels must align with query and memory rows")
    if query_keys is not None and len(query_keys) != len(query):
        raise ValueError("query_keys must align with query rows")
    if memory_keys is not None and len(memory_keys) != len(memory):
        raise ValueError("memory_keys must align with memory rows")
    results: list[dict[str, Any]] = []
    memory_task_array = np.asarray(memory_tasks, dtype=object)
    for index, vector in enumerate(query):
        legal = memory_task_array == query_tasks[index]
        if query_keys is not None and memory_keys is not None:
            legal &= np.asarray(memory_keys, dtype=object) != query_keys[index]
        legal_indices = np.flatnonzero(legal)
        if len(legal_indices) == 0:
            raise ValueError(f"no same-task memory rows for query {index}")
        distances = np.sum(np.square(memory[legal_indices] - vector.reshape(1, -1)), axis=1)
        order = np.argsort(distances, kind="mergesort")[: int(k)]
        selected = legal_indices[order]
        results.append(
            {
                "indices": selected.astype(np.int64),
                "distances": distances[order].astype(np.float64),
                "available_count": int(len(legal_indices)),
            }
        )
    return results


def uniform_anchor(memory_chunks: Any, retrievals: Sequence[Mapping[str, Any]]) -> np.ndarray:
    chunks = _chunk_matrix(memory_chunks)
    anchors = []
    for retrieval in retrievals:
        indices = np.asarray(retrieval["indices"], dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("retrieval must contain at least one index")
        anchors.append(chunks[indices].mean(axis=0))
    return np.asarray(anchors, dtype=np.float64)


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
    phase_values = list(phases)
    for chunk, task, phase in zip(source, task_values, phase_values, strict=True):
        buckets.setdefault((task, phase_bin(float(phase), bins)), []).append(chunk)
    task_fallbacks: dict[Any, np.ndarray] = {}
    for task in sorted({str(value) for value in task_values}):
        selected = source[np.asarray(task_values, dtype=object) == task]
        task_fallbacks[task] = selected.mean(axis=0)
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


def retrieval_noncollapse(
    retrievals: Sequence[Mapping[str, Any]],
    memory_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    top1_counts: dict[str, int] = {}
    unique_demo_counts = []
    for retrieval in retrievals:
        indices = np.asarray(retrieval["indices"], dtype=np.int64)
        if len(indices) == 0:
            continue
        demos = {int(memory_rows[int(index)]["demo_id"]) for index in indices}
        unique_demo_counts.append(len(demos))
        top1_key = str(memory_rows[int(indices[0])]["row_key"])
        top1_counts[top1_key] = top1_counts.get(top1_key, 0) + 1
    total = max(sum(top1_counts.values()), 1)
    return {
        "median_topk_unique_demos": float(np.median(unique_demo_counts)) if unique_demo_counts else 0.0,
        "top1_max_source_fraction": max((count / total for count in top1_counts.values()), default=1.0),
        "top1_unique_source_count": int(len(top1_counts)),
    }


def flattened_chunks(chunks: Any) -> np.ndarray:
    value = _chunk_matrix(chunks)
    return value.reshape(value.shape[0], CHUNK_SIZE * ACTION_DIM)


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
    median_topk_unique_demos: float
    top1_max_source_fraction: float
    anchor_relative_improvement: float
    anchor_absolute_huber_improvement: float
    residual_variance_all_positive: bool
    residual_probe_relative_improvement: float
    residual_probe_absolute_huber_improvement: float
    anchor_and_residual_paths_distinct: bool
    finite_objectives_and_gradients: bool
    rap_gradient_nonzero: bool
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
        return "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < 512
        or int(inputs.minimum_validation_windows) < 128
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or float(inputs.median_topk_unique_demos) < MIN_TOPK_UNIQUE_DEMOS
        or float(inputs.top1_max_source_fraction) > MAX_TOP1_SOURCE_FRACTION
        or not inputs.residual_variance_all_positive
    ):
        return "RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    anchor_ok = (
        float(inputs.anchor_relative_improvement) >= ANCHOR_RELATIVE_GATE
        or float(inputs.anchor_absolute_huber_improvement) >= ANCHOR_ABSOLUTE_HUBER_GATE
    )
    residual_ok = (
        float(inputs.residual_probe_relative_improvement) >= RESIDUAL_RELATIVE_GATE
        or float(inputs.residual_probe_absolute_huber_improvement) >= RESIDUAL_ABSOLUTE_HUBER_GATE
    )
    if not anchor_ok or not residual_ok:
        return "RAP_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not inputs.anchor_and_residual_paths_distinct
        or not inputs.finite_objectives_and_gradients
        or not inputs.rap_gradient_nonzero
        or not inputs.gradient_ratio_at_most_100
        or int(inputs.frozen_parameter_gradient_count) != 0
    ):
        return "RAP_STAGE_0_DESIGN_FAILURE"
    return "RAP_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
