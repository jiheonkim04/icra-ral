"""Frozen URF-VLA Stage 0 uncertainty-routed residual helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532"
ACTION_DIM = 7
PROPRIO_DIM = 8
VISUAL_FEATURE_DIM = 960
TASK_COUNT = 4
PHASE_BINS = 10
CHUNK_SIZE = 50
RESIDUAL_SCALE_FLOOR = 1e-4
RESIDUAL_SCALE_CEILING = 10.0
LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 4.0
DEFAULT_RESIDUAL_CAP = 2.0
DEFAULT_G_MAX = 0.10
DEFAULT_KAPPA = 1.0
HUBER_DELTA = 1.0
ACTION_HUBER_DELTA = 0.05
RIDGE_COEFFICIENT = 1e-4
ROUTE_POSITIVE_MIN = 0.02
ROUTE_POSITIVE_MAX = 0.80
UNCERTAINTY_SPEARMAN_MIN = 0.20
HEADROOM_RELATIVE_GATE = 0.05
HEADROOM_ABSOLUTE_HUBER_GATE = 0.005
GRADIENT_RATIO_MAX = 100.0


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


def urf_row_key(row: Mapping[str, Any]) -> str:
    fields = [
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["policy_probe"],
    ]
    for name in ("model_or_probe", "proxy_variant", "g_max", "lambda_clean", "tau_g_family"):
        if name in row and row[name] is not None:
            fields.append(f"{name}={row[name]}")
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [urf_row_key(row) for row in manifest_rows]
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


def flattened_chunks(action_chunks: Any) -> np.ndarray:
    chunks = _chunk_matrix(action_chunks)
    return chunks.reshape(chunks.shape[0], CHUNK_SIZE * ACTION_DIM)


def fit_residual_scale(
    base_chunks: Any,
    expert_chunks: Any,
    *,
    floor: float = RESIDUAL_SCALE_FLOOR,
    ceiling: float = RESIDUAL_SCALE_CEILING,
) -> dict[str, Any]:
    base = _chunk_matrix(base_chunks, "base_chunks")
    expert = _chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunk shapes differ: {base.shape} vs {expert.shape}")
    residual = expert - base
    flat = residual.reshape(-1, ACTION_DIM)
    raw_scale = np.std(flat, axis=0, ddof=0)
    collapsed = raw_scale < float(floor)
    scale = np.clip(np.maximum(raw_scale, float(floor)), float(floor), float(ceiling))
    return {
        "action_dimension": ACTION_DIM,
        "scale": scale,
        "raw_scale": raw_scale,
        "collapsed_residual_scale_mask": collapsed,
        "collapsed_residual_scale_count": int(collapsed.sum()),
        "residual_scale_min": float(np.min(scale)),
        "residual_scale_max": float(np.max(scale)),
        "residual_scale_noncollapsed": bool(not np.any(collapsed)),
    }


def normalized_residual(base_chunks: Any, expert_chunks: Any, residual_scale: Any) -> np.ndarray:
    base = _chunk_matrix(base_chunks, "base_chunks")
    expert = _chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunk shapes differ: {base.shape} vs {expert.shape}")
    scale = _scale_vector(residual_scale)
    return (expert - base) / scale.reshape(1, 1, ACTION_DIM)


def route_thresholds(normalized_residuals: Any, *, quantile: float = 0.70, floor: float = 0.25) -> np.ndarray:
    value = np.abs(_chunk_matrix(normalized_residuals, "normalized_residuals")).reshape(-1, ACTION_DIM)
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0,1)")
    thresholds = np.quantile(value, float(quantile), axis=0)
    return np.maximum(thresholds, float(floor))


def route_labels(normalized_residuals: Any, thresholds: Any | None = None) -> np.ndarray:
    value = _chunk_matrix(normalized_residuals, "normalized_residuals")
    threshold = route_thresholds(value) if thresholds is None else _scale_vector(thresholds)
    return np.abs(value) >= threshold.reshape(1, 1, ACTION_DIM)


def route_positive_fraction(labels: Any) -> float:
    value = np.asarray(labels, dtype=bool)
    if value.size == 0:
        raise ValueError("labels must be nonempty")
    return float(np.mean(value))


def route_label_health(labels: Any) -> dict[str, Any]:
    value = np.asarray(labels, dtype=bool)
    positive = route_positive_fraction(value)
    by_dim = value.reshape(-1, ACTION_DIM).mean(axis=0)
    return {
        "route_label_positive_fraction": positive,
        "route_label_noncollapsed": bool(ROUTE_POSITIVE_MIN <= positive <= ROUTE_POSITIVE_MAX),
        "route_label_positive_fraction_by_dim": by_dim,
        "route_label_all_zero": bool(np.all(~value)),
        "route_label_all_one": bool(np.all(value)),
    }


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


def heteroscedastic_huber_nll(
    target: Any,
    mean: Any,
    log_var: Any,
    *,
    delta: float = HUBER_DELTA,
    reduce: bool = True,
) -> float | np.ndarray:
    y = _chunk_matrix(target, "target")
    mu = _chunk_matrix(mean, "mean")
    ell = _broadcast_chunk(log_var, y.shape, "log_var")
    if y.shape != mu.shape:
        raise ValueError(f"target and mean shapes differ: {y.shape} vs {mu.shape}")
    ell = np.clip(ell, LOG_VAR_MIN, LOG_VAR_MAX)
    loss = np.exp(-ell) * huber_values(y - mu, delta=delta) + 0.5 * ell
    if reduce:
        return float(np.mean(loss))
    return loss


def route_logits(
    residual_mean: Any,
    log_var: Any,
    *,
    q_base: Any | None = None,
    alpha_m: float = 1.0,
    alpha_u: float = 1.0,
    tau_g: float = 0.0,
) -> np.ndarray:
    mu = _chunk_matrix(residual_mean, "residual_mean")
    ell = _broadcast_chunk(log_var, mu.shape, "log_var")
    if q_base is None:
        base = np.zeros_like(mu)
    else:
        base = _broadcast_chunk(q_base, mu.shape, "q_base")
    std = np.sqrt(np.exp(np.clip(ell, LOG_VAR_MIN, LOG_VAR_MAX)))
    return base + float(alpha_m) * np.abs(mu) - float(alpha_u) * std - float(tau_g)


def route_gate(route_logit: Any, *, eta: float = 1.0, g_max: float = DEFAULT_G_MAX) -> np.ndarray:
    logits = np.asarray(route_logit, dtype=np.float64)
    clipped = np.clip(logits, -60.0, 60.0)
    return float(eta) * float(g_max) / (1.0 + np.exp(-clipped))


def urf_gate_components(
    residual_mean: Any,
    log_var: Any,
    *,
    q_base: Any | None = None,
    eta: float = 1.0,
    g_max: float = DEFAULT_G_MAX,
    alpha_m: float = 1.0,
    alpha_u: float = 1.0,
    tau_g: float = 0.0,
) -> dict[str, np.ndarray]:
    logits = route_logits(
        residual_mean,
        log_var,
        q_base=q_base,
        alpha_m=alpha_m,
        alpha_u=alpha_u,
        tau_g=tau_g,
    )
    ell = _broadcast_chunk(log_var, logits.shape, "log_var")
    return {
        "route_logits": logits,
        "route_gate": route_gate(logits, eta=eta, g_max=g_max),
        "predicted_std": np.sqrt(np.exp(np.clip(ell, LOG_VAR_MIN, LOG_VAR_MAX))),
    }


def apply_urf_residual(
    base_chunks: Any,
    residual_mean: Any,
    log_var: Any,
    residual_scale: Any,
    *,
    eta: float = 0.0,
    g_max: float = DEFAULT_G_MAX,
    r_max: float = DEFAULT_RESIDUAL_CAP,
    alpha_m: float = 1.0,
    alpha_u: float = 1.0,
    tau_g: float = 0.0,
    q_base: Any | None = None,
) -> np.ndarray:
    base = _chunk_matrix(base_chunks, "base_chunks")
    mu = _chunk_matrix(residual_mean, "residual_mean")
    if base.shape != mu.shape:
        raise ValueError(f"base and residual mean shapes differ: {base.shape} vs {mu.shape}")
    scale = _scale_vector(residual_scale).reshape(1, 1, ACTION_DIM)
    components = urf_gate_components(
        mu,
        log_var,
        q_base=q_base,
        eta=eta,
        g_max=g_max,
        alpha_m=alpha_m,
        alpha_u=alpha_u,
        tau_g=tau_g,
    )
    bounded_residual = np.clip(mu, -float(r_max), float(r_max))
    action = base + scale * components["route_gate"] * bounded_residual
    if not np.isfinite(action).all():
        raise ValueError("URF action contains nonfinite values")
    return action


def fit_ridge(features: Any, targets: Any, *, coefficient: float = RIDGE_COEFFICIENT) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"ridge shapes are inconsistent: {x.shape} vs {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs contain nonfinite values")
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    gram = design.T @ design
    penalty = float(coefficient) * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ y)
    return {"weights": weights[:-1], "bias": weights[-1], "coefficient": np.asarray(float(coefficient))}


def predict_ridge(model: Mapping[str, Any], features: Any) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != weights.shape[0]:
        raise ValueError(f"feature shape {x.shape} is incompatible with weights {weights.shape}")
    return x @ weights + bias


def uncertainty_monotonicity(predicted_std: Any, residual_error: Any, *, bins: int = 5) -> dict[str, Any]:
    std = np.asarray(predicted_std, dtype=np.float64).reshape(-1)
    err = np.asarray(residual_error, dtype=np.float64).reshape(-1)
    if std.shape != err.shape:
        raise ValueError(f"uncertainty and error shapes differ: {std.shape} vs {err.shape}")
    finite = np.isfinite(std) & np.isfinite(err)
    std = std[finite]
    err = err[finite]
    if std.size < int(bins) or np.max(std) == np.min(std) or np.max(err) == np.min(err):
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(_rankdata(std), _rankdata(err))[0, 1])
        if not np.isfinite(spearman):
            spearman = 0.0
    order = np.argsort(std, kind="mergesort")
    groups = [chunk for chunk in np.array_split(order, int(bins)) if len(chunk)]
    bin_means = np.asarray([float(np.mean(err[group])) for group in groups], dtype=np.float64)
    std_means = np.asarray([float(np.mean(std[group])) for group in groups], dtype=np.float64)
    binned_non_decreasing = bool(len(bin_means) >= 2 and np.all(np.diff(bin_means) >= -1e-12))
    passed = bool(spearman >= UNCERTAINTY_SPEARMAN_MIN or binned_non_decreasing)
    return {
        "uncertainty_strata_count": int(len(groups)),
        "uncertainty_strata_noncollapsed": bool(len(groups) >= 2 and np.max(std) > np.min(std)),
        "uncertainty_monotonicity_spearman": spearman,
        "uncertainty_binned_error_mean": bin_means,
        "uncertainty_binned_std_mean": std_means,
        "uncertainty_binned_monotonic": binned_non_decreasing,
        "uncertainty_monotonicity_passed": passed,
    }


def action_delta_summary(
    base_chunks: Any,
    prediction_chunks: Any,
    *,
    translation_p95_limit: float = 0.25,
    rotation_p95_limit: float = 0.25,
    gripper_p95_limit: float = 1.00,
) -> dict[str, Any]:
    base = _chunk_matrix(base_chunks, "base_chunks")
    prediction = _chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction shapes differ: {base.shape} vs {prediction.shape}")
    delta = prediction - base

    def p95(value: np.ndarray) -> float:
        return float(np.percentile(np.abs(value), 95))

    translation = p95(delta[:, :, 0:3])
    rotation = p95(delta[:, :, 3:6])
    gripper = p95(delta[:, :, 6:7])
    changed = np.abs(delta) > 1e-12
    return {
        "changed_cell_fraction": float(changed.mean()),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta_p95": translation,
        "rotation_delta_p95": rotation,
        "gripper_delta_p95": gripper,
        "action_deltas_bounded": bool(
            translation <= float(translation_p95_limit)
            and rotation <= float(rotation_p95_limit)
            and gripper <= float(gripper_p95_limit)
        ),
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
    residual_scales_noncollapsed: bool
    residual_targets_noncollapsed: bool
    route_labels_noncollapsed: bool
    route_positive_fraction: float
    uncertainty_strata_noncollapsed: bool
    task_phase_action_group_coverage_ok: bool
    base_residual_headroom_ok: bool
    hetero_beats_homoscedastic_relative: float
    hetero_beats_homoscedastic_absolute_huber: float
    hetero_beats_task_phase_relative: float
    hetero_beats_task_phase_absolute_huber: float
    uncertainty_enters_route_gate: bool
    uncertainty_monotonicity_spearman: float
    uncertainty_binned_monotonic: bool
    sureflow_proxy_headroom_relative: float
    sureflow_proxy_headroom_absolute_huber: float
    no_uncertainty_ablation_distinct: bool
    urf_beats_ablation_relative: float
    urf_beats_ablation_absolute_huber: float
    route_activation_fraction: float
    route_all_zero: bool
    route_all_one: bool
    route_globally_active: bool
    action_validity_ok: bool
    identity_max_abs_error: float
    checkpoint_reload_ok: bool
    finite_objectives_and_gradients: bool
    urf_gradient_nonzero: bool
    frozen_parameter_gradient_count: int
    weighted_gradient_norm_ratio_max: float
    action_deltas_bounded: bool
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
        return "URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < 512
        or int(inputs.minimum_validation_windows) < 128
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.residual_scales_noncollapsed
        or not inputs.residual_targets_noncollapsed
        or not inputs.route_labels_noncollapsed
        or not (ROUTE_POSITIVE_MIN <= float(inputs.route_positive_fraction) <= ROUTE_POSITIVE_MAX)
        or not inputs.uncertainty_strata_noncollapsed
        or not inputs.task_phase_action_group_coverage_ok
    ):
        return "URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if not inputs.base_residual_headroom_ok:
        return "URF_STAGE_0_NO_USABLE_HEADROOM"
    if not (
        _headroom_gate(
            inputs.hetero_beats_homoscedastic_relative,
            inputs.hetero_beats_homoscedastic_absolute_huber,
        )
        and _headroom_gate(
            inputs.hetero_beats_task_phase_relative,
            inputs.hetero_beats_task_phase_absolute_huber,
        )
    ):
        return "URF_STAGE_0_NO_USABLE_HEADROOM"
    if not inputs.uncertainty_enters_route_gate:
        return "URF_STAGE_0_DESIGN_FAILURE"
    if not (
        float(inputs.uncertainty_monotonicity_spearman) >= UNCERTAINTY_SPEARMAN_MIN
        or bool(inputs.uncertainty_binned_monotonic)
    ):
        return "URF_STAGE_0_DESIGN_FAILURE"
    if not _headroom_gate(inputs.sureflow_proxy_headroom_relative, inputs.sureflow_proxy_headroom_absolute_huber):
        return "URF_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not inputs.no_uncertainty_ablation_distinct
        or not _headroom_gate(inputs.urf_beats_ablation_relative, inputs.urf_beats_ablation_absolute_huber)
    ):
        return "URF_STAGE_0_DESIGN_FAILURE"
    if (
        bool(inputs.route_all_zero)
        or bool(inputs.route_all_one)
        or bool(inputs.route_globally_active)
        or not (ROUTE_POSITIVE_MIN <= float(inputs.route_activation_fraction) <= ROUTE_POSITIVE_MAX)
    ):
        return "URF_STAGE_0_DESIGN_FAILURE"
    if (
        not inputs.action_validity_ok
        or float(inputs.identity_max_abs_error) > 1e-6
        or not inputs.checkpoint_reload_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.urf_gradient_nonzero
        or int(inputs.frozen_parameter_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
        or not inputs.action_deltas_bounded
    ):
        return "URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    return "URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _headroom_gate(relative_improvement: float, absolute_huber_gain: float) -> bool:
    return bool(
        float(relative_improvement) >= HEADROOM_RELATIVE_GATE
        or float(absolute_huber_gain) >= HEADROOM_ABSOLUTE_HUBER_GATE
    )


def _chunk_matrix(value: Any, name: str = "chunks") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array.reshape(1, *array.shape)
    if array.ndim != 3 or array.shape[1:] != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"{name} must have shape [N,{CHUNK_SIZE},{ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _scale_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != ACTION_DIM:
        raise ValueError(f"scale must have {ACTION_DIM} elements, got {array.size}")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("scale must be finite and positive")
    return array


def _broadcast_chunk(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    try:
        broadcast = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast from {array.shape} to {shape}") from exc
    if not np.isfinite(broadcast).all():
        raise ValueError(f"{name} contains nonfinite values")
    return np.asarray(broadcast, dtype=np.float64)


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks
