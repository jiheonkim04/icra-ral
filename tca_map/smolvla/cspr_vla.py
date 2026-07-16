"""Frozen CSPR-VLA Stage 0 critical-step refinement audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7"
HORIZON = 50
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 960
PROPRIO_DIM = 8
TRANSLATION_CAP_SMALL = 0.005
ROTATION_CAP_SMALL = 0.0125
GRIPPER_CAP_SMALL = 0.0625
TRANSLATION_CAP = 0.01
ROTATION_CAP = 0.025
GRIPPER_CAP = 0.125
TRANSLATION_CAP_LARGE = 0.02
ROTATION_CAP_LARGE = 0.05
GRIPPER_CAP_LARGE = 0.25
DEFAULT_TAU_QUANTILE = 0.95
SOFT_GATE_TEMPERATURE = 0.05
IDENTITY_TOLERANCE = 1e-7
GRADIENT_RATIO_MAX = 100.0
PREDICTABILITY_MARGIN_MIN = 0.02
COMPARATOR_MARGIN_MIN = 0.005
HUBER_DELTA = 0.05
STD_FLOOR = 1e-12


CRITICALITY_WEIGHTS = {
    "error": 1.0,
    "curvature": 0.5,
    "acceleration": 0.5,
    "gripper_event": 1.0,
}


POLICY_ROWS = (
    "smolvla_base",
    "dysl_action_importance_proxy",
    "cspr_full",
    "cspr_uniform_refinement_ablation",
    "critical_step_threshold_simple_killer",
    "criticality_label_health_diagnostic",
    "criticality_predictability_diagnostic",
    "identity_passthrough_reload_diagnostic",
    "objective_gradient_scale_diagnostic",
)


@dataclass(frozen=True)
class Stage0DecisionInputs:
    proposal_hash_ok: bool
    serializer_preflight_ok: bool
    official_prior_asset_check_persisted: bool
    preflight_passed: bool
    manifest_integrity_ok: bool
    source_alignment_ok: bool
    action_semantics_ok: bool
    base_chunks_valid: bool
    feature_caches_valid: bool
    labels_noncollapsed: bool
    criticality_score_variance_ok: bool
    enough_discovery_rows: bool
    enough_validation_rows: bool
    validation_task_coverage_ok: bool
    maximum_validation_task_fraction: float
    validation_positive_count: int
    validation_negative_count: int
    validation_positive_fraction: float
    largest_positive_task_fraction: float
    criticality_predictability_margin: float
    base_residual_headroom: float
    dysl_residual_headroom: float
    simple_killer_residual_headroom: float
    cspr_beats_comparators: bool
    cspr_differs_from_base: bool
    cspr_differs_from_ablation: bool
    simple_killer_explains_gain: bool
    identity_reload_error: float
    finite_nonzero_gradients: bool
    frozen_base_gradient_count: int
    weighted_gradient_norm_ratio_max: float
    intervention_fraction: float
    action_deltas_bounded: bool
    action_validity_ok: bool
    clean_retention_ok: bool
    reward_read_count: int
    success_read_count: int
    done_read_count: int
    confirmatory_records_read: int
    simulator_load_count: int
    closed_loop_experiment_happened: bool
    training_happened: bool
    validation_search_happened: bool
    exception_count: int


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


def cspr_row_key(row: Mapping[str, Any]) -> str:
    fields: list[Any] = [
        row["split"],
        row["task_suite"],
        row["task_identity"],
        row["demo_id"],
        row["frame_index"],
        row["source_edge_sha256"],
        row["model_or_probe"],
        row["config_label"],
    ]
    if "probe_label" in row:
        fields.append(row["probe_label"])
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [cspr_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = [
            row["task_suite"],
            row["task_identity"],
            row["demo_id"],
            row["frame_index"],
            row["source_edge_sha256"],
            row["model_or_probe"],
            row["config_label"],
        ]
        if "probe_label" in row:
            values.append(row["probe_label"])
        return tuple(values)

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


def feature_matrix(value: Any, name: str = "features") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape == (VISUAL_FEATURE_DIM,):
        array = array.reshape(1, VISUAL_FEATURE_DIM)
    if array.ndim != 2 or array.shape[1] != VISUAL_FEATURE_DIM:
        raise ValueError(f"{name} must have shape [N,{VISUAL_FEATURE_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def residual_targets(base_chunks: Any, expert_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunks differ: {base.shape} vs {expert.shape}")
    return expert - base


def cap_tuple(cap_group: str = "mid") -> tuple[float, float, float]:
    if cap_group == "small":
        return TRANSLATION_CAP_SMALL, ROTATION_CAP_SMALL, GRIPPER_CAP_SMALL
    if cap_group == "mid":
        return TRANSLATION_CAP, ROTATION_CAP, GRIPPER_CAP
    if cap_group == "large":
        return TRANSLATION_CAP_LARGE, ROTATION_CAP_LARGE, GRIPPER_CAP_LARGE
    raise ValueError(f"unknown CSPR cap group {cap_group!r}")


def group_clip(residual: Any, *, cap_group: str = "mid") -> np.ndarray:
    trans, rot, grip = cap_tuple(cap_group)
    value = chunk_matrix(residual, "residual").copy()
    value[:, :, 0:3] = np.clip(value[:, :, 0:3], -trans, trans)
    value[:, :, 3:6] = np.clip(value[:, :, 3:6], -rot, rot)
    value[:, :, 6:7] = np.clip(value[:, :, 6:7], -grip, grip)
    return value


def huber_values(error: Any, *, delta: float = HUBER_DELTA) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("huber input contains nonfinite values")
    threshold = float(delta)
    if threshold <= 0.0:
        raise ValueError("delta must be positive")
    absolute = np.abs(value)
    return np.where(absolute <= threshold, 0.5 * np.square(value), threshold * (absolute - 0.5 * threshold))


def mean_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    return float(np.mean(huber_values(pred - tgt, delta=delta)))


def cell_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> np.ndarray:
    pred = chunk_matrix(prediction, "prediction")
    tgt = chunk_matrix(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    return huber_values(pred - tgt, delta=delta)


def _pad_diff(values: np.ndarray, order: int) -> np.ndarray:
    diff = np.diff(values, n=order, axis=1)
    if diff.shape[1] == 0:
        return np.zeros_like(values)
    pad_front = order // 2
    pad_back = HORIZON - diff.shape[1] - pad_front
    return np.pad(diff, ((0, 0), (pad_front, pad_back), (0, 0)), mode="edge")


def _gripper_event_cells(expert: np.ndarray, *, threshold: float = 0.05) -> np.ndarray:
    events = np.zeros_like(expert, dtype=np.float64)
    gripper = expert[:, :, 6]
    change = np.abs(np.diff(gripper, axis=1)) >= threshold
    for batch_index in range(len(expert)):
        event_steps = np.where(change[batch_index])[0]
        protected: set[int] = set()
        for step in event_steps:
            protected.update({max(0, int(step) - 1), int(step), min(HORIZON - 1, int(step) + 1)})
        for step in protected:
            events[batch_index, step, 6] = 1.0
    return events


def _robust_stats(component: np.ndarray, discovery_mask: np.ndarray) -> tuple[float, float, bool]:
    source = component[discovery_mask]
    if source.size == 0:
        source = component.reshape(-1)
    median = float(np.median(source))
    q25 = float(np.quantile(source, 0.25))
    q75 = float(np.quantile(source, 0.75))
    iqr = q75 - q25
    return median, max(iqr, 1.0), bool(np.isfinite(source).all() and np.var(source) > STD_FLOOR)


def criticality_components(base_chunks: Any, expert_chunks: Any) -> dict[str, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunks differ: {base.shape} vs {expert.shape}")
    return {
        "error": np.abs(base - expert),
        "curvature": np.abs(_pad_diff(expert, 2)),
        "acceleration": np.abs(_pad_diff(expert, 1)),
        "gripper_event": _gripper_event_cells(expert),
    }


def construct_criticality_labels(
    base_chunks: Any,
    expert_chunks: Any,
    discovery_rows: Any,
    *,
    tau_quantile: float = DEFAULT_TAU_QUANTILE,
) -> dict[str, Any]:
    components = criticality_components(base_chunks, expert_chunks)
    base = chunk_matrix(base_chunks, "base_chunks")
    discovery = np.asarray(discovery_rows, dtype=bool)
    if discovery.shape != (len(base),):
        raise ValueError("discovery_rows must have shape [N]")
    discovery_cells = np.broadcast_to(discovery[:, None, None], base.shape)
    normalized: dict[str, np.ndarray] = {}
    normalizers: dict[str, Any] = {}
    variance_ok = True
    for name, component in components.items():
        median, scale, ok = _robust_stats(component, discovery_cells)
        normalized[name] = np.maximum((component - median) / scale, 0.0)
        normalizers[name] = {"median": median, "scale": scale, "variance_ok": ok}
        if name != "gripper_event":
            variance_ok = variance_ok and ok
    score = (
        CRITICALITY_WEIGHTS["error"] * normalized["error"]
        + CRITICALITY_WEIGHTS["curvature"] * normalized["curvature"]
        + CRITICALITY_WEIGHTS["acceleration"] * normalized["acceleration"]
        + CRITICALITY_WEIGHTS["gripper_event"] * components["gripper_event"]
    )
    discovery_scores = score[discovery_cells]
    if discovery_scores.size == 0:
        discovery_scores = score.reshape(-1)
    q_tau = float(np.quantile(discovery_scores, tau_quantile))
    labels = (score >= q_tau).astype(np.int64)
    return {
        "score": score,
        "labels": labels,
        "q_tau": q_tau,
        "tau_quantile": float(tau_quantile),
        "normalizers": normalizers,
        "criticality_score_variance_ok": bool(variance_ok and np.var(discovery_scores) > STD_FLOOR),
    }


def label_health(labels: Any, task_ids: Sequence[str], frame_indices: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"labels must have shape [N,{HORIZON},{ACTION_DIM}], got {values.shape}")
    positive = int(np.sum(values == 1))
    negative = int(np.sum(values == 0))
    count = int(values.size)
    positive_fraction = float(positive / max(count, 1))
    positive_by_task: dict[str, int] = {}
    count_by_task: dict[str, int] = {}
    quartile_positive = {str(index): 0 for index in range(4)}
    quartile_count = {str(index): 0 for index in range(4)}
    for row_index, task in enumerate(task_ids):
        task = str(task)
        row_values = values[row_index]
        positive_by_task[task] = positive_by_task.get(task, 0) + int(np.sum(row_values == 1))
        count_by_task[task] = count_by_task.get(task, 0) + int(row_values.size)
        frame_offset = int(frame_indices[row_index]) % HORIZON
        for step in range(HORIZON):
            quartile = str(min(3, int(((frame_offset + step) / HORIZON) * 4)))
            quartile_positive[quartile] += int(np.sum(row_values[step] == 1))
            quartile_count[quartile] += ACTION_DIM
    largest_positive_task_fraction = 0.0 if positive == 0 else max(positive_by_task.values()) / positive
    return {
        "label_count": count,
        "positive_count": positive,
        "negative_count": negative,
        "positive_fraction": positive_fraction,
        "positive_by_task": positive_by_task,
        "count_by_task": count_by_task,
        "quartile_positive_fraction": {
            key: float(quartile_positive[key] / max(quartile_count[key], 1)) for key in sorted(quartile_count)
        },
        "largest_positive_task_fraction": float(largest_positive_task_fraction),
        "labels_noncollapsed": bool(positive >= 1 and negative >= 1 and 0.0 < positive_fraction < 1.0),
    }


def base_criticality_proxy(base_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    velocity = np.abs(_pad_diff(base, 1))
    acceleration = np.abs(_pad_diff(base, 2))
    gripper = _gripper_event_cells(base, threshold=0.02)
    score = velocity + 0.5 * acceleration + gripper
    maximum = np.max(score)
    if maximum > 0.0:
        score = score / maximum
    return score


def _balanced_accuracy_from_score(score: np.ndarray, target: np.ndarray) -> float:
    flat_score = np.asarray(score, dtype=np.float64).reshape(-1)
    flat_target = np.asarray(target, dtype=np.int64).reshape(-1)
    if flat_score.size == 0 or len(np.unique(flat_target)) < 2:
        return 0.5
    threshold = float(np.quantile(flat_score, 1.0 - np.mean(flat_target)))
    pred = flat_score >= threshold
    positives = flat_target == 1
    negatives = ~positives
    tpr = float(np.mean(pred[positives])) if np.any(positives) else 0.0
    tnr = float(np.mean(~pred[negatives])) if np.any(negatives) else 0.0
    return 0.5 * (tpr + tnr)


def criticality_predictability_diagnostics(
    labels: Any,
    legal_score: Any,
    task_ids: Sequence[str],
    frame_indices: Sequence[int],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(legal_score, dtype=np.float64)
    if target.shape != score.shape:
        raise ValueError(f"labels and legal score differ: {target.shape} vs {score.shape}")
    legal = _balanced_accuracy_from_score(score, target)
    task_score = np.zeros_like(score)
    for task in sorted(set(str(item) for item in task_ids)):
        indexes = [index for index, item in enumerate(task_ids) if str(item) == task]
        rate = float(np.mean(target[indexes]))
        task_score[indexes] = rate
    frame_score = np.zeros_like(score)
    for row_index, frame in enumerate(frame_indices):
        phase = (int(frame) % HORIZON) / max(HORIZON - 1, 1)
        frame_score[row_index, :, :] = phase
    majority = np.full_like(score, float(np.mean(target)), dtype=np.float64)
    baselines = {
        "majority": _balanced_accuracy_from_score(majority, target),
        "task_mean": _balanced_accuracy_from_score(task_score, target),
        "frame_index_audit_proxy": _balanced_accuracy_from_score(frame_score, target),
    }
    strongest = max(baselines.values())
    return {
        "legal_score_balanced_accuracy": legal,
        "strongest_trivial_baseline_score": strongest,
        "strongest_trivial_baseline_name": max(baselines, key=baselines.get),
        "criticality_predictability_margin": float(legal - strongest),
        "criticality_predictable": bool(legal - strongest >= PREDICTABILITY_MARGIN_MIN),
        "baseline_scores": baselines,
    }


def apply_cspr_refinement(
    base_chunks: Any,
    residual_prediction: Any,
    criticality_score: Any,
    *,
    tau: float,
    cap_group: str = "mid",
) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")), cap_group=cap_group)
    score = np.asarray(criticality_score, dtype=np.float64)
    if score.shape != base.shape:
        raise ValueError(f"criticality score must match action chunks, got {score.shape}")
    gate = (score >= float(tau)).astype(np.float64)
    output = base + gate * residual
    if not np.isfinite(output).all():
        raise ValueError("CSPR output contains nonfinite values")
    return output, gate


def uniform_refinement_ablation(
    base_chunks: Any,
    residual_prediction: Any,
    *,
    intervention_fraction: float,
    cap_group: str = "mid",
) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")), cap_group=cap_group)
    fraction = float(np.clip(intervention_fraction, 0.0, 1.0))
    gate = np.full_like(base, fraction, dtype=np.float64)
    return base + gate * residual, gate


def critical_step_threshold_simple_killer(
    base_chunks: Any,
    residual_prediction: Any,
    *,
    cap_group: str = "mid",
    quantile: float = DEFAULT_TAU_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    proxy = base_criticality_proxy(base)
    threshold = float(np.quantile(proxy, quantile))
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")), cap_group=cap_group)
    gate = (proxy >= threshold).astype(np.float64)
    return base + gate * residual, gate


def action_delta_summary(base_chunks: Any, prediction_chunks: Any, *, cap_group: str = "mid") -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    prediction = chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction chunks differ: {base.shape} vs {prediction.shape}")
    trans_cap, rot_cap, grip_cap = cap_tuple(cap_group)
    delta = prediction - base

    def p95(value: np.ndarray) -> float:
        return float(np.percentile(np.abs(value), 95))

    translation = p95(delta[:, :, 0:3])
    rotation = p95(delta[:, :, 3:6])
    gripper = p95(delta[:, :, 6:7])
    return {
        "changed_cell_fraction": float(np.mean(np.abs(delta) > 1e-12)),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "translation_delta_p95": translation,
        "rotation_delta_p95": rotation,
        "gripper_delta_p95": gripper,
        "action_deltas_bounded": bool(
            translation <= trans_cap + 1e-12 and rotation <= rot_cap + 1e-12 and gripper <= grip_cap + 1e-12
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
    inactive_delta = inactive - base
    return {
        "identity_max_abs_error": identity_error,
        "inactive_gate_max_abs_error": float(np.max(np.abs(inactive_delta))),
        "clean_translation_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 0:3]), 95)),
        "clean_rotation_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 3:6]), 95)),
        "clean_gripper_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 6:7]), 95)),
        "clean_retention_ok": bool(
            identity_error <= IDENTITY_TOLERANCE
            and float(np.percentile(np.abs(inactive_delta[:, :, 0:3]), 95)) <= 0.10 * TRANSLATION_CAP + 1e-12
            and float(np.percentile(np.abs(inactive_delta[:, :, 3:6]), 95)) <= 0.10 * ROTATION_CAP + 1e-12
            and float(np.percentile(np.abs(inactive_delta[:, :, 6:7]), 95)) <= 0.10 * GRIPPER_CAP + 1e-12
        ),
    }


def action_validity_summary(chunks: Any) -> dict[str, Any]:
    action = chunk_matrix(chunks, "chunks")
    return {
        "action_shape": list(action.shape),
        "finite_fraction": float(np.mean(np.isfinite(action))),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
        "postprocessed_action_validity": True,
        "action_validity_ok": bool(np.isfinite(action).all()),
    }


def gradient_smoke(
    base_chunks: Any,
    residual_prediction: Any,
    gate: Any,
    expert_chunks: Any,
    labels: Any,
) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = chunk_matrix(residual_prediction, "residual_prediction")
    gate_array = np.asarray(gate, dtype=np.float64)
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    label_array = np.asarray(labels, dtype=np.float64)
    if gate_array.shape != base.shape or label_array.shape != base.shape:
        raise ValueError("gate and labels must match action chunks")
    residual_grad_norm = float(np.linalg.norm((base + gate_array * residual) - expert))
    # Stage 0 smoke checks that a noncollapsed BCE-style criticality target
    # would give the predictor a finite nonzero gradient before real training.
    criticality_grad_norm = float(np.linalg.norm(label_array - 0.5))
    keep_grad_norm = float(np.linalg.norm((1.0 - label_array) * gate_array * residual))
    bound_grad_norm = float(np.linalg.norm(np.maximum(np.abs(base + gate_array * residual) - 1.0, 0.0)))
    norms = np.asarray([criticality_grad_norm, residual_grad_norm, keep_grad_norm, bound_grad_norm], dtype=np.float64)
    nonzero = norms[norms > STD_FLOOR]
    ratio = float(np.max(nonzero) / max(np.min(nonzero), STD_FLOOR)) if len(nonzero) else 0.0
    return {
        "L_crit": criticality_grad_norm,
        "L_fit": residual_grad_norm,
        "L_keep": keep_grad_norm,
        "L_bound": bound_grad_norm,
        "finite_nonzero_gradients": bool(np.isfinite(norms).all() and criticality_grad_norm > 0.0 and residual_grad_norm > 0.0),
        "expected_parameter_gradient_nonzero": bool(criticality_grad_norm > 0.0 and residual_grad_norm > 0.0),
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": ratio,
        "weighted_gradient_norm_ratio_ok": bool(ratio <= GRADIENT_RATIO_MAX),
    }


def classify_stage0(inputs: Stage0DecisionInputs) -> str:
    privileged_or_runtime_defect = (
        inputs.reward_read_count
        or inputs.success_read_count
        or inputs.done_read_count
        or inputs.confirmatory_records_read
        or inputs.simulator_load_count
        or inputs.closed_loop_experiment_happened
        or inputs.training_happened
        or inputs.validation_search_happened
        or inputs.exception_count
    )
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or not inputs.official_prior_asset_check_persisted
        or not inputs.preflight_passed
        or not inputs.action_semantics_ok
        or privileged_or_runtime_defect
    ):
        return "CSPR_STAGE_0_IMPLEMENTATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.base_chunks_valid
        or not inputs.feature_caches_valid
        or not inputs.labels_noncollapsed
        or not inputs.criticality_score_variance_ok
        or not inputs.enough_discovery_rows
        or not inputs.enough_validation_rows
        or not inputs.validation_task_coverage_ok
        or inputs.maximum_validation_task_fraction > 0.40
        or inputs.validation_positive_count < 8
        or inputs.validation_negative_count < 8
        or not (0.02 <= inputs.validation_positive_fraction <= 0.80)
        or inputs.largest_positive_task_fraction > 0.75
    ):
        return "CSPR_STAGE_0_DATA_FAILURE"
    if (
        inputs.criticality_predictability_margin < PREDICTABILITY_MARGIN_MIN
        or inputs.base_residual_headroom <= 0.0
        or inputs.dysl_residual_headroom <= 0.0
        or inputs.simple_killer_residual_headroom <= 0.0
    ):
        return "CSPR_STAGE_0_NO_USABLE_HEADROOM"
    if (
        inputs.identity_reload_error > IDENTITY_TOLERANCE
        or not inputs.finite_nonzero_gradients
        or inputs.frozen_base_gradient_count != 0
        or inputs.weighted_gradient_norm_ratio_max > GRADIENT_RATIO_MAX
        or not inputs.action_deltas_bounded
        or not inputs.action_validity_ok
        or not inputs.clean_retention_ok
    ):
        return "CSPR_STAGE_0_IMPLEMENTATION_FAILURE"
    if (
        not inputs.cspr_beats_comparators
        or not inputs.cspr_differs_from_base
        or not inputs.cspr_differs_from_ablation
        or inputs.simple_killer_explains_gain
        or not (0.02 <= inputs.intervention_fraction <= 0.80)
    ):
        return "CSPR_STAGE_0_DESIGN_FAILURE"
    return "CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
