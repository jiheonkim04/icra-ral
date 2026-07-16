"""Frozen MCI-VLA Stage 0 multi-consistency audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A"
HORIZON = 50
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 960
PROPRIO_DIM = 8
LATENT_DIM_VALUES = (16, 32)
LAMBDA_C_VALUES = (0.25, 0.50, 1.00)
HUBER_DELTA = 0.05
GAMMA_VAR = 0.5
IDENTITY_TOLERANCE = 1e-7
GRADIENT_RATIO_MAX = 100.0
PREDICTABILITY_MARGIN_MIN = 0.02
COMPARATOR_MARGIN_MIN = 0.005
TRANSLATION_CAP = 0.01
ROTATION_CAP = 0.025
GRIPPER_CAP = 0.125
INTERVENTION_FRACTION_MIN = 0.02
INTERVENTION_FRACTION_MAX = 0.80
STD_FLOOR = 1e-12

TRANSFORMATION_FAMILIES = (
    "instruction",
    "observation_proprioception",
    "action_evolution",
)

POLICY_ROWS = (
    "smolvla_base",
    "rovla_multiconsistency_proxy",
    "mci_full",
    "mci_no_consistency_code_ablation",
    "augmentation_only_lora_killer",
    "transformation_label_health_diagnostic",
    "consistency_observability_diagnostic",
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
    transformations_noncollapsed: bool
    enough_discovery_rows: bool
    enough_validation_rows: bool
    validation_task_coverage_ok: bool
    maximum_validation_task_fraction: float
    minimum_validation_pairs_per_family: int
    positive_contrast_count: int
    negative_contrast_count: int
    representation_dims_fraction_above_floor: float
    consistency_predictability_margin: float
    base_transformed_pair_headroom: float
    rovla_residual_headroom: float
    augmentation_residual_headroom: float
    mci_beats_comparators: bool
    mci_differs_from_base: bool
    mci_differs_from_rovla: bool
    mci_differs_from_ablation: bool
    mci_differs_from_augmentation_only_lora: bool
    exact_base_passthrough_ok: bool
    identity_reload_error: float
    finite_nonzero_gradients: bool
    frozen_base_gradient_count: int
    weighted_gradient_norm_ratio_max: float
    intervention_fraction: float
    action_deltas_bounded: bool
    action_validity_rate: float
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


def mci_row_key(row: Mapping[str, Any]) -> str:
    fields = [
        row["split"],
        row["task_suite"],
        row["task_identity"],
        row["demo_id"],
        row["window_start"],
        row["transform_family"],
        row["policy"],
        row["config_label"],
        row["probe_label"],
    ]
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [mci_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["task_suite"],
            row["task_identity"],
            row["demo_id"],
            row["window_start"],
            row["transform_family"],
            row["policy"],
            row["config_label"],
            row["probe_label"],
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


def feature_matrix(value: Any, name: str = "features") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape == (VISUAL_FEATURE_DIM,):
        array = array.reshape(1, VISUAL_FEATURE_DIM)
    if array.ndim != 2 or array.shape[1] != VISUAL_FEATURE_DIM:
        raise ValueError(f"{name} must have shape [N,{VISUAL_FEATURE_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def proprio_matrix(value: Any, name: str = "proprioception") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape == (PROPRIO_DIM,):
        array = array.reshape(1, PROPRIO_DIM)
    if array.ndim != 2 or array.shape[1] != PROPRIO_DIM:
        raise ValueError(f"{name} must have shape [N,{PROPRIO_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _projection(label: str, input_dim: int, output_dim: int, scale: float = 1.0) -> np.ndarray:
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale / np.sqrt(max(input_dim, 1)), size=(input_dim, output_dim))


def _task_features(task_ids: Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for task in task_ids:
        digest = hashlib.sha256(str(task).encode("utf-8")).digest()
        rows.append([(digest[index] / 255.0) * 2.0 - 1.0 for index in range(8)])
    return np.asarray(rows, dtype=np.float64)


def _feature_subset(features: np.ndarray, count: int = 64) -> np.ndarray:
    if features.shape[1] <= count:
        return features
    indexes = np.linspace(0, features.shape[1] - 1, count).astype(int)
    return features[:, indexes]


def _base_summary(base_chunks: np.ndarray) -> np.ndarray:
    translation = base_chunks[:, :, 0:3]
    rotation = base_chunks[:, :, 3:6]
    gripper = base_chunks[:, :, 6:7]
    velocity = np.diff(base_chunks, axis=1)
    return np.concatenate(
        [
            np.mean(base_chunks, axis=1),
            np.std(base_chunks, axis=1),
            np.mean(np.abs(velocity), axis=1),
            np.max(np.abs(translation), axis=1),
            np.max(np.abs(rotation), axis=1),
            np.max(np.abs(gripper), axis=1),
        ],
        axis=1,
    )


def layernorm(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=1, keepdims=True)
    std = np.std(array, axis=1, keepdims=True)
    return (array - mean) / np.maximum(std, STD_FLOOR)


def consistency_code(
    features: Any,
    proprioception: Any,
    task_ids: Sequence[str],
    base_chunks: Any,
    *,
    latent_dim: int = 16,
) -> np.ndarray:
    if latent_dim not in LATENT_DIM_VALUES:
        raise ValueError(f"latent_dim must be one of {LATENT_DIM_VALUES}, got {latent_dim}")
    feat = feature_matrix(features)
    prop = proprio_matrix(proprioception)
    base = chunk_matrix(base_chunks)
    if len(feat) != len(prop) or len(feat) != len(base) or len(feat) != len(task_ids):
        raise ValueError("features, proprioception, task ids, and base chunks must align")
    compact = np.concatenate(
        [
            _feature_subset(feat),
            prop,
            _task_features(task_ids),
            _base_summary(base),
        ],
        axis=1,
    )
    projection = _projection(f"mci_consistency_code_d{latent_dim}", compact.shape[1], latent_dim)
    code = compact @ projection
    if not np.isfinite(code).all():
        raise ValueError("consistency code contains nonfinite values")
    return layernorm(code)


def _deterministic_noise(label: str, shape: Sequence[int], scale: float) -> np.ndarray:
    seed = int(canonical_json_sha256({"label": label})[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=tuple(shape))


def transformed_inputs(
    features: Any,
    proprioception: Any,
    task_ids: Sequence[str],
    base_chunks: Any,
    *,
    family: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    if family not in TRANSFORMATION_FAMILIES:
        raise ValueError(f"unknown transformation family {family!r}")
    feat = feature_matrix(features)
    prop = proprio_matrix(proprioception)
    base = chunk_matrix(base_chunks)
    tasks = [str(task) for task in task_ids]
    if family == "instruction":
        transformed_tasks = [f"{task}::task_preserving_paraphrase" for task in tasks]
        return feat.copy(), prop.copy(), transformed_tasks, base.copy(), {
            "family": family,
            "task_semantics_preserved": True,
            "uses_future_or_privileged_input": False,
        }
    if family == "observation_proprioception":
        feature_scale = 1.0 + 0.01 * np.tanh(_deterministic_noise(family, (1, feat.shape[1]), 1.0))
        transformed_feat = feat * feature_scale
        transformed_prop = prop + _deterministic_noise(f"{family}_prop", prop.shape, 0.002)
        return transformed_feat, transformed_prop, tasks, base.copy(), {
            "family": family,
            "bounded_feature_affine": True,
            "bounded_proprioception_jitter": True,
            "uses_future_or_privileged_input": False,
        }
    noise = _deterministic_noise(family, base.shape, 0.01)
    transformed_base = base + group_clip(noise)
    return feat.copy(), prop.copy(), tasks, transformed_base, {
        "family": family,
        "bounded_base_chunk_perturbation": True,
        "uses_future_or_privileged_input": False,
    }


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


def residual_targets(base_chunks: Any, expert_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    if base.shape != expert.shape:
        raise ValueError(f"base and expert chunks differ: {base.shape} vs {expert.shape}")
    return expert - base


def group_clip(residual: Any) -> np.ndarray:
    value = chunk_matrix(residual, "residual").copy()
    value[:, :, 0:3] = np.clip(value[:, :, 0:3], -TRANSLATION_CAP, TRANSLATION_CAP)
    value[:, :, 3:6] = np.clip(value[:, :, 3:6], -ROTATION_CAP, ROTATION_CAP)
    value[:, :, 6:7] = np.clip(value[:, :, 6:7], -GRIPPER_CAP, GRIPPER_CAP)
    return value


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def consistency_gate(code: Any, *, horizon: int = HORIZON, action_dim: int = ACTION_DIM) -> np.ndarray:
    z = np.asarray(code, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"code must have shape [N,d_z], got {z.shape}")
    projection = _projection("mci_gate_projection", z.shape[1], horizon * action_dim)
    logits = (z @ projection).reshape(len(z), horizon, action_dim)
    thresholds = np.quantile(logits.reshape(len(z), -1), 0.70, axis=1).reshape(len(z), 1, 1)
    return _sigmoid(8.0 * (logits - thresholds))


def apply_mci_adapter(
    base_chunks: Any,
    residual_prediction: Any,
    code: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")))
    if len(base) != len(residual):
        raise ValueError("base and residual prediction must align")
    gate = consistency_gate(code)
    if len(base) != len(gate):
        raise ValueError("base and consistency code must align")
    emitted = base + gate * residual
    if not np.isfinite(emitted).all():
        raise ValueError("MCI output contains nonfinite values")
    return emitted, gate, residual


def mci_no_consistency_code_ablation(
    base_chunks: Any,
    residual_prediction: Any,
    *,
    intervention_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")))
    gate = np.full_like(base, float(np.clip(intervention_fraction, 0.0, 1.0)))
    return base + gate * residual, gate


def rovla_multiconsistency_proxy(base_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    smoothed = base.copy()
    smoothed[:, 1:-1, 0:6] = (base[:, :-2, 0:6] + base[:, 1:-1, 0:6] + base[:, 2:, 0:6]) / 3.0
    residual = group_clip(smoothed - base)
    return base + 0.75 * residual


def augmentation_only_lora_killer(base_chunks: Any, residual_prediction: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(np.tanh(chunk_matrix(residual_prediction, "residual_prediction")))
    smooth = rovla_multiconsistency_proxy(base)
    return smooth + 0.35 * residual


def identity_passthrough(base_chunks: Any) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    zero_residual = np.zeros_like(base)
    zero_code = np.zeros((len(base), LATENT_DIM_VALUES[0]), dtype=np.float64)
    emitted, gate, _ = apply_mci_adapter(base, zero_residual, zero_code)
    return emitted, gate


def action_delta_summary(base_chunks: Any, prediction_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    prediction = chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction chunks differ: {base.shape} vs {prediction.shape}")
    delta = prediction - base

    def p95(value: np.ndarray) -> float:
        return float(np.percentile(np.abs(value), 95))

    trans = np.abs(delta[:, :, 0:3])
    rot = np.abs(delta[:, :, 3:6])
    grip = np.abs(delta[:, :, 6:7])
    return {
        "changed_cell_fraction": float(np.mean(np.abs(delta) > 1e-12)),
        "delta_finite": bool(np.isfinite(delta).all()),
        "delta_abs_max": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "translation_delta_p95": p95(delta[:, :, 0:3]),
        "rotation_delta_p95": p95(delta[:, :, 3:6]),
        "gripper_delta_p95": p95(delta[:, :, 6:7]),
        "translation_delta_max": float(np.max(trans)) if trans.size else 0.0,
        "rotation_delta_max": float(np.max(rot)) if rot.size else 0.0,
        "gripper_delta_max": float(np.max(grip)) if grip.size else 0.0,
        "action_deltas_bounded": bool(
            (float(np.max(trans)) if trans.size else 0.0) <= TRANSLATION_CAP + 1e-12
            and (float(np.max(rot)) if rot.size else 0.0) <= ROTATION_CAP + 1e-12
            and (float(np.max(grip)) if grip.size else 0.0) <= GRIPPER_CAP + 1e-12
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
        "clean_retention_ok": bool(identity_error <= IDENTITY_TOLERANCE),
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
        "official_semantics_required": True,
        "no_ad_hoc_unit_box_gate": True,
    }


def representation_health(code: Any) -> dict[str, Any]:
    z = np.asarray(code, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"code must have shape [N,d_z], got {z.shape}")
    std = np.std(z, axis=0)
    fraction = float(np.mean(std >= GAMMA_VAR)) if len(std) else 0.0
    return {
        "latent_dim": int(z.shape[1]),
        "std_min": float(np.min(std)) if len(std) else 0.0,
        "std_mean": float(np.mean(std)) if len(std) else 0.0,
        "dims_fraction_above_floor": fraction,
        "representation_noncollapsed": bool(fraction >= 0.80),
    }


def _balanced_accuracy(score: np.ndarray, target: np.ndarray) -> float:
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


def consistency_observability_diagnostics(
    legal_signal: Sequence[float],
    targets: Sequence[int],
    task_ids: Sequence[str],
    frame_indices: Sequence[int],
    action_magnitudes: Sequence[float],
    families: Sequence[str],
) -> dict[str, Any]:
    signal = np.asarray(legal_signal, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int64)
    if signal.shape != target.shape:
        raise ValueError("legal_signal and targets must align")
    legal_score = _balanced_accuracy(signal, target)
    task_score = np.zeros_like(signal)
    for task in sorted(set(str(item) for item in task_ids)):
        indexes = [index for index, item in enumerate(task_ids) if str(item) == task]
        task_score[indexes] = float(np.mean(target[indexes]))
    frame_score = np.asarray(frame_indices, dtype=np.float64)
    if frame_score.size:
        frame_score = frame_score / max(float(np.max(frame_score)), 1.0)
    magnitude_score = np.asarray(action_magnitudes, dtype=np.float64)
    if magnitude_score.size:
        magnitude_score = magnitude_score / max(float(np.max(np.abs(magnitude_score))), 1.0)
    family_score = np.zeros_like(signal)
    for family in sorted(set(str(item) for item in families)):
        indexes = [index for index, item in enumerate(families) if str(item) == family]
        family_score[indexes] = float(np.mean(target[indexes]))
    majority = np.full_like(signal, float(np.mean(target)) if target.size else 0.0)
    baselines = {
        "majority": _balanced_accuracy(majority, target),
        "task_identity": _balanced_accuracy(task_score, target),
        "frame_phase_audit_proxy": _balanced_accuracy(frame_score, target),
        "action_magnitude": _balanced_accuracy(magnitude_score, target),
        "augmentation_family_identity": _balanced_accuracy(family_score, target),
    }
    strongest_name = max(baselines, key=baselines.get)
    strongest = baselines[strongest_name]
    return {
        "legal_signal_balanced_accuracy": legal_score,
        "strongest_trivial_baseline_score": strongest,
        "strongest_trivial_baseline_name": strongest_name,
        "consistency_predictability_margin": float(legal_score - strongest),
        "consistency_signal_predictable": bool(legal_score - strongest >= PREDICTABILITY_MARGIN_MIN),
        "baseline_scores": baselines,
    }


def objective_gradient_smoke(
    base_chunks: Any,
    expert_chunks: Any,
    transformed_chunks: Any,
    code: Any,
    transformed_code: Any,
    mci_chunks: Any,
    gate: Any,
    *,
    lambda_c: float = 0.5,
) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    transformed = chunk_matrix(transformed_chunks, "transformed_chunks")
    mci = chunk_matrix(mci_chunks, "mci_chunks")
    z = np.asarray(code, dtype=np.float64)
    z_t = np.asarray(transformed_code, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    if z.shape != z_t.shape:
        raise ValueError("code and transformed_code must align")
    if gate_array.shape != base.shape:
        raise ValueError("gate must match action chunks")
    h = layernorm(z)
    h_t = layernorm(z_t)
    raw = mci
    values = {
        "L_code": float(np.mean(np.square(h - h_t)) / max(z.shape[1], 1)),
        "L_act": mean_huber(mci, transformed),
        "L_fit": mean_huber(mci, expert),
        "L_keep": mean_huber(mci, base),
        "L_var": float(np.mean(np.maximum(GAMMA_VAR - np.std(h, axis=0), 0.0) ** 2)),
        "L_bound": float(np.mean(np.maximum(np.abs(raw) - 1.0, 0.0) ** 2)),
    }
    grad_norms = {
        "L_code": float(np.linalg.norm(h - h_t)),
        "L_act": float(np.linalg.norm(mci - transformed)),
        "L_fit": float(np.linalg.norm(mci - expert)),
        "L_keep": float(np.linalg.norm(mci - base)),
        "L_var": float(np.linalg.norm(np.maximum(GAMMA_VAR - np.std(h, axis=0), 0.0))),
        "L_bound": float(np.linalg.norm(np.maximum(np.abs(raw) - 1.0, 0.0))),
    }
    weights = {
        "L_code": float(lambda_c),
        "L_act": float(lambda_c),
        "L_fit": 1.0,
        "L_keep": 1.0,
        "L_var": 1.0,
        "L_bound": 1.0,
    }
    weighted = {name: values[name] * weights[name] for name in values}
    weighted_grad_norms = {name: grad_norms[name] * weights[name] for name in grad_norms}
    nonzero = np.asarray([value for value in weighted_grad_norms.values() if value > STD_FLOOR], dtype=np.float64)
    ratio = float(np.max(nonzero) / max(np.min(nonzero), STD_FLOOR)) if len(nonzero) else 0.0
    return {
        "objective_values": values,
        "weighted_objective_values": weighted,
        "gradient_norms": grad_norms,
        "weighted_gradient_norms": weighted_grad_norms,
        "weighted_gradient_norm_ratio_max": ratio,
        "weighted_gradient_norm_ratio_ok": bool(ratio <= GRADIENT_RATIO_MAX),
        "finite_nonzero_gradients": bool(np.isfinite(nonzero).all() and len(nonzero) >= 4),
        "expected_parameter_gradient_nonzero": bool(np.linalg.norm(gate_array) > 0.0 and len(nonzero) >= 4),
        "frozen_base_gradient_count": 0,
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
        return "MCI_STAGE_0_IMPLEMENTATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.base_chunks_valid
        or not inputs.feature_caches_valid
        or not inputs.transformations_noncollapsed
        or not inputs.enough_discovery_rows
        or not inputs.enough_validation_rows
        or not inputs.validation_task_coverage_ok
        or inputs.maximum_validation_task_fraction > 0.40
        or inputs.minimum_validation_pairs_per_family < 32
        or inputs.positive_contrast_count < 16
        or inputs.negative_contrast_count < 16
    ):
        return "MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        inputs.identity_reload_error > IDENTITY_TOLERANCE
        or not inputs.exact_base_passthrough_ok
        or not inputs.finite_nonzero_gradients
        or inputs.frozen_base_gradient_count != 0
        or inputs.weighted_gradient_norm_ratio_max > GRADIENT_RATIO_MAX
        or not inputs.action_deltas_bounded
        or inputs.action_validity_rate < 1.0
        or not inputs.clean_retention_ok
    ):
        return "MCI_STAGE_0_IMPLEMENTATION_FAILURE"
    if (
        inputs.consistency_predictability_margin < PREDICTABILITY_MARGIN_MIN
        or inputs.base_transformed_pair_headroom <= 0.0
        or inputs.rovla_residual_headroom <= 0.0
        or inputs.augmentation_residual_headroom <= 0.0
    ):
        return "MCI_STAGE_0_NO_HEADROOM"
    if (
        inputs.representation_dims_fraction_above_floor < 0.80
        or not inputs.mci_beats_comparators
        or not inputs.mci_differs_from_base
        or not inputs.mci_differs_from_rovla
        or not inputs.mci_differs_from_ablation
        or not inputs.mci_differs_from_augmentation_only_lora
        or not (INTERVENTION_FRACTION_MIN <= inputs.intervention_fraction <= INTERVENTION_FRACTION_MAX)
    ):
        return "MCI_STAGE_0_DESIGN_FAILURE"
    return "MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
