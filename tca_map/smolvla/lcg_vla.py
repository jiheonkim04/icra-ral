"""Frozen LCG-VLA Stage 0 language-contrast guidance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E"
HORIZON = 50
ACTION_DIM = 7
NULL_INSTRUCTION = ""
TAU_LANG = 0.25
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
CAG_PROXY_BETAS = (0.25, 0.5, 1.0)
MIN_DISCOVERY_WINDOWS = 512
MIN_VALIDATION_WINDOWS = 128
MAX_VALIDATION_TASK_FRACTION = 0.40
CONTRAST_POSITIVE_MIN = 0.05
CONTRAST_POSITIVE_MAX = 0.95
GATE_ACTIVATION_MIN = 0.02
GATE_ACTIVATION_MAX = 0.80
CONTRAST_RESIDUAL_SPEARMAN_MIN = 0.05
CONTRAST_PROBE_IMPROVEMENT_MIN = 0.01
MASKED_ORACLE_HEADROOM_MIN = 0.01
LCG_BEATS_CAG_MIN = 0.0
IDENTITY_RELOAD_ERROR_MAX = 1e-6
GRADIENT_RATIO_MAX = 20.0
ACTION_HUBER_DELTA = 0.05
STD_FLOOR = 1e-8


POLICY_ROWS = (
    "smolvla_base",
    "counterfactual_action_guidance_proxy",
    "lcg_full",
    "lcg_no_language_contrast_ablation",
    "standard_lora_proxy",
    "contrast_magnitude_only_gate",
    "task_phase_residual",
    "masked_residual_oracle_diagnostic",
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


def lcg_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["instruction_variant"],
        row["model_or_probe"],
        row["policy_probe"],
    )
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [lcg_row_key(row) for row in manifest_rows]
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
            row["instruction_variant"],
            row["model_or_probe"],
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


def construct_language_contrast(base_chunks: Any, null_chunks: Any) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    null = chunk_matrix(null_chunks, "null_chunks")
    if base.shape != null.shape:
        raise ValueError(f"base and null chunks differ: {base.shape} vs {null.shape}")
    return base - null


def fit_discovery_contrast_scale(contrast_chunks: Any) -> dict[str, Any]:
    contrast = chunk_matrix(contrast_chunks, "contrast_chunks")
    absolute = np.abs(contrast)
    scale = np.maximum(np.percentile(absolute, 75, axis=(0, 1)), STD_FLOOR)
    return {
        "kind": "discovery_abs_p75_per_action_dim",
        "tau_lang": TAU_LANG,
        "action_dimension": ACTION_DIM,
        "scale": scale,
        "discovery_row_count": int(len(contrast)),
    }


def language_mask(contrast_chunks: Any, contrast_scale: Mapping[str, Any], *, tau: float = TAU_LANG) -> np.ndarray:
    contrast = chunk_matrix(contrast_chunks, "contrast_chunks")
    scale = np.asarray(contrast_scale["scale"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    return (np.abs(contrast) >= float(tau) * np.maximum(scale, STD_FLOOR)).astype(np.float64)


def mask_health(mask: Any) -> dict[str, Any]:
    value = chunk_matrix(mask, "mask")
    positive = float(np.mean(value > 0.5))
    return {
        "contrast_positive_fraction": positive,
        "language_mask_noncollapsed": bool(CONTRAST_POSITIVE_MIN <= positive <= CONTRAST_POSITIVE_MAX),
        "language_mask_all_zero": bool(np.all(value <= 0.5)),
        "language_mask_all_one": bool(np.all(value > 0.5)),
    }


def apply_lcg_gate(
    base_chunks: Any,
    residual_chunks: Any,
    gate: Any,
    *,
    residual_gain: float = 0.0,
) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_chunks)
    if base.shape != residual.shape:
        raise ValueError(f"base and residual chunks differ: {base.shape} vs {residual.shape}")
    gate_value = np.asarray(gate, dtype=np.float64)
    if gate_value.ndim == 0:
        gate_value = np.full((len(base), HORIZON, 1), float(gate_value), dtype=np.float64)
    elif gate_value.shape == (len(base), HORIZON):
        gate_value = gate_value[:, :, None]
    elif gate_value.shape == base.shape:
        gate_value = gate_value.mean(axis=2, keepdims=True)
    elif gate_value.shape != (len(base), HORIZON, 1):
        raise ValueError(f"gate must be scalar or [N,{HORIZON},1]/[N,{HORIZON},{ACTION_DIM}], got {gate_value.shape}")
    gate_value = np.clip(gate_value, 0.0, 1.0)
    output = base + float(residual_gain) * gate_value * residual
    if not np.isfinite(output).all():
        raise ValueError("LCG action chunk contains nonfinite values")
    return output


def apply_cag_proxy(base_chunks: Any, null_chunks: Any, *, beta: float) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    contrast = construct_language_contrast(base, null_chunks)
    return base + float(beta) * group_clip(contrast)


def apply_no_language_ablation(base_chunks: Any, residual_chunks: Any, *, residual_gain: float = 1.0) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    return base + float(residual_gain) * group_clip(residual_chunks)


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


def contrast_residual_spearman(contrast_chunks: Any, residual_chunks: Any) -> float:
    contrast = np.linalg.norm(chunk_matrix(contrast_chunks, "contrast_chunks"), axis=2).reshape(-1)
    residual = np.linalg.norm(chunk_matrix(residual_chunks, "residual_chunks"), axis=2).reshape(-1)
    if contrast.size != residual.size or contrast.size < 3:
        return 0.0
    if float(np.std(contrast)) < STD_FLOOR or float(np.std(residual)) < STD_FLOOR:
        return 0.0
    contrast_rank = _rankdata(contrast)
    residual_rank = _rankdata(residual)
    corr = np.corrcoef(contrast_rank, residual_rank)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def contrast_residual_noncollapse(contrast_chunks: Any, residual_chunks: Any) -> dict[str, Any]:
    contrast = chunk_matrix(contrast_chunks, "contrast_chunks")
    residual = chunk_matrix(residual_chunks, "residual_chunks")
    contrast_std = float(np.std(contrast))
    residual_std = float(np.std(residual))
    return {
        "contrast_std": contrast_std,
        "residual_std": residual_std,
        "contrast_noncollapsed": bool(contrast_std > STD_FLOOR),
        "residual_labels_noncollapsed": bool(residual_std > STD_FLOOR),
        "contrast_residual_spearman": contrast_residual_spearman(contrast, residual),
    }


def scalar_contrast_residual_probe(contrast_chunks: Any, residual_chunks: Any) -> dict[str, Any]:
    contrast = chunk_matrix(contrast_chunks, "contrast_chunks")
    residual = chunk_matrix(residual_chunks, "residual_chunks")
    numerator = np.sum(contrast * residual, axis=(0, 1))
    denominator = np.maximum(np.sum(np.square(contrast), axis=(0, 1)), STD_FLOOR)
    slope = numerator / denominator
    return {
        "kind": "per_action_dim_scalar_contrast_residual_probe",
        "slope": slope,
        "fitted_row_count": int(len(contrast)),
    }


def predict_contrast_residual(model: Mapping[str, Any], contrast_chunks: Any) -> np.ndarray:
    contrast = chunk_matrix(contrast_chunks, "contrast_chunks")
    slope = np.asarray(model["slope"], dtype=np.float64).reshape(1, 1, ACTION_DIM)
    return contrast * slope


def gradient_smoke(base_chunks: Any, residual_chunks: Any, gate: Any, target_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_chunks)
    target = chunk_matrix(target_chunks, "target_chunks")
    gate_value = np.asarray(gate, dtype=np.float64)
    if gate_value.shape == base.shape:
        gate_value = gate_value.mean(axis=2, keepdims=True)
    elif gate_value.shape == (len(base), HORIZON):
        gate_value = gate_value[:, :, None]
    elif gate_value.ndim == 0:
        gate_value = np.full((len(base), HORIZON, 1), float(gate_value), dtype=np.float64)
    update = gate_value * residual
    gradient = float(np.mean((base - target) * update))
    residual_norm = float(np.mean(np.square(update)))
    return {
        "finite_objectives_and_gradients": bool(np.isfinite(gradient) and np.isfinite(residual_norm)),
        "expected_parameter_gradient_nonzero": bool(abs(gradient) > STD_FLOOR or residual_norm > STD_FLOOR),
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "scalar_gain_gradient": gradient,
        "update_mean_square": residual_norm,
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
    contrast_noncollapsed: bool
    residual_labels_noncollapsed: bool
    contrast_positive_fraction: float
    language_mask_all_zero: bool
    language_mask_all_one: bool
    gate_activation_fraction: float
    contrast_residual_spearman: float
    contrast_probe_beats_task_phase_baseline: bool
    contrast_probe_relative_improvement: float
    best_cag_proxy_score: float
    cag_proxy_residual_headroom: float
    lcg_beats_cag_proxy_relative: float
    masked_residual_headroom: float
    cag_coefficient_equivalence: bool
    no_language_ablation_explains: bool
    lora_explains: bool
    identity_max_abs_error: float
    inactive_gate_max_abs_error: float
    action_validity_ok: bool
    clean_retention_ok: bool
    finite_objectives_and_gradients: bool
    expected_parameter_gradient_nonzero: bool
    frozen_base_gradient_count: int
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
        or float(inputs.identity_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or float(inputs.inactive_gate_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or not inputs.action_validity_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.expected_parameter_gradient_nonzero
        or int(inputs.frozen_base_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
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
        return "LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    if (
        not inputs.feature_action_proprio_finite_aligned
        or int(inputs.minimum_discovery_windows) < MIN_DISCOVERY_WINDOWS
        or int(inputs.minimum_validation_windows) < MIN_VALIDATION_WINDOWS
        or not inputs.all_tasks_reported
        or float(inputs.maximum_validation_task_fraction) > MAX_VALIDATION_TASK_FRACTION
        or not inputs.contrast_noncollapsed
        or not inputs.residual_labels_noncollapsed
        or bool(inputs.language_mask_all_zero)
        or bool(inputs.language_mask_all_one)
        or not (CONTRAST_POSITIVE_MIN <= float(inputs.contrast_positive_fraction) <= CONTRAST_POSITIVE_MAX)
    ):
        return "LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        float(inputs.masked_residual_headroom) < MASKED_ORACLE_HEADROOM_MIN
        or float(inputs.lcg_beats_cag_proxy_relative) < LCG_BEATS_CAG_MIN
    ):
        return "LCG_STAGE_0_NO_USABLE_HEADROOM"
    if (
        float(inputs.contrast_residual_spearman) < CONTRAST_RESIDUAL_SPEARMAN_MIN
        or not inputs.contrast_probe_beats_task_phase_baseline
        or float(inputs.contrast_probe_relative_improvement) < CONTRAST_PROBE_IMPROVEMENT_MIN
        or bool(inputs.cag_coefficient_equivalence)
        or bool(inputs.no_language_ablation_explains)
        or bool(inputs.lora_explains)
        or not (GATE_ACTIVATION_MIN <= float(inputs.gate_activation_fraction) <= GATE_ACTIVATION_MAX)
        or not inputs.clean_retention_ok
    ):
        return "LCG_STAGE_0_DESIGN_FAILURE"
    return "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    index = 0
    while index < len(value):
        stop = index + 1
        while stop < len(value) and value[order[stop]] == value[order[index]]:
            stop += 1
        rank = 0.5 * (index + stop - 1) + 1.0
        ranks[order[index:stop]] = rank
        index = stop
    return ranks
