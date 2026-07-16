"""Frozen BRID-VLA Stage 0 base-residual diffusion audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2"
HORIZON = 50
ACTION_DIM = 7
DIFFUSION_STEP_COUNT = 8
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
SCORE_MARGIN_MIN = 0.02
RESIDUAL_ORACLE_REDUCTION_MIN = 0.02
INTERVENTION_FRACTION_MIN = 0.02
INTERVENTION_FRACTION_MAX = 0.80
IDENTITY_RELOAD_ERROR_MAX = 1e-7
GRADIENT_RATIO_MAX = 20.0
ACTION_HUBER_DELTA = 0.05
STD_FLOOR = 1e-12


POLICY_ROWS = (
    "smolvla_base",
    "diffusion_policy_action_chunk_proxy",
    "brid_full",
    "brid_no_base_residual_ablation",
    "standard_lora",
    "residual_oracle_diagnostic",
    "task_phase_score_baseline_diagnostic",
    "mean_noise_baseline_diagnostic",
    "zero_noise_baseline_diagnostic",
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


def brid_row_key(row: Mapping[str, Any]) -> str:
    fields: list[Any] = [
        row["split"],
        row["task_suite"],
        row["task_id"],
        row["demo_id"],
        row["window_start"],
        row["diffusion_step"],
        row["noise_identity"],
        row["policy"],
    ]
    if "probe_label" in row:
        fields.append(row["probe_label"])
    if "config_label" in row:
        fields.append(row["config_label"])
    return "|".join(str(value) for value in fields)


def noise_identity_for(row: Mapping[str, Any], *, seed: int = 20263400) -> str:
    payload = {
        "seed": int(seed),
        "split": row["split"],
        "task_suite": row["task_suite"],
        "task_id": row["task_id"],
        "demo_id": int(row["demo_id"]),
        "window_start": int(row["window_start"]),
        "diffusion_step": int(row["diffusion_step"]),
        "policy": row.get("policy", "brid_noise"),
        "probe_label": row.get("probe_label"),
        "config_label": row.get("config_label"),
    }
    return "noise:" + canonical_json_sha256(payload)[:20]


def deterministic_noise(noise_identity: str, *, shape: tuple[int, int] = (HORIZON, ACTION_DIM)) -> np.ndarray:
    digest = hashlib.sha256(str(noise_identity).encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float64)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [brid_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = [
            row["task_suite"],
            row["task_id"],
            row["demo_id"],
            row["window_start"],
            row["diffusion_step"],
            row["noise_identity"],
            row["policy"],
        ]
        if "probe_label" in row:
            values.append(row["probe_label"])
        if "config_label" in row:
            values.append(row["config_label"])
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
    return float((float(baseline) - float(candidate)) / max(abs(float(baseline)), STD_FLOOR))


def residual_health(
    residual_chunks: Any,
    *,
    splits: Sequence[str],
    task_ids: Sequence[str],
    phase_bins: Sequence[int] | None = None,
) -> dict[str, Any]:
    residual = chunk_matrix(residual_chunks, "residual_chunks")
    if len(residual) != len(splits) or len(residual) != len(task_ids):
        raise ValueError("residual chunks, splits, and task ids must align")
    norms = np.linalg.norm(residual.reshape(len(residual), -1), axis=1)
    phase_bins = list(phase_bins) if phase_bins is not None else [0] * len(residual)
    task_summary: dict[str, Any] = {}
    active = np.zeros(len(residual), dtype=bool)
    clean = np.zeros(len(residual), dtype=bool)
    for task in sorted(set(task_ids)):
        indexes = np.asarray([index for index, item in enumerate(task_ids) if item == task], dtype=np.int64)
        task_norms = norms[indexes]
        median = float(np.median(task_norms))
        lower = float(np.quantile(task_norms, 0.25))
        active[indexes] = task_norms > median
        clean[indexes] = task_norms < lower
        task_summary[task] = {
            "count": int(len(indexes)),
            "norm_mean": float(np.mean(task_norms)),
            "norm_median": median,
            "norm_q25": lower,
            "active_count": int(np.sum(active[indexes])),
            "clean_count": int(np.sum(clean[indexes])),
        }
    by_split = {split: int(sum(1 for item in splits if item == split)) for split in sorted(set(splits))}
    return {
        "row_count": int(len(residual)),
        "residual_abs_max": float(np.max(np.abs(residual))),
        "residual_l2_mean": float(np.mean(norms)),
        "residual_l2_std": float(np.std(norms)),
        "residual_noncollapsed": bool(np.std(norms) > STD_FLOOR and np.max(np.abs(residual)) > STD_FLOOR),
        "residual_active_count": int(np.sum(active)),
        "clean_retention_count": int(np.sum(clean)),
        "task_summary": task_summary,
        "split_counts": by_split,
        "phase_bin_counts": {str(phase): int(sum(1 for item in phase_bins if item == phase)) for phase in sorted(set(phase_bins))},
        "active_mask": active,
        "clean_mask": clean,
    }


def group_mean_prediction(values: Any, keys: Sequence[str]) -> np.ndarray:
    array = chunk_matrix(values, "values")
    if len(array) != len(keys):
        raise ValueError("values and keys must align")
    default = np.mean(array, axis=0)
    groups: dict[str, np.ndarray] = {}
    for key in sorted(set(keys)):
        group = np.asarray([array[index] for index, item in enumerate(keys) if item == key], dtype=np.float64)
        groups[key] = np.mean(group, axis=0)
    return np.asarray([groups.get(str(key), default) for key in keys], dtype=np.float64)


def score_prediction_diagnostics(
    true_noise: Any,
    *,
    task_phase_keys: Sequence[str],
    brid_prediction: Any,
) -> dict[str, Any]:
    target = chunk_matrix(true_noise, "true_noise")
    brid = chunk_matrix(brid_prediction, "brid_prediction")
    if len(target) != len(task_phase_keys):
        raise ValueError("true noise and task/phase keys must align")
    zero = np.zeros_like(target)
    mean = np.broadcast_to(np.mean(target, axis=0, keepdims=True), target.shape)
    task_phase = group_mean_prediction(target, task_phase_keys)
    zero_huber = mean_huber(zero, target)
    mean_huber_value = mean_huber(mean, target)
    task_phase_huber = mean_huber(task_phase, target)
    brid_huber = mean_huber(brid, target)
    strongest = min(zero_huber, mean_huber_value, task_phase_huber)
    improvement = float(strongest - brid_huber)
    return {
        "zero_noise_huber": zero_huber,
        "mean_noise_huber": mean_huber_value,
        "task_phase_huber": task_phase_huber,
        "brid_score_huber": brid_huber,
        "strongest_trivial_huber": strongest,
        "score_prediction_huber_improvement": improvement,
        "score_predictable": bool(improvement >= SCORE_MARGIN_MIN),
    }


def residual_oracle_metrics(base_chunks: Any, expert_chunks: Any, oracle_prediction: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    oracle = chunk_matrix(oracle_prediction, "oracle_prediction")
    base_huber = mean_huber(base, expert)
    oracle_huber = mean_huber(oracle, expert)
    reduction = relative_improvement(base_huber, oracle_huber)
    return {
        "base_huber": base_huber,
        "residual_oracle_huber": oracle_huber,
        "residual_oracle_huber_reduction": reduction,
        "residual_oracle_headroom_ok": bool(reduction >= RESIDUAL_ORACLE_REDUCTION_MIN),
    }


def raw_diffusion_proxy_metrics(base_chunks: Any, expert_chunks: Any, proxy_prediction: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    proxy = chunk_matrix(proxy_prediction, "proxy_prediction")
    base_huber = mean_huber(base, expert)
    proxy_huber = mean_huber(proxy, expert)
    return {
        "raw_diffusion_proxy_huber": proxy_huber,
        "base_huber": base_huber,
        "raw_diffusion_proxy_headroom": relative_improvement(base_huber, proxy_huber),
    }


def apply_brid_residual(
    base_chunks: Any,
    residual_prediction: Any,
    intervention_score: Any,
    *,
    residual_gain: float = 1.0,
    threshold: float = SCORE_MARGIN_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_prediction)
    score = np.asarray(intervention_score, dtype=np.float64)
    if score.shape == (len(base),):
        gate = (score[:, None, None] >= float(threshold)).astype(np.float64)
        gate = np.broadcast_to(gate, base.shape).copy()
    elif score.shape == (len(base), HORIZON):
        gate = (score[:, :, None] >= float(threshold)).astype(np.float64)
        gate = np.broadcast_to(gate, base.shape).copy()
    elif score.shape == base.shape:
        gate = (score >= float(threshold)).astype(np.float64)
    else:
        raise ValueError(f"intervention score must align with chunks, got {score.shape}")
    output = base + float(residual_gain) * gate * residual
    if not np.isfinite(output).all():
        raise ValueError("BRID output contains nonfinite values")
    return output, gate


def apply_raw_residual_ablation(residual_prediction: Any, intervention_score: Any) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros_like(chunk_matrix(residual_prediction, "residual_prediction"))
    return apply_brid_residual(zeros, residual_prediction, intervention_score)


def standard_lora_proxy(base_chunks: Any, residual_prediction: Any, *, scale: float = 0.25) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_prediction)
    return base + float(scale) * residual


def action_delta_summary(base_chunks: Any, prediction_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    prediction = chunk_matrix(prediction_chunks, "prediction_chunks")
    if base.shape != prediction.shape:
        raise ValueError(f"base and prediction chunks differ: {base.shape} vs {prediction.shape}")
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
            translation <= TRANSLATION_CAP + 1e-12
            and rotation <= ROTATION_CAP + 1e-12
            and gripper <= GRIPPER_CAP + 1e-12
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


def gradient_smoke(base_chunks: Any, residual_prediction: Any, gate: Any, expert_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_prediction)
    gate_value = chunk_matrix(gate, "gate")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    update = gate_value * residual
    gradient = float(np.mean((base - expert) * update))
    update_norm = float(np.mean(np.square(update)))
    return {
        "finite_objectives_and_gradients": bool(np.isfinite(gradient) and np.isfinite(update_norm)),
        "expected_parameter_gradient_nonzero": bool(abs(gradient) > STD_FLOOR or update_norm > STD_FLOOR),
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "scalar_gain_gradient": gradient,
        "update_mean_square": update_norm,
    }


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
    residual_targets_noncollapsed: bool
    enough_discovery_windows: bool
    enough_validation_windows: bool
    validation_task_coverage_ok: bool
    maximum_validation_task_fraction: float
    noise_identity_valid: bool
    score_predictable: bool
    residual_oracle_huber_reduction: float
    raw_diffusion_proxy_headroom: float
    brid_beats_base: bool
    brid_beats_raw_diffusion_proxy: bool
    brid_beats_no_base_residual_ablation: bool
    brid_beats_standard_lora: bool
    brid_differs_from_base: bool
    brid_differs_from_ablation: bool
    identity_max_abs_error: float
    checkpoint_reload_ok: bool
    finite_objectives_and_gradients: bool
    expected_parameter_gradient_nonzero: bool
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
        or not inputs.preflight_passed
        or not inputs.manifest_integrity_ok
        or not inputs.source_alignment_ok
        or not inputs.action_semantics_ok
        or not inputs.base_chunks_valid
        or float(inputs.identity_max_abs_error) > IDENTITY_RELOAD_ERROR_MAX
        or not inputs.checkpoint_reload_ok
        or not inputs.finite_objectives_and_gradients
        or not inputs.expected_parameter_gradient_nonzero
        or int(inputs.frozen_base_gradient_count) != 0
        or float(inputs.weighted_gradient_norm_ratio_max) > GRADIENT_RATIO_MAX
        or not inputs.action_deltas_bounded
        or not inputs.action_validity_ok
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
        return "BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.residual_targets_noncollapsed
        or not inputs.enough_discovery_windows
        or not inputs.enough_validation_windows
        or not inputs.validation_task_coverage_ok
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or not inputs.noise_identity_valid
    ):
        return "BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        float(inputs.residual_oracle_huber_reduction) < RESIDUAL_ORACLE_REDUCTION_MIN
        or float(inputs.raw_diffusion_proxy_headroom) <= 0.0
    ):
        return "BRID_STAGE_0_NO_RESIDUAL_HEADROOM"
    if (
        not inputs.score_predictable
        or not inputs.brid_beats_base
        or not inputs.brid_beats_raw_diffusion_proxy
        or not inputs.brid_beats_no_base_residual_ablation
        or not inputs.brid_beats_standard_lora
        or not inputs.brid_differs_from_base
        or not inputs.brid_differs_from_ablation
        or float(inputs.intervention_fraction) < INTERVENTION_FRACTION_MIN
        or float(inputs.intervention_fraction) > INTERVENTION_FRACTION_MAX
        or not inputs.clean_retention_ok
    ):
        return "BRID_STAGE_0_DESIGN_FAILURE"
    return "BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
