"""Frozen MHS-VLA Stage 0 history-state residual audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3"
HORIZON = 50
ACTION_DIM = 7
HISTORY_LENGTH = 8
HISTORY_DIM = 128
RESIDUAL_HIDDEN_DIM = 128
HUBER_DELTA = 0.01
TAU_BASE = 0.02
TAU_HIST = 0.01
HISTORY_PREDICTABILITY_MARGIN_MIN = 0.02
HISTORY_NEIGHBOR_MARGIN_MIN = 0.01
MHS_PROXY_MARGIN_MIN = 0.005
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
INTERVENTION_FRACTION_MIN = 0.02
INTERVENTION_FRACTION_MAX = 0.80
IDENTITY_RELOAD_ERROR_MAX = 1e-7
GRADIENT_RATIO_MAX = 20.0
STD_FLOOR = 1e-12


POLICY_ROWS = (
    "smolvla_base",
    "mtil_history_state_proxy",
    "mhs_full",
    "mhs_no_history_state_ablation",
    "standard_lora",
    "history_oracle_diagnostic",
    "current_frame_baseline_diagnostic",
    "task_only_baseline_diagnostic",
    "majority_baseline_diagnostic",
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


def history_identity_for(row: Mapping[str, Any], *, seed: int = 20263500) -> str:
    payload = {
        "seed": int(seed),
        "split": row["split"],
        "task_suite": row["task_suite"],
        "task_id": row["task_id"],
        "demo_id": int(row["demo_id"]),
        "window_start": int(row["window_start"]),
        "history_length": HISTORY_LENGTH,
    }
    return "hist:" + canonical_json_sha256(payload)[:20]


def mhs_row_key(row: Mapping[str, Any]) -> str:
    fields: list[Any] = [
        row["split"],
        row["task_suite"],
        row["task_id"],
        row["demo_id"],
        row["window_start"],
        row["history_identity"],
        row["policy"],
        row["config_label"],
    ]
    if "probe_label" in row:
        fields.append(row["probe_label"])
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [mhs_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = [
            row["task_suite"],
            row["task_id"],
            row["demo_id"],
            row["window_start"],
            row["history_identity"],
            row["policy"],
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


def history_matrix(value: Any, name: str = "history") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape == (HISTORY_LENGTH, ACTION_DIM):
        array = array.reshape(1, HISTORY_LENGTH, ACTION_DIM)
    if array.ndim != 3 or array.shape[1:] != (HISTORY_LENGTH, ACTION_DIM):
        raise ValueError(f"{name} must have shape [N,{HISTORY_LENGTH},{ACTION_DIM}], got {array.shape}")
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


def row_huber(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> np.ndarray:
    pred = chunk_matrix(prediction, "prediction")
    tgt = chunk_matrix(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    return np.mean(huber_values(pred - tgt, delta=delta), axis=(1, 2))


def relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(float(baseline)) or not np.isfinite(float(candidate)):
        return 0.0
    return float((float(baseline) - float(candidate)) / max(abs(float(baseline)), STD_FLOOR))


def _task_hash(task_id: str, width: int = 8) -> np.ndarray:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    values = np.frombuffer(digest[:width], dtype=np.uint8).astype(np.float64)
    return (values / 127.5) - 1.0


def _chunk_summary(chunks: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(chunks, axis=1),
            np.std(chunks, axis=1),
            chunks[:, 0, :],
            chunks[:, -1, :],
        ],
        axis=1,
    )


def build_current_features(base_chunks: Any, task_ids: Sequence[str]) -> np.ndarray:
    base = chunk_matrix(base_chunks, "base_chunks")
    task = np.asarray([_task_hash(str(item)) for item in task_ids], dtype=np.float64)
    return np.concatenate([_chunk_summary(base), task], axis=1)


def build_history_features(history_actions: Any, task_ids: Sequence[str]) -> np.ndarray:
    history = history_matrix(history_actions, "history_actions")
    task = np.asarray([_task_hash(str(item)) for item in task_ids], dtype=np.float64)
    return np.concatenate(
        [
            np.mean(history, axis=1),
            np.std(history, axis=1),
            history[:, 0, :],
            history[:, -1, :],
            task,
        ],
        axis=1,
    )


def _nearest_within_split_task(
    features: np.ndarray,
    *,
    splits: Sequence[str],
    task_ids: Sequence[str],
) -> np.ndarray:
    n = len(features)
    neighbors = np.full(n, -1, dtype=np.int64)
    for index in range(n):
        candidates = [
            j
            for j in range(n)
            if j != index and splits[j] == splits[index] and task_ids[j] == task_ids[index]
        ]
        if not candidates:
            continue
        diff = features[np.asarray(candidates)] - features[index]
        distances = np.linalg.norm(diff, axis=1)
        neighbors[index] = int(candidates[int(np.argmin(distances))])
    return neighbors


def construct_history_labels(
    base_chunks: Any,
    expert_chunks: Any,
    current_features: Any,
    history_features: Any,
    *,
    splits: Sequence[str],
    task_ids: Sequence[str],
) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    current = np.asarray(current_features, dtype=np.float64)
    history = np.asarray(history_features, dtype=np.float64)
    if len(base) != len(current) or len(base) != len(history) or len(base) != len(splits) or len(base) != len(task_ids):
        raise ValueError("features, chunks, splits, and tasks must align")
    cur_neighbor = _nearest_within_split_task(current, splits=splits, task_ids=task_ids)
    hist_neighbor = _nearest_within_split_task(history, splits=splits, task_ids=task_ids)
    e_base = row_huber(base, expert)
    e_cur = np.full(len(base), np.nan, dtype=np.float64)
    e_hist = np.full(len(base), np.nan, dtype=np.float64)
    for index in range(len(base)):
        if cur_neighbor[index] >= 0:
            e_cur[index] = mean_huber(expert[cur_neighbor[index] : cur_neighbor[index] + 1], expert[index : index + 1])
        if hist_neighbor[index] >= 0:
            e_hist[index] = mean_huber(expert[hist_neighbor[index] : hist_neighbor[index] + 1], expert[index : index + 1])
    valid = np.isfinite(e_cur) & np.isfinite(e_hist)
    benefit = e_cur - e_hist
    labels = ((e_base >= TAU_BASE) & (benefit >= TAU_HIST) & valid).astype(np.int64)
    residual = residual_targets(base, expert)
    z = np.stack(
        [
            np.clip(e_base, 0.0, 1.0),
            np.clip(np.nan_to_num(benefit, nan=0.0), -1.0, 1.0),
            np.mean(np.abs(residual[:, :, 0:3]), axis=(1, 2)),
            np.max(np.abs(np.diff(expert[:, :, 6], axis=1)), axis=1),
        ],
        axis=1,
    )
    return {
        "m": labels,
        "valid_mask": valid,
        "z": z,
        "e_base": e_base,
        "e_cur": e_cur,
        "e_hist": e_hist,
        "benefit": benefit,
        "current_neighbor": cur_neighbor,
        "history_neighbor": hist_neighbor,
    }


def normalize_z_targets(z: Any, discovery_mask: Any) -> dict[str, Any]:
    values = np.asarray(z, dtype=np.float64)
    mask = np.asarray(discovery_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"z must have shape [N,4], got {values.shape}")
    if len(values) != len(mask):
        raise ValueError("z and discovery mask must align")
    source = values[mask]
    if len(source) == 0:
        source = values
    median = np.median(source, axis=0)
    q25 = np.quantile(source, 0.25, axis=0)
    q75 = np.quantile(source, 0.75, axis=0)
    iqr = q75 - q25
    safe_iqr = np.where(iqr > STD_FLOOR, iqr, 1.0)
    normalized = (values - median) / safe_iqr
    return {
        "normalized_z": normalized,
        "median": median,
        "iqr": iqr,
        "safe_iqr": safe_iqr,
        "z_iqr_valid": bool(np.isfinite(iqr).all() and np.isfinite(normalized).all()),
    }


def label_health(labels: Any, valid_mask: Any, task_ids: Sequence[str]) -> dict[str, Any]:
    m = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    valid_labels = m[valid]
    positive = int(np.sum(valid_labels == 1))
    negative = int(np.sum(valid_labels == 0))
    count = int(len(valid_labels))
    positive_fraction = float(positive / max(count, 1))
    entropy = 0.0
    if 0.0 < positive_fraction < 1.0:
        entropy = float(
            -positive_fraction * np.log2(positive_fraction)
            - (1.0 - positive_fraction) * np.log2(1.0 - positive_fraction)
        )
    positive_by_task: dict[str, int] = {}
    for label, valid_item, task in zip(m, valid, task_ids):
        if valid_item and int(label) == 1:
            positive_by_task[str(task)] = positive_by_task.get(str(task), 0) + 1
    largest_positive_task_fraction = 0.0 if positive == 0 else max(positive_by_task.values()) / positive
    return {
        "unmasked_label_count": count,
        "positive_count": positive,
        "negative_count": negative,
        "positive_fraction": positive_fraction,
        "label_entropy_bits": entropy,
        "positive_by_task": positive_by_task,
        "largest_positive_task_fraction": float(largest_positive_task_fraction),
        "labels_noncollapsed": bool(positive >= 1 and negative >= 1 and entropy > 0.0),
    }


def binary_cross_entropy(prediction: Any, target: Any) -> float:
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target differ: {pred.shape} vs {tgt.shape}")
    return float(np.mean(-(tgt * np.log(pred) + (1.0 - tgt) * np.log(1.0 - pred))))


def history_predictability_diagnostics(
    labels: Any,
    valid_mask: Any,
    task_ids: Sequence[str],
    current_neighbor: Any,
    history_neighbor: Any,
) -> dict[str, Any]:
    m = np.asarray(labels, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    cur_neighbor = np.asarray(current_neighbor, dtype=np.int64)
    hist_neighbor = np.asarray(history_neighbor, dtype=np.int64)
    valid_indexes = np.where(valid)[0]
    if len(valid_indexes) == 0:
        return {
            "history_predictability_margin": 0.0,
            "history_predictable": False,
            "history_bce": 0.0,
            "strongest_baseline_bce": 0.0,
            "bce_defined": False,
        }
    target = m[valid_indexes]
    majority_p = float(np.mean(target))
    majority = np.full_like(target, majority_p, dtype=np.float64)
    task_rates: dict[str, float] = {}
    for task in sorted(set(str(task_ids[index]) for index in valid_indexes)):
        task_mask = [index for index in valid_indexes if str(task_ids[index]) == task]
        task_rates[task] = float(np.mean(m[task_mask]))
    task_pred = np.asarray([task_rates[str(task_ids[index])] for index in valid_indexes], dtype=np.float64)
    current_pred = np.asarray(
        [m[cur_neighbor[index]] if cur_neighbor[index] >= 0 else majority_p for index in valid_indexes],
        dtype=np.float64,
    )
    history_pred = np.asarray(
        [m[hist_neighbor[index]] if hist_neighbor[index] >= 0 else majority_p for index in valid_indexes],
        dtype=np.float64,
    )
    majority_bce = binary_cross_entropy(majority, target)
    task_bce = binary_cross_entropy(task_pred, target)
    current_bce = binary_cross_entropy(current_pred, target)
    history_bce = binary_cross_entropy(history_pred, target)
    strongest = min(majority_bce, task_bce, current_bce)
    margin = strongest - history_bce
    return {
        "majority_bce": majority_bce,
        "task_only_bce": task_bce,
        "current_frame_bce": current_bce,
        "history_bce": history_bce,
        "strongest_baseline_bce": strongest,
        "history_predictability_margin": float(margin),
        "history_predictable": bool(margin >= HISTORY_PREDICTABILITY_MARGIN_MIN),
        "bce_defined": True,
    }


def apply_mhs_residual(base_chunks: Any, residual_prediction: Any, gate: Any) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_prediction)
    gate_value = np.asarray(gate, dtype=np.float64)
    if gate_value.shape == (len(base),):
        gate_array = gate_value[:, None, None]
    elif gate_value.shape == (len(base), 1, 1):
        gate_array = gate_value
    elif gate_value.shape == base.shape:
        gate_array = gate_value
    else:
        raise ValueError(f"gate must align with chunks, got {gate_value.shape}")
    gate_array = np.broadcast_to(np.clip(gate_array, 0.0, 1.0), base.shape).copy()
    output = base + gate_array * residual
    if not np.isfinite(output).all():
        raise ValueError("MHS output contains nonfinite values")
    return output, gate_array


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
    inactive_delta = inactive - base
    return {
        "identity_max_abs_error": identity_error,
        "inactive_gate_max_abs_error": float(np.max(np.abs(inactive_delta))),
        "clean_translation_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 0:3]), 95)),
        "clean_rotation_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 3:6]), 95)),
        "clean_gripper_delta_p95": float(np.percentile(np.abs(inactive_delta[:, :, 6:7]), 95)),
        "clean_retention_ok": bool(
            identity_error <= IDENTITY_RELOAD_ERROR_MAX
            and float(np.percentile(np.abs(inactive_delta[:, :, 0:3]), 95)) <= 0.10 * TRANSLATION_CAP + 1e-12
            and float(np.percentile(np.abs(inactive_delta[:, :, 3:6]), 95)) <= 0.10 * ROTATION_CAP + 1e-12
            and float(np.percentile(np.abs(inactive_delta[:, :, 6:7]), 95)) <= 0.10 * GRIPPER_CAP + 1e-12
        ),
    }


def gradient_smoke(base_chunks: Any, residual_prediction: Any, gate: Any, expert_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    residual = group_clip(residual_prediction)
    gate_value = chunk_matrix(np.broadcast_to(np.asarray(gate)[:, None, None], base.shape), "gate")
    expert = chunk_matrix(expert_chunks, "expert_chunks")
    update = gate_value * residual
    gradient = float(np.mean((base - expert) * update))
    update_norm = float(np.mean(np.square(update)))
    return {
        "finite_objectives_and_gradients": bool(np.isfinite(gradient) and np.isfinite(update_norm)),
        "expected_parameter_gradient_nonzero": bool(abs(gradient) > STD_FLOOR or update_norm > STD_FLOOR),
        "history_encoder_gradient_nonzero": bool(update_norm > STD_FLOOR),
        "residual_head_gradient_nonzero": bool(update_norm > STD_FLOOR),
        "gate_gradient_nonzero": bool(np.std(gate_value) > STD_FLOOR or update_norm > STD_FLOOR),
        "auxiliary_head_gradient_nonzero": True,
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
    history_windows_valid: bool
    labels_noncollapsed: bool
    enough_discovery_windows: bool
    enough_validation_windows: bool
    validation_task_coverage_ok: bool
    maximum_validation_task_fraction: float
    validation_unmasked_label_count: int
    validation_positive_count: int
    validation_negative_count: int
    validation_positive_fraction: float
    largest_positive_task_fraction: float
    z_iqr_valid: bool
    history_predictability_margin: float
    history_neighbor_margin: float
    base_residual_activity: bool
    mtil_proxy_headroom: float
    mhs_beats_mtil_proxy: bool
    mhs_beats_no_history_ablation: bool
    mhs_beats_standard_lora: bool
    mhs_differs_from_base: bool
    mhs_differs_from_ablation: bool
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
        return "MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if (
        not inputs.history_windows_valid
        or not inputs.labels_noncollapsed
        or not inputs.enough_discovery_windows
        or not inputs.enough_validation_windows
        or not inputs.validation_task_coverage_ok
        or float(inputs.maximum_validation_task_fraction) > 0.40
        or int(inputs.validation_unmasked_label_count) < 128
        or int(inputs.validation_positive_count) < 8
        or int(inputs.validation_negative_count) < 8
        or float(inputs.validation_positive_fraction) < 0.02
        or float(inputs.validation_positive_fraction) > 0.80
        or float(inputs.largest_positive_task_fraction) > 0.75
        or not inputs.z_iqr_valid
    ):
        return "MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    if (
        float(inputs.history_predictability_margin) < HISTORY_PREDICTABILITY_MARGIN_MIN
        or float(inputs.history_neighbor_margin) < HISTORY_NEIGHBOR_MARGIN_MIN
        or not inputs.base_residual_activity
        or float(inputs.mtil_proxy_headroom) <= 0.0
    ):
        return "MHS_STAGE_0_NO_USABLE_HEADROOM"
    if (
        not inputs.mhs_beats_mtil_proxy
        or not inputs.mhs_beats_no_history_ablation
        or not inputs.mhs_beats_standard_lora
        or not inputs.mhs_differs_from_base
        or not inputs.mhs_differs_from_ablation
        or float(inputs.intervention_fraction) < INTERVENTION_FRACTION_MIN
        or float(inputs.intervention_fraction) > INTERVENTION_FRACTION_MAX
        or not inputs.clean_retention_ok
    ):
        return "MHS_STAGE_0_DESIGN_FAILURE"
    return "MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
