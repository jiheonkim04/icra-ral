"""Frozen DCCG-VLA Stage 0 coherence-guidance audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E"
HORIZON = 50
ACTION_DIM = 7
FEATURE_COUNT = 10
TAIL_TAU = 0.02
TAU_PAUSE = 0.005
EPSILON_PAUSE = 0.0025
TAU_GRIP = 0.05
EPSILON_SCALE = 1e-6
HUBER_DELTA = 1.0
TRANSLATION_CAP = 0.02
ROTATION_CAP = 0.05
GRIPPER_CAP = 0.25
GATE_ACTIVATION_MIN = 0.02
GATE_ACTIVATION_MAX = 0.80
ACG_HEADROOM_MIN = 0.01
DISTINCTION_MARGIN_MIN = 1e-6
GRADIENT_EPSILON = 1e-4


POLICY_ROWS = (
    "smolvla_base",
    "acg_official_proxy",
    "dccg_full",
    "dccg_no_demo_calibration_ablation",
    "action_smoothing_simple_killer",
    "expert_demo_coherence_diagnostic",
    "synthetic_jitter_diagnostic",
    "synthetic_pause_diagnostic",
    "synthetic_gripper_corruption_diagnostic",
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
    features_noncollapsed: bool
    bins_noncollapsed: bool
    enough_discovery_windows: bool
    enough_validation_windows: bool
    validation_task_coverage_ok: bool
    maximum_validation_task_fraction: float
    gate_activation_fraction: float
    base_acg_headroom: float
    dccg_differs_from_base: bool
    dccg_differs_from_acg: bool
    dccg_differs_from_ablation: bool
    dccg_differs_from_smoothing: bool
    finite_nonzero_gradients: bool
    exact_base_passthrough_ok: bool
    gripper_event_preservation_ok: bool
    normalized_action_validity_ok: bool
    postprocessed_action_validity_ok: bool
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


def dccg_row_key(row: Mapping[str, Any]) -> str:
    fields: list[Any] = [
        row["split"],
        row["task_suite"],
        row["task_id"],
        row["demo_id"],
        row["window_start"],
        row["bin_key"],
        row["policy"],
        row["config_label"],
    ]
    if "probe_label" in row:
        fields.append(row["probe_label"])
    return "|".join(str(value) for value in fields)


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [dccg_row_key(row) for row in manifest_rows]
    completed = [str(row["row_key"]) for row in partial_rows]
    expected_set = set(expected)
    completed_set = set(completed)

    def split_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = [
            row["task_suite"],
            row["task_id"],
            row["demo_id"],
            row["window_start"],
            row["bin_key"],
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


def _finite_difference(values: np.ndarray, order: int) -> np.ndarray:
    result = values
    for _ in range(order):
        result = np.diff(result, axis=1)
    return result


def _stable_tail(values: np.ndarray, *, tau: float = TAIL_TAU) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    scaled = values / tau
    maximum = np.max(scaled, axis=1, keepdims=True)
    return tau * (np.log(np.mean(np.exp(scaled - maximum), axis=1)) + maximum[:, 0])


def _dct_high_frequency_energy(translation: np.ndarray) -> np.ndarray:
    centered = translation - np.mean(translation, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    if spectrum.shape[1] <= 3:
        return np.zeros(len(translation), dtype=np.float64)
    high = spectrum[:, 3:, :]
    return np.mean(np.abs(high) ** 2, axis=(1, 2)).astype(np.float64)


def coherence_features(chunks: Any) -> np.ndarray:
    """Return the frozen differentiable DCCG feature proxy."""
    action = chunk_matrix(chunks, "chunks")
    translation = action[:, :, 0:3]
    rotation = action[:, :, 3:6]
    gripper = action[:, :, 6]

    d1_t = _finite_difference(translation, 1)
    d2_t = _finite_difference(translation, 2)
    d3_t = _finite_difference(translation, 3)
    d1_r = _finite_difference(rotation, 1)
    d2_r = _finite_difference(rotation, 2)
    d3_r = _finite_difference(rotation, 3)

    v_t = np.linalg.norm(d1_t, axis=2)
    a_t = np.linalg.norm(d2_t, axis=2)
    j_t = np.linalg.norm(d3_t, axis=2)
    v_r = np.linalg.norm(d1_r, axis=2)
    a_r = np.linalg.norm(d2_r, axis=2)
    j_r = np.linalg.norm(d3_r, axis=2)

    pause_arg = np.clip((v_t - EPSILON_PAUSE) / TAU_PAUSE, -60.0, 60.0)
    pause = np.mean(1.0 / (1.0 + np.exp(pause_arg)), axis=1)
    soft = np.tanh(gripper / TAU_GRIP)
    transition = np.sum((1.0 - soft[:, 1:] * soft[:, :-1]) / 2.0, axis=1)
    reversal = np.sum(np.abs(np.diff(soft, n=2, axis=1)), axis=1)

    features = np.stack(
        [
            _stable_tail(v_t),
            _stable_tail(a_t),
            _stable_tail(j_t),
            _stable_tail(v_r),
            _stable_tail(a_r),
            _stable_tail(j_r),
            pause,
            _dct_high_frequency_energy(translation),
            transition,
            reversal,
        ],
        axis=1,
    )
    if features.shape[1] != FEATURE_COUNT:
        raise AssertionError(f"expected {FEATURE_COUNT} features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("coherence features contain nonfinite values")
    return features.astype(np.float64)


def deployment_bin_key(chunk: Any, *, task_family: str, queue_index: int | None = None) -> str:
    action = chunk_matrix(chunk, "chunk")[0]
    translation_mag = float(np.mean(np.linalg.norm(np.diff(action[:, 0:3], axis=0), axis=1)))
    rotation_mag = float(np.mean(np.linalg.norm(np.diff(action[:, 3:6], axis=0), axis=1)))
    gripper_mag = float(np.mean(np.abs(action[:, 6])))
    gripper_changes = int(np.sum(np.diff(np.signbit(action[:, 6])) != 0))
    queue = "qNA" if queue_index is None else f"q{int(queue_index) // 10}"
    return (
        f"{task_family}|{queue}|t{int(translation_mag > 0.01)}|"
        f"r{int(rotation_mag > 0.01)}|g{int(gripper_mag > 0.1)}|c{int(gripper_changes > 0)}"
    )


def fit_demo_statistics(features: Any, bin_keys: Sequence[str]) -> dict[str, dict[str, Any]]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise ValueError(f"features must have shape [N,{FEATURE_COUNT}], got {values.shape}")
    if len(values) != len(bin_keys):
        raise ValueError("features and bin keys must align")
    stats: dict[str, dict[str, Any]] = {}
    for key in sorted(set(str(item) for item in bin_keys)):
        mask = np.asarray([str(item) == key for item in bin_keys], dtype=bool)
        selected = values[mask]
        center = np.median(selected, axis=0)
        q75 = np.percentile(selected, 75, axis=0)
        q25 = np.percentile(selected, 25, axis=0)
        scale = np.maximum(q75 - q25, EPSILON_SCALE)
        stats[key] = {
            "count": int(mask.sum()),
            "center": center,
            "scale": scale,
            "feature_variance_mean": float(np.mean(np.var(selected, axis=0))) if len(selected) > 1 else 0.0,
        }
    return stats


def _stats_for(stats: Mapping[str, Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    if key in stats:
        item = stats[key]
    elif "global" in stats:
        item = stats["global"]
    else:
        first_key = sorted(stats)[0]
        item = stats[first_key]
    center = np.asarray(item["center"], dtype=np.float64)
    scale = np.maximum(np.asarray(item["scale"], dtype=np.float64), EPSILON_SCALE)
    return center, scale


def huber_values(error: Any, *, delta: float = HUBER_DELTA) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("huber input contains nonfinite values")
    threshold = float(delta)
    absolute = np.abs(value)
    return np.where(absolute <= threshold, 0.5 * np.square(value), threshold * (absolute - 0.5 * threshold))


def coherence_energy(chunks: Any, bin_keys: Sequence[str], stats: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    action = chunk_matrix(chunks, "chunks")
    if len(action) != len(bin_keys):
        raise ValueError("chunks and bin keys must align")
    features = coherence_features(action)
    energies = []
    for feature, key in zip(features, bin_keys, strict=True):
        center, scale = _stats_for(stats, str(key))
        z = (feature - center) / scale
        energies.append(float(np.mean(huber_values(z))))
    return np.asarray(energies, dtype=np.float64)


def numerical_energy_gradient(
    chunks: Any,
    bin_keys: Sequence[str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float = GRADIENT_EPSILON,
) -> np.ndarray:
    action = chunk_matrix(chunks, "chunks")
    gradient = np.zeros_like(action)
    for row in range(len(action)):
        key = [str(bin_keys[row])]
        for h in range(HORIZON):
            for d in range(ACTION_DIM):
                plus = action[row : row + 1].copy()
                minus = action[row : row + 1].copy()
                plus[0, h, d] += epsilon
                minus[0, h, d] -= epsilon
                e_plus = coherence_energy(plus, key, stats)[0]
                e_minus = coherence_energy(minus, key, stats)[0]
                gradient[row, h, d] = (e_plus - e_minus) / (2.0 * epsilon)
    if not np.isfinite(gradient).all():
        raise ValueError("energy gradient contains nonfinite values")
    return gradient


def group_clip(vector: Any) -> np.ndarray:
    value = chunk_matrix(vector, "vector").copy()
    translation = value[:, :, 0:3]
    translation_norm = np.linalg.norm(translation, axis=2, keepdims=True)
    value[:, :, 0:3] = translation * np.minimum(
        1.0, TRANSLATION_CAP / np.maximum(translation_norm, EPSILON_SCALE)
    )
    rotation = value[:, :, 3:6]
    rotation_norm = np.linalg.norm(rotation, axis=2, keepdims=True)
    value[:, :, 3:6] = rotation * np.minimum(1.0, ROTATION_CAP / np.maximum(rotation_norm, EPSILON_SCALE))
    value[:, :, 6:7] = np.clip(value[:, :, 6:7], -GRIPPER_CAP, GRIPPER_CAP)
    return value


def apply_dccg_guidance(base_chunks: Any, gradient: Any, gate: Any, *, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    base = chunk_matrix(base_chunks, "base_chunks")
    direction = group_clip(gradient)
    gate_array = np.asarray(gate, dtype=np.float64).reshape(-1, 1, 1)
    if len(base) != len(direction) or len(base) != len(gate_array):
        raise ValueError("base chunks, gradient, and gate must align")
    guided = base - float(gamma) * gate_array * direction
    gripper_sign_changed = np.signbit(guided[:, :, 6]) != np.signbit(base[:, :, 6])
    guided[:, :, 6] = np.where(gripper_sign_changed, base[:, :, 6], guided[:, :, 6])
    return guided.astype(np.float64), gate_array.reshape(-1)


def smoothing_simple_killer(chunks: Any, *, preserve_gripper_events: bool = True) -> np.ndarray:
    action = chunk_matrix(chunks, "chunks").copy()
    smoothed = action.copy()
    smoothed[:, 1:-1, 0:6] = (action[:, :-2, 0:6] + action[:, 1:-1, 0:6] + action[:, 2:, 0:6]) / 3.0
    if not preserve_gripper_events:
        smoothed[:, 1:-1, 6] = (action[:, :-2, 6] + action[:, 1:-1, 6] + action[:, 2:, 6]) / 3.0
    return smoothed


def no_demo_calibration_stats(features: Any) -> dict[str, dict[str, Any]]:
    values = np.asarray(features, dtype=np.float64)
    center = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), EPSILON_SCALE)
    return {"global": {"count": int(len(values)), "center": center, "scale": scale, "feature_variance_mean": float(np.mean(np.var(values, axis=0)))}}


def feature_health(features: Any, bin_keys: Sequence[str]) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    variance = np.var(values, axis=0) if len(values) else np.zeros(FEATURE_COUNT)
    counts = {str(key): int(sum(str(item) == str(key) for item in bin_keys)) for key in sorted(set(bin_keys))}
    return {
        "feature_count": int(values.shape[1]) if values.ndim == 2 else 0,
        "feature_variance_min": float(np.min(variance)) if len(variance) else 0.0,
        "feature_variance_mean": float(np.mean(variance)) if len(variance) else 0.0,
        "features_noncollapsed": bool(values.ndim == 2 and values.shape[1] == FEATURE_COUNT and np.max(variance) > 0.0),
        "bin_count": len(counts),
        "bins_noncollapsed": bool(len(counts) > 0 and min(counts.values()) > 0),
        "bin_counts": counts,
    }


def gripper_event_summary(base_chunks: Any, candidate_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    candidate = chunk_matrix(candidate_chunks, "candidate_chunks")
    if base.shape != candidate.shape:
        raise ValueError("base and candidate chunks must align")

    def transitions(array: np.ndarray) -> np.ndarray:
        return np.sum(np.diff(np.signbit(array[:, :, 6]), axis=1) != 0, axis=1)

    def reversals(array: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(np.diff(np.tanh(array[:, :, 6] / TAU_GRIP), n=2, axis=1)) > 0.5, axis=1)

    base_t = transitions(base)
    cand_t = transitions(candidate)
    base_r = reversals(base)
    cand_r = reversals(candidate)
    return {
        "base_transition_mean": float(np.mean(base_t)),
        "candidate_transition_mean": float(np.mean(cand_t)),
        "transition_delta_max": int(np.max(np.abs(cand_t - base_t))) if len(base_t) else 0,
        "base_reversal_mean": float(np.mean(base_r)),
        "candidate_reversal_mean": float(np.mean(cand_r)),
        "reversal_delta_max": int(np.max(np.abs(cand_r - base_r))) if len(base_r) else 0,
        "gripper_event_preservation_ok": bool(np.max(np.abs(cand_t - base_t)) <= 1 if len(base_t) else True),
    }


def action_delta_summary(base_chunks: Any, candidate_chunks: Any) -> dict[str, Any]:
    base = chunk_matrix(base_chunks, "base_chunks")
    candidate = chunk_matrix(candidate_chunks, "candidate_chunks")
    delta = candidate - base
    trans = np.abs(delta[:, :, 0:3])
    rot = np.abs(delta[:, :, 3:6])
    grip = np.abs(delta[:, :, 6:7])
    changed = np.abs(delta) > 1e-12
    return {
        "translation_delta_p95": float(np.percentile(trans, 95)),
        "rotation_delta_p95": float(np.percentile(rot, 95)),
        "gripper_delta_p95": float(np.percentile(grip, 95)),
        "translation_delta_max": float(np.max(trans)),
        "rotation_delta_max": float(np.max(rot)),
        "gripper_delta_max": float(np.max(grip)),
        "changed_cell_fraction": float(np.mean(changed)),
        "action_deltas_bounded": bool(
            np.max(trans) <= TRANSLATION_CAP + 1e-12
            and np.max(rot) <= ROTATION_CAP + 1e-12
            and np.max(grip) <= GRIPPER_CAP + 1e-12
        ),
    }


def action_validity_summary(chunks: Any) -> dict[str, Any]:
    try:
        action = chunk_matrix(chunks, "chunks")
    except ValueError as exc:
        return {
            "action_validity_ok": False,
            "error": str(exc),
            "finite_fraction": 0.0,
            "shape": None,
        }
    return {
        "action_validity_ok": True,
        "finite_fraction": float(np.mean(np.isfinite(action))),
        "shape": list(action.shape),
        "normalized_range_min": float(np.min(action)),
        "normalized_range_max": float(np.max(action)),
    }


def gradient_smoke(chunks: Any, bin_keys: Sequence[str], stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gradient = numerical_energy_gradient(chunks, bin_keys, stats)
    norms = np.linalg.norm(gradient.reshape(len(gradient), -1), axis=1)
    group_norms = {
        "translation": float(np.linalg.norm(gradient[:, :, 0:3])),
        "rotation": float(np.linalg.norm(gradient[:, :, 3:6])),
        "gripper": float(np.linalg.norm(gradient[:, :, 6:7])),
    }
    return {
        "finite_nonzero_gradients": bool(np.isfinite(gradient).all() and np.any(np.abs(gradient) > 0.0)),
        "gradient_norm_mean": float(np.mean(norms)),
        "gradient_norm_max": float(np.max(norms)),
        "group_norms": group_norms,
        "gradient": gradient,
    }


def relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(float(baseline)) or not np.isfinite(float(candidate)):
        return 0.0
    return float((float(baseline) - float(candidate)) / max(abs(float(baseline)), EPSILON_SCALE))


def classify_stage0(inputs: Stage0DecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.serializer_preflight_ok
        or not inputs.official_prior_asset_check_persisted
        or not inputs.preflight_passed
        or not inputs.source_alignment_ok
        or not inputs.action_semantics_ok
        or not inputs.base_chunks_valid
        or not inputs.exact_base_passthrough_ok
        or inputs.reward_read_count
        or inputs.success_read_count
        or inputs.done_read_count
        or inputs.confirmatory_records_read
        or inputs.closed_loop_experiment_happened
        or inputs.simulator_load_count
        or inputs.training_happened
        or inputs.validation_search_happened
        or inputs.exception_count
    ):
        return "DCCG_STAGE_0_IMPLEMENTATION_FAILURE"
    if (
        not inputs.manifest_integrity_ok
        or not inputs.features_noncollapsed
        or not inputs.bins_noncollapsed
        or not inputs.enough_discovery_windows
        or not inputs.enough_validation_windows
        or not inputs.validation_task_coverage_ok
        or inputs.maximum_validation_task_fraction > 0.40
    ):
        return "DCCG_STAGE_0_DATA_FAILURE"
    if (
        not inputs.finite_nonzero_gradients
        or not inputs.normalized_action_validity_ok
        or not inputs.postprocessed_action_validity_ok
    ):
        return "DCCG_STAGE_0_IMPLEMENTATION_FAILURE"
    if inputs.base_acg_headroom < ACG_HEADROOM_MIN:
        return "DCCG_STAGE_0_NO_HEADROOM"
    if (
        inputs.gate_activation_fraction < GATE_ACTIVATION_MIN
        or inputs.gate_activation_fraction > GATE_ACTIVATION_MAX
        or not inputs.dccg_differs_from_base
        or not inputs.dccg_differs_from_acg
        or not inputs.dccg_differs_from_ablation
        or not inputs.dccg_differs_from_smoothing
        or not inputs.gripper_event_preservation_ok
        or not inputs.clean_retention_ok
    ):
        return "DCCG_STAGE_0_DESIGN_FAILURE"
    return "DCCG_STAGE_0_PASS_TO_VALIDATION_SEARCH"
