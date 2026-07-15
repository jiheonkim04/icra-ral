"""Frozen HASTE-VLA event labels, probe losses, and Stage 0A decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6"
ACTION_DIM = 7
ARM_DIM = 6
EVENT_HORIZONS = (20, 50)
EVENT_THRESHOLD = 1e-8
STD_FLOOR = 1e-6


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def event_row_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
        row["event_horizon"],
    )
    return "|".join(str(value) for value in fields)


def frame_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        row["demo_id"],
        row["frame_index"],
    )
    return "|".join(str(value) for value in fields)


def construct_event_label(
    actions: Any,
    frame_index: int,
    event_horizon: int,
    *,
    threshold: float = EVENT_THRESHOLD,
) -> dict[str, Any]:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTION_DIM}], received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("actions contain nonfinite values")
    frame = int(frame_index)
    horizon = int(event_horizon)
    if horizon not in EVENT_HORIZONS:
        raise ValueError(f"event horizon must be one of {EVENT_HORIZONS}")
    if frame < 0 or frame >= len(array) - 1:
        raise ValueError("frame must have at least one observable future interval")

    valid_intervals = min(horizon, len(array) - 1 - frame)
    transition_offset: int | None = None
    for offset in range(1, valid_intervals + 1):
        before = float(array[frame + offset - 1, 6])
        after = float(array[frame + offset, 6])
        if abs(after - before) > float(threshold):
            transition_offset = offset
            break

    event = np.zeros(horizon, dtype=np.float32)
    survival_mask = np.zeros(horizon, dtype=bool)
    likelihood_terms = transition_offset if transition_offset is not None else valid_intervals
    survival_mask[:likelihood_terms] = True
    displacement: np.ndarray | None = None
    if transition_offset is not None:
        event[transition_offset - 1] = 1.0
        displacement = array[frame : frame + transition_offset + 1, :ARM_DIM].sum(axis=0)

    return {
        "valid_interval_count": int(valid_intervals),
        "likelihood_term_count": int(likelihood_terms),
        "transition_offset": transition_offset,
        "censored": transition_offset is None,
        "censor_reason": None if transition_offset is not None else "right_boundary_or_horizon",
        "event_target": event,
        "survival_mask": survival_mask,
        "relative_displacement": displacement,
    }


def displacement_statistics(displacements: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    if not displacements:
        raise ValueError("uncensored discovery displacement targets are required")
    values = np.stack([np.asarray(value, dtype=np.float64).reshape(ARM_DIM) for value in displacements])
    if not np.isfinite(values).all():
        raise ValueError("displacements contain nonfinite values")
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0, ddof=0), STD_FLOOR)
    return mean, std


def normalize_displacement(value: Any, mean: Any, std: Any) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64).reshape(ARM_DIM)
    mu = np.asarray(mean, dtype=np.float64).reshape(ARM_DIM)
    scale = np.maximum(np.asarray(std, dtype=np.float64).reshape(ARM_DIM), STD_FLOOR)
    return (target - mu) / scale


def fit_constant_hazard(rows: Sequence[Mapping[str, Any]], event_horizon: int) -> np.ndarray:
    horizon = int(event_horizon)
    risk = np.zeros(horizon, dtype=np.float64)
    events = np.zeros(horizon, dtype=np.float64)
    for row in rows:
        if int(row["event_horizon"]) != horizon:
            continue
        mask = np.asarray(row["survival_mask"], dtype=bool).reshape(horizon)
        target = np.asarray(row["event_target"], dtype=np.float64).reshape(horizon)
        risk += mask.astype(np.float64)
        events += target
    return (events + 1.0) / (risk + 2.0)


def hazard_nll_from_probabilities(probabilities: Any, rows: Sequence[Mapping[str, Any]]) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.repeat(probs[None, :], len(rows), axis=0)
    if probs.shape[0] != len(rows):
        raise ValueError("hazard probability row count mismatch")
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    losses = []
    for index, row in enumerate(rows):
        horizon = int(row["event_horizon"])
        mask = np.asarray(row["survival_mask"], dtype=bool).reshape(horizon)
        target = np.asarray(row["event_target"], dtype=np.float64).reshape(horizon)
        selected = probs[index, :horizon]
        terms = -(target * np.log(selected) + (1.0 - target) * np.log(1.0 - selected))
        losses.append(float(terms[mask].mean()))
    return float(np.mean(losses)) if losses else float("nan")


def huber_loss(prediction: Any, target: Any, *, delta: float = 1.0) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    values = np.where(absolute <= delta, 0.5 * np.square(error), delta * (absolute - 0.5 * delta))
    return float(np.mean(values))


def offset_quintile(offset: int, event_horizon: int) -> int:
    fraction = (int(offset) - 1) / max(int(event_horizon), 1)
    return min(4, max(0, int(np.floor(5.0 * fraction))))


def event_stratum(row: Mapping[str, Any]) -> str:
    offset = row.get("transition_offset")
    if offset is None:
        return "censored"
    return "event_near" if int(offset) <= 10 else "event_far"


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    manifest_keys = [str(row["event_row_key"]) for row in manifest_rows]
    partial_keys = [str(row["event_row_key"]) for row in partial_rows]
    manifest_set = set(manifest_keys)
    partial_set = set(partial_keys)
    discovery = {str(row["event_row_key"]) for row in manifest_rows if row["partition"] == "discovery"}
    validation = {str(row["event_row_key"]) for row in manifest_rows if row["partition"] == "validation"}
    return {
        "manifest_row_count": len(manifest_keys),
        "partial_row_count": len(partial_keys),
        "duplicate_manifest_key_count": len(manifest_keys) - len(manifest_set),
        "duplicate_partial_key_count": len(partial_keys) - len(partial_set),
        "missing_manifest_key_count": len(manifest_set - partial_set),
        "extra_partial_key_count": len(partial_set - manifest_set),
        "partition_overlap_count": len(discovery & validation),
        "key_sets_equal": manifest_set == partial_set,
    }


@dataclass(frozen=True)
class Stage0ADecisionInputs:
    proposal_hash_ok: bool
    manifest_integrity_ok: bool
    finite_source_and_features: bool
    discovery_uncensored_count: int
    discovery_censored_count: int
    minimum_validation_uncensored_per_task: int
    occupied_offset_quintile_count: int
    displacement_variance_all_positive: bool
    maximum_uncensored_task_fraction: float
    base_event_near_headroom: bool
    hazard_probe_improvement: float
    displacement_probe_improvement: float
    identity_max_error: float
    base_hash_unchanged: bool
    checkpoint_reload_ok: bool
    exception_count: int


def classify_stage0a(inputs: Stage0ADecisionInputs) -> str:
    if (
        not inputs.proposal_hash_ok
        or not inputs.manifest_integrity_ok
        or int(inputs.discovery_uncensored_count) < 128
        or int(inputs.discovery_censored_count) < 128
        or int(inputs.minimum_validation_uncensored_per_task) < 16
        or int(inputs.occupied_offset_quintile_count) < 5
        or not inputs.displacement_variance_all_positive
        or float(inputs.maximum_uncensored_task_fraction) > 0.40
    ):
        return "HASTE_STAGE_0A_DATA_FAILURE"
    if (
        not inputs.finite_source_and_features
        or float(inputs.identity_max_error) > 1e-6
        or not inputs.base_hash_unchanged
        or not inputs.checkpoint_reload_ok
        or int(inputs.exception_count) != 0
    ):
        return "HASTE_STAGE_0A_IMPLEMENTATION_FAILURE"
    if not inputs.base_event_near_headroom:
        return "HASTE_STAGE_0A_NO_HEADROOM"
    if float(inputs.hazard_probe_improvement) < 0.02 or float(inputs.displacement_probe_improvement) < 0.02:
        return "HASTE_STAGE_0A_DESIGN_FAILURE"
    return "HASTE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
