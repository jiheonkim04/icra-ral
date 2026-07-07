"""Event-locked retiming diagnostic for temporal action-chunk mismatch.

This runner is bounded replay/control diagnostic evidence only. It uses local
LIBERO HDF5 expert demonstrations and exact-init replay to test whether
event-locked action timing can recover from temporal perturbations better than
simple timing and action-only baselines. It performs no training, model
loading, VLA inference, GPU work, downloads, OpenVLA-OFT execution, benchmark
sweep, or paper-grade claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import (
    _action_stats,
    _as_path,
    _compact,
    _load_env_class,
    _load_json,
    _write_json,
)
from tca_map.datasets.libero_full_demo_expert_replay_sanity import (
    _controller_summary,
    _read_demo_full,
    _safe_l2,
    _sim_state_array,
)
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _extract_eef,
    _extract_pos,
    _object_position_keys,
)
from tca_map.execspec.mismatch_diagnostic import action_metrics


SCHEMA_VERSION = "2026-07-07.phase_locked_retiming.v1"
TASK_GATE = "ALLOW_PHASE_LOCKED_RETIMING"
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
    "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS",
    "ALLOW_FULL_DEMO_EXPERT_REPLAY",
    "ALLOW_EXECSPEC_MISMATCH_REPLAY",
    "ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY",
    "ALLOW_EXECSPEC_STATE3_REPLAY_VALIDATION",
    "ALLOW_AMP_GD_STATE2",
    "ALLOW_RESETSPEC_RETARGET",
)

PERTURBATION_NAMES = (
    "gripper_close_delayed",
    "gripper_close_early",
    "lift_phase_delayed",
    "lift_phase_early",
    "chunk_shifted_forward",
    "chunk_shifted_backward",
    "time_stretch",
    "time_compression",
    "chunk_boundary_offset",
)
STATIC_BASELINES = (
    "raw_perturbed_replay",
    "fixed_time_shift",
    "repeat_last_hold",
    "gripper_only_timing_correction",
    "global_scale",
    "diagonal_affine",
    "linear_time_warp",
)
DYNAMIC_BASELINES = (
    "nearest_progress_demo",
    "event_locked_retiming",
)
ALL_BASELINES = (*STATIC_BASELINES, *DYNAMIC_BASELINES)


def _round(value: float | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _norm(values: Any) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return float(np.linalg.norm(arr))


def _md(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return str(_round(value, 6))
    return str(value)


def _range(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "max_abs": None}
    return {
        "min": _round(float(arr.min()), 6),
        "max": _round(float(arr.max()), 6),
        "mean": _round(float(arr.mean()), 6),
        "std": _round(float(arr.std()), 6),
        "max_abs": _round(float(np.max(np.abs(arr))), 6),
    }


def _success(result: dict[str, Any]) -> bool:
    return bool(result.get("final_success") or result.get("done_seen") or float(result.get("reward_sum") or 0.0) > 0.0)


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    for index, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _first_gripper_nonnegative(actions: np.ndarray) -> int | None:
    if actions.ndim != 2 or actions.shape[1] < 7:
        return None
    for index, value in enumerate(actions[:, 6]):
        if float(value) >= 0.0:
            return int(index)
    return None


def _gripper_timing_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = _first_gripper_nonnegative(reference)
    cand = _first_gripper_nonnegative(candidate)
    return {
        "reference_first_nonnegative_index": ref,
        "candidate_first_nonnegative_index": cand,
        "absolute_error": None if ref is None or cand is None else abs(int(ref) - int(cand)),
    }


def _clip_stats(raw_actions: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw_actions, dtype=np.float64)
    if raw.ndim != 2 or raw.size == 0:
        return {"controller_valid_action_rate": 0.0, "clip_rate_element": 0.0, "clip_rate_step": 0.0}
    finite = np.all(np.isfinite(raw), axis=1)
    in_range = np.all(np.abs(raw) <= 1.0, axis=1)
    clipped = np.abs(raw) > 1.0
    return {
        "controller_valid_action_rate": _round(float(np.mean(finite & in_range)), 6),
        "clip_rate_element": _round(float(np.mean(clipped)), 6),
        "clip_rate_step": _round(float(np.mean(np.any(clipped, axis=1))), 6),
    }


def _target_audit(obs: Any, instruction: str) -> dict[str, Any]:
    audit = _best_object_key(obs, instruction)
    return {
        "source": "natural_language_instruction_text_plus_visible_observation_object_keys",
        "uses_bddl_metadata": False,
        "uses_dataset_target_labels": False,
        "uses_eval_labels": False,
        "uses_task_id_filename_or_manifest_target_field_as_target_proxy": False,
        "best_key": audit.get("best_key"),
        "best_score": audit.get("best_score"),
        "best_overlap": audit.get("best_overlap"),
        "available_object_position_keys": audit.get("available_object_position_keys") or [],
    }


def _read_hdf5_eef_positions(path: Path, demo_name: str, limit: int) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        obs = demo.get("obs")
        if obs is None:
            return {"available": False, "key": None, "positions": None, "obs_keys": []}
        keys = sorted(str(key) for key in obs.keys())
        for key in ("robot0_eef_pos", "ee_pos", "eef_pos"):
            if key in obs:
                positions = np.asarray(obs[key][: max(1, min(limit, obs[key].shape[0]))], dtype=np.float64)
                if positions.ndim == 2 and positions.shape[1] >= 3:
                    return {"available": True, "key": key, "positions": positions[:, :3], "obs_keys": keys}
        return {"available": False, "key": None, "positions": None, "obs_keys": keys}


def _read_hdf5_object_positions(path: Path, demo_name: str, instruction: str, limit: int) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        obs = demo.get("obs")
        if obs is None:
            return {"available": False, "key": None, "positions": None, "audit": None, "obs_keys": []}
        keys = sorted(str(key) for key in obs.keys())
        first_obs: dict[str, Any] = {}
        for key in keys:
            if key.endswith("_pos") and not key.startswith("robot") and key not in {"ee_pos", "eef_pos"}:
                first_obs[key] = np.asarray(obs[key][0], dtype=np.float64)
        audit = _best_object_key(first_obs, instruction)
        key = audit.get("best_key")
        if key is not None and key in obs:
            positions = np.asarray(obs[key][: max(1, min(limit, obs[key].shape[0]))], dtype=np.float64)
            if positions.ndim == 2 and positions.shape[1] >= 3:
                return {"available": True, "key": key, "positions": positions[:, :3], "audit": audit, "obs_keys": keys}
        return {"available": False, "key": key, "positions": None, "audit": audit, "obs_keys": keys}


def _manifest_pair(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("counterfactual split manifest has no counterfactual_pairs")
    return pairs[0]


def _shift_sequence(actions: np.ndarray, offset: int, dims: list[int] | None = None) -> np.ndarray:
    """Shift an action sequence in time. Positive offset delays events."""
    source = np.asarray(actions, dtype=np.float64)
    out = source.copy()
    if source.ndim != 2 or source.shape[0] == 0:
        return out
    idx = np.arange(source.shape[0]) - int(offset)
    idx = np.clip(idx, 0, source.shape[0] - 1)
    shifted = source[idx]
    if dims is None:
        return shifted.copy()
    out[:, dims] = shifted[:, dims]
    return out


def _shift_after_anchor(actions: np.ndarray, anchor: int | None, offset: int, dims: list[int]) -> np.ndarray:
    source = np.asarray(actions, dtype=np.float64)
    out = source.copy()
    if anchor is None or source.ndim != 2 or source.shape[0] == 0:
        return out
    start = int(np.clip(anchor, 0, source.shape[0] - 1))
    local = np.arange(source.shape[0] - start) - int(offset)
    idx = np.clip(start + local, start, source.shape[0] - 1)
    out[start:, dims] = source[idx, :][:, dims]
    return out


def _time_warp(actions: np.ndarray, factor: float) -> np.ndarray:
    """Resample by a fixed factor. factor > 1 stretches/slows the sequence."""
    source = np.asarray(actions, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0 or not np.isfinite(factor) or factor <= 0:
        return source.copy()
    idx = np.floor(np.arange(source.shape[0], dtype=np.float64) / float(factor)).astype(int)
    idx = np.clip(idx, 0, source.shape[0] - 1)
    return source[idx].copy()


def _repeat_last_hold(actions: np.ndarray) -> np.ndarray:
    source = np.asarray(actions, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] <= 1:
        return source.copy()
    out = source.copy()
    out[1:] = source[:-1]
    return out


def _trajectory_drift(observed_eef: list[list[float]], demo_eef_positions: np.ndarray | None) -> dict[str, Any]:
    if not observed_eef or demo_eef_positions is None:
        return {"available": False, "mean_l2": None, "final_l2": None, "sample_count": 0}
    count = min(len(observed_eef), demo_eef_positions.shape[0])
    if count == 0:
        return {"available": False, "mean_l2": None, "final_l2": None, "sample_count": 0}
    actual = np.asarray(observed_eef[:count], dtype=np.float64)
    expected = np.asarray(demo_eef_positions[:count, :3], dtype=np.float64)
    distances = np.linalg.norm(actual - expected, axis=1)
    return {
        "available": True,
        "mean_l2": _round(float(np.mean(distances)), 6),
        "final_l2": _round(float(distances[-1]), 6),
        "sample_count": int(count),
    }


def _first_object_motion(object_positions: np.ndarray | None, threshold: float = 0.005) -> int | None:
    if object_positions is None or object_positions.ndim != 2 or object_positions.shape[0] < 2:
        return None
    start = np.asarray(object_positions[0, :3], dtype=np.float64)
    deltas = np.linalg.norm(np.asarray(object_positions[:, :3], dtype=np.float64) - start.reshape(1, 3), axis=1)
    return _first_index(deltas, threshold)


def _first_lift(object_positions: np.ndarray | None, eef_positions: np.ndarray | None, close_index: int | None, threshold: float = 0.02) -> int | None:
    if object_positions is not None and object_positions.ndim == 2 and object_positions.shape[0] > 1:
        z = np.asarray(object_positions[:, 2], dtype=np.float64)
        lift = _first_index(z - float(z[0]), threshold)
        if lift is not None:
            return lift
    if eef_positions is not None and eef_positions.ndim == 2 and eef_positions.shape[0] > 1:
        start = int(close_index or 0)
        start = int(np.clip(start, 0, eef_positions.shape[0] - 1))
        z = np.asarray(eef_positions[:, 2], dtype=np.float64)
        local = _first_index(z[start:] - float(z[start]), threshold)
        if local is not None:
            return start + int(local)
    return None


def extract_event_anchors(
    actions: np.ndarray,
    eef_positions: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    horizon = int(actions.shape[0]) if actions.ndim == 2 else 0
    close = _first_gripper_nonnegative(actions)
    distances = None
    approach = None
    if eef_positions is not None:
        eef = np.asarray(eef_positions, dtype=np.float64)
        if object_positions is not None:
            obj = np.asarray(object_positions, dtype=np.float64)
            count = min(eef.shape[0], obj.shape[0], horizon)
            if count > 0:
                distances = np.linalg.norm(eef[:count, :3] - obj[:count, :3], axis=1)
        elif eef.ndim == 2 and eef.shape[0] > 0:
            count = min(eef.shape[0], horizon)
            distances = np.linalg.norm(eef[:count, :3] - eef[0, :3].reshape(1, 3), axis=1)
        if distances is not None and distances.size:
            stop = min(close if close is not None else distances.size, distances.size)
            stop = max(1, stop)
            approach = int(np.argmin(distances[:stop]))
    if approach is None and actions.ndim == 2 and horizon > 0:
        trans = np.linalg.norm(actions[:, :3], axis=1)
        approach = _first_index(trans, 1e-6) or 0
    motion = _first_object_motion(object_positions)
    lift = _first_lift(object_positions, eef_positions, close)
    place = None
    if object_positions is not None and lift is not None:
        obj = np.asarray(object_positions, dtype=np.float64)
        z = obj[:, 2]
        if obj.shape[0] > lift + 2:
            peak = int(lift + np.argmax(z[lift:]))
            if peak + 1 < obj.shape[0]:
                down = _first_index(float(z[peak]) - z[peak:], 0.01)
                if down is not None:
                    place = peak + int(down)
    anchors = {
        "approach_index": approach,
        "gripper_close_index": close,
        "object_motion_onset_index": motion,
        "lift_index": lift,
        "place_or_contact_index": place,
        "horizon": horizon,
    }
    if distances is not None and distances.size:
        anchors["demo_eef_object_distance"] = {
            "available": True,
            "min": _round(float(np.min(distances)), 6),
            "min_index": int(np.argmin(distances)),
            "start": _round(float(distances[0]), 6),
            "final": _round(float(distances[-1]), 6),
        }
    else:
        anchors["demo_eef_object_distance"] = {"available": False}
    return anchors


def _anchor_or_default(anchors: dict[str, Any], key: str, default: int) -> int:
    value = anchors.get(key)
    if isinstance(value, int):
        return int(value)
    return int(default)


def build_phase_perturbations(
    actions: np.ndarray,
    anchors: dict[str, Any],
    *,
    offset_steps: int = 18,
    stretch_factor: float = 1.15,
    compression_factor: float = 0.85,
) -> dict[str, dict[str, Any]]:
    actions = np.asarray(actions, dtype=np.float64)
    close = _anchor_or_default(anchors, "gripper_close_index", max(1, actions.shape[0] // 4))
    lift = _anchor_or_default(anchors, "lift_index", max(close + 1, actions.shape[0] // 2))
    boundary = _anchor_or_default(anchors, "gripper_close_index", max(1, actions.shape[0] // 3))
    offset = int(offset_steps)
    perturbations = {
        "gripper_close_delayed": {
            "actions": _shift_sequence(actions, offset, dims=[6]),
            "family": "gripper_shift",
            "offset_steps": offset,
            "description": "Delay gripper timing only.",
        },
        "gripper_close_early": {
            "actions": _shift_sequence(actions, -offset, dims=[6]),
            "family": "gripper_shift",
            "offset_steps": -offset,
            "description": "Advance gripper timing only.",
        },
        "lift_phase_delayed": {
            "actions": _shift_after_anchor(actions, lift, offset, dims=[0, 1, 2, 3, 4, 5]),
            "family": "phase_shift_after_anchor",
            "anchor": "lift_index",
            "anchor_index": lift,
            "offset_steps": offset,
            "description": "Delay post-lift translation/rotation timing.",
        },
        "lift_phase_early": {
            "actions": _shift_after_anchor(actions, lift, -offset, dims=[0, 1, 2, 3, 4, 5]),
            "family": "phase_shift_after_anchor",
            "anchor": "lift_index",
            "anchor_index": lift,
            "offset_steps": -offset,
            "description": "Advance post-lift translation/rotation timing.",
        },
        "chunk_shifted_forward": {
            "actions": _shift_sequence(actions, -offset),
            "family": "global_shift",
            "offset_steps": -offset,
            "description": "Advance the full action chunk.",
        },
        "chunk_shifted_backward": {
            "actions": _shift_sequence(actions, offset),
            "family": "global_shift",
            "offset_steps": offset,
            "description": "Delay the full action chunk.",
        },
        "time_stretch": {
            "actions": _time_warp(actions, stretch_factor),
            "family": "linear_time_warp",
            "factor": float(stretch_factor),
            "description": "Slow the full action chunk with a fixed stretch.",
        },
        "time_compression": {
            "actions": _time_warp(actions, compression_factor),
            "family": "linear_time_warp",
            "factor": float(compression_factor),
            "description": "Speed up the full action chunk with a fixed compression.",
        },
        "chunk_boundary_offset": {
            "actions": _shift_after_anchor(actions, boundary, offset, dims=[0, 1, 2, 3, 4, 5, 6]),
            "family": "boundary_offset",
            "anchor": "gripper_close_index",
            "anchor_index": boundary,
            "offset_steps": offset,
            "description": "Delay all actions after the gripper-close chunk boundary.",
        },
    }
    return perturbations


def _baseline_static_actions(
    baseline: str,
    expert_actions: np.ndarray,
    perturbed_actions: np.ndarray,
    perturbation: dict[str, Any],
    *,
    global_scale: float,
) -> np.ndarray | None:
    if baseline == "raw_perturbed_replay":
        return np.asarray(perturbed_actions, dtype=np.float64).copy()
    if baseline == "fixed_time_shift":
        offset = int(perturbation.get("offset_steps") or 0)
        return _shift_sequence(perturbed_actions, -offset)
    if baseline == "repeat_last_hold":
        return _repeat_last_hold(perturbed_actions)
    if baseline == "gripper_only_timing_correction":
        out = np.asarray(perturbed_actions, dtype=np.float64).copy()
        out[:, 6] = np.asarray(expert_actions, dtype=np.float64)[:, 6]
        return out
    if baseline == "global_scale":
        out = np.asarray(perturbed_actions, dtype=np.float64).copy()
        out[:, :6] *= float(global_scale)
        return out
    if baseline == "diagonal_affine":
        return np.asarray(perturbed_actions, dtype=np.float64).copy()
    if baseline == "linear_time_warp":
        factor = perturbation.get("factor")
        if isinstance(factor, (int, float)) and float(factor) > 0:
            return _time_warp(perturbed_actions, 1.0 / float(factor))
        return np.asarray(perturbed_actions, dtype=np.float64).copy()
    return None


def _demo_distance_series(case: dict[str, Any], demo_object: list[float] | None) -> np.ndarray | None:
    eef = case.get("hdf5_eef_positions")
    if eef is None or demo_object is None:
        return None
    eef_arr = np.asarray(eef, dtype=np.float64)
    if eef_arr.ndim != 2 or eef_arr.shape[0] == 0:
        return None
    obj = np.asarray(demo_object, dtype=np.float64).reshape(1, 3)
    count = min(eef_arr.shape[0], case["actions"].shape[0])
    return np.linalg.norm(eef_arr[:count, :3] - obj, axis=1)


def _nearest_progress_index(
    *,
    case: dict[str, Any],
    obs: Any,
    target_key: str | None,
    demo_object: list[float] | None,
    previous_index: int | None,
) -> int:
    actions = np.asarray(case["actions"], dtype=np.float64)
    distances = _demo_distance_series(case, demo_object)
    current_eef = _extract_eef(obs)
    current_object = _extract_pos(obs, target_key)
    if distances is None or current_eef is None or current_object is None:
        base = 0 if previous_index is None else min(previous_index + 1, actions.shape[0] - 1)
        return int(base)
    current_distance = _norm(np.asarray(current_eef, dtype=np.float64) - np.asarray(current_object, dtype=np.float64))
    if current_distance is None:
        return int(0 if previous_index is None else min(previous_index + 1, actions.shape[0] - 1))
    index = int(np.argmin(np.abs(distances - float(current_distance))))
    if previous_index is not None:
        index = max(int(previous_index), index)
    return int(np.clip(index, 0, actions.shape[0] - 1))


def select_event_locked_index(
    *,
    case: dict[str, Any],
    obs: Any,
    target_key: str | None,
    demo_object: list[float] | None,
    start_object: list[float] | None,
    previous_index: int | None,
) -> tuple[int, dict[str, Any]]:
    actions = np.asarray(case["actions"], dtype=np.float64)
    anchors = case.get("event_anchors") or {}
    nearest = _nearest_progress_index(
        case=case,
        obs=obs,
        target_key=target_key,
        demo_object=demo_object,
        previous_index=previous_index,
    )
    current_eef = _extract_eef(obs)
    current_object = _extract_pos(obs, target_key)
    current_distance = _distance(current_eef, current_object)
    object_motion = _distance(start_object, current_object)
    close = anchors.get("gripper_close_index")
    motion = anchors.get("object_motion_onset_index")
    lift = anchors.get("lift_index")
    selected = nearest
    phase = "nearest_progress"
    distances = _demo_distance_series(case, demo_object)
    if current_distance is not None and distances is not None and close is not None:
        close_idx = int(np.clip(close, 0, len(distances) - 1))
        close_distance = float(distances[close_idx])
        if float(current_distance) <= close_distance + 0.015:
            selected = max(selected, close_idx)
            phase = "gripper_close_event"
    if object_motion is not None and float(object_motion) > 0.004 and motion is not None:
        selected = max(selected, int(motion))
        phase = "object_motion_event"
    if object_motion is not None and float(object_motion) > 0.015 and lift is not None:
        selected = max(selected, int(lift))
        phase = "lift_event"
    if previous_index is not None:
        selected = max(int(previous_index), int(selected))
    selected = int(np.clip(selected, 0, actions.shape[0] - 1))
    return selected, {
        "phase": phase,
        "nearest_index": nearest,
        "current_distance": current_distance,
        "object_motion_l2": object_motion,
        "selected_index": selected,
    }


def build_phase_locked_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 320,
    post_signal_margin: int = 20,
    offset_steps: int = 18,
    stretch_factor: float = 1.15,
    compression_factor: float = 0.85,
) -> dict[str, Any]:
    pair = _manifest_pair(manifest_path)
    positive = _read_demo_full(_as_path(pair["positive_demo_file"]), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    actions = np.asarray(positive["actions"], dtype=np.float64)
    limit = int(actions.shape[0]) + 1
    eef = _read_hdf5_eef_positions(_as_path(positive["path"]), positive["demo_name"], limit)
    obj = _read_hdf5_object_positions(_as_path(positive["path"]), positive["demo_name"], pair["positive_instruction"], limit)
    eef_positions = eef["positions"] if eef.get("available") else None
    object_positions = obj["positions"] if obj.get("available") else None
    anchors = extract_event_anchors(actions, eef_positions, object_positions)
    perturbations = build_phase_perturbations(
        actions,
        anchors,
        offset_steps=offset_steps,
        stretch_factor=stretch_factor,
        compression_factor=compression_factor,
    )
    return {
        "pair_id": pair["pair_id"],
        "suite": pair.get("suite") or "libero_10",
        "task_id": pair["positive_task_id"],
        "instruction": pair["positive_instruction"],
        "counterfactual_task_id": pair.get("counterfactual_task_id"),
        "counterfactual_instruction": pair.get("counterfactual_instruction"),
        "positive_demo_path": positive["path"],
        "demo_name": positive["demo_name"],
        "init_state": positive["init_state"],
        "actions": actions,
        "hdf5_eef_positions": eef_positions,
        "hdf5_object_positions": object_positions,
        "hdf5_eef_source": {"available": bool(eef.get("available")), "key": eef.get("key"), "obs_keys": eef.get("obs_keys") or []},
        "hdf5_object_source": {
            "available": bool(obj.get("available")),
            "key": obj.get("key"),
            "audit": obj.get("audit"),
            "obs_keys": obj.get("obs_keys") or [],
        },
        "event_anchors": anchors,
        "perturbation_parameters": {
            "offset_steps": int(offset_steps),
            "stretch_factor": float(stretch_factor),
            "compression_factor": float(compression_factor),
        },
        "perturbations": perturbations,
        "target_horizon": int(actions.shape[0]),
        "max_steps_cap": int(max_steps_cap),
        "post_signal_margin": int(post_signal_margin),
        "hdf5_metadata": {
            "full_action_steps": positive["full_action_steps"],
            "num_samples_attr": positive["num_samples_attr"],
            "first_positive_reward_index": positive["first_reward_index"],
            "first_done_index": positive["first_done_index"],
            "first_signal_index": positive["first_signal_index"],
            "target_horizon": positive["target_horizon"],
            "states0_l2_to_init_state": positive["states0_l2_to_init_state"],
            "model_file_available": positive["model_file_available"],
            "hdf5_object_pose_keys_available": positive["first_obs"].get("object_position_keys", []),
        },
    }


def _inspect_start_pose(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    instruction: str,
    seed: int,
) -> dict[str, Any]:
    env = None
    summary = {
        "env_created": False,
        "reset_ok": False,
        "set_init_state_used": False,
        "set_init_state_ok": False,
        "sim_state_l2_to_hdf5_init": None,
        "target_audit": None,
        "target_key": None,
        "target_pos": None,
        "eef_pos": None,
        "object_position_keys_available": False,
        "error": None,
    }
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        env.seed(seed)
        obs = env.reset()
        summary["reset_ok"] = True
        summary["set_init_state_used"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["sim_state_l2_to_hdf5_init"] = _safe_l2(_sim_state_array(env), init_state)
        audit = _target_audit(obs, instruction)
        summary["target_audit"] = audit
        summary["target_key"] = audit.get("best_key")
        summary["target_pos"] = _extract_pos(obs, audit.get("best_key"))
        summary["eef_pos"] = _extract_eef(obs)
        summary["object_position_keys_available"] = bool(_object_position_keys(obs))
    except Exception as exc:  # noqa: BLE001
        summary["error"] = _compact(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return summary


def _action_for_dynamic(
    *,
    baseline: str,
    case: dict[str, Any],
    obs: Any,
    target_key: str | None,
    demo_object: list[float] | None,
    start_object: list[float] | None,
    previous_index: int | None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    actions = np.asarray(case["actions"], dtype=np.float64)
    if baseline == "nearest_progress_demo":
        index = _nearest_progress_index(
            case=case,
            obs=obs,
            target_key=target_key,
            demo_object=demo_object,
            previous_index=previous_index,
        )
        return actions[index].copy(), index, {"phase": "nearest_progress", "selected_index": index}
    index, trace = select_event_locked_index(
        case=case,
        obs=obs,
        target_key=target_key,
        demo_object=demo_object,
        start_object=start_object,
        previous_index=previous_index,
    )
    return actions[index].copy(), index, trace


def _event_timing_error(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("approach_index", "gripper_close_index", "object_motion_onset_index", "lift_index", "place_or_contact_index")
    errors = {}
    values = []
    for key in keys:
        ref = reference.get(key)
        cand = candidate.get(key)
        error = None if ref is None or cand is None else abs(int(ref) - int(cand))
        errors[key] = {"reference": ref, "candidate": cand, "absolute_error": error}
        if error is not None:
            values.append(float(error))
    return {
        "per_event": errors,
        "mean_absolute_error": None if not values else _round(float(np.mean(values)), 6),
        "max_absolute_error": None if not values else _round(float(np.max(values)), 6),
    }


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    case: dict[str, Any],
    perturbation_name: str,
    baseline: str,
    static_actions: np.ndarray | None,
    exact_start: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    expert_actions = np.asarray(case["actions"], dtype=np.float64)
    steps = int(expert_actions.shape[0])
    claim_role = "event_locked_retiming_method" if baseline == "event_locked_retiming" else "simple_baseline"
    if baseline == "raw_perturbed_replay":
        claim_role = "raw_perturbation_control"
    summary: dict[str, Any] = {
        "perturbation": perturbation_name,
        "baseline": baseline,
        "claim_role": claim_role,
        "init_mode": "exact",
        "env_created": False,
        "reset_ok": False,
        "set_init_state_used": False,
        "set_init_state_ok": False,
        "steps_requested": steps,
        "steps_performed": 0,
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "final_success": None,
        "done_seen": False,
        "first_positive_reward_index": None,
        "first_done_index": None,
        "first_success_index": None,
        "target_audit": None,
        "target_key": None,
        "start_target_pos": None,
        "final_target_pos": None,
        "target_object_movement_l2": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "eef_object_distance": None,
        "trajectory_drift": {"available": False, "mean_l2": None, "final_l2": None, "sample_count": 0},
        "event_timing_error": None,
        "action_metrics_vs_expert": None,
        "controller_valid_action_rate": None,
        "clip_rate_element": None,
        "clip_rate_step": None,
        "gripper_timing_error": None,
        "executed_action_stats": None,
        "action_index_trace_first_8": [],
        "phase_trace_first_8": [],
        "controller": None,
        "after_set_state_l2_to_hdf5_init": None,
        "error": None,
    }
    env = None
    executed_raw_actions: list[np.ndarray] = []
    executed_env_actions: list[np.ndarray] = []
    observed_eef: list[list[float]] = []
    observed_object: list[list[float]] = []
    previous_index: int | None = None
    obs: Any = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        summary["controller"] = _controller_summary(env)
        env.seed(seed)
        obs = env.reset()
        summary["reset_ok"] = True
        summary["set_init_state_used"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["after_set_state_l2_to_hdf5_init"] = _safe_l2(_sim_state_array(env), init_state)
        audit = _target_audit(obs, case["instruction"])
        target_key = audit.get("best_key")
        target_start = _extract_pos(obs, target_key)
        eef_start = _extract_eef(obs)
        summary["target_audit"] = audit
        summary["target_key"] = target_key
        summary["start_target_pos"] = target_start
        summary["eef_start"] = eef_start
        if eef_start is not None:
            observed_eef.append(eef_start)
        if target_start is not None:
            observed_object.append(target_start)
        start_distance = _distance(eef_start, target_start)
        demo_object = exact_start.get("target_pos")
        for step in range(steps):
            if static_actions is not None:
                action_raw = np.asarray(static_actions[step], dtype=np.float64).copy()
                selected_index = step
                trace = {"phase": "static_sequence", "selected_index": selected_index}
            else:
                action_raw, selected_index, trace = _action_for_dynamic(
                    baseline=baseline,
                    case=case,
                    obs=obs,
                    target_key=target_key,
                    demo_object=demo_object,
                    start_object=target_start,
                    previous_index=previous_index,
                )
                previous_index = selected_index
            if len(summary["action_index_trace_first_8"]) < 8:
                summary["action_index_trace_first_8"].append(int(selected_index))
                summary["phase_trace_first_8"].append(trace)
            env_action = np.clip(np.asarray(action_raw, dtype=np.float64), -1.0, 1.0)
            executed_raw_actions.append(np.asarray(action_raw, dtype=np.float64))
            executed_env_actions.append(env_action)
            obs, reward, done, _info = env.step(env_action)
            reward_value = float(reward)
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            current_eef = _extract_eef(obs)
            current_obj = _extract_pos(obs, target_key)
            if current_eef is not None:
                observed_eef.append(current_eef)
            if current_obj is not None:
                observed_object.append(current_obj)
            try:
                success_value = bool(env.check_success())
            except Exception:
                success_value = None
            if reward_value > 0.0 and summary["first_positive_reward_index"] is None:
                summary["first_positive_reward_index"] = int(step)
            if bool(done):
                summary["done_seen"] = True
                if summary["first_done_index"] is None:
                    summary["first_done_index"] = int(step)
            if success_value and summary["first_success_index"] is None:
                summary["first_success_index"] = int(step)
            if bool(done) or reward_value > 0.0 or success_value:
                break
        eef_final = _extract_eef(obs)
        target_final = _extract_pos(obs, target_key)
        summary["eef_final"] = eef_final
        summary["final_target_pos"] = target_final
        summary["eef_displacement_l2"] = _round(_distance(eef_start, eef_final), 6)
        summary["target_object_movement_l2"] = _round(_distance(target_start, target_final), 6)
        final_distance = _distance(eef_final, target_final)
        summary["eef_object_distance"] = {
            "available": start_distance is not None and final_distance is not None,
            "start": start_distance,
            "final": final_distance,
            "change": None if start_distance is None or final_distance is None else _round(float(final_distance) - float(start_distance), 6),
        }
        summary["trajectory_drift"] = _trajectory_drift(observed_eef, case.get("hdf5_eef_positions"))
        observed_obj_arr = np.asarray(observed_object, dtype=np.float64) if observed_object else None
        observed_eef_arr = np.asarray(observed_eef, dtype=np.float64) if observed_eef else None
        candidate_anchors = extract_event_anchors(
            np.asarray(executed_env_actions, dtype=np.float64) if executed_env_actions else np.zeros((0, 7)),
            observed_eef_arr,
            observed_obj_arr,
        )
        summary["event_timing_error"] = _event_timing_error(case.get("event_anchors") or {}, candidate_anchors)
        try:
            summary["final_success"] = bool(env.check_success())
        except Exception:
            summary["final_success"] = None
    except Exception as exc:  # noqa: BLE001
        summary["error"] = _compact(f"{type(exc).__name__}: {exc}")
        summary["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    if executed_raw_actions:
        raw_executed = np.asarray(executed_raw_actions, dtype=np.float64)
        env_executed = np.asarray(executed_env_actions, dtype=np.float64)
        reference = expert_actions[: raw_executed.shape[0]]
        summary["action_metrics_vs_expert"] = action_metrics(reference, env_executed)
        summary.update(_clip_stats(raw_executed))
        summary["gripper_timing_error"] = _gripper_timing_error(reference, env_executed)
        summary["executed_action_stats"] = _action_stats(env_executed)
    else:
        summary.update(_clip_stats(np.zeros((0, 7), dtype=np.float64)))
        summary["gripper_timing_error"] = {"reference_first_nonnegative_index": None, "candidate_first_nonnegative_index": None, "absolute_error": None}
    summary["passed"] = bool(
        summary["env_created"]
        and summary["reset_ok"]
        and summary["set_init_state_ok"]
        and summary["steps_performed"] > 0
        and summary["error"] is None
    )
    return summary


def _run_exact_expert(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    case: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return _run_variant(
        env_cls=env_cls,
        bddl_file=bddl_file,
        camera_size=camera_size,
        init_state=init_state,
        case=case,
        perturbation_name="none",
        baseline="hdf5_expert_replay_exact_init",
        static_actions=np.asarray(case["actions"], dtype=np.float64),
        exact_start={"target_pos": None},
        seed=seed,
    )


def _progress_tuple(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    success = 1.0 if _success(result) else 0.0
    reward = float(result.get("reward_sum") or 0.0)
    done = result.get("first_done_index")
    done_score = 0.0 if done is None else -float(done)
    dist_change = ((result.get("eef_object_distance") or {}).get("change"))
    approach = -float(dist_change) if isinstance(dist_change, (int, float)) else 0.0
    movement = float(result.get("target_object_movement_l2") or 0.0)
    return success, reward, done_score, approach, movement


def _beats(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-6) -> bool:
    left_tuple = _progress_tuple(left)
    right_tuple = _progress_tuple(right)
    for lval, rval in zip(left_tuple, right_tuple):
        if lval > rval + eps:
            return True
        if lval + eps < rval:
            return False
    return False


def _matches_or_beats(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-6) -> bool:
    left_tuple = _progress_tuple(left)
    right_tuple = _progress_tuple(right)
    return all(lval + eps >= rval for lval, rval in zip(left_tuple, right_tuple))


def _degraded(raw: dict[str, Any], exact: dict[str, Any]) -> bool:
    if _success(exact) and not _success(raw):
        return True
    exact_done = exact.get("first_done_index")
    raw_done = raw.get("first_done_index")
    if exact_done is not None and raw_done is not None and int(raw_done) > int(exact_done):
        return True
    if exact_done is not None and raw_done is None:
        return True
    exact_reward = float(exact.get("reward_sum") or 0.0)
    raw_reward = float(raw.get("reward_sum") or 0.0)
    if exact_reward > raw_reward:
        return True
    exact_progress = _progress_tuple(exact)[3]
    raw_progress = _progress_tuple(raw)[3]
    return bool(exact_progress > raw_progress + 1e-6)


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    case = (report.get("cases") or [{}])[0]
    exact = case.get("exact_expert_replay") or {}
    perturbation_summaries = []
    degraded_count = 0
    event_beats_best_count = 0
    event_recovers_count = 0
    simple_matches_event_count = 0
    event_progress_only_count = 0
    for perturbation in case.get("perturbations", []):
        variants = {item["baseline"]: item for item in perturbation.get("variants", [])}
        raw = variants.get("raw_perturbed_replay", {})
        event = variants.get("event_locked_retiming", {})
        simple = [payload for name, payload in variants.items() if name != "event_locked_retiming"]
        simple = [payload for payload in simple if payload.get("baseline") != "raw_perturbed_replay"]
        best_simple = max(simple, key=_progress_tuple) if simple else {}
        degraded = _degraded(raw, exact) if raw else False
        event_beats_best = bool(event and best_simple and _beats(event, best_simple))
        simple_matches_event = bool(event and best_simple and _matches_or_beats(best_simple, event))
        event_recovers = bool(event and raw and _beats(event, raw))
        event_success = _success(event)
        event_progress_only = bool(event_recovers and not event_success and not event_beats_best)
        degraded_count += int(degraded)
        event_beats_best_count += int(event_beats_best)
        simple_matches_event_count += int(simple_matches_event)
        event_recovers_count += int(event_recovers)
        event_progress_only_count += int(event_progress_only)
        perturbation_summaries.append(
            {
                "perturbation": perturbation.get("name"),
                "raw_degraded": degraded,
                "raw_success": _success(raw),
                "event_locked_success": _success(event),
                "event_locked_beats_raw": event_recovers,
                "event_locked_beats_best_simple": event_beats_best,
                "best_simple_baseline": best_simple.get("baseline"),
                "best_simple_success": _success(best_simple),
                "best_simple_matches_or_beats_event_locked": simple_matches_event,
                "raw_progress_tuple": _progress_tuple(raw) if raw else None,
                "event_progress_tuple": _progress_tuple(event) if event else None,
                "best_simple_progress_tuple": _progress_tuple(best_simple) if best_simple else None,
            }
        )
    exact_success = _success(exact)
    replay_metric = bool(report.get("policy", {}).get("replay_or_rollout_performed"))
    if not replay_metric:
        decision = "kill"
        reason = "No real replay/control metric appeared."
    elif not exact_success:
        decision = "kill"
        reason = "Exact-init expert replay did not succeed, so the action bridge is not a clean upper bound."
    elif degraded_count == 0:
        decision = "kill"
        reason = "Phase perturbations did not degrade exact-init replay."
    elif event_recovers_count == 0:
        decision = "kill"
        reason = "Event-locked retiming did not improve replay/progress over raw perturbed replay."
    elif simple_matches_event_count > 0:
        decision = "kill"
        reason = "A simple timing/progress baseline matched or beat event-locked retiming."
    elif event_beats_best_count == 0:
        decision = "kill"
        reason = "Event-locked retiming did not beat the best simple baseline on a meaningful replay metric."
    else:
        decision = "continue"
        reason = "Event-locked retiming recovered degraded replay and beat the best simple baseline on at least one meaningful metric."
    return {
        "continue_or_kill": decision,
        "reason": reason,
        "next_state": "STATE 2: broaden phase-locked retiming across tasks/demos" if decision == "continue" else "archive_or_reframe_phase_locked_retiming",
        "exact_init_expert_replay_success": exact_success,
        "phase_perturbations_tested": len(case.get("perturbations", [])),
        "baselines_tested_per_perturbation": list(ALL_BASELINES),
        "phase_perturbations_degraded_replay_count": degraded_count,
        "event_locked_recovered_over_raw_count": event_recovers_count,
        "event_locked_beats_best_simple_count": event_beats_best_count,
        "simple_baseline_matches_or_beats_event_locked_count": simple_matches_event_count,
        "event_progress_only_count": event_progress_only_count,
        "perturbation_summaries": perturbation_summaries,
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_phase_locked_retiming": True,
        "task_local_gate_required": f"{TASK_GATE}=1",
        "task_local_gate_set": os.environ.get(TASK_GATE) == "1",
        "downloads_performed": False,
        "installs_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "lora_training_performed": False,
        "loss_computed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "learned_policy_inference_performed": False,
        "simulator_environment_created": False,
        "replay_or_rollout_performed": False,
        "diagnostic_rollouts_performed": False,
        "benchmark_rollouts_performed": False,
        "multi_seed_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
        "forbidden_gates_set": forbidden,
    }


def _serializable_case_header(case: dict[str, Any], bddl_file: Path, exact_start: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": case["pair_id"],
        "task_id": case["task_id"],
        "instruction": case["instruction"],
        "counterfactual_task_id": case["counterfactual_task_id"],
        "counterfactual_instruction": case["counterfactual_instruction"],
        "positive_demo_path": case["positive_demo_path"],
        "demo_name": case["demo_name"],
        "bddl_file": str(bddl_file),
        "target_horizon": case["target_horizon"],
        "hdf5_metadata": case["hdf5_metadata"],
        "hdf5_eef_source": case["hdf5_eef_source"],
        "hdf5_object_source": case["hdf5_object_source"],
        "event_anchors": case["event_anchors"],
        "perturbation_parameters": case["perturbation_parameters"],
        "exact_start": exact_start,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary") or {}
    case = (report.get("cases") or [{}])[0]
    lines = [
        "# Phase-Locked Retiming STATE 1 Result",
        "",
        "Bounded replay/control diagnostic only. This is not benchmark success, paper-grade evidence, or a policy rollout claim.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- reason: {summary.get('reason')}",
        f"- replay happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- loss computed: `{report.get('policy', {}).get('loss_computed')}`",
        f"- GPU/download/OpenVLA-OFT: `{report.get('policy', {}).get('gpu_jobs_performed')}` / `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}`",
        f"- demos/tasks: `1 / 1`",
        f"- perturbations tested: `{summary.get('phase_perturbations_tested')}`",
        f"- baselines tested: `{', '.join(summary.get('baselines_tested_per_perturbation') or [])}`",
        f"- phase mismatch degraded replay count: `{summary.get('phase_perturbations_degraded_replay_count')}`",
        f"- event-locked beats best simple count: `{summary.get('event_locked_beats_best_simple_count')}`",
        f"- simple baseline matches/beats event count: `{summary.get('simple_baseline_matches_or_beats_event_locked_count')}`",
        f"- next state: `{summary.get('next_state')}`",
        "",
        "## Case",
        "",
        f"- task: `{case.get('task_id')}`",
        f"- instruction: {case.get('instruction')}",
        f"- selected horizon: `{case.get('target_horizon')}`",
        f"- HDF5 first reward/done/signal: `{(case.get('hdf5_metadata') or {}).get('first_positive_reward_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_done_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_signal_index')}`",
        f"- HDF5 EEF source: `{(case.get('hdf5_eef_source') or {}).get('key')}`",
        f"- HDF5 object source: `{(case.get('hdf5_object_source') or {}).get('key')}`",
        f"- event anchors: `{case.get('event_anchors')}`",
        "",
        "## Exact Expert Replay",
        "",
        "| reward | success | first done | steps | dist change | object move | traj drift |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    exact = case.get("exact_expert_replay") or {}
    dist = exact.get("eef_object_distance") or {}
    drift = exact.get("trajectory_drift") or {}
    lines.append(
        "| "
        + " | ".join(
            [
                _md(exact.get("reward_sum")),
                _md(_success(exact)),
                _md(exact.get("first_done_index")),
                _md(exact.get("steps_performed")),
                _md(dist.get("change")),
                _md(exact.get("target_object_movement_l2")),
                _md(drift.get("mean_l2")),
            ]
        )
        + " |"
    )
    lines.extend(
        [
            "",
            "## Perturbation Summary",
            "",
            "| perturbation | raw degraded | raw success | event success | best simple | best simple success | event beats best simple | simple matches/beats event |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary.get("perturbation_summaries") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("perturbation")),
                    _md(item.get("raw_degraded")),
                    _md(item.get("raw_success")),
                    _md(item.get("event_locked_success")),
                    str(item.get("best_simple_baseline")),
                    _md(item.get("best_simple_success")),
                    _md(item.get("event_locked_beats_best_simple")),
                    _md(item.get("best_simple_matches_or_beats_event_locked")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Replay Metrics",
            "",
            "| perturbation | baseline | reward | success | first done | steps | dist change | object move | event err | grip err | traj drift | clip step |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for perturbation in case.get("perturbations", []):
        for item in perturbation.get("variants", []):
            dist = item.get("eef_object_distance") or {}
            drift = item.get("trajectory_drift") or {}
            grip = item.get("gripper_timing_error") or {}
            event_err = item.get("event_timing_error") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("perturbation")),
                        str(item.get("baseline")),
                        _md(item.get("reward_sum")),
                        _md(_success(item)),
                        _md(item.get("first_done_index")),
                        _md(item.get("steps_performed")),
                        _md(dist.get("change")),
                        _md(item.get("target_object_movement_l2")),
                        _md(event_err.get("mean_absolute_error")),
                        _md(grip.get("absolute_error")),
                        _md(drift.get("mean_l2")),
                        _md(item.get("clip_rate_step")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Non-Leakage Notes",
            "",
            "- Target object key is resolved from natural-language instruction text plus visible observation object keys.",
            "- Event-locked retiming uses the demonstration chunk being retimed and current observation progress; it does not use reward labels or success labels for action selection.",
            "- Nearest-progress demo is reported as a strong simple baseline, not as method novelty.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    readiness_path = _as_path(args.readiness_report)
    readiness = _load_json(readiness_path) if readiness_path.exists() else {}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "phase_locked_retiming_state1",
        "policy": _policy(forbidden),
        "inputs": vars(args).copy(),
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "summary": {},
        "result": {"passed": False, "blocked_reason": None, "total_steps_performed": 0, "variant_count": 0},
        "elapsed_seconds": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden gates set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for bounded phase-locked retiming replay")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("rollout readiness gate is not green/authorized")
    if args.max_steps_cap < 1 or args.max_steps_cap > 320:
        stop_reasons.append("max_steps_cap must be between 1 and 320")
    if args.post_signal_margin < 0 or args.post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if args.camera_size < 16 or args.camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    if args.offset_steps < 1 or args.offset_steps > 60:
        stop_reasons.append("offset_steps must be between 1 and 60")
    if stop_reasons:
        report["result"]["blocked_reason"] = "; ".join(stop_reasons)
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_phase_locked_retiming_blocker"}
        report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
        return report
    try:
        case = build_phase_locked_case(
            _as_path(args.manifest),
            max_steps_cap=args.max_steps_cap,
            post_signal_margin=args.post_signal_margin,
            offset_steps=args.offset_steps,
            stretch_factor=args.stretch_factor,
            compression_factor=args.compression_factor,
        )
        env_cls = _load_env_class(_as_path(args.libero_root), _as_path(args.robosuite_root))
        bddl_file = _as_path(args.libero_root) / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
        exact_start = _inspect_start_pose(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=args.camera_size,
            init_state=case["init_state"],
            instruction=case["instruction"],
            seed=args.seed,
        )
        case_summary = _serializable_case_header(case, bddl_file, exact_start)
        exact_expert = _run_exact_expert(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=args.camera_size,
            init_state=case["init_state"],
            case=case,
            seed=args.seed,
        )
        case_summary["exact_expert_replay"] = exact_expert
        case_summary["perturbations"] = []
        total_steps = int(exact_expert.get("steps_performed") or 0)
        variant_count = 1
        for perturbation_name in PERTURBATION_NAMES:
            perturbation = case["perturbations"][perturbation_name]
            perturbed_actions = np.asarray(perturbation["actions"], dtype=np.float64)
            pert_summary = {
                "name": perturbation_name,
                "family": perturbation.get("family"),
                "description": perturbation.get("description"),
                "offset_steps": perturbation.get("offset_steps"),
                "factor": perturbation.get("factor"),
                "anchor": perturbation.get("anchor"),
                "anchor_index": perturbation.get("anchor_index"),
                "variants": [],
            }
            for baseline in ALL_BASELINES:
                static_actions = _baseline_static_actions(
                    baseline,
                    np.asarray(case["actions"], dtype=np.float64),
                    perturbed_actions,
                    perturbation,
                    global_scale=args.global_scale,
                )
                result = _run_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=args.camera_size,
                    init_state=case["init_state"],
                    case=case,
                    perturbation_name=perturbation_name,
                    baseline=baseline,
                    static_actions=static_actions,
                    exact_start=exact_start,
                    seed=args.seed,
                )
                pert_summary["variants"].append(result)
                total_steps += int(result.get("steps_performed") or 0)
                variant_count += 1
            case_summary["perturbations"].append(pert_summary)
        report["cases"].append(case_summary)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["replay_or_rollout_performed"] = total_steps > 0
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = variant_count
        all_variants = [case_summary["exact_expert_replay"]]
        for perturbation in case_summary["perturbations"]:
            all_variants.extend(perturbation.get("variants", []))
        report["result"]["passed"] = all(item.get("passed") for item in all_variants)
        report["summary"] = _summarize(report)
    except Exception as exc:  # noqa: BLE001
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_phase_locked_retiming_blocker"}
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-steps-cap", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--offset-steps", type=int, default=18)
    parser.add_argument("--stretch-factor", type=float, default=1.15)
    parser.add_argument("--compression-factor", type=float, default=0.85)
    parser.add_argument("--global-scale", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="reports/phase_locked_retiming_state1_result.json")
    parser.add_argument("--report-md", default="reports/phase_locked_retiming_state1_result.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    console = {
        "result": report.get("result"),
        "summary": {
            key: value
            for key, value in (report.get("summary") or {}).items()
            if key != "perturbation_summaries"
        },
        "report_json": str(report_json),
        "replay_or_rollout_performed": report.get("policy", {}).get("replay_or_rollout_performed"),
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] or os.environ.get(TASK_GATE) != "1" else 1


if __name__ == "__main__":
    sys.exit(main())
