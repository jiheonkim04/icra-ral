"""STATE 1 smoke for ContactTube-Aug.

This runner extracts contact-tube structure from local LIBERO HDF5 expert
demonstrations, builds small action-level augmentation baselines, and can run a
bounded exact/default-reset replay smoke behind a task-local gate. It performs
no training, model loading, VLA inference, GPU work, downloads, OpenVLA-OFT
execution, benchmark sweep, or paper-grade claim.
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


SCHEMA_VERSION = "2026-07-08.contacttube_aug_state1.v1"
TASK_GATE = "ALLOW_CONTACTTUBE_AUG_STATE1"
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
    "ALLOW_PHASE_LOCKED_RETIMING",
    "ALLOW_TL_CHUNKREPAIR_STATE1",
)

BASELINE_VARIANTS = (
    "raw_demo_replay",
    "random_pose_jitter",
    "simple_object_relative_translation_retarget",
    "random_action_jitter",
)
METHOD_VARIANT = "contacttube_aug"
REPLAY_VARIANTS = ("exact_init_noop_upper_bound", *BASELINE_VARIANTS, METHOD_VARIANT)


def _round(value: float | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _md(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return str(_round(value, 6))
    return str(value)


def _norm(values: Any) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return float(np.linalg.norm(arr))


def _success(result: dict[str, Any]) -> bool:
    return bool(result.get("final_success") or result.get("done_seen") or float(result.get("reward_sum") or 0.0) > 0.0)


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    for index, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


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


def _first_gripper_nonnegative(actions: np.ndarray) -> int | None:
    if actions.ndim != 2 or actions.shape[1] < 7:
        return None
    for index, value in enumerate(actions[:, 6]):
        if float(value) >= 0.0:
            return int(index)
    return None


def _release_transitions(actions: np.ndarray) -> list[int]:
    if actions.ndim != 2 or actions.shape[1] < 7:
        return []
    grip = np.asarray(actions[:, 6], dtype=np.float64).reshape(-1)
    return [int(index) for index in range(1, grip.size) if float(grip[index - 1]) >= 0.0 and float(grip[index]) < 0.0]


def _close_transitions(actions: np.ndarray) -> list[int]:
    if actions.ndim != 2 or actions.shape[1] < 7:
        return []
    grip = np.asarray(actions[:, 6], dtype=np.float64).reshape(-1)
    return [int(index) for index in range(1, grip.size) if float(grip[index - 1]) < 0.0 and float(grip[index]) >= 0.0]


def _gripper_timing_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = _first_gripper_nonnegative(reference)
    cand = _first_gripper_nonnegative(candidate)
    ref_releases = _release_transitions(reference)
    cand_releases = _release_transitions(candidate)
    release_error = None
    if ref_releases and cand_releases:
        release_error = abs(int(ref_releases[-1]) - int(cand_releases[-1]))
    return {
        "reference_first_nonnegative_index": ref,
        "candidate_first_nonnegative_index": cand,
        "close_absolute_error": None if ref is None or cand is None else abs(int(ref) - int(cand)),
        "reference_release_index": ref_releases[-1] if ref_releases else None,
        "candidate_release_index": cand_releases[-1] if cand_releases else None,
        "release_absolute_error": release_error,
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


def _estimate_translation_unit(actions: np.ndarray, eef_positions: np.ndarray | None) -> dict[str, Any]:
    if eef_positions is None or eef_positions.shape[0] < 2 or actions.shape[0] < 2:
        return {"available": False, "meters_per_action_unit": 0.01, "source": "fallback_no_hdf5_eef_delta"}
    steps = min(actions.shape[0] - 1, eef_positions.shape[0] - 1)
    action_norm = np.linalg.norm(np.asarray(actions[:steps, :3], dtype=np.float64), axis=1)
    eef_delta = np.linalg.norm(np.diff(np.asarray(eef_positions[: steps + 1], dtype=np.float64), axis=0), axis=1)
    mask = (action_norm > 1e-6) & np.isfinite(action_norm) & np.isfinite(eef_delta)
    ratios = eef_delta[mask] / np.maximum(action_norm[mask], 1e-9)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if ratios.size == 0:
        return {"available": False, "meters_per_action_unit": 0.01, "source": "fallback_degenerate_action_eef_delta"}
    value = float(np.median(ratios))
    value = float(np.clip(value, 0.002, 0.08))
    return {
        "available": True,
        "meters_per_action_unit": _round(value, 9),
        "source": "median_hdf5_eef_delta_norm_per_translation_action_norm",
        "sample_count": int(ratios.size),
        "raw_ratio_range": _range(ratios),
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


def _object_motion_onset(object_positions: np.ndarray | None, threshold: float = 0.005) -> int | None:
    if object_positions is None or object_positions.ndim != 2 or object_positions.shape[0] < 2:
        return None
    start = np.asarray(object_positions[0, :3], dtype=np.float64)
    deltas = np.linalg.norm(np.asarray(object_positions[:, :3], dtype=np.float64) - start.reshape(1, 3), axis=1)
    return _first_index(deltas, threshold)


def _lift_index(
    *,
    object_positions: np.ndarray | None,
    eef_positions: np.ndarray | None,
    close_index: int | None,
    threshold: float = 0.02,
) -> int | None:
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


def _place_index(object_positions: np.ndarray | None, lift_index: int | None, release_index: int | None) -> int | None:
    if object_positions is not None and lift_index is not None and object_positions.ndim == 2 and object_positions.shape[0] > int(lift_index) + 2:
        obj = np.asarray(object_positions, dtype=np.float64)
        z = obj[:, 2]
        peak = int(lift_index) + int(np.argmax(z[int(lift_index) :]))
        if peak + 1 < obj.shape[0]:
            down = _first_index(float(z[peak]) - z[peak:], 0.01)
            if down is not None:
                return peak + int(down)
    return release_index


def extract_contact_tube(
    *,
    actions: np.ndarray,
    eef_positions: np.ndarray | None,
    object_positions: np.ndarray | None,
    source: str,
) -> dict[str, Any]:
    """Extract a compact contact-tube summary from actions and state traces."""

    actions = np.asarray(actions, dtype=np.float64)
    horizon = int(actions.shape[0]) if actions.ndim == 2 else 0
    close = _first_gripper_nonnegative(actions)
    closes = _close_transitions(actions)
    releases = _release_transitions(actions)
    release = releases[-1] if releases else None
    eef_available = bool(eef_positions is not None and np.asarray(eef_positions).ndim == 2 and np.asarray(eef_positions).shape[1] >= 3)
    object_available = bool(object_positions is not None and np.asarray(object_positions).ndim == 2 and np.asarray(object_positions).shape[1] >= 3)
    distance_profile = None
    relative_profile = None
    contact_indices: list[int] = []
    proximity_threshold = None
    if eef_available and object_available:
        eef = np.asarray(eef_positions, dtype=np.float64)
        obj = np.asarray(object_positions, dtype=np.float64)
        count = min(eef.shape[0], obj.shape[0], max(1, horizon))
        if count > 0:
            relative_profile = eef[:count, :3] - obj[:count, :3]
            distance_profile = np.linalg.norm(relative_profile, axis=1)
            min_dist = float(np.min(distance_profile))
            proximity_threshold = min(0.08, max(min_dist + 0.025, 0.035))
            contact_indices = [int(index) for index, value in enumerate(distance_profile) if float(value) <= proximity_threshold]
    if not contact_indices and close is not None:
        stop = release if release is not None else min(horizon, int(close) + 40)
        contact_indices = list(range(int(close), max(int(close), int(stop)) + 1))
    motion = _object_motion_onset(object_positions)
    lift = _lift_index(object_positions=object_positions, eef_positions=eef_positions, close_index=close)
    place = _place_index(object_positions, lift, release)
    contact_window = {
        "available": bool(contact_indices),
        "start_index": min(contact_indices) if contact_indices else None,
        "end_index": max(contact_indices) if contact_indices else None,
        "index_count": len(contact_indices),
        "source": "eef_object_distance" if distance_profile is not None else ("gripper_surrogate" if contact_indices else "unavailable"),
        "proximity_threshold_m": _round(proximity_threshold, 6),
    }
    return {
        "source": source,
        "horizon": horizon,
        "eef_available": eef_available,
        "object_pose_available": object_available,
        "distance_profile_available": distance_profile is not None,
        "relative_profile_available": relative_profile is not None,
        "gripper_close_index": close,
        "close_transitions": closes,
        "release_index": release,
        "release_transitions": releases,
        "object_motion_onset_index": motion,
        "lift_index": lift,
        "place_or_release_index": place,
        "contact_window": contact_window,
        "distance_profile_summary": {
            "available": distance_profile is not None,
            "start": None if distance_profile is None else _round(float(distance_profile[0]), 6),
            "min": None if distance_profile is None else _round(float(np.min(distance_profile)), 6),
            "min_index": None if distance_profile is None else int(np.argmin(distance_profile)),
            "final": None if distance_profile is None else _round(float(distance_profile[-1]), 6),
            "mean": None if distance_profile is None else _round(float(np.mean(distance_profile)), 6),
        },
        "observable": bool(eef_available and (object_available or close is not None)),
        "_distance_profile": distance_profile,
        "_relative_profile": relative_profile,
    }


def _json_tube(tube: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tube.items() if not key.startswith("_")}


def _event_error(reference: dict[str, Any], candidate: dict[str, Any], key: str) -> dict[str, Any]:
    ref = reference.get(key)
    cand = candidate.get(key)
    return {"reference": ref, "candidate": cand, "absolute_error": None if ref is None or cand is None else abs(int(ref) - int(cand))}


def _contact_window_iou(reference: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    ref_window = reference.get("contact_window") or {}
    cand_window = candidate.get("contact_window") or {}
    if not ref_window.get("available") or not cand_window.get("available"):
        return None
    ref_start, ref_end = ref_window.get("start_index"), ref_window.get("end_index")
    cand_start, cand_end = cand_window.get("start_index"), cand_window.get("end_index")
    if ref_start is None or ref_end is None or cand_start is None or cand_end is None:
        return None
    ref_set = set(range(int(ref_start), int(ref_end) + 1))
    cand_set = set(range(int(cand_start), int(cand_end) + 1))
    if not ref_set or not cand_set:
        return None
    return _round(float(len(ref_set & cand_set) / len(ref_set | cand_set)), 6)


def _tube_preservation_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_dist = reference.get("_distance_profile")
    cand_dist = candidate.get("_distance_profile")
    distance_mae = None
    distance_max = None
    if ref_dist is not None and cand_dist is not None:
        count = min(np.asarray(ref_dist).shape[0], np.asarray(cand_dist).shape[0])
        if count > 0:
            diff = np.abs(np.asarray(ref_dist[:count], dtype=np.float64) - np.asarray(cand_dist[:count], dtype=np.float64))
            distance_mae = _round(float(np.mean(diff)), 6)
            distance_max = _round(float(np.max(diff)), 6)
    ref_rel = reference.get("_relative_profile")
    cand_rel = candidate.get("_relative_profile")
    relative_mae = None
    if ref_rel is not None and cand_rel is not None:
        count = min(np.asarray(ref_rel).shape[0], np.asarray(cand_rel).shape[0])
        if count > 0:
            diff = np.linalg.norm(np.asarray(ref_rel[:count], dtype=np.float64) - np.asarray(cand_rel[:count], dtype=np.float64), axis=1)
            relative_mae = _round(float(np.mean(diff)), 6)
    event_errors = {
        "object_motion_onset": _event_error(reference, candidate, "object_motion_onset_index"),
        "lift": _event_error(reference, candidate, "lift_index"),
        "place_or_release": _event_error(reference, candidate, "place_or_release_index"),
    }
    event_values = [
        float(payload["absolute_error"])
        for payload in event_errors.values()
        if payload.get("absolute_error") is not None
    ]
    score_parts = []
    if distance_mae is not None:
        score_parts.append(float(distance_mae))
    if relative_mae is not None:
        score_parts.append(float(relative_mae))
    if event_values:
        score_parts.append(float(np.mean(event_values)) * 0.002)
    return {
        "contact_tube_preservation_error": None if not score_parts else _round(float(np.sum(score_parts)), 6),
        "eef_object_distance_profile_mae": distance_mae,
        "eef_object_distance_profile_max": distance_max,
        "eef_object_relative_profile_mae": relative_mae,
        "contact_window_iou": _contact_window_iou(reference, candidate),
        "object_motion_onset_error": event_errors["object_motion_onset"],
        "lift_phase_error": event_errors["lift"],
        "place_phase_error": event_errors["place_or_release"],
    }


def _manifest_pair(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("counterfactual split manifest has no counterfactual_pairs")
    return pairs[0]


def _static_random_pose_actions(actions: np.ndarray, *, rng: np.random.Generator, meters_per_unit: float, pose_jitter_meters: float) -> tuple[np.ndarray, list[float]]:
    source = np.asarray(actions, dtype=np.float64)
    out = source.copy()
    jitter = rng.normal(0.0, float(pose_jitter_meters), size=3)
    jitter[2] *= 0.25
    ramp = max(1, min(24, source.shape[0]))
    if np.isfinite(meters_per_unit) and meters_per_unit > 0:
        out[:ramp, :3] += jitter.reshape(1, 3) / (float(meters_per_unit) * float(ramp))
    return out, [float(value) for value in jitter.tolist()]


def _static_random_action_jitter(actions: np.ndarray, *, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    source = np.asarray(actions, dtype=np.float64)
    noise = rng.normal(0.0, float(noise_std), size=source.shape)
    noise[:, 6] *= 0.15
    out = source + noise
    out[:, 6] = source[:, 6]
    return out


def build_contacttube_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 300,
    post_signal_margin: int = 20,
    seed: int = 0,
    noise_std: float = 0.025,
    pose_jitter_meters: float = 0.02,
) -> dict[str, Any]:
    pair = _manifest_pair(manifest_path)
    positive = _read_demo_full(_as_path(pair["positive_demo_file"]), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    actions = np.asarray(positive["actions"], dtype=np.float64)
    limit = int(actions.shape[0]) + 1
    eef = _read_hdf5_eef_positions(_as_path(positive["path"]), positive["demo_name"], limit)
    obj = _read_hdf5_object_positions(_as_path(positive["path"]), positive["demo_name"], pair["positive_instruction"], limit)
    eef_positions = eef["positions"] if eef.get("available") else None
    object_positions = obj["positions"] if obj.get("available") else None
    tube = extract_contact_tube(actions=actions, eef_positions=eef_positions, object_positions=object_positions, source="hdf5_demo")
    unit = _estimate_translation_unit(actions, eef_positions)
    rng = np.random.default_rng(int(seed))
    pose_actions, pose_delta = _static_random_pose_actions(
        actions,
        rng=rng,
        meters_per_unit=float(unit["meters_per_action_unit"]),
        pose_jitter_meters=pose_jitter_meters,
    )
    jitter_actions = _static_random_action_jitter(actions, rng=rng, noise_std=noise_std)
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
        "hdf5_contact_tube": tube,
        "translation_unit": unit,
        "target_horizon": int(actions.shape[0]),
        "max_steps_cap": int(max_steps_cap),
        "post_signal_margin": int(post_signal_margin),
        "augmentation_parameters": {
            "seed": int(seed),
            "noise_std": float(noise_std),
            "pose_jitter_meters": float(pose_jitter_meters),
            "random_pose_jitter_delta_m": pose_delta,
        },
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
        "static_variant_actions": {
            "exact_init_noop_upper_bound": actions,
            "raw_demo_replay": actions,
            "random_pose_jitter": pose_actions,
            "random_action_jitter": jitter_actions,
        },
        "dynamic_variants": {
            "simple_object_relative_translation_retarget": {
                "description": "Translate the HDF5 EEF path by the start-object delta only.",
                "uses_current_object_during_contact": False,
            },
            METHOD_VARIANT: {
                "description": "Preserve EEF-object relative tube during contact/proximity windows and copy gripper timing.",
                "uses_current_object_during_contact": True,
            },
        },
    }


def _desired_eef(
    *,
    variant: str,
    step: int,
    case: dict[str, Any],
    reference_trace: dict[str, Any],
    current_object: list[float] | None,
    object_delta: list[float] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    ref_eef = reference_trace.get("eef_positions") or []
    ref_obj = reference_trace.get("object_positions") or []
    if not ref_eef:
        return None, {"retarget_available": False, "reason": "reference_eef_trace_unavailable"}
    index = min(int(step) + 1, len(ref_eef) - 1)
    desired = np.asarray(ref_eef[index], dtype=np.float64).reshape(-1)[:3]
    if object_delta is not None:
        desired = desired + np.asarray(object_delta, dtype=np.float64).reshape(-1)[:3]
    phase = "free_space_shifted_eef_path"
    uses_current_object = False
    if variant == METHOD_VARIANT and current_object is not None and ref_obj and index < len(ref_obj):
        ref_relative = np.asarray(ref_eef[index], dtype=np.float64).reshape(-1)[:3] - np.asarray(ref_obj[index], dtype=np.float64).reshape(-1)[:3]
        tube = reference_trace.get("contact_tube") or {}
        window = tube.get("contact_window") or {}
        start, end = window.get("start_index"), window.get("end_index")
        in_contact_window = bool(start is not None and end is not None and int(start) <= int(step) <= int(end))
        if in_contact_window or (tube.get("object_motion_onset_index") is not None and int(step) >= int(tube["object_motion_onset_index"])):
            desired = np.asarray(current_object, dtype=np.float64).reshape(-1)[:3] + ref_relative
            phase = "contact_tube_relative_to_current_object"
            uses_current_object = True
    return desired, {
        "retarget_available": True,
        "phase": phase,
        "reference_index": int(index),
        "uses_current_object": uses_current_object,
    }


def _variant_action_for_step(
    *,
    variant: str,
    step: int,
    obs: Any,
    case: dict[str, Any],
    static_actions: np.ndarray | None,
    reference_trace: dict[str, Any],
    target_key: str | None,
    object_delta: list[float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_actions = np.asarray(case["actions"], dtype=np.float64)
    if static_actions is not None:
        return np.asarray(static_actions[step], dtype=np.float64).copy(), {"retarget_available": False, "phase": "static_action_replay"}
    raw = np.asarray(raw_actions[step], dtype=np.float64).copy()
    current_eef = _extract_eef(obs)
    current_object = _extract_pos(obs, target_key)
    desired, trace = _desired_eef(
        variant=variant,
        step=step,
        case=case,
        reference_trace=reference_trace,
        current_object=current_object,
        object_delta=object_delta,
    )
    if desired is not None and current_eef is not None:
        unit = float(case["translation_unit"]["meters_per_action_unit"])
        if np.isfinite(unit) and unit > 0:
            raw[:3] = (np.asarray(desired, dtype=np.float64).reshape(-1)[:3] - np.asarray(current_eef, dtype=np.float64).reshape(-1)[:3]) / unit
            trace["desired_eef"] = [float(value) for value in np.asarray(desired, dtype=np.float64).reshape(-1)[:3].tolist()]
            trace["current_eef"] = current_eef
            trace["current_object"] = current_object
    raw[6] = raw_actions[step, 6]
    return raw, trace


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    case: dict[str, Any],
    variant: str,
    init_mode: str,
    reference_trace: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    raw_actions = np.asarray(case["actions"], dtype=np.float64)
    static_actions = case["static_variant_actions"].get(variant)
    if static_actions is not None:
        static_actions = np.asarray(static_actions, dtype=np.float64)
    steps = int(raw_actions.shape[0])
    summary: dict[str, Any] = {
        "variant": variant,
        "claim_role": "noop_original_replay_upper_bound" if variant == "exact_init_noop_upper_bound" else ("contacttube_aug_method" if variant == METHOD_VARIANT else "simple_baseline"),
        "init_mode": init_mode,
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
        "object_delta_from_exact_start": None,
        "action_metrics_vs_raw_expert": None,
        "controller_valid_action_rate": None,
        "clip_rate_element": None,
        "clip_rate_step": None,
        "gripper_timing_error": None,
        "executed_action_stats": None,
        "contact_tube_metrics": None,
        "contact_tube": None,
        "retarget_trace_first_8": [],
        "controller": None,
        "after_set_state_l2_to_hdf5_init": None,
        "available_obs_keys": [],
        "object_position_keys_available": False,
        "error": None,
    }
    env = None
    obs: Any = None
    executed_raw_actions: list[np.ndarray] = []
    executed_env_actions: list[np.ndarray] = []
    observed_eef: list[list[float]] = []
    observed_object: list[list[float]] = []
    object_delta: list[float] | None = None
    reference_trace = reference_trace or {}
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        summary["controller"] = _controller_summary(env)
        env.seed(seed)
        obs = env.reset()
        summary["reset_ok"] = True
        if init_mode == "exact":
            summary["set_init_state_used"] = True
            obs = env.set_init_state(init_state)
            summary["set_init_state_ok"] = True
            summary["after_set_state_l2_to_hdf5_init"] = _safe_l2(_sim_state_array(env), init_state)
        if isinstance(obs, dict):
            keys = sorted(str(key) for key in obs.keys())
            summary["available_obs_keys"] = keys[:80]
            summary["object_position_keys_available"] = bool(_object_position_keys(obs))
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
        ref_start = reference_trace.get("start_target_pos")
        if target_start is not None and ref_start is not None:
            object_delta = [
                float(a - b)
                for a, b in zip(
                    np.asarray(target_start, dtype=np.float64).reshape(-1)[:3],
                    np.asarray(ref_start, dtype=np.float64).reshape(-1)[:3],
                )
            ]
            summary["object_delta_from_exact_start"] = object_delta
        start_distance = _distance(eef_start, target_start)
        for step in range(steps):
            action_raw, trace = _variant_action_for_step(
                variant=variant,
                step=step,
                obs=obs,
                case=case,
                static_actions=static_actions,
                reference_trace=reference_trace,
                target_key=target_key,
                object_delta=object_delta,
            )
            if len(summary["retarget_trace_first_8"]) < 8 and variant in case["dynamic_variants"]:
                compact_trace = {
                    "step": int(step),
                    "phase": trace.get("phase"),
                    "retarget_available": trace.get("retarget_available"),
                    "uses_current_object": trace.get("uses_current_object"),
                    "desired_eef": trace.get("desired_eef"),
                }
                summary["retarget_trace_first_8"].append(compact_trace)
            env_action = np.clip(np.asarray(action_raw, dtype=np.float64), -1.0, 1.0)
            executed_raw_actions.append(np.asarray(action_raw, dtype=np.float64))
            executed_env_actions.append(env_action)
            obs, reward, done, _info = env.step(env_action)
            reward_value = float(reward)
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            current_eef = _extract_eef(obs)
            current_object = _extract_pos(obs, target_key)
            if current_eef is not None:
                observed_eef.append(current_eef)
            if current_object is not None:
                observed_object.append(current_object)
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
            if success_value:
                if summary["first_success_index"] is None:
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
        reference = raw_actions[: raw_executed.shape[0]]
        summary["action_metrics_vs_raw_expert"] = action_metrics(reference, env_executed)
        summary.update(_clip_stats(raw_executed))
        summary["gripper_timing_error"] = _gripper_timing_error(reference, env_executed)
        summary["executed_action_stats"] = _action_stats(env_executed)
        tube = extract_contact_tube(
            actions=env_executed,
            eef_positions=np.asarray(observed_eef, dtype=np.float64) if observed_eef else None,
            object_positions=np.asarray(observed_object, dtype=np.float64) if observed_object else None,
            source=f"runtime_replay:{variant}",
        )
        summary["contact_tube"] = _json_tube(tube)
        ref_tube = reference_trace.get("contact_tube_full")
        if ref_tube is not None:
            summary["contact_tube_metrics"] = _tube_preservation_metrics(ref_tube, tube)
    else:
        summary.update(_clip_stats(np.zeros((0, 7), dtype=np.float64)))
        summary["gripper_timing_error"] = {
            "reference_first_nonnegative_index": None,
            "candidate_first_nonnegative_index": None,
            "close_absolute_error": None,
            "reference_release_index": None,
            "candidate_release_index": None,
            "release_absolute_error": None,
        }
    summary["trace"] = {
        "eef_positions": observed_eef,
        "object_positions": observed_object,
        "start_target_pos": summary.get("start_target_pos"),
        "contact_tube": summary.get("contact_tube"),
    }
    summary["passed"] = bool(
        summary["env_created"]
        and summary["reset_ok"]
        and (init_mode != "exact" or summary["set_init_state_ok"])
        and summary["steps_performed"] > 0
        and summary["error"] is None
    )
    return summary


def _progress_tuple(result: dict[str, Any]) -> tuple[float, float, float, float]:
    success = 1.0 if _success(result) else 0.0
    reward = float(result.get("reward_sum") or 0.0)
    dist_change = ((result.get("eef_object_distance") or {}).get("change"))
    approach = -float(dist_change) if isinstance(dist_change, (int, float)) else 0.0
    movement = float(result.get("target_object_movement_l2") or 0.0)
    return success, reward, approach, movement


def _lower_metric(result: dict[str, Any], key: str) -> float | None:
    metrics = result.get("contact_tube_metrics") or {}
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _tube_score(result: dict[str, Any]) -> float | None:
    return _lower_metric(result, "contact_tube_preservation_error")


def _beats_by_tube(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-6) -> bool:
    left_score = _tube_score(left)
    right_score = _tube_score(right)
    if left_score is None or right_score is None:
        return False
    if left_score + eps < right_score:
        return True
    if right_score + eps < left_score:
        return False
    left_clip = float(left.get("clip_rate_step") or 0.0)
    right_clip = float(right.get("clip_rate_step") or 0.0)
    if left_clip + eps < right_clip:
        return True
    return False


def _matches_or_beats(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-6) -> bool:
    left_score = _tube_score(left)
    right_score = _tube_score(right)
    if left_score is not None and right_score is not None:
        return left_score <= right_score + eps
    return _progress_tuple(left) >= _progress_tuple(right)


def _case_header(case: dict[str, Any], bddl_file: Path) -> dict[str, Any]:
    return {
        "pair_id": case["pair_id"],
        "task_id": case["task_id"],
        "instruction": case["instruction"],
        "counterfactual_task_id": case.get("counterfactual_task_id"),
        "counterfactual_instruction": case.get("counterfactual_instruction"),
        "positive_demo_path": case["positive_demo_path"],
        "demo_name": case["demo_name"],
        "bddl_file": str(bddl_file),
        "target_horizon": case["target_horizon"],
        "hdf5_metadata": case["hdf5_metadata"],
        "hdf5_eef_source": case["hdf5_eef_source"],
        "hdf5_object_source": case["hdf5_object_source"],
        "hdf5_contact_tube": _json_tube(case["hdf5_contact_tube"]),
        "translation_unit": case["translation_unit"],
        "augmentation_parameters": case["augmentation_parameters"],
        "variants": [],
    }


def _run_replay_case(args: argparse.Namespace, report: dict[str, Any]) -> None:
    case = build_contacttube_case(
        _as_path(args.manifest),
        max_steps_cap=args.max_steps_cap,
        post_signal_margin=args.post_signal_margin,
        seed=args.seed,
        noise_std=args.noise_std,
        pose_jitter_meters=args.pose_jitter_meters,
    )
    env_cls = _load_env_class(_as_path(args.libero_root), _as_path(args.robosuite_root))
    bddl_file = _as_path(args.libero_root) / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
    case_summary = _case_header(case, bddl_file)
    exact = _run_variant(
        env_cls=env_cls,
        bddl_file=bddl_file,
        camera_size=args.camera_size,
        init_state=case["init_state"],
        case=case,
        variant="exact_init_noop_upper_bound",
        init_mode="exact",
        reference_trace=None,
        seed=args.seed,
    )
    exact_trace = exact.pop("trace")
    exact_tube_full = extract_contact_tube(
        actions=np.asarray(case["actions"][: int(exact.get("steps_performed") or 0)], dtype=np.float64),
        eef_positions=np.asarray(exact_trace.get("eef_positions") or [], dtype=np.float64) if exact_trace.get("eef_positions") else None,
        object_positions=np.asarray(exact_trace.get("object_positions") or [], dtype=np.float64) if exact_trace.get("object_positions") else None,
        source="runtime_replay:exact_init_reference",
    )
    exact_trace["contact_tube_full"] = exact_tube_full
    exact_trace["contact_tube"] = _json_tube(exact_tube_full)
    exact["contact_tube"] = _json_tube(exact_tube_full)
    case_summary["variants"].append(exact)
    total_steps = int(exact.get("steps_performed") or 0)
    for name in (*BASELINE_VARIANTS, METHOD_VARIANT):
        result = _run_variant(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=args.camera_size,
            init_state=case["init_state"],
            case=case,
            variant=name,
            init_mode="default_reset",
            reference_trace=exact_trace,
            seed=args.seed,
        )
        result.pop("trace", None)
        case_summary["variants"].append(result)
        total_steps += int(result.get("steps_performed") or 0)
    report["cases"].append(case_summary)
    report["policy"]["simulator_environment_created"] = True
    report["policy"]["replay_or_rollout_performed"] = total_steps > 0
    report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
    report["result"]["total_steps_performed"] = total_steps
    report["result"]["variant_count"] = len(case_summary["variants"])


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("cases"):
        return {"continue_or_kill": "blocked", "reason": "no ContactTube-Aug case was built", "next_state": "resolve_contacttube_aug_blocker"}
    case = report["cases"][0]
    variants = {item["variant"]: item for item in case.get("variants", [])}
    exact = variants.get("exact_init_noop_upper_bound", {})
    method = variants.get(METHOD_VARIANT, {})
    simple = variants.get("simple_object_relative_translation_retarget", {})
    random_jitter = variants.get("random_action_jitter", {})
    random_pose = variants.get("random_pose_jitter", {})
    replay_metric = bool(report.get("policy", {}).get("replay_or_rollout_performed"))
    hdf5_tube = case.get("hdf5_contact_tube") or {}
    runtime_tube = exact.get("contact_tube") or {}
    object_state_available = bool(hdf5_tube.get("object_pose_available") or runtime_tube.get("object_pose_available"))
    contact_extractable = bool(hdf5_tube.get("observable") or runtime_tube.get("observable"))
    method_valid = bool(method.get("passed") and float(method.get("controller_valid_action_rate") or 0.0) >= 0.85)
    method_beats_random_jitter = bool(method and random_jitter and _beats_by_tube(method, random_jitter))
    method_beats_random_pose = bool(method and random_pose and _beats_by_tube(method, random_pose))
    method_beats_simple = bool(method and simple and _beats_by_tube(method, simple))
    simple_matches = bool(simple and method and _matches_or_beats(simple, method))
    exact_success = _success(exact)
    method_success = _success(method)
    variant_scores = {
        name: {
            "success": _success(payload),
            "progress_tuple": _progress_tuple(payload),
            "tube_score": _tube_score(payload),
            "controller_valid_action_rate": payload.get("controller_valid_action_rate"),
            "clip_rate_step": payload.get("clip_rate_step"),
            "reward_sum": payload.get("reward_sum"),
            "first_done_index": payload.get("first_done_index"),
        }
        for name, payload in variants.items()
    }
    if not replay_metric:
        decision = "blocked"
        reason = "No real replay/control metric was produced; set the bounded task gate after risk assessment."
    elif not contact_extractable:
        decision = "kill"
        reason = "Contact tube extraction was not reliable from HDF5 or replay traces."
    elif not object_state_available:
        decision = "kill"
        reason = "Object/contact state was unavailable for ContactTube-Aug validation."
    elif not exact_success:
        decision = "kill"
        reason = "No-op exact-init expert replay did not succeed, so augmentation replay validity cannot be judged."
    elif not method_valid:
        decision = "kill"
        reason = "ContactTube-Aug trajectory was not controller-valid/replay-valid."
    elif not method_beats_random_jitter:
        decision = "kill"
        reason = "Random action jitter matched or beat ContactTube-Aug on contact-tube preservation."
    elif not method_beats_random_pose:
        decision = "kill"
        reason = "Random pose jitter matched or beat ContactTube-Aug on contact-tube preservation."
    elif not method_beats_simple or simple_matches:
        decision = "kill"
        reason = "Simple object-relative translation retargeting matched or beat ContactTube-Aug."
    else:
        decision = "continue"
        reason = "ContactTube-Aug preserved contact/gripper/object-motion structure better than random jitter, random pose jitter, and simple object-relative retargeting while remaining replay-valid."
    return {
        "continue_or_kill": decision,
        "reason": reason,
        "next_state": (
            "STATE 2: tiny BC/action-head diagnostic on original versus augmented demos"
            if decision == "continue"
            else ("resolve_contacttube_aug_blocker" if decision == "blocked" else "archive_or_reframe_contacttube_aug_before_training")
        ),
        "replay_metric_produced": replay_metric,
        "exact_init_noop_upper_bound_success": exact_success,
        "contacttube_aug_success": method_success,
        "contact_tube_extractable": contact_extractable,
        "object_state_available": object_state_available,
        "hdf5_object_pose_available": bool(hdf5_tube.get("object_pose_available")),
        "runtime_object_pose_available": bool(runtime_tube.get("object_pose_available")),
        "contacttube_aug_controller_valid": method_valid,
        "contacttube_aug_beats_random_action_jitter": method_beats_random_jitter,
        "contacttube_aug_beats_random_pose_jitter": method_beats_random_pose,
        "contacttube_aug_beats_simple_object_relative": method_beats_simple,
        "simple_object_relative_matches_or_beats_contacttube_aug": simple_matches,
        "baselines_tested": list(BASELINE_VARIANTS),
        "method_variant": METHOD_VARIANT,
        "variant_scores": variant_scores,
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_contacttube_aug_state1": True,
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


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary") or {}
    case = (report.get("cases") or [{}])[0]
    lines = [
        "# ContactTube-Aug STATE 1 Result",
        "",
        "Bounded replay/control diagnostic only. This is not benchmark success, paper-grade evidence, or a policy-training result.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- reason: {summary.get('reason')}",
        f"- replay happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- loss computed: `{report.get('policy', {}).get('loss_computed')}`",
        f"- GPU/download/OpenVLA-OFT: `{report.get('policy', {}).get('gpu_jobs_performed')}` / `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}`",
        f"- demos/tasks: `1 / 1`",
        f"- contact-tube extraction success: `{summary.get('contact_tube_extractable')}`",
        f"- HDF5 object pose available: `{summary.get('hdf5_object_pose_available')}`",
        f"- runtime object pose available: `{summary.get('runtime_object_pose_available')}`",
        f"- augmentation validity: `{summary.get('contacttube_aug_controller_valid')}`",
        f"- simple object-relative matches/beats ContactTube-Aug: `{summary.get('simple_object_relative_matches_or_beats_contacttube_aug')}`",
        f"- ContactTube-Aug beats simple object-relative: `{summary.get('contacttube_aug_beats_simple_object_relative')}`",
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
        f"- translation unit source: `{(case.get('translation_unit') or {}).get('source')}`",
        "",
        "## Replay Metrics",
        "",
        "| variant | init | reward | success | first done | steps | tube score | dist MAE | motion err | lift err | place err | grip close err | valid rate | clip step |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in case.get("variants", []):
        metrics = item.get("contact_tube_metrics") or {}
        grip = item.get("gripper_timing_error") or {}
        motion = metrics.get("object_motion_onset_error") or {}
        lift = metrics.get("lift_phase_error") or {}
        place = metrics.get("place_phase_error") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant")),
                    str(item.get("init_mode")),
                    _md(item.get("reward_sum")),
                    _md(_success(item)),
                    _md(item.get("first_done_index")),
                    _md(item.get("steps_performed")),
                    _md(metrics.get("contact_tube_preservation_error")),
                    _md(metrics.get("eef_object_distance_profile_mae")),
                    _md(motion.get("absolute_error")),
                    _md(lift.get("absolute_error")),
                    _md(place.get("absolute_error")),
                    _md(grip.get("close_absolute_error")),
                    _md(item.get("controller_valid_action_rate")),
                    _md(item.get("clip_rate_step")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Baseline Gate",
            "",
            f"- baselines tested: `{', '.join(summary.get('baselines_tested') or [])}`",
            f"- method variant: `{summary.get('method_variant')}`",
            f"- ContactTube-Aug beats random action jitter: `{summary.get('contacttube_aug_beats_random_action_jitter')}`",
            f"- ContactTube-Aug beats random pose jitter: `{summary.get('contacttube_aug_beats_random_pose_jitter')}`",
            f"- ContactTube-Aug beats simple object-relative: `{summary.get('contacttube_aug_beats_simple_object_relative')}`",
            "",
            "## Feasibility Notes",
            "",
            "- Object pose shift is represented by the simulator default-reset object start relative to exact HDF5 init when available.",
            "- Reset pose shift uses default reset versus exact HDF5 init; no task-generic init-state object editor is assumed.",
            "- Distractor insertion/relabeling and camera perturbation are logged as not feasible in this smoke because no training/render augmentation is run.",
            "- Exact-init no-op replay is the upper bound/control; default-reset variants are diagnostics for augmentation validity.",
            "- The runner does not use reward labels or success labels to choose ContactTube-Aug actions.",
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
        "evidence_label": "contacttube_aug_state1",
        "policy": _policy(forbidden),
        "inputs": vars(args).copy(),
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "skipped_conditions": {
            "object_pose_shift": "represented_by_default_reset_object_delta_when_runtime_object_pose_is_available",
            "reset_pose_shift": "default_reset_vs_exact_hdf5_init",
            "distractor_insertion_or_relabeling": "not_feasible_without_task_generic_scene_editor_or_training_image_pipeline",
            "viewpoint_camera_perturbation": "not_feasible_without render/image augmentation in this replay-only smoke",
            "visual_object_swap_only": "not_run_no_training_or_image_only_policy_evaluation_in_state1",
        },
        "summary": {},
        "result": {"passed": False, "blocked_reason": None, "total_steps_performed": 0, "variant_count": 0},
        "elapsed_seconds": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden gates set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for bounded ContactTube-Aug replay")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("rollout readiness gate is not green/authorized")
    if args.max_steps_cap < 1 or args.max_steps_cap > 320:
        stop_reasons.append("max_steps_cap must be between 1 and 320")
    if args.post_signal_margin < 0 or args.post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if args.camera_size < 16 or args.camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    if args.noise_std < 0.0 or args.noise_std > 0.20:
        stop_reasons.append("noise_std must be between 0.0 and 0.20")
    if args.pose_jitter_meters < 0.0 or args.pose_jitter_meters > 0.08:
        stop_reasons.append("pose_jitter_meters must be between 0.0 and 0.08")
    if stop_reasons:
        try:
            case = build_contacttube_case(
                _as_path(args.manifest),
                max_steps_cap=args.max_steps_cap,
                post_signal_margin=args.post_signal_margin,
                seed=args.seed,
                noise_std=args.noise_std,
                pose_jitter_meters=args.pose_jitter_meters,
            )
            report["cases"].append(_case_header(case, Path("<not-run-without-gate>")))
        except Exception as exc:  # noqa: BLE001
            stop_reasons.append(f"failed to build ContactTube-Aug case: {type(exc).__name__}: {exc}")
        report["result"]["blocked_reason"] = "; ".join(stop_reasons)
        report["summary"] = summarize_report(report)
        if report["summary"].get("continue_or_kill") == "blocked":
            report["summary"]["reason"] = report["result"]["blocked_reason"]
        report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
        return report
    try:
        _run_replay_case(args, report)
        all_variants = (report["cases"][0].get("variants") or []) if report.get("cases") else []
        report["result"]["passed"] = bool(all_variants and all(item.get("passed") for item in all_variants))
        report["summary"] = summarize_report(report)
        if not report["result"]["passed"] and report["result"].get("blocked_reason") is None:
            report["result"]["blocked_reason"] = "one or more ContactTube-Aug replay variants failed"
    except Exception as exc:  # noqa: BLE001
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_contacttube_aug_blocker"}
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-steps-cap", type=int, default=300)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--pose-jitter-meters", type=float, default=0.02)
    parser.add_argument("--report-json", default="reports/contacttube_aug_state1_result.json")
    parser.add_argument("--report-md", default="reports/contacttube_aug_state1_result.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    console = {
        "result": report.get("result"),
        "summary": {key: value for key, value in (report.get("summary") or {}).items() if key != "variant_scores"},
        "report_json": str(report_json),
        "replay_or_rollout_performed": report.get("policy", {}).get("replay_or_rollout_performed"),
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] or os.environ.get(TASK_GATE) != "1" else 1


if __name__ == "__main__":
    sys.exit(main())
