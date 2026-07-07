"""Object-relative replay retargeting diagnostic for reset mismatch.

This runner is bounded diagnostic evidence only. It replays local LIBERO HDF5
expert actions under exact-init and reset-mismatched conditions, then tests a
small object-relative translation retargeter against simple action-only
baselines. It performs no training, model loading, GPU work, downloads,
OpenVLA-OFT execution, benchmark sweep, or paper-grade claim.
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


SCHEMA_VERSION = "2026-07-07.resetspec_retarget.v1"
TASK_GATE = "ALLOW_RESETSPEC_RETARGET"
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
)

DEFAULT_BASELINE_VARIANTS = (
    "hdf5_expert_replay_default_reset",
    "default_reset_diagonal_affine_replay",
    "default_reset_global_scale_replay",
    "default_reset_clipping_replay",
)
RETARGET_VARIANTS = (
    "object_relative_translation_retarget",
    "object_relative_translation_gripper_phase_retarget",
)
ALL_EXECUTED_VARIANTS = (
    "hdf5_expert_replay_exact_init",
    *DEFAULT_BASELINE_VARIANTS,
    *RETARGET_VARIANTS,
)


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


def _success(result: dict[str, Any]) -> bool:
    return bool(result.get("final_success") or result.get("done_seen") or float(result.get("reward_sum") or 0.0) > 0.0)


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
                    return {
                        "available": True,
                        "key": key,
                        "positions": positions[:, :3],
                        "obs_keys": keys,
                    }
        return {"available": False, "key": None, "positions": None, "obs_keys": keys}


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


def _manifest_pair(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("counterfactual split manifest has no counterfactual_pairs")
    return pairs[0]


def build_resetspec_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 320,
    post_signal_margin: int = 20,
    global_scale: float = 0.85,
) -> dict[str, Any]:
    pair = _manifest_pair(manifest_path)
    positive = _read_demo_full(_as_path(pair["positive_demo_file"]), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    actions = np.asarray(positive["actions"], dtype=np.float64)
    eef = _read_hdf5_eef_positions(_as_path(positive["path"]), positive["demo_name"], int(actions.shape[0]) + 1)
    eef_positions = eef["positions"] if eef.get("available") else None
    unit = _estimate_translation_unit(actions, eef_positions)
    diagonal_actions = actions.copy()
    global_actions = actions.copy()
    global_actions[:, :6] *= float(global_scale)
    clipping_actions = np.clip(actions, -1.0, 1.0)
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
        "hdf5_eef_source": {
            "available": bool(eef.get("available")),
            "key": eef.get("key"),
            "obs_keys": eef.get("obs_keys") or [],
        },
        "translation_unit": unit,
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
        "action_baselines": {
            "diagonal_affine_policy": {
                "description": "Action-only diagonal affine baseline has no non-leaking reset-state calibration target, so the predeclared calibration is identity.",
                "scale": [1.0] * 7,
                "bias": [0.0] * 7,
                "uses_eval_success_labels": False,
                "uses_object_pose": False,
            },
            "global_scale_policy": {
                "description": "Fixed global scale baseline multiplies translation and rotation dimensions before replay; gripper sign is preserved.",
                "scale": float(global_scale),
                "uses_eval_success_labels": False,
                "uses_object_pose": False,
            },
            "clipping_policy": {
                "description": "Controller range clipping only.",
                "uses_eval_success_labels": False,
                "uses_object_pose": False,
            },
        },
        "static_variant_actions": {
            "hdf5_expert_replay_exact_init": actions,
            "hdf5_expert_replay_default_reset": actions,
            "default_reset_diagonal_affine_replay": diagonal_actions,
            "default_reset_global_scale_replay": global_actions,
            "default_reset_clipping_replay": clipping_actions,
        },
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


def _inspect_start_pose(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    instruction: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    env = None
    summary = {
        "mode": mode,
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
        if mode == "exact":
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


def retarget_translation_action(
    *,
    raw_action: np.ndarray,
    current_eef: list[float] | None,
    desired_eef: np.ndarray | None,
    meters_per_action_unit: float,
) -> tuple[np.ndarray, bool]:
    action = np.asarray(raw_action, dtype=np.float64).copy()
    if current_eef is None or desired_eef is None or not np.isfinite(meters_per_action_unit) or meters_per_action_unit <= 0:
        return action, False
    delta = np.asarray(desired_eef, dtype=np.float64).reshape(-1)[:3] - np.asarray(current_eef, dtype=np.float64).reshape(-1)[:3]
    action[:3] = delta / float(meters_per_action_unit)
    return action, True


def _phase_gripper_value(
    *,
    raw_actions: np.ndarray,
    current_eef: list[float] | None,
    current_object: list[float] | None,
    demo_eef_positions: np.ndarray | None,
    demo_object: list[float] | None,
) -> tuple[float | None, int | None]:
    if current_eef is None or current_object is None or demo_eef_positions is None or demo_object is None:
        return None, None
    if raw_actions.ndim != 2 or raw_actions.shape[1] < 7 or demo_eef_positions.shape[0] == 0:
        return None, None
    current_dist = _norm(np.asarray(current_eef, dtype=np.float64) - np.asarray(current_object, dtype=np.float64))
    if current_dist is None:
        return None, None
    demo_obj = np.asarray(demo_object, dtype=np.float64).reshape(-1)[:3]
    demo_count = min(raw_actions.shape[0], demo_eef_positions.shape[0])
    demo_dist = np.linalg.norm(np.asarray(demo_eef_positions[:demo_count, :3], dtype=np.float64) - demo_obj.reshape(1, 3), axis=1)
    index = int(np.argmin(np.abs(demo_dist - float(current_dist))))
    return float(raw_actions[index, 6]), index


def _desired_eef_for_step(
    *,
    demo_eef_positions: np.ndarray | None,
    demo_object: list[float] | None,
    current_start_object: list[float] | None,
    step: int,
) -> tuple[np.ndarray | None, list[float] | None]:
    if demo_eef_positions is None or demo_object is None or current_start_object is None:
        return None, None
    demo_obj = np.asarray(demo_object, dtype=np.float64).reshape(-1)[:3]
    current_obj = np.asarray(current_start_object, dtype=np.float64).reshape(-1)[:3]
    object_delta = current_obj - demo_obj
    index = min(step + 1, demo_eef_positions.shape[0] - 1)
    return np.asarray(demo_eef_positions[index, :3], dtype=np.float64) + object_delta, [float(v) for v in object_delta.tolist()]


def _trajectory_drift(
    observed_eef: list[list[float]],
    demo_eef_positions: np.ndarray | None,
    object_delta: list[float] | None,
) -> dict[str, Any]:
    if not observed_eef or demo_eef_positions is None or object_delta is None:
        return {"available": False, "mean_l2": None, "final_l2": None}
    count = min(len(observed_eef), demo_eef_positions.shape[0])
    if count == 0:
        return {"available": False, "mean_l2": None, "final_l2": None}
    expected = np.asarray(demo_eef_positions[:count, :3], dtype=np.float64) + np.asarray(object_delta, dtype=np.float64).reshape(1, 3)
    actual = np.asarray(observed_eef[:count], dtype=np.float64)
    distances = np.linalg.norm(actual - expected, axis=1)
    return {
        "available": True,
        "mean_l2": _round(float(np.mean(distances)), 6),
        "final_l2": _round(float(distances[-1]), 6),
        "sample_count": int(count),
    }


def _variant_action_for_step(
    *,
    variant: str,
    step: int,
    obs: Any,
    raw_actions: np.ndarray,
    static_actions: np.ndarray | None,
    demo_eef_positions: np.ndarray | None,
    demo_object: list[float] | None,
    current_start_object: list[float] | None,
    meters_per_action_unit: float,
    target_key: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_actions[step], dtype=np.float64)
    if static_actions is not None:
        return np.asarray(static_actions[step], dtype=np.float64).copy(), {"retarget_translation_available": False}
    current_eef = _extract_eef(obs)
    desired, object_delta = _desired_eef_for_step(
        demo_eef_positions=demo_eef_positions,
        demo_object=demo_object,
        current_start_object=current_start_object,
        step=step,
    )
    action, available = retarget_translation_action(
        raw_action=raw,
        current_eef=current_eef,
        desired_eef=desired,
        meters_per_action_unit=meters_per_action_unit,
    )
    phase_index = None
    if variant == "object_relative_translation_gripper_phase_retarget":
        current_object = _extract_pos(obs, target_key)
        gripper, phase_index = _phase_gripper_value(
            raw_actions=raw_actions,
            current_eef=current_eef,
            current_object=current_object,
            demo_eef_positions=demo_eef_positions,
            demo_object=demo_object,
        )
        if gripper is not None:
            action[6] = gripper
    return action, {
        "retarget_translation_available": available,
        "desired_eef": None if desired is None else [float(value) for value in desired.tolist()],
        "object_delta_from_exact_init": object_delta,
        "gripper_phase_index": phase_index,
    }


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    case: dict[str, Any],
    variant: str,
    init_mode: str,
    reference_exact_start: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    raw_actions = np.asarray(case["actions"], dtype=np.float64)
    static_actions = case["static_variant_actions"].get(variant)
    if static_actions is not None:
        static_actions = np.asarray(static_actions, dtype=np.float64)
    steps = int(raw_actions.shape[0])
    summary: dict[str, Any] = {
        "variant": variant,
        "init_mode": init_mode,
        "claim_role": "expert_upper_bound" if init_mode == "exact" else ("object_relative_retarget_diagnostic" if variant in RETARGET_VARIANTS else "simple_baseline"),
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
        "trajectory_drift": {"available": False, "mean_l2": None, "final_l2": None},
        "action_metrics_vs_raw_expert": None,
        "controller_valid_action_rate": None,
        "clip_rate_element": None,
        "clip_rate_step": None,
        "gripper_timing_error": None,
        "translation_retarget_available_steps": 0,
        "retarget_trace_first_5": [],
        "controller": None,
        "after_set_state_l2_to_hdf5_init": None,
        "error": None,
    }
    env = None
    executed_raw_actions: list[np.ndarray] = []
    executed_env_actions: list[np.ndarray] = []
    observed_eef: list[list[float]] = []
    object_delta: list[float] | None = None
    obs: Any = None
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
        start_distance = _distance(eef_start, target_start)
        for step in range(steps):
            action_raw, trace = _variant_action_for_step(
                variant=variant,
                step=step,
                obs=obs,
                raw_actions=raw_actions,
                static_actions=static_actions,
                demo_eef_positions=case.get("hdf5_eef_positions"),
                demo_object=reference_exact_start.get("target_pos"),
                current_start_object=target_start,
                meters_per_action_unit=float(case["translation_unit"]["meters_per_action_unit"]),
                target_key=target_key,
            )
            if trace.get("retarget_translation_available"):
                summary["translation_retarget_available_steps"] += 1
            if trace.get("object_delta_from_exact_init") is not None:
                object_delta = trace["object_delta_from_exact_init"]
            if len(summary["retarget_trace_first_5"]) < 5 and variant in RETARGET_VARIANTS:
                compact_trace = {
                    "step": step,
                    "retarget_translation_available": trace.get("retarget_translation_available"),
                    "desired_eef": trace.get("desired_eef"),
                    "object_delta_from_exact_init": trace.get("object_delta_from_exact_init"),
                    "gripper_phase_index": trace.get("gripper_phase_index"),
                }
                summary["retarget_trace_first_5"].append(compact_trace)
            env_action = np.clip(np.asarray(action_raw, dtype=np.float64), -1.0, 1.0)
            executed_raw_actions.append(np.asarray(action_raw, dtype=np.float64))
            executed_env_actions.append(env_action)
            obs, reward, done, _info = env.step(env_action)
            reward_value = float(reward)
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            current_eef = _extract_eef(obs)
            if current_eef is not None:
                observed_eef.append(current_eef)
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
        if object_delta is None and target_start is not None and reference_exact_start.get("target_pos") is not None:
            object_delta = [
                float(a - b)
                for a, b in zip(
                    np.asarray(target_start, dtype=np.float64).reshape(-1)[:3],
                    np.asarray(reference_exact_start["target_pos"], dtype=np.float64).reshape(-1)[:3],
                )
            ]
        summary["trajectory_drift"] = _trajectory_drift(observed_eef, case.get("hdf5_eef_positions"), object_delta)
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
    else:
        summary.update(_clip_stats(np.zeros((0, 7), dtype=np.float64)))
        summary["gripper_timing_error"] = {"reference_first_nonnegative_index": None, "candidate_first_nonnegative_index": None, "absolute_error": None}
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


def _beats(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-6) -> bool:
    left_tuple = _progress_tuple(left)
    right_tuple = _progress_tuple(right)
    for lval, rval in zip(left_tuple, right_tuple):
        if lval > rval + eps:
            return True
        if lval + eps < rval:
            return False
    left_done = left.get("first_done_index")
    right_done = right.get("first_done_index")
    if left_done is not None and right_done is None:
        return True
    if left_done is not None and right_done is not None and int(left_done) < int(right_done):
        return True
    return False


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    variants = {item["variant"]: item for item in (report.get("cases") or [{}])[0].get("variants", [])}
    exact = variants.get("hdf5_expert_replay_exact_init", {})
    default_raw = variants.get("hdf5_expert_replay_default_reset", {})
    retargets = [variants.get(name, {}) for name in RETARGET_VARIANTS if variants.get(name)]
    simple = [variants.get(name, {}) for name in DEFAULT_BASELINE_VARIANTS if variants.get(name)]
    exact_success = _success(exact)
    default_success = _success(default_raw)
    retarget_best = max(retargets, key=_progress_tuple) if retargets else {}
    best_simple = max(simple, key=_progress_tuple) if simple else {}
    object_pose_available = bool((report.get("cases") or [{}])[0].get("object_pose_audit", {}).get("object_poses_available_for_retarget"))
    raw_gap = bool(exact_success and not default_success)
    retarget_improves_default = bool(retarget_best and _beats(retarget_best, default_raw))
    retarget_beats_simple = bool(retarget_best and best_simple and _beats(retarget_best, best_simple))
    if not object_pose_available:
        decision = "kill"
        reason = "Object poses were unavailable for non-leaking retargeting."
    elif default_success:
        decision = "kill"
        reason = "Default-reset raw replay already succeeded, so there is no reset-mismatch baseline gap."
    elif not raw_gap:
        decision = "kill"
        reason = "Exact-init expert replay did not create a clean exact-vs-default reset gap."
    elif not retarget_improves_default:
        decision = "kill"
        reason = "Object-relative retargeting did not improve replay/progress over default raw replay."
    elif not retarget_beats_simple:
        decision = "kill"
        reason = "Object-relative retargeting did not beat the simple action-only baselines."
    else:
        decision = "continue"
        reason = "Object-relative retargeting improved reset-mismatched replay and beat the simple baselines on at least one progress metric."
    return {
        "continue_or_kill": decision,
        "reason": reason,
        "next_state": "STATE 2: broaden object-relative retargeting across demos/tasks" if decision == "continue" else "archive_or_reframe_resetspec_retarget",
        "exact_init_expert_replay_success": exact_success,
        "default_reset_raw_replay_success": default_success,
        "default_reset_gap_observed": raw_gap,
        "object_pose_available": object_pose_available,
        "best_retarget_variant": retarget_best.get("variant"),
        "best_simple_baseline": best_simple.get("variant"),
        "object_relative_improves_default_raw": retarget_improves_default,
        "object_relative_beats_simple_baselines": retarget_beats_simple,
        "baseline_progress_tuples": {name: _progress_tuple(payload) for name, payload in variants.items()},
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_resetspec_retarget": True,
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


def _md(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return str(_round(value, 6))
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary") or {}
    case = (report.get("cases") or [{}])[0]
    lines = [
        "# ResetSpec-Retarget STATE 1 Result",
        "",
        "Bounded replay/retarget diagnostic only. This is not benchmark success, paper-grade evidence, or a policy rollout claim.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- reason: {summary.get('reason')}",
        f"- replay happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- loss computed: `{report.get('policy', {}).get('loss_computed')}`",
        f"- exact-init expert replay success: `{summary.get('exact_init_expert_replay_success')}`",
        f"- default-reset raw replay success: `{summary.get('default_reset_raw_replay_success')}`",
        f"- object poses available: `{summary.get('object_pose_available')}`",
        f"- best retarget variant: `{summary.get('best_retarget_variant')}`",
        f"- best simple baseline: `{summary.get('best_simple_baseline')}`",
        f"- object-relative beats simple baselines: `{summary.get('object_relative_beats_simple_baselines')}`",
        f"- next state: `{summary.get('next_state')}`",
        "",
        "## Case",
        "",
        f"- task: `{case.get('task_id')}`",
        f"- instruction: {case.get('instruction')}",
        f"- selected horizon: `{case.get('target_horizon')}`",
        f"- HDF5 first reward/done/signal: `{(case.get('hdf5_metadata') or {}).get('first_positive_reward_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_done_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_signal_index')}`",
        f"- HDF5 EEF trajectory source: `{(case.get('hdf5_eef_source') or {}).get('key')}`",
        f"- translation unit source: `{(case.get('translation_unit') or {}).get('source')}`",
        "",
        "## Replay Metrics",
        "",
        "| variant | init | reward | success | first done | steps | dist change | object move | traj drift | clip step | trans err | rot err | grip timing err |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in case.get("variants", []):
        metrics = item.get("action_metrics_vs_raw_expert") or {}
        dist = item.get("eef_object_distance") or {}
        drift = item.get("trajectory_drift") or {}
        grip = item.get("gripper_timing_error") or {}
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
                    _md(dist.get("change")),
                    _md(item.get("target_object_movement_l2")),
                    _md(drift.get("mean_l2")),
                    _md(item.get("clip_rate_step")),
                    _md(metrics.get("translation_drift_mean")),
                    _md(metrics.get("rotation_drift_mean")),
                    _md(grip.get("absolute_error")),
                ]
            )
            + " |"
        )
    skipped = report.get("skipped_conditions") or {}
    lines.extend(
        [
            "",
            "## Skipped Conditions",
            "",
            f"- perturbed-init raw replay: `{skipped.get('perturbed_init_raw_replay')}`",
            f"- nearest-demo replay: `{skipped.get('nearest_demo_replay')}`",
            "",
            "## Non-Leakage Notes",
            "",
            "- Target object key is resolved from natural-language instruction text plus visible observation object keys.",
            "- The runner does not use BDDL target metadata, eval labels, dataset target labels, task IDs, filenames, or manifest target fields as inference-time target proxies.",
            "- Retargeted actions use demonstration EEF trajectory and current object/EEF state as replay diagnostics, not as an online policy-performance claim.",
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
        "evidence_label": "resetspec_retarget_state1",
        "policy": _policy(forbidden),
        "inputs": vars(args).copy(),
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "skipped_conditions": {
            "perturbed_init_raw_replay": "not_run_no_task_generic_safe_state_perturbation_helper",
            "nearest_demo_replay": "not_run_no_nonleaking_nearest_demo_selector_with_object_pose_cache",
        },
        "summary": {},
        "result": {"passed": False, "blocked_reason": None, "total_steps_performed": 0, "variant_count": 0},
        "elapsed_seconds": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden gates set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for bounded ResetSpec retarget replay")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("rollout readiness gate is not green/authorized")
    if args.max_steps_cap < 1 or args.max_steps_cap > 320:
        stop_reasons.append("max_steps_cap must be between 1 and 320")
    if args.post_signal_margin < 0 or args.post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if args.camera_size < 16 or args.camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    if stop_reasons:
        report["result"]["blocked_reason"] = "; ".join(stop_reasons)
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_resetspec_retarget_blocker"}
        report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
        return report
    try:
        case = build_resetspec_case(
            _as_path(args.manifest),
            max_steps_cap=args.max_steps_cap,
            post_signal_margin=args.post_signal_margin,
            global_scale=args.global_scale,
        )
        env_cls = _load_env_class(_as_path(args.libero_root), _as_path(args.robosuite_root))
        bddl_file = _as_path(args.libero_root) / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
        exact_start = _inspect_start_pose(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=args.camera_size,
            init_state=case["init_state"],
            instruction=case["instruction"],
            mode="exact",
            seed=args.seed,
        )
        default_start = _inspect_start_pose(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=args.camera_size,
            init_state=case["init_state"],
            instruction=case["instruction"],
            mode="default",
            seed=args.seed,
        )
        case_summary = {
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
            "translation_unit": case["translation_unit"],
            "action_baselines": case["action_baselines"],
            "object_pose_audit": {
                "exact_start": exact_start,
                "default_start": default_start,
                "object_poses_available_for_retarget": bool(exact_start.get("target_pos") and default_start.get("target_pos") and exact_start.get("eef_pos") and default_start.get("eef_pos")),
                "default_target_delta_from_exact_init_l2": _round(
                    _distance(exact_start.get("target_pos"), default_start.get("target_pos")),
                    6,
                ),
                "default_eef_delta_from_exact_init_l2": _round(
                    _distance(exact_start.get("eef_pos"), default_start.get("eef_pos")),
                    6,
                ),
            },
            "variants": [],
        }
        total_steps = 0
        for variant in ALL_EXECUTED_VARIANTS:
            init_mode = "exact" if variant == "hdf5_expert_replay_exact_init" else "default"
            result = _run_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=args.camera_size,
                init_state=case["init_state"],
                case=case,
                variant=variant,
                init_mode=init_mode,
                reference_exact_start=exact_start,
                seed=args.seed,
            )
            case_summary["variants"].append(result)
            total_steps += int(result.get("steps_performed") or 0)
        report["cases"].append(case_summary)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["replay_or_rollout_performed"] = total_steps > 0
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = len(case_summary["variants"])
        report["result"]["passed"] = all(item.get("passed") for item in case_summary["variants"])
        report["summary"] = _summarize(report)
    except Exception as exc:  # noqa: BLE001
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_resetspec_retarget_blocker"}
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
    parser.add_argument("--global-scale", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="reports/resetspec_state1_result.json")
    parser.add_argument("--report-md", default="reports/resetspec_state1_result.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    console = {
        "result": report.get("result"),
        "summary": report.get("summary"),
        "report_json": str(report_json),
        "replay_or_rollout_performed": report.get("policy", {}).get("replay_or_rollout_performed"),
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] or os.environ.get(TASK_GATE) != "1" else 1


if __name__ == "__main__":
    sys.exit(main())
