"""Full-demo expert replay sanity for bounded LIBERO rollout diagnostics.

This module replays local HDF5 expert actions long enough to reach the first
recorded reward/done index when possible. It is bounded diagnostic evidence
only: no training, no model loading, no VLA inference, no GPU jobs, no
downloads, no OpenVLA-OFT, no benchmark sweep, and no paper-grade claims.
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
    _policy,
    _write_json,
)
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _distance_delta,
    _extract_eef,
    _extract_pos,
    _object_position_keys,
)

SCHEMA_VERSION = "2026-07-06.libero_full_demo_expert_replay_sanity.v1"
TASK_GATE = "ALLOW_FULL_DEMO_EXPERT_REPLAY"
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
    "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = report.get("result", {})
    decision = report.get("decision") or {}
    lines = [
        "# Full-Demo Expert Replay Sanity",
        "",
        "This is a bounded diagnostic only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- diagnostic passed: `{result.get('passed')}`",
        f"- rollout happened: `{report['policy']['diagnostic_rollouts_performed']}`",
        f"- full-demo expert replay happened: `{result.get('full_demo_expert_replay_happened')}`",
        f"- expert replay succeeded: `{decision.get('expert_replay_succeeded')}`",
        f"- observed first signal index: `{decision.get('observed_first_signal_index')}`",
        f"- HDF5 first signal index: `{decision.get('hdf5_first_signal_index')}`",
        f"- blocker classification: `{decision.get('blocker_classification')}`",
        f"- bridge green for longer-horizon method rollout: `{decision.get('bridge_green_for_longer_horizon_method_rollout')}`",
        f"- recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_index(values: np.ndarray, threshold: float = 0.0) -> int | None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _safe_l2(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    width = min(left_arr.size, right_arr.size)
    if width == 0:
        return None
    return round(float(np.linalg.norm(left_arr[:width] - right_arr[:width])), 9)


def _sim_state_array(env: Any) -> list[float] | None:
    try:
        state = env.sim.get_state()
        if hasattr(state, "flatten"):
            return [float(value) for value in np.asarray(state.flatten(), dtype=np.float64).reshape(-1)]
    except Exception:
        pass
    try:
        qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).reshape(-1)
        return [float(value) for value in np.concatenate([qpos, qvel], axis=0)]
    except Exception:
        return None


def _hdf5_first_obs_snapshot(demo: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "available": False,
        "keys": [],
        "eef_pos": None,
        "gripper_state": None,
        "object_position_keys": [],
    }
    if "obs" not in demo:
        return snapshot
    obs_group = demo["obs"]
    keys = sorted(str(key) for key in obs_group.keys())
    snapshot["available"] = True
    snapshot["keys"] = keys
    for key in ("robot0_eef_pos", "ee_pos", "eef_pos"):
        if key in obs_group:
            arr = np.asarray(obs_group[key][0], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                snapshot["eef_pos"] = [float(value) for value in arr[:3]]
                break
    for key in ("robot0_gripper_qpos", "gripper_states"):
        if key in obs_group:
            snapshot["gripper_state"] = [float(value) for value in np.asarray(obs_group[key][0], dtype=np.float64).reshape(-1)]
            break
    snapshot["object_position_keys"] = [
        key
        for key in keys
        if key.endswith("_pos") and not key.startswith("robot") and key not in {"ee_pos", "eef_pos"}
    ]
    return snapshot


def _read_demo_full(path: Path, *, max_steps_cap: int, post_signal_margin: int) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        demo_name = sorted(data_group.keys())[0]
        demo = data_group[demo_name]
        if "actions" not in demo:
            raise ValueError(f"{path} demo {demo_name} has no actions dataset")
        if "init_state" not in demo.attrs:
            raise ValueError(f"{path} demo {demo_name} has no init_state attribute")
        full_actions = np.asarray(demo["actions"], dtype=np.float64)
        if full_actions.ndim != 2 or full_actions.shape[1] != 7:
            raise ValueError(f"{path} actions must be [T, 7], got {list(full_actions.shape)}")
        rewards = np.asarray(demo["rewards"], dtype=np.float64).reshape(-1) if "rewards" in demo else np.zeros((full_actions.shape[0],))
        dones = np.asarray(demo["dones"], dtype=np.float64).reshape(-1) if "dones" in demo else np.zeros((full_actions.shape[0],))
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64).reshape(-1)
        states0 = np.asarray(demo["states"][0], dtype=np.float64).reshape(-1) if "states" in demo and demo["states"].shape[0] > 0 else None
        first_obs = _hdf5_first_obs_snapshot(demo)
        model_file_available = "model_file" in demo.attrs
        num_samples = int(demo.attrs.get("num_samples", full_actions.shape[0]))
    first_reward_index = _first_index(rewards, 0.0)
    first_done_index = _first_index(dones, 0.5)
    signals = [index for index in (first_reward_index, first_done_index) if index is not None]
    first_signal_index = min(signals) if signals else None
    if first_signal_index is None:
        target_horizon = min(int(full_actions.shape[0]), int(max_steps_cap))
    else:
        target_horizon = min(int(first_signal_index) + int(post_signal_margin), int(full_actions.shape[0]), int(max_steps_cap))
    target_horizon = max(1, target_horizon)
    return {
        "path": str(path),
        "demo_name": demo_name,
        "init_state": init_state,
        "full_actions": full_actions,
        "actions": full_actions[:target_horizon],
        "first_reward_index": first_reward_index,
        "first_done_index": first_done_index,
        "first_signal_index": first_signal_index,
        "target_horizon": int(target_horizon),
        "full_action_steps": int(full_actions.shape[0]),
        "num_samples_attr": num_samples,
        "states0_l2_to_init_state": _safe_l2(states0, init_state),
        "first_obs": first_obs,
        "model_file_available": bool(model_file_available),
        "action_stats": _action_stats(full_actions[:target_horizon]),
    }


def build_full_demo_expert_replay_case(
    manifest_path: Path,
    *,
    max_tasks: int = 1,
    max_steps_cap: int = 320,
    post_signal_margin: int = 20,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")
    pairs = manifest.get("counterfactual_pairs", [])
    if not pairs:
        raise ValueError("counterfactual split manifest has no pairs")
    pair = pairs[0]
    positive = _read_demo_full(_as_path(pair["positive_demo_file"]), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    actions = positive["actions"]
    zero_actions = np.zeros_like(actions)
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
        "target_horizon": positive["target_horizon"],
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
            "first_obs": positive["first_obs"],
            "model_file_available": positive["model_file_available"],
        },
        "action_diagnostics": {
            "expert_full_demo_window": positive["action_stats"],
            "zero_action_window": _action_stats(zero_actions),
            "hdf5_action_distribution_matches_expected_range": bool(
                positive["action_stats"]["finite"] and float(positive["action_stats"]["range"]["max_abs"] or 0.0) <= 1.0
            ),
        },
        "variants": [
            {"name": "zero_action_exact_init", "claim_role": "negative_control", "actions": zero_actions, "use_exact_init_state": True},
            {"name": "hdf5_expert_replay_exact_init", "claim_role": "expert_replay_sanity", "actions": actions, "use_exact_init_state": True},
            {"name": "hdf5_expert_replay_default_reset", "claim_role": "init_state_control", "actions": actions, "use_exact_init_state": False},
        ],
    }


def _controller_summary(env: Any) -> dict[str, Any]:
    summary = {"controller_type": None, "control_freq": None, "timestep": None}
    try:
        summary["control_freq"] = getattr(env, "control_freq", None)
    except Exception:
        pass
    try:
        summary["timestep"] = getattr(getattr(env, "sim", None), "model", None).opt.timestep
    except Exception:
        pass
    try:
        robots = getattr(env, "robots", [])
        if robots:
            controller = getattr(robots[0], "controller", None)
            summary["controller_type"] = getattr(controller, "name", None) or (type(controller).__name__ if controller is not None else None)
    except Exception:
        pass
    return summary


def _gripper_timing(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0 or actions.shape[1] < 7:
        return {"available": False}
    grip = np.asarray(actions[:, 6], dtype=np.float64).reshape(-1)
    signs = [int(np.sign(value)) for value in grip]
    sign_changes = [index for index in range(1, len(signs)) if signs[index] != signs[index - 1]]
    return {
        "available": True,
        "min": round(float(grip.min()), 6),
        "max": round(float(grip.max()), 6),
        "mean": round(float(grip.mean()), 6),
        "first_nonnegative_index": next((index for index, value in enumerate(grip) if value >= 0.0), None),
        "first_positive_index": next((index for index, value in enumerate(grip) if value > 0.0), None),
        "sign_change_indices_first_10": sign_changes[:10],
    }


def _run_replay_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    variant: dict[str, Any],
    instruction: str,
) -> dict[str, Any]:
    actions = np.asarray(variant["actions"], dtype=np.float64)
    summary: dict[str, Any] = {
        "variant": variant["name"],
        "claim_role": variant["claim_role"],
        "use_exact_init_state": bool(variant.get("use_exact_init_state")),
        "action_shape": list(actions.shape),
        "env_action_shape": 7,
        "action_stats": _action_stats(actions),
        "gripper_timing": _gripper_timing(actions),
        "env_created": False,
        "reset_ok": False,
        "set_init_state_used": False,
        "set_init_state_ok": False,
        "after_set_state_l2_to_hdf5_init": None,
        "steps_requested": int(actions.shape[0]),
        "steps_performed": 0,
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "final_success": None,
        "done_seen": False,
        "first_positive_reward_index": None,
        "first_done_index": None,
        "first_success_index": None,
        "reward_trajectory": [],
        "done_indices": [],
        "success_indices": [],
        "available_obs_keys": [],
        "camera_keys": [],
        "state_keys": [],
        "target_key_audit": None,
        "target_directed_movement": None,
        "object_movement": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "controller": None,
        "delta_vs_absolute_action_convention_evidence": "not_established_until_replay_succeeds",
        "error": None,
    }
    env = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        summary["controller"] = _controller_summary(env)
        env.seed(0)
        obs = env.reset()
        summary["reset_ok"] = True
        if bool(variant.get("use_exact_init_state")):
            summary["set_init_state_used"] = True
            obs = env.set_init_state(init_state)
            summary["set_init_state_ok"] = True
            summary["after_set_state_l2_to_hdf5_init"] = _safe_l2(_sim_state_array(env), init_state)
        summary["eef_start"] = _extract_eef(obs)
        if isinstance(obs, dict):
            keys = sorted(str(key) for key in obs.keys())
            summary["available_obs_keys"] = keys[:80]
            summary["camera_keys"] = [key for key in keys if "image" in key or "rgb" in key]
            summary["state_keys"] = [key for key in keys if "state" in key or key.endswith("_pos") or key.endswith("_quat")]
        target_audit = _best_object_key(obs, instruction)
        target_key = target_audit["best_key"]
        target_start = _extract_pos(obs, target_key)
        summary["target_key_audit"] = target_audit
        for index, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            reward_value = float(reward)
            try:
                success_value = bool(env.check_success())
            except Exception:
                success_value = None
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            summary["reward_trajectory"].append(round(reward_value, 6))
            if reward_value > 0.0 and summary["first_positive_reward_index"] is None:
                summary["first_positive_reward_index"] = int(index)
            if bool(done):
                summary["done_seen"] = True
                summary["done_indices"].append(int(index))
                if summary["first_done_index"] is None:
                    summary["first_done_index"] = int(index)
            if success_value:
                summary["success_indices"].append(int(index))
                if summary["first_success_index"] is None:
                    summary["first_success_index"] = int(index)
            if bool(done) or reward_value > 0.0 or success_value:
                break
        summary["eef_final"] = _extract_eef(obs)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            summary["eef_displacement_l2"] = round(float(np.linalg.norm(np.asarray(summary["eef_final"]) - np.asarray(summary["eef_start"]))), 6)
        target_final = _extract_pos(obs, target_key)
        summary["target_directed_movement"] = _distance_delta(summary["eef_start"], summary["eef_final"], target_start, target_final)
        target_object_distance = _distance(target_start, target_final)
        summary["object_movement"] = {
            "available": target_object_distance is not None,
            "target_object_key": target_key,
            "target_object_displacement_l2": target_object_distance,
            "object_position_keys_missing": not bool(_object_position_keys(obs)),
        }
        try:
            summary["final_success"] = bool(env.check_success())
        except Exception:
            summary["final_success"] = None
        if summary["first_positive_reward_index"] is not None or summary["first_done_index"] is not None or summary["first_success_index"] is not None:
            summary["delta_vs_absolute_action_convention_evidence"] = "raw_hdf5_actions_reached_reward_done_or_success"
    except Exception as exc:
        summary["error"] = _compact(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    rewards = summary["reward_trajectory"]
    summary["reward_trajectory_summary"] = {
        "length": len(rewards),
        "nonzero_indices": [index for index, value in enumerate(rewards) if float(value) > 0.0][:20],
        "first_10": rewards[:10],
        "last_10": rewards[-10:],
    }
    summary["passed"] = bool(
        summary["env_created"]
        and summary["reset_ok"]
        and ((not bool(variant.get("use_exact_init_state"))) or summary["set_init_state_ok"])
        and summary["steps_performed"] > 0
        and summary["error"] is None
    )
    return summary


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    case = (report.get("cases") or [{}])[0]
    hdf5 = case.get("hdf5_metadata", {})
    variants = case.get("variants", [])
    exact = next((item for item in variants if item.get("variant") == "hdf5_expert_replay_exact_init"), {})
    default = next((item for item in variants if item.get("variant") == "hdf5_expert_replay_default_reset"), {})
    zero = next((item for item in variants if item.get("variant") == "zero_action_exact_init"), {})
    hdf5_index = hdf5.get("first_signal_index")
    observed_values = [exact.get("first_positive_reward_index"), exact.get("first_done_index"), exact.get("first_success_index")]
    observed = min([int(value) for value in observed_values if value is not None], default=None)
    expert_ok = bool(exact.get("final_success") or exact.get("done_seen") or float(exact.get("reward_sum") or 0.0) > 0.0)
    default_ok = bool(default.get("final_success") or default.get("done_seen") or float(default.get("reward_sum") or 0.0) > 0.0)
    zero_ok = bool(zero.get("final_success") or zero.get("done_seen") or float(zero.get("reward_sum") or 0.0) > 0.0)
    exact_init_used = bool(exact.get("set_init_state_used") and exact.get("set_init_state_ok"))
    state_l2 = exact.get("after_set_state_l2_to_hdf5_init")
    state_match = bool(state_l2 is not None and float(state_l2) <= 1e-6)
    matches_hdf5 = bool(observed is not None and hdf5_index is not None and abs(int(observed) - int(hdf5_index)) <= 1)
    if not exact_init_used:
        blocker = "init_state_replay_blocker"
        bridge_green = False
    elif expert_ok and matches_hdf5:
        blocker = "bridge_init_action_convention_green"
        bridge_green = True
    elif expert_ok:
        blocker = "expert_replay_succeeds_but_timing_differs"
        bridge_green = True
    elif observed is None and hdf5_index is not None and int(exact.get("steps_performed") or 0) > int(hdf5_index):
        blocker = "action_convention_or_demo_mapping_mismatch"
        bridge_green = False
    else:
        blocker = "expert_replay_inconclusive_or_short"
        bridge_green = False
    return {
        "blocker_classification": blocker,
        "bridge_green_for_longer_horizon_method_rollout": bridge_green,
        "expert_replay_succeeded": expert_ok,
        "default_reset_expert_replay_succeeded": default_ok,
        "zero_action_succeeded": zero_ok,
        "default_reset_label": "expert_requires_exact_demo_init_state" if expert_ok and not default_ok else ("expert_succeeds_from_default_reset_too" if expert_ok and default_ok else "default_reset_not_decisive_until_exact_replay_succeeds"),
        "exact_init_state_used": exact_init_used,
        "exact_sim_state_match_to_hdf5_init": state_match,
        "after_set_state_l2_to_hdf5_init": state_l2,
        "hdf5_first_signal_index": hdf5_index,
        "hdf5_first_positive_reward_index": hdf5.get("first_positive_reward_index"),
        "hdf5_first_done_index": hdf5.get("first_done_index"),
        "observed_first_signal_index": observed,
        "observed_first_positive_reward_index": exact.get("first_positive_reward_index"),
        "observed_first_done_index": exact.get("first_done_index"),
        "observed_first_success_index": exact.get("first_success_index"),
        "matches_hdf5_index_271": matches_hdf5,
        "steps_requested": exact.get("steps_requested"),
        "steps_performed": exact.get("steps_performed"),
        "expert_reward_sum": round(float(exact.get("reward_sum") or 0.0), 6),
        "zero_action_reward_sum": round(float(zero.get("reward_sum") or 0.0), 6),
        "action_convention_diagnosis": exact.get("delta_vs_absolute_action_convention_evidence"),
    }


def run_full_demo_expert_replay_sanity(
    *,
    manifest_path: Path,
    readiness_report_path: Path,
    report_json: Path,
    report_md: Path,
    libero_root: Path,
    robosuite_root: Path,
    max_tasks: int = 1,
    max_steps_cap: int = 320,
    post_signal_margin: int = 20,
    camera_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    readiness = _load_json(readiness_report_path) if readiness_report_path.exists() else {}
    policy = _policy()
    policy["bounded_full_demo_expert_replay_sanity"] = True
    policy["task_local_gate_required"] = f"{TASK_GATE}=1"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "inputs": {
            "manifest_path": str(manifest_path),
            "readiness_report_path": str(readiness_report_path),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "max_tasks": max_tasks,
            "max_steps_cap": max_steps_cap,
            "post_signal_margin": post_signal_margin,
            "camera_size": camera_size,
        },
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "result": {"passed": False, "reason": None, "total_steps_performed": 0, "variant_count": 0, "full_demo_expert_replay_happened": False},
        "forbidden_gates_set": forbidden,
        "decision": None,
        "elapsed_seconds": None,
        "recommended_next_step": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for this bounded full-demo expert replay sanity check")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("fixed-prior rollout readiness gate is not green/authorized")
    if max_tasks != 1:
        stop_reasons.append("max_tasks must be exactly 1 for this milestone")
    if max_steps_cap < 1 or max_steps_cap > 320:
        stop_reasons.append("max_steps_cap must be between 1 and 320")
    if post_signal_margin < 0 or post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if camera_size < 16 or camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    try:
        case = build_full_demo_expert_replay_case(manifest_path, max_tasks=max_tasks, max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    except Exception as exc:
        case = None
        stop_reasons.append(f"failed to build full-demo expert replay case: {type(exc).__name__}: {exc}")
    if stop_reasons:
        report["result"]["reason"] = "; ".join(stop_reasons)
        report["recommended_next_step"] = "Resolve listed blockers before full-demo expert replay sanity."
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(report_json, report)
        _write_markdown(report_md, report)
        return report
    try:
        assert case is not None
        env_cls = _load_env_class(libero_root, robosuite_root)
        bddl_file = libero_root / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
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
            "action_diagnostics": case["action_diagnostics"],
            "init_state_compatibility": {
                "uses_exact_hdf5_initial_state": True,
                "state_setting_function": "env.set_init_state(init_state)",
                "hdf5_states0_l2_to_init_state": case["hdf5_metadata"].get("states0_l2_to_init_state"),
                "hdf5_object_pose_keys_available": case["hdf5_metadata"].get("first_obs", {}).get("object_position_keys", []),
                "object_pose_match_to_demo": "not_available_hdf5_obs_lacks_object_pos_keys" if not case["hdf5_metadata"].get("first_obs", {}).get("object_position_keys") else "available_in_variant_metrics",
            },
            "variants": [],
        }
        total_steps = 0
        for variant in case["variants"]:
            variant_summary = _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=camera_size,
                init_state=case["init_state"],
                variant=variant,
                instruction=case["instruction"],
            )
            case_summary["variants"].append(variant_summary)
            total_steps += int(variant_summary.get("steps_performed") or 0)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = len(case_summary["variants"])
        report["result"]["full_demo_expert_replay_happened"] = any(v.get("variant") == "hdf5_expert_replay_exact_init" and int(v.get("steps_performed") or 0) > 0 for v in case_summary["variants"])
        report["cases"].append(case_summary)
        report["decision"] = _classify(report)
        report["result"]["passed"] = all(variant.get("passed") for variant in case_summary["variants"])
        report["result"]["reason"] = "bounded full-demo expert replay sanity completed" if report["result"]["passed"] else "one or more replay variants failed"
        report["recommended_next_step"] = {
            "bridge_init_action_convention_green": "A. bounded longer-horizon fixed-prior method rollout, using matched init states and the validated expert horizon as the risk boundary.",
            "expert_replay_succeeds_but_timing_differs": "A. bounded longer-horizon fixed-prior method rollout, but document timing mismatch first.",
            "init_state_replay_blocker": "B. fix init-state replay before method rollout.",
            "action_convention_or_demo_mapping_mismatch": "C. fix action convention/controller bridge before method rollout.",
            "expert_replay_inconclusive_or_short": "C. inspect replay horizon/action convention before method rollout.",
        }.get(report["decision"]["blocker_classification"], "Inspect expert replay sanity before method rollout scaling.")
    except Exception as exc:
        report["result"]["reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["recommended_next_step"] = "Diagnose simulator, init-state, or action convention error before any method rollout."
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--report-json", default="reports/full_demo_expert_replay_sanity_report.json")
    parser.add_argument("--report-md", default="reports/full_demo_expert_replay_sanity_report.md")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-steps-cap", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args()
    report = run_full_demo_expert_replay_sanity(
        manifest_path=_as_path(args.manifest),
        readiness_report_path=_as_path(args.readiness_report),
        report_json=_as_path(args.report_json),
        report_md=_as_path(args.report_md),
        libero_root=_as_path(args.libero_root),
        robosuite_root=_as_path(args.robosuite_root),
        max_tasks=args.max_tasks,
        max_steps_cap=args.max_steps_cap,
        post_signal_margin=args.post_signal_margin,
        camera_size=args.camera_size,
    )
    if os.environ.get(TASK_GATE) == "1":
        summary = {
            "schema_version": report.get("schema_version"),
            "report_json": str(_as_path(args.report_json)),
            "result": report.get("result"),
            "decision": report.get("decision"),
            "recommended_next_step": report.get("recommended_next_step"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
