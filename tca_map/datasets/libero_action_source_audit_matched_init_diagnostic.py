"""Action-source audit plus matched-init candidate replay diagnostic.

This module checks whether rollout actions are online policy outputs or are
derived from future HDF5 demonstration actions. It can then run a bounded
matched-init replay diagnostic, explicitly labeled as candidate replay when the
actions come from offline HDF5 candidates. It performs no training, no model
loading, no VLA inference, no GPU jobs, no downloads, no OpenVLA-OFT, no full
benchmark rollout, and no paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
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
from tca_map.datasets.libero_full_demo_expert_replay_sanity import (
    _controller_summary,
    _gripper_timing,
    _read_demo_full,
    _safe_l2,
    _sim_state_array,
)
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _distance_delta,
    _extract_eef,
    _extract_pos,
    _object_position_keys,
)

SCHEMA_VERSION = "2026-07-06.libero_action_source_audit_matched_init_diagnostic.v1"
TASK_GATE = "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT"
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
    "ALLOW_FULL_DEMO_EXPERT_REPLAY",
    "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
)
METHOD_VARIANTS = {
    "actionmap_style_target_agnostic_mean",
    "fixed_prior_tca_candidate_replay",
    "hard_learned_target_tca_candidate_replay",
}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = report.get("result", {})
    decision = report.get("decision") or {}
    lines = [
        "# Action-Source Audit And Matched-Init Diagnostic",
        "",
        "This is bounded diagnostic evidence only. It is not closed-loop policy success, standard success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- diagnostic passed: `{result.get('passed')}`",
        f"- rollout happened: `{report['policy']['diagnostic_rollouts_performed']}`",
        f"- action-source audit happened: `{result.get('action_source_audit_happened')}`",
        f"- evidence type: `{decision.get('evidence_type')}`",
        f"- blocker classification: `{decision.get('blocker_classification')}`",
        f"- fixed-prior valid rollout support: `{decision.get('fixed_prior_valid_rollout_support')}`",
        f"- fixed-prior uses future HDF5 expert actions: `{decision.get('fixed_prior_uses_future_hdf5_expert_actions')}`",
        f"- recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_l2(left: np.ndarray, right: np.ndarray) -> float:
    steps = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    if steps == 0 or width == 0:
        return 0.0
    diff = left[:steps, :width] - right[:steps, :width]
    return round(float(np.mean(np.linalg.norm(diff, axis=1))), 9)


def _match_stats(actions: np.ndarray, reference: np.ndarray, *, near_tol: float = 1e-6) -> dict[str, Any]:
    steps = min(actions.shape[0], reference.shape[0])
    width = min(actions.shape[1], reference.shape[1])
    if steps == 0 or width == 0:
        return {
            "steps_compared": 0,
            "exact_match_rate": 0.0,
            "near_match_rate": 0.0,
            "mean_l2": None,
            "max_l2": None,
            "near_tol": near_tol,
        }
    diff = actions[:steps, :width] - reference[:steps, :width]
    row_l2 = np.linalg.norm(diff, axis=1)
    exact = np.all(actions[:steps, :width] == reference[:steps, :width], axis=1)
    near = row_l2 <= near_tol
    return {
        "steps_compared": int(steps),
        "exact_match_rate": round(float(np.mean(exact)), 9),
        "near_match_rate": round(float(np.mean(near)), 9),
        "mean_l2": round(float(np.mean(row_l2)), 9),
        "max_l2": round(float(np.max(row_l2)), 9),
        "near_tol": near_tol,
    }


def _source_template(name: str) -> dict[str, Any]:
    templates = {
        "zero_action_exact_init": {
            "action_source_class": "constant_diagnostic_control",
            "candidate_provenance": "programmatic_zero_action",
            "uses_future_hdf5_actions_unavailable_at_deployment": False,
            "online_generated_policy_action": False,
            "model_head_decoded_action": False,
            "selected_from_offline_candidate_set": False,
            "copied_from_hdf5_expert_action_at_same_timestep": False,
            "mean_or_aggregate_of_hdf5_actions": False,
            "oracle_or_upper_bound_action": False,
        },
        "hdf5_expert_replay_exact_init": {
            "action_source_class": "copied_from_hdf5_expert_action_at_same_timestep",
            "candidate_provenance": "positive_demo_hdf5_actions",
            "uses_future_hdf5_actions_unavailable_at_deployment": True,
            "online_generated_policy_action": False,
            "model_head_decoded_action": False,
            "selected_from_offline_candidate_set": True,
            "copied_from_hdf5_expert_action_at_same_timestep": True,
            "mean_or_aggregate_of_hdf5_actions": False,
            "oracle_or_upper_bound_action": True,
        },
        "actionmap_style_target_agnostic_mean": {
            "action_source_class": "mean_aggregate_of_hdf5_positive_and_counterfactual_actions",
            "candidate_provenance": "mean_of_positive_and_counterfactual_hdf5_action_sequences",
            "uses_future_hdf5_actions_unavailable_at_deployment": True,
            "online_generated_policy_action": False,
            "model_head_decoded_action": False,
            "selected_from_offline_candidate_set": True,
            "copied_from_hdf5_expert_action_at_same_timestep": False,
            "mean_or_aggregate_of_hdf5_actions": True,
            "oracle_or_upper_bound_action": False,
        },
        "fixed_prior_tca_candidate_replay": {
            "action_source_class": "copied_from_hdf5_expert_action_at_same_timestep",
            "candidate_provenance": "fixed_target_prior_selects_positive_demo_hdf5_actions",
            "uses_future_hdf5_actions_unavailable_at_deployment": True,
            "online_generated_policy_action": False,
            "model_head_decoded_action": False,
            "selected_from_offline_candidate_set": True,
            "copied_from_hdf5_expert_action_at_same_timestep": True,
            "mean_or_aggregate_of_hdf5_actions": False,
            "oracle_or_upper_bound_action": False,
        },
        "hard_learned_target_tca_candidate_replay": {
            "action_source_class": "selected_from_offline_counterfactual_hdf5_candidate_set",
            "candidate_provenance": "counterfactual_demo_hdf5_actions_as_wrong-target_candidate_proxy",
            "uses_future_hdf5_actions_unavailable_at_deployment": True,
            "online_generated_policy_action": False,
            "model_head_decoded_action": False,
            "selected_from_offline_candidate_set": True,
            "copied_from_hdf5_expert_action_at_same_timestep": False,
            "mean_or_aggregate_of_hdf5_actions": False,
            "oracle_or_upper_bound_action": False,
        },
    }
    return dict(templates[name])


def _audit_variant_actions(
    *,
    name: str,
    actions: np.ndarray,
    expert_actions: np.ndarray,
    counter_actions: np.ndarray,
    actionmap_actions: np.ndarray,
) -> dict[str, Any]:
    audit = _source_template(name)
    audit["action_shape"] = list(actions.shape)
    audit["action_stats"] = _action_stats(actions)
    audit["match_to_hdf5_expert"] = _match_stats(actions, expert_actions)
    audit["match_to_counterfactual_hdf5"] = _match_stats(actions, counter_actions)
    audit["match_to_actionmap_mean"] = _match_stats(actions, actionmap_actions)
    audit["l2_to_hdf5_expert_mean"] = audit["match_to_hdf5_expert"]["mean_l2"]
    audit["l2_to_actionmap_mean"] = audit["match_to_actionmap_mean"]["mean_l2"]
    audit["valid_for_closed_loop_policy_claim"] = False if audit["uses_future_hdf5_actions_unavailable_at_deployment"] else name == "zero_action_exact_init"
    audit["valid_for_method_rollout_claim"] = bool(name in METHOD_VARIANTS and not audit["uses_future_hdf5_actions_unavailable_at_deployment"])
    if name == "fixed_prior_tca_candidate_replay":
        if audit["match_to_hdf5_expert"]["near_match_rate"] == 1.0:
            audit["fixed_prior_action_equals"] = "hdf5_expert_action"
        elif audit["match_to_actionmap_mean"]["near_match_rate"] == 1.0:
            audit["fixed_prior_action_equals"] = "actionmap_mean_action"
        else:
            audit["fixed_prior_action_equals"] = "distinct_candidate"
    return audit


def build_action_source_audit_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 300,
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
    counter = _read_demo_full(_as_path(pair["counterfactual_demo_file"]), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    horizon = min(int(positive["target_horizon"]), int(counter["full_action_steps"]), int(max_steps_cap))
    expert_actions = np.asarray(positive["full_actions"][:horizon], dtype=np.float64)
    counter_actions = np.asarray(counter["full_actions"][:horizon], dtype=np.float64)
    zero_actions = np.zeros_like(expert_actions)
    actionmap_actions = np.clip((expert_actions + counter_actions) / 2.0, -1.0, 1.0)
    variants = [
        {"name": "zero_action_exact_init", "claim_role": "negative_control", "actions": zero_actions},
        {"name": "hdf5_expert_replay_exact_init", "claim_role": "expert_replay_sanity", "actions": expert_actions},
        {"name": "actionmap_style_target_agnostic_mean", "claim_role": "baseline_candidate_replay_diagnostic", "actions": actionmap_actions},
        {"name": "fixed_prior_tca_candidate_replay", "claim_role": "fixed_prior_candidate_replay_diagnostic", "actions": expert_actions},
        {"name": "hard_learned_target_tca_candidate_replay", "claim_role": "wrong_target_candidate_replay_diagnostic", "actions": counter_actions},
    ]
    action_source_audit = {
        variant["name"]: _audit_variant_actions(
            name=variant["name"],
            actions=np.asarray(variant["actions"], dtype=np.float64),
            expert_actions=expert_actions,
            counter_actions=counter_actions,
            actionmap_actions=actionmap_actions,
        )
        for variant in variants
    }
    return {
        "pair_id": pair["pair_id"],
        "suite": pair.get("suite") or "libero_10",
        "task_id": pair["positive_task_id"],
        "instruction": pair["positive_instruction"],
        "counterfactual_task_id": pair["counterfactual_task_id"],
        "counterfactual_instruction": pair["counterfactual_instruction"],
        "positive_demo_path": positive["path"],
        "counterfactual_demo_path": counter["path"],
        "demo_name": positive["demo_name"],
        "init_state": positive["init_state"],
        "target_horizon": horizon,
        "hdf5_metadata": {
            "positive_first_signal_index": positive["first_signal_index"],
            "positive_first_positive_reward_index": positive["first_reward_index"],
            "positive_first_done_index": positive["first_done_index"],
            "positive_full_action_steps": positive["full_action_steps"],
            "counterfactual_first_signal_index": counter["first_signal_index"],
            "counterfactual_full_action_steps": counter["full_action_steps"],
            "states0_l2_to_init_state": positive["states0_l2_to_init_state"],
            "first_obs": positive["first_obs"],
        },
        "action_source_audit": action_source_audit,
        "action_pair_distances": {
            "actionmap_vs_fixed_prior_mean_l2": _mean_l2(actionmap_actions, expert_actions),
            "hard_learned_vs_fixed_prior_mean_l2": _mean_l2(counter_actions, expert_actions),
            "hard_learned_vs_actionmap_mean_l2": _mean_l2(counter_actions, actionmap_actions),
        },
        "variants": variants,
    }


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    variant: dict[str, Any],
    instruction: str,
    counterfactual_instruction: str,
) -> dict[str, Any]:
    actions = np.asarray(variant["actions"], dtype=np.float64)
    summary: dict[str, Any] = {
        "variant": variant["name"],
        "claim_role": variant["claim_role"],
        "evidence_type": "candidate_replay_diagnostic_not_closed_loop_policy",
        "action_shape": list(actions.shape),
        "env_action_shape": 7,
        "action_stats": _action_stats(actions),
        "gripper_timing": _gripper_timing(actions),
        "env_created": False,
        "reset_ok": False,
        "set_init_state_used": True,
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
        "done_indices": [],
        "success_indices": [],
        "reward_trajectory_summary": {"length": 0, "nonzero_indices": [], "first_10": [], "last_10": []},
        "available_obs_keys": [],
        "camera_keys": [],
        "state_keys": [],
        "target_key_audit": None,
        "target_directed_movement": None,
        "wrong_target_movement": None,
        "target_directed_movement_score": None,
        "object_movement": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "controller": None,
        "error": None,
    }
    env = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        summary["controller"] = _controller_summary(env)
        env.seed(0)
        env.reset()
        summary["reset_ok"] = True
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
        wrong_audit = _best_object_key(obs, counterfactual_instruction)
        target_key = target_audit["best_key"]
        wrong_key = wrong_audit["best_key"]
        target_start = _extract_pos(obs, target_key)
        wrong_start = _extract_pos(obs, wrong_key)
        summary["target_key_audit"] = target_audit
        first_10: list[float] = []
        last_10: deque[float] = deque(maxlen=10)
        nonzero: list[int] = []
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
            rounded_reward = round(reward_value, 6)
            if len(first_10) < 10:
                first_10.append(rounded_reward)
            last_10.append(rounded_reward)
            if reward_value > 0.0:
                nonzero.append(index)
                if summary["first_positive_reward_index"] is None:
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
        summary["reward_trajectory_summary"] = {
            "length": int(summary["steps_performed"]),
            "nonzero_indices": nonzero[:20],
            "first_10": first_10,
            "last_10": list(last_10),
        }
        summary["eef_final"] = _extract_eef(obs)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            summary["eef_displacement_l2"] = round(float(np.linalg.norm(np.asarray(summary["eef_final"]) - np.asarray(summary["eef_start"]))), 6)
        target_final = _extract_pos(obs, target_key)
        wrong_final = _extract_pos(obs, wrong_key)
        target_delta = _distance_delta(summary["eef_start"], summary["eef_final"], target_start, target_final)
        wrong_delta = _distance_delta(summary["eef_start"], summary["eef_final"], wrong_start, wrong_final)
        summary["target_directed_movement"] = target_delta
        summary["wrong_target_movement"] = wrong_delta
        if target_delta.get("available") and wrong_delta.get("available"):
            summary["target_directed_movement_score"] = round(-float(target_delta["distance_change"]) + float(wrong_delta["distance_change"]), 6)
        elif target_delta.get("available"):
            summary["target_directed_movement_score"] = round(-float(target_delta["distance_change"]), 6)
        else:
            summary["target_directed_movement_score"] = "not_available_missing_target_or_eef_position"
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
    except Exception as exc:
        summary["error"] = _compact(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    summary["passed"] = bool(
        summary["env_created"]
        and summary["reset_ok"]
        and summary["set_init_state_ok"]
        and summary["steps_performed"] > 0
        and summary["error"] is None
    )
    return summary


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    case = (report.get("cases") or [{}])[0]
    audits = case.get("action_source_audit", {})
    variants = {item.get("variant"): item for item in case.get("matched_init_rollout_results", [])}
    fixed_audit = audits.get("fixed_prior_tca_candidate_replay", {})
    actionmap_audit = audits.get("actionmap_style_target_agnostic_mean", {})
    method_audits = [audit for name, audit in audits.items() if name in METHOD_VARIANTS]
    any_method_future = any(bool(audit.get("uses_future_hdf5_actions_unavailable_at_deployment")) for audit in method_audits)
    fixed_uses_future = bool(fixed_audit.get("uses_future_hdf5_actions_unavailable_at_deployment"))
    fixed_exact_expert = fixed_audit.get("match_to_hdf5_expert", {}).get("near_match_rate") == 1.0
    fixed = variants.get("fixed_prior_tca_candidate_replay", {})
    actionmap = variants.get("actionmap_style_target_agnostic_mean", {})
    expert = variants.get("hdf5_expert_replay_exact_init", {})
    fixed_success = bool(fixed.get("final_success") or fixed.get("done_seen") or float(fixed.get("reward_sum") or 0.0) > 0.0)
    actionmap_success = bool(actionmap.get("final_success") or actionmap.get("done_seen") or float(actionmap.get("reward_sum") or 0.0) > 0.0)
    expert_success = bool(expert.get("final_success") or expert.get("done_seen") or float(expert.get("reward_sum") or 0.0) > 0.0)
    if any_method_future:
        blocker = "expert_action_leakage_candidate_replay_only"
        evidence_type = "candidate_replay_diagnostic_not_closed_loop_policy"
        valid_support = False
    elif fixed_success and not actionmap_success:
        blocker = "non_leaking_matched_init_method_support"
        evidence_type = "matched_init_method_rollout_diagnostic"
        valid_support = True
    elif not fixed_success and not actionmap_success and expert_success:
        blocker = "candidate_quality_or_action_selection_failure"
        evidence_type = "matched_init_method_rollout_diagnostic"
        valid_support = False
    else:
        blocker = "matched_init_rollout_inconclusive"
        evidence_type = "matched_init_method_rollout_diagnostic"
        valid_support = False
    return {
        "blocker_classification": blocker,
        "evidence_type": evidence_type,
        "expert_replay_succeeded": expert_success,
        "fixed_prior_succeeded": fixed_success,
        "actionmap_succeeded": actionmap_success,
        "fixed_prior_valid_rollout_support": valid_support,
        "fixed_prior_uses_future_hdf5_expert_actions": fixed_uses_future,
        "fixed_prior_action_equals": fixed_audit.get("fixed_prior_action_equals"),
        "fixed_prior_near_match_rate_to_expert": fixed_audit.get("match_to_hdf5_expert", {}).get("near_match_rate"),
        "actionmap_near_match_rate_to_expert": actionmap_audit.get("match_to_hdf5_expert", {}).get("near_match_rate"),
        "any_method_variant_uses_future_hdf5_actions": any_method_future,
        "fixed_prior_exact_expert_replay": fixed_exact_expert,
        "actionmap_vs_fixed_prior_mean_l2": case.get("action_pair_distances", {}).get("actionmap_vs_fixed_prior_mean_l2"),
        "hard_learned_vs_fixed_prior_mean_l2": case.get("action_pair_distances", {}).get("hard_learned_vs_fixed_prior_mean_l2"),
        "recommended_evidence_wording": (
            "candidate-replay diagnostic / action-bridge evidence only; not closed-loop policy rollout success"
            if any_method_future
            else "matched-init bounded method rollout diagnostic"
        ),
    }


def run_action_source_audit_matched_init_diagnostic(
    *,
    manifest_path: Path,
    readiness_report_path: Path,
    report_json: Path,
    report_md: Path,
    libero_root: Path,
    robosuite_root: Path,
    max_steps_cap: int = 300,
    post_signal_margin: int = 20,
    camera_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    readiness = _load_json(readiness_report_path) if readiness_report_path.exists() else {}
    policy = _policy()
    policy["bounded_action_source_audit_matched_init_diagnostic"] = True
    policy["task_local_gate_required"] = f"{TASK_GATE}=1"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "inputs": {
            "manifest_path": str(manifest_path),
            "readiness_report_path": str(readiness_report_path),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "max_steps_cap": max_steps_cap,
            "post_signal_margin": post_signal_margin,
            "camera_size": camera_size,
        },
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "result": {"passed": False, "reason": None, "total_steps_performed": 0, "variant_count": 0, "action_source_audit_happened": False},
        "forbidden_gates_set": forbidden,
        "decision": None,
        "elapsed_seconds": None,
        "recommended_next_step": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for this bounded action-source audit diagnostic")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("fixed-prior rollout readiness gate is not green/authorized")
    if max_steps_cap < 1 or max_steps_cap > 300:
        stop_reasons.append("max_steps_cap must be between 1 and 300")
    if post_signal_margin < 0 or post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if camera_size < 16 or camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    try:
        case = build_action_source_audit_case(
            manifest_path,
            max_steps_cap=max_steps_cap,
            post_signal_margin=post_signal_margin,
        )
    except Exception as exc:
        case = None
        stop_reasons.append(f"failed to build action-source audit case: {type(exc).__name__}: {exc}")
    if stop_reasons:
        report["result"]["reason"] = "; ".join(stop_reasons)
        report["recommended_next_step"] = "Resolve listed blockers before action-source audit and matched-init diagnostic."
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
            "counterfactual_demo_path": case["counterfactual_demo_path"],
            "demo_name": case["demo_name"],
            "bddl_file": str(bddl_file),
            "target_horizon": case["target_horizon"],
            "hdf5_metadata": case["hdf5_metadata"],
            "action_source_audit": case["action_source_audit"],
            "action_pair_distances": case["action_pair_distances"],
            "matched_init_rollout_results": [],
        }
        total_steps = 0
        for variant in case["variants"]:
            result = _run_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=camera_size,
                init_state=case["init_state"],
                variant=variant,
                instruction=case["instruction"],
                counterfactual_instruction=case["counterfactual_instruction"],
            )
            case_summary["matched_init_rollout_results"].append(result)
            total_steps += int(result.get("steps_performed") or 0)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = len(case_summary["matched_init_rollout_results"])
        report["result"]["action_source_audit_happened"] = True
        report["cases"].append(case_summary)
        report["decision"] = _classify(report)
        report["result"]["passed"] = all(item.get("passed") for item in case_summary["matched_init_rollout_results"])
        report["result"]["reason"] = "bounded action-source audit and matched-init diagnostic completed" if report["result"]["passed"] else "one or more matched-init variants failed"
        report["recommended_next_step"] = {
            "expert_action_leakage_candidate_replay_only": "D. paper-readiness package with honest rollout caveat, or B. online action-generation bridge before any method rollout claim.",
            "non_leaking_matched_init_method_support": "A. second matched-init task diagnostic under the same audit rules.",
            "candidate_quality_or_action_selection_failure": "C. candidate quality/action selection diagnosis before scaling.",
            "matched_init_rollout_inconclusive": "C. inspect action source and target-directed metrics before scaling.",
        }.get(report["decision"]["blocker_classification"], "Inspect action-source audit before any rollout claim.")
    except Exception as exc:
        report["result"]["reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["recommended_next_step"] = "Diagnose simulator, action-source, or matched-init error before any method rollout."
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--report-json", default="reports/action_source_audit_matched_init_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/action_source_audit_matched_init_diagnostic_report.md")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-steps-cap", type=int, default=300)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args()
    report = run_action_source_audit_matched_init_diagnostic(
        manifest_path=_as_path(args.manifest),
        readiness_report_path=_as_path(args.readiness_report),
        report_json=_as_path(args.report_json),
        report_md=_as_path(args.report_md),
        libero_root=_as_path(args.libero_root),
        robosuite_root=_as_path(args.robosuite_root),
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


