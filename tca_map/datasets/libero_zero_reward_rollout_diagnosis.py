"""Zero-reward rollout diagnosis for bounded fixed-prior LIBERO diagnostics.

This module compares short-horizon zero-action, ActionMap-style candidate
actions, HDF5 expert replay, and fixed-prior TCA proxy actions. It is bounded
diagnostic evidence only: no training, no model loading, no VLA inference, no
GPU jobs, no downloads, no OpenVLA-OFT, and no paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    _mean_l1,
    _policy,
    _range,
    _write_json,
    _write_markdown,
)

SCHEMA_VERSION = "2026-07-06.libero_zero_reward_rollout_diagnosis.v1"
TASK_GATE = "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS"
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
)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    stop = {"the", "a", "an", "and", "in", "on", "of", "to", "it", "put", "pick", "up", "turn", "close", "open"}
    return {token for token in tokens if token not in stop and not token.isdigit()}


def _object_position_keys(obs: Any) -> list[str]:
    if not isinstance(obs, dict):
        return []
    keys = []
    for key in obs:
        if not isinstance(key, str) or not key.endswith("_pos"):
            continue
        if key.startswith("robot") or key in {"ee_pos", "eef_pos"}:
            continue
        keys.append(key)
    return sorted(keys)


def _best_object_key(obs: Any, instruction: str) -> dict[str, Any]:
    keys = _object_position_keys(obs)
    instruction_tokens = _tokenize(instruction)
    scored = []
    for key in keys:
        key_tokens = _tokenize(key.replace("_pos", ""))
        overlap = sorted(instruction_tokens & key_tokens)
        scored.append({"key": key, "score": len(overlap), "overlap": overlap})
    scored.sort(key=lambda item: (-int(item["score"]), str(item["key"])))
    best = scored[0] if scored and int(scored[0]["score"]) > 0 else None
    return {
        "instruction": instruction,
        "instruction_tokens": sorted(instruction_tokens),
        "available_object_position_keys": keys,
        "best_key": best["key"] if best else None,
        "best_score": int(best["score"]) if best else 0,
        "best_overlap": best["overlap"] if best else [],
        "all_scores": scored,
    }


def _extract_pos(obs: Any, key: str | None) -> list[float] | None:
    if not key or not isinstance(obs, dict) or key not in obs:
        return None
    arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return [float(value) for value in arr[:3]]


def _extract_eef(obs: Any) -> list[float] | None:
    if not isinstance(obs, dict):
        return None
    for key in ("robot0_eef_pos", "ee_pos", "eef_pos"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return [float(value) for value in arr[:3]]
    return None


def _distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))), 6)


def _distance_delta(start_eef: list[float] | None, final_eef: list[float] | None, start_obj: list[float] | None, final_obj: list[float] | None) -> dict[str, Any]:
    start_dist = _distance(start_eef, start_obj)
    final_dist = _distance(final_eef, final_obj)
    if start_dist is None or final_dist is None:
        return {"available": False, "start_distance": start_dist, "final_distance": final_dist, "distance_change": None}
    return {
        "available": True,
        "start_distance": start_dist,
        "final_distance": final_dist,
        "distance_change": round(final_dist - start_dist, 6),
    }


def _first_k_direction_consistency(actions: np.ndarray, start_eef: list[float] | None, target_pos: list[float] | None, *, k: int = 5) -> Any:
    if start_eef is None or target_pos is None or actions.size == 0:
        return "not_available_missing_eef_or_target_position"
    vector = np.asarray(target_pos, dtype=np.float64) - np.asarray(start_eef, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return "not_available_zero_target_vector"
    unit = vector / norm
    scores = []
    for row in actions[:k, :3]:
        row_norm = float(np.linalg.norm(row))
        if row_norm <= 1e-12:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(row / row_norm, unit)))
    return {
        "k": min(k, int(actions.shape[0])),
        "mean_cosine": round(float(np.mean(scores)), 6) if scores else None,
        "positive_fraction": round(float(np.mean([score > 0.0 for score in scores])), 6) if scores else None,
        "scores": [round(float(score), 6) for score in scores],
    }


def _read_demo(path: Path, max_steps: int) -> dict[str, Any]:
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
        actions = full_actions[:max_steps]
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        dones = np.asarray(demo["dones"], dtype=np.float64).reshape(-1) if "dones" in demo else np.zeros((full_actions.shape[0],))
        rewards = np.asarray(demo["rewards"], dtype=np.float64).reshape(-1) if "rewards" in demo else np.zeros((full_actions.shape[0],))
    done_indices = [int(index) for index, value in enumerate(dones) if float(value) > 0.5]
    reward_indices = [int(index) for index, value in enumerate(rewards) if float(value) > 0.0]
    return {
        "path": str(path),
        "demo_name": demo_name,
        "init_state": init_state,
        "actions": actions,
        "full_action_steps": int(full_actions.shape[0]),
        "first_done_index": done_indices[0] if done_indices else None,
        "first_positive_reward_index": reward_indices[0] if reward_indices else None,
        "reward_sum_first_max_steps": round(float(np.sum(rewards[:max_steps])), 6),
    }


def build_zero_reward_diagnosis_cases(
    manifest_path: Path,
    *,
    horizons: list[int],
    max_tasks: int = 1,
) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")
    max_horizon = max(horizons)
    cases = []
    for pair in manifest.get("counterfactual_pairs", [])[:max_tasks]:
        positive = _read_demo(_as_path(pair["positive_demo_file"]), max_horizon)
        counter = _read_demo(_as_path(pair["counterfactual_demo_file"]), max_horizon)
        steps_available = min(int(positive["actions"].shape[0]), int(counter["actions"].shape[0]), max_horizon)
        if steps_available < max_horizon:
            raise ValueError(f"pair {pair.get('pair_id')} has only {steps_available} actions, requested {max_horizon}")
        positive_actions = positive["actions"]
        counter_actions = counter["actions"]
        actionmap_actions = np.clip((positive_actions + counter_actions) / 2.0, -1.0, 1.0)
        zero_actions = np.zeros_like(positive_actions)
        variants = [
            ("zero_action", "negative_control", zero_actions),
            ("actionmap_style_target_agnostic_mean", "baseline_style_diagnostic", actionmap_actions),
            ("hdf5_expert_replay", "expert_replay_sanity", positive_actions),
            ("fixed_semantic_target_prior_tca_proxy", "main_fixed_prior_proxy_diagnostic", positive_actions),
        ]
        cases.append(
            {
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
                "horizons": horizons,
                "positive_demo_metadata": {
                    "full_action_steps": positive["full_action_steps"],
                    "first_done_index": positive["first_done_index"],
                    "first_positive_reward_index": positive["first_positive_reward_index"],
                    "reward_sum_first_max_steps": positive["reward_sum_first_max_steps"],
                },
                "action_diagnostics": {
                    "positive_demo_full_50": _action_stats(positive_actions),
                    "counterfactual_demo_full_50": _action_stats(counter_actions),
                    "candidate_positive_vs_counter_l1": round(_mean_l1(positive_actions, counter_actions), 6),
                    "actionmap_l1_to_positive": round(_mean_l1(actionmap_actions, positive_actions), 6),
                    "fixed_prior_l1_to_positive": 0.0,
                    "fixed_prior_actions_identical_to_expert_replay": True,
                },
                "variants": [
                    {"name": name, "claim_role": role, "actions": actions}
                    for name, role, actions in variants
                ],
            }
        )
    return cases


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    variant: dict[str, Any],
    horizon: int,
    instruction: str,
    counterfactual_instruction: str,
) -> dict[str, Any]:
    actions = np.asarray(variant["actions"][:horizon], dtype=np.float64)
    summary: dict[str, Any] = {
        "variant": variant["name"],
        "claim_role": variant["claim_role"],
        "horizon": horizon,
        "action_shape": list(actions.shape),
        "env_action_shape": 7,
        "action_stats": _action_stats(actions),
        "env_created": False,
        "reset_ok": False,
        "set_init_state_ok": False,
        "steps_performed": 0,
        "reward_sum": 0.0,
        "success_check": None,
        "done_seen": False,
        "available_obs_keys": [],
        "target_key_audit": None,
        "wrong_target_key_audit": None,
        "target_directed_movement": None,
        "wrong_target_movement": None,
        "target_directed_movement_score": None,
        "first_k_action_direction_consistency": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "object_position_keys_missing": False,
        "error": None,
    }
    env = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        env.seed(0)
        env.reset()
        summary["reset_ok"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["eef_start"] = _extract_eef(obs)
        if isinstance(obs, dict):
            summary["available_obs_keys"] = sorted(str(key) for key in obs.keys())[:60]
        target_audit = _best_object_key(obs, instruction)
        wrong_audit = _best_object_key(obs, counterfactual_instruction)
        target_key = target_audit["best_key"]
        wrong_key = wrong_audit["best_key"]
        target_start = _extract_pos(obs, target_key)
        wrong_start = _extract_pos(obs, wrong_key)
        summary["target_key_audit"] = target_audit
        summary["wrong_target_key_audit"] = wrong_audit
        summary["object_position_keys_missing"] = not bool(_object_position_keys(obs))
        summary["first_k_action_direction_consistency"] = _first_k_direction_consistency(actions, summary["eef_start"], target_start)
        for action in actions:
            obs, reward, done, info = env.step(action)
            summary["steps_performed"] += 1
            summary["reward_sum"] += float(reward)
            summary["done_seen"] = bool(summary["done_seen"] or done)
        summary["eef_final"] = _extract_eef(obs)
        target_final = _extract_pos(obs, target_key)
        wrong_final = _extract_pos(obs, wrong_key)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            start = np.asarray(summary["eef_start"], dtype=np.float64)
            final = np.asarray(summary["eef_final"], dtype=np.float64)
            summary["eef_displacement_l2"] = round(float(np.linalg.norm(final - start)), 6)
        target_delta = _distance_delta(summary["eef_start"], summary["eef_final"], target_start, target_final)
        wrong_delta = _distance_delta(summary["eef_start"], summary["eef_final"], wrong_start, wrong_final)
        summary["target_directed_movement"] = target_delta
        summary["wrong_target_movement"] = wrong_delta
        if target_delta.get("available") and wrong_delta.get("available"):
            target_change = float(target_delta["distance_change"])
            wrong_change = float(wrong_delta["distance_change"])
            summary["target_directed_movement_score"] = round((-target_change) - (-wrong_change), 6)
        elif target_delta.get("available"):
            summary["target_directed_movement_score"] = round(-float(target_delta["distance_change"]), 6)
        else:
            summary["target_directed_movement_score"] = "not_available_missing_target_or_eef_position"
        try:
            summary["success_check"] = bool(env.check_success())
        except Exception:
            summary["success_check"] = None
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
        and summary["steps_performed"] == horizon
        and summary["error"] is None
    )
    return summary


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    variants = [
        variant
        for case in report.get("cases", [])
        for horizon in case.get("horizon_results", [])
        for variant in horizon.get("variants", [])
    ]
    expert = [item for item in variants if item.get("variant") == "hdf5_expert_replay"]
    fixed = [item for item in variants if item.get("variant") == "fixed_semantic_target_prior_tca_proxy"]
    actionmap = [item for item in variants if item.get("variant") == "actionmap_style_target_agnostic_mean"]
    any_success = any(bool(item.get("success_check")) for item in variants)
    expert_success = any(bool(item.get("success_check")) for item in expert)
    expert_reward = max((float(item.get("reward_sum") or 0.0) for item in expert), default=0.0)
    object_metric_available = any(isinstance(item.get("target_directed_movement"), dict) and item["target_directed_movement"].get("available") for item in variants)
    fixed_scores = {int(item["horizon"]): item.get("target_directed_movement_score") for item in fixed}
    actionmap_scores = {int(item["horizon"]): item.get("target_directed_movement_score") for item in actionmap}
    fixed_beats_actionmap = []
    for horizon, score in fixed_scores.items():
        base = actionmap_scores.get(horizon)
        if isinstance(score, (int, float)) and isinstance(base, (int, float)):
            fixed_beats_actionmap.append(float(score) > float(base))
    demo_meta = (report.get("cases") or [{}])[0].get("positive_demo_metadata", {})
    first_done = demo_meta.get("first_done_index")
    max_horizon = max((int(item.get("horizon") or 0) for item in variants), default=0)
    if not expert_success and expert_reward <= 0.0 and first_done is not None and int(first_done) > max_horizon:
        blocker = "sparse_reward_or_short_horizon"
    elif not expert_success and expert_reward <= 0.0:
        blocker = "init_state_or_action_convention_mismatch"
    elif expert_success and not any(bool(item.get("success_check")) for item in fixed):
        blocker = "policy_action_quality_issue"
    elif not object_metric_available:
        blocker = "insufficient_target_directed_metric_availability"
    elif fixed_beats_actionmap and any(fixed_beats_actionmap):
        blocker = "partial_target_directed_support_without_success"
    else:
        blocker = "target_prior_or_action_selection_issue"
    return {
        "blocker_classification": blocker,
        "expert_replay_success": expert_success,
        "expert_replay_max_reward_sum": round(expert_reward, 6),
        "any_variant_success": any_success,
        "object_metric_available": object_metric_available,
        "fixed_prior_target_directed_score_by_horizon": fixed_scores,
        "actionmap_target_directed_score_by_horizon": actionmap_scores,
        "fixed_prior_beats_actionmap_target_directed_by_horizon": fixed_beats_actionmap,
        "hdf5_first_done_index": first_done,
        "max_horizon_tested": max_horizon,
        "ten_step_zero_reward_expected": bool(first_done is not None and int(first_done) > 10),
    }


def run_zero_reward_rollout_diagnosis(
    *,
    manifest_path: Path,
    readiness_report_path: Path,
    report_json: Path,
    report_md: Path,
    libero_root: Path,
    robosuite_root: Path,
    horizons: list[int],
    max_tasks: int = 1,
    camera_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    readiness = _load_json(readiness_report_path) if readiness_report_path.exists() else {}
    policy = _policy()
    policy["bounded_zero_reward_rollout_diagnosis"] = True
    policy["task_local_gate_required"] = f"{TASK_GATE}=1"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "inputs": {
            "manifest_path": str(manifest_path),
            "readiness_report_path": str(readiness_report_path),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "horizons": horizons,
            "max_tasks": max_tasks,
            "camera_size": camera_size,
        },
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "result": {"passed": False, "reason": None, "total_steps_performed": 0, "variant_count": 0},
        "forbidden_gates_set": forbidden,
        "decision": None,
        "elapsed_seconds": None,
        "recommended_next_step": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for this bounded rollout diagnosis")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("fixed-prior rollout readiness gate is not green/authorized")
    if max_tasks < 1 or max_tasks > 2:
        stop_reasons.append("max_tasks must be between 1 and 2")
    if not horizons or max(horizons) > 50 or min(horizons) < 1:
        stop_reasons.append("horizons must be between 1 and 50")
    if camera_size < 16 or camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    try:
        cases = build_zero_reward_diagnosis_cases(manifest_path, horizons=horizons, max_tasks=max_tasks)
    except Exception as exc:
        cases = []
        stop_reasons.append(f"failed to build zero-reward diagnosis cases: {type(exc).__name__}: {exc}")
    if stop_reasons:
        report["result"]["reason"] = "; ".join(stop_reasons)
        report["recommended_next_step"] = "Resolve listed blockers before zero-reward rollout diagnosis."
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(report_json, report)
        _write_markdown(report_md, report)
        return report
    try:
        env_cls = _load_env_class(libero_root, robosuite_root)
        total_steps = 0
        variant_count = 0
        for case in cases:
            bddl_file = libero_root / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
            case_summary = {
                "pair_id": case["pair_id"],
                "task_id": case["task_id"],
                "instruction": case["instruction"],
                "counterfactual_task_id": case["counterfactual_task_id"],
                "counterfactual_instruction": case["counterfactual_instruction"],
                "demo_name": case["demo_name"],
                "positive_demo_metadata": case["positive_demo_metadata"],
                "action_diagnostics": case["action_diagnostics"],
                "bddl_file": str(bddl_file),
                "horizon_results": [],
            }
            for horizon in horizons:
                horizon_summary = {"horizon": horizon, "variants": []}
                for variant in case["variants"]:
                    result = _run_variant(
                        env_cls=env_cls,
                        bddl_file=bddl_file,
                        camera_size=camera_size,
                        init_state=case["init_state"],
                        variant=variant,
                        horizon=horizon,
                        instruction=case["instruction"],
                        counterfactual_instruction=case["counterfactual_instruction"],
                    )
                    horizon_summary["variants"].append(result)
                    total_steps += int(result["steps_performed"])
                    variant_count += 1
                case_summary["horizon_results"].append(horizon_summary)
            report["cases"].append(case_summary)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = variant_count
        report["decision"] = _classify(report)
        report["result"]["passed"] = all(
            variant.get("passed")
            for case in report["cases"]
            for horizon in case.get("horizon_results", [])
            for variant in horizon.get("variants", [])
        )
        report["result"]["reason"] = "zero-reward rollout diagnosis completed"
        report["recommended_next_step"] = {
            "sparse_reward_or_short_horizon": "Run a separately gated longer-horizon demonstration-aligned replay before method rollout scaling.",
            "init_state_or_action_convention_mismatch": "Fix init-state/action convention and rerun expert replay sanity.",
            "policy_action_quality_issue": "Diagnose candidate/action quality before scaling fixed-prior rollout.",
            "partial_target_directed_support_without_success": "Consider a bounded longer-horizon rollout only after documenting sparse-reward limits.",
            "insufficient_target_directed_metric_availability": "Implement better target-directed metrics before interpreting method rollout.",
            "target_prior_or_action_selection_issue": "Redesign target-prior/action selection before rollout scaling.",
        }.get(report["decision"]["blocker_classification"], "Inspect diagnosis before scaling rollout.")
    except Exception as exc:
        report["result"]["reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["recommended_next_step"] = "Diagnose simulator or bridge error before any larger rollout."
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    return report


def _parse_horizons(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--report-json", default="reports/zero_reward_rollout_diagnosis_report.json")
    parser.add_argument("--report-md", default="reports/zero_reward_rollout_diagnosis_report.md")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--horizons", default="10,25,50")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args()
    report = run_zero_reward_rollout_diagnosis(
        manifest_path=_as_path(args.manifest),
        readiness_report_path=_as_path(args.readiness_report),
        report_json=_as_path(args.report_json),
        report_md=_as_path(args.report_md),
        libero_root=_as_path(args.libero_root),
        robosuite_root=_as_path(args.robosuite_root),
        horizons=_parse_horizons(args.horizons),
        max_tasks=args.max_tasks,
        camera_size=args.camera_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if os.environ.get(TASK_GATE) == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
