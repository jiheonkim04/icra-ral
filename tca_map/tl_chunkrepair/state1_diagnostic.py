"""STATE 1 diagnostic for Temporal-Logic-Guided Action Chunk Repair.

This runner uses local LIBERO HDF5 expert chunks and bounded exact-init replay
to test whether a finite-state temporal monitor plus minimal chunk repair can
recover temporal manipulation failures beyond simple baselines. It performs no
training, model loading, VLA inference, GPU work, downloads, OpenVLA-OFT
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

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _as_path, _compact, _load_env_class, _load_json, _write_json
from tca_map.phase_locked import retiming


SCHEMA_VERSION = "2026-07-07.tl_chunkrepair_state1.v1"
TASK_GATE = "ALLOW_TL_CHUNKREPAIR_STATE1"
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
    "ALLOW_PHASE_LOCKED_RETIMING",
)

PERTURBATION_NAMES = (
    "early_gripper_release",
    "delayed_gripper_close",
    "lift_before_grasp",
    "transport_with_gripper_open",
    "premature_place_release",
    "chunk_truncation",
    "phase_skip",
    "inserted_unsafe_contact_action",
)
BASELINES = (
    "no_repair",
    "clipping_only",
    "safety_only_one_step_filter",
    "gripper_only_timing_fix",
    "fixed_delay_shift",
    "linear_time_warp",
    "abort_to_stop",
    "repeat_last_hold",
    "tl_chunkrepair",
)


def _round(value: float | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _first_true(values: np.ndarray) -> int | None:
    for index, value in enumerate(np.asarray(values, dtype=bool).reshape(-1)):
        if bool(value):
            return int(index)
    return None


def _safe_index(value: Any, default: int, horizon: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(np.clip(int(value), 0, max(0, horizon - 1)))
    return int(np.clip(default, 0, max(0, horizon - 1)))


def _release_transitions(grip: np.ndarray) -> list[int]:
    return [int(i) for i in range(1, len(grip)) if float(grip[i - 1]) >= 0.0 and float(grip[i]) < 0.0]


def _close_transitions(grip: np.ndarray) -> list[int]:
    return [int(i) for i in range(1, len(grip)) if float(grip[i - 1]) < 0.0 and float(grip[i]) >= 0.0]


def _event_indices(actions: np.ndarray, anchors: dict[str, Any] | None = None) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    horizon = int(actions.shape[0]) if actions.ndim == 2 else 0
    if horizon == 0 or actions.shape[1] < 7:
        return {"horizon": horizon, "observable": False}
    grip = actions[:, 6]
    close = _first_true(grip >= 0.0)
    releases = _release_transitions(grip)
    closes = _close_transitions(grip)
    anchor_lift = (anchors or {}).get("lift_index")
    z_motion = np.abs(actions[:, 2]) > 0.025
    fallback_lift = _first_true(z_motion)
    lift = anchor_lift if isinstance(anchor_lift, (int, np.integer)) else fallback_lift
    if lift is None and close is not None:
        after_close = np.where(np.linalg.norm(actions[:, :3], axis=1) > 0.04)[0]
        lift = int(after_close[0]) if after_close.size else None
    anchor_release = (anchors or {}).get("safe_release_index")
    safe_release = releases[-1] if releases else None
    if isinstance(anchor_release, (int, np.integer)):
        safe_release = int(anchor_release) if safe_release is None else max(int(safe_release), int(anchor_release))
    first_release = releases[0] if releases else None
    transport_start = min([idx for idx in (close, lift) if idx is not None], default=0)
    transport_stop = safe_release if safe_release is not None else horizon
    transport_open = []
    for index in range(int(transport_start), int(np.clip(transport_stop, 0, horizon))):
        if float(grip[index]) < 0.0 and float(np.linalg.norm(actions[index, :3])) > 0.02:
            transport_open.append(int(index))
    unsafe_contact = [
        int(index)
        for index in range(0, int(np.clip(close if close is not None else lift if lift is not None else horizon, 0, horizon)))
        if float(np.linalg.norm(actions[index, :2])) > 0.95
    ]
    return {
        "horizon": horizon,
        "observable": True,
        "gripper_close_index": close,
        "close_transitions": closes,
        "first_release_index": first_release,
        "safe_release_index": safe_release,
        "release_transitions": releases,
        "lift_index": lift,
        "transport_start_index": transport_start,
        "transport_stop_index": transport_stop,
        "transport_open_indices_first_12": transport_open[:12],
        "unsafe_contact_indices_first_12": unsafe_contact[:12],
    }


def monitor_chunk(actions: np.ndarray, anchors: dict[str, Any] | None = None) -> dict[str, Any]:
    events = _event_indices(actions, anchors)
    if not events.get("observable"):
        return {"observable": False, "events": events, "violations": {}, "violation_count": 0, "first_violation_index": None}
    close = events.get("gripper_close_index")
    lift = events.get("lift_index")
    first_release = events.get("first_release_index")
    safe_release = events.get("safe_release_index")
    transport_open = events.get("transport_open_indices_first_12") or []
    unsafe_contact = events.get("unsafe_contact_indices_first_12") or []
    violations = {
        "grasp_before_lift": close is None or (lift is not None and close > lift),
        "keep_grasp_until_placement": bool(first_release is not None and safe_release is not None and first_release < safe_release),
        "do_not_release_before_target_region": bool(first_release is not None and safe_release is not None and first_release < safe_release),
        "do_not_move_object_while_gripper_open": bool(transport_open),
        "avoid_forbidden_contact_before_safe_phase": bool(unsafe_contact),
        "mechanism_action_onset_order": bool(close is None or (lift is not None and close > lift) or (first_release is not None and lift is not None and first_release < lift)),
    }
    boundary_candidates = []
    if violations["grasp_before_lift"] and lift is not None:
        boundary_candidates.append(int(lift))
    if violations["keep_grasp_until_placement"] and first_release is not None:
        boundary_candidates.append(int(first_release))
    if transport_open:
        boundary_candidates.append(int(transport_open[0]))
    if unsafe_contact:
        boundary_candidates.append(int(unsafe_contact[0]))
    return {
        "observable": True,
        "events": events,
        "violations": violations,
        "violation_count": int(sum(bool(value) for value in violations.values())),
        "property_violation_rate": _round(float(np.mean(list(violations.values()))), 6),
        "first_violation_index": min(boundary_candidates) if boundary_candidates else None,
    }


def _zero_segment(actions: np.ndarray, indices: list[int]) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float64).copy()
    for index in indices:
        if 0 <= int(index) < out.shape[0]:
            out[int(index), :6] = 0.0
    return out


def _force_grasp_until_safe(actions: np.ndarray, monitor: dict[str, Any]) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float64).copy()
    events = monitor.get("events") or {}
    horizon = int(out.shape[0])
    close = events.get("gripper_close_index")
    lift = events.get("lift_index")
    safe_release = events.get("safe_release_index")
    start = min([int(v) for v in (close, lift) if isinstance(v, (int, np.integer))], default=0)
    stop = _safe_index(safe_release, horizon - 1, horizon) if safe_release is not None else horizon - 1
    if stop >= start:
        out[start : stop + 1, 6] = 1.0
    return out


def repair_tl_chunk(actions: np.ndarray, anchors: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(actions, dtype=np.float64)
    out = raw.copy()
    monitor = monitor_chunk(out, anchors)
    events = monitor.get("events") or {}
    violations = monitor.get("violations") or {}
    edits: list[str] = []
    if violations.get("grasp_before_lift") or violations.get("keep_grasp_until_placement") or violations.get("do_not_release_before_target_region") or violations.get("do_not_move_object_while_gripper_open"):
        out = _force_grasp_until_safe(out, monitor)
        edits.append("force_grasp_maintenance_until_safe_release")
    unsafe = events.get("unsafe_contact_indices_first_12") or []
    if unsafe:
        out = _zero_segment(out, [int(i) for i in unsafe])
        edits.append("remove_pregrasp_unsafe_contact_segment")
    close = events.get("gripper_close_index")
    lift = events.get("lift_index")
    if isinstance(close, int) and isinstance(lift, int) and lift < close:
        out[lift:close, :6] = 0.0
        out[lift:close, 6] = 1.0
        edits.append("hold_premature_lift_until_grasp")
    repaired_monitor = monitor_chunk(out, anchors)
    return np.clip(out, -1.0, 1.0), {
        "input_monitor": monitor,
        "output_monitor": repaired_monitor,
        "edits": edits or ["no_temporal_edit"],
        "first_violation_index": monitor.get("first_violation_index"),
    }


def _first_close(actions: np.ndarray) -> int:
    close = _event_indices(actions).get("gripper_close_index")
    return int(close) if isinstance(close, int) else max(1, actions.shape[0] // 4)


def _safe_release(actions: np.ndarray) -> int:
    releases = _release_transitions(np.asarray(actions[:, 6], dtype=np.float64))
    return int(releases[-1]) if releases else max(1, actions.shape[0] - 1)


def build_temporal_perturbations(actions: np.ndarray, anchors: dict[str, Any], *, offset_steps: int = 18, unsafe_span: int = 6) -> dict[str, dict[str, Any]]:
    actions = np.asarray(actions, dtype=np.float64)
    horizon = int(actions.shape[0])
    close = _safe_index(anchors.get("gripper_close_index"), _first_close(actions), horizon)
    lift = _safe_index(anchors.get("lift_index"), max(close + 1, horizon // 2), horizon)
    release = _safe_index(_safe_release(actions), horizon - 1, horizon)
    offset = int(offset_steps)
    out: dict[str, dict[str, Any]] = {}

    early = actions.copy()
    start = min(close + max(2, offset // 3), release)
    stop = min(release, start + max(2, offset // 2))
    early[start:stop, 6] = -1.0
    early[stop:release, 6] = 1.0
    out["early_gripper_release"] = {"actions": early, "family": "release_too_early", "offset_steps": offset}

    delayed = actions.copy()
    delayed[:, 6] = retiming._shift_sequence(actions, offset, dims=[6])[:, 6]
    out["delayed_gripper_close"] = {"actions": delayed, "family": "delayed_grasp", "offset_steps": offset}

    lift_before = delayed.copy()
    src_start = min(lift, horizon - 1)
    dst_start = max(0, close - offset)
    width = max(1, min(offset, horizon - src_start, close - dst_start if close > dst_start else offset))
    lift_before[dst_start : dst_start + width, :6] = actions[src_start : src_start + width, :6]
    out["lift_before_grasp"] = {"actions": lift_before, "family": "lift_before_grasp", "offset_steps": offset}

    open_transport = actions.copy()
    stop = min(release, lift + max(2, offset))
    open_transport[lift:stop, 6] = -1.0
    open_transport[stop:release, 6] = 1.0
    out["transport_with_gripper_open"] = {"actions": open_transport, "family": "open_transport", "offset_steps": offset}

    premature = actions.copy()
    place_src = max(lift, release - offset)
    place_dst = min(release - 1, lift + max(1, offset // 2))
    width = max(1, min(offset, horizon - place_src, horizon - place_dst))
    premature[place_dst : place_dst + width, :6] = actions[place_src : place_src + width, :6]
    stop = min(release, place_dst + max(2, offset // 2))
    premature[place_dst:stop, 6] = -1.0
    premature[stop:release, 6] = 1.0
    out["premature_place_release"] = {"actions": premature, "family": "premature_release_and_place", "offset_steps": offset}

    trunc = actions.copy()
    trunc[max(1, horizon - offset) :, :6] = 0.0
    trunc[max(1, horizon - offset) :, 6] = -1.0
    out["chunk_truncation"] = {"actions": trunc, "family": "truncated_completion", "offset_steps": offset}

    skip = actions.copy()
    skip_start = max(0, close)
    skip_src = min(horizon - 1, close + offset)
    skip[skip_start:, :] = actions[np.clip(np.arange(skip_start, horizon) + (skip_src - skip_start), 0, horizon - 1)]
    out["phase_skip"] = {"actions": skip, "family": "skip_safe_phase", "offset_steps": offset}

    unsafe = actions.copy()
    start = max(0, close - offset)
    stop = min(horizon, start + int(unsafe_span))
    unsafe[start:stop, :3] = np.asarray([1.0, -1.0, 0.0], dtype=np.float64)
    out["inserted_unsafe_contact_action"] = {"actions": unsafe, "family": "unsafe_pregrasp_contact", "offset_steps": offset, "unsafe_span": int(stop - start)}
    return out


def build_tl_chunkrepair_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 320,
    post_signal_margin: int = 20,
    offset_steps: int = 18,
    unsafe_span: int = 6,
) -> dict[str, Any]:
    case = retiming.build_phase_locked_case(
        manifest_path,
        max_steps_cap=max_steps_cap,
        post_signal_margin=post_signal_margin,
        offset_steps=offset_steps,
    )
    case["event_anchors"]["safe_release_index"] = _safe_release(np.asarray(case["actions"], dtype=np.float64))
    perturbations = build_temporal_perturbations(np.asarray(case["actions"], dtype=np.float64), case.get("event_anchors") or {}, offset_steps=offset_steps, unsafe_span=unsafe_span)
    case["perturbations"] = perturbations
    case["perturbation_parameters"]["unsafe_span"] = int(unsafe_span)
    return case


def _safety_one_step(actions: np.ndarray, anchors: dict[str, Any]) -> np.ndarray:
    monitor = monitor_chunk(actions, anchors)
    boundary = monitor.get("first_violation_index")
    if not isinstance(boundary, int):
        return np.asarray(actions, dtype=np.float64).copy()
    return _zero_segment(actions, [boundary])


def _abort_to_stop(actions: np.ndarray, anchors: dict[str, Any]) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float64).copy()
    boundary = monitor_chunk(out, anchors).get("first_violation_index")
    if isinstance(boundary, int):
        out[boundary:, :6] = 0.0
    return out


def _baseline_actions(baseline: str, expert: np.ndarray, perturbed: np.ndarray, perturbation: dict[str, Any], anchors: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if baseline == "no_repair":
        return np.asarray(perturbed, dtype=np.float64).copy(), {}
    if baseline == "clipping_only":
        return np.clip(perturbed, -1.0, 1.0), {}
    if baseline == "safety_only_one_step_filter":
        return np.clip(_safety_one_step(perturbed, anchors), -1.0, 1.0), {}
    if baseline == "gripper_only_timing_fix":
        repaired = _force_grasp_until_safe(perturbed, monitor_chunk(perturbed, anchors))
        return np.clip(repaired, -1.0, 1.0), {"repair": "gripper_only_force_closed_until_safe_release"}
    if baseline == "fixed_delay_shift":
        offset = int(perturbation.get("offset_steps") or 0)
        return np.clip(retiming._shift_sequence(perturbed, -offset), -1.0, 1.0), {}
    if baseline == "linear_time_warp":
        return np.clip(retiming._time_warp(perturbed, 0.9), -1.0, 1.0), {}
    if baseline == "abort_to_stop":
        return np.clip(_abort_to_stop(perturbed, anchors), -1.0, 1.0), {}
    if baseline == "repeat_last_hold":
        return np.clip(retiming._repeat_last_hold(perturbed), -1.0, 1.0), {}
    if baseline == "tl_chunkrepair":
        return repair_tl_chunk(perturbed, anchors)
    raise ValueError(f"unknown baseline: {baseline}")


def _edit_metrics(source: np.ndarray, repaired: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    repaired = np.asarray(repaired, dtype=np.float64)
    steps = min(source.shape[0], repaired.shape[0])
    if steps == 0:
        return {"action_l1_mean": 0.0, "changed_step_rate": 0.0, "changed_steps": 0}
    diff = np.abs(source[:steps] - repaired[:steps])
    changed = np.any(diff > 1e-9, axis=1)
    return {
        "action_l1_mean": _round(float(np.mean(diff)), 9),
        "action_l2_mean": _round(float(np.mean(np.linalg.norm(source[:steps] - repaired[:steps], axis=1))), 9),
        "changed_step_rate": _round(float(np.mean(changed)), 6),
        "changed_steps": int(np.sum(changed)),
    }


def _case_header(case: dict[str, Any], bddl_file: Path, exact_start: dict[str, Any]) -> dict[str, Any]:
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
        "event_anchors": case["event_anchors"],
        "perturbation_parameters": case["perturbation_parameters"],
        "temporal_properties": [
            "grasp_before_lift",
            "keep_grasp_until_placement",
            "do_not_release_before_target_region",
            "do_not_move_object_while_gripper_open",
            "avoid_forbidden_contact_before_safe_phase",
            "mechanism_action_onset_order",
        ],
        "exact_start": exact_start,
    }


def _run_replay_case(args: argparse.Namespace, report: dict[str, Any]) -> None:
    case = build_tl_chunkrepair_case(
        _as_path(args.manifest),
        max_steps_cap=args.max_steps_cap,
        post_signal_margin=args.post_signal_margin,
        offset_steps=args.offset_steps,
        unsafe_span=args.unsafe_span,
    )
    env_cls = _load_env_class(_as_path(args.libero_root), _as_path(args.robosuite_root))
    bddl_file = _as_path(args.libero_root) / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
    exact_start = retiming._inspect_start_pose(
        env_cls=env_cls,
        bddl_file=bddl_file,
        camera_size=args.camera_size,
        init_state=case["init_state"],
        instruction=case["instruction"],
        seed=args.seed,
    )
    case_summary = _case_header(case, bddl_file, exact_start)
    exact = retiming._run_exact_expert(
        env_cls=env_cls,
        bddl_file=bddl_file,
        camera_size=args.camera_size,
        init_state=case["init_state"],
        case=case,
        seed=args.seed,
    )
    exact["temporal_monitor"] = monitor_chunk(np.asarray(case["actions"], dtype=np.float64), case.get("event_anchors") or {})
    case_summary["exact_expert_replay"] = exact
    case_summary["perturbations"] = []
    total_steps = int(exact.get("steps_performed") or 0)
    variant_count = 1
    expert = np.asarray(case["actions"], dtype=np.float64)
    anchors = case.get("event_anchors") or {}
    for name in PERTURBATION_NAMES:
        perturbation = case["perturbations"][name]
        perturbed = np.asarray(perturbation["actions"], dtype=np.float64)
        pert_summary = {
            "name": name,
            "family": perturbation.get("family"),
            "offset_steps": perturbation.get("offset_steps"),
            "unsafe_span": perturbation.get("unsafe_span"),
            "input_monitor": monitor_chunk(perturbed, anchors),
            "variants": [],
        }
        for baseline in BASELINES:
            actions, repair_info = _baseline_actions(baseline, expert, perturbed, perturbation, anchors)
            result = retiming._run_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=args.camera_size,
                init_state=case["init_state"],
                case=case,
                perturbation_name=name,
                baseline=baseline,
                static_actions=actions,
                exact_start=exact_start,
                seed=args.seed,
            )
            result["claim_role"] = "tl_chunkrepair_method" if baseline == "tl_chunkrepair" else ("raw_perturbation_control" if baseline == "no_repair" else "simple_baseline")
            result["temporal_monitor"] = monitor_chunk(actions, anchors)
            result["repair_info"] = repair_info
            result["edit_metrics_vs_perturbed"] = _edit_metrics(perturbed, actions)
            result["safe_success"] = bool(retiming._success(result) and result["temporal_monitor"].get("violation_count") == 0)
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


def _method_totals(case: dict[str, Any]) -> dict[str, Any]:
    totals = {name: {"safe_success": 0, "success": 0, "reward": 0.0, "edit_l1": 0.0, "count": 0} for name in BASELINES}
    for perturbation in case.get("perturbations", []):
        for item in perturbation.get("variants", []):
            baseline = item.get("baseline")
            if baseline not in totals:
                continue
            totals[baseline]["safe_success"] += int(bool(item.get("safe_success")))
            totals[baseline]["success"] += int(retiming._success(item))
            totals[baseline]["reward"] += float(item.get("reward_sum") or 0.0)
            totals[baseline]["edit_l1"] += float((item.get("edit_metrics_vs_perturbed") or {}).get("action_l1_mean") or 0.0)
            totals[baseline]["count"] += 1
    for payload in totals.values():
        count = max(1, int(payload["count"]))
        payload["reward"] = _round(float(payload["reward"]), 6)
        payload["mean_edit_l1"] = _round(float(payload.pop("edit_l1")) / count, 9)
    return totals


def _best_simple(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    simple = [payload for name, payload in variants.items() if name != "tl_chunkrepair"]
    return max(simple, key=retiming._progress_tuple) if simple else {}


def _degraded(raw: dict[str, Any], exact: dict[str, Any]) -> bool:
    exact_success = retiming._success(exact)
    raw_success = retiming._success(raw)
    if exact_success and not raw_success:
        return True
    if exact_success and raw_success:
        exact_done = exact.get("first_done_index")
        raw_done = raw.get("first_done_index")
        if exact_done is not None and raw_done is not None:
            return int(raw_done) > int(exact_done)
        return False
    exact_reward = float(exact.get("reward_sum") or 0.0)
    raw_reward = float(raw.get("reward_sum") or 0.0)
    if exact_reward > raw_reward:
        return True
    exact_tuple = retiming._progress_tuple(exact)
    raw_tuple = retiming._progress_tuple(raw)
    return any(e > r + 1e-6 for e, r in zip(exact_tuple, raw_tuple))


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("cases"):
        return {"continue_or_kill": "blocked", "reason": "no replay case was built", "next_state": "resolve_tl_chunkrepair_blocker"}
    case = report["cases"][0]
    exact = case.get("exact_expert_replay") or {}
    replay_metric = bool(report.get("policy", {}).get("replay_or_rollout_performed"))
    exact_success = retiming._success(exact)
    perturbation_summaries = []
    degraded_count = 0
    tl_violation_reduction_count = 0
    tl_beats_best_count = 0
    simple_matches_count = 0
    tl_safe_success_count = 0
    for perturbation in case.get("perturbations", []):
        variants = {item["baseline"]: item for item in perturbation.get("variants", [])}
        raw = variants.get("no_repair", {})
        tl = variants.get("tl_chunkrepair", {})
        best_simple = _best_simple(variants)
        degraded = bool(raw and _degraded(raw, exact))
        degraded_count += int(degraded)
        raw_violations = int((raw.get("temporal_monitor") or {}).get("violation_count") or 0)
        tl_violations = int((tl.get("temporal_monitor") or {}).get("violation_count") or 0)
        violation_reduced = tl_violations < raw_violations
        tl_violation_reduction_count += int(violation_reduced)
        tl_beats_best = bool(tl and best_simple and retiming._beats(tl, best_simple))
        simple_matches = bool(tl and best_simple and retiming._matches_or_beats(best_simple, tl))
        tl_beats_best_count += int(tl_beats_best)
        simple_matches_count += int(simple_matches)
        tl_safe_success_count += int(bool(tl.get("safe_success")))
        perturbation_summaries.append(
            {
                "perturbation": perturbation.get("name"),
                "family": perturbation.get("family"),
                "raw_degraded": degraded,
                "raw_violation_count": raw_violations,
                "tl_violation_count": tl_violations,
                "tl_reduces_violations": violation_reduced,
                "tl_safe_success": bool(tl.get("safe_success")),
                "tl_progress_tuple": retiming._progress_tuple(tl) if tl else None,
                "raw_progress_tuple": retiming._progress_tuple(raw) if raw else None,
                "best_simple_baseline": best_simple.get("baseline"),
                "best_simple_safe_success": bool(best_simple.get("safe_success")),
                "best_simple_progress_tuple": retiming._progress_tuple(best_simple) if best_simple else None,
                "tl_beats_best_simple": tl_beats_best,
                "best_simple_matches_or_beats_tl": simple_matches,
            }
        )
    totals = _method_totals(case)
    simple_totals = {name: payload for name, payload in totals.items() if name != "tl_chunkrepair"}
    best_single_name = max(simple_totals, key=lambda name: (simple_totals[name]["safe_success"], simple_totals[name]["success"], simple_totals[name]["reward"])) if simple_totals else None
    best_single = simple_totals.get(best_single_name or "", {})
    tl_total = totals.get("tl_chunkrepair", {})
    beats_best_single = bool(
        tl_total
        and best_single
        and (
            int(tl_total["safe_success"]) > int(best_single["safe_success"])
            or (
                int(tl_total["safe_success"]) == int(best_single["safe_success"])
                and int(tl_total["success"]) > int(best_single["success"])
            )
        )
    )
    beats_per_failure_mode = bool(degraded_count > 0 and tl_beats_best_count == degraded_count)
    if not replay_metric:
        decision = "kill"
        reason = "No real replay/control metric was produced."
    elif not exact_success:
        decision = "kill"
        reason = "Exact-init expert replay did not succeed, so repair cannot be judged cleanly."
    elif degraded_count == 0:
        decision = "kill"
        reason = "Temporal perturbations did not meaningfully degrade replay."
    elif tl_violation_reduction_count == 0:
        decision = "kill"
        reason = "TL-ChunkRepair did not reduce temporal property violations."
    elif not beats_best_single:
        decision = "kill"
        reason = "TL-ChunkRepair did not beat the best single simple baseline."
    elif not beats_per_failure_mode:
        decision = "kill"
        reason = "TL-ChunkRepair did not beat the best per-failure-mode simple baseline."
    elif simple_matches_count > 0:
        decision = "kill"
        reason = "A simple baseline matched or beat TL-ChunkRepair on at least one degraded perturbation."
    else:
        decision = "continue"
        reason = "TL-ChunkRepair reduced violations, improved replay/control metrics, and beat both simple-baseline gates."
    return {
        "continue_or_kill": decision,
        "reason": reason,
        "next_state": "STATE 2: broaden across tasks and non-oracle predicate sources" if decision == "continue" else "archive_or_reframe_tl_chunkrepair",
        "exact_init_expert_replay_success": exact_success,
        "temporal_properties_observable": True,
        "perturbations_tested": len(case.get("perturbations", [])),
        "baselines_tested": list(BASELINES),
        "perturbations_degraded_replay_count": degraded_count,
        "tl_violation_reduction_count": tl_violation_reduction_count,
        "tl_safe_success_count": tl_safe_success_count,
        "tl_beats_best_simple_count": tl_beats_best_count,
        "simple_baseline_matches_or_beats_tl_count": simple_matches_count,
        "best_single_simple_baseline": best_single_name,
        "best_single_simple_totals": best_single,
        "tl_totals": tl_total,
        "tl_beats_best_single_simple_baseline": beats_best_single,
        "tl_beats_best_per_failure_mode_simple_baseline": beats_per_failure_mode,
        "method_totals": totals,
        "perturbation_summaries": perturbation_summaries,
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_tl_chunkrepair_state1": True,
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
        "# TL-ChunkRepair STATE 1 Result",
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
        f"- temporal properties tested: `{', '.join((case.get('temporal_properties') or []))}`",
        f"- perturbations tested: `{summary.get('perturbations_tested')}`",
        f"- baselines tested: `{', '.join(summary.get('baselines_tested') or [])}`",
        f"- perturbations degraded replay: `{summary.get('perturbations_degraded_replay_count')}`",
        f"- TL violation reductions: `{summary.get('tl_violation_reduction_count')}`",
        f"- TL safe-success count: `{summary.get('tl_safe_success_count')}`",
        f"- best single simple baseline: `{summary.get('best_single_simple_baseline')}`",
        f"- TL beats best single baseline: `{summary.get('tl_beats_best_single_simple_baseline')}`",
        f"- TL beats best per-failure baseline: `{summary.get('tl_beats_best_per_failure_mode_simple_baseline')}`",
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
        "## Perturbation Summary",
        "",
        "| perturbation | raw degraded | raw violations | TL violations | TL safe success | best simple | best simple safe success | TL beats best simple |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in summary.get("perturbation_summaries") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("perturbation")),
                    _md(item.get("raw_degraded")),
                    _md(item.get("raw_violation_count")),
                    _md(item.get("tl_violation_count")),
                    _md(item.get("tl_safe_success")),
                    str(item.get("best_simple_baseline")),
                    _md(item.get("best_simple_safe_success")),
                    _md(item.get("tl_beats_best_simple")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Method Totals", ""])
    lines.append("| method | safe success | success | reward sum | mean edit L1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, payload in (summary.get("method_totals") or {}).items():
        lines.append(f"| {name} | {_md(payload.get('safe_success'))} | {_md(payload.get('success'))} | {_md(payload.get('reward'))} | {_md(payload.get('mean_edit_l1'))} |")
    lines.extend(
        [
            "",
            "## Non-Leakage Notes",
            "",
            "- Predicate source is action chunk timing plus HDF5/visible EEF state where available.",
            "- The diagnostic does not use reward labels, success labels, task ids, BDDL target fields, or dataset target labels to select repair actions.",
            "- Exact-init replay is a bounded local control diagnostic, not a benchmark or paper-grade claim.",
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
        "evidence_label": "tl_chunkrepair_state1",
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
        stop_reasons.append(f"{TASK_GATE}=1 is required for bounded TL-ChunkRepair replay")
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
    if args.unsafe_span < 1 or args.unsafe_span > 20:
        stop_reasons.append("unsafe_span must be between 1 and 20")
    if stop_reasons:
        report["result"]["blocked_reason"] = "; ".join(stop_reasons)
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_tl_chunkrepair_blocker"}
        report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
        return report
    try:
        _run_replay_case(args, report)
        all_variants = [report["cases"][0]["exact_expert_replay"]]
        for perturbation in report["cases"][0].get("perturbations", []):
            all_variants.extend(perturbation.get("variants", []))
        report["result"]["passed"] = all(item.get("passed") for item in all_variants)
        report["summary"] = summarize_report(report)
    except Exception as exc:  # noqa: BLE001
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "reason": report["result"]["blocked_reason"], "next_state": "resolve_tl_chunkrepair_blocker"}
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
    parser.add_argument("--unsafe-span", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="reports/tl_chunkrepair_state1_result.json")
    parser.add_argument("--report-md", default="reports/tl_chunkrepair_state1_result.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    console = {
        "result": report.get("result"),
        "summary": {key: value for key, value in (report.get("summary") or {}).items() if key != "perturbation_summaries" and key != "method_totals"},
        "report_json": str(report_json),
        "replay_or_rollout_performed": report.get("policy", {}).get("replay_or_rollout_performed"),
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] or os.environ.get(TASK_GATE) != "1" else 1


if __name__ == "__main__":
    sys.exit(main())
