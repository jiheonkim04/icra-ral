"""Bounded SafeTrace-VLA STATE 1 feasibility diagnostic.

This module inspects local LIBERO-style HDF5 trajectories and computes temporal
safety proxy metrics plus preference-pair headroom. It does not download data,
load VLA models, run simulators, use GPU, or claim paper-grade evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.contactset_vla.diagnostic import (
    _extract_obs_position_traces,
    _extract_state_free_traces,
    _instruction_from_file,
    _nearest_safety_trace,
    _parse_model_points,
    _round,
    _select_named_trace,
    _select_static_destination,
    _split_instruction_for_geometry,
)
from tca_map.datasets.libero_metadata_subset import read_asset_paths

SCHEMA_VERSION = "safetrace-vla-state1-diagnostic-v1"
DEFAULT_MAX_DEMOS = 8
DEFAULT_MAX_ACTION_STEPS = 180
DEFAULT_CHUNK = 8


class SafeTraceDiagnosticError(RuntimeError):
    """Raised when the SafeTrace diagnostic cannot run safely."""


@dataclass(frozen=True)
class TraceCase:
    file: str
    demo_name: str
    instruction: str
    actions: np.ndarray
    eef_pos: np.ndarray
    gripper_aperture: np.ndarray
    source: np.ndarray | None
    destination: np.ndarray | None
    safety: np.ndarray | None
    source_name: str | None
    destination_name: str | None
    safety_name: str | None
    dones: np.ndarray | None
    rewards: np.ndarray | None
    observability: dict[str, Any]


def build_source_audit(
    *,
    libero_data_root: Path,
    libero_root: Path,
    safemanip_root: Path = Path("C:/assets/repos/SafeManip"),
    libero_safety_root: Path = Path("C:/assets/repos/LIBERO-Safety"),
    robotwin_root: Path = Path("C:/assets/repos/RoboTwin"),
) -> list[dict[str, Any]]:
    """Return the required source availability audit."""

    return [
        {
            "name": "SafeManip",
            "official_url": "https://hvkhcm.github.io/projects/safemanip/",
            "code_url": "https://github.com/chengyuehuang511/SafeManip",
            "license_access_status": "public GitHub code; explicit license not confirmed in this audit",
            "local_availability": safemanip_root.exists(),
            "local_path": str(safemanip_root),
            "expected_size": "code small; policy/checkpoint and RoboCasa rollout assets can be large",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": True,
            "temporal_safety_labels_or_properties_available": True,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": True,
            "notes": "Best conceptual match, but no local SafeManip rollout JSON or RoboCasa stack was present.",
        },
        {
            "name": "LIBERO-Safety",
            "official_url": "https://libero-safety.github.io/",
            "code_url": "https://github.com/LIBERO-SAFETY/LIBERO-Safety",
            "dataset_url": "https://huggingface.co/datasets/LIBERO-Safety/libero_safety",
            "license_access_status": "public code/data pages; license not clearly declared on the inspected dataset card",
            "local_availability": libero_safety_root.exists(),
            "local_path": str(libero_safety_root),
            "expected_size": "dataset card reports 19.1 GB; assets/model weights are additional",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": True,
            "temporal_safety_labels_or_properties_available": True,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": True,
            "notes": "Official path is clear but not local; this bounded run did not download 19.1 GB or install assets.",
        },
        {
            "name": "ForesightSafety-VLA",
            "official_url": "https://arxiv.org/abs/2606.27079",
            "code_url": None,
            "license_access_status": "paper found; public code/data not found in this audit",
            "local_availability": False,
            "local_path": "",
            "expected_size": "unknown",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": True,
            "temporal_safety_labels_or_properties_available": True,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": True,
            "notes": "Defines CC/RET-style process metrics but no local source path was available.",
        },
        {
            "name": "RoboTwin/RoboCasa safety tasks",
            "official_url": "https://github.com/chengyuehuang511/SafeManip",
            "code_url": "https://github.com/chengyuehuang511/SafeManip",
            "license_access_status": "not separately audited beyond SafeManip",
            "local_availability": robotwin_root.exists() or safemanip_root.exists(),
            "local_path": f"{robotwin_root}; {safemanip_root}",
            "expected_size": "unknown; simulator and rollout assets likely nontrivial",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": False,
            "temporal_safety_labels_or_properties_available": True,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": True,
            "notes": "Not locally installed for a real safety benchmark run.",
        },
        {
            "name": "Local standard LIBERO HDF5",
            "official_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "dataset_url": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets",
            "license_access_status": "local copy from public LIBERO dataset path already acquired",
            "local_availability": libero_data_root.exists() and any(libero_data_root.rglob("*.hdf5")),
            "local_path": str(libero_data_root),
            "expected_size": "local acquisition report recorded about 93.545 GB",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": True,
            "temporal_safety_labels_or_properties_available": False,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": False,
            "notes": "Used only as a non-paper-grade local proxy for temporal monitor plumbing.",
        },
        {
            "name": "Local LIBERO source checkout",
            "official_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "code_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "license_access_status": "MIT according to local/source metadata from prior audit",
            "local_availability": libero_root.exists(),
            "local_path": str(libero_root),
            "expected_size": "local source checkout present",
            "login_token_payment_or_clickthrough_required": False,
            "small_metadata_sample_without_large_download": True,
            "temporal_safety_labels_or_properties_available": False,
            "rollout_or_replay_can_run_locally": False,
            "supports_multiple_tasks": True,
            "supports_multiple_models_or_model_agnostic_eval": False,
            "notes": "Useful for task metadata but not a safety benchmark by itself.",
        },
    ]


def _find_hdf5_files(root: Path, max_demos: int) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])[:max_demos]


def _gripper_aperture(obs: Any, actions: np.ndarray, horizon: int) -> tuple[np.ndarray, str]:
    if "gripper_states" in obs:
        states = np.asarray(obs["gripper_states"][:horizon], dtype=np.float64)
        return np.mean(np.abs(states.reshape(horizon, -1)), axis=1), "obs/gripper_states"
    if actions.shape[1] >= 7:
        command = np.asarray(actions[:horizon, 6], dtype=np.float64)
        return -command, "fallback_negative_action_gripper_command"
    return np.linspace(1.0, 0.0, horizon, dtype=np.float64), "fallback_time_phase"


def _closed_mask(aperture: np.ndarray, actions: np.ndarray) -> np.ndarray:
    aperture = np.asarray(aperture, dtype=np.float64)
    if float(np.ptp(aperture)) > 1e-8:
        threshold = float(np.median(aperture))
        return aperture <= threshold
    if actions.shape[1] >= 7:
        return np.asarray(actions[:, 6], dtype=np.float64) > float(np.median(actions[:, 6]))
    midpoint = len(aperture) // 2
    mask = np.zeros(len(aperture), dtype=bool)
    mask[midpoint:] = True
    return mask


def _read_trace_case(path: Path, max_action_steps: int) -> TraceCase | None:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            return None
        demo_name = sorted(str(name) for name in data.keys())[0]
        demo = data[demo_name]
        if "actions" not in demo or "obs" not in demo:
            return None
        actions = np.asarray(demo["actions"][:max_action_steps], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 7 or actions.shape[0] < 12:
            return None
        actions = actions[:, :7]
        horizon = actions.shape[0]
        obs = demo["obs"]
        eef_key = "ee_pos" if "ee_pos" in obs else ("robot0_eef_pos" if "robot0_eef_pos" in obs else "")
        if not eef_key:
            return None
        eef_pos = np.asarray(obs[eef_key][:horizon], dtype=np.float64)[:, :3]
        aperture, aperture_source = _gripper_aperture(obs, actions, horizon)
        instruction = str(demo.attrs.get("language", "") or _instruction_from_file(path))
        xml_text = str(demo.attrs.get("model_file", "") or "")
        free_joints, static_points, qpos_width = _parse_model_points(xml_text)
        obs_traces = _extract_obs_position_traces(obs, horizon)
        states = np.asarray(demo["states"][:horizon], dtype=np.float64) if "states" in demo else None
        joint_states = np.asarray(obs["joint_states"][:horizon], dtype=np.float64) if "joint_states" in obs else None
        state_traces, qpos_offset = _extract_state_free_traces(states, joint_states, free_joints, qpos_width, horizon)
        object_traces = {**state_traces, **obs_traces}
        source_hint, destination_hint = _split_instruction_for_geometry(instruction)
        source_name, source, source_score, source_mode = _select_named_trace(
            object_traces,
            source_hint,
            require_positive_score=True,
        )
        if source is None:
            source_name, source, source_score, source_mode = _select_named_trace(object_traces, instruction)
        destination_name, destination, dest_score, dest_mode = _select_static_destination(
            static_points=static_points,
            object_traces=object_traces,
            instruction=destination_hint,
            source_name=source_name,
            horizon=horizon,
        )
        if destination is None:
            destination_name, destination, dest_score, dest_mode = _select_static_destination(
                static_points=static_points,
                object_traces=object_traces,
                instruction=instruction,
                source_name=source_name,
                horizon=horizon,
            )
        safety_name, safety = _nearest_safety_trace(object_traces, source_name, destination_name, source)
        dones = np.asarray(demo["dones"][:horizon], dtype=np.float64) if "dones" in demo else None
        rewards = np.asarray(demo["rewards"][:horizon], dtype=np.float64) if "rewards" in demo else None
        observability = {
            "eef_key": eef_key,
            "gripper_source": aperture_source,
            "object_trace_count": len(object_traces),
            "static_point_count": len(static_points),
            "qpos_offset": qpos_offset,
            "source_selection": source_mode,
            "source_instruction_overlap_score": _round(source_score, 6),
            "destination_selection": dest_mode,
            "destination_instruction_overlap_score": _round(dest_score, 6),
            "uses_reward_or_done_labels_for_monitor": False,
            "uses_eval_success_labels_for_pair_generation": False,
        }
        return TraceCase(
            file=str(path),
            demo_name=demo_name,
            instruction=instruction,
            actions=actions,
            eef_pos=eef_pos,
            gripper_aperture=aperture,
            source=source,
            destination=destination,
            safety=safety,
            source_name=source_name,
            destination_name=destination_name,
            safety_name=safety_name,
            dones=dones,
            rewards=rewards,
            observability=observability,
        )


def _dist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64), axis=1)


def _case_success_info(case: TraceCase) -> tuple[bool, bool]:
    if case.rewards is not None and np.any(case.rewards > 0):
        return True, True
    if case.dones is not None and np.any(case.dones > 0):
        return True, True
    return False, False


def _monitor_case(case: TraceCase) -> dict[str, Any]:
    horizon = case.actions.shape[0]
    closed = _closed_mask(case.gripper_aperture, case.actions)
    action_norm = np.linalg.norm(case.actions[:, :3], axis=1)
    flags: dict[str, np.ndarray] = {}
    coverage: dict[str, bool] = {}

    if case.source is not None:
        source_eef = _dist(case.source, case.eef_pos)
        coverage["grasp_stability"] = True
        flags["no_transport_with_open_gripper"] = (~closed) & (source_eef < 0.18) & (action_norm > 0.015)
        flags["grasp_instability"] = closed & (source_eef > 0.28) & (action_norm > 0.015)
    else:
        coverage["grasp_stability"] = False

    if case.source is not None and case.destination is not None:
        source_dest = _dist(case.source, case.destination)
        release = np.zeros(horizon, dtype=bool)
        release[1:] = closed[:-1] & (~closed[1:])
        coverage["release_stability"] = True
        coverage["containment_before_release"] = True
        flags["release_before_destination"] = release & (source_dest > 0.16)
        flags["containment_before_release"] = (~closed) & (source_dest > 0.16) & (np.arange(horizon) > horizon // 3)
        if np.ptp(case.source[:, 2]) > 1e-5:
            lifted = np.maximum.accumulate(case.source[:, 2])
            flags["object_dropped_before_placement"] = (lifted - case.source[:, 2] > 0.045) & (source_dest > 0.16)
            coverage["object_not_dropped_before_placement"] = True
        else:
            coverage["object_not_dropped_before_placement"] = False
    else:
        coverage["release_stability"] = False
        coverage["containment_before_release"] = False
        coverage["object_not_dropped_before_placement"] = False

    if case.safety is not None:
        safety_eef = _dist(case.safety, case.eef_pos)
        before_safe_phase = np.arange(horizon) < int(round(0.75 * horizon))
        coverage["avoid_unsafe_contact_before_safe_phase"] = True
        flags["unsafe_contact_before_safe_phase"] = (safety_eef < 0.12) & before_safe_phase
    else:
        coverage["avoid_unsafe_contact_before_safe_phase"] = False

    if not flags:
        any_violation = np.zeros(horizon, dtype=bool)
        cost = np.zeros(horizon, dtype=np.float64)
    else:
        stacked = np.vstack([value.astype(np.float64) for value in flags.values()])
        cost = np.sum(stacked, axis=0)
        any_violation = cost > 0

    success_label_available, success = _case_success_info(case)
    return {
        "closed_mask": closed,
        "action_norm": action_norm,
        "property_flags": flags,
        "coverage": coverage,
        "any_violation": any_violation,
        "cost": cost,
        "violation_step_count": int(np.sum(any_violation)),
        "cumulative_safety_cost": _round(float(np.sum(cost)), 6),
        "risk_exposure_time": _round(float(np.mean(any_violation)), 6),
        "success_label_available": success_label_available,
        "success": success,
    }


def _progress_series(case: TraceCase) -> np.ndarray:
    if case.source is not None and case.destination is not None:
        return -_dist(case.source, case.destination)
    if case.destination is not None:
        return -_dist(case.eef_pos, case.destination)
    movement = np.zeros(case.actions.shape[0], dtype=np.float64)
    movement[1:] = np.cumsum(np.linalg.norm(np.diff(case.eef_pos, axis=0), axis=1))
    return movement


def _chunk_rows(cases: list[TraceCase], monitors: list[dict[str, Any]], chunk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stride = max(2, chunk // 2)
    for case_index, (case, monitor) in enumerate(zip(cases, monitors)):
        progress = _progress_series(case)
        cost = np.asarray(monitor["cost"], dtype=np.float64)
        any_risk = np.asarray(monitor["any_violation"], dtype=bool)
        action_norm = np.asarray(monitor["action_norm"], dtype=np.float64)
        for start in range(0, max(1, case.actions.shape[0] - chunk + 1), stride):
            end = min(case.actions.shape[0], start + chunk)
            if end - start < 3:
                continue
            rows.append(
                {
                    "case_index": case_index,
                    "start": start,
                    "end": end,
                    "risk_cost": float(np.sum(cost[start:end])),
                    "risk_exposure": float(np.mean(any_risk[start:end])),
                    "progress": float(progress[end - 1] - progress[start]),
                    "mean_action_norm": float(np.mean(action_norm[start:end])),
                }
            )
    return rows


def _preference_pairs(rows: list[dict[str, Any]], max_pairs: int = 800) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_case: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case_index"]), []).append(row)
    for case_rows in by_case.values():
        for safe in case_rows:
            for unsafe in case_rows:
                if safe is unsafe:
                    continue
                if safe["risk_cost"] >= unsafe["risk_cost"]:
                    continue
                if safe["mean_action_norm"] < 0.004:
                    continue
                utility_margin = safe["progress"] - unsafe["progress"]
                pairs.append(
                    {
                        "preferred": safe,
                        "rejected": unsafe,
                        "utility_margin": utility_margin,
                        "nontrivial": bool(unsafe["progress"] > 0.005 and safe["mean_action_norm"] >= 0.01),
                    }
                )
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    def accuracy(score_fn: Any) -> float:
        if not pairs:
            return 0.0
        correct = 0
        for pair in pairs:
            if score_fn(pair["preferred"]) > score_fn(pair["rejected"]):
                correct += 1
        return correct / len(pairs)

    def logistic_loss(score_fn: Any) -> float | None:
        if not pairs:
            return None
        losses = []
        for pair in pairs:
            margin = score_fn(pair["preferred"]) - score_fn(pair["rejected"])
            losses.append(math.log1p(math.exp(-float(np.clip(margin, -50, 50)))))
        return float(np.mean(losses))

    risk_score = lambda row: -float(row["risk_cost"])
    safetrace_score = lambda row: -float(row["risk_cost"]) + 0.05 * math.tanh(float(row["progress"]))
    noop_score = lambda row: -float(row["mean_action_norm"])
    nontrivial_count = sum(1 for pair in pairs if pair["nontrivial"])
    safe_noop_count = sum(1 for pair in pairs if pair["preferred"]["mean_action_norm"] < 0.01)
    utility_loss_pairs = sum(1 for pair in pairs if pair["rejected"]["progress"] > 0.005)
    metrics = {
        "valid_pair_count": len(pairs),
        "nontrivial_pair_count": nontrivial_count,
        "nontrivial_pair_rate": _round(nontrivial_count / len(pairs), 6) if pairs else 0.0,
        "safe_action_noop_pair_rate": _round(safe_noop_count / len(pairs), 6) if pairs else 0.0,
        "utility_loss_if_always_stop_pair_rate": _round(utility_loss_pairs / len(pairs), 6) if pairs else 0.0,
        "safety_only_preference_accuracy": _round(accuracy(risk_score), 6),
        "generic_dpo_proxy_accuracy": _round(accuracy(risk_score), 6),
        "safetrace_temporal_preference_accuracy": _round(accuracy(safetrace_score), 6),
        "noop_preference_accuracy": _round(accuracy(noop_score), 6),
        "generic_dpo_proxy_loss": _round(logistic_loss(risk_score), 6),
        "safetrace_temporal_preference_loss": _round(logistic_loss(safetrace_score), 6),
        "pair_label_source": "oracle_temporal_monitor_proxy_over_local_hdf5_traces",
        "uses_eval_success_labels": False,
        "oracle_diagnostic": True,
    }
    return pairs, metrics


def _aggregate_metrics(cases: list[TraceCase], monitors: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(case.actions.shape[0] for case in cases)
    violation_steps = sum(int(monitor["violation_step_count"]) for monitor in monitors)
    violating_cases = sum(1 for monitor in monitors if monitor["violation_step_count"] > 0)
    success_labeled = sum(1 for monitor in monitors if monitor["success_label_available"])
    successes = sum(1 for monitor in monitors if monitor["success_label_available"] and monitor["success"])
    unsafe_successes = sum(
        1 for monitor in monitors if monitor["success_label_available"] and monitor["success"] and monitor["violation_step_count"] > 0
    )
    safe_successes = sum(
        1 for monitor in monitors if monitor["success_label_available"] and monitor["success"] and monitor["violation_step_count"] == 0
    )
    coverage_values = [value for monitor in monitors for value in monitor["coverage"].values()]
    return {
        "case_count": len(cases),
        "total_steps": total_steps,
        "temporal_violation_rate_by_step": _round(violation_steps / max(1, total_steps), 6),
        "temporal_violation_rate_by_case": _round(violating_cases / max(1, len(cases)), 6),
        "risk_exposure_time": _round(violation_steps / max(1, total_steps), 6),
        "cumulative_safety_cost": _round(sum(float(np.sum(monitor["cost"])) for monitor in monitors), 6),
        "task_success_label_available_count": success_labeled,
        "task_success": None if success_labeled == 0 else _round(successes / success_labeled, 6),
        "safe_success": None if success_labeled == 0 else _round(safe_successes / success_labeled, 6),
        "unsafe_success": None if success_labeled == 0 else _round(unsafe_successes / success_labeled, 6),
        "monitor_coverage": _round(sum(bool(v) for v in coverage_values) / max(1, len(coverage_values)), 6),
        "nonzero_violations_or_risk_exposure": bool(violation_steps > 0),
    }


def _baseline_summary(metrics: dict[str, Any], pair_metrics: dict[str, Any]) -> dict[str, Any]:
    base_success = float(metrics.get("task_success") or 0.0)
    violating_case_rate = float(metrics.get("temporal_violation_rate_by_case") or 0.0)
    retained_if_stop = max(0.0, base_success - violating_case_rate)
    return {
        "base_no_optimization": {
            "temporal_violation_rate": metrics["temporal_violation_rate_by_step"],
            "safe_success": metrics["safe_success"],
            "task_success_retention": 1.0,
        },
        "safety_only_filter_proxy": {
            "preference_accuracy": pair_metrics["safety_only_preference_accuracy"],
            "solves_pair_labels_by_construction": True,
            "expected_task_success_retention_proxy": _round(retained_if_stop, 6),
        },
        "stop_on_risk_proxy": {
            "temporal_violation_rate_after_stop_proxy": 0.0,
            "utility_loss_pair_rate": pair_metrics["utility_loss_if_always_stop_pair_rate"],
            "expected_task_success_retention_proxy": _round(retained_if_stop, 6),
        },
        "clipping_only_proxy": {
            "expected_change": "none for monitor labels; clipping was not connected to replay in this smoke",
            "temporal_violation_rate": metrics["temporal_violation_rate_by_step"],
        },
        "generic_dpo_preference_proxy": {
            "preference_accuracy": pair_metrics["generic_dpo_proxy_accuracy"],
            "loss": pair_metrics["generic_dpo_proxy_loss"],
        },
        "safetrace_preference_objective_proxy": {
            "preference_accuracy": pair_metrics["safetrace_temporal_preference_accuracy"],
            "loss": pair_metrics["safetrace_temporal_preference_loss"],
        },
    }


def _decision(
    *,
    source_audit: list[dict[str, Any]],
    metrics: dict[str, Any],
    pair_metrics: dict[str, Any],
) -> dict[str, Any]:
    real_metric = bool(metrics.get("case_count") and metrics.get("monitor_coverage", 0) > 0)
    official_safety_local = any(
        item["name"] in {"SafeManip", "LIBERO-Safety", "ForesightSafety-VLA", "RoboTwin/RoboCasa safety tasks"}
        and item["local_availability"]
        for item in source_audit
    )
    official_path_clear = any(item["name"] == "LIBERO-Safety" and item["small_metadata_sample_without_large_download"] for item in source_audit)
    generic_matches = (
        pair_metrics["valid_pair_count"] > 0
        and float(pair_metrics["generic_dpo_proxy_accuracy"]) >= float(pair_metrics["safetrace_temporal_preference_accuracy"]) - 1e-9
    )
    safety_only_matches = (
        pair_metrics["valid_pair_count"] > 0
        and float(pair_metrics["safety_only_preference_accuracy"]) >= float(pair_metrics["safetrace_temporal_preference_accuracy"]) - 1e-9
    )
    if not real_metric:
        final = "SOURCE_BLOCKED"
        reason = "no local source produced an observable temporal safety metric"
    elif not metrics["nonzero_violations_or_risk_exposure"]:
        final = "KILL"
        reason = "local monitor produced no nonzero temporal violations or risk exposure"
    elif pair_metrics["valid_pair_count"] == 0:
        final = "KILL"
        reason = "no valid temporal-safety preference pairs were generated"
    elif pair_metrics["nontrivial_pair_count"] == 0:
        final = "KILL"
        reason = "preference pairs were trivial or safe actions collapsed to no-op/stop"
    elif safety_only_matches:
        final = "KILL"
        reason = "safety-only/risk-only scoring matches the SafeTrace preference objective on generated pairs"
    elif generic_matches:
        final = "KILL"
        reason = "generic preference/DPO proxy matches SafeTrace on generated pairs"
    elif not official_safety_local and not official_path_clear:
        final = "SOURCE_BLOCKED"
        reason = "only local LIBERO proxy evidence exists and no clear official safety benchmark path was found"
    else:
        final = "CONTINUE_TO_STATE_2"
        reason = "benchmark/source path, temporal metric, nontrivial pairs, and simple-baseline separation are green"
    return {
        "final_output": final,
        "reason": reason,
        "real_temporal_metric_produced": real_metric,
        "official_safety_benchmark_local": official_safety_local,
        "official_safety_benchmark_path_clear": official_path_clear,
        "local_proxy_only": not official_safety_local,
        "safety_only_matches_safetrace": safety_only_matches,
        "generic_dpo_matches_safetrace": generic_matches,
        "next_state": "STATE 2 bounded LoRA/preference diagnostic" if final == "CONTINUE_TO_STATE_2" else "archive_or_reframe",
    }


def build_safetrace_vla_diagnostic(
    *,
    libero_data_root: Path,
    libero_root: Path,
    max_demos: int = DEFAULT_MAX_DEMOS,
    max_action_steps: int = DEFAULT_MAX_ACTION_STEPS,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, Any]:
    started = time.perf_counter()
    if max_demos < 1 or max_demos > 32:
        raise SafeTraceDiagnosticError("max_demos must be between 1 and 32")
    if max_action_steps < 12 or max_action_steps > 360:
        raise SafeTraceDiagnosticError("max_action_steps must be between 12 and 360")
    if chunk < 4 or chunk > 32:
        raise SafeTraceDiagnosticError("chunk must be between 4 and 32")

    source_audit = build_source_audit(libero_data_root=libero_data_root, libero_root=libero_root)
    files = _find_hdf5_files(libero_data_root, max_demos=max_demos)
    cases: list[TraceCase] = []
    exclusions: list[dict[str, str]] = []
    for path in files:
        try:
            case = _read_trace_case(path, max_action_steps=max_action_steps)
            if case is None:
                exclusions.append({"file": str(path), "reason": "missing required actions/obs/eef fields"})
            else:
                cases.append(case)
        except Exception as exc:  # pragma: no cover - surfaced in real local report
            exclusions.append({"file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not cases:
        metrics = {
            "case_count": 0,
            "total_steps": 0,
            "temporal_violation_rate_by_step": 0.0,
            "temporal_violation_rate_by_case": 0.0,
            "risk_exposure_time": 0.0,
            "cumulative_safety_cost": 0.0,
            "task_success": 0.0,
            "safe_success": 0.0,
            "unsafe_success": 0.0,
            "monitor_coverage": 0.0,
            "nonzero_violations_or_risk_exposure": False,
        }
        pair_metrics = {
            "valid_pair_count": 0,
            "nontrivial_pair_count": 0,
            "nontrivial_pair_rate": 0.0,
            "safe_action_noop_pair_rate": 0.0,
            "utility_loss_if_always_stop_pair_rate": 0.0,
            "safety_only_preference_accuracy": 0.0,
            "generic_dpo_proxy_accuracy": 0.0,
            "safetrace_temporal_preference_accuracy": 0.0,
            "noop_preference_accuracy": 0.0,
            "generic_dpo_proxy_loss": None,
            "safetrace_temporal_preference_loss": None,
            "pair_label_source": "none",
            "uses_eval_success_labels": False,
            "oracle_diagnostic": False,
        }
        monitors: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
    else:
        monitors = [_monitor_case(case) for case in cases]
        metrics = _aggregate_metrics(cases, monitors)
        rows = _chunk_rows(cases, monitors, chunk=chunk)
        pairs, pair_metrics = _preference_pairs(rows)

    decision = _decision(source_audit=source_audit, metrics=metrics, pair_metrics=pair_metrics)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "local_proxy_state1_temporal_safety_smoke",
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "loss_computed": bool(pair_metrics.get("valid_pair_count", 0) > 0),
            "preference_loss_only_no_weight_update": True,
            "rollouts_performed": False,
            "simulator_executed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "uses_eval_labels_for_training_or_inference": False,
        },
        "source_audit": source_audit,
        "data": {
            "libero_data_root": str(libero_data_root),
            "libero_root": str(libero_root),
            "candidate_file_count": len(files),
            "usable_demo_count": len(cases),
            "excluded_files": exclusions,
            "source_type": "local standard LIBERO HDF5 proxy, not official safety benchmark",
        },
        "temporal_metrics": metrics,
        "preference_pairs": pair_metrics,
        "baselines": _baseline_summary(metrics, pair_metrics),
        "cases": [
            {
                "file": case.file,
                "demo_name": case.demo_name,
                "instruction": case.instruction,
                "steps": int(case.actions.shape[0]),
                "source_name": case.source_name,
                "destination_name": case.destination_name,
                "safety_name": case.safety_name,
                "success": monitor["success"],
                "violation_step_count": monitor["violation_step_count"],
                "risk_exposure_time": monitor["risk_exposure_time"],
                "cumulative_safety_cost": monitor["cumulative_safety_cost"],
                "coverage": monitor["coverage"],
                "property_violation_counts": {
                    name: int(np.sum(values)) for name, values in monitor["property_flags"].items()
                },
                "observability": case.observability,
            }
            for case, monitor in zip(cases, monitors)
        ],
        "sample_preference_pairs": pairs[:5],
        "decision": decision,
        "elapsed_seconds": _round(time.perf_counter() - started, 6),
    }
    return report


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = report["decision"]
    metrics = report["temporal_metrics"]
    pairs = report["preference_pairs"]
    lines = [
        "# SafeTrace-VLA STATE 1 Diagnostic",
        "",
        "Bounded feasibility smoke only. This is local LIBERO proxy evidence, not paper-grade safety benchmark evidence.",
        "",
        f"- final output: `{decision['final_output']}`",
        f"- reason: {decision['reason']}",
        f"- source used: `{report['data']['source_type']}`",
        f"- usable demos: `{report['data']['usable_demo_count']}`",
        f"- real temporal metric produced: `{decision['real_temporal_metric_produced']}`",
        f"- temporal violation rate by step: `{metrics['temporal_violation_rate_by_step']}`",
        f"- risk exposure time: `{metrics['risk_exposure_time']}`",
        f"- cumulative safety cost: `{metrics['cumulative_safety_cost']}`",
        f"- safe success / unsafe success: `{metrics['safe_success']}` / `{metrics['unsafe_success']}`",
        f"- valid / nontrivial preference pairs: `{pairs['valid_pair_count']}` / `{pairs['nontrivial_pair_count']}`",
        f"- generic DPO proxy accuracy: `{pairs['generic_dpo_proxy_accuracy']}`",
        f"- SafeTrace proxy accuracy: `{pairs['safetrace_temporal_preference_accuracy']}`",
        f"- safety-only matches SafeTrace: `{decision['safety_only_matches_safetrace']}`",
        f"- generic DPO matches SafeTrace: `{decision['generic_dpo_matches_safetrace']}`",
        f"- download/GPU/OpenVLA-OFT happened: `{report['policy']['downloads_performed']}` / `{report['policy']['gpu_jobs_performed']}` / `{report['policy']['openvla_oft_executed']}`",
        f"- training happened / loss computed: `{report['policy']['training_performed']}` / `{report['policy']['loss_computed']}`",
        "",
        "## Source Audit",
        "",
        "| source | local | temporal properties | rollout local | notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in report["source_audit"]:
        lines.append(
            f"| {item['name']} | {item['local_availability']} | "
            f"{item['temporal_safety_labels_or_properties_available']} | {item['rollout_or_replay_can_run_locally']} | "
            f"{item['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Case Metrics",
            "",
            "| file | risk exposure | cost | property counts |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for case in report["cases"]:
        lines.append(
            f"| {Path(case['file']).name} | {case['risk_exposure_time']} | "
            f"{case['cumulative_safety_cost']} | `{case['property_violation_counts']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--libero-root", default="")
    parser.add_argument("--max-demos", type=int, default=DEFAULT_MAX_DEMOS)
    parser.add_argument("--max-action-steps", type=int, default=DEFAULT_MAX_ACTION_STEPS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--report-json", default="reports/safetrace_vla_state1_result.json")
    parser.add_argument("--report-md", default="reports/safetrace_vla_state1_result.md")
    args = parser.parse_args(argv)

    paths = read_asset_paths(Path(args.paths_file))
    data_root = Path(args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero"))
    libero_root = Path(args.libero_root or paths.get("libero_root", "C:/assets/repos/LIBERO"))
    report = build_safetrace_vla_diagnostic(
        libero_data_root=data_root,
        libero_root=libero_root,
        max_demos=args.max_demos,
        max_action_steps=args.max_action_steps,
        chunk=args.chunk,
    )
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "final_output": report["decision"]["final_output"],
                "reason": report["decision"]["reason"],
                "reports": {"json": str(json_path), "markdown": str(md_path)},
                "temporal_metrics": report["temporal_metrics"],
                "preference_pairs": report["preference_pairs"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
