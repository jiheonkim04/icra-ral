#!/usr/bin/env python3
"""Run the sole sealed Epoch 9E exact-pair joint certification serially."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.epoch9e_nondrag_controller import inward_approach_y, run_nondrag_probe
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9e_joint_certification_protocol.json"
SEAL_PATH = REPORTS / "epoch9e_joint_execution_seal.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
OUTPUT_ROOT = REPORTS / "epoch9e_joint_certification"
RESULT_PATH = OUTPUT_ROOT / "result.json"
TRACE_ROOT = OUTPUT_ROOT / "traces"
SHAM_TRACE_ROOT = OUTPUT_ROOT / "sham_traces"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def memory_sample() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            meminfo[key] = int(raw.strip().split()[0]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return {
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "wsl_mem_used_bytes": int(meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0)),
        "wsl_swap_used_bytes": int(meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)),
    }


def gpu_sample() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
        return {"status": "AVAILABLE", "query": output}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"status": "UNAVAILABLE"}


def update_resource_peaks(result: dict[str, Any]) -> None:
    sample = memory_sample()
    resource = result["resource_monitor"]
    resource["process_max_rss_bytes"] = max(resource["process_max_rss_bytes"], sample["process_max_rss_bytes"])
    resource["wsl_mem_used_peak_bytes"] = max(resource["wsl_mem_used_peak_bytes"], sample["wsl_mem_used_bytes"])
    resource["wsl_swap_used_peak_bytes"] = max(resource["wsl_swap_used_peak_bytes"], sample["wsl_swap_used_bytes"])
    resource["last_sample"] = sample


def validate_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    if not SEAL_PATH.exists():
        raise FileNotFoundError("missing Epoch 9E joint execution seal")
    seal = load(SEAL_PATH)
    protocol = load(PROTOCOL_PATH)
    bindings = {
        "protocol": sha256(PROTOCOL_PATH) == seal["joint_protocol_sha256"],
        "runner": sha256(Path(__file__)) == seal["runner_sha256"],
        "controller": sha256(ROOT / seal["controller_path"]) == seal["controller_sha256"],
        "original_runner": sha256(ROOT / seal["original_runner_path"]) == seal["original_runner_sha256"],
    }
    if not all(bindings.values()):
        raise RuntimeError(f"joint execution seal mismatch: {bindings}")
    if seal["joint_outcomes_accessed_before_seal"]:
        raise RuntimeError("joint seal records prior outcome access")
    return seal, protocol


def base_lookup(protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["base_identity_id"]): row for row in protocol["base_states"]}


def initial_localization(observation: dict[str, Any], calibration: dict[str, Any]) -> dict[str, dict[str, float]]:
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
    values: dict[str, dict[str, float]] = {}
    for slot in ("front", "back"):
        _, _, metric = campaign.localize_candidate(frame, slot, calibration)
        values[slot] = {
            "subpixel_dx": float(metric["subpixel_dx"]),
            "subpixel_dy": float(metric["subpixel_dy"]),
            "quality": float(metric["quality"]),
        }
    return values


def make_exact_env(
    env_class: Any,
    base: dict[str, Any],
    task_bddl: str,
    mass_factor: dict[str, float],
    calibration: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / task_bddl),
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(int(base["generator_seed"]))
    env.reset()
    state = np.asarray(base["base_state_vector_float64"], dtype=np.float64)
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    observation = campaign.forced_observation(env)
    before = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    if before != base["first_agentview_rgb_sha256"]:
        env.close()
        raise RuntimeError(f"frozen first RGB mismatch for base {base['base_identity_id']}")
    localization_before = initial_localization(observation, calibration)
    baseline_masses = {
        slot: float(env.sim.model.body_mass[int(env.sim.model.body_name2id(body))])
        for slot, body in campaign.BODY_BY_SLOT.items()
    }
    for slot, factor in mass_factor.items():
        if float(factor) != 1.0:
            apply_intervention(
                env.sim.model,
                {
                    "axis": "target_mass",
                    "body_name": campaign.BODY_BY_SLOT[slot],
                    "arrays": ["body_mass", "body_inertia"],
                    "factor": float(factor),
                },
            )
    env.sim.forward()
    observation = campaign.forced_observation(env)
    after = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    localization_after = initial_localization(observation, calibration)
    if after != before or localization_after != localization_before:
        env.close()
        raise RuntimeError(f"mass assignment changed the ordinary initial observation for base {base['base_identity_id']}")
    applied_masses = {
        slot: float(env.sim.model.body_mass[int(env.sim.model.body_name2id(body))])
        for slot, body in campaign.BODY_BY_SLOT.items()
    }
    audit = {
        "base_state_vector_sha256": base["base_state_vector_sha256"],
        "expected_first_rgb_sha256": base["first_agentview_rgb_sha256"],
        "first_rgb_before_mass_sha256": before,
        "first_rgb_after_mass_sha256": after,
        "first_rgb_exact": before == after == base["first_agentview_rgb_sha256"],
        "initial_rgb_localization_audit": localization_after,
        "initial_rgb_localization_matches_frozen_base": localization_after == base["initial_rgb_localization_audit"],
        "baseline_body_mass_eval_construction_only": baseline_masses,
        "applied_body_mass_eval_construction_only": applied_masses,
        "mass_factor_eval_construction_only": mass_factor,
        "task_bddl": task_bddl,
    }
    return env, observation, audit


def response_from_trace(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        estimated = np.asarray(trace["estimated_world_delta_xy_m"], dtype=np.float64)
        quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
        target_contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
    indices = np.flatnonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response", "sham_fixed_micro_impulse", "sham_post_impulse_response"]))
    if indices.size != 5 or not np.all(np.diff(indices) == 1):
        raise RuntimeError(f"trace does not contain the frozen five-step response window: {path}")
    start = int(indices[0])
    baseline = np.median(estimated[max(0, start - 3):start], axis=0)
    response_axis = estimated[indices, 0] - baseline[0]
    front, back = positions[:, 0, :], positions[:, 1, :]
    low = np.asarray([-0.25, -0.05, 0.85])
    high = np.asarray([0.25, 0.25, 1.10])
    return {
        "response_window_indices": indices.tolist(),
        "observation_response_expected_axis_peak_m": float(max(np.max(response_axis), 0.0)),
        "observation_response_expected_axis_by_step_m": response_axis.tolist(),
        "response_tracker_quality_min": float(np.min(quality[indices])),
        "sampled_target_contact": bool(np.any(target_contact)),
        "fall": bool(np.any(front[:, 2] < 0.85) or np.any(back[:, 2] < 0.85)),
        "workspace_exit": bool(np.any(front < low) or np.any(front > high) or np.any(back < low) or np.any(back > high)),
        "identity_ordering_reversed": bool(np.any(front[:, 0] <= back[:, 0])),
        "candidate_positions_finite": bool(np.isfinite(positions).all()),
    }


def cross_lane_identity_swap(protocol: dict[str, Any], trace_path: Path) -> bool:
    with np.load(trace_path, allow_pickle=False) as trace:
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
    front, back = positions[:, 0, :2], positions[:, 1, :2]

    def inside(values: np.ndarray, lane: dict[str, list[float]]) -> np.ndarray:
        return ((values[:, 0] >= lane["x"][0]) & (values[:, 0] <= lane["x"][1]) &
                (values[:, 1] >= lane["y"][0]) & (values[:, 1] <= lane["y"][1]))

    return bool(np.any(inside(front, protocol["safe_center_lanes_m"]["back"])) or
                np.any(inside(back, protocol["safe_center_lanes_m"]["front"])))


def probe_audit(probe: dict[str, Any], original_protocol: dict[str, Any]) -> dict[str, Any]:
    trace_path = ROOT / probe["trace_path"]
    trace = response_from_trace(trace_path)
    nondrag = probe.get("epoch9e_nondrag_disengagement", {})
    attempts = nondrag.get("attempts", [])
    return {
        **trace,
        "trace_path": relative(trace_path),
        "trace_sha256": sha256(trace_path),
        "identity_swap": bool(trace["identity_ordering_reversed"] or cross_lane_identity_swap(original_protocol, trace_path)),
        "unrecoverable_track_loss": bool(float(probe["online_localization"]["quality"]) < 0.50 or trace["response_tracker_quality_min"] < 0.50 or not np.isfinite(trace["observation_response_expected_axis_peak_m"])),
        "unintended_collision": bool(probe["candidate_pair_collision_eval_only"] or probe["candidate_distractor_collision_eval_only"]),
        "finite_bounded_actions": bool(probe["finite_bounded_actions"]),
        "intended_contact_or_excitation": bool(probe["intended_target_contact_or_excitation_eval_only"]),
        "lane_and_reachability_continuous_pass": bool(probe["lane_and_reachability_continuous_pass"]),
        "forbidden_online_inputs_used": probe["forbidden_online_inputs_used"],
        "simulator_state_used_for_actions": bool(probe["simulator_state_used_for_actions"]),
        "mass_or_property_used_for_actions": bool(probe["mass_or_property_used_for_actions"]),
        "nondrag_attempt_count": len(attempts),
        "nondrag_liftoff_planar_commands_exact_zero": bool(nondrag.get("all_liftoff_planar_commands_exact_zero")),
        "nondrag_all_separations_verified": bool(attempts and all(row["separation_verified_ordinary_observations"] for row in attempts)),
        "nondrag_forbidden_inputs_used": nondrag.get("forbidden_inputs_used", ["missing audit"]),
    }


def sham_probe(
    env: Any,
    observation: dict[str, Any],
    slot: str,
    sham_id: str,
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
    estimated_target, center_xy, localization = campaign.localize_candidate(initial_frame, slot, calibration)
    recorder = campaign.EpisodeRecorder(env, initial_frame, slot, center_xy, calibration)
    for _ in range(config.close_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_preclose_paddle")
    approach_y, orientation = inward_approach_y(slot, float(estimated_target[1]), config.approach_start_offset_x_m, config, original_protocol)
    high = np.asarray([estimated_target[0] + config.approach_start_offset_x_m, approach_y, 1.02], dtype=np.float64)
    observation, high_reached = campaign.move_to(observation, high, recorder, config, "sham_transit_high", gripper=config.gripper_closed_command)
    for _ in range(config.impulse_steps):
        action = np.zeros(7, dtype=np.float32)
        action[0] = config.impulse_action_x
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_fixed_micro_impulse")
    for _ in range(config.coast_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_post_impulse_response")
    trace_path = SHAM_TRACE_ROOT / f"{sham_id}_{slot}.npz"
    campaign.persist_trace(trace_path, recorder)
    trace = response_from_trace(trace_path)
    actions = np.asarray(recorder.action, dtype=np.float32)
    return observation, {
        "slot": slot,
        "online_localization": localization,
        "orientation_from_rgb_lane_geometry": orientation,
        "finite_bounded_actions": bool(actions.size and all(campaign.legal_action(value) for value in actions)),
        "intended_target_contact_eval_only": bool(any(recorder.target_contact_eval)),
        "intended_target_contact_or_excitation_eval_only": bool(any(recorder.target_contact_eval)),
        "candidate_pair_collision_eval_only": bool(any(recorder.candidate_pair_collision_eval)),
        "candidate_distractor_collision_eval_only": bool(any(recorder.distractor_collision_eval)),
        "response_estimated_displacement_m": trace["observation_response_expected_axis_peak_m"],
        "response_tracker_quality_min": trace["response_tracker_quality_min"],
        "high_clearance_reached": bool(high_reached),
        "trace_path": relative(trace_path),
        "forbidden_online_inputs_used": [],
        "simulator_state_used_for_actions": False,
        "mass_or_property_used_for_actions": False,
    }


def run_primary(
    env_class: Any,
    assignment: dict[str, Any],
    base: dict[str, Any],
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
    controller_contract: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "row_key": f"primary:{assignment['scene_id']}",
        "row_type": "PRIMARY_ASSIGNMENT",
        "scene_id": assignment["scene_id"],
        "base_identity_id": assignment["base_identity_id"],
        "assignment": assignment["assignment"],
        "heavy_slot_eval_only": assignment["heavy_slot_eval_only"],
        "completion_target_slot_eval_only": assignment["completion_target_slot_eval_only"],
        "instruction_property": assignment["instruction_property"],
        "probe_order": assignment["probe_order"],
        "exception": None,
    }
    try:
        env, observation, exact = make_exact_env(env_class, base, assignment["task_bddl"], assignment["mass_factor"], calibration)
        probes = []
        for slot in assignment["probe_order"]:
            observation, probe = run_nondrag_probe(
                env, observation, slot, assignment["scene_id"], config, calibration,
                original_protocol, controller_contract, TRACE_ROOT,
            )
            probes.append(probe)
        audits = {probe["slot"]: probe_audit(probe, original_protocol) for probe in probes}
        responses = {probe["slot"]: float(probe["response_estimated_displacement_m"]) for probe in probes}
        threshold = float(config.back_heavy_threshold_m)
        scores = {"front": threshold - responses["back"], "back": responses["back"] - threshold}
        predicted = min(scores, key=scores.get)
        observation, oracle = campaign.oracle_completion(
            env, observation, assignment["completion_target_slot_eval_only"], calibration, config
        )
        row.update({
            "completed": True,
            "exact_state_audit": exact,
            "responses_m": responses,
            "candidate_scores_m": scores,
            "predicted_heavy_slot": predicted,
            "heavy_rank_correct_eval_only": predicted == assignment["heavy_slot_eval_only"],
            "probes": probes,
            "probe_audits": audits,
            "both_candidates_excited_eval_only": all(audit["intended_contact_or_excitation"] for audit in audits.values()),
            "oracle_completion": oracle,
            "method_information_boundary": {
                "mass_assignment_used_only_to_construct_intervention": True,
                "probe_and_score_inputs": ["ordinary RGB", "proprioception", "frozen geometry", "action trace"],
                "mass_property_simulator_pose_force_reward_success_segmentation_oracle_passed_to_probe_or_score": False,
                "oracle_privilege_evaluation_only_after_both_probes": True,
            },
        })
    except Exception as exc:
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = float(time.monotonic() - started)
    row["resource_after_row"] = memory_sample()
    return row


def run_sham(
    env_class: Any,
    sham: dict[str, Any],
    base: dict[str, Any],
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "row_key": f"sham:{sham['sham_id']}",
        "row_type": "SHAM_CONTROL",
        "sham_id": sham["sham_id"],
        "base_identity_id": sham["base_identity_id"],
        "assignment": sham["assignment"],
        "slot": sham["slot"],
        "counts_toward_primary_accuracy_or_completion": False,
        "exception": None,
    }
    try:
        assignment = next(row for row in load(PROTOCOL_PATH)["assignments"] if row["base_identity_id"] == sham["base_identity_id"] and row["assignment"] == sham["assignment"])
        env, observation, exact = make_exact_env(env_class, base, assignment["task_bddl"], sham["mass_factor"], calibration)
        observation, probe = sham_probe(env, observation, sham["slot"], sham["sham_id"], config, calibration, original_protocol)
        audit = probe_audit({**probe, "epoch9e_nondrag_disengagement": {"attempts": [], "all_liftoff_planar_commands_exact_zero": False, "forbidden_inputs_used": []}, "lane_and_reachability_continuous_pass": True}, original_protocol)
        response = float(probe["response_estimated_displacement_m"])
        threshold = float(config.back_heavy_threshold_m)
        scores = {"front": threshold - response, "back": response - threshold}
        row.update({
            "completed": True,
            "exact_state_audit": exact,
            "back_response_m": response,
            "candidate_scores_m": scores,
            "predicted_heavy_slot": min(scores, key=scores.get),
            "probe": probe,
            "probe_audit": audit,
            "method_information_boundary": {
                "mass_assignment_used_only_to_construct_intervention": True,
                "mass_property_or_simulator_pose_passed_to_sham_action_or_score": False,
                "no_completion_oracle_run": True,
            },
        })
    except Exception as exc:
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = float(time.monotonic() - started)
    row["resource_after_row"] = memory_sample()
    return row


def summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "primary_rows": sum(row["row_type"] == "PRIMARY_ASSIGNMENT" for row in rows),
        "sham_rows": sum(row["row_type"] == "SHAM_CONTROL" for row in rows),
        "complete_rows": sum(bool(row.get("completed")) for row in rows),
        "exceptions": sum(row.get("exception") is not None for row in rows),
    }


def main() -> int:
    if RESULT_PATH.exists():
        raise FileExistsError("refusing to overwrite or resume the one-shot joint result")
    _, protocol = validate_seal()
    original_protocol = load(ORIGINAL_PROTOCOL_PATH)
    calibration = campaign.load_calibration()
    config = campaign.ControllerConfig(**protocol["controller_contract"]["base_controller_config"])
    result: dict[str, Any] = {
        "schema_version": "epoch9e.joint_certification_result.v1",
        "started_at": timestamp(),
        "pid": os.getpid(),
        "evidence_partition": protocol["evidence_partition"],
        "protocol_path": relative(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "execution_seal_path": relative(SEAL_PATH),
        "execution_seal_sha256": sha256(SEAL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "controller_name": protocol["controller_contract"]["name"],
        "controller_config": config.as_dict(),
        "one_shot_no_resume": True,
        "rows": [],
        "summary": summary([]),
        "validation_accessed": False,
        "confirmation_accessed": False,
        "resource_monitor": {
            "process_max_rss_bytes": 0,
            "wsl_mem_used_peak_bytes": 0,
            "wsl_swap_used_peak_bytes": 0,
            "gpu_initial": gpu_sample(),
        },
    }
    atomic_write_json(RESULT_PATH, result)
    env_class = campaign.load_env_class()
    bases = base_lookup(protocol)
    manifests = [("primary", row) for row in protocol["assignments"]] + [("sham", row) for row in protocol["sham_control"]["manifest"]]
    for kind, manifest in manifests:
        base = bases[int(manifest["base_identity_id"])]
        row = (run_primary(env_class, manifest, base, config, calibration, original_protocol, protocol["controller_contract"])
               if kind == "primary" else run_sham(env_class, manifest, base, config, calibration, original_protocol))
        result["rows"].append(row)
        result["summary"] = summary(result["rows"])
        update_resource_peaks(result)
        atomic_write_json(RESULT_PATH, result)
        if not row.get("completed"):
            raise RuntimeError(f"joint row failed and was preserved without resume authority: {row['row_key']} {row['exception']}")
        if result["resource_monitor"]["wsl_swap_used_peak_bytes"] != 0:
            raise RuntimeError("WSL swap use detected; joint result preserved and execution stopped")
    result["completed_at"] = timestamp()
    result["resource_monitor"]["gpu_final"] = gpu_sample()
    result["summary"] = summary(result["rows"])
    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
