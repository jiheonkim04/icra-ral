#!/usr/bin/env python3
"""Run the sealed Epoch 9D exact-state mass-swap causal panel serially."""

from __future__ import annotations

import argparse
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
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9d_causal_panel_protocol.json"
EXECUTION_SEAL_PATH = REPORTS / "epoch9d_causal_execution_seal.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
OUTPUT_ROOT = REPORTS / "epoch9d_causal_panel"
RESULT_PATH = OUTPUT_ROOT / "result.json"
TRACE_ROOT = OUTPUT_ROOT / "traces"
SHAM_TRACE_ROOT = OUTPUT_ROOT / "sham_traces"
PLATE_BODY = "plate_1_main"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def memory_sample() -> dict[str, Any]:
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
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
        return {"status": "AVAILABLE", "query": output}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"status": "UNAVAILABLE"}


def update_resource_peaks(result: dict[str, Any]) -> None:
    sample = memory_sample()
    peaks = result.setdefault(
        "resource_monitor",
        {
            "process_max_rss_bytes": 0,
            "wsl_mem_used_peak_bytes": 0,
            "wsl_swap_used_peak_bytes": 0,
            "gpu_initial": gpu_sample(),
        },
    )
    peaks["process_max_rss_bytes"] = max(peaks["process_max_rss_bytes"], sample["process_max_rss_bytes"])
    peaks["wsl_mem_used_peak_bytes"] = max(peaks["wsl_mem_used_peak_bytes"], sample["wsl_mem_used_bytes"])
    peaks["wsl_swap_used_peak_bytes"] = max(peaks["wsl_swap_used_peak_bytes"], sample["wsl_swap_used_bytes"])
    peaks["last_sample"] = sample


def validate_execution_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    if not EXECUTION_SEAL_PATH.exists():
        raise FileNotFoundError("missing causal execution seal")
    seal = load_json(EXECUTION_SEAL_PATH)
    protocol = load_json(PROTOCOL_PATH)
    if sha256(PROTOCOL_PATH) != seal["causal_protocol_sha256"]:
        raise RuntimeError("causal protocol hash mismatch")
    if sha256(Path(__file__)) != seal["runner_sha256"]:
        raise RuntimeError("causal runner hash mismatch")
    if seal["outcomes_accessed_before_seal"]:
        raise RuntimeError("execution seal records prior outcome access")
    return seal, protocol


def base_by_identity(protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["base_identity_id"]): row for row in protocol["base_states"]}


def make_exact_env(
    env_class: Any,
    base: dict[str, Any],
    mass_factor: dict[str, float],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    task = campaign.TASKS["front"]
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / task["bddl"]),
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(int(base["generator_seed"]))
    env.reset()
    state = np.asarray(base["base_state_vector_float64"], dtype=np.float64)
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    observation = campaign.forced_observation(env)
    before_mass_hash = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    if before_mass_hash != base["first_agentview_rgb_sha256"]:
        env.close()
        raise RuntimeError(
            f"frozen first-observation mismatch for base {base['base_identity_id']}: "
            f"{before_mass_hash} != {base['first_agentview_rgb_sha256']}"
        )
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
    after_mass_hash = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    if after_mass_hash != before_mass_hash:
        env.close()
        raise RuntimeError(f"mass assignment changed first RGB for base {base['base_identity_id']}")
    applied_masses = {
        slot: float(env.sim.model.body_mass[int(env.sim.model.body_name2id(body))])
        for slot, body in campaign.BODY_BY_SLOT.items()
    }
    audit = {
        "base_state_vector_sha256": base["base_state_vector_sha256"],
        "expected_first_rgb_sha256": base["first_agentview_rgb_sha256"],
        "first_rgb_before_mass_sha256": before_mass_hash,
        "first_rgb_after_mass_sha256": after_mass_hash,
        "first_rgb_exact": before_mass_hash == after_mass_hash == base["first_agentview_rgb_sha256"],
        "baseline_body_mass_eval_construction_only": baseline_masses,
        "applied_body_mass_eval_construction_only": applied_masses,
        "mass_factor_eval_construction_only": mass_factor,
    }
    return env, observation, audit


def initial_localization_audit(
    observation: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, dict[str, float]]:
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
    result: dict[str, dict[str, float]] = {}
    for slot in ("front", "back"):
        _, _, metric = campaign.localize_candidate(frame, slot, calibration)
        result[slot] = {
            "subpixel_dx": float(metric["subpixel_dx"]),
            "subpixel_dy": float(metric["subpixel_dy"]),
            "quality": float(metric["quality"]),
        }
    return result


def response_from_trace(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        estimated = np.asarray(trace["estimated_world_delta_xy_m"], dtype=np.float64)
        quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
        target_contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
    indices = np.flatnonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response", "sham_fixed_micro_impulse", "sham_post_impulse_response"]))
    if indices.size != 5 or not np.all(np.diff(indices) == 1):
        raise RuntimeError(f"trace does not contain one five-step response window: {path}")
    start = int(indices[0])
    baseline = np.median(estimated[max(0, start - 3) : start], axis=0)
    response_axis = estimated[indices, 0] - baseline[0]
    front = positions[:, 0, :]
    back = positions[:, 1, :]
    workspace_low = np.asarray([-0.25, -0.05, 0.85])
    workspace_high = np.asarray([0.25, 0.25, 1.10])
    workspace_exit = bool(
        np.any(front < workspace_low) or np.any(front > workspace_high) or np.any(back < workspace_low) or np.any(back > workspace_high)
    )
    fall = bool(np.any(front[:, 2] < 0.85) or np.any(back[:, 2] < 0.85))
    ordering_reversed = bool(np.any(front[:, 0] <= back[:, 0]))
    return {
        "response_window_indices": indices.tolist(),
        "observation_response_expected_axis_peak_m": float(max(np.max(response_axis), 0.0)),
        "observation_response_expected_axis_by_step_m": response_axis.tolist(),
        "response_tracker_quality_min": float(np.min(quality[indices])),
        "sampled_target_contact": bool(np.any(target_contact)),
        "fall": fall,
        "workspace_exit": workspace_exit,
        "identity_ordering_reversed": ordering_reversed,
        "candidate_positions_finite": bool(np.isfinite(positions).all()),
    }


def cross_lane_identity_swap(protocol: dict[str, Any], trace_path: Path) -> bool:
    with np.load(trace_path, allow_pickle=False) as trace:
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
    front = positions[:, 0, :2]
    back = positions[:, 1, :2]
    front_lane = protocol["safe_center_lanes_m"]["front"]
    back_lane = protocol["safe_center_lanes_m"]["back"]

    def inside(values: np.ndarray, lane: dict[str, list[float]]) -> np.ndarray:
        return (
            (values[:, 0] >= lane["x"][0])
            & (values[:, 0] <= lane["x"][1])
            & (values[:, 1] >= lane["y"][0])
            & (values[:, 1] <= lane["y"][1])
        )

    return bool(np.any(inside(front, back_lane)) or np.any(inside(back, front_lane)))


def primary_probe_audit(probe: dict[str, Any], original_protocol: dict[str, Any]) -> dict[str, Any]:
    trace_path = ROOT / probe["trace_path"]
    trace = response_from_trace(trace_path)
    identity_swap = trace["identity_ordering_reversed"] or cross_lane_identity_swap(original_protocol, trace_path)
    track_loss = bool(
        float(probe["online_localization"]["quality"]) < 0.50
        or trace["response_tracker_quality_min"] < 0.50
        or not np.isfinite(trace["observation_response_expected_axis_peak_m"])
    )
    return {
        **trace,
        "trace_path": relative(trace_path),
        "trace_sha256": sha256(trace_path),
        "identity_swap": identity_swap,
        "unrecoverable_track_loss": track_loss,
        "unintended_collision": bool(
            probe["candidate_pair_collision_eval_only"] or probe["candidate_distractor_collision_eval_only"]
        ),
        "finite_bounded_actions": bool(probe["finite_bounded_actions"]),
        "intended_contact_or_excitation": bool(probe["intended_target_contact_or_excitation_eval_only"]),
        "forbidden_online_inputs_used": probe["forbidden_online_inputs_used"],
        "simulator_state_used_for_actions": bool(probe["simulator_state_used_for_actions"]),
        "mass_or_property_used_for_actions": bool(probe["mass_or_property_used_for_actions"]),
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
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    initial_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    recorder = campaign.EpisodeRecorder(env, initial_frame, slot, center_xy, calibration)
    for _ in range(config.close_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_preclose_paddle")
    prehigh = np.asarray(
        [
            estimated_target[0] + config.approach_start_offset_x_m,
            campaign.approach_y(slot, float(estimated_target[1]), config.approach_start_offset_x_m, config),
            1.02,
        ],
        dtype=np.float64,
    )
    precontact = prehigh.copy()
    precontact[2] = config.paddle_z_m
    observation, transit = campaign.move_to(
        observation, prehigh, recorder, config, "sham_transit_high", gripper=config.gripper_closed_command
    )
    observation, descend = campaign.move_to(
        observation, precontact, recorder, config, "sham_descend_clearance", gripper=config.gripper_closed_command
    )
    for _ in range(config.impulse_steps):
        action = np.zeros(7, dtype=np.float32)
        action[0] = config.impulse_action_x
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_fixed_micro_impulse")
    for _ in range(config.coast_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "sham_post_impulse_response")
    current = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    retreat = current.copy()
    retreat[0] -= 0.055
    observation, retreat_side = campaign.move_to(
        observation, retreat, recorder, config, "sham_retreat_side", gripper=config.gripper_closed_command
    )
    retreat_high = retreat.copy()
    retreat_high[2] = 1.04
    observation, retreat_up = campaign.move_to(
        observation, retreat_high, recorder, config, "sham_retreat_high", gripper=config.gripper_closed_command
    )
    observation, returned = campaign.move_to(
        observation,
        initial_eef,
        recorder,
        config,
        "sham_return_neutral",
        gripper=config.gripper_closed_command,
        tolerance=config.neutral_tolerance_m,
        target_quat=initial_quat,
    )
    trace_path = SHAM_TRACE_ROOT / f"{sham_id}_{slot}.npz"
    campaign.persist_trace(trace_path, recorder)
    trace = response_from_trace(trace_path)
    actions = np.asarray(recorder.action, dtype=np.float32)
    result = {
        "slot": slot,
        "online_localization": localization,
        "finite_bounded_actions": bool(actions.size and all(campaign.legal_action(value) for value in actions)),
        "intended_target_contact_eval_only": bool(any(recorder.target_contact_eval)),
        "intended_target_contact_or_excitation_eval_only": bool(any(recorder.target_contact_eval)),
        "candidate_pair_collision_eval_only": bool(any(recorder.candidate_pair_collision_eval)),
        "candidate_distractor_collision_eval_only": bool(any(recorder.distractor_collision_eval)),
        "response_estimated_displacement_m": trace["observation_response_expected_axis_peak_m"],
        "response_tracker_quality_min": trace["response_tracker_quality_min"],
        "phases_reached": {
            "transit_high": transit,
            "descend_clearance": descend,
            "retreat_side": retreat_side,
            "retreat_high": retreat_up,
            "return_neutral": returned,
        },
        "trace_path": relative(trace_path),
        "forbidden_online_inputs_used": [],
        "simulator_state_used_for_actions": False,
        "mass_or_property_used_for_actions": False,
    }
    return observation, result


def run_primary_scene(
    env_class: Any,
    assignment: dict[str, Any],
    base: dict[str, Any],
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
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
        "probe_order": assignment["probe_order"],
        "exception": None,
    }
    try:
        env, observation, exact = make_exact_env(env_class, base, assignment["mass_factor"])
        exact["initial_rgb_localization_audit"] = initial_localization_audit(observation, calibration)
        exact["initial_rgb_localization_matches_frozen_base"] = (
            exact["initial_rgb_localization_audit"] == base["initial_rgb_localization_audit"]
        )
        probes = []
        for slot in assignment["probe_order"]:
            observation, probe = campaign.probe_candidate(
                env,
                observation,
                slot,
                assignment["scene_id"],
                config,
                calibration,
                original_protocol,
                TRACE_ROOT,
            )
            probes.append(probe)
        audits = {probe["slot"]: primary_probe_audit(probe, original_protocol) for probe in probes}
        responses = {probe["slot"]: float(probe["response_estimated_displacement_m"]) for probe in probes}
        threshold = float(config.back_heavy_threshold_m)
        scores = {"front": threshold - responses["back"], "back": responses["back"] - threshold}
        predicted = min(scores, key=scores.get)
        row.update(
            {
                "completed": True,
                "exact_state_audit": exact,
                "responses_m": responses,
                "candidate_scores_m": scores,
                "predicted_heavy_slot": predicted,
                "heavy_rank_correct_eval_only": predicted == assignment["heavy_slot_eval_only"],
                "probes": probes,
                "probe_audits": audits,
                "both_candidates_excited_eval_only": all(
                    probe["intended_target_contact_or_excitation_eval_only"] for probe in probes
                ),
                "method_information_boundary": {
                    "mass_assignment_used_only_to_construct_intervention": True,
                    "mass_or_property_passed_to_probe_or_score": False,
                    "simulator_pose_passed_to_probe_or_score": False,
                },
            }
        )
    except Exception as exc:
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = float(time.monotonic() - started)
    row["resource_after_row"] = memory_sample()
    return row


def run_sham_row(
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
        "counts_toward_32_scene_accuracy_gate": False,
        "exception": None,
    }
    try:
        env, observation, exact = make_exact_env(env_class, base, sham["mass_factor"])
        exact["initial_rgb_localization_audit"] = initial_localization_audit(observation, calibration)
        exact["initial_rgb_localization_matches_frozen_base"] = (
            exact["initial_rgb_localization_audit"] == base["initial_rgb_localization_audit"]
        )
        observation, probe = sham_probe(
            env,
            observation,
            sham["slot"],
            sham["sham_id"],
            config,
            calibration,
            original_protocol,
        )
        audit = primary_probe_audit(probe, original_protocol)
        response = float(probe["response_estimated_displacement_m"])
        threshold = float(config.back_heavy_threshold_m)
        scores = {"front": threshold - response, "back": response - threshold}
        row.update(
            {
                "completed": True,
                "exact_state_audit": exact,
                "back_response_m": response,
                "candidate_scores_m": scores,
                "predicted_heavy_slot": min(scores, key=scores.get),
                "probe": probe,
                "probe_audit": audit,
                "method_information_boundary": {
                    "mass_assignment_used_only_to_construct_intervention": True,
                    "mass_or_property_passed_to_sham_action_or_score": False,
                    "simulator_pose_passed_to_sham_action_or_score": False,
                },
            }
        )
    except Exception as exc:
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = float(time.monotonic() - started)
    row["resource_after_row"] = memory_sample()
    return row


def expected_rows(protocol: dict[str, Any], mode: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if mode in ("primary", "all"):
        rows.extend(("primary", row) for row in protocol["assignments"])
    if mode in ("sham", "all"):
        rows.extend(("sham", row) for row in protocol["sham_control"]["manifest"])
    return rows


def run(mode: str, resume: bool) -> dict[str, Any]:
    seal, protocol = validate_execution_seal()
    original_protocol = load_json(ORIGINAL_PROTOCOL_PATH)
    calibration = campaign.load_calibration()
    config = campaign.ControllerConfig(**protocol["original_controller_freeze"]["controller_config"])
    bases = base_by_identity(protocol)
    if RESULT_PATH.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {RESULT_PATH}")
        result = load_json(RESULT_PATH)
        if result["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("resume protocol mismatch")
        if result["execution_seal_sha256"] != sha256(EXECUTION_SEAL_PATH):
            raise RuntimeError("resume execution seal mismatch")
    else:
        result = {
            "schema_version": "epoch9d.causal_mass_swap_panel_result.v1",
            "started_at": timestamp(),
            "pid": os.getpid(),
            "mode": mode,
            "evidence_partition": "DEVELOPMENT_CAUSAL_INTERVENTION",
            "protocol_path": relative(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "execution_seal_path": relative(EXECUTION_SEAL_PATH),
            "execution_seal_sha256": sha256(EXECUTION_SEAL_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "controller_config": config.as_dict(),
            "rows": [],
            "validation_accessed": False,
            "confirmation_accessed": False,
            "resource_monitor": {
                "process_max_rss_bytes": 0,
                "wsl_mem_used_peak_bytes": 0,
                "wsl_swap_used_peak_bytes": 0,
                "gpu_initial": gpu_sample(),
            },
        }
    completed = {row["row_key"] for row in result["rows"]}
    env_class = campaign.load_env_class()
    for kind, manifest in expected_rows(protocol, mode):
        key = f"primary:{manifest['scene_id']}" if kind == "primary" else f"sham:{manifest['sham_id']}"
        if key in completed:
            continue
        base = bases[int(manifest["base_identity_id"])]
        row = (
            run_primary_scene(env_class, manifest, base, config, calibration, original_protocol)
            if kind == "primary"
            else run_sham_row(env_class, manifest, base, config, calibration, original_protocol)
        )
        result["rows"].append(row)
        update_resource_peaks(result)
        atomic_write_json(RESULT_PATH, result)
        if not row.get("completed"):
            raise RuntimeError(f"row failed and was preserved: {row['row_key']} {row['exception']}")
        if result["resource_monitor"]["wsl_swap_used_peak_bytes"] != 0:
            raise RuntimeError("WSL swap use detected; result preserved and execution stopped")
    result["completed_at"] = timestamp()
    result["resource_monitor"]["gpu_final"] = gpu_sample()
    result["summary"] = {
        "rows": len(result["rows"]),
        "primary_rows": sum(row["row_type"] == "PRIMARY_ASSIGNMENT" for row in result["rows"]),
        "sham_rows": sum(row["row_type"] == "SHAM_CONTROL" for row in result["rows"]),
        "complete_rows": sum(bool(row.get("completed")) for row in result["rows"]),
        "exceptions": sum(row.get("exception") is not None for row in result["rows"]),
    }
    atomic_write_json(RESULT_PATH, result)
    return result


def preflight() -> dict[str, Any]:
    seal, protocol = validate_execution_seal()
    del seal
    env_class = campaign.load_env_class()
    bases = base_by_identity(protocol)
    audits = []
    for assignment in protocol["assignments"][:2]:
        env = None
        try:
            env, _, audit = make_exact_env(
                env_class, bases[int(assignment["base_identity_id"])], assignment["mass_factor"]
            )
            audits.append({"assignment": assignment["assignment"], **audit})
        finally:
            if env is not None:
                env.close()
    result = {
        "schema_version": "epoch9d.causal_execution_preflight.v1",
        "timestamp": timestamp(),
        "base_identity_id": protocol["assignments"][0]["base_identity_id"],
        "assignments_checked": [row["assignment"] for row in audits],
        "first_rgb_hashes": [row["first_rgb_after_mass_sha256"] for row in audits],
        "exact_pair_first_rgb": len(set(row["first_rgb_after_mass_sha256"] for row in audits)) == 1,
        "mass_factors_change_only_model_dynamics": True,
        "outcomes_accessed": [],
        "resource": memory_sample(),
    }
    destination = REPORTS / "epoch9d_causal_execution_preflight.json"
    if destination.exists():
        raise FileExistsError("refusing to overwrite causal execution preflight")
    atomic_write_json(destination, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "primary", "sham", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = preflight() if args.mode == "preflight" else run(args.mode, args.resume)
    print(json.dumps(result.get("summary", result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
