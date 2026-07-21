#!/usr/bin/env python3
"""Run the sealed, label-blind Epoch 9E mechanics-only smoke set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.epoch9e_nondrag_controller import run_nondrag_probe
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from scripts.run_epoch9d_causal_panel import memory_sample, sha256
from tca_map.epoch7_latent_dynamics import atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9e_mechanics_smoke_protocol.json"
SEAL_PATH = REPORTS / "epoch9e_mechanics_execution_seal.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
OUTPUT_ROOT = REPORTS / "epoch9e_mechanics_smoke"
RESULT_PATH = OUTPUT_ROOT / "result.json"
TRACE_ROOT = OUTPUT_ROOT / "traces"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def initial_localizations(observation: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
    result = {}
    for slot in ("front", "back"):
        _, _, metric = campaign.localize_candidate(frame, slot, calibration)
        result[slot] = {"subpixel_dx": float(metric["subpixel_dx"]), "subpixel_dy": float(metric["subpixel_dy"]), "quality": float(metric["quality"])}
    return result


def make_env(env_class: Any, scene: dict[str, Any], calibration: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    task = campaign.TASKS["front"]
    env = env_class(bddl_file_name=str(BDDL_ROOT / task["bddl"]), camera_heights=128, camera_widths=128)
    env.seed(int(scene["generator_seed"]))
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(scene["base_state_vector_float64"], dtype=np.float64))
    env.sim.forward()
    observation = campaign.forced_observation(env)
    frame_hash = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    localization = initial_localizations(observation, calibration)
    if frame_hash != scene["first_agentview_rgb_sha256"] or localization != scene["initial_rgb_localization_audit"]:
        env.close()
        raise RuntimeError(f"smoke exact state mismatch: {scene['scene_id']}")
    return env, observation, {"first_rgb_exact": True, "initial_localization_exact": True, "mass_factor": scene["mass_factor"]}


def mechanics_audit(probe: dict[str, Any], original_protocol: dict[str, Any]) -> dict[str, Any]:
    trace_path = ROOT / probe["trace_path"]
    with np.load(trace_path, allow_pickle=False) as trace:
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
        quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        actions = np.asarray(trace["action"], dtype=np.float32)
        phases = np.asarray(trace["phase"]).astype(str)
    margins = {"front": [], "back": []}
    for pair in positions:
        for slot_index, slot in enumerate(("front", "back")):
            xyz = pair[slot_index]
            lane = original_protocol["safe_center_lanes_m"][slot]
            reach = original_protocol["reachable_center_envelope_m"]
            margins[slot].append(float(min(xyz[0]-lane["x"][0], lane["x"][1]-xyz[0], xyz[1]-lane["y"][0], lane["y"][1]-xyz[1], xyz[2]-reach["z"][0], reach["z"][1]-xyz[2])))
    workspace_low = np.asarray([-0.25, -0.05, 0.85])
    workspace_high = np.asarray([0.25, 0.25, 1.10])
    front, back = positions[:, 0], positions[:, 1]
    identity_swap = bool(np.any(front[:, 0] <= back[:, 0]))
    for slot_index, slot in enumerate(("front", "back")):
        other = "back" if slot == "front" else "front"
        lane = original_protocol["safe_center_lanes_m"][other]
        xy = positions[:, slot_index, :2]
        identity_swap = identity_swap or bool(np.any((xy[:,0]>=lane["x"][0])&(xy[:,0]<=lane["x"][1])&(xy[:,1]>=lane["y"][0])&(xy[:,1]<=lane["y"][1])))
    nondrag = probe["epoch9e_nondrag_disengagement"]
    return {
        "slot": probe["slot"], "trace_path": relative(trace_path), "trace_sha256": sha256(trace_path),
        "finite_bounded_actions": bool(actions.size and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.0),
        "intended_contact_or_excitation": bool(probe["intended_target_contact_or_excitation_eval_only"]),
        "sampled_intended_contact": bool(probe["intended_target_contact_eval_only"]),
        "full_trajectory_lane_reachable": bool(probe["lane_and_reachability_continuous_pass"]),
        "minimum_continuous_lane_margin_m_eval_only": {slot: min(values) for slot, values in margins.items()},
        "unintended_collision": bool(probe["candidate_pair_collision_eval_only"] or probe["candidate_distractor_collision_eval_only"]),
        "identity_swap": identity_swap, "fall": bool(np.any(positions[:,:,2] < 0.85)),
        "workspace_exit": bool(np.any(positions < workspace_low) or np.any(positions > workspace_high)),
        "unrecoverable_track_loss": bool(not np.isfinite(quality).all() or np.min(quality) < 0.50),
        "disengagement_attempt_count": len(nondrag["attempts"]),
        "all_liftoff_planar_commands_exact_zero": bool(nondrag["all_liftoff_planar_commands_exact_zero"]),
        "all_attempts_separation_verified": bool(nondrag["attempts"] and all(row["separation_verified_ordinary_observations"] for row in nondrag["attempts"])),
        "nondrag_attempts": nondrag["attempts"],
        "phase_counts": {phase: int(np.count_nonzero(phases == phase)) for phase in sorted(set(phases))},
        "mass_rank_computed": False, "mass_conditioned_response_computed_or_revealed": False,
        "oracle_task_success_accessed": False, "reward_done_success_accessed": False,
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("completed")]
    audits = [audit for row in valid for audit in row["probe_audits"]]
    safety = sum(bool(audit[key]) for audit in audits for key in ("unintended_collision", "identity_swap", "fall", "workspace_exit", "unrecoverable_track_loss"))
    return {"scenes": len(rows), "complete_scenes": len(valid), "probes": len(audits), "finite_bounded_actions": sum(audit["finite_bounded_actions"] for audit in audits), "intended_contact_or_excitation": sum(audit["intended_contact_or_excitation"] for audit in audits), "both_candidates_excited_scenes": sum(all(audit["intended_contact_or_excitation"] for audit in row["probe_audits"]) for row in valid), "full_trajectory_lane_reachable": sum(audit["full_trajectory_lane_reachable"] for audit in audits), "safety_or_track_events": safety, "liftoff_zero_planar_probes": sum(audit["all_liftoff_planar_commands_exact_zero"] for audit in audits), "separation_verified_probes": sum(audit["all_attempts_separation_verified"] for audit in audits)}


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--resume", action="store_true")
    resume = args.parse_args().resume
    protocol = load(PROTOCOL_PATH)
    seal = load(SEAL_PATH)
    if sha256(PROTOCOL_PATH) != seal["smoke_protocol_sha256"] or sha256(Path(__file__)) != seal["runner_sha256"]:
        raise RuntimeError("mechanics execution seal mismatch")
    if RESULT_PATH.exists():
        if not resume: raise FileExistsError("refusing to overwrite mechanics smoke result")
        result = load(RESULT_PATH)
    else:
        result = {"schema_version": "epoch9e.mechanics_smoke_result.v1", "started_at": campaign.timestamp(), "pid": os.getpid(), "protocol_path": relative(PROTOCOL_PATH), "protocol_sha256": sha256(PROTOCOL_PATH), "execution_seal_path": relative(SEAL_PATH), "execution_seal_sha256": sha256(SEAL_PATH), "runner_sha256": sha256(Path(__file__)), "rows": [], "mass_rank_computed": False, "mass_conditioned_response_computed_or_revealed": False, "oracle_task_success_accessed": False, "reward_done_success_accessed": False, "validation_accessed": False, "confirmation_accessed": False, "resource": {"process_max_rss_bytes": 0, "wsl_mem_used_peak_bytes": 0, "wsl_swap_used_peak_bytes": 0}}
    original = load(ORIGINAL_PROTOCOL_PATH)
    calibration = campaign.load_calibration()
    config = campaign.ControllerConfig(**protocol["controller_contract"]["base_controller_config"])
    env_class = campaign.load_env_class()
    completed = {row["row_key"] for row in result["rows"]}
    for scene in protocol["manifest"]:
        if scene["scene_id"] in completed: continue
        started = time.monotonic(); env = None
        row = {"row_key": scene["scene_id"], "scene_identity": scene["generated_identity_id"], "spatial_stratum": scene["spatial_stratum"], "probe_order": scene["probe_order"], "completed": False, "exception": None}
        try:
            env, observation, exact = make_env(env_class, scene, calibration)
            probe_audits = []
            for slot in scene["probe_order"]:
                observation, probe = run_nondrag_probe(env, observation, slot, scene["scene_id"], config, calibration, original, protocol["controller_contract"], TRACE_ROOT)
                probe_audits.append(mechanics_audit(probe, original))
            row.update({"completed": True, "exact_state_audit": exact, "probe_audits": probe_audits, "mass_rank_computed": False, "mass_conditioned_response_computed_or_revealed": False, "oracle_task_success_accessed": False, "reward_done_success_accessed": False})
        except Exception as exc:
            row["exception"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None: env.close()
        row["wall_seconds"] = float(time.monotonic()-started)
        result["rows"].append(row); result["summary"] = counts(result["rows"])
        sample = memory_sample(); resource = result["resource"]
        resource["process_max_rss_bytes"] = max(resource["process_max_rss_bytes"], sample["process_max_rss_bytes"])
        resource["wsl_mem_used_peak_bytes"] = max(resource["wsl_mem_used_peak_bytes"], sample["wsl_mem_used_bytes"])
        resource["wsl_swap_used_peak_bytes"] = max(resource["wsl_swap_used_peak_bytes"], sample["wsl_swap_used_bytes"])
        atomic_write_json(RESULT_PATH, result)
        if not row["completed"]: raise RuntimeError(f"smoke row failed: {row['row_key']} {row['exception']}")
        if resource["wsl_swap_used_peak_bytes"] != 0: raise RuntimeError("WSL swap use detected")
    result["completed_at"] = campaign.timestamp(); result["summary"] = counts(result["rows"]); atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
