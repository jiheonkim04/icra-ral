#!/usr/bin/env python3
"""Run the sealed 12-scene Epoch 9D guarded-controller pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from scripts.run_epoch9d_causal_panel import (
    gpu_sample,
    memory_sample,
    primary_probe_audit,
    sha256,
    update_resource_peaks,
)
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9d_controller_development_protocol.json"
EXECUTION_SEAL_PATH = REPORTS / "epoch9d_controller_pilot_execution_seal.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
OUTPUT_ROOT = REPORTS / "epoch9d_controller_development"
RESULT_PATH = OUTPUT_ROOT / "variant1_pilot_result.json"
TRACE_ROOT = OUTPUT_ROOT / "variant1_pilot_traces"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    seal = load(EXECUTION_SEAL_PATH)
    protocol = load(PROTOCOL_PATH)
    if sha256(PROTOCOL_PATH) != seal["protocol_sha256"]:
        raise RuntimeError("controller pilot protocol hash mismatch")
    if sha256(Path(__file__)) != seal["runner_sha256"]:
        raise RuntimeError("controller pilot runner hash mismatch")
    if seal["outcomes_accessed_before_seal"]:
        raise RuntimeError("invalid execution seal")
    return seal, protocol


def initial_localization(
    observation: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, dict[str, float]]:
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
    result = {}
    for slot in ("front", "back"):
        _, _, metric = campaign.localize_candidate(frame, slot, calibration)
        result[slot] = {
            "subpixel_dx": float(metric["subpixel_dx"]),
            "subpixel_dy": float(metric["subpixel_dy"]),
            "quality": float(metric["quality"]),
        }
    return result


def lane_guard_approach_y(
    candidate_slot: str,
    estimated_target_y: float,
    offset_x: float,
    controller_config: campaign.ControllerConfig,
    original_protocol: dict[str, Any],
    guard: dict[str, Any],
    original_approach_y: float,
) -> tuple[float, dict[str, Any] | None]:
    """Apply the frozen label-blind RGB/geometry lane guard."""

    active_offset = float(guard["active_only_after_approach_x_offset_m_at_or_above"])
    if offset_x < active_offset:
        return float(original_approach_y), None
    lane = original_protocol["safe_center_lanes_m"][candidate_slot]["y"]
    lower_margin = float(estimated_target_y - lane[0])
    upper_margin = float(lane[1] - estimated_target_y)
    signed_margin = min(lower_margin, upper_margin)
    trigger_margin = float(guard["trigger_signed_y_margin_m_at_or_below"])
    if signed_margin > trigger_margin:
        return float(original_approach_y), None
    boundary = "lower" if lower_margin <= upper_margin else "upper"
    paddle_offset = abs(float(controller_config.paddle_y_offset_m))
    guarded_y = (
        float(estimated_target_y - paddle_offset)
        if boundary == "lower"
        else float(estimated_target_y + paddle_offset)
    )
    return guarded_y, {
        "slot": candidate_slot,
        "estimated_target_y_m": float(estimated_target_y),
        "signed_y_lane_margin_m": signed_margin,
        "nearest_boundary": boundary,
        "approach_offset_x_m": float(offset_x),
        "original_approach_y_m": float(original_approach_y),
        "guarded_approach_y_m": guarded_y,
    }


def make_env(
    env_class: Any,
    scene: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    task = campaign.TASKS[scene["completion_target_slot_eval_only"]]
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / task["bddl"]),
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(int(scene["generator_seed"]))
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(scene["base_state_vector_float64"], dtype=np.float64))
    env.sim.forward()
    observation = campaign.forced_observation(env)
    before = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
    if before != scene["first_agentview_rgb_sha256"]:
        env.close()
        raise RuntimeError(f"pilot frozen first RGB mismatch: {scene['scene_id']}")
    for slot, factor in scene["mass_factor"].items():
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
    localization = initial_localization(observation, calibration)
    if after != before or localization != scene["initial_rgb_localization_audit"]:
        env.close()
        raise RuntimeError(f"pilot mass assignment changed initial observation: {scene['scene_id']}")
    return env, observation, {
        "base_state_vector_sha256": scene["base_state_vector_sha256"],
        "first_rgb_before_mass_sha256": before,
        "first_rgb_after_mass_sha256": after,
        "first_rgb_exact": before == after == scene["first_agentview_rgb_sha256"],
        "initial_rgb_localization_audit": localization,
        "mass_factor_eval_construction_only": scene["mass_factor"],
    }


def guarded_probe(
    env: Any,
    observation: dict[str, Any],
    slot: str,
    scene_id: str,
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
    guard: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_approach_y = campaign.approach_y
    events: list[dict[str, Any]] = []

    def guarded_approach_y(
        candidate_slot: str,
        estimated_target_y: float,
        offset_x: float,
        controller_config: campaign.ControllerConfig,
    ) -> float:
        original = float(
            original_approach_y(candidate_slot, estimated_target_y, offset_x, controller_config)
        )
        guarded_y, event = lane_guard_approach_y(
            candidate_slot,
            estimated_target_y,
            offset_x,
            controller_config,
            original_protocol,
            guard,
            original,
        )
        if event is not None:
            events.append(event)
        return guarded_y

    campaign.approach_y = guarded_approach_y
    try:
        observation, probe = campaign.probe_candidate(
            env,
            observation,
            slot,
            scene_id,
            config,
            calibration,
            original_protocol,
            TRACE_ROOT,
        )
    finally:
        campaign.approach_y = original_approach_y
    probe["predictive_lane_guard"] = {
        "triggered": bool(events),
        "event_count": len(events),
        "events": events,
        "admissible_rgb_geometry_only": True,
        "mass_or_property_input": False,
    }
    return observation, probe


def run_scene(
    env_class: Any,
    scene: dict[str, Any],
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "row_key": scene["scene_id"],
        "scene": scene,
        "exception": None,
    }
    try:
        env, observation, exact = make_env(env_class, scene, calibration)
        probes = []
        for slot in scene["probe_order"]:
            observation, probe = guarded_probe(
                env,
                observation,
                slot,
                scene["scene_id"],
                config,
                calibration,
                original_protocol,
                guard,
            )
            probes.append(probe)
        audits = {probe["slot"]: primary_probe_audit(probe, original_protocol) for probe in probes}
        responses = {probe["slot"]: float(probe["response_estimated_displacement_m"]) for probe in probes}
        threshold = float(config.back_heavy_threshold_m)
        scores = {"front": threshold - responses["back"], "back": responses["back"] - threshold}
        predicted = min(scores, key=scores.get)
        observation, oracle = campaign.oracle_completion(
            env,
            observation,
            scene["completion_target_slot_eval_only"],
            calibration,
            config,
        )
        row.update(
            {
                "completed": True,
                "exact_state_audit": exact,
                "probes": probes,
                "probe_audits": audits,
                "responses_m": responses,
                "candidate_scores_m": scores,
                "predicted_heavy_slot": predicted,
                "heavy_rank_correct": predicted == scene["heavy_slot"],
                "oracle_completion": oracle,
                "method_information_boundary": {
                    "guard_inputs": ["ordinary RGB localization", "offline frozen lane geometry"],
                    "mass_or_property_passed_to_guard_probe_or_score": False,
                    "simulator_pose_passed_to_guard_probe_or_score": False,
                    "oracle_privilege_evaluation_only": True,
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("completed")]
    audits = [audit for row in valid for audit in row["probe_audits"].values()]
    by_heavy = {}
    for slot in ("front", "back"):
        subset = [row for row in valid if row["scene"]["heavy_slot"] == slot]
        by_heavy[slot] = {"correct": sum(row["heavy_rank_correct"] for row in subset), "total": len(subset)}
    counts = {
        "scenes": len(rows),
        "complete_scenes": len(valid),
        "probes": len(audits),
        "finite_bounded_actions": sum(audit["finite_bounded_actions"] for audit in audits),
        "intended_contact_or_excitation": sum(audit["intended_contact_or_excitation"] for audit in audits),
        "lane_and_reachability": sum(
            probe["lane_and_reachability_continuous_pass"] for row in valid for probe in row["probes"]
        ),
        "collisions": sum(audit["unintended_collision"] for audit in audits),
        "identity_swaps": sum(audit["identity_swap"] for audit in audits),
        "falls": sum(audit["fall"] for audit in audits),
        "workspace_exits": sum(audit["workspace_exit"] for audit in audits),
        "track_losses": sum(audit["unrecoverable_track_loss"] for audit in audits),
        "rank_correct": sum(row["heavy_rank_correct"] for row in valid),
        "rank_by_heavy_position": by_heavy,
        "oracle_completion": sum(row["oracle_completion"]["official_task_success"] for row in valid),
        "guard_triggered_probes": sum(
            probe["predictive_lane_guard"]["triggered"] for row in valid for probe in row["probes"]
        ),
    }
    counts["pilot_selection_gate"] = bool(
        counts["complete_scenes"] == 12
        and counts["finite_bounded_actions"] == 24
        and counts["intended_contact_or_excitation"] >= 23
        and counts["lane_and_reachability"] == 24
        and counts["collisions"] + counts["identity_swaps"] + counts["falls"] + counts["workspace_exits"] == 0
        and counts["rank_correct"] >= 10
        and all(value["correct"] >= 5 for value in by_heavy.values())
        and counts["oracle_completion"] >= 10
    )
    return counts


def run(resume: bool) -> dict[str, Any]:
    seal, protocol = validate_seal()
    del seal
    original_protocol = load(ORIGINAL_PROTOCOL_PATH)
    calibration = campaign.load_calibration()
    config = campaign.ControllerConfig(**protocol["variant1"]["base_controller_config"])
    if RESULT_PATH.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {RESULT_PATH}")
        result = load(RESULT_PATH)
        if result["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError("pilot resume protocol mismatch")
    else:
        result = {
            "schema_version": "epoch9d.controller_variant1_pilot_result.v1",
            "started_at": campaign.timestamp(),
            "pid": os.getpid(),
            "protocol_path": relative(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "execution_seal_path": relative(EXECUTION_SEAL_PATH),
            "execution_seal_sha256": sha256(EXECUTION_SEAL_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "controller": protocol["variant1"],
            "manifest": protocol["variant1_pilot_manifest"],
            "rows": [],
            "resource_monitor": {
                "process_max_rss_bytes": 0,
                "wsl_mem_used_peak_bytes": 0,
                "wsl_swap_used_peak_bytes": 0,
                "gpu_initial": gpu_sample(),
            },
            "validation_accessed": False,
            "confirmation_accessed": False,
        }
    completed = {row["row_key"] for row in result["rows"]}
    env_class = campaign.load_env_class()
    for scene in protocol["variant1_pilot_manifest"]:
        if scene["scene_id"] in completed:
            continue
        row = run_scene(
            env_class,
            scene,
            config,
            calibration,
            original_protocol,
            protocol["variant1"]["guard"],
        )
        result["rows"].append(row)
        result["summary"] = summarize(result["rows"])
        update_resource_peaks(result)
        atomic_write_json(RESULT_PATH, result)
        if not row.get("completed"):
            raise RuntimeError(f"pilot row failed and was preserved: {row['row_key']} {row['exception']}")
        if result["resource_monitor"]["wsl_swap_used_peak_bytes"] != 0:
            raise RuntimeError("WSL swap use detected")
    result["completed_at"] = campaign.timestamp()
    result["summary"] = summarize(result["rows"])
    result["resource_monitor"]["gpu_final"] = gpu_sample()
    atomic_write_json(RESULT_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().resume)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
