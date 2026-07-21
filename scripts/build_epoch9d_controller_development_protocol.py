#!/usr/bin/env python3
"""Freeze the bounded Epoch 9D task-preserving controller development plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from tca_map.epoch7_latent_dynamics import atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "epoch9d_controller_development_protocol.json"
CAUSAL_ADJUDICATION = REPORTS / "epoch9d_causal_panel_adjudication.json"
DIAGNOSTIC = REPORTS / "epoch9d_existing_trace_causal_diagnostic.json"
ORIGINAL_PROTOCOL = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
ORIGINAL_FREEZE = REPORTS / "epoch9b_dynamic_nudge/controller_freeze.json"
CALIBRATION = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"
PILOT_IDENTITIES = list(range(72, 84))
PILOT_SEEDS = list(range(914200, 914212))


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


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def geometry(index: int) -> tuple[dict[str, list[float]], str]:
    pattern = index % 6
    repeat = index // 6
    front_y, back_y, name = (
        (0.172, 0.050, "front_upper_edge"),
        (0.118, 0.050, "front_lower_edge"),
        (0.145, 0.087, "back_upper_edge"),
        (0.145, 0.018, "back_lower_edge"),
        (0.168, 0.022, "dual_edge"),
        (0.145, 0.052, "interior"),
    )[pattern]
    front_x = 0.086 + 0.010 * ((2 * pattern + repeat) % 4)
    back_x = -0.168 + 0.010 * ((3 * pattern + repeat) % 4)
    return {"front": [front_x, front_y], "back": [back_x, back_y]}, name


def scene_semantics(index: int) -> dict[str, Any]:
    heavy = "front" if index % 4 in (0, 1) else "back"
    instruction = "heaviest" if index % 2 == 0 else "lightest"
    target = heavy if instruction == "heaviest" else ("back" if heavy == "front" else "front")
    order = ["front", "back"] if (index // 2) % 2 == 0 else ["back", "front"]
    light = "back" if heavy == "front" else "front"
    return {
        "heavy_slot": heavy,
        "instruction_property": instruction,
        "completion_target_slot_eval_only": target,
        "probe_order": order,
        "mass_factor": {heavy: 8.0, light: 1.0},
    }


def lane_margin(protocol: dict[str, Any], slot: str, xyz: np.ndarray) -> float:
    lane = protocol["safe_center_lanes_m"][slot]
    reach = protocol["reachable_center_envelope_m"]
    return float(
        min(
            xyz[0] - lane["x"][0],
            lane["x"][1] - xyz[0],
            xyz[1] - lane["y"][0],
            lane["y"][1] - xyz[1],
            xyz[2] - reach["z"][0],
            reach["z"][1] - xyz[2],
        )
    )


def build_pilot_states() -> list[dict[str, Any]]:
    original_protocol = json.loads(ORIGINAL_PROTOCOL.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    env_class = campaign.load_env_class()
    rows = []
    for index, (identity, seed) in enumerate(zip(PILOT_IDENTITIES, PILOT_SEEDS, strict=True)):
        semantics = scene_semantics(index)
        candidate_xy, stratum = geometry(index)
        task = campaign.TASKS[semantics["completion_target_slot_eval_only"]]
        env = None
        try:
            env = env_class(
                bddl_file_name=str(BDDL_ROOT / task["bddl"]),
                camera_heights=128,
                camera_widths=128,
            )
            env.seed(seed)
            observation = env.reset()
            for _ in range(10):
                observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            campaign.set_scene_candidates(
                env, {"candidate_initial_xy_m": candidate_xy, "mass_factor": {"front": 1.0, "back": 1.0}}
            )
            for _ in range(10):
                observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            observation = campaign.forced_observation(env)
            frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
            state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            positions = {
                slot: campaign.body_position(env.sim, body) for slot, body in campaign.BODY_BY_SLOT.items()
            }
            localization = {}
            for slot in ("front", "back"):
                _, _, metric = campaign.localize_candidate(frame, slot, calibration)
                localization[slot] = {
                    "subpixel_dx": float(metric["subpixel_dx"]),
                    "subpixel_dy": float(metric["subpixel_dy"]),
                    "quality": float(metric["quality"]),
                }
                if localization[slot]["quality"] < 0.50:
                    raise RuntimeError(f"pilot localization quality failed: {identity} {slot}")
                if lane_margin(original_protocol, slot, positions[slot]) <= 0:
                    raise RuntimeError(f"pilot starts outside lane: {identity} {slot}")
            rows.append(
                {
                    "scene_id": f"epoch9d_controller_pilot_{identity:03d}",
                    "generated_identity_id": identity,
                    "generator_seed": seed,
                    "partition": "CONTROLLER_PILOT_DEVELOPMENT",
                    "spatial_stratum": stratum,
                    "candidate_initial_xy_command_m": candidate_xy,
                    "candidate_initial_xyz_eval_only": {slot: positions[slot].tolist() for slot in ("front", "back")},
                    "candidate_initial_lane_margin_m_eval_only": {
                        slot: lane_margin(original_protocol, slot, positions[slot]) for slot in ("front", "back")
                    },
                    "initial_rgb_localization_audit": localization,
                    "base_state_vector_float64": state.tolist(),
                    "base_state_vector_sha256": array_sha256(state),
                    "first_agentview_rgb_sha256": rgb_sha256(frame),
                    "task_bddl": task["bddl"],
                    **semantics,
                    "outcomes_accessed_during_construction": [],
                    "mass_assignment_applied_during_construction": False,
                }
            )
        finally:
            if env is not None:
                env.close()
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite controller development protocol")
    for path in (CAUSAL_ADJUDICATION, DIAGNOSTIC, ORIGINAL_PROTOCOL, ORIGINAL_FREEZE, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)
    causal = json.loads(CAUSAL_ADJUDICATION.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if causal["decision"] != "CAUSAL_SIGNAL_GO":
        raise RuntimeError("controller development is not authorized")
    pilot = build_pilot_states()
    original_freeze = json.loads(ORIGINAL_FREEZE.read_text(encoding="utf-8"))
    protocol = {
        "schema_version": "epoch9d.task_preserving_controller_development_protocol.v1",
        "frozen_at": timestamp(),
        "branch": git("branch", "--show-current"),
        "source_checkpoint": git("rev-parse", "HEAD"),
        "causal_signal_go": {"path": relative(CAUSAL_ADJUDICATION), "sha256": sha256(CAUSAL_ADJUDICATION)},
        "phase_a_diagnostic": {"path": relative(DIAGNOSTIC), "sha256": sha256(DIAGNOSTIC)},
        "diagnosed_limitation": (
            "the original response is causal, the sole original lane failure began from an RGB estimate already beyond the front "
            "upper lane boundary, and the original 21/24 completion headroom already passes the final completion threshold"
        ),
        "search_budget": {
            "maximum_variants_beyond_original": 2,
            "maximum_total_fresh_pilot_scenes": 24,
            "maximum_adjustment_rounds_per_variant": 1,
            "variant1_initial_pilot_scenes": 12,
            "variant1_adjustment_reserved_identity_ids": list(range(84, 96)),
            "variant2_authorized_only_if": "pilot evidence shows post-probe pose, rather than completion control, is limiting",
            "classifier_or_threshold_tuning_on_final_panel": False,
        },
        "variant1": {
            "name": "original_response_with_predictive_inward_lane_guard_and_neutral_retract",
            "base_controller_config": original_freeze["controller_config"],
            "response_window_unchanged": True,
            "score_and_threshold_unchanged": True,
            "guard": {
                "admissible_position_source": "ordinary RGB localization through the frozen pixel-to-world calibration",
                "lane_source": "offline frozen workspace geometry",
                "trigger_signed_y_margin_m_at_or_below": 0.010,
                "trigger_derivation": (
                    "0.007188 m maximum historical observation-derived lateral excursion plus a conservative margin, rounded to 0.010 m"
                ),
                "active_only_after_approach_x_offset_m_at_or_above": -0.062,
                "action": (
                    "place the paddle on the nearer y-boundary side of the candidate so any incidental lateral component points "
                    "toward lane interior; otherwise retain the original slot-specific approach"
                ),
                "identical_for_heavy_and_light": True,
                "mass_or_property_input": False,
            },
            "neutral_retract": "unchanged label-blind -x side retreat, vertical clearance, and proprioceptive neutral return after the five-step response window",
        },
        "variant1_pilot_manifest": pilot,
        "pilot_selection_gate": {
            "complete_scenes": {"minimum": 12, "denominator": 12},
            "finite_bounded_actions": {"minimum": 24, "denominator": 24},
            "intended_contact_or_excitation": {"minimum": 23, "denominator": 24},
            "lane_and_reachability": {"minimum": 24, "denominator": 24},
            "zero_collision_identity_swap_fall_workspace_exit": True,
            "rank_correct": {"minimum": 10, "denominator": 12},
            "rank_each_heavy_position": {"minimum": 5, "denominator": 6},
            "oracle_completion": {"minimum": 10, "denominator": 12},
        },
        "variant1_conditional_adjustment": {
            "authorized_only_if": (
                "the initial pilot passes contact, safety, ranking, and oracle gates but has one or more lane failures whose "
                "pre-contact RGB signed margin lies in (0.010, 0.014] m"
            ),
            "only_change": "increase guard trigger signed y margin from 0.010 m to 0.014 m",
            "fresh_identity_ids": list(range(84, 96)),
            "otherwise_run": False,
        },
        "final_controller_panel": {
            "locked_until_pilot_selection": True,
            "fresh_identity_ids": list(range(96, 120)),
            "scene_count": 24,
            "thresholds": "exact TASK_PRESERVING_CONTROLLER_GO thresholds in the Epoch 9D authority prompt",
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
        "outcomes_accessed_during_protocol_construction": [],
    }
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({
        "output": relative(OUTPUT),
        "pilot_scenes": len(pilot),
        "identity_range": [PILOT_IDENTITIES[0], PILOT_IDENTITIES[-1]],
        "min_initial_lane_margin_m": min(
            value for row in pilot for value in row["candidate_initial_lane_margin_m_eval_only"].values()
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
