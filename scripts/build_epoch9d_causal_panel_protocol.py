#!/usr/bin/env python3
"""Freeze exact generated base states and the Epoch 9D causal protocol.

This script performs outcome-suppressed simulator setup only. It does not
apply a heavy-mass assignment, execute a probe, inspect reward/success, or
access validation/confirmation identities.
"""

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
OUTPUT = REPORTS / "epoch9d_causal_panel_protocol.json"
DIAGNOSTIC = REPORTS / "epoch9d_existing_trace_causal_diagnostic.json"
CAMPAIGN_STATE = REPORTS / "epoch9d_campaign_state.json"
IDENTITY_INVENTORY = REPORTS / "epoch9d_identity_seed_inventory.json"
ORIGINAL_CONTROLLER_FREEZE = REPORTS / "epoch9b_dynamic_nudge/controller_freeze.json"
ORIGINAL_PROTOCOL = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
CALIBRATION = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"
BRANCH = "codex/epoch9d-causal-probe-bounded-convergence"
BASE_IDENTITIES = list(range(56, 72))
BASE_SEEDS = list(range(914100, 914116))


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def candidate_geometry(index: int) -> tuple[dict[str, list[float]], str]:
    # Four label-blind spatial strata, each repeated four times. Coordinates
    # stay within the historical RGB calibration search region and at least
    # 11 mm inside every frozen lane boundary.
    stratum = index % 4
    repeat = index // 4
    front_x = [0.082, 0.094, 0.106, 0.118][(stratum + repeat) % 4]
    front_y = [0.128, 0.142, 0.156, 0.164][stratum]
    back_x = [-0.172, -0.158, -0.144, -0.130][(2 * stratum + repeat) % 4]
    back_y = [0.028, 0.044, 0.060, 0.076][stratum]
    return {
        "candidate_1_front": [front_x, front_y],
        "candidate_2_back": [back_x, back_y],
    }, f"spatial_{stratum}"


def probe_order(index: int) -> list[str]:
    # Two orders per spatial stratum in each half of the panel.
    return ["front", "back"] if ((index // 4) + (index % 2)) % 2 == 0 else ["back", "front"]


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


def build_base_states() -> list[dict[str, Any]]:
    protocol = json.loads(ORIGINAL_PROTOCOL.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    env_class = campaign.load_env_class()
    task = campaign.TASKS["front"]
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / task["bddl"]),
        camera_heights=128,
        camera_widths=128,
    )
    bases: list[dict[str, Any]] = []
    try:
        for index, (identity, seed) in enumerate(zip(BASE_IDENTITIES, BASE_SEEDS, strict=True)):
            geometry, stratum = candidate_geometry(index)
            env.seed(seed)
            observation = env.reset()
            for _ in range(10):
                observation, _, _, _ = env.step(
                    np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                )
            scene = {
                "candidate_initial_xy_m": {
                    "front": geometry["candidate_1_front"],
                    "back": geometry["candidate_2_back"],
                },
                "mass_factor": {"front": 1.0, "back": 1.0},
            }
            campaign.set_scene_candidates(env, scene)
            observation = campaign.forced_observation(env)
            for _ in range(10):
                observation, _, _, _ = env.step(
                    np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                )
            observation = campaign.forced_observation(env)
            state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
            positions = {
                slot: campaign.body_position(env.sim, body) for slot, body in campaign.BODY_BY_SLOT.items()
            }
            localizations: dict[str, Any] = {}
            for slot in ("front", "back"):
                _, _, metric = campaign.localize_candidate(frame, slot, calibration)
                localizations[slot] = {
                    "subpixel_dx": float(metric["subpixel_dx"]),
                    "subpixel_dy": float(metric["subpixel_dy"]),
                    "quality": float(metric["quality"]),
                }
                if float(metric["quality"]) < 0.50:
                    raise RuntimeError(f"untrackable frozen base {identity} {slot}: {metric}")
                if lane_margin(protocol, slot, positions[slot]) < 0.010:
                    raise RuntimeError(f"insufficient frozen base lane margin {identity} {slot}: {positions[slot]}")
            bases.append(
                {
                    "base_identity_id": identity,
                    "generator_seed": seed,
                    "partition": "DEVELOPMENT_CAUSAL_PANEL",
                    "spatial_stratum": stratum,
                    "probe_order": probe_order(index),
                    "candidate_mapping": {"candidate_1": "front", "candidate_2": "back"},
                    "candidate_initial_xy_command_m": {
                        "front": geometry["candidate_1_front"],
                        "back": geometry["candidate_2_back"],
                    },
                    "candidate_initial_xyz_eval_only": {
                        slot: positions[slot].tolist() for slot in ("front", "back")
                    },
                    "candidate_initial_lane_margin_m_eval_only": {
                        slot: lane_margin(protocol, slot, positions[slot]) for slot in ("front", "back")
                    },
                    "candidate_center_separation_m_eval_only": float(
                        np.linalg.norm(positions["front"] - positions["back"])
                    ),
                    "initial_rgb_localization_audit": localizations,
                    "base_state_vector_float64": state.tolist(),
                    "base_state_vector_sha256": array_sha256(state),
                    "first_agentview_rgb_sha256": rgb_sha256(frame),
                    "outcomes_accessed_during_construction": [],
                    "mass_assignment_applied_during_construction": False,
                }
            )
    finally:
        env.close()
    return bases


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite frozen Epoch 9D causal protocol")
    for path in (
        DIAGNOSTIC,
        CAMPAIGN_STATE,
        IDENTITY_INVENTORY,
        ORIGINAL_CONTROLLER_FREEZE,
        ORIGINAL_PROTOCOL,
        CALIBRATION,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("causal protocol must be frozen on the Epoch 9D continuation branch")
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    state = json.loads(CAMPAIGN_STATE.read_text(encoding="utf-8"))
    inventory = json.loads(IDENTITY_INVENTORY.read_text(encoding="utf-8"))
    if diagnostic["decision"] != "FREEZE_ORIGINAL_PRIMARY_SCORE_AND_RUN_EXACT_STATE_MASS_SWAP_CAUSAL_PANEL":
        raise RuntimeError("Phase A did not authorize the causal protocol")
    if diagnostic["frozen_primary_score"]["secondary_score_frozen"]:
        raise RuntimeError("unexpected secondary score")
    if state["validation_accessed"] or state["confirmation_accessed"]:
        raise RuntimeError("sealed identities have been accessed")
    if state["identity_and_seed_allocations"]["causal_panel_generated_base_identity_ids"] != BASE_IDENTITIES:
        raise RuntimeError("base identity allocation drift")
    if set(BASE_SEEDS).intersection(inventory["seed_values"]):
        raise RuntimeError("causal generator seed reuse")

    bases = build_base_states()
    assignments = []
    for base in bases:
        base_id = int(base["base_identity_id"])
        assignments.extend(
            (
                {
                    "scene_id": f"epoch9d_causal_base_{base_id:03d}_assignment_A",
                    "base_identity_id": base_id,
                    "assignment": "A",
                    "mass_factor": {"front": 1.0, "back": 8.0},
                    "heavy_slot_eval_only": "back",
                    "probe_order": base["probe_order"],
                },
                {
                    "scene_id": f"epoch9d_causal_base_{base_id:03d}_assignment_B",
                    "base_identity_id": base_id,
                    "assignment": "B",
                    "mass_factor": {"front": 8.0, "back": 1.0},
                    "heavy_slot_eval_only": "front",
                    "probe_order": base["probe_order"],
                },
            )
        )
    sham = []
    for base in bases[:8]:
        for assignment, factors in (
            ("A", {"front": 1.0, "back": 8.0}),
            ("B", {"front": 8.0, "back": 1.0}),
        ):
            sham.append(
                {
                    "sham_id": f"epoch9d_sham_base_{base['base_identity_id']:03d}_assignment_{assignment}",
                    "base_identity_id": base["base_identity_id"],
                    "assignment": assignment,
                    "mass_factor": factors,
                    "slot": "back",
                    "sham_action": (
                        "hold at the original high-clearance approach-start pose; apply the identical three +x action=0.65 "
                        "steps and two zero-action response steps without entering the guarded contact approach"
                    ),
                    "counts_toward_32_scene_accuracy_gate": False,
                }
            )

    controller_freeze = json.loads(ORIGINAL_CONTROLLER_FREEZE.read_text(encoding="utf-8"))
    score = diagnostic["frozen_primary_score"]
    protocol = {
        "schema_version": "epoch9d.causal_mass_swap_panel_protocol.v1",
        "frozen_at": timestamp(),
        "branch": BRANCH,
        "source_checkpoint": git("rev-parse", "HEAD"),
        "evidence_partition": "DEVELOPMENT_CAUSAL_INTERVENTION",
        "outcomes_accessed_during_protocol_construction": [],
        "validation_accessed": False,
        "confirmation_accessed": False,
        "phase_a_diagnostic": {"path": relative(DIAGNOSTIC), "sha256": sha256(DIAGNOSTIC)},
        "original_controller_freeze": {
            "path": relative(ORIGINAL_CONTROLLER_FREEZE),
            "sha256": sha256(ORIGINAL_CONTROLLER_FREEZE),
            "controller_config": controller_freeze["controller_config"],
            "controller_must_match_exactly": True,
        },
        "original_response_window": {
            "impulse_action_x": 0.65,
            "impulse_steps": 3,
            "coast_steps": 2,
            "window_steps": 5,
            "action_sequence_x": [0.65, 0.65, 0.65, 0.0, 0.0],
            "safety_supervisor_inside_window_authorized": False,
        },
        "primary_score": score,
        "secondary_score": None,
        "base_state_count": 16,
        "assignment_scene_count": 32,
        "candidate_probe_count": 64,
        "base_states": bases,
        "assignments": assignments,
        "balance_requirements": {
            "assignments_per_base": 2,
            "heavy_position_counts": {"front": 16, "back": 16},
            "probe_order_counts": {"front_first": 16, "back_first": 16},
            "spatial_strata": {f"spatial_{index}": 4 for index in range(4)},
            "exact_pair_rule": (
                "A and B load the same frozen float64 simulator state, use the same geometry, rendering, robot state, "
                "probe order, and controller; only body mass/inertia factors differ. First RGB hashes must match exactly."
            ),
        },
        "sham_control": {
            "base_state_count": 8,
            "sham_row_count": 16,
            "manifest": sham,
            "required_findings": {
                "intended_or_sampled_contact_rows": 0,
                "unintended_collision_rows": 0,
                "assignment_pair_prediction_flips": 0,
                "paired_mass_contrast_95_interval_must_include_zero": True,
                "ranking_rows_do_not_count_toward_primary_gate": True,
            },
        },
        "causal_signal_go": {
            "finite_bounded_actions_probes": {"minimum": 64, "denominator": 64},
            "intended_contact_or_excitation_probes": {"minimum": 61, "denominator": 64},
            "both_candidates_excited_scenes": {"minimum": 29, "denominator": 32},
            "collision_identity_swap_fall_workspace_exit_unrecoverable_track_loss": {"maximum": 0},
            "correct_heavy_light_rank_scenes": {"minimum": 26, "denominator": 32},
            "correct_rank_each_heavy_position_stratum": {"minimum": 12, "denominator": 16},
            "exact_state_pairs_both_assignments_flip_correctly": {"minimum": 12, "denominator": 16},
            "mass_intervention_test": {
                "pair_contrast": "back_response_when_back_light_assignment_B - back_response_when_back_heavy_assignment_A",
                "expected_direction": "positive",
                "one_sided_exact_sign_test_p_strictly_less_than": 0.01,
                "zero_differences": "discard only for the exact sign test and report count",
                "paired_interval": "two-sided 95% Student-t interval for the mean of 16 exact-pair contrasts",
                "paired_interval_required": "lower endpoint > 0",
            },
            "position_lane_order_adjustment": {
                "model": (
                    "OLS on the 16 pair contrasts with centered initial back x, centered initial back y, centered initial "
                    "back lane margin, probe-order indicator, and three spatial-stratum indicators"
                ),
                "effect": "intercept at centered covariates",
                "uncertainty": "HC3 two-sided 95% interval",
                "required": "positive intercept and lower interval endpoint > 0",
                "pairing_note": "exact swapping already blocks every base-state fixed nuisance",
            },
            "precontact_and_position_controls": {
                "first_rgb_hash_match_pairs": {"minimum": 16, "denominator": 16},
                "position_order_only_max_correct": {"maximum": 16, "denominator": 32},
                "position_order_only_max_pair_flips": 0,
                "precontact_rgb_assignment_contrast": "must be exactly zero for hash and stored localization fields",
                "sham_must_not_reproduce_primary_signal": True,
            },
            "primary_score_inputs_must_be_admissible": True,
            "all_rows_reported": True,
        },
        "near_miss_replication": {
            "eligible_only_if": (
                "no collision, identity swap, leakage, missing row, or protocol violation and exactly one numerical signal "
                "gate misses by no more than one scene or two probes"
            ),
            "replication": "one independently frozen fresh identical panel that must independently pass every gate",
            "pooling": False,
            "second_replication": False,
        },
        "failure_decision": "ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_NOT_CONFIRMED",
        "success_decision": "CAUSAL_SIGNAL_GO",
        "resource_contract": {
            "simulator_environments_at_once": 1,
            "resident_vla_models": 0,
            "serial_execution": True,
            "host_ram_ceiling_percent": 82.0,
            "wsl_swap_peak_bytes": 0,
            "hidden_offload": False,
            "result_changing_precision": False,
            "missing_key_only_resume": True,
        },
        "safety_definitions": {
            "identity_swap": "front or back body crosses into the other candidate's frozen lane or their x ordering reverses",
            "fall": "either candidate center z < 0.85 m at any stored step",
            "workspace_exit": "either candidate center leaves x [-0.25,0.25], y [-0.05,0.25], z [0.85,1.10] m",
            "unrecoverable_track_loss": (
                "initial localization or any response-window tracker quality below 0.50, or a non-finite observation-derived response"
            ),
            "collision": "any candidate-pair or candidate-distractor contact; intended paddle-candidate contact is not a collision",
        },
    }
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({
        "output": relative(OUTPUT),
        "base_states": len(bases),
        "assignments": len(assignments),
        "sham_rows": len(sham),
        "first_state": bases[0]["base_state_vector_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
