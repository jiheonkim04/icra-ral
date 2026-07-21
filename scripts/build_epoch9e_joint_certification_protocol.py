#!/usr/bin/env python3
"""Freeze exact states and the one-shot Epoch 9E joint certification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
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
OUTPUT_JSON = REPORTS / "epoch9e_joint_certification_protocol.json"
OUTPUT_MD = REPORTS / "epoch9e_joint_certification_protocol.md"
SCOPE = REPORTS / "epoch9e_scope_and_authority_correction.json"
ENDPOINT = REPORTS / "epoch9e_endpoint_construct_audit.json"
IDENTITIES = REPORTS / "epoch9e_fresh_identity_manifest.json"
ORIGINAL_CONTROLLER = REPORTS / "epoch9b_dynamic_nudge/controller_freeze.json"
ORIGINAL_PROTOCOL = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
CALIBRATION = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"
BRANCH = "codex/epoch9e-nondrag-disengagement-convergence"


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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def geometry(index: int) -> tuple[dict[str, list[float]], str, str]:
    pattern = index % 6
    repeat = index // 6
    front_y, back_y, stratum, lane_stratum = (
        (0.170, 0.050, "front_upper_edge", "front_edge"),
        (0.120, 0.050, "front_lower_edge", "front_edge"),
        (0.145, 0.085, "back_upper_edge", "back_edge"),
        (0.145, 0.020, "back_lower_edge", "back_edge"),
        (0.168, 0.022, "dual_edge", "dual_edge"),
        (0.145, 0.052, "interior", "interior"),
    )[pattern]
    front_x = 0.084 + 0.010 * ((2 * pattern + repeat) % 4)
    back_x = -0.170 + 0.010 * ((3 * pattern + repeat) % 4)
    return {"front": [front_x, front_y], "back": [back_x, back_y]}, stratum, lane_stratum


def probe_order(index: int) -> list[str]:
    return ["front", "back"] if index % 4 in (0, 3) else ["back", "front"]


def instruction_property(index: int) -> str:
    return "heaviest" if index % 2 == 0 else "lightest"


def lane_margin(protocol: dict[str, Any], slot: str, xyz: np.ndarray) -> float:
    lane = protocol["safe_center_lanes_m"][slot]
    reach = protocol["reachable_center_envelope_m"]
    return float(min(xyz[0] - lane["x"][0], lane["x"][1] - xyz[0], xyz[1] - lane["y"][0], lane["y"][1] - xyz[1], xyz[2] - reach["z"][0], reach["z"][1] - xyz[2]))


def build_base_states(ids: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    original_protocol = load(ORIGINAL_PROTOCOL)
    calibration = load(CALIBRATION)
    env_class = campaign.load_env_class()
    task = campaign.TASKS["front"]
    env = env_class(bddl_file_name=str(BDDL_ROOT / task["bddl"]), camera_heights=128, camera_widths=128)
    rows = []
    try:
        for index, (identity, seed) in enumerate(zip(ids, seeds, strict=True)):
            candidate_xy, stratum, lane_stratum = geometry(index)
            env.seed(seed)
            observation = env.reset()
            for _ in range(10):
                observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            campaign.set_scene_candidates(env, {"candidate_initial_xy_m": candidate_xy, "mass_factor": {"front": 1.0, "back": 1.0}})
            for _ in range(10):
                observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            observation = campaign.forced_observation(env)
            state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
            positions = {slot: campaign.body_position(env.sim, body) for slot, body in campaign.BODY_BY_SLOT.items()}
            localizations = {}
            for slot in ("front", "back"):
                _, _, metric = campaign.localize_candidate(frame, slot, calibration)
                localizations[slot] = {"subpixel_dx": float(metric["subpixel_dx"]), "subpixel_dy": float(metric["subpixel_dy"]), "quality": float(metric["quality"])}
                if localizations[slot]["quality"] < 0.50:
                    raise RuntimeError(f"untrackable joint base {identity} {slot}")
                if lane_margin(original_protocol, slot, positions[slot]) <= 0.002:
                    raise RuntimeError(f"joint base lacks preregistered initial lane margin: {identity} {slot}")
            rows.append({
                "base_identity_id": identity,
                "generator_seed": seed,
                "partition": "EPOCH9E_JOINT_CERTIFICATION",
                "spatial_stratum": stratum,
                "lane_stratum": lane_stratum,
                "probe_order": probe_order(index),
                "instruction_property": instruction_property(index),
                "candidate_mapping": {"candidate_1": "front", "candidate_2": "back"},
                "candidate_initial_xy_command_m": candidate_xy,
                "candidate_initial_xyz_eval_only": {slot: positions[slot].tolist() for slot in ("front", "back")},
                "candidate_initial_lane_margin_m_eval_only": {slot: lane_margin(original_protocol, slot, positions[slot]) for slot in ("front", "back")},
                "candidate_center_separation_m_eval_only": float(np.linalg.norm(positions["front"] - positions["back"])),
                "initial_rgb_localization_audit": localizations,
                "base_state_vector_float64": state.tolist(),
                "base_state_vector_sha256": array_sha256(state),
                "first_agentview_rgb_sha256": rgb_sha256(frame),
                "outcomes_accessed_during_construction": [],
                "mass_assignment_applied_during_construction": False,
                "validation_accessed": False,
                "confirmation_accessed": False,
            })
    finally:
        env.close()
    return rows


def completion_target(heavy_slot: str, property_name: str) -> str:
    if property_name == "heaviest":
        return heavy_slot
    return "back" if heavy_slot == "front" else "front"


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite Epoch 9E joint protocol")
    for path in (SCOPE, ENDPOINT, IDENTITIES, ORIGINAL_CONTROLLER, ORIGINAL_PROTOCOL, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong Epoch 9E branch")
    scope = load(SCOPE)
    endpoint = load(ENDPOINT)
    identities = load(IDENTITIES)
    if scope["paper_status"] != "PAPER_NOT_AUTHORIZED" or scope["sealed_source_demo_identities"]["accessed_by_epoch9e"]:
        raise RuntimeError("sealed-stage contamination")
    if endpoint["joint_certification_endpoint_decision"]["full_trajectory_lane_reachability_gate"] != "48/48 RETAINED":
        raise RuntimeError("endpoint audit did not retain the lane gate")
    allocation = identities["allocations"]["joint_certification_base_pairs"]
    ids = allocation["identity_ids"]
    seeds = allocation["generator_seeds"]
    if len(ids) != 12 or len(seeds) != 12:
        raise RuntimeError("joint identity allocation must contain 12 bases")
    bases = build_base_states(ids, seeds)
    assignments = []
    for base in bases:
        for assignment, factors, heavy in (
            ("A", {"front": 1.0, "back": 8.0}, "back"),
            ("B", {"front": 8.0, "back": 1.0}, "front"),
        ):
            target = completion_target(heavy, base["instruction_property"])
            assignments.append({
                "scene_id": f"epoch9e_joint_base_{base['base_identity_id']}_assignment_{assignment}",
                "base_identity_id": base["base_identity_id"],
                "assignment": assignment,
                "mass_factor": factors,
                "heavy_slot_eval_only": heavy,
                "instruction_property": base["instruction_property"],
                "completion_target_slot_eval_only": target,
                "task_bddl": campaign.TASKS[target]["bddl"],
                "probe_order": base["probe_order"],
            })
    shams = []
    for base in bases[:6]:
        for assignment, factors in (("A", {"front": 1.0, "back": 8.0}), ("B", {"front": 8.0, "back": 1.0})):
            shams.append({
                "sham_id": f"epoch9e_joint_sham_base_{base['base_identity_id']}_assignment_{assignment}",
                "base_identity_id": base["base_identity_id"],
                "assignment": assignment,
                "mass_factor": factors,
                "slot": "back",
                "sham_action": "hold at the high-clearance approach-start pose and execute the same five-step response action sequence without descending into contact",
                "counts_toward_primary_accuracy_or_completion": False,
            })
    orders = Counter(tuple(row["probe_order"]) for row in bases)
    properties = Counter(row["instruction_property"] for row in bases)
    lanes = Counter(row["lane_stratum"] for row in bases)
    controller = load(ORIGINAL_CONTROLLER)
    protocol = {
        "schema_version": "epoch9e.joint_causal_task_certification_protocol.v1",
        "frozen_at": timestamp(),
        "branch": BRANCH,
        "source_checkpoint": git("rev-parse", "HEAD"),
        "evidence_partition": "EPOCH9E_JOINT_CERTIFICATION_ONE_SHOT",
        "authority": {"path": relative(SCOPE), "sha256": sha256(SCOPE)},
        "endpoint_audit": {"path": relative(ENDPOINT), "sha256": sha256(ENDPOINT)},
        "identity_manifest": {"path": relative(IDENTITIES), "sha256": sha256(IDENTITIES)},
        "outcomes_accessed_during_protocol_construction": [],
        "validation_accessed": False,
        "confirmation_accessed": False,
        "one_shot": {"panels": 1, "near_miss_rerun": False, "row_replacement": False, "endpoint_repair_after_outcome": False},
        "base_state_count": 12,
        "assignment_scene_count": 24,
        "candidate_probe_count": 48,
        "base_states": bases,
        "assignments": assignments,
        "balance_audit": {
            "assignments_per_base": 2,
            "heavy_position_counts": {"front": 12, "back": 12},
            "probe_order_base_counts": {"front_first": orders[("front", "back")], "back_first": orders[("back", "front")]},
            "probe_order_scene_counts": {"front_first": 2 * orders[("front", "back")], "back_first": 2 * orders[("back", "front")]},
            "instruction_property_base_counts": dict(properties),
            "instruction_property_scene_counts": {key: 2 * value for key, value in properties.items()},
            "lane_stratum_base_counts": dict(lanes),
            "candidate_identity_mapping": {"candidate_1": "front", "candidate_2": "back"},
            "exact_pair_rule": "A and B load the same float64 state with identical geometry, rendering, robot state, first RGB, probe order, instruction property, and controller; only body mass/inertia assignment differs",
        },
        "controller_contract": {
            "name": "epoch9e_one_shot_nondrag_disengagement",
            "base_controller_config": controller["controller_config"],
            "only_replaced_phase": "pre-response contact_verify_retract",
            "contact_trigger": {"visual_displacement_pixels_at_least": 0.55, "tracker_quality_at_least": 0.50, "ordinary_observations_only": True},
            "inward_orientation": {"source": "initial ordinary RGB localization and frozen lane geometry", "rule": "place paddle on the nearer y-boundary side so the candidate-facing normal points toward lane interior; exact ties use lower side for front and upper side for back", "mirrored_front_back": True, "mass_or_label_input": False},
            "nondrag_lift": {"direction_world_xyz": [0.0, 0.0, 1.0], "raw_action_xyz": [0.0, 0.0, 0.65], "commanded_planar_component": 0.0, "minimum_proprioceptive_lift_m": 0.030, "target_proprioceptive_lift_m": 0.035, "maximum_steps": 12, "fixed_direction": True},
            "separation_trackability_verification": {"eef_lift_m_at_least": 0.030, "rgb_tracker_quality_at_least": 0.50, "persistent_target_displacement_pixels_at_least": 0.55, "hold_steps": 3, "simulator_contact_or_force_input": False},
            "response_reentry": {"only_after_separation_verified": True, "high_clearance_x_reposition_m": -0.008, "descend_to_paddle_z_m": 0.930, "planar_clearance_source": "verified visual-contact approach offset", "response_baseline_hold_steps": 3},
            "response_window": {"unchanged": True, "impulse_action_x": 0.65, "impulse_steps": 3, "coast_steps": 2, "window_steps": 5, "action_sequence_x": [0.65, 0.65, 0.65, 0.0, 0.0]},
            "primary_score": {"unchanged": True, "threshold_m": 0.005219466062047384, "front_score_m": "threshold - back_response", "back_score_m": "back_response - threshold", "smaller_score_predicts_heavy": True},
            "forbidden_action_inputs": ["mass", "property label", "response-score sign", "simulator pose", "force", "reward", "success", "segmentation", "oracle identity"],
        },
        "mechanics_authority": {"smoke_scene_maximum": 8, "mass_assignment": {"front": 1.0, "back": 1.0}, "may_inspect": ["action finiteness", "contact", "collision", "trackability", "continuous lane margin", "swept volume", "kinematics"], "must_not_compute_or_reveal": ["mass rank", "mass-conditioned response", "oracle task success"], "implementation_repairs_maximum": 1, "repair_only_for_unit_testable_outcome_independent_defect": True},
        "sham_control": {"base_state_count": 6, "row_count": 12, "manifest": shams, "required": {"sampled_contact_rows": 0, "collision_rows": 0, "prediction_flips": 0, "paired_mass_contrast_95_interval_includes_zero": True}},
        "paired_test": {"contrast": "back_response_when_back_light_B minus back_response_when_back_heavy_A", "expected_direction": "positive", "primary_one_sided_test": "exact sign test discarding and reporting zeros", "p_strictly_less_than": 0.01, "paired_interval": "two-sided 95% Student-t interval for 12 exact-pair contrasts", "interval_requirement": "lower endpoint > 0"},
        "position_order_control": {"first_rgb_exact_pairs": 12, "initial_localization_exact_pairs": 12, "position_order_only_max_correct": 12, "position_order_only_denominator": 24, "position_order_only_pair_flips": 0, "adjusted_model": "OLS pair contrast on centered initial back x, back y, back lane margin, and back-first indicator with HC3 95% interval", "adjusted_requirement": "positive estimate and lower endpoint > 0"},
        "joint_go": {
            "finite_bounded_actions": {"minimum": 48, "denominator": 48},
            "intended_contact_or_excitation": {"minimum": 46, "denominator": 48},
            "both_candidates_excited": {"minimum": 22, "denominator": 24},
            "full_trajectory_lane_reachable": {"minimum": 48, "denominator": 48},
            "collision_identity_swap_fall_workspace_exit_track_loss": {"maximum": 0},
            "rank_correct": {"minimum": 20, "denominator": 24},
            "rank_correct_each_heavy_position": {"minimum": 10, "denominator": 12},
            "exact_pair_both_assignments_flip_correctly": {"minimum": 9, "denominator": 12},
            "paired_mass_response_test": "paired_test must pass",
            "precontact_position_order_sham_controls": "all controls must pass",
            "completion_oracle": {"minimum": 20, "denominator": 24},
            "completion_oracle_each_heavy_position": {"minimum": 9, "denominator": 12},
            "all_misses_and_continuous_lane_displacement_disclosed": True,
        },
        "success_decision": "EPOCH9E_JOINT_CERTIFICATION_GO",
        "failure_decision": "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED",
        "resource_contract": {"simulator_environments_at_once": 1, "resident_models_at_once": 0, "serial": True, "host_ram_ceiling_percent": 82.0, "wsl_swap_used_peak_bytes": 0, "model_offload": False, "missing_key_only_resume": True},
    }
    if protocol["balance_audit"]["probe_order_base_counts"] != {"front_first": 6, "back_first": 6} or properties != {"heaviest": 6, "lightest": 6}:
        raise RuntimeError("joint panel balance failure")
    atomic_write_json(OUTPUT_JSON, protocol)
    atomic_write_text(OUTPUT_MD, f"""# Epoch 9E one-shot joint certification protocol

This is the only authorized joint panel: 12 fresh exact states, two physical mass swaps per state, 24 primary scenes, 48 probes, and 12 sham rows on six bases. A/B pairs are identical in state, geometry, rendering, first RGB, controller, order, and instruction; only mass/inertia assignment differs.

The controller changes only pre-response `contact_verify_retract`: contact is detected from ordinary RGB/proprioception, contact is broken by a world-vertical action with exactly zero commanded planar component, separation and tracking are verified from ordinary observations, and only then is the frozen response clearance re-entered. The five-step response window, threshold, score, and rank sign remain unchanged. Every action rule is label-blind and mirrored across slots.

`EPOCH9E_JOINT_CERTIFICATION_GO` requires all gates in `{relative(OUTPUT_JSON)}` on the first panel, including lane/reach 48/48, rank 20/24 and 10/12 per heavy stratum, correct exact flips 9/12, one-sided paired p < 0.01, positive paired and adjusted confidence intervals, valid pre-contact/position/sham controls, and completion 20/24 with 9/12 per heavy stratum. There is no near-miss rerun or endpoint repair.
""")
    print(json.dumps({"output": relative(OUTPUT_JSON), "bases": len(bases), "assignments": len(assignments), "shams": len(shams), "min_initial_margin_m": min(value for row in bases for value in row["candidate_initial_lane_margin_m_eval_only"].values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
