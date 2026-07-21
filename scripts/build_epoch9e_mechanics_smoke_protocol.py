#!/usr/bin/env python3
"""Freeze eight label-blind mechanics-only smoke states for Epoch 9E."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.build_epoch9e_joint_certification_protocol import geometry, lane_margin
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from tca_map.epoch7_latent_dynamics import atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "epoch9e_mechanics_smoke_protocol.json"
IDENTITIES = REPORTS / "epoch9e_fresh_identity_manifest.json"
JOINT = REPORTS / "epoch9e_joint_certification_protocol.json"
ORIGINAL_PROTOCOL = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
CALIBRATION = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"


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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite mechanics smoke protocol")
    identities = load(IDENTITIES)
    joint = load(JOINT)
    original = load(ORIGINAL_PROTOCOL)
    calibration = load(CALIBRATION)
    allocation = identities["allocations"]["mechanics_smoke"]
    env_class = campaign.load_env_class()
    task = campaign.TASKS["front"]
    env = env_class(bddl_file_name=str(BDDL_ROOT / task["bddl"]), camera_heights=128, camera_widths=128)
    rows = []
    try:
        for index, (identity, seed) in enumerate(zip(allocation["identity_ids"], allocation["generator_seeds"], strict=True)):
            candidate_xy, stratum, lane_stratum = geometry(index % 6)
            if index == 6:
                candidate_xy, stratum, lane_stratum = {"front": [0.114, 0.170], "back": [-0.150, 0.020]}, "dual_opposed_edge", "dual_edge"
            elif index == 7:
                candidate_xy, stratum, lane_stratum = {"front": [0.094, 0.120], "back": [-0.130, 0.085]}, "dual_inverse_edge", "dual_edge"
            env.seed(seed)
            observation = env.reset()
            for _ in range(10): observation, _, _, _ = env.step(np.asarray([0,0,0,0,0,0,1], dtype=np.float32))
            campaign.set_scene_candidates(env, {"candidate_initial_xy_m": candidate_xy, "mass_factor": {"front": 1.0, "back": 1.0}})
            for _ in range(10): observation, _, _, _ = env.step(np.asarray([0,0,0,0,0,0,1], dtype=np.float32))
            observation = campaign.forced_observation(env)
            state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
            positions = {slot: campaign.body_position(env.sim, body) for slot, body in campaign.BODY_BY_SLOT.items()}
            localization = {}
            for slot in ("front", "back"):
                _, _, metric = campaign.localize_candidate(frame, slot, calibration)
                localization[slot] = {"subpixel_dx": float(metric["subpixel_dx"]), "subpixel_dy": float(metric["subpixel_dy"]), "quality": float(metric["quality"])}
                if localization[slot]["quality"] < 0.50 or lane_margin(original, slot, positions[slot]) <= 0.002:
                    raise RuntimeError(f"invalid smoke base {identity} {slot}")
            rows.append({"scene_id": f"epoch9e_mechanics_smoke_{identity}", "generated_identity_id": identity, "generator_seed": seed, "partition": "EPOCH9E_MECHANICS_ONLY", "spatial_stratum": stratum, "lane_stratum": lane_stratum, "probe_order": ["front", "back"] if index % 2 == 0 else ["back", "front"], "candidate_initial_xy_command_m": candidate_xy, "candidate_initial_xyz_eval_only": {slot: positions[slot].tolist() for slot in ("front", "back")}, "candidate_initial_lane_margin_m_eval_only": {slot: lane_margin(original, slot, positions[slot]) for slot in ("front", "back")}, "initial_rgb_localization_audit": localization, "base_state_vector_float64": state.tolist(), "base_state_vector_sha256": array_sha256(state), "first_agentview_rgb_sha256": rgb_sha256(frame), "mass_factor": {"front": 1.0, "back": 1.0}, "outcomes_accessed_during_construction": [], "mass_rank_authorized": False, "oracle_success_authorized": False})
    finally:
        env.close()
    protocol = {"schema_version": "epoch9e.mechanics_smoke_protocol.v1", "frozen_at": timestamp(), "joint_protocol_path": "reports/epoch9e_joint_certification_protocol.json", "joint_protocol_sha256": sha256(JOINT), "controller_contract": joint["controller_contract"], "scene_count": 8, "candidate_probe_count": 16, "manifest": rows, "mass_assignment": {"front": 1.0, "back": 1.0}, "may_inspect": joint["mechanics_authority"]["may_inspect"], "must_not_compute_or_reveal": joint["mechanics_authority"]["must_not_compute_or_reveal"], "mechanics_pass": {"complete_scenes": 8, "finite_bounded_actions": 16, "intended_contact_or_excitation_minimum": 15, "both_candidates_excited_scenes_minimum": 7, "full_trajectory_lane_reachable": 16, "zero_collision_identity_swap_fall_workspace_exit_track_loss": True, "all_liftoff_planar_commands_exact_zero": True, "all_disengagement_attempts_separation_verified": True}, "implementation_repair": {"maximum": 1, "authorized_only_for": "unit-testable outcome-independent execution defect", "rank_lane_or_task_outcome_repair": False}, "scientific_outcomes_accessed": False, "validation_accessed": False, "confirmation_accessed": False}
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"scenes": len(rows), "minimum_initial_margin_m": min(value for row in rows for value in row["candidate_initial_lane_margin_m_eval_only"].values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
