#!/usr/bin/env python3
"""Freeze the final planar-push headroom attempt for attribution rotation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


PARENT = ROOT / "reports/epoch8_latent_dynamics_feedback_development.json"
EXPERT = ROOT / "reports/epoch7_latent_dynamics_attribution/expert_feasibility.json"
OUTPUT = ROOT / "reports/epoch9c_attribution_feedback_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    expert = json.loads(EXPERT.read_text(encoding="utf-8"))
    task = next(row for row in expert["tasks"] if int(row["eval_id"]) == 2)
    protocol = {
        "schema_version": "epoch9c.attribution_feedback_protocol.v1",
        "status": "FROZEN_BEFORE_POSE_ADAPTIVE_PUSH_OUTCOMES",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "decisive third-family feasibility gate for semantic-versus-physical causal attribution after active-property closure",
        "active_property_terminal_input": "ACTIVE_PROPERTY_THESIS_EMPIRICALLY_INFEASIBLE_ROTATING",
        "parent_zero_rotation_result": {
            "path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(PARENT),
            "decision": "FEEDBACK_EXPERT_DEVELOPMENT_FAIL_ROTATE",
            "scope": "zero-rotation Cartesian sweep only",
        },
        "parent_expert_feasibility": {
            "path": str(EXPERT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(EXPERT),
            "already_eligible_families": ["articulated_drawer", "pick_transport_place"],
        },
        "task": {
            "eval_id": 2,
            "family": "planar_push",
            "goal_bddl": "push_the_plate_to_the_front_of_the_stove.bddl",
            "target_body": "plate_1_main",
            "goal_site": "main_table_stove_front_region",
            "selected_standard_demo": task["selected_demo"],
            "hdf5_path": task["hdf5_path"],
            "intervention": {
                "axis": "target_contact_friction",
                "body_name": "plate_1_main",
                "array": "geom_friction",
                "components": [0, 1, 2],
                "collision_geoms_only": True,
                "factor": 0.25,
            },
        },
        "controller": {
            "phase_1": "replay the frozen selected standard demonstration prefix only until first target contact",
            "phase_2": "recompute the target-to-goal XY unit vector from privileged feasibility geometry and issue a fixed low-gain delta translation",
            "feedback_translation_gain": 0.08,
            "feedback_rotation_action": [0.0, 0.0, 0.0],
            "gripper_action": -1.0,
            "maximum_prefix_steps": 70,
            "maximum_feedback_steps": 200,
            "action_bounds": [-1.0, 1.0],
            "success_stopping_only": True,
            "intervention_label_used_for_actions": False,
            "demonstration_wrist_pose_preserved_by_zero_delta_rotation_after_contact": True,
        },
        "role_and_information_boundary": {
            "execution_type": "PRIVILEGED_FEASIBILITY_ORACLE_NO_POLICY",
            "legal": ["standard demo prefix", "current target position", "fixed goal-site position", "official success as stopping predicate"],
            "forbidden": ["intervention label for action selection", "friction values", "reward shaping", "future states", "VLA features or actions"],
            "expert_counted_as_policy_success": False,
        },
        "paired_conditions": ["standard", "latent_dynamics_intervention"],
        "gate": {
            "episodes": 2,
            "finite_bounded_actions": "2/2",
            "paired_initial_state_exact": True,
            "first_observation_exact": True,
            "target_contact": "2/2",
            "official_success": "2/2",
            "pass": "ATTRIBUTION_THIRD_FAMILY_HEADROOM_RESTORED",
            "fail": "NO_DEFENSIBLE_LOCAL_PATH_AFTER_EMPIRICAL_ROTATIONS",
        },
        "sealed_validation_or_confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"status": protocol["status"], "gain": protocol["controller"]["feedback_translation_gain"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
