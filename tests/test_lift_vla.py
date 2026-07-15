from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from tca_map.datasets.lift_counterfactual_manifest import (
    PARTITIONS,
    build_counterfactual_manifest,
)
from tca_map.smolvla.lift_vla import CAG, LAST_STEP_ABLATION, LIFT, sample_flow_variant


def _affine_field(branch, latent, timestep):
    offset = 0.4 if branch == "conditioned" else -0.2
    curvature = 0.08 if branch == "conditioned" else -0.03
    return 0.15 * latent + curvature * latent.square() + offset + 0.01 * timestep[:, None, None]


def test_lift_omega_one_is_base_identity():
    noise = torch.linspace(-1.0, 1.0, 64).reshape(1, 2, 32)
    base = sample_flow_variant(_affine_field, noise, variant="base", omega=1.0)
    lift = sample_flow_variant(_affine_field, noise, variant=LIFT, omega=1.0)

    assert torch.equal(base.native, lift.native)
    assert base.field_evaluations["total"] == 10
    assert lift.field_evaluations == {"conditioned": 10, "empty": 10, "total": 20}


def test_cag_and_pathwise_lift_follow_different_frozen_equations():
    noise = torch.zeros((1, 2, 32), dtype=torch.float32)
    cag = sample_flow_variant(_affine_field, noise, variant=CAG, omega=1.5)
    lift = sample_flow_variant(_affine_field, noise, variant=LIFT, omega=1.5)

    assert cag.native.shape == (1, 2, 32)
    assert not torch.allclose(cag.native, lift.native)
    assert cag.field_evaluations["total"] == 20
    assert lift.field_evaluations["total"] == 20


def test_last_step_ablation_is_compute_matched_but_not_full_lift():
    noise = torch.ones((1, 3, 32), dtype=torch.float32)
    ablation = sample_flow_variant(_affine_field, noise, variant=LAST_STEP_ABLATION, omega=2.0)
    lift = sample_flow_variant(_affine_field, noise, variant=LIFT, omega=2.0)

    assert ablation.field_evaluations == {"conditioned": 10, "empty": 10, "total": 20}
    assert not torch.allclose(ablation.native, lift.native)


def _task(index: int) -> dict:
    partition = next(name for name, indices in PARTITIONS.items() if index in indices)
    del partition
    goal = [["on", f"object_{index}", "shared_receptacle"]]
    return {
        "sorted_index": index,
        "task_id": f"task_{index}",
        "bddl_path": str(Path(f"task_{index}.bddl")),
        "bddl_sha256": f"bddl-{index}",
        "language": f"put object {index} on the shared receptacle",
        "problem_name": "LIBERO_Tabletop_Manipulation",
        "objects": [*(f"object_{item}" for item in range(10)), "shared_receptacle"],
        "fixtures": [],
        "regions": [],
        "obj_of_interest": [f"object_{index}"],
        "goal_state": goal,
        "initial_predicates": [["on", f"object_{item}", "table"] for item in range(10)],
        "scene_sha256": "same-scene",
        "parsed": {},
    }


def test_manifest_preserves_frozen_partitions_and_zero_overlap():
    tasks = [_task(index) for index in range(10)]

    def load_state(source, state_index):
        return np.asarray([source["sorted_index"], state_index, 0.25], dtype=np.float64)

    manifest = build_counterfactual_manifest(
        tasks,
        load_state,
        dynamic_validator=lambda row, state: {
            "valid": True,
            "target_scorer_instantiated": True,
            "source_state_shape_compatible": np.asarray(state).shape == (3,),
            "errors": [],
        },
    )

    assert manifest["ready_for_stage_0_model_load"] is True
    assert manifest["row_count"] == 20
    assert manifest["development_scoreable_episode_count"] == 14
    assert manifest["target_task_counts"] == {"discovery": 4, "validation": 3, "confirmatory": 3}
    assert manifest["partition_overlaps"] == []
    assert manifest["confirmatory_policy_observations_decoded"] == 0
    assert manifest["confirmatory_policy_actions_computed"] == 0
    for row in manifest["rows"]:
        assert row["source_sorted_index"] in PARTITIONS[row["evidence_partition"]]
        assert row["target_sorted_index"] in PARTITIONS[row["evidence_partition"]]
        assert row["source_goal_state"] != row["target_goal_state"]


def test_persisted_stage0_result_preserves_the_frozen_stop():
    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "reports" / "lift_vla"
    result = json.loads((report_dir / "stage_0_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((report_dir / "counterfactual_manifest.json").read_text(encoding="utf-8"))
    thresholds = json.loads((report_dir / "discovery_thresholds.json").read_text(encoding="utf-8"))

    assert result["final_decision"] == "LIFT_COMPUTE_INFEASIBLE"
    assert result["smoke"]["shape_gate_passed"] is True
    assert result["smoke"]["identity_gate_passed"] is True
    assert result["smoke"]["activation_gate_passed"] is True
    assert result["smoke"]["compute_gate_passed"] is True
    assert result["smoke"]["action_gate_passed"] is False
    assert result["smoke"]["action_validity"]["range_valid_fraction"] == 0.8023809523809524
    assert "headroom" not in result
    assert result["validation_search_happened"] is False
    assert result["confirmatory_policy_observations_decoded"] == 0
    assert result["confirmatory_policy_actions_computed"] == 0
    assert manifest["ready_for_stage_0_model_load"] is True
    assert manifest["valid_row_count"] == 20
    assert manifest["partition_overlaps"] == []
    assert thresholds["validation_data_used"] is False
    assert thresholds["confirmatory_data_used"] is False
