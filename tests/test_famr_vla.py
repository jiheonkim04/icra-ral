import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tca_map.smolvla.famr_vla import (
    COEFFICIENT_STEPS,
    PROPOSAL_HASH,
    TARGET_TASK_IDENTITIES,
    action_scales,
    action_validity,
    assign_parameter_groups,
    build_response_matrix,
    canonical_task_identity,
    classify_stage0,
    episode_partitions,
    missing_manifest_rows,
    objective_and_gradients,
    ordering_agreement,
    practical_action_threshold,
    predict_linear_actions,
    resource_overlap,
    response_fidelity,
    scale_lora_b,
    solve_coefficients,
    task_identity_audit,
    validate_episode_partitions,
    validate_partial_payload,
    validate_result_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parameter_names() -> list[str]:
    names = []
    for layer in range(16):
        for projection in ("q_proj", "v_proj"):
            for side in ("lora_A", "lora_B"):
                names.append(
                    f"base.model.vlm_with_expert.lm_expert.model.text_model.layers.{layer}."
                    f"self_attn.{projection}.{side}.default.weight"
                )
    for module in (
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ):
        for side in ("lora_A", "lora_B"):
            names.append(f"base.model.{module}.{side}.default.weight")
    return names


def test_task_identity_and_episode_partitions_are_exact_and_disjoint() -> None:
    assert canonical_task_identity(TARGET_TASK_IDENTITIES[0]) == "put_the_frying_pan_under_the_cabinet_shelf"
    audit = task_identity_audit(
        ["pick up the black bowl and put it on the plate", "turn on the stove"],
        TARGET_TASK_IDENTITIES,
    )
    assert audit["intersection_count"] == 0
    partitions = episode_partitions()
    summary = validate_episode_partitions(partitions, range(50))
    assert summary["counts"] == {"train": 35, "validation": 10, "test": 5}
    assert all(not overlap for overlap in summary["pairwise_overlap"].values())


def test_parameter_assignment_is_exhaustive_disjoint_and_rejects_unknowns() -> None:
    names = _parameter_names()
    groups = assign_parameter_groups(names)
    assert set(groups["coarse"]) == set(names)
    assert set(groups["fine"]) == set(names)
    assert set(groups["coarse"].values()) == {"vlm_expert", "action_flow", "state_projection"}
    assert set(groups["fine"].values()) == {
        "vlm_layers_0_7",
        "vlm_layers_8_15",
        "action_input_output",
        "action_time_mlp",
        "state_projection",
    }
    with pytest.raises(ValueError, match="outside frozen"):
        assign_parameter_groups(names + ["base.model.unexpected.lora_A.default.weight"])


def test_response_objective_and_projected_solver_are_deterministic_and_bounded() -> None:
    base = np.zeros((6, 7), dtype=np.float64)
    group_actions = np.zeros((6, 2, 7), dtype=np.float64)
    group_actions[:, 0, 0] = 1.0
    group_actions[:, 1, 1] = 1.0
    responses = build_response_matrix(base, group_actions)
    target = np.zeros_like(base)
    target[:, 0] = 0.8
    target[:, 1] = 0.2
    retention = np.zeros((4, 7, 2), dtype=np.float64)
    retention[:, 0, 0] = 0.2
    retention[:, 1, 1] = 0.1
    scales = np.ones(7)

    initial = objective_and_gradients(np.array([0.5, 0.5]), base, responses, target, retention, scales, 1.0)
    first = solve_coefficients(base, responses, target, retention, scales, 1.0)
    second = solve_coefficients(base, responses, target, retention, scales, 1.0)
    coefficients = first["coefficients"]
    assert first["steps"] == COEFFICIENT_STEPS
    assert np.array_equal(coefficients, second["coefficients"])
    assert np.all((0.0 <= coefficients) & (coefficients <= 1.0))
    assert first["final"]["total_loss"] < initial["total_loss"]
    assert coefficients[0] > coefficients[1]


def test_action_scale_and_lora_b_scaling_endpoints() -> None:
    actions = np.zeros((8, 7))
    actions[:, 0] = np.arange(8)
    scales = action_scales(actions)
    assert scales[0] > 0.05
    assert np.all(scales[1:] == 0.05)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    full = b @ a
    assert torch.equal(scale_lora_b(b, 0.0) @ a, torch.zeros_like(full))
    assert torch.equal(scale_lora_b(b, 1.0) @ a, full)
    assert torch.equal(scale_lora_b(b, 0.25) @ a, 0.25 * full)


def test_response_fidelity_ordering_and_practical_threshold() -> None:
    base = np.zeros((4, 7))
    direct = np.zeros((4, 7))
    direct[:, 0] = [0.1, 0.2, 0.3, 0.4]
    linear = direct.copy()
    linear[:, 0] += 0.01
    metrics = response_fidelity(base, linear, direct, np.full(7, 0.1))
    assert metrics["normalized_rmse"] < 0.05
    assert metrics["norm_correlation"] == pytest.approx(1.0)
    assert ordering_agreement([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0
    assert practical_action_threshold([0.0, 2e-4]) > 1e-4


def test_action_validity_uses_absolute_and_base_relative_gates() -> None:
    base = np.zeros((20, 7))
    healthy = np.full((20, 7), 0.2)
    assert action_validity(healthy, base)["passed"] is True
    destructive = healthy.copy()
    destructive[:, 0] = 1.3
    result = action_validity(destructive, base)
    assert result["passed"] is False
    assert result["candidate_abs_max"] > result["absolute_limit"]


def _row(policy: str, reset: int, **extra: object) -> dict[str, object]:
    return {
        "policy": policy,
        "suite": "target",
        "task": "task_a",
        "reset_identity": reset,
        **extra,
    }


def test_manifest_resume_and_partial_validation_reject_corruption() -> None:
    manifest = [_row("base", 1), _row("ours", 1), _row("base", 2)]
    completed = [_row("base", 1), _row("base", 2)]
    assert missing_manifest_rows(manifest, completed) == [_row("ours", 1)]
    summary = validate_result_manifest(manifest, completed)
    assert summary["missing_keys"] == [("ours", "target", "task_a", 1)]
    assert summary["passed"] is False
    duplicate = validate_result_manifest(manifest, completed + [_row("base", 1)])
    assert duplicate["duplicate_result_key_count"] == 1
    assert validate_partial_payload('{"completed_count": 2, "planned_count": 3, "exception_count": 0}')[
        "parsed"
    ]
    with pytest.raises(ValueError, match="valid JSON"):
        validate_partial_payload("{")


def test_resource_overlap_quarantines_timing_evidence() -> None:
    clean = resource_overlap(100.0, 200.0, [{"started_unix": 10.0, "finished_unix": 20.0}])
    assert clean["timing_resource_evidence_eligible"] is True
    overlap = resource_overlap(
        100.0,
        200.0,
        [{"started_unix": 150.0, "finished_unix": 250.0, "kind": "windows_efficiency_mode"}],
    )
    assert overlap["timing_resource_evidence_eligible"] is False
    assert overlap["overlap_count"] == 1


def _healthy_summary(**overrides: object) -> dict[str, object]:
    summary = {
        "essential_source_unavailable": False,
        "target_overlap_count": 0,
        "preflight_passed": True,
        "data_semantics_passed": True,
        "split_integrity_passed": True,
        "identity_passed": True,
        "target_modules_passed": True,
        "gradient_health_passed": True,
        "checkpoint_reload_passed": True,
        "group_assignment_passed": True,
        "scaling_identity_passed": True,
        "base_unchanged": True,
        "memory_passed": True,
        "confirmatory_sealed": True,
        "subset_fit_passed": True,
        "capacity_check_used": False,
    }
    summary.update(overrides)
    return summary


def test_stage0_classification_preserves_false_negative_guard() -> None:
    assert classify_stage0(_healthy_summary()) == "FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED"
    assert (
        classify_stage0(_healthy_summary(subset_fit_passed=False))
        == "FAMR_UNDERPOWERED_ONE_CHECK_ALLOWED"
    )
    assert (
        classify_stage0(_healthy_summary(subset_fit_passed=False, capacity_check_used=True))
        == "FAMR_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT"
    )
    assert (
        classify_stage0(_healthy_summary(data_semantics_passed=False))
        == "FAMR_IMPLEMENTATION_OR_DATA_FAILURE"
    )
    assert classify_stage0(_healthy_summary(target_overlap_count=1)) == "FAMR_FATAL_PREIMPLEMENTATION"


def test_frozen_artifact_contract_matches_proposal_and_seals_test() -> None:
    assert (REPO_ROOT / "reports" / "famr_vla" / "proposal_hash.txt").read_text().strip() == PROPOSAL_HASH
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))
    contract = state["epoch_4_cycle_17_famr_pre_stage_0a"]
    assert contract["target_task_count"] == 3
    assert contract["stage_0a_micro_fit_steps"] == 20
    assert contract["confirmatory_observations_decoded_max"] == 0
    assert contract["confirmatory_actions_computed_max"] == 0
