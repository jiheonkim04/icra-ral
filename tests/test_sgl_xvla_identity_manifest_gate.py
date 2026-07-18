from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_identity_manifest_gate import (
    build_sgl_identity_manifest_gate,
    initial_state_index,
    validate_sgl_identity_manifest_gate,
    write_sgl_identity_manifest_gate,
)


def test_identity_manifest_gate_is_valid_and_report_only() -> None:
    gate = build_sgl_identity_manifest_gate()
    execution = gate["execution_classification"]

    assert validate_sgl_identity_manifest_gate(gate) == []
    assert gate["decision"] == "SGL_HELDOUT_IDENTITY_MANIFEST_FROZEN_NO_TRAINING_NO_OURS"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_identity_manifest_freezes_disjoint_role_sets() -> None:
    gate = build_sgl_identity_manifest_gate()
    roles = gate["identity_roles"]

    assert [item["reset_identity"] for item in roles["development_residual_identities"]] == [
        20260727,
        20260730,
        20260733,
    ]
    assert [item["reset_identity"] for item in roles["clean_retention_identities"]] == [20260731, 20260732]
    assert [item["reset_identity"] for item in roles["held_out_confirmatory_identities"]] == [
        20260734,
        20260735,
        20260736,
        20260737,
    ]
    assert roles["role_sets_disjoint"] is True
    assert roles["role_lists_individually_sorted"] is True
    assert roles["all_frozen_identities_sorted"] == [
        20260727,
        20260730,
        20260731,
        20260732,
        20260733,
        20260734,
        20260735,
        20260736,
        20260737,
    ]


def test_identity_manifest_records_initial_state_mapping() -> None:
    gate = build_sgl_identity_manifest_gate()
    all_records = (
        gate["identity_roles"]["development_residual_identities"]
        + gate["identity_roles"]["clean_retention_identities"]
        + gate["identity_roles"]["held_out_confirmatory_identities"]
    )

    assert gate["target"]["identity_mapping_rule"] == "initial_state_index = reset_identity - 20260711 for task5"
    for record in all_records:
        assert record["initial_state_index"] == initial_state_index(record["reset_identity"])


def test_identity_manifest_forbids_cherry_picking_and_test_set_tuning() -> None:
    gate = build_sgl_identity_manifest_gate()
    policy = gate["role_usage_policy"]
    rules = " ".join(gate["anti_cherry_pick_rules"])

    assert "checkpoint selection" in policy["held_out_confirmatory"]["forbidden_use"]
    assert "dropping identities after failures" in policy["clean_retention"]["forbidden_use"]
    assert "Do not add, drop, reorder, or relabel identities" in rules
    assert "Do not inspect held-out outcomes" in rules
    assert gate["identity_roles"]["frozen_before_any_sgl_ours_result"] is True
    assert gate["identity_roles"]["frozen_before_any_simple_control_rollout"] is True


def test_identity_manifest_records_all_stage0_checks_without_rollout_authority() -> None:
    gate = build_sgl_identity_manifest_gate()
    conclusion = gate["bounded_conclusion"]

    assert set(conclusion["stage0_required_checks_now_frozen"]) == {
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    assert conclusion["candidate_can_advance_to_stage0_completion_adjudication"] is True
    assert conclusion["candidate_can_train"] is False
    assert conclusion["candidate_can_roll_out_ours"] is False
    assert conclusion["control_can_roll_out_now"] is False
    assert gate["comparator_role_calibration"]["universal_beat_all_rule_applied"] is False


def test_identity_manifest_gate_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "identity_manifest.json"
    gate = write_sgl_identity_manifest_gate(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert gate["candidate_id"] == "SGL-XVLA"
    assert '"identity_manifest_frozen": true' in text
    assert '"candidate_can_train": false' in text
    assert '"control_can_roll_out_now": false' in text
