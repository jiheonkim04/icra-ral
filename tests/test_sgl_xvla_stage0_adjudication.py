from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_stage0_adjudication import (
    build_sgl_stage0_adjudication,
    validate_sgl_stage0_adjudication,
    write_sgl_stage0_adjudication,
)


def test_stage0_adjudication_is_valid_and_report_only() -> None:
    adjudication = build_sgl_stage0_adjudication()
    execution = adjudication["execution_classification"]

    assert validate_sgl_stage0_adjudication(adjudication) == []
    assert adjudication["decision"] == "SGL_STAGE0_COMPLETE_PROTOCOL_FREEZE_AUTHORIZED_NO_TRAINING_NO_OURS_ROLLOUT"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_stage0_adjudication_records_all_required_checks() -> None:
    adjudication = build_sgl_stage0_adjudication()
    statuses = adjudication["stage0_check_status"]

    assert set(statuses) == {
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    assert statuses["support_observability_no_training"]["status"] == "PASS_LANGUAGE_LEVEL_ONLY"
    assert statuses["support_observability_no_training"]["visual_progress_observability_verified"] is False
    assert statuses["action_bias_bounds_no_optimizer"]["status"] == "PASS_FROZEN_POST_CLAMP_BOUNDS"
    assert statuses["simple_fixed_lift_control_frozen"]["exactly_one_simple_control"] is True
    assert statuses["held_out_identity_manifest_frozen"]["held_out_confirmatory_identity_pool"] == [
        20260734,
        20260735,
        20260736,
        20260737,
    ]


def test_stage0_adjudication_has_no_rollout_or_training_authority() -> None:
    adjudication = build_sgl_stage0_adjudication()
    auth = adjudication["authorization_boundary"]
    conclusion = adjudication["bounded_conclusion"]

    assert auth["stage0_complete"] is True
    assert auth["candidate_can_advance_to_no_training_rollout_protocol_freeze"] is True
    assert auth["training_authorized"] is False
    assert auth["lora_or_qlora_training_authorized"] is False
    assert auth["checkpoint_write_authorized"] is False
    assert auth["control_rollout_authorized"] is False
    assert auth["ours_rollout_authorized"] is False
    assert auth["paper_candidate_go"] is False
    assert auth["prototype_go"] is False
    assert conclusion["candidate_can_train"] is False
    assert conclusion["candidate_can_roll_out_ours"] is False
    assert conclusion["control_can_roll_out_now"] is False


def test_stage0_adjudication_uses_calibrated_comparator_statuses() -> None:
    adjudication = build_sgl_stage0_adjudication()
    statuses = adjudication["comparator_role_statuses"]

    assert statuses["BASE_CLAIM_STATUS"] == "NOT_TESTED_STAGE0_ONLY"
    assert statuses["PRIOR_ADVANCE_STATUS"] == "NOT_TESTED_STAGE0_ONLY"
    assert statuses["SIMPLE_EXPLANATION_STATUS"] == "CONTROL_FROZEN_NOT_TESTED"
    assert statuses["CLEAN_RETENTION_STATUS"] == "REQUIRED_NOT_TESTED"
    assert statuses["GENERALIZATION_STATUS"] == "HELD_OUT_MANIFEST_FROZEN_NOT_TESTED"
    assert statuses["OVERALL_PAPER_CANDIDATE_STATUS"] == "PRIOR_ADVANCE_NOT_ESTABLISHED"
    assert "Stage0-only" in statuses["status_scope"]


def test_stage0_adjudication_preserves_identity_roles() -> None:
    adjudication = build_sgl_stage0_adjudication()
    target = adjudication["target"]

    assert target["residual_identities"] == [20260727, 20260730, 20260733]
    assert target["clean_retention_identities"] == [20260731, 20260732]
    assert target["held_out_confirmatory_identity_pool"] == [20260734, 20260735, 20260736, 20260737]


def test_stage0_adjudication_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "stage0_adjudication.json"
    adjudication = write_sgl_stage0_adjudication(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert adjudication["candidate_id"] == "SGL-XVLA"
    assert '"stage0_complete_for_protocol_freeze": true' in text
    assert '"candidate_can_train": false' in text
    assert '"ours_rollout_authorized": false' in text
