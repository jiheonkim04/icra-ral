from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_observability_audit import (
    build_sgl_observability_audit,
    validate_sgl_observability_audit,
    write_sgl_observability_audit,
)


def test_observability_audit_is_valid_and_report_only() -> None:
    audit = build_sgl_observability_audit()
    execution = audit["execution_classification"]

    assert validate_sgl_observability_audit(audit) == []
    assert audit["decision"] == "SGL_STAGE0_SUPPORT_OBSERVABILITY_LANGUAGE_PASS_VISUAL_PROGRESS_UNVERIFIED_NO_TRAINING"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["visual_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_observability_audit_support_gate_passes_only_from_language() -> None:
    audit = build_sgl_observability_audit()
    obs = audit["observability"]

    assert obs["language_support_observable"] is True
    assert obs["allowed_activation_sources"] == ["language instruction"]
    assert obs["forbidden_sources_consulted"] == []
    assert obs["privileged_state_consulted"] is False
    assert obs["reward_or_success_consulted"] is False
    assert obs["reset_identity_consulted_for_activation"] is False
    assert obs["visual_support_detector_trained_or_run"] is False


def test_observability_audit_does_not_claim_progress_detector() -> None:
    audit = build_sgl_observability_audit()

    assert audit["observability"]["visual_progress_observability_verified"] is False
    assert audit["bounded_conclusion"]["support_gate_observability_passed"] is True
    assert audit["bounded_conclusion"]["progress_detector_observability_passed"] is False
    assert audit["bounded_conclusion"]["candidate_can_advance_to_action_bias_bounds_gate"] is True
    assert audit["bounded_conclusion"]["candidate_can_train"] is False
    assert audit["bounded_conclusion"]["candidate_can_roll_out_ours"] is False


def test_observability_audit_keeps_clean_retention_identities_required() -> None:
    audit = build_sgl_observability_audit()
    retention = audit["clean_retention_implication"]

    assert retention["support_gate_would_activate_on_clean_retention_identities"] is True
    assert retention["clean_retention_identities_required_in_next_gate"] == [20260731, 20260732]


def test_observability_audit_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "observability.json"
    audit = write_sgl_observability_audit(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert audit["candidate_id"] == "SGL-XVLA"
    assert '"candidate_can_train": false' in text
    assert '"visual_progress_observability_verified": false' in text
