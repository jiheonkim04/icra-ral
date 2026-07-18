from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_runner_preflight import (
    build_sgl_runner_preflight,
    templates_equivalent,
    validate_sgl_runner_preflight,
    write_sgl_runner_preflight,
)


def test_runner_preflight_is_valid_and_report_only() -> None:
    preflight = build_sgl_runner_preflight()
    execution = preflight["execution_classification"]

    assert validate_sgl_runner_preflight(preflight) == []
    assert preflight["decision"] == "SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_runner_preflight_detects_static_simple_control_equivalence() -> None:
    preflight = build_sgl_runner_preflight()
    equivalence = preflight["static_behavior_equivalence"]

    assert templates_equivalent(equivalence["sgl_frozen_template"], equivalence["fixed_control_template"]) is True
    assert equivalence["templates_equivalent_under_current_frozen_stage0"] is True
    assert "visual/progress detector remains unverified" in equivalence["missing_distinct_sgl_component_reasons"]
    assert "fixed lift/regrasp schedule equals the primary simple-control schedule" in equivalence[
        "missing_distinct_sgl_component_reasons"
    ]


def test_runner_preflight_blocks_rollout_and_training() -> None:
    preflight = build_sgl_runner_preflight()
    auth = preflight["authorization_boundary"]
    conclusion = preflight["bounded_conclusion"]

    assert auth["primary_sgl_xvla_current_frozen_executable_blocked"] is True
    assert auth["simulator_episode_authorized"] is False
    assert auth["control_rollout_authorized"] is False
    assert auth["ours_rollout_authorized"] is False
    assert auth["training_authorized"] is False
    assert auth["checkpoint_write_authorized"] is False
    assert conclusion["preflight_passed_for_rollout"] is False
    assert conclusion["blocked_before_simulator_episode"] is True
    assert conclusion["current_sgl_candidate_killed"] is True
    assert conclusion["candidate_can_roll_out_ours"] is False


def test_runner_preflight_uses_simple_control_comparator_status() -> None:
    preflight = build_sgl_runner_preflight()
    statuses = preflight["comparator_role_statuses"]

    assert statuses["SIMPLE_EXPLANATION_STATUS"] == "SIMPLE_CONTROL_EXPLAINS_GAIN"
    assert statuses["ABLATION_COMPONENT_STATUS"] == "KEY_COMPONENT_NOT_SUPPORTED"
    assert statuses["OVERALL_PAPER_CANDIDATE_STATUS"] == "SIMPLE_CONTROL_EXPLAINS_GAIN"
    assert "Static preflight block only" in statuses["status_scope"]


def test_runner_preflight_records_action_binding_evidence_without_outcome_tuning() -> None:
    preflight = build_sgl_runner_preflight()
    binding = preflight["action_binding_preflight"]

    assert binding["lift_axis_dimension_index"] == 2
    assert binding["gripper_dimension_index"] == 6
    assert binding["binding_source_semantics_required_before_any_future_runner"] is True
    assert binding["outcome_tuned_binding_allowed"] is False
    assert len(binding["binding_evidence"]) >= 3


def test_runner_preflight_advances_to_backup_candidate_stage0() -> None:
    preflight = build_sgl_runner_preflight()

    assert preflight["authorization_boundary"]["backup_candidate_stage0_authorized_next"] is True
    assert preflight["bounded_conclusion"]["backup_ocr_xvla_can_start_stage0"] is True
    assert "Start Stage 0 report-only gating for the backup OCR-XVLA candidate" in preflight["next_action"]


def test_runner_preflight_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "runner_preflight.json"
    preflight = write_sgl_runner_preflight(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert preflight["candidate_id"] == "SGL-XVLA"
    assert '"blocked_before_simulator_episode": true' in text
    assert '"current_sgl_candidate_killed": true' in text
    assert '"backup_ocr_xvla_can_start_stage0": true' in text
