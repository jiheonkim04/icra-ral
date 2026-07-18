from pathlib import Path

from tca_map.xvla_spatial_task5.ocr_observability_audit import (
    build_ocr_observability_audit,
    validate_ocr_observability_audit,
    write_ocr_observability_audit,
)


def test_ocr_observability_audit_is_valid_and_report_only() -> None:
    audit = build_ocr_observability_audit()
    execution = audit["execution_classification"]

    assert validate_ocr_observability_audit(audit) == []
    assert audit["decision"] == "OCR_TRIGGER_OBSERVABILITY_BLOCKED_NO_ALLOWED_PROGRESS_TRACE_NO_ROLLOUT"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_ocr_observability_audit_records_missing_allowed_trace_fields() -> None:
    audit = build_ocr_observability_audit()
    trigger = audit["trigger_observability"]

    assert trigger["trigger_observable_from_existing_artifacts"] is False
    assert trigger["deterministic_trigger_can_be_frozen_now"] is False
    assert set(trigger["missing_required_trace_fields"]) == {
        "per-step RGB or video frames",
        "per-step proprio/eef/gripper trace",
        "per-step executed or proposed action history",
        "a frozen observation-only object-separation/progress signal",
        "timestamps or step indices linking observations to first grasp/lift attempt",
    }


def test_ocr_observability_audit_inventory_has_only_summaries() -> None:
    audit = build_ocr_observability_audit()
    inventory = audit["existing_artifact_inventory"]

    assert inventory["video_search_result"]["video_or_image_files_found"] == []
    assert len(inventory["xvla_trace_artifacts"]) == 2
    for artifact in inventory["xvla_trace_artifacts"]:
        assert artifact["has_per_step_rgb_or_video"] is False
        assert artifact["has_per_step_proprio"] is False
        assert artifact["has_per_step_action_history"] is False
        assert artifact["has_object_separation_signal_from_allowed_observation"] is False
        assert "action_chunk_ranges" in artifact["available_episode_fields"]


def test_ocr_observability_audit_does_not_use_forbidden_proxy_fields() -> None:
    audit = build_ocr_observability_audit()
    forbidden = set(audit["forbidden_proxy_fields_not_used"])

    for field in ["success", "final_reward", "done", "reset_identity", "initial_state_index"]:
        assert field in forbidden
    assert audit["comparator_role_statuses"]["OVERALL_PAPER_CANDIDATE_STATUS"] == (
        "IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE"
    )


def test_ocr_observability_audit_exhausts_task5_candidates_without_rollout() -> None:
    audit = build_ocr_observability_audit()
    conclusion = audit["bounded_conclusion"]

    assert conclusion["trigger_observability_passed"] is False
    assert conclusion["ocr_candidate_can_advance_to_action_bounds"] is False
    assert conclusion["ocr_candidate_can_train"] is False
    assert conclusion["ocr_candidate_can_roll_out_ours"] is False
    assert conclusion["ocr_candidate_blocked_from_existing_artifacts"] is True
    assert conclusion["task5_candidate_set_exhausted"] is True
    assert "Do not run OCR-XVLA" in audit["next_action"]


def test_ocr_observability_audit_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "ocr_observability.json"
    audit = write_ocr_observability_audit(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert audit["candidate_id"] == "OCR-XVLA"
    assert '"task5_candidate_set_exhausted": true' in text
    assert '"trigger_observability_passed": false' in text
    assert '"ours_rollout_happened": false' in text
