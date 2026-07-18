from pathlib import Path

from tca_map.xvla_spatial_task5.ocr_stage0_gate import (
    OCR_CANDIDATE_ID,
    build_ocr_xvla_stage0_gate,
    validate_ocr_xvla_stage0_gate,
    write_ocr_xvla_stage0_gate,
)


def test_ocr_stage0_gate_is_valid_and_no_training() -> None:
    spec = build_ocr_xvla_stage0_gate()
    boundary = spec["authorization_boundary"]

    assert validate_ocr_xvla_stage0_gate(spec) == []
    assert spec["candidate"]["candidate_id"] == OCR_CANDIDATE_ID
    assert boundary["training_happened_at_freeze"] is False
    assert boundary["optimizer_step_happened_at_freeze"] is False
    assert boundary["checkpoint_written_at_freeze"] is False
    assert boundary["closed_loop_ours_evaluation_happened_at_freeze"] is False
    assert boundary["model_loaded_at_freeze"] is False
    assert boundary["simulator_episode_count_at_freeze"] == 0
    assert boundary["training_authorized_by_this_gate"] is False
    assert boundary["ours_rollout_authorized_by_this_gate"] is False
    assert boundary["control_rollout_authorized_by_this_gate"] is False


def test_ocr_stage0_gate_requires_sgl_static_block() -> None:
    spec = build_ocr_xvla_stage0_gate()
    upstream = spec["upstream_primary_candidate"]

    assert upstream["candidate_id"] == "SGL-XVLA"
    assert upstream["valid"] is True
    assert upstream["blocked_before_simulator_episode"] is True
    assert upstream["backup_candidate_stage0_authorized_next"] is True
    assert "SGL-XVLA-current-frozen-executable" in spec["closed_methods_not_reopened"]


def test_ocr_stage0_gate_freezes_identity_roles() -> None:
    spec = build_ocr_xvla_stage0_gate()
    target = spec["target"]

    assert target["repeated_shared_failure_identities"] == [20260727, 20260730, 20260733]
    assert target["xvla_solved_clean_retention_identities"] == [20260731, 20260732]
    assert target["held_out_confirmatory_identity_pool"] == [20260734, 20260735, 20260736, 20260737]
    assert target["held_out_pool_frozen_before_any_ocr_result"] is True


def test_ocr_stage0_gate_allows_action_history_but_forbids_privileged_inputs() -> None:
    spec = build_ocr_xvla_stage0_gate()
    policy = spec["inference_input_policy"]

    assert "recent action history already produced by the policy" in policy["allowed_inputs"]
    assert policy["privileged_state_at_inference"] is False
    assert policy["reward_or_success_at_inference"] is False
    assert policy["reset_identity_at_inference"] is False
    for forbidden in [
        "simulator object pose",
        "simulator contact state",
        "reward",
        "success flag",
        "HDF5 demo identity",
        "reset identity label",
        "phase label",
        "task success oracle",
    ]:
        assert forbidden in policy["forbidden_inputs"]


def test_ocr_stage0_gate_freezes_required_checks_and_comparator_roles() -> None:
    spec = build_ocr_xvla_stage0_gate()
    check_ids = {item["check_id"] for item in spec["stage0_required_checks"]}
    roles = spec["future_comparator_roles"]

    assert check_ids == {
        "observation_consistency_trigger_observability_no_training",
        "retry_action_bounds_no_optimizer",
        "simple_timeout_retry_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    assert roles["simple_control"]["comparator_role"] == "CONTROL"
    assert "fixed timeout retry" in roles["simple_control"]["scientific_question"].lower()
    assert roles["key_ablation"]["comparator_role"] == "ABLATION"


def test_ocr_stage0_gate_writes_json_without_runtime(tmp_path: Path) -> None:
    output = tmp_path / "ocr_stage0_gate.json"
    spec = write_ocr_xvla_stage0_gate(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert spec["freeze_id"] == "epoch5_ocr_xvla_task5_stage0_gate_v1"
    assert '"training_authorized_by_this_gate": false' in text
    assert '"ours_rollout_authorized_by_this_gate": false' in text
