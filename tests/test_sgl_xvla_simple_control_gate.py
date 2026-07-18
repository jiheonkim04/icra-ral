from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_simple_control_gate import (
    CONTROL_ID,
    build_sgl_simple_control_gate,
    validate_sgl_simple_control_gate,
    write_sgl_simple_control_gate,
)


def test_simple_control_gate_is_valid_and_report_only() -> None:
    gate = build_sgl_simple_control_gate()
    execution = gate["execution_classification"]

    assert validate_sgl_simple_control_gate(gate) == []
    assert gate["decision"] == "SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_FROZEN_NO_TRAINING_NO_OURS"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_simple_control_freezes_exactly_one_control_for_the_objection() -> None:
    gate = build_sgl_simple_control_gate()
    control = gate["simple_control"]
    conclusion = gate["bounded_conclusion"]

    assert control["control_id"] == CONTROL_ID
    assert control["comparator_role"] == "SIMPLE_CONTROL"
    assert control["is_primary_simple_control"] is True
    assert control["other_simple_controls_frozen_for_same_objection"] == []
    assert conclusion["exactly_one_simple_control_for_this_objection"] is True


def test_simple_control_template_respects_action_bounds() -> None:
    gate = build_sgl_simple_control_gate()
    template = gate["simple_control"]["template"]
    schedule = template["schedule"]

    assert template["nonadaptive"] is True
    assert template["max_activated_chunks"] == 2
    assert template["max_activated_steps"] == 60
    assert template["zero_bias_after_chunk_index"] == 1
    assert template["post_bias_action_clamp_abs"] == 1.0
    assert template["saturation_guard_inherited"] is True
    assert [item["chunk_index"] for item in schedule] == [0, 1]
    for item in schedule:
        assert item["lift_axis_translation_bias_abs"] <= 0.20
        assert item["gripper_close_bias_abs"] <= 0.25
        assert item["lateral_translation_bias_abs"] == 0.0
        assert item["rotation_bias_abs"] == 0.0


def test_simple_control_activation_uses_no_privileged_inputs_and_keeps_retention() -> None:
    gate = build_sgl_simple_control_gate()
    activation = gate["simple_control"]["activation_condition"]
    target = gate["target"]

    assert activation["allowed_activation_sources"] == ["language instruction"]
    assert activation["uses_visual_progress_feedback"] is False
    assert activation["uses_simulator_state"] is False
    assert activation["uses_reward_or_success"] is False
    assert activation["uses_reset_identity"] is False
    assert activation["activates_on_clean_retention_identities"] == [20260731, 20260732]
    assert target["residual_identities"] == [20260727, 20260730, 20260733]
    assert target["clean_retention_identities"] == [20260731, 20260732]


def test_simple_control_uses_calibrated_comparator_role() -> None:
    gate = build_sgl_simple_control_gate()
    calibration = gate["comparator_role_calibration"]

    assert calibration["universal_beat_all_rule_applied"] is False
    assert "substantially all future SGL gain" in calibration["scientific_question"]
    assert "equal/lower cost" in calibration["blocking_condition"]
    assert gate["bounded_conclusion"]["candidate_can_train"] is False
    assert gate["bounded_conclusion"]["candidate_can_roll_out_ours"] is False
    assert gate["bounded_conclusion"]["control_can_roll_out_now"] is False


def test_simple_control_gate_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "simple_control.json"
    gate = write_sgl_simple_control_gate(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert gate["simple_control"]["control_id"] == CONTROL_ID
    assert '"simple_control_frozen": true' in text
    assert '"control_rollout_happened": false' in text
    assert '"candidate_can_train": false' in text
