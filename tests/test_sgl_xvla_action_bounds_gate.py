from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_action_bounds_gate import (
    build_sgl_action_bounds_gate,
    validate_sgl_action_bounds_gate,
    write_sgl_action_bounds_gate,
)


def test_action_bounds_gate_is_valid_and_report_only() -> None:
    gate = build_sgl_action_bounds_gate()
    execution = gate["execution_classification"]

    assert validate_sgl_action_bounds_gate(gate) == []
    assert gate["decision"] == "SGL_ACTION_BIAS_BOUNDS_FROZEN_POST_CLAMP_NO_OPTIMIZER_NO_TRAINING"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False
    assert execution["lora_or_qlora_training_happened"] is False


def test_action_bounds_are_conservative_against_expert_stats() -> None:
    gate = build_sgl_action_bounds_gate()
    bounds = gate["frozen_action_bias_bounds"]
    expert = gate["expert_action_evidence"]["selected_demo_action_stats"]

    assert expert["finite"] is True
    assert expert["clip_rate_if_env_adapter_applied"] == 0.0
    assert bounds["lift_axis_translation_bias_max_abs_per_step"] <= expert["translation_range"]["max_abs"] * 0.25
    assert bounds["lateral_translation_bias_max_abs_per_step"] == 0.0
    assert bounds["rotation_bias_max_abs_per_step"] == 0.0
    assert bounds["gripper_bias_max_abs_per_step"] <= expert["gripper_range"]["max_abs"] * 0.25
    assert bounds["max_activated_chunks"] == 2
    assert bounds["max_activated_steps"] == 60
    assert bounds["post_bias_action_clamp_abs"] == 1.0
    assert bounds["no_optimizer"] is True
    assert bounds["no_learned_parameters"] is True


def test_action_bounds_record_xvla_saturation_guard() -> None:
    gate = build_sgl_action_bounds_gate()
    evidence = gate["xvla_action_range_evidence"]
    saturation = gate["saturation_guard"]

    assert evidence["known_raw_xvla_first_two_chunks_exceed_env_bound"] is True
    assert evidence["max_observed_first_two_abs"] > 1.0
    assert saturation["post_bias_action_clamp_required"] is True
    assert saturation["post_bias_action_clamp_abs"] == 1.0
    assert saturation["suppress_bias_if_component_already_saturated"] is True
    assert saturation["forbid_bias_that_increases_existing_saturation"] is True
    assert saturation["fail_if_clean_retention_added_clip_count_positive"] is True


def test_action_bounds_preserve_identity_roles_and_clean_retention() -> None:
    gate = build_sgl_action_bounds_gate()
    target = gate["target"]
    evidence_identities = {item["reset_identity"] for item in gate["xvla_action_range_evidence"]["records"]}

    assert target["residual_identities"] == [20260727, 20260730, 20260733]
    assert target["clean_retention_identities"] == [20260731, 20260732]
    assert {20260730, 20260731, 20260732, 20260733}.issubset(evidence_identities)
    assert gate["bounded_conclusion"]["candidate_can_train"] is False
    assert gate["bounded_conclusion"]["candidate_can_roll_out_ours"] is False
    assert gate["bounded_conclusion"]["candidate_can_advance_to_simple_fixed_lift_control_gate"] is True


def test_action_bounds_include_comparator_role_implications() -> None:
    gate = build_sgl_action_bounds_gate()
    roles = {item["comparator_role"]: item for item in gate["comparator_role_implications"]}

    assert "BASE" in roles
    assert "SIMPLE_CONTROL" in roles
    assert "CLEAN_RETENTION" in roles
    assert roles["SIMPLE_CONTROL"]["does_this_gate_answer_it"] is False
    assert "fixed lift" in roles["SIMPLE_CONTROL"]["scientific_question"].lower()


def test_action_bounds_gate_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "action_bounds.json"
    gate = write_sgl_action_bounds_gate(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert gate["candidate_id"] == "SGL-XVLA"
    assert '"action_bounds_frozen": true' in text
    assert '"candidate_can_train": false' in text
    assert '"post_bias_action_clamp_abs": 1.0' in text
