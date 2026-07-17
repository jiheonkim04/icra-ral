from __future__ import annotations

from tca_map.xvla_task6.training_spec import build_mpr_xvla_training_spec, validate_mpr_xvla_training_spec


def test_mpr_xvla_training_spec_freezes_two_arms_without_training() -> None:
    spec = build_mpr_xvla_training_spec()

    assert validate_mpr_xvla_training_spec(spec) == []
    assert spec["training_happened_at_freeze"] is False
    assert spec["optimizer_step_happened_at_freeze"] is False
    assert spec["checkpoint_written_at_freeze"] is False
    assert spec["closed_loop_ours_evaluation_happened_at_freeze"] is False
    assert spec["method"] == "MPR-XVLA"
    assert len(spec["arms"]) == 2
    assert {arm["role"] for arm in spec["arms"]} == {"primary_selected_method", "uniform_weight_ablation"}


def test_mpr_xvla_training_spec_requires_task6_gates_and_uniform_ablation() -> None:
    spec = build_mpr_xvla_training_spec()

    assert spec["prerequisites"]["second_prior_gate"] == "TASK6_NOT_SOLVED_BY_OPENVLA_OFT_INT4"
    assert spec["interface_audit"]["official_training_script_has_peft_lora"] is True
    assert spec["interface_audit"]["local_data_adapter_required_before_gradient_smoke"] is True
    assert spec["interface_audit"]["raw_libero_hdf5_is_not_direct_official_xvla_training_input"] is True
    assert spec["pre_optimizer_required_gates"]["optimizer_step_allowed_before_all_gates"] is False
    assert spec["data"]["residual_failure_reset_identities"] == [20260725, 20260731]
    assert spec["data"]["residual_initial_state_overlap_count"] == 0
    assert spec["data"]["privileged_state_at_inference"] is False
    assert "uniform ablation" in spec["validation_selection"]["primary_vs_ablation_rule"]
    assert "using simulator object state as an inference input" in spec["kill_rules"]


def test_mpr_xvla_training_spec_rejects_extra_arm_or_missing_second_prior_gate() -> None:
    spec = build_mpr_xvla_training_spec()
    spec["arms"].append({**spec["arms"][0], "arm_id": "extra_forbidden_task6_arm"})
    spec["prerequisites"]["second_prior_gate"] = "MISSING"

    errors = validate_mpr_xvla_training_spec(spec)

    assert "spec must contain exactly two training arms" in errors
    assert "second-prior no-solve gate must be recorded" in errors
