from __future__ import annotations

from tca_map.xvla_task1.training_spec import build_br_xvla_training_spec, validate_br_xvla_training_spec


def test_br_xvla_training_spec_freezes_two_arms_without_training() -> None:
    spec = build_br_xvla_training_spec()

    assert validate_br_xvla_training_spec(spec) == []
    assert spec["training_happened_at_freeze"] is False
    assert spec["optimizer_step_happened_at_freeze"] is False
    assert spec["checkpoint_written_at_freeze"] is False
    assert spec["method"] == "BR-XVLA"
    assert len(spec["arms"]) == 2
    assert {arm["role"] for arm in spec["arms"]} == {"primary_selected_method", "uniform_weight_ablation"}


def test_br_xvla_training_spec_keeps_adapter_and_residual_gates_closed() -> None:
    spec = build_br_xvla_training_spec()

    assert spec["interface_audit"]["official_training_script_has_peft_lora"] is True
    assert spec["interface_audit"]["local_data_adapter_required_before_gradient_smoke"] is True
    assert spec["interface_audit"]["raw_libero_hdf5_is_not_direct_official_xvla_training_input"] is True
    assert spec["pre_optimizer_required_gates"]["optimizer_step_allowed_before_all_gates"] is False
    assert spec["data"]["residual_failure_reset_identity"] == 20260727
    assert spec["data"]["excluded_xvla_regression_identity"] == 20260725
    assert "using X-VLA regression identity 20260725 as an Ours target" in spec["kill_rules"]
