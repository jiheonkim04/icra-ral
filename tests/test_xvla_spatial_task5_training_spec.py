from pathlib import Path

from tca_map.xvla_spatial_task5.training_spec import (
    build_r2p_xvla_training_spec,
    validate_r2p_xvla_training_spec,
    write_r2p_xvla_training_spec,
)


def test_r2p_xvla_training_spec_is_valid_and_frozen() -> None:
    spec = build_r2p_xvla_training_spec()

    assert validate_r2p_xvla_training_spec(spec) == []
    assert spec["method"] == "R2P-XVLA"
    assert spec["training_happened_at_freeze"] is False
    assert spec["optimizer_step_happened_at_freeze"] is False
    assert spec["checkpoint_written_at_freeze"] is False
    assert spec["closed_loop_ours_evaluation_happened_at_freeze"] is False
    assert spec["pre_optimizer_required_gates"]["optimizer_step_allowed_before_all_gates"] is False
    assert spec["deployment_input_policy"]["privileged_state_at_inference"] is False
    assert spec["deployment_input_policy"]["phase_labels_at_inference"] is False
    assert spec["single_identity_safeguard"]["paper_claim_from_single_identity_allowed"] is False


def test_r2p_xvla_training_spec_has_exactly_primary_and_uniform_ablation() -> None:
    spec = build_r2p_xvla_training_spec()
    arms = spec["arms"]

    assert len(arms) == 2
    assert {arm["role"] for arm in arms} == {"primary_selected_method", "uniform_weight_ablation"}
    by_role = {arm["role"]: arm for arm in arms}
    assert by_role["primary_selected_method"]["method"] == "R2P-XVLA"
    assert by_role["primary_selected_method"]["phase_weight_lambda"] == 2.0
    assert by_role["primary_selected_method"]["phase_loss_weights"]["transit"] > 1.0
    assert by_role["primary_selected_method"]["phase_loss_weights"]["target_on_plate"] > 1.0
    assert by_role["uniform_weight_ablation"]["phase_weight_lambda"] == 0.0
    assert set(by_role["uniform_weight_ablation"]["phase_loss_weights"].values()) == {1.0}
    assert spec["matrix_limits"]["max_total_training_arms"] == 2
    assert spec["matrix_limits"]["new_configs_after_residual_rollout_allowed"] is False


def test_r2p_xvla_training_spec_writes_json_without_training(tmp_path: Path) -> None:
    output = tmp_path / "spec.json"
    spec = write_r2p_xvla_training_spec(output)

    assert output.exists()
    assert spec["freeze_id"] == "epoch5_r2p_xvla_task5_training_spec_v1"
    text = output.read_text(encoding="utf-8")
    assert '"optimizer_step_happened_at_freeze": false' in text
    assert '"closed_loop_ours_evaluation_happened_at_freeze": false' in text
