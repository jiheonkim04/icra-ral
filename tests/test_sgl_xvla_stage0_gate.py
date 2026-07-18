from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_stage0_gate import (
    CLEAN_RETENTION_IDENTITIES,
    HELD_OUT_CONFIRMATORY_IDENTITY_POOL,
    RESIDUAL_IDENTITIES,
    build_sgl_xvla_stage0_gate,
    validate_sgl_xvla_stage0_gate,
    write_sgl_xvla_stage0_gate,
)


def test_sgl_stage0_gate_is_valid_and_no_training() -> None:
    spec = build_sgl_xvla_stage0_gate()
    boundary = spec["authorization_boundary"]

    assert validate_sgl_xvla_stage0_gate(spec) == []
    assert spec["candidate"]["candidate_id"] == "SGL-XVLA"
    assert boundary["training_happened_at_freeze"] is False
    assert boundary["optimizer_step_happened_at_freeze"] is False
    assert boundary["checkpoint_written_at_freeze"] is False
    assert boundary["closed_loop_ours_evaluation_happened_at_freeze"] is False
    assert boundary["model_loaded_at_freeze"] is False
    assert boundary["simulator_episode_count_at_freeze"] == 0
    assert boundary["training_authorized_by_this_gate"] is False
    assert boundary["ours_rollout_authorized_by_this_gate"] is False


def test_sgl_stage0_gate_freezes_identity_roles() -> None:
    spec = build_sgl_xvla_stage0_gate()
    target = spec["target"]

    assert target["repeated_shared_failure_identities"] == RESIDUAL_IDENTITIES
    assert target["xvla_solved_clean_retention_identities"] == CLEAN_RETENTION_IDENTITIES
    assert target["held_out_confirmatory_identity_pool"] == HELD_OUT_CONFIRMATORY_IDENTITY_POOL
    assert target["held_out_pool_frozen_before_any_ours_result"] is True


def test_sgl_stage0_gate_forbids_privileged_inference_inputs() -> None:
    spec = build_sgl_xvla_stage0_gate()
    policy = spec["inference_input_policy"]

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


def test_sgl_stage0_gate_does_not_reopen_archived_methods() -> None:
    spec = build_sgl_xvla_stage0_gate()

    assert "R2P-XVLA" in spec["closed_methods_not_reopened"]
    assert "MPR-XVLA" in spec["closed_methods_not_reopened"]
    assert "R2P-XVLA or any closed method is reopened" in " ".join(spec["kill_rules"])


def test_sgl_stage0_gate_writes_json_without_runtime(tmp_path: Path) -> None:
    output = tmp_path / "sgl_stage0_gate.json"
    spec = write_sgl_xvla_stage0_gate(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert spec["freeze_id"] == "epoch5_sgl_xvla_task5_stage0_gate_v1"
    assert '"training_authorized_by_this_gate": false' in text
    assert '"closed_loop_ours_evaluation_happened_at_freeze": false' in text
