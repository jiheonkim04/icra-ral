from pathlib import Path

from tca_map.xvla_spatial_task5.sgl_rollout_protocol import (
    DEVELOPMENT_EVALUATION_IDENTITIES,
    ROLLOUT_ARMS,
    build_sgl_rollout_protocol,
    validate_sgl_rollout_protocol,
    write_sgl_rollout_protocol,
)


def test_rollout_protocol_is_valid_and_report_only() -> None:
    protocol = build_sgl_rollout_protocol()
    execution = protocol["execution_classification"]

    assert validate_sgl_rollout_protocol(protocol) == []
    assert protocol["decision"] == "SGL_NO_TRAINING_ROLLOUT_PROTOCOL_FROZEN_NO_EPISODES_RUN"
    assert execution["execution_type"] == "REPORT_ONLY"
    assert execution["simulator_episode_count"] == 0
    assert execution["vla_model_loaded"] is False
    assert execution["training_happened"] is False
    assert execution["optimizer_step_happened"] is False
    assert execution["checkpoint_written"] is False
    assert execution["control_rollout_happened"] is False
    assert execution["closed_loop_ours_evaluation_happened"] is False


def test_rollout_protocol_freezes_identities_and_arms() -> None:
    protocol = build_sgl_rollout_protocol()
    target = protocol["target"]

    assert target["development_residual_identities"] == [20260727, 20260730, 20260733]
    assert target["clean_retention_identities"] == [20260731, 20260732]
    assert target["development_evaluation_identities"] == DEVELOPMENT_EVALUATION_IDENTITIES
    assert target["held_out_confirmatory_identity_pool"] == [20260734, 20260735, 20260736, 20260737]
    assert target["held_out_used_in_this_protocol"] is False
    assert [arm["arm_id"] for arm in protocol["rollout_arms"]] == ROLLOUT_ARMS


def test_rollout_protocol_freezes_episode_budget_without_authorizing_episodes() -> None:
    protocol = build_sgl_rollout_protocol()
    budget = protocol["episode_budget_if_later_authorized"]
    auth = protocol["authorization_boundary"]

    assert budget["control_development_episodes"] == len(DEVELOPMENT_EVALUATION_IDENTITIES)
    assert budget["sgl_development_episodes"] == len(DEVELOPMENT_EVALUATION_IDENTITIES)
    assert budget["xvla_reference_rerun_episodes"] == 0
    assert budget["held_out_episodes"] == 0
    assert budget["total_new_simulator_episodes"] == len(DEVELOPMENT_EVALUATION_IDENTITIES) * 2
    assert budget["episode_budget_is_frozen_before_any_new_simulator_episode"] is True
    assert auth["simulator_episode_authorized_by_this_artifact"] is False
    assert auth["control_rollout_authorized_now"] is False
    assert auth["ours_rollout_authorized_now"] is False
    assert auth["held_out_rollout_authorized"] is False


def test_rollout_protocol_decision_rules_are_comparator_specific() -> None:
    protocol = build_sgl_rollout_protocol()
    rules = protocol["development_decision_rules"]
    calibration = protocol["comparator_role_calibration"]

    assert "at least 2 of 3" in rules["primary_residual_pass_condition"]
    assert "both clean-retention identities" in rules["clean_retention_pass_condition"]
    assert "fixed lift/regrasp control blocks" in rules["simple_control_blocking_condition"]
    assert "Held-out identities are not run" in rules["held_out_advancement_condition"]
    assert calibration["universal_beat_all_rule_applied"] is False


def test_rollout_protocol_requires_durable_future_workers_and_forbids_training() -> None:
    protocol = build_sgl_rollout_protocol()
    durability_files = set(protocol["durability_requirements_for_future_worker"]["required_files"])
    forbidden = " ".join(protocol["forbidden_actions"])

    for required in ["heartbeat.txt", "exit_code.txt", "stdout.log", "stderr.log", "result.json", "result.md"]:
        assert required in durability_files
    assert "No training" in forbidden
    assert "No held-out identity rollout" in forbidden
    assert "No simulator object state" in forbidden
    assert "No reopening R2P-XVLA" in forbidden
    assert protocol["bounded_conclusion"]["candidate_can_train"] is False
    assert protocol["bounded_conclusion"]["candidate_can_roll_out_ours_now"] is False


def test_rollout_protocol_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "rollout_protocol.json"
    protocol = write_sgl_rollout_protocol(output)
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert protocol["candidate_id"] == "SGL-XVLA"
    assert '"rollout_protocol_frozen": true' in text
    assert '"simulator_episode_authorized_by_this_artifact": false' in text
    assert '"candidate_can_train": false' in text
