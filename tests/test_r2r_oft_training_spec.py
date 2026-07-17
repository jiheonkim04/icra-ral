from __future__ import annotations

import json
from pathlib import Path

from tca_map.r2r_oft.training_spec import (
    build_epoch5_training_spec,
    validate_training_spec,
    write_training_spec,
)


def test_epoch5_training_spec_is_bounded_and_valid() -> None:
    spec = build_epoch5_training_spec()

    assert validate_training_spec(spec) == []
    assert spec["training_happened_at_freeze"] is False
    assert spec["optimizer_step_happened_at_freeze"] is False
    assert spec["checkpoint_written_at_freeze"] is False
    assert spec["shared_training"]["load_in_4bit"] is True
    assert spec["shared_training"]["full_bf16_attempted"] is False
    assert spec["shared_training"]["max_optimizer_steps"] == 64
    assert spec["matrix_limits"]["max_total_training_arms"] == 2
    assert spec["matrix_limits"]["new_configs_after_confirmatory_rollout_allowed"] is False


def test_epoch5_training_spec_contains_primary_and_uniform_ablation_only() -> None:
    spec = build_epoch5_training_spec()
    arms = spec["arms"]

    assert len(arms) == 2
    by_role = {arm["role"]: arm for arm in arms}
    assert set(by_role) == {"primary_selected_method", "uniform_weight_ablation"}
    assert by_role["primary_selected_method"]["phase_weight_lambda"] == 2.0
    assert by_role["uniform_weight_ablation"]["phase_weight_lambda"] == 0.0
    assert by_role["primary_selected_method"]["lora_rank"] == by_role["uniform_weight_ablation"]["lora_rank"] == 4
    assert by_role["primary_selected_method"]["sampler"] == by_role["uniform_weight_ablation"]["sampler"]


def test_epoch5_training_spec_blocks_confirmatory_reset_tuning() -> None:
    spec = build_epoch5_training_spec()

    assert spec["data"]["residual_failure_reset_identities"] == [20260721, 20260722]
    assert spec["validation_selection"]["closed_loop_residual_resets_used_for_model_selection"] is False
    assert any("retuning based on residual reset identities" in rule for rule in spec["kill_rules"])
    assert "offline_validation_only_before_closed_loop" == spec["validation_selection"]["selection_source"]


def test_write_training_spec_round_trips_json(tmp_path: Path) -> None:
    output = tmp_path / "spec.json"
    spec = write_training_spec(output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == spec
    assert loaded["freeze_id"] == "epoch5_r2r_oft_training_spec_v1"
