from __future__ import annotations

import json
import pathlib

import pytest
import torch

from tca_map.action_consistent_missing_view_distillation.stage0 import (
    adjudicate_stage0,
    arm_loss,
    bootstrap_difference,
    learning_rate_for_step,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "action_consistent_missing_view_distillation_xvla_stage0_execution_contract.json"


class _FakeActionSpace:
    def compute_loss(self, _pred: torch.Tensor, _target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "position_loss": torch.tensor(500.0),
            "rotate6D_loss": torch.tensor(10.0),
            "gripper_loss": torch.tensor(1.0),
        }


class _FakeModel:
    action_space = _FakeActionSpace()


def test_frozen_learning_rate_schedule_endpoints() -> None:
    assert learning_rate_for_step(1) == pytest.approx(3e-4 / 8)
    assert learning_rate_for_step(8) == pytest.approx(3e-4)
    assert learning_rate_for_step(128) == pytest.approx(3e-5)
    with pytest.raises(ValueError):
        learning_rate_for_step(0)


def test_all_four_frozen_arm_losses() -> None:
    components = {
        "hidden_mse": torch.tensor(1.0),
        "translation_mse": torch.tensor(1.0),
        "rotation_mse": torch.tensor(1.0),
        "raw_gripper_margin_mse": torch.tensor(1.0),
        "wrist_reconstruction_mse": torch.tensor(1.0),
    }
    denominators = {key: 1.0 for key in components}
    raw = torch.zeros(1, 30, 20)
    assert arm_loss("OURS_FULL", components, denominators, _FakeModel(), raw, raw)[0].item() == pytest.approx(3.5)
    assert arm_loss("NO_RECONSTRUCTION", components, denominators, _FakeModel(), raw, raw)[0].item() == pytest.approx(3.25)
    assert arm_loss("NO_RAW_GRIPPER_MARGIN", components, denominators, _FakeModel(), raw, raw)[0].item() == pytest.approx(2.5)
    assert arm_loss("GENERIC_WRIST_DROPOUT_ADAPTER", components, denominators, _FakeModel(), raw, raw)[0].item() == pytest.approx(3.0)


def test_bootstrap_is_deterministic_and_paired() -> None:
    first = bootstrap_difference([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], seed=7)
    second = bootstrap_difference([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], seed=7)
    assert first == second
    assert first["ci95_low"] == pytest.approx(1.0)
    assert first["ci95_high"] == pytest.approx(1.0)


def test_stage0_decision_precedence_is_frozen() -> None:
    comparison = {
        "point_gate": True,
        "ci_gate": True,
        "other_metric_nonregression": True,
        "gripper_nonregression": True,
        "metrics": {"x": {"absolute_improvement": 1.0}},
    }
    comparisons = {
        "full_vs_no_reconstruction": comparison,
        "full_vs_generic": comparison,
    }
    gates = {
        "execution_valid": True,
        "action_legality_and_smoothness": True,
        "reconstruction_gate": True,
        "base_directional_gate": True,
    }
    assert adjudicate_stage0(gates, comparisons) == "STAGE0_GO"
    assert adjudicate_stage0({**gates, "execution_valid": False}, comparisons) == "STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE"
    assert adjudicate_stage0({**gates, "action_legality_and_smoothness": False}, comparisons) == "STAGE0_ACTION_LEGALITY_FAILURE"
    generic_failed = {**comparison, "point_gate": False, "ci_gate": False}
    assert adjudicate_stage0(gates, {**comparisons, "full_vs_generic": generic_failed}) == "STAGE0_GENERIC_ADAPTATION_EXPLAINS_GAIN"


def test_execution_contract_locks_measured_inputs_and_no_confirmation_access() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["microbatch_preflight"]["selected_microbatch"] == 8
    assert contract["microbatch_preflight"]["gradient_accumulation"] == 1
    assert contract["optimization"]["optimizer_steps_per_arm"] == 128
    assert contract["optimization"]["checkpoint_steps"] == [64, 128]
    assert contract["data_execution"]["training_record_count"] == 480
    assert contract["data_execution"]["validation_record_count"] == 12
    assert contract["data_execution"]["confirmation_demo_indices_inaccessible"] == "41..49 inclusive"
    assert contract["prohibitions"]["confirmatory_access"] is False
    assert contract["prohibitions"]["physical_robot_manipulation"] is False
