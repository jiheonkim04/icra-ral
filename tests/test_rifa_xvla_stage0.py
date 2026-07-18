from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tca_map.rifa_xvla.stage0 import (
    RIFAAdapter,
    action_delta,
    apply_stage0_decision,
    load_frozen_contract,
    normalized_context,
    plan_to_libero_actions,
    trainable_parameter_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "configs" / "rifa_xvla_stage0_frozen_contract.json"


def test_frozen_contract_preserves_panel_split_and_budget() -> None:
    contract = load_frozen_contract(CONTRACT)
    assert contract["execution_classification"] == "OURS_VLA_TRAINING"
    assert [(row["suite"], row["task_id"], row["identities"]) for row in contract["panel"]] == [
        ("libero_goal", 0, [20260733, 20260734, 20260735]),
        ("libero_object", 0, [20260733, 20260734, 20260735]),
        ("libero_spatial", 5, [20260731, 20260732, 20260735]),
    ]
    assert contract["data_split"]["training_demo_indices"] == "0..39"
    assert contract["data_split"]["validation_demo_indices"] == "40..49"
    assert contract["training_budget"]["configuration_count"] == 1
    assert contract["training_budget"]["optimizer_steps_per_arm"] == 6
    assert contract["execution_boundary"]["closed_loop_rollout_authorized"] is False
    json.dumps(contract)


def test_rifa_is_exact_base_passthrough_at_initialization_and_on_clean() -> None:
    torch.manual_seed(7)
    adapter = RIFAAdapter(hidden_size=16, imputed_dim=8, bottleneck_dim=4, residual_scale=0.05)
    hidden = torch.randn(2, 3, 16)
    imputed = torch.randn(2, 8)
    reliability = torch.randn(2, 3)
    missing = torch.ones(2, 1)
    output, telemetry = adapter(hidden, imputed, reliability, missing)
    assert torch.equal(output, hidden)
    assert torch.count_nonzero(telemetry["residual"]).item() == 0

    with torch.no_grad():
        adapter.residual_projection.weight.fill_(0.2)
        adapter.residual_projection.bias.fill_(0.1)
    clean, clean_telemetry = adapter(hidden, imputed, reliability, torch.zeros(2, 1))
    assert torch.equal(clean, hidden)
    assert torch.count_nonzero(clean_telemetry["gate"]).item() == 0


def test_no_reliability_ablation_has_matched_capacity_and_neutral_features() -> None:
    torch.manual_seed(11)
    full = RIFAAdapter(hidden_size=12, imputed_dim=6, bottleneck_dim=5, no_reliability=False)
    ablation = RIFAAdapter(hidden_size=12, imputed_dim=6, bottleneck_dim=5, no_reliability=True)
    ablation.load_state_dict(full.state_dict())
    assert trainable_parameter_count(full) == trainable_parameter_count(ablation) > 0
    with torch.no_grad():
        full.residual_projection.weight.fill_(0.1)
        ablation.residual_projection.weight.fill_(0.1)
        full.reliability_gate.weight.fill_(0.5)
        ablation.reliability_gate.weight.fill_(0.5)
    hidden = torch.zeros(1, 2, 12)
    imputed = torch.ones(1, 6)
    missing = torch.ones(1, 1)
    low = torch.zeros(1, 3)
    high = torch.ones(1, 3)
    full_low, _ = full(hidden, imputed, low, missing)
    full_high, _ = full(hidden, imputed, high, missing)
    ablation_low, _ = ablation(hidden, imputed, low, missing)
    ablation_high, _ = ablation(hidden, imputed, high, missing)
    assert not torch.equal(full_low, full_high)
    assert torch.equal(ablation_low, ablation_high)


def test_context_normalization_uses_frozen_train_statistics() -> None:
    context = {
        "imputed_feature": np.asarray([1.0, 2.0], dtype=np.float32),
        "reliability_raw": np.asarray([2.0, 4.0, 8.0], dtype=np.float32),
        "missing_indicator": 1.0,
    }
    normalized = normalized_context(
        context,
        {
            "mean": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
            "std": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        },
    )
    np.testing.assert_allclose(normalized["reliability"], np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(normalized["imputed_feature"], [1.0, 2.0])
    np.testing.assert_allclose(normalized["missing_indicator"], [1.0])


def test_action_delta_uses_libero_7d_semantics() -> None:
    plan = np.zeros((2, 20), dtype=np.float32)
    plan[:, 3] = 1.0
    plan[:, 7] = 1.0
    plan[:, 9] = 0.6
    converted = plan_to_libero_actions(plan)
    assert converted.shape == (2, 7)
    assert np.isfinite(converted).all()
    assert np.all(converted[:, 6] == 1.0)
    same = action_delta(plan, plan.copy())
    assert same["rms"] == 0.0
    assert same["max_abs"] == 0.0
    assert same["gripper_flip_count"] == 0


def test_stage0_decision_rule_is_frozen_and_precise() -> None:
    names = [
        "real_xvla_forward_path",
        "cuda_execution",
        "trainable_parameters_nonzero_and_matched",
        "finite_nonzero_gradients",
        "optimizer_steps_exact",
        "weights_changed",
        "checkpoint_write_and_disk_reload",
        "base_preserving_initialization",
        "missing_modality_signal_observable",
        "rl4il_reliability_features_nonconstant",
        "full_vs_no_reliability_difference",
        "bounded_action_delta",
        "clean_validation_retained",
        "action_outputs_finite",
    ]
    gates = {name: True for name in names}
    assert apply_stage0_decision(gates) == "RIFA_XVLA_STAGE0_PASS"
    data_failure = dict(gates, rl4il_reliability_features_nonconstant=False)
    assert apply_stage0_decision(data_failure) == "RIFA_XVLA_STAGE0_DATA_OR_SUPERVISION_FAILURE"
    implementation_failure = dict(gates, finite_nonzero_gradients=False)
    assert apply_stage0_decision(implementation_failure) == "RIFA_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    design_failure = dict(gates, clean_validation_retained=False)
    assert apply_stage0_decision(design_failure) == "RIFA_XVLA_STAGE0_DESIGN_FAILURE"
    unresolved = dict(gates, full_vs_no_reliability_difference=False)
    assert apply_stage0_decision(unresolved) == "RIFA_XVLA_STAGE0_UNDERPOWERED_OR_UNRESOLVED"
