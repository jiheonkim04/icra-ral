from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_epoch7_contact_topology_stage0b.py"
SPEC = importlib.util.spec_from_file_location("epoch7_contact_stage0b", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage0b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage0b)


def test_contract_and_stage0a_authorization_hashes_are_frozen() -> None:
    assert stage0b.sha256_file(stage0b.PROTOCOL_PATH) == stage0b.EXPECTED_PROTOCOL_SHA256
    assert stage0b.sha256_file(stage0b.CONTRACT_PATH) == stage0b.EXPECTED_CONTRACT_SHA256
    assert stage0b.EXPECTED_STAGE0A_RESULT_SHA256 == (
        "3D18F9D7C6FA2E6311E4667853D1247FFC515B1BCD3325671A46BB0994160E06"
    )


def test_boundary_mask_is_demo_local_plus_or_minus_two() -> None:
    typed = np.zeros((9, 10), dtype=np.uint8)
    typed[4, 2] = 1
    assert stage0b.boundary_mask(typed).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_visual_probe_matches_frozen_four_block_architecture() -> None:
    model = stage0b.VisualProbe(nonvisual_dim=30)
    convolutions = [layer for layer in model.encoder if isinstance(layer, torch.nn.Conv2d)]
    normalizations = [layer for layer in model.encoder if isinstance(layer, torch.nn.GroupNorm)]
    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (12, 16),
        (16, 32),
        (32, 64),
        (64, 64),
    ]
    assert len(normalizations) == 4
    output = model(torch.zeros(2, 12, 64, 64), torch.zeros(2, 30))
    assert output.shape == (2, 11)


def test_nonvisual_probe_has_same_64_unit_fusion_width() -> None:
    model = stage0b.NonvisualProbe(nonvisual_dim=30)
    linear = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(30, 64), (64, 11)]


def test_within_group_shuffle_never_crosses_demo_groups() -> None:
    values = np.arange(24).reshape(6, 4)
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    shuffled = stage0b.shuffled_within_groups(values, groups, np.random.default_rng(7))
    assert sorted(map(tuple, shuffled[:3])) == sorted(map(tuple, values[:3]))
    assert sorted(map(tuple, shuffled[3:])) == sorted(map(tuple, values[3:]))


def test_ridge_alpha_selection_uses_only_tune_and_returns_finite_validation() -> None:
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(40, 5))
    x_tune = rng.normal(size=(15, 5))
    x_validation = rng.normal(size=(12, 5))
    weights = rng.normal(size=(5, 6))
    y_train = x_train @ weights
    y_tune = x_tune @ weights
    y_validation = x_validation @ weights
    result = stage0b.fit_ridge_family(
        x_train, x_tune, x_validation, y_train, y_tune, y_validation
    )
    assert result["selected_alpha"] in stage0b.ALPHAS
    assert len(result["alpha_scores"]) == 7
    assert np.isfinite(result["validation_prediction"]).all()


def test_stage0b_loader_source_never_requests_outcome_datasets_or_steps() -> None:
    source = inspect.getsource(stage0b.load_rows)
    assert '["rewards"]' not in source
    assert '["dones"]' not in source
    assert ".step(" not in source
    assert "check_success" not in source
    assert set(stage0b.ALLOWED_DATASETS) == {
        "actions",
        "obs/agentview_rgb",
        "obs/eye_in_hand_rgb",
        "obs/ee_pos",
        "obs/ee_ori",
        "obs/gripper_states",
        "obs/joint_states",
    }


def test_adjudication_requires_both_gate_families() -> None:
    visual = {
        "all_gates_passed": True,
        "metrics": {
            "any_transition_auprc_over_nonvisual": 0.2,
            "supported_typed_bin_macro_ap_over_nonvisual": 0.1,
        },
    }
    oracle = {
        "all_gates_passed": True,
        "metrics": {"aggregate_arm_nrmse_relative_reduction": 0.06},
    }
    assert stage0b.adjudicate(visual, oracle) == "CONTACT_TOPOLOGY_PREMETHOD_STAGE0_GO"
    oracle["all_gates_passed"] = False
    oracle["metrics"]["aggregate_arm_nrmse_relative_reduction"] = -0.01
    assert stage0b.adjudicate(visual, oracle) == "STAGE0_TRIVIAL_EQUIVALENCE"
