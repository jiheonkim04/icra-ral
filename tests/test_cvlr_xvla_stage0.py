from __future__ import annotations

import numpy as np
import torch

from tca_map.cvlr_xvla.stage0 import (
    AuxVisualTokenHook,
    CVLRLatentPredictor,
    apply_stage0_decision,
    load_frozen_contract,
    semantic_action_delta,
    trainable_parameter_count,
)


def test_frozen_contract_and_parameterization_are_exact() -> None:
    contract = load_frozen_contract()
    assert contract["method"] == "CVLR_XVLA"
    assert contract["training_budget"]["optimizer_steps_exact"] == 96
    assert contract["action_safety_thresholds"]["universal_max_absolute_threshold_used"] is False
    predictor = CVLRLatentPredictor()
    assert trainable_parameter_count(predictor) == 422144
    assert torch.count_nonzero(predictor.output.weight).item() == 0
    output = predictor(torch.randn(1, 50, 1024), torch.randn(1, 1024), torch.randn(1, 20))
    assert torch.count_nonzero(output).item() == 0


def test_auxiliary_hook_has_exact_clean_bypass_and_wrist_only_replacement() -> None:
    hook = AuxVisualTokenHook(50)
    auxiliary = torch.randn(1, 100, 1024)
    replacement = torch.randn(1, 50, 1024)
    kwargs = {"aux_visual_inputs": auxiliary, "other": torch.ones(1)}
    hook.activate(replacement, missing=False)
    args, clean = hook(None, (), kwargs)  # type: ignore[arg-type]
    assert args == ()
    assert clean is kwargs
    assert clean["aux_visual_inputs"] is auxiliary

    hook.activate(replacement, missing=True)
    _, dropout = hook(None, (), kwargs)  # type: ignore[arg-type]
    assert dropout is not kwargs
    assert torch.equal(dropout["aux_visual_inputs"][:, :50], replacement)
    assert torch.equal(dropout["aux_visual_inputs"][:, 50:], auxiliary[:, 50:])
    assert torch.equal(auxiliary, kwargs["aux_visual_inputs"])


def test_action_semantics_do_not_collapse_binary_gripper_into_continuous_axes() -> None:
    base = np.zeros((2, 20), dtype=np.float32)
    full = np.zeros((2, 20), dtype=np.float32)
    for plan in (base, full):
        plan[:, 3] = 1.0
        plan[:, 7] = 1.0
    base[:, 9] = 0.51
    full[:, 9] = 0.49
    full[:, 0] = 0.001
    delta = semantic_action_delta(full, base)
    assert delta["translation_rms"] > 0.0
    assert delta["rotation_rms"] == 0.0
    assert delta["raw_gripper_max_abs_delta"] < 0.1
    assert delta["gripper_flip_count"] == 2


def test_decision_rule_separates_validity_design_and_mechanism() -> None:
    names = [
        "target_records_valid",
        "split_integrity",
        "real_xvla_forward_path",
        "cuda_execution",
        "trainable_parameter_count_exact",
        "finite_nonzero_gradients",
        "optimizer_steps_exact",
        "weights_changed",
        "checkpoint_write_and_disk_reload",
        "xvla_frozen",
        "wrist_insertion_path_active",
        "reconstruction_meaningfully_beats_controls",
        "prediction_noncollapsed",
        "meaningful_full_vs_no_reconstruction_action_effect",
        "exact_clean_bypass",
        "semantic_action_safety",
        "action_outputs_finite",
    ]
    gates = {name: True for name in names}
    assert apply_stage0_decision(gates) == "CVLR_XVLA_STAGE0_PASS"
    assert (
        apply_stage0_decision(dict(gates, target_records_valid=False))
        == "CVLR_XVLA_STAGE0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        apply_stage0_decision(dict(gates, optimizer_steps_exact=False))
        == "CVLR_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        apply_stage0_decision(dict(gates, exact_clean_bypass=False))
        == "CVLR_XVLA_STAGE0_DESIGN_FAILURE"
    )
    assert (
        apply_stage0_decision(dict(gates, reconstruction_meaningfully_beats_controls=False))
        == "CVLR_XVLA_STAGE0_KEY_COMPONENT_NOT_USEFUL"
    )
