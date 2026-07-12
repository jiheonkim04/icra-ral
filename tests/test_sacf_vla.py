from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.sacf_vla import (
    SACFConfig,
    SACFExample,
    assert_no_privileged_inference_fields,
    instruction_to_demo_filename,
    make_sacf_features,
    predict_plain_action,
    predict_sacf_action,
    semantic_hash_features,
    task_phase_mean_action,
    train_plain_bc_prefix,
    train_sacf_policy,
)


def _examples(count: int = 72) -> list[SACFExample]:
    config = SACFConfig()
    rows: list[SACFExample] = []
    instructions = [
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the ketchup and place it in the basket",
        "pick up the milk and place it in the basket",
    ]
    families = ["libero_spatial", "libero_spatial", "libero_object", "libero_object"]
    semantic_prototypes = {
        instructions[0]: np.asarray([0.35, -0.2, 0.1, 0.0, 0.0, 0.05, -0.4], dtype=np.float32),
        instructions[1]: np.asarray([-0.25, 0.25, 0.08, 0.0, 0.0, 0.02, -0.4], dtype=np.float32),
        instructions[2]: np.asarray([0.18, 0.10, -0.05, 0.0, 0.0, -0.02, -0.4], dtype=np.float32),
        instructions[3]: np.asarray([-0.16, -0.12, -0.03, 0.0, 0.0, -0.02, -0.4], dtype=np.float32),
    }
    for index in range(count):
        task = index % len(instructions)
        frac = (index % 18) / 17.0
        phase = int(frac * config.phase_bins)
        state = np.asarray(
            [
                np.sin(frac * np.pi),
                np.cos(frac * np.pi),
                frac,
                frac * frac,
                1.0 - frac,
                (-1.0) ** index * 0.05,
                0.2,
                -0.2,
            ],
            dtype=np.float32,
        )
        shared = np.asarray([0.02 * phase, -0.01 * phase, 0.03 * frac, 0.01, -0.01, 0.0, 0.25], dtype=np.float32)
        action = np.clip(shared + semantic_prototypes[instructions[task]] + 0.03 * state[:7], -1.0, 1.0)
        rows.append(
            SACFExample(
                state=[float(x) for x in state],
                action=[float(x) for x in action],
                instruction=instructions[task],
                family=families[task],
                task_key=f"{families[task]}/task_{task}",
                step_fraction=float(frac),
                phase_index=phase,
            )
        )
    return rows


def test_semantic_features_are_stable_and_not_empty() -> None:
    a = semantic_hash_features("pick up the ketchup and place it in the basket", width=16)
    b = semantic_hash_features("pick up the ketchup and place it in the basket", width=16)
    c = semantic_hash_features("pick up the milk and place it in the basket", width=16)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.linalg.norm(a) > 0.9


def test_feature_widths_and_filename_mapping() -> None:
    config = SACFConfig()
    shared, semantic = make_sacf_features(
        [0.0] * config.state_dim,
        instruction="pick up the ketchup and place it in the basket",
        family="libero_object",
        step_fraction=0.2,
        config=config,
    )
    assert len(shared) == config.shared_input_dim
    assert len(semantic) == config.semantic_input_dim
    assert instruction_to_demo_filename("Pick up the ketchup and place it in the basket") == "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5"


def test_privileged_fields_blocked() -> None:
    with pytest.raises(ValueError, match="object_pose"):
        assert_no_privileged_inference_fields(["state", "object_pose"])
    assert_no_privileged_inference_fields(["state", "instruction", "phase"])


def test_training_reduces_losses_and_actions_are_distinct() -> None:
    config = SACFConfig(hidden_dim=32)
    examples = _examples()
    full, full_stats = train_sacf_policy(examples, config=config, epochs=70, lr=0.004, seed=3)
    plain, plain_stats = train_plain_bc_prefix(examples, config=config, epochs=70, lr=0.004, seed=4)
    assert full_stats["loss_decreased"]
    assert full_stats["bc_loss_decreased"]
    assert full_stats["counterfactual_pair_count"] > 0
    assert plain_stats["loss_decreased"]
    sample = examples[0]
    full_action, diag = predict_sacf_action(
        full,
        state=sample.state,
        instruction=sample.instruction,
        family=sample.family,
        step_fraction=sample.step_fraction,
    )
    plain_action = predict_plain_action(
        plain,
        state=sample.state,
        instruction=sample.instruction,
        family=sample.family,
        step_fraction=sample.step_fraction,
    )
    mean_action = task_phase_mean_action(full_stats, task_key=sample.task_key, step_fraction=sample.step_fraction, config=config)
    assert full_action.shape == (config.action_dim,)
    assert plain_action.shape == (config.action_dim,)
    assert mean_action.shape == (config.action_dim,)
    assert diag["semantic_component_norm"] > 0.01
    assert np.linalg.norm(full_action - mean_action) > 0.001
