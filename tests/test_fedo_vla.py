from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla.fedo_vla import (
    FEDOConfig,
    apex_feedback_proxy_action,
    apply_control_fault,
    assert_no_privileged_inference_fields,
    build_fedo_examples,
    inverse_fault_command,
    load_fedo_checkpoint,
    make_fedo_features,
    predict_fedo_command,
    save_fedo_checkpoint,
    static_inverse_gain_action,
    train_fedo_compensator,
)


def _actions(count: int = 24) -> list[np.ndarray]:
    rows = []
    for index in range(count):
        frac = index / max(1, count - 1)
        rows.append(
            np.asarray(
                [
                    0.25 * np.sin(frac * np.pi),
                    -0.20 * np.cos(frac * np.pi),
                    0.12 * (frac - 0.5),
                    0.03,
                    -0.02,
                    0.01,
                    0.8 if frac < 0.45 else -0.8,
                ],
                dtype=np.float32,
            )
        )
    return rows


def test_fedo_features_have_expected_width():
    config = FEDOConfig()
    features = make_fedo_features(
        np.zeros(7, dtype=np.float32),
        previous_command=np.ones(7, dtype=np.float32) * 0.1,
        previous_realized=np.ones(7, dtype=np.float32) * 0.05,
        step_fraction=0.35,
        task_key="libero_spatial/task_4",
        config=config,
    )
    assert len(features) == config.input_dim


def test_fault_inverse_reconstructs_intended_action():
    intended = np.asarray([0.25, -0.15, 0.1, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    command = inverse_fault_command(intended, identity=20260713, step_fraction=0.4)
    realized = apply_control_fault(command, identity=20260713, step_fraction=0.4)
    assert np.linalg.norm(realized - intended) < 1e-5


def test_static_inverse_is_not_identity_under_fault():
    intended = np.asarray([0.25, -0.15, 0.1, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    command = static_inverse_gain_action(intended, step_fraction=0.4)
    assert np.linalg.norm(command - intended) > 1e-3


def test_apex_feedback_proxy_uses_previous_error():
    intended = np.zeros(7, dtype=np.float32)
    previous_command = np.ones(7, dtype=np.float32) * 0.4
    previous_realized = np.ones(7, dtype=np.float32) * 0.1
    command = apex_feedback_proxy_action(
        intended,
        previous_command=previous_command,
        previous_realized=previous_realized,
        feedback_gain=0.5,
        smoothing=0.0,
    )
    assert np.allclose(command, np.ones(7, dtype=np.float32) * 0.15)


def test_fedo_training_save_load_predicts_residual(tmp_path: Path):
    config = FEDOConfig(hidden_dim=32)
    examples = build_fedo_examples(
        _actions(32),
        identities=[20260713, 20260714, 20260715],
        task_keys=["libero_spatial/task_4", "libero_10/task_4"],
        config=config,
    )
    model, stats = train_fedo_compensator(examples, config=config, epochs=80, lr=3e-3, seed=7)
    assert stats["loss_decreased"]
    path = tmp_path / "fedo.pt"
    save_fedo_checkpoint(path, model, stats)
    loaded, loaded_stats = load_fedo_checkpoint(path)
    assert loaded_stats["example_count"] == len(examples)
    action = _actions(1)[0]
    command = predict_fedo_command(loaded, examples[0].features, action)
    assert command.shape == (7,)
    assert np.isfinite(command).all()


def test_privileged_inference_guard():
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["current_action", "sim_state"])
