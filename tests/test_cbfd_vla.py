import numpy as np
import pytest

from tca_map.smolvla.cbfd_vla import (
    CBFDConfig,
    CBFDExample,
    assert_no_privileged_inference_fields,
    make_cbfd_features,
    memory_action,
    predict_cbfd_action,
    stage_a_decision,
    train_cbfd_policy,
)


def _examples() -> list[CBFDExample]:
    rows: list[CBFDExample] = []
    for index in range(40):
        frac = (index % 10) / 9.0
        task = "libero_spatial/task_4" if index % 2 == 0 else "libero_10/task_4"
        code = -1.0 if "spatial" in task else 1.0
        state = np.asarray([frac, code, np.sin(frac), np.cos(frac), 0.1 * index, 0.0, 0.0, 1.0], dtype=np.float32)
        teacher_action = np.asarray([0.3 * code, frac, -frac, 0.0, 0.0, 0.1, -0.5], dtype=np.float32)
        rows.append(
            CBFDExample(
                state=state.tolist(),
                action=teacher_action.tolist(),
                task_key=task,
                step_fraction=frac,
                source="teacher",
                failure_weight=1.0,
            )
        )
        retention_action = np.asarray([0.05 * code, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
        rows.append(
            CBFDExample(
                state=(state + 0.01).tolist(),
                action=retention_action.tolist(),
                task_key=task,
                step_fraction=frac,
                source="retention",
                failure_weight=1.0,
            )
        )
    return rows


def test_features_have_expected_width_and_task_sensitivity() -> None:
    config = CBFDConfig()
    left = make_cbfd_features([0.0] * 8, step_fraction=0.1, task_key_value="libero_spatial/task_4", config=config)
    right = make_cbfd_features([0.0] * 8, step_fraction=0.1, task_key_value="libero_10/task_4", config=config)

    assert len(left) == config.input_dim
    assert left != right


def test_cbfd_training_decreases_loss_and_predicts_finite_action() -> None:
    config = CBFDConfig(hidden_dim=24)
    model, stats = train_cbfd_policy(_examples(), config=config, epochs=30, lr=3e-3, seed=4)
    action = predict_cbfd_action(
        model,
        state=[0.5, -1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 1.0],
        step_fraction=0.5,
        task_key_value="libero_spatial/task_4",
    )

    assert stats["loss_decreased"]
    assert stats["teacher_example_count"] == 40
    assert stats["retention_example_count"] == 40
    assert action.shape == (7,)
    assert np.isfinite(action).all()


def test_memory_action_uses_only_teacher_rows() -> None:
    config = CBFDConfig()
    action, diag = memory_action(
        _examples(),
        state=[0.5, -1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 1.0],
        step_fraction=0.5,
        task_key_value="libero_spatial/task_4",
        config=config,
    )

    assert action.shape == (7,)
    assert diag["memory_score"] >= 0.0
    assert diag["memory_task_key"] == "libero_spatial/task_4"


def test_stage_a_decision_kills_only_clear_bad_cases() -> None:
    summary = {
        "mechanism_active": True,
        "by_variant": {
            "cbfd_full": {"successes": 0, "total": 10, "task_balanced_success_rate": 0.0},
            "direct_distill_proxy": {"successes": 4, "total": 10, "task_balanced_success_rate": 0.4},
        },
    }

    assert stage_a_decision(summary) == "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"


def test_privileged_fields_rejected() -> None:
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["state", "teacher_action"])
