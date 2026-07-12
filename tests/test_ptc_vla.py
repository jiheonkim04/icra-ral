import numpy as np
import pytest

from tca_map.smolvla.ptc_vla import (
    PTCConfig,
    PTCExample,
    assert_no_privileged_inference_fields,
    load_ptc_checkpoint,
    make_ptc_features,
    mean_action_from_stats,
    phase_from_fraction,
    predict_ptc_action,
    save_ptc_checkpoint,
    train_ptc_policy,
    transition_context,
)


def _examples(count: int = 48) -> list[PTCExample]:
    rows = []
    for index in range(count):
        frac = index / max(1, count - 1)
        state = np.asarray(
            [
                np.sin(frac * np.pi),
                np.cos(frac * np.pi),
                frac,
                frac * frac,
                1.0 - frac,
                (-1.0) ** index * 0.1,
            ],
            dtype=np.float32,
        )
        transition = np.asarray([0.2 * frac, -0.1 * frac, 0.05, 0.0, 0.03 * np.sin(frac), -0.02], dtype=np.float32)
        action = np.asarray([state[0], state[1], transition[0], transition[1], 0.1, -0.1, -0.8], dtype=np.float32) * 0.35
        rows.append(
            PTCExample(
                state=[float(x) for x in state],
                transition=[float(x) for x in transition],
                action=[float(x) for x in action],
                task_key="libero_spatial/task_4" if index % 2 == 0 else "libero_10/task_4",
                step_fraction=frac,
                phase=phase_from_fraction(frac),
            )
        )
    return rows


def test_make_ptc_features_width_and_transition_ablation() -> None:
    config = PTCConfig()
    state = np.arange(6, dtype=np.float32)
    transition = np.ones(6, dtype=np.float32) * 0.25

    full = make_ptc_features(state, transition, step_fraction=0.3, task_key="libero_spatial/task_4", config=config)
    ablation = make_ptc_features(state, transition, step_fraction=0.3, task_key="libero_spatial/task_4", config=config, use_transition=False)

    assert len(full) == config.input_dim
    assert full[:6] == pytest.approx(state.tolist())
    assert full[6:12] == pytest.approx(transition.tolist())
    assert ablation[6:12] == pytest.approx([0.0] * 6)


def test_privileged_guard_blocks_forbidden_fields() -> None:
    with pytest.raises(ValueError, match="privileged"):
        assert_no_privileged_inference_fields(["current_policy_state", "success"])


def test_transition_context_blends_recent_and_prior() -> None:
    config = PTCConfig(transition_blend=0.5)
    context = transition_context(
        current_state=np.asarray([2, 2, 2, 2, 2, 2], dtype=np.float32),
        previous_state=np.asarray([1, 1, 1, 1, 1, 1], dtype=np.float32),
        prior_transition=np.asarray([0, 2, 0, 2, 0, 2], dtype=np.float32),
        config=config,
    )

    assert context.tolist() == pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5])


def test_train_ptc_policy_decreases_loss_and_uses_transition(tmp_path) -> None:
    config = PTCConfig(hidden_dim=32)
    examples = _examples()
    full_model, full_stats = train_ptc_policy(examples, config=config, epochs=80, lr=2e-3, seed=7, use_transition=True)
    ablation_model, ablation_stats = train_ptc_policy(examples, config=config, epochs=80, lr=2e-3, seed=8, use_transition=False)

    assert full_stats["loss_decreased"] is True
    assert ablation_stats["loss_decreased"] is True
    assert full_stats["uses_transition"] is True
    assert ablation_stats["uses_transition"] is False
    assert full_stats["final_loss"] < full_stats["initial_loss"]

    state = examples[10].state
    transition = examples[10].transition
    full_action, full_scale = predict_ptc_action(
        full_model,
        state=state,
        transition=transition,
        step_fraction=examples[10].step_fraction,
        task_key=examples[10].task_key,
    )
    ablated_action, _ = predict_ptc_action(
        ablation_model,
        state=state,
        transition=transition,
        step_fraction=examples[10].step_fraction,
        task_key=examples[10].task_key,
        use_transition=False,
    )

    assert full_action.shape == (7,)
    assert full_scale.shape == (7,)
    assert np.isfinite(full_action).all()
    assert np.linalg.norm(full_action - ablated_action) > 1e-3

    checkpoint = tmp_path / "ptc.pt"
    save_ptc_checkpoint(checkpoint, full_model, full_stats)
    loaded_model, loaded_stats = load_ptc_checkpoint(checkpoint)
    loaded_action, _ = predict_ptc_action(
        loaded_model,
        state=state,
        transition=transition,
        step_fraction=examples[10].step_fraction,
        task_key=examples[10].task_key,
    )
    assert loaded_stats["example_count"] == len(examples)
    assert np.allclose(full_action, loaded_action)


def test_mean_action_stats_fallback_is_finite() -> None:
    config = PTCConfig(hidden_dim=16)
    model, stats = train_ptc_policy(_examples(12), config=config, epochs=10, lr=1e-3, seed=3)
    action = mean_action_from_stats(stats, phase="approach", task_key="unknown", config=model.config)

    assert action.shape == (7,)
    assert np.isfinite(action).all()
