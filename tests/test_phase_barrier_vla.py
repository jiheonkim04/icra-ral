import numpy as np

from tca_map.smolvla.phase_barrier_vla import (
    BarrierRecord,
    action_feature_dict,
    fit_phase_barrier,
    infer_phase_from_step,
    pre_vla_style_halt_proxy,
    project_action_with_barrier,
    simple_global_damping,
)


def _action(scale=1.0):
    return np.asarray([[0.2 * scale, -0.1 * scale, 0.05 * scale, 0.3 * scale, 0.0, -0.2 * scale, -1.0]])


def test_fit_phase_barrier_separates_simple_training_records():
    good = action_feature_dict(_action(0.5), eef=[0.0, 0.0, 0.2], step_fraction=0.2)
    bad = action_feature_dict(_action(3.0), eef=[0.0, 0.0, 0.2], step_fraction=0.2)
    model = fit_phase_barrier(
        [
            BarrierRecord("contact", good, 1.0),
            BarrierRecord("contact", bad, -1.0),
            BarrierRecord("transport", good, 1.0),
            BarrierRecord("transport", bad, -1.0),
        ]
    )

    assert model.score(good, "contact") > model.score(bad, "contact")
    assert model.to_json()["use_phase"] is True


def test_project_action_with_barrier_changes_only_risky_actions():
    action = _action(1.0)

    safe = project_action_with_barrier(action, margin=1.0, phase="contact")
    risky = project_action_with_barrier(action, margin=-2.0, phase="contact")

    assert np.allclose(safe, action)
    assert np.linalg.norm(risky[:, :2]) < np.linalg.norm(action[:, :2])
    assert risky[0, 2] > action[0, 2]


def test_baseline_action_transforms_are_distinct():
    action = _action(1.0)

    damped = simple_global_damping(action, scale=0.5)
    halted = pre_vla_style_halt_proxy(action, margin=-1.0)

    assert np.linalg.norm(damped[:, :6]) < np.linalg.norm(action[:, :6])
    assert np.allclose(halted[:, :6], 0.0)
    assert halted[0, 6] == action[0, 6]


def test_phase_inference_is_deterministic_by_episode_fraction():
    assert infer_phase_from_step(0, 100) == "approach"
    assert infer_phase_from_step(30, 100) == "contact"
    assert infer_phase_from_step(60, 100) == "transport"
    assert infer_phase_from_step(90, 100) == "placement"
