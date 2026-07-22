from __future__ import annotations

import gc

import numpy as np

from scripts.run_epoch10b_fresh_controller_assay import (
    CONTROL_NAMES,
    _index_manifest_states,
    action_queue_audit,
    closing_environment,
    control_actions,
    seed_runtime,
)


class _FakeController:
    def __init__(self) -> None:
        self.action_queue = []


class _FakeRobot:
    def __init__(self) -> None:
        self.controller = _FakeController()


class _FakeEnv:
    closes = 0

    def __init__(self) -> None:
        self.env = self
        self.robots = [_FakeRobot()]
        self.termination_flag = True

    def close(self) -> None:
        type(self).closes += 1


def test_closing_environment_closes_every_branch() -> None:
    _FakeEnv.closes = 0
    audits = []
    for _ in range(2):
        with closing_environment(_FakeEnv) as (env, audit):
            assert env is not None
            audits.append(audit)
        gc.collect()
    assert _FakeEnv.closes == 2
    assert all(audit["close_called"] for audit in audits)


def test_fresh_factory_reinitializes_controller_and_action_queue() -> None:
    first = _FakeEnv()
    second = _FakeEnv()
    assert first.robots[0].controller is not second.robots[0].controller
    first.robots[0].controller.action_queue.append(1)
    assert second.robots[0].controller.action_queue == []
    audit = action_queue_audit(second)
    queue = next(row for row in audit["queue_like_fields"] if row["attribute"] == "action_queue")
    assert queue["length"] == 0


def test_rng_identity_repeats_python_and_numpy_streams() -> None:
    import random

    seed_runtime(1234)
    left = (random.random(), np.random.standard_normal(4))
    seed_runtime(1234)
    right = (random.random(), np.random.standard_normal(4))
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])


def test_controls_cover_nominal_sham_noop_harm_and_symmetric_pairs() -> None:
    expert = np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    controls = control_actions(expert, "approach", "state")
    assert tuple(controls) == CONTROL_NAMES
    np.testing.assert_array_equal(controls["nominal_a"], controls["nominal_b"])
    np.testing.assert_array_equal(controls["nominal_a"], controls["sham"])
    np.testing.assert_allclose(
        controls["small_plus"] + controls["small_minus"],
        2.0 * controls["nominal_a"],
        atol=1e-12,
    )
    assert np.linalg.norm(controls["medium_plus"] - controls["nominal_a"]) > np.linalg.norm(
        controls["small_plus"] - controls["nominal_a"]
    )


def test_contact_harmful_control_flips_gripper_without_illegal_action() -> None:
    expert = np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3, 1.0], dtype=np.float64)
    harmful = control_actions(expert, "contact_grasp_release", "state")["harmful_phase_matched"]
    np.testing.assert_array_equal(harmful[:6], expert[:6])
    assert harmful[6] == -1.0
    assert np.max(np.abs(harmful)) <= 1.0


def test_manifest_index_preserves_reverse_registration_from_index_zero() -> None:
    states = [
        {
            "state_id": "colliding_state",
            "phase_index": 0,
            "reverse_order_duplicate": True,
        },
        {
            "state_id": "colliding_state",
            "phase_index": 1,
            "reverse_order_duplicate": False,
        },
    ]
    indexed = _index_manifest_states(states)
    assert tuple(indexed) == ("colliding_state",)
    assert indexed["colliding_state"]["phase_index"] == 1
    assert indexed["colliding_state"]["reverse_order_duplicate"] is True


def test_manifest_index_does_not_invent_reverse_registration() -> None:
    indexed = _index_manifest_states(
        [
            {"state_id": "state_a", "reverse_order_duplicate": False},
            {"state_id": "state_b", "reverse_order_duplicate": True},
        ]
    )
    assert indexed["state_a"]["reverse_order_duplicate"] is False
    assert indexed["state_b"]["reverse_order_duplicate"] is True
