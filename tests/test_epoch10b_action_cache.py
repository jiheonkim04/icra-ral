import numpy as np
import pytest

from scripts.run_epoch10b_action_cache import (
    EXPECTED_DEVELOPMENT_ROWS,
    RESPONSIVE_FRACTIONS,
    CacheError,
    baseline_metrics,
    cache_origin_and_offset,
    state_is_selected,
    unique_increasing_phase_frames,
)


@pytest.mark.parametrize(
    ("frame", "origin", "offset"),
    [(0, 0, 0), (49, 0, 49), (50, 50, 0), (137, 100, 37), (299, 250, 49)],
)
def test_queue_origin_and_offset(frame: int, origin: int, offset: int) -> None:
    assert cache_origin_and_offset(frame) == (origin, offset)


def test_negative_frame_is_rejected() -> None:
    with pytest.raises(CacheError, match="Negative frame"):
        cache_origin_and_offset(-1)


def test_mechanics_only_selector_excludes_only_object_contact_stratum() -> None:
    selected = {
        (suite, phase)
        for suite, phases in RESPONSIVE_FRACTIONS.items()
        for phase in phases
        if state_is_selected(suite, phase)
    }
    assert len(selected) == 15
    assert ("libero_object", "contact_grasp_release") not in selected
    assert EXPECTED_DEVELOPMENT_ROWS == 3840


def test_outcome_blind_backward_collision_rule_makes_frames_unique() -> None:
    frames = unique_increasing_phase_frames(79)
    assert frames == tuple(sorted(set(frames)))
    assert len(frames) == 8


def test_equal_weight_arm_gripper_baseline() -> None:
    expert = np.zeros(7)
    candidate = np.array([1, 1, 1, 1, 1, 1, 2], dtype=float)
    metrics = baseline_metrics(candidate, expert, np.ones(7), 0.5)
    assert metrics["raw_mse"] == pytest.approx(10 / 7)
    assert metrics["arm_gripper_equal_weight_mse"] == pytest.approx(2.5)
    assert metrics["phase_state_criticality_weighted_normalized_mse"] == pytest.approx(5 / 7)
