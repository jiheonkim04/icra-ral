from __future__ import annotations

import numpy as np

from tca_map.xvla_task6.gradient_smoke import select_mug_done_pudding_clip_start, task6_phase_labels


def test_task6_phase_labels_detect_mug_done_then_pudding_remaining() -> None:
    states = np.zeros((4, 71), dtype=np.float64)
    plate = np.array([0.1, 0.0, 0.4])
    mug_far = plate + np.array([0.2, 0.0, 0.0])
    mug_on = plate + np.array([0.01, 0.0, 0.0])
    pudding_far = plate + np.array([-0.2, 0.11, 0.0])
    pudding_right = plate + np.array([0.0, 0.11, 0.0])
    states[:, 24:27] = plate
    states[:, 10:13] = mug_far
    states[:, 31:34] = pudding_far
    states[1:, 10:13] = mug_on
    states[3:, 31:34] = pudding_right

    labels = task6_phase_labels(
        states,
        mug_plate_xy_threshold=0.05,
        pudding_abs_dx_threshold=0.07,
        pudding_dy_min=0.08,
        pudding_dy_max=0.16,
    )

    assert labels["completed_count"].tolist() == [0, 1, 1, 2]
    assert labels["mug_done_pudding_remaining"].tolist() == [False, True, True, False]


def test_select_mug_done_pudding_clip_start_prefers_viable_phase_index() -> None:
    states = np.zeros((20, 71), dtype=np.float64)
    plate = np.array([0.1, 0.0, 0.4])
    states[:, 24:27] = plate
    states[:, 10:13] = plate + np.array([0.2, 0.0, 0.0])
    states[:, 31:34] = plate + np.array([-0.2, 0.11, 0.0])
    states[5:, 10:13] = plate + np.array([0.01, 0.0, 0.0])
    states[15:, 31:34] = plate + np.array([0.0, 0.11, 0.0])

    start = select_mug_done_pudding_clip_start(
        states,
        mug_plate_xy_threshold=0.05,
        pudding_abs_dx_threshold=0.07,
        pudding_dy_min=0.08,
        pudding_dy_max=0.16,
        clip_steps=10,
    )

    assert start == 5
