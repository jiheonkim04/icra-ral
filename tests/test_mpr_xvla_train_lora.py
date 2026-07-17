from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tca_map.xvla_task6.train_lora import (
    _arm_by_id,
    _phase_for_step,
    build_phase_clip_index,
    materialize_xvla_clip,
    select_clip_for_step,
)
from tca_map.xvla_task6.training_spec import build_mpr_xvla_training_spec


def _write_phase_demo(group, name: str, *, offset: float = 0.0, steps: int = 24) -> None:
    demo = group.create_group(name)
    states = np.zeros((steps, 71), dtype=np.float64)
    plate = np.array([0.10 + offset, 0.0, 0.44])
    states[:, 24:27] = plate
    states[:, 10:13] = plate + np.array([0.20, -0.10, 0.0])
    states[:, 31:34] = plate + np.array([-0.20, 0.11, 0.0])
    states[6:, 10:13] = plate + np.array([0.01, 0.0, 0.0])
    states[14:, 31:34] = plate + np.array([0.0, 0.11, 0.0])
    demo.create_dataset("states", data=states)
    actions = np.zeros((steps, 7), dtype=np.float64)
    actions[:, 6] = np.linspace(-1.0, 1.0, steps)
    demo.create_dataset("actions", data=actions)
    robot_states = np.zeros((steps, 9), dtype=np.float64)
    robot_states[:, 2:5] = np.array([0.1, 0.2, 0.3])
    robot_states[:, 5:9] = np.array([1.0, 0.0, 0.0, 0.0])
    demo.create_dataset("robot_states", data=robot_states)
    obs = demo.create_group("obs")
    obs.create_dataset("agentview_rgb", data=np.zeros((steps, 8, 8, 3), dtype=np.uint8))
    obs.create_dataset("eye_in_hand_rgb", data=np.zeros((steps, 8, 8, 3), dtype=np.uint8))


def _write_hdf5(path: Path, demos: int = 4) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_phase_demo(data, f"demo_{index}", offset=0.001 * index)


def test_mpr_arm_lookup_rejects_unfrozen_arm() -> None:
    spec = build_mpr_xvla_training_spec()

    assert _arm_by_id(spec, "mpr_xvla_rank8_lambda2_lr1e4_steps64")["role"] == "primary_selected_method"
    with pytest.raises(ValueError):
        _arm_by_id(spec, "third_unfrozen_mpr_xvla_arm")


def test_mpr_phase_cycle_matches_frozen_order() -> None:
    cycle = [1, 0, 1, 2]

    assert [_phase_for_step(cycle, index) for index in range(8)] == [1, 0, 1, 2, 1, 0, 1, 2]


def test_task6_build_phase_clip_index_and_select_clip(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task6.hdf5"
    _write_hdf5(hdf5_path)
    spec = build_mpr_xvla_training_spec()
    layout = spec["data"]["phase_state_layout"]

    grouped = build_phase_clip_index(
        hdf5_path,
        demo_indices=[0, 1, 2],
        clip_steps=8,
        mug_plate_xy_threshold=layout["mug_plate_xy_threshold"],
        pudding_abs_dx_threshold=layout["pudding_abs_dx_threshold"],
        pudding_dy_min=layout["pudding_dy_min"],
        pudding_dy_max=layout["pudding_dy_max"],
    )

    assert all(grouped[phase] for phase in (0, 1, 2))
    rng = np.random.default_rng(20260717)
    selected = [
        select_clip_for_step(grouped, cycle=[1, 0, 1, 2], step_index_zero_based=index, rng=rng)
        for index in range(4)
    ]
    assert [item["phase_count_in_basket"] for item in selected] == [1, 0, 1, 2]


def test_task6_materialize_xvla_clip_writes_reader_meta(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task6.hdf5"
    _write_hdf5(hdf5_path, demos=1)
    spec = build_mpr_xvla_training_spec()
    clip = {
        "demo_index": 0,
        "demo_name": "demo_0",
        "source_start_index": 5,
        "source_end_index": 17,
        "phase_count_in_basket": 0,
    }

    materialized = materialize_xvla_clip(hdf5_path, tmp_path / "adapter", clip, spec)

    assert Path(materialized["meta_path"]).is_file()
    assert Path(materialized["clip_hdf5"]).is_file()
    assert materialized["clip_steps"] == 12
    assert materialized["abs_action_6d_shape"] == [12, 10]
    assert materialized["completed_count_first"] == 0
