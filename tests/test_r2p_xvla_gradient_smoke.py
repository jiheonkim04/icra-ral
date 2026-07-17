from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_spatial_task5.gradient_smoke import (
    materialize_phase_rich_clip,
    select_phase_rich_clip_start,
    task5_phase_labels,
)


def _write_phase_demo(group, name: str, *, steps: int = 24) -> None:
    demo = group.create_group(name)
    states = np.zeros((steps, 45), dtype=np.float64)
    states[:, 31:34] = np.array([0.0, 0.0, 0.0])
    states[:, 38:41] = np.array([1.0, 0.0, 0.0])
    states[0:6, 10:13] = np.array([0.0, 0.0, 0.0])
    states[6:14, 10:13] = np.array([0.5, 0.5, 0.0])
    states[14:, 10:13] = np.array([1.0, 0.0, 0.0])
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


def test_task5_phase_labels_find_source_transit_and_target() -> None:
    states = np.zeros((6, 45), dtype=np.float64)
    states[:, 31:34] = [0.0, 0.0, 0.0]
    states[:, 38:41] = [1.0, 0.0, 0.0]
    states[0:2, 10:13] = [0.0, 0.0, 0.0]
    states[2:4, 10:13] = [0.4, 0.4, 0.0]
    states[4:6, 10:13] = [1.0, 0.0, 0.0]

    labels = task5_phase_labels(states)

    assert list(labels["phase"]) == [
        "source_on_ramekin",
        "source_on_ramekin",
        "transit",
        "transit",
        "target_on_plate",
        "target_on_plate",
    ]


def test_select_phase_rich_clip_prefers_transit_start() -> None:
    states = np.zeros((24, 45), dtype=np.float64)
    states[:, 31:34] = [0.0, 0.0, 0.0]
    states[:, 38:41] = [1.0, 0.0, 0.0]
    states[0:6, 10:13] = [0.0, 0.0, 0.0]
    states[6:14, 10:13] = [0.4, 0.4, 0.0]
    states[14:, 10:13] = [1.0, 0.0, 0.0]

    assert select_phase_rich_clip_start(states, clip_steps=8) == 6


def test_materialize_phase_rich_clip_writes_reader_meta(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    with h5py.File(hdf5_path, "w") as h5:
        _write_phase_demo(h5.create_group("data"), "demo_0")

    materialized = materialize_phase_rich_clip(hdf5_path, tmp_path / "adapter", demo_name="demo_0", clip_steps=12)

    assert Path(materialized["meta_path"]).is_file()
    assert Path(materialized["clip_hdf5"]).is_file()
    assert materialized["clip_steps"] == 12
    assert materialized["abs_action_6d_shape"] == [12, 10]
    assert materialized["phase_counts"]["transit"] > 0
    assert materialized["phase_weight_mean"] > 1.0
