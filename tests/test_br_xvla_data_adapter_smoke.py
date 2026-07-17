from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

from tca_map.xvla_task1 import data_adapter_smoke
from tca_map.xvla_task1.data_adapter_smoke import build_abs_action_6d, materialize_xvla_demo


def test_build_abs_action_6d_uses_robot_state_pose_and_gripper() -> None:
    robot_states = np.zeros((2, 9), dtype=np.float64)
    robot_states[:, 2:5] = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    robot_states[:, 5:9] = [1.0, 0.0, 0.0, 0.0]
    actions = np.zeros((2, 7), dtype=np.float64)
    actions[:, 6] = [-1.0, 1.0]

    out = build_abs_action_6d(robot_states, actions)

    assert out.shape == (2, 10)
    np.testing.assert_allclose(out[:, :3], robot_states[:, 2:5])
    np.testing.assert_allclose(out[:, 3:9], np.array([[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]))
    np.testing.assert_allclose(out[:, 9], [-1.0, 1.0])


def test_materialize_xvla_demo_writes_official_libero_keys(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float64))
        robot_states = np.zeros((3, 9), dtype=np.float64)
        robot_states[:, 5] = 1.0
        demo.create_dataset("robot_states", data=robot_states)
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))

    output = tmp_path / "converted.hdf5"
    row = materialize_xvla_demo(source, "demo_0", output)

    assert row["abs_action_6d_shape"] == [3, 10]
    with h5py.File(output, "r") as handle:
        assert set(handle.keys()) == {"abs_action_6d", "agentview_rgb", "eye_in_hand_rgb", "language_instruction"}
        assert handle["language_instruction"][()].decode()


def test_mmengine_fileio_shim_has_importlib_specs(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "mmengine", raising=False)
    monkeypatch.delitem(sys.modules, "mmengine.fileio", raising=False)
    original_find_spec = data_adapter_smoke.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "mmengine":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(data_adapter_smoke.importlib.util, "find_spec", fake_find_spec)

    assert data_adapter_smoke._install_mmengine_fileio_shim_if_needed() is True
    assert sys.modules["mmengine"].__spec__ is not None
    assert sys.modules["mmengine.fileio"].__spec__ is not None
