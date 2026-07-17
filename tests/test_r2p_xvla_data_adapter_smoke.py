from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_spatial_task5 import data_adapter_smoke
from tca_map.xvla_spatial_task5.data_adapter_smoke import TASK_DESCRIPTION, materialize_xvla_demo, run_data_adapter_smoke


def _write_source(path: Path, demos: int = 1) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            steps = 6
            actions = np.zeros((steps, 7), dtype=np.float64)
            actions[:, 6] = [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
            robot_states = np.zeros((steps, 9), dtype=np.float64)
            robot_states[:, 2:5] = np.array([0.1 + index, 0.2, 0.3])
            robot_states[:, 5] = 1.0
            states = np.zeros((steps, 45), dtype=np.float64)
            states[:, 31:34] = np.array([0.0, 0.0, 0.0])
            states[:, 38:41] = np.array([1.0, 0.0, 0.0])
            states[0:2, 10:13] = np.array([0.0, 0.0, 0.0])
            states[2:4, 10:13] = np.array([0.5, 0.5, 0.0])
            states[4:6, 10:13] = np.array([1.0, 0.0, 0.0])
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((steps, 8, 8, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((steps, 8, 8, 3), dtype=np.uint8))
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("robot_states", data=robot_states)
            demo.create_dataset("states", data=states)


def test_task5_materialize_xvla_demo_writes_instruction_keys_and_phase_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    _write_source(source)

    output = tmp_path / "converted.hdf5"
    row = materialize_xvla_demo(source, "demo_0", output)

    assert row["abs_action_6d_shape"] == [6, 10]
    assert row["phase_step_counts"] == {
        "source_on_ramekin": 2,
        "target_on_plate": 2,
        "transit": 2,
    }
    with h5py.File(output, "r") as handle:
        assert set(handle.keys()) == {"abs_action_6d", "agentview_rgb", "eye_in_hand_rgb", "language_instruction"}
        assert handle["language_instruction"][()].decode() == TASK_DESCRIPTION


def test_task5_data_adapter_smoke_keeps_preoptimizer_gates_closed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.hdf5"
    _write_source(source, demos=2)

    def fake_reader(_xvla_root: Path, _meta_path: Path):
        return {
            "action": {"shape": [30, 20], "dtype": "torch.float32", "finite": True},
            "proprio": {"shape": [20], "dtype": "torch.float32", "finite": True},
            "image_input": {"shape": [3, 3, 224, 224], "dtype": "torch.float32", "finite": True},
            "domain_id": {"shape": [], "dtype": "torch.int64", "finite": True},
        }

    monkeypatch.setattr(data_adapter_smoke, "smoke_xvla_reader", fake_reader)

    output_dir = tmp_path / "adapter_smoke"
    report = run_data_adapter_smoke(source, output_dir, Path("/unused/xvla"), ["demo_1"])

    assert report["passed"] is True
    assert report["decision"] == "R2P_XVLA_DATA_ADAPTER_SMOKE_PASS"
    assert report["policy"]["training_happened"] is False
    assert report["policy"]["optimizer_step_happened"] is False
    assert report["policy"]["checkpoint_written"] is False
    assert report["policy"]["model_loaded"] is False
    assert report["phase_coverage"] == {
        "source_on_ramekin": 2,
        "target_on_plate": 2,
        "transit": 2,
    }
    assert (output_dir / "result.json").exists()
