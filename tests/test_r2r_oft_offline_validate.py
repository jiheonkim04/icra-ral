from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.r2r_oft.offline_validate import _summarize_predictions, select_fixed_validation_chunks
from tca_map.r2r_oft.training_spec import build_epoch5_training_spec


def _write_demo(group, name: str, *, offset: float) -> None:
    demo = group.create_group(name)
    steps = 20
    states = np.zeros((steps, 47), dtype=np.float64)
    states[:, 10:13] = np.array([-0.20 + offset, 0.30, 0.966])
    states[:, 17:20] = np.array([0.20 + offset, 0.30, 0.966])
    states[6:, 10:13] = np.array([0.0 + offset, 0.0, 1.0])
    states[10:, 17:20] = np.array([0.03 + offset, 0.0, 1.0])
    demo.create_dataset("states", data=states)


def _write_hdf5(path: Path, demos: int = 50) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_demo(data, f"demo_{index}", offset=0.001 * index)


def test_select_fixed_validation_chunks_uses_validation_split(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task8.hdf5"
    _write_hdf5(hdf5_path)
    spec = build_epoch5_training_spec()

    chunks = select_fixed_validation_chunks(hdf5_path=hdf5_path, spec=spec, num_chunks=8)

    assert len(chunks) == 8
    assert all(chunk["demo_index"] >= 40 for chunk in chunks)
    assert [chunk["phase_count_on"] for chunk in chunks[:4]] == [1, 0, 1, 2]


def test_summarize_predictions_reports_phase_l1() -> None:
    rows = [
        {"phase_count_on": 0, "l1": 0.2},
        {"phase_count_on": 1, "l1": 0.1},
        {"phase_count_on": 1, "l1": 0.3},
        {"phase_count_on": 2, "l1": 0.4},
    ]

    summary = _summarize_predictions(rows)

    assert summary["count"] == 4
    assert summary["phase_1_count"] == 2
    assert summary["phase_1_mean_l1"] == 0.2
