from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tca_map.r2r_oft.train_qlora import (
    _arm_by_id,
    _phase_for_step,
    build_phase_chunk_index,
    select_chunk_for_step,
)
from tca_map.r2r_oft.training_spec import build_epoch5_training_spec


def _write_phase_demo(group, name: str, *, offset: float) -> None:
    demo = group.create_group(name)
    steps = 18
    states = np.zeros((steps, 47), dtype=np.float64)
    states[:, 10:13] = np.array([-0.20 + offset, 0.30, 0.966])
    states[:, 17:20] = np.array([0.20 + offset, 0.30, 0.966])
    states[5:, 10:13] = np.array([0.0 + offset, 0.0, 1.0])
    states[12:, 17:20] = np.array([0.03 + offset, 0.0, 1.0])
    demo.create_dataset("states", data=states)


def _write_hdf5(path: Path, demos: int = 6) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_phase_demo(data, f"demo_{index}", offset=0.001 * index)


def test_arm_lookup_rejects_non_frozen_arm() -> None:
    spec = build_epoch5_training_spec()

    assert _arm_by_id(spec, "r2r_oft_rank4_lambda2_lr2e4_steps64")["role"] == "primary_selected_method"
    with pytest.raises(ValueError):
        _arm_by_id(spec, "third_unfrozen_config")


def test_phase_cycle_matches_frozen_order() -> None:
    cycle = [1, 0, 1, 2]

    assert [_phase_for_step(cycle, index) for index in range(8)] == [1, 0, 1, 2, 1, 0, 1, 2]


def test_build_phase_chunk_index_and_select_chunk(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task8.hdf5"
    _write_hdf5(hdf5_path, demos=6)

    grouped = build_phase_chunk_index(
        hdf5_path,
        demo_indices=[0, 1, 2, 3],
        chunk_size=4,
        train_demo_count_for_target_xy=4,
    )

    assert all(grouped[phase] for phase in (0, 1, 2))
    rng = np.random.default_rng(20260717)
    selected = [select_chunk_for_step(grouped, cycle=[1, 0, 1, 2], step_index_zero_based=i, rng=rng) for i in range(4)]
    assert [item["phase_count_on"] for item in selected] == [1, 0, 1, 2]
