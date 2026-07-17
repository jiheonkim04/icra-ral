from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_task6.offline_validate import (
    _summarize_action_delta,
    _summarize_policy_rows,
    select_fixed_validation_clips,
)
from tca_map.xvla_task6.training_spec import build_mpr_xvla_training_spec


def _write_demo(group, name: str, *, offset: float = 0.0, steps: int = 24) -> None:
    demo = group.create_group(name)
    states = np.zeros((steps, 71), dtype=np.float64)
    plate = np.array([0.10 + offset, 0.0, 0.44])
    states[:, 24:27] = plate
    states[:, 10:13] = plate + np.array([0.20, -0.10, 0.0])
    states[:, 31:34] = plate + np.array([-0.20, 0.11, 0.0])
    states[6:, 10:13] = plate + np.array([0.01, 0.0, 0.0])
    states[14:, 31:34] = plate + np.array([0.0, 0.11, 0.0])
    demo.create_dataset("states", data=states)


def _write_hdf5(path: Path, demos: int = 50) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_demo(data, f"demo_{index}", offset=0.0001 * index)


def test_task6_select_fixed_validation_clips_uses_validation_split(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task6.hdf5"
    _write_hdf5(hdf5_path)
    spec = build_mpr_xvla_training_spec()

    clips = select_fixed_validation_clips(hdf5_path=hdf5_path, spec=spec, num_chunks=8, clip_steps=8)

    assert len(clips) == 8
    assert all(int(clip["demo_index"]) >= 40 for clip in clips)
    assert [clip["phase_count_in_basket"] for clip in clips[:4]] == [1, 0, 1, 2]
    assert [clip["validation_index"] for clip in clips] == list(range(8))


def test_task6_summarize_policy_rows_reports_phase_losses() -> None:
    rows = [
        {"phase_count_in_basket": 0, "loss_total": 0.2, "weighted_loss": 0.2},
        {"phase_count_in_basket": 1, "loss_total": 0.1, "weighted_loss": 0.3},
        {"phase_count_in_basket": 1, "loss_total": 0.3, "weighted_loss": 0.9},
        {"phase_count_in_basket": 2, "loss_total": 0.4, "weighted_loss": 0.4},
    ]

    summary = _summarize_policy_rows(rows)

    assert summary["count"] == 4
    assert summary["phase_1_count"] == 2
    assert summary["phase_1_mean_loss_total"] == 0.2
    assert summary["all_losses_finite"] is True


def test_task6_summarize_action_delta_against_prior() -> None:
    prior_rows = [
        {"validation_index": 0, "generated_action": [[[0.0, 0.0], [1.0, 1.0]]]},
        {"validation_index": 1, "generated_action": [[[1.0, 1.0], [2.0, 2.0]]]},
    ]
    rows = [
        {"validation_index": 0, "generated_action": [[[0.5, 0.0], [1.0, 1.5]]]},
        {"validation_index": 1, "generated_action": [[[1.0, 0.0], [3.0, 2.0]]]},
    ]

    delta = _summarize_action_delta(rows, prior_rows)

    assert delta["fixed_chunk_mean_abs_action_delta"] == 0.375
    assert delta["fixed_chunk_max_abs_action_delta"] == 1.0
