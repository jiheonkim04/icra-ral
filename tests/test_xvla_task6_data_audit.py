from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_task6.data_audit import Task6SpatialAuditConfig, audit_task6_spatial_data, init_state_sha256


def _write_demo(group, name: str, *, offset: float) -> np.ndarray:
    demo = group.create_group(name)
    steps = 24
    states = np.zeros((steps, 71), dtype=np.float64)
    actions = np.zeros((steps, 7), dtype=np.float64)
    rewards = np.zeros((steps,), dtype=np.uint8)
    dones = np.zeros((steps,), dtype=np.uint8)
    plate = np.array([0.10 + offset, 0.0, 0.44])
    mug_initial = np.array([-0.20 + offset, -0.10, 0.44])
    red_mug = np.array([-0.25 + offset, 0.08, 0.44])
    pudding_initial = np.array([-0.12 + offset, 0.10, 0.45])
    mug_final = plate + np.array([0.01, -0.01, 0.01])
    pudding_final = plate + np.array([-0.02, 0.11, 0.01])
    states[:, 10:13] = mug_initial
    states[:, 17:20] = red_mug
    states[:, 24:27] = plate
    states[:, 31:34] = pudding_initial
    states[8:, 10:13] = mug_final
    states[17:, 31:34] = pudding_final
    actions[:, -1] = np.where(np.arange(steps) < 12, -1.0, 1.0)
    rewards[-1] = 1
    dones[-1] = 1
    init = states[0].copy()
    demo.attrs["init_state"] = init
    demo.create_dataset("states", data=states)
    demo.create_dataset("actions", data=actions)
    demo.create_dataset("rewards", data=rewards)
    demo.create_dataset("dones", data=dones)
    return init


def _write_hdf5(path: Path, demos: int = 22) -> list[np.ndarray]:
    inits = []
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            inits.append(_write_demo(data, f"demo_{index}", offset=0.001 * index))
    return inits


def test_task6_spatial_audit_passes_synthetic_mug_then_pudding_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task6.hdf5"
    _write_hdf5(hdf5_path, demos=22)

    report = audit_task6_spatial_data(
        Task6SpatialAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=16,
            chunk_size=4,
        )
    )

    assert report["passes_data_health_gate"] is True
    assert report["dataset_summary"]["subgoal_order_counts"]["mug_first"] == 22
    assert report["split_summary"]["train"]["mug_done_pudding_remaining_chunks"] > 0
    assert report["split_summary"]["validation"]["mug_done_pudding_remaining_chunks"] > 0
    assert report["deployment_input_policy"]["privileged_state_at_inference"] is False


def test_task6_spatial_audit_detects_residual_initial_state_overlap(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task6.hdf5"
    inits = _write_hdf5(hdf5_path, demos=8)
    overlap = init_state_sha256(inits[0])

    report = audit_task6_spatial_data(
        Task6SpatialAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=6,
            residual_initial_state_sha256=(overlap,),
        )
    )

    assert report["passes_data_health_gate"] is False
    assert report["gate_checks"]["initial_states_do_not_overlap_residual_failures"] is False
    assert report["residual_overlap"]["overlap_hashes"] == [overlap]
