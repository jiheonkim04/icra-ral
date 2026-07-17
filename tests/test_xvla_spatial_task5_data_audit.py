from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_spatial_task5.data_audit import (
    SpatialTask5DataAuditConfig,
    audit_spatial_task5_data,
    init_state_sha256,
)


def _write_demo(group, name: str, *, offset: float) -> np.ndarray:
    demo = group.create_group(name)
    steps = 24
    states = np.zeros((steps, 92), dtype=np.float64)
    actions = np.zeros((steps, 7), dtype=np.float64)
    rewards = np.zeros((steps,), dtype=np.uint8)
    dones = np.zeros((steps,), dtype=np.uint8)
    bowl_source = np.array([-0.20 + offset, 0.20, 0.95])
    ramekin = np.array([-0.21 + offset, 0.20, 0.90])
    plate = np.array([0.05 + offset, 0.21, 0.90])
    states[:, 10:13] = bowl_source
    states[:, 31:34] = ramekin
    states[:, 38:41] = plate
    for step in range(8, steps):
        alpha = (step - 8) / max(1, steps - 9)
        states[step, 10:13] = (1.0 - alpha) * bowl_source + alpha * (plate + np.array([0.01, 0.0, 0.01]))
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


def test_spatial_task5_data_audit_passes_synthetic_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    _write_hdf5(hdf5_path, demos=50)

    report = audit_spatial_task5_data(
        SpatialTask5DataAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=40,
            chunk_size=4,
        )
    )

    assert report["passes_data_health_gate"] is True
    assert report["split_summary"]["train"]["phase_chunk_counts"]["source_on_ramekin"] > 0
    assert report["split_summary"]["train"]["phase_chunk_counts"]["target_on_plate"] > 0
    assert report["split_summary"]["validation"]["phase_chunk_counts"]["transit"] > 0
    assert report["deployment_input_policy"]["privileged_state_at_inference"] is False


def test_spatial_task5_data_audit_detects_residual_initial_state_overlap(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    inits = _write_hdf5(hdf5_path, demos=8)
    overlap = init_state_sha256(inits[0])

    report = audit_spatial_task5_data(
        SpatialTask5DataAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=6,
            residual_initial_state_sha256=(overlap,),
        )
    )

    assert report["passes_data_health_gate"] is False
    assert report["gate_checks"]["initial_states_do_not_overlap_residual_failures"] is False
    assert report["residual_overlap"]["overlap_hashes"] == [overlap]
