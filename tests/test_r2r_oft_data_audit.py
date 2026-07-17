from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.r2r_oft.data_audit import R2ROFTAuditConfig, audit_r2r_oft_data, init_state_sha256


def _write_demo(group, name: str, *, offset: float, one_pot_first: bool = True) -> np.ndarray:
    demo = group.create_group(name)
    steps = 20
    states = np.zeros((steps, 47), dtype=np.float64)
    actions = np.zeros((steps, 7), dtype=np.float64)
    rewards = np.zeros((steps,), dtype=np.uint8)
    dones = np.zeros((steps,), dtype=np.uint8)
    init = np.zeros((47,), dtype=np.float64)

    # Initial: both pots off the inferred stove region.
    states[:, 10:13] = np.array([-0.20 + offset, 0.30, 0.966])
    states[:, 17:20] = np.array([0.20 + offset, 0.30, 0.966])

    # Middle: exactly one pot in the stove region.
    if one_pot_first:
        states[6:, 10:13] = np.array([0.0 + offset, 0.0, 1.0])
        states[14:, 17:20] = np.array([0.03 + offset, 0.0, 1.0])
    else:
        states[6:, 17:20] = np.array([0.03 + offset, 0.0, 1.0])
        states[14:, 10:13] = np.array([0.0 + offset, 0.0, 1.0])

    init[:] = states[0]
    actions[:, -1] = np.where(np.arange(steps) < 10, -1.0, 1.0)
    rewards[-1] = 1
    dones[-1] = 1
    demo.attrs["init_state"] = init
    demo.create_dataset("states", data=states)
    demo.create_dataset("actions", data=actions)
    demo.create_dataset("rewards", data=rewards)
    demo.create_dataset("dones", data=dones)
    return init


def _write_hdf5(path: Path, demos: int = 6) -> list[np.ndarray]:
    inits = []
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            inits.append(
                _write_demo(
                    data,
                    f"demo_{index}",
                    offset=0.001 * index,
                    one_pot_first=(index % 2 == 0),
                )
            )
    return inits


def test_audit_passes_on_balanced_synthetic_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task8.hdf5"
    _write_hdf5(hdf5_path, demos=22)

    report = audit_r2r_oft_data(
        R2ROFTAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=16,
            chunk_size=4,
            stove_xy_threshold=0.12,
            stove_z_min=0.98,
        )
    )

    assert report["passes_data_health_gate"] is True
    assert report["split_summary"]["train"]["phase_chunk_counts"]["1"] > 0
    assert report["split_summary"]["validation"]["phase_chunk_counts"]["1"] > 0
    assert report["deployment_input_policy"]["privileged_state_at_inference"] is False


def test_audit_detects_residual_initial_state_overlap(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task8.hdf5"
    inits = _write_hdf5(hdf5_path, demos=6)
    overlap = init_state_sha256(inits[0])

    report = audit_r2r_oft_data(
        R2ROFTAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=4,
            chunk_size=4,
            residual_initial_state_sha256=(overlap,),
        )
    )

    assert report["passes_data_health_gate"] is False
    assert report["gate_checks"]["initial_states_do_not_overlap_residual_failures"] is False
    assert report["residual_overlap"]["overlap_hashes"] == [overlap]
