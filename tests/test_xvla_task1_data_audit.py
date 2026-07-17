from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tca_map.xvla_task1.data_audit import Task1BasketAuditConfig, audit_task1_basket_data, init_state_sha256


def _write_demo(group, name: str, *, offset: float, first_target: str = "cream") -> np.ndarray:
    demo = group.create_group(name)
    steps = 24
    states = np.zeros((steps, 123), dtype=np.float64)
    actions = np.zeros((steps, 7), dtype=np.float64)
    rewards = np.zeros((steps,), dtype=np.uint8)
    dones = np.zeros((steps,), dtype=np.uint8)
    basket = np.array([0.0 + offset, 0.25, 0.43])
    cream_initial = np.array([0.30 + offset, -0.15, 0.45])
    butter_initial = np.array([-0.25 + offset, 0.05, 0.45])
    cream_final = basket + np.array([0.01, -0.01, 0.03])
    butter_final = basket + np.array([-0.01, 0.01, 0.04])
    states[:, 59:62] = basket
    states[:, 17:20] = cream_initial
    states[:, 52:55] = butter_initial
    if first_target == "cream":
        states[8:, 17:20] = cream_final
        states[17:, 52:55] = butter_final
    else:
        states[8:, 52:55] = butter_final
        states[17:, 17:20] = cream_final
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
            inits.append(
                _write_demo(
                    data,
                    f"demo_{index}",
                    offset=0.001 * index,
                    first_target="cream" if index % 2 == 0 else "butter",
                )
            )
    return inits


def test_task1_basket_audit_passes_balanced_synthetic_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task1.hdf5"
    _write_hdf5(hdf5_path, demos=22)

    report = audit_task1_basket_data(
        Task1BasketAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=16,
            chunk_size=4,
            basket_xy_threshold=0.08,
        )
    )

    assert report["passes_data_health_gate"] is True
    assert report["split_summary"]["train"]["phase_chunk_counts"]["1"] > 0
    assert report["split_summary"]["validation"]["phase_chunk_counts"]["1"] > 0
    assert report["deployment_input_policy"]["privileged_state_at_inference"] is False


def test_task1_basket_audit_detects_residual_initial_state_overlap(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task1.hdf5"
    inits = _write_hdf5(hdf5_path, demos=8)
    overlap = init_state_sha256(inits[0])

    report = audit_task1_basket_data(
        Task1BasketAuditConfig(
            hdf5_path=hdf5_path,
            train_demo_count=6,
            chunk_size=4,
            residual_initial_state_sha256=(overlap,),
        )
    )

    assert report["passes_data_health_gate"] is False
    assert report["gate_checks"]["initial_states_do_not_overlap_residual_failures"] is False
    assert report["residual_overlap"]["overlap_hashes"] == [overlap]
