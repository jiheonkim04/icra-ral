from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tca_map.xvla_spatial_task5.train_lora import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_XVLA_ROOT,
    TrainArmConfig,
    _arm_by_id,
    _assert_output_root_allowed,
    _phase_cycle_from_sampler,
    _phase_for_step,
    build_phase_clip_index,
    materialize_xvla_clip,
    run_training_arm,
    select_clip_for_step,
)
from tca_map.xvla_spatial_task5.training_spec import build_r2p_xvla_training_spec


def _write_phase_demo(group, name: str, *, offset: float = 0.0, steps: int = 24) -> None:
    demo = group.create_group(name)
    states = np.zeros((steps, 45), dtype=np.float64)
    states[:, 31:34] = np.array([0.0 + offset, 0.0, 0.0])
    states[:, 38:41] = np.array([1.0 + offset, 0.0, 0.0])
    states[0:6, 10:13] = np.array([0.0 + offset, 0.0, 0.0])
    states[6:14, 10:13] = np.array([0.5 + offset, 0.5, 0.0])
    states[14:, 10:13] = np.array([1.0 + offset, 0.0, 0.0])
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


def _write_hdf5(path: Path, demos: int = 4) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_phase_demo(data, f"demo_{index}", offset=0.001 * index)


def test_r2p_arm_lookup_rejects_unfrozen_arm() -> None:
    spec = build_r2p_xvla_training_spec()

    assert _arm_by_id(spec, "r2p_xvla_rank8_phase_weights_lr1e4_steps64")["role"] == "primary_selected_method"
    with pytest.raises(ValueError):
        _arm_by_id(spec, "third_unfrozen_r2p_xvla_arm")


def test_r2p_phase_cycle_matches_frozen_order() -> None:
    spec = build_r2p_xvla_training_spec()
    cycle = _phase_cycle_from_sampler(spec["arms"][0]["sampler"])

    assert cycle == [
        "source_on_ramekin",
        "transit",
        "transit",
        "target_on_plate",
        "target_on_plate",
    ]
    assert [_phase_for_step(cycle, index) for index in range(6)] == [
        "source_on_ramekin",
        "transit",
        "transit",
        "target_on_plate",
        "target_on_plate",
        "source_on_ramekin",
    ]


def test_task5_build_phase_clip_index_and_select_clip(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    _write_hdf5(hdf5_path)
    spec = build_r2p_xvla_training_spec()
    cycle = _phase_cycle_from_sampler(spec["arms"][0]["sampler"])

    grouped = build_phase_clip_index(hdf5_path, demo_indices=[0, 1, 2], clip_steps=8)

    assert all(grouped[phase] for phase in ("source_on_ramekin", "transit", "target_on_plate"))
    rng = np.random.default_rng(20260718)
    selected = [
        select_clip_for_step(grouped, cycle=cycle, step_index_zero_based=index, rng=rng)
        for index in range(5)
    ]
    assert [item["phase_label"] for item in selected] == cycle


def test_task5_materialize_xvla_clip_writes_reader_meta_and_phase_weight(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    _write_hdf5(hdf5_path, demos=1)
    spec = build_r2p_xvla_training_spec()
    arm = _arm_by_id(spec, "r2p_xvla_rank8_phase_weights_lr1e4_steps64")
    clip = {
        "demo_index": 0,
        "demo_name": "demo_0",
        "source_start_index": 5,
        "source_end_index": 17,
        "phase_label": "source_on_ramekin",
    }

    materialized = materialize_xvla_clip(hdf5_path, tmp_path / "adapter", clip, spec, arm)

    assert Path(materialized["meta_path"]).is_file()
    assert Path(materialized["clip_hdf5"]).is_file()
    assert materialized["clip_steps"] == 12
    assert materialized["abs_action_6d_shape"] == [12, 10]
    assert materialized["phase_counts"]["transit"] > 0
    assert materialized["phase_weight_mean"] > 1.0
    assert materialized["residual_reset_used_for_sampling"] is False
    assert materialized["privileged_state_at_inference"] is False


def test_output_root_guard_rejects_noncanonical_path(tmp_path: Path) -> None:
    _assert_output_root_allowed(DEFAULT_OUTPUT_ROOT)
    with pytest.raises(ValueError):
        _assert_output_root_allowed(tmp_path / "not_the_frozen_output_root")


def test_run_training_arm_rejects_downloads_before_runtime_artifacts() -> None:
    spec = build_r2p_xvla_training_spec()

    with pytest.raises(ValueError, match="downloads are not allowed"):
        run_training_arm(
            TrainArmConfig(
                arm_id=spec["arms"][0]["arm_id"],
                local_files_only=False,
            )
        )


def test_train_lora_default_xvla_root_is_wsl_path() -> None:
    assert DEFAULT_XVLA_ROOT.as_posix() == "/mnt/c/assets/repos/X-VLA"
