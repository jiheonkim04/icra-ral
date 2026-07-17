from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tca_map.xvla_spatial_task5.offline_validate import (
    DEFAULT_OUTPUT,
    OfflineValidationConfig,
    _assert_output_path_allowed,
    _default_adapter_dirs,
    _summarize_policy_rows,
    compute_offline_selection_decision,
    run_offline_validation,
    select_fixed_validation_clips,
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


def _write_hdf5(path: Path, demos: int = 50) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for index in range(demos):
            _write_phase_demo(data, f"demo_{index}", offset=0.001 * index)


def _rows(losses: dict[str, list[float]], weights: dict[str, float]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    index = 0
    for phase, values in losses.items():
        for value in values:
            rows.append(
                {
                    "validation_index": index,
                    "phase_label": phase,
                    "loss_total": float(value),
                    "weighted_loss": float(value) * float(weights[phase]),
                }
            )
            index += 1
    return rows


def test_select_fixed_validation_clips_uses_heldout_split_and_phase_cycle(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "task5.hdf5"
    _write_hdf5(hdf5_path)
    spec = build_r2p_xvla_training_spec()

    clips = select_fixed_validation_clips(hdf5_path=hdf5_path, spec=spec, num_chunks=5, clip_steps=8)

    assert [clip["validation_index"] for clip in clips] == [0, 1, 2, 3, 4]
    assert [clip["phase_label"] for clip in clips] == [
        "source_on_ramekin",
        "transit",
        "transit",
        "target_on_plate",
        "target_on_plate",
    ]
    assert all(int(clip["demo_index"]) >= 40 for clip in clips)


def test_default_adapter_dirs_are_frozen_step64_paths() -> None:
    spec = build_r2p_xvla_training_spec()
    primary, ablation = _default_adapter_dirs(OfflineValidationConfig(), spec)

    assert primary.as_posix().endswith(
        "runs/xvla_prior/epoch5_r2p_xvla_task5_training/"
        "r2p_xvla_rank8_phase_weights_lr1e4_steps64/checkpoints/step_0064/adapter"
    )
    assert ablation.as_posix().endswith(
        "runs/xvla_prior/epoch5_r2p_xvla_task5_training/"
        "uniform_task5_xvla_rank8_lambda0_lr1e4_steps64/checkpoints/step_0064/adapter"
    )


def test_offline_selection_uses_common_r2p_weighted_metric() -> None:
    spec = build_r2p_xvla_training_spec()
    weights = spec["arms"][0]["phase_loss_weights"]
    summaries = {
        "xvla_prior_base": _summarize_policy_rows(
            _rows(
                {
                    "source_on_ramekin": [1.0],
                    "transit": [1.2],
                    "target_on_plate": [1.1],
                },
                weights,
            )
        ),
        "r2p_xvla_primary": _summarize_policy_rows(
            _rows(
                {
                    "source_on_ramekin": [1.0],
                    "transit": [0.7],
                    "target_on_plate": [0.8],
                },
                weights,
            )
        ),
        "uniform_task5_xvla_ablation": _summarize_policy_rows(
            _rows(
                {
                    "source_on_ramekin": [1.0],
                    "transit": [1.0],
                    "target_on_plate": [0.95],
                },
                weights,
            )
        ),
    }
    summaries["r2p_xvla_primary"]["delta_vs_prior"] = {
        "fixed_chunk_mean_abs_action_delta": 0.10,
        "fixed_chunk_max_abs_action_delta": 0.30,
    }
    summaries["uniform_task5_xvla_ablation"]["delta_vs_prior"] = {
        "fixed_chunk_mean_abs_action_delta": 0.10,
        "fixed_chunk_max_abs_action_delta": 0.30,
    }
    runtimes = {
        label: {"cuda_memory_after_eval": {"max_allocated_mib": 1000.0}}
        for label in summaries
    }

    decision = compute_offline_selection_decision(spec=spec, summaries=summaries, runtimes=runtimes)

    assert decision["decision"] == "R2P_XVLA_OFFLINE_PASS_BEATS_UNIFORM_ABLATION"
    assert decision["primary_beats_uniform_on_phase_weighted_validation_loss"] is True


def test_offline_selection_blocks_source_phase_degradation() -> None:
    spec = build_r2p_xvla_training_spec()
    weights = spec["arms"][0]["phase_loss_weights"]
    summaries = {
        "xvla_prior_base": _summarize_policy_rows(
            _rows({"source_on_ramekin": [1.0], "transit": [1.0], "target_on_plate": [1.0]}, weights)
        ),
        "r2p_xvla_primary": _summarize_policy_rows(
            _rows({"source_on_ramekin": [1.2], "transit": [0.5], "target_on_plate": [0.5]}, weights)
        ),
        "uniform_task5_xvla_ablation": _summarize_policy_rows(
            _rows({"source_on_ramekin": [1.0], "transit": [1.0], "target_on_plate": [1.0]}, weights)
        ),
    }
    summaries["r2p_xvla_primary"]["delta_vs_prior"] = {
        "fixed_chunk_mean_abs_action_delta": 0.10,
        "fixed_chunk_max_abs_action_delta": 0.30,
    }
    summaries["uniform_task5_xvla_ablation"]["delta_vs_prior"] = {
        "fixed_chunk_mean_abs_action_delta": 0.10,
        "fixed_chunk_max_abs_action_delta": 0.30,
    }
    runtimes = {
        label: {"cuda_memory_after_eval": {"max_allocated_mib": 1000.0}}
        for label in summaries
    }

    decision = compute_offline_selection_decision(spec=spec, summaries=summaries, runtimes=runtimes)

    assert decision["decision"] == "R2P_XVLA_OFFLINE_SELECTION_NOT_PASSED"
    assert "source_phase_degradation_vs_uniform_exceeds_bound" in decision["reasons"]


def test_output_path_guard_rejects_noncanonical_path(tmp_path: Path) -> None:
    _assert_output_path_allowed(DEFAULT_OUTPUT)
    with pytest.raises(ValueError):
        _assert_output_path_allowed(tmp_path / "offline_validation.json")


def test_run_offline_validation_rejects_downloads_before_runtime_artifacts() -> None:
    with pytest.raises(ValueError, match="downloads are not allowed"):
        run_offline_validation(OfflineValidationConfig(local_files_only=False))
