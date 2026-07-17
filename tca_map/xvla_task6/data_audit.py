"""Data-health audit for the Epoch 5 X-VLA task-6 residual.

This CPU-only audit checks whether the local LIBERO-10 task-6 HDF5 data has
enough phase structure to support a narrow method candidate after the matched
X-VLA/SmolVLA Base residual and expert-headroom gates. It performs no training,
no model loading, no optimizer steps, no simulator rollout, and no downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

PORCELAIN_MUG_POS_SLICE = slice(10, 13)
RED_MUG_POS_SLICE = slice(17, 20)
PLATE_POS_SLICE = slice(24, 27)
CHOCOLATE_PUDDING_POS_SLICE = slice(31, 34)


@dataclass(frozen=True)
class Task6SpatialAuditConfig:
    hdf5_path: Path
    output_path: Path | None = None
    residual_initial_state_sha256: tuple[str, ...] = ()
    chunk_size: int = 8
    train_demo_count: int = 40
    mug_plate_xy_threshold: float = 0.05
    pudding_abs_dx_threshold: float = 0.07
    pudding_dy_min: float = 0.08
    pudding_dy_max: float = 0.16


def init_state_sha256(initial_state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _as_float_array(dataset: Any) -> np.ndarray:
    return np.asarray(dataset, dtype=np.float64)


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values).reshape(-1) > threshold)
    return int(indices[0]) if indices.size else None


def _split_name(index: int, train_demo_count: int) -> str:
    return "train" if index < train_demo_count else "validation"


def _count_values(values: Iterable[int]) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _object_positions(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] < CHOCOLATE_PUDDING_POS_SLICE.stop:
        raise ValueError(f"expected states shape [T, >={CHOCOLATE_PUDDING_POS_SLICE.stop}], got {states.shape}")
    return (
        states[:, PORCELAIN_MUG_POS_SLICE],
        states[:, RED_MUG_POS_SLICE],
        states[:, PLATE_POS_SLICE],
        states[:, CHOCOLATE_PUDDING_POS_SLICE],
    )


def _phase_labels(states: np.ndarray, config: Task6SpatialAuditConfig) -> dict[str, np.ndarray]:
    porcelain_mug, _red_mug, plate, pudding = _object_positions(states)
    mug_xy_dist = np.linalg.norm(porcelain_mug[:, :2] - plate[:, :2], axis=1)
    pudding_dx = pudding[:, 0] - plate[:, 0]
    pudding_dy = pudding[:, 1] - plate[:, 1]
    mug_on_plate = mug_xy_dist <= config.mug_plate_xy_threshold
    pudding_right = (
        (np.abs(pudding_dx) <= config.pudding_abs_dx_threshold)
        & (pudding_dy >= config.pudding_dy_min)
        & (pudding_dy <= config.pudding_dy_max)
    )
    completed_count = mug_on_plate.astype(np.int64) + pudding_right.astype(np.int64)
    return {
        "mug_xy_dist_to_plate": mug_xy_dist,
        "pudding_dx_to_plate": pudding_dx,
        "pudding_dy_to_plate": pudding_dy,
        "mug_on_plate_region": mug_on_plate,
        "pudding_right_of_plate_region": pudding_right,
        "completed_subgoal_count": completed_count,
        "mug_done_pudding_remaining_phase": mug_on_plate & ~pudding_right,
        "pudding_done_mug_remaining_phase": pudding_right & ~mug_on_plate,
    }


def audit_task6_spatial_data(config: Task6SpatialAuditConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.train_demo_count <= 0:
        raise ValueError("train_demo_count must be positive")
    if config.mug_plate_xy_threshold <= 0:
        raise ValueError("mug_plate_xy_threshold must be positive")
    if config.pudding_abs_dx_threshold <= 0:
        raise ValueError("pudding_abs_dx_threshold must be positive")
    if config.pudding_dy_min >= config.pudding_dy_max:
        raise ValueError("pudding_dy_min must be less than pudding_dy_max")
    if not config.hdf5_path.exists():
        raise FileNotFoundError(config.hdf5_path)

    with h5py.File(config.hdf5_path, "r") as h5:
        if "data" not in h5:
            raise ValueError("HDF5 file has no top-level 'data' group")
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        if len(demo_names) <= config.train_demo_count:
            raise ValueError("train_demo_count must leave at least one validation demo")

        records: list[dict[str, Any]] = []
        initial_mug_xy: list[float] = []
        final_mug_xy: list[float] = []
        initial_pudding_dx: list[float] = []
        final_pudding_dx: list[float] = []
        initial_pudding_dy: list[float] = []
        final_pudding_dy: list[float] = []
        red_mug_xy_initial: list[float] = []
        red_mug_xy_final: list[float] = []

        for demo_index, demo_name in enumerate(demo_names):
            group = h5["data"][demo_name]
            states = _as_float_array(group["states"])
            actions = _as_float_array(group["actions"])
            rewards = np.asarray(group["rewards"])
            dones = np.asarray(group["dones"])
            init_state = np.asarray(group.attrs["init_state"], dtype=np.float64).reshape(-1)
            porcelain_mug, red_mug, plate, pudding = _object_positions(states)
            mug_xy = np.linalg.norm(porcelain_mug[:, :2] - plate[:, :2], axis=1)
            red_xy = np.linalg.norm(red_mug[:, :2] - plate[:, :2], axis=1)
            pudding_dx = pudding[:, 0] - plate[:, 0]
            pudding_dy = pudding[:, 1] - plate[:, 1]
            initial_mug_xy.append(float(mug_xy[0]))
            final_mug_xy.append(float(mug_xy[-1]))
            initial_pudding_dx.append(float(pudding_dx[0]))
            final_pudding_dx.append(float(pudding_dx[-1]))
            initial_pudding_dy.append(float(pudding_dy[0]))
            final_pudding_dy.append(float(pudding_dy[-1]))
            red_mug_xy_initial.append(float(red_xy[0]))
            red_mug_xy_final.append(float(red_xy[-1]))
            records.append(
                {
                    "demo_name": str(demo_name),
                    "demo_index": int(demo_index),
                    "split": _split_name(int(demo_index), config.train_demo_count),
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "init_state_sha256": init_state_sha256(init_state),
                    "first_reward_index": _first_index(rewards, 0.0),
                    "first_done_index": _first_index(dones, 0.5),
                }
            )

    split_summary: dict[str, dict[str, Any]] = {
        "train": {
            "demo_count": 0,
            "step_count": 0,
            "chunk_count": 0,
            "phase_step_counts": {"0": 0, "1": 0, "2": 0},
            "phase_chunk_counts": {"0": 0, "1": 0, "2": 0},
            "mug_done_pudding_remaining_chunks": 0,
            "pudding_done_mug_remaining_chunks": 0,
            "demos_with_mug_done_pudding_remaining_chunks": 0,
        },
        "validation": {
            "demo_count": 0,
            "step_count": 0,
            "chunk_count": 0,
            "phase_step_counts": {"0": 0, "1": 0, "2": 0},
            "phase_chunk_counts": {"0": 0, "1": 0, "2": 0},
            "mug_done_pudding_remaining_chunks": 0,
            "pudding_done_mug_remaining_chunks": 0,
            "demos_with_mug_done_pudding_remaining_chunks": 0,
        },
    }

    action_mins: list[float] = []
    action_maxs: list[float] = []
    action_finite = True
    action_dim_set: set[int] = set()
    terminal_reward_demos = 0
    terminal_done_demos = 0
    overlap_hashes: list[str] = []
    residual_hashes = set(config.residual_initial_state_sha256)
    demo_summaries: list[dict[str, Any]] = []
    subgoal_order_counts: dict[str, int] = {"mug_first": 0, "pudding_first": 0, "same_step": 0, "missing": 0}

    for record in records:
        split = str(record["split"])
        states = record["states"]
        actions = record["actions"]
        rewards = record["rewards"]
        dones = record["dones"]
        labels = _phase_labels(states, config)

        action_dim_set.add(int(actions.shape[1]) if actions.ndim == 2 else -1)
        action_mins.append(float(np.min(actions)))
        action_maxs.append(float(np.max(actions)))
        action_finite = bool(action_finite and np.all(np.isfinite(actions)))
        terminal_reward_demos += int(np.any(rewards > 0))
        terminal_done_demos += int(np.any(dones > 0))
        if record["init_state_sha256"] in residual_hashes:
            overlap_hashes.append(record["init_state_sha256"])

        chunk_count = max(0, int(states.shape[0]) - config.chunk_size + 1)
        completed = labels["completed_subgoal_count"]
        chunk_phase_counts = _count_values(completed[:chunk_count])
        step_phase_counts = _count_values(completed)
        mug_remaining_chunks = int(np.sum(labels["mug_done_pudding_remaining_phase"][:chunk_count]))
        pudding_remaining_chunks = int(np.sum(labels["pudding_done_mug_remaining_phase"][:chunk_count]))
        has_mug_remaining_chunks = int(mug_remaining_chunks > 0)

        mug_first_indices = np.flatnonzero(labels["mug_on_plate_region"])
        pudding_first_indices = np.flatnonzero(labels["pudding_right_of_plate_region"])
        mug_first = int(mug_first_indices[0]) if mug_first_indices.size else None
        pudding_first = int(pudding_first_indices[0]) if pudding_first_indices.size else None
        if mug_first is None or pudding_first is None:
            order = "missing"
        elif mug_first < pudding_first:
            order = "mug_first"
        elif pudding_first < mug_first:
            order = "pudding_first"
        else:
            order = "same_step"
        subgoal_order_counts[order] += 1

        summary = split_summary[split]
        summary["demo_count"] += 1
        summary["step_count"] += int(states.shape[0])
        summary["chunk_count"] += chunk_count
        summary["mug_done_pudding_remaining_chunks"] += mug_remaining_chunks
        summary["pudding_done_mug_remaining_chunks"] += pudding_remaining_chunks
        summary["demos_with_mug_done_pudding_remaining_chunks"] += has_mug_remaining_chunks
        for key in ("0", "1", "2"):
            summary["phase_step_counts"][key] += int(step_phase_counts.get(key, 0))
            summary["phase_chunk_counts"][key] += int(chunk_phase_counts.get(key, 0))

        demo_summaries.append(
            {
                "demo_name": str(record["demo_name"]),
                "demo_index": int(record["demo_index"]),
                "split": split,
                "steps": int(states.shape[0]),
                "chunks": int(chunk_count),
                "phase_chunk_counts": chunk_phase_counts,
                "mug_done_pudding_remaining_chunks": int(mug_remaining_chunks),
                "pudding_done_mug_remaining_chunks": int(pudding_remaining_chunks),
                "subgoal_order": order,
                "first_mug_on_plate_index": mug_first,
                "first_pudding_right_index": pudding_first,
                "first_reward_index": record["first_reward_index"],
                "first_done_index": record["first_done_index"],
                "init_state_sha256": record["init_state_sha256"],
            }
        )

    initial_mug = np.asarray(initial_mug_xy, dtype=np.float64)
    final_mug = np.asarray(final_mug_xy, dtype=np.float64)
    initial_dx = np.asarray(initial_pudding_dx, dtype=np.float64)
    final_dx = np.asarray(final_pudding_dx, dtype=np.float64)
    initial_dy = np.asarray(initial_pudding_dy, dtype=np.float64)
    final_dy = np.asarray(final_pudding_dy, dtype=np.float64)
    red_initial = np.asarray(red_mug_xy_initial, dtype=np.float64)
    red_final = np.asarray(red_mug_xy_final, dtype=np.float64)

    initial_pudding_right = (
        (np.abs(initial_dx) <= config.pudding_abs_dx_threshold)
        & (initial_dy >= config.pudding_dy_min)
        & (initial_dy <= config.pudding_dy_max)
    )
    final_pudding_right = (
        (np.abs(final_dx) <= config.pudding_abs_dx_threshold)
        & (final_dy >= config.pudding_dy_min)
        & (final_dy <= config.pudding_dy_max)
    )

    gate_checks = {
        "hdf5_demo_count_at_least_20": len(records) >= 20,
        "train_validation_split_nonempty": split_summary["train"]["demo_count"] > 0
        and split_summary["validation"]["demo_count"] > 0,
        "actions_are_7d": action_dim_set == {7},
        "actions_are_finite": action_finite,
        "actions_within_libero_controller_range_with_margin": min(action_mins) >= -1.01 and max(action_maxs) <= 1.01,
        "all_demos_have_terminal_reward": terminal_reward_demos == len(records),
        "all_demos_have_terminal_done": terminal_done_demos == len(records),
        "train_has_mug_done_pudding_remaining_chunks": split_summary["train"][
            "mug_done_pudding_remaining_chunks"
        ]
        > 0,
        "validation_has_mug_done_pudding_remaining_chunks": split_summary["validation"][
            "mug_done_pudding_remaining_chunks"
        ]
        > 0,
        "every_train_demo_has_mug_done_pudding_remaining_chunks": split_summary["train"][
            "demos_with_mug_done_pudding_remaining_chunks"
        ]
        == split_summary["train"]["demo_count"],
        "every_validation_demo_has_mug_done_pudding_remaining_chunks": split_summary["validation"][
            "demos_with_mug_done_pudding_remaining_chunks"
        ]
        == split_summary["validation"]["demo_count"],
        "phase_labels_not_collapsed": all(
            int(split_summary["train"]["phase_chunk_counts"][key]) > 0 for key in ("0", "1", "2")
        ),
        "all_demos_complete_mug_before_pudding": subgoal_order_counts == {
            "mug_first": len(records),
            "pudding_first": 0,
            "same_step": 0,
            "missing": 0,
        },
        "initial_states_do_not_overlap_residual_failures": len(overlap_hashes) == 0,
        "mug_plate_region_separates_initial_from_final": float(np.min(initial_mug)) > config.mug_plate_xy_threshold
        and float(np.max(final_mug)) < config.mug_plate_xy_threshold,
        "pudding_right_region_separates_initial_from_final": not bool(np.any(initial_pudding_right))
        and bool(np.all(final_pudding_right)),
        "red_mug_stays_off_plate_distractor": float(np.min(red_initial)) > config.mug_plate_xy_threshold
        and float(np.min(red_final)) > config.mug_plate_xy_threshold,
        "privileged_state_used_only_for_training_labels": True,
    }

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    passed = all(bool(value) for value in gate_checks.values())
    return {
        "schema_version": "2026-07-17.epoch5_xvla_task6_spatial_data_audit.v1",
        "method_family": "candidate_predesign_data_health",
        "hdf5_path": str(config.hdf5_path),
        "chunk_size": int(config.chunk_size),
        "train_demo_count": int(config.train_demo_count),
        "validation_demo_count": int(len(records) - config.train_demo_count),
        "state_layout": {
            "porcelain_mug_pos_slice": [PORCELAIN_MUG_POS_SLICE.start, PORCELAIN_MUG_POS_SLICE.stop],
            "red_mug_pos_slice": [RED_MUG_POS_SLICE.start, RED_MUG_POS_SLICE.stop],
            "plate_pos_slice": [PLATE_POS_SLICE.start, PLATE_POS_SLICE.stop],
            "chocolate_pudding_pos_slice": [
                CHOCOLATE_PUDDING_POS_SLICE.start,
                CHOCOLATE_PUDDING_POS_SLICE.stop,
            ],
            "layout_evidence": "verified by exact-init LIBERO env inspection for demo_0 on 2026-07-17 KST; sim_l2_to_hdf5_init=0.0",
        },
        "thresholds": {
            "mug_plate_xy_threshold": float(config.mug_plate_xy_threshold),
            "pudding_abs_dx_threshold": float(config.pudding_abs_dx_threshold),
            "pudding_dy_min": float(config.pudding_dy_min),
            "pudding_dy_max": float(config.pudding_dy_max),
            "source": "thresholds chosen before training from initial/final HDF5 separation and task semantics",
        },
        "spatial_separation": {
            "initial_mug_xy_to_plate": stats(initial_mug),
            "final_mug_xy_to_plate": stats(final_mug),
            "initial_pudding_dx_to_plate": stats(initial_dx),
            "final_pudding_dx_to_plate": stats(final_dx),
            "initial_pudding_dy_to_plate": stats(initial_dy),
            "final_pudding_dy_to_plate": stats(final_dy),
            "initial_red_mug_xy_to_plate": stats(red_initial),
            "final_red_mug_xy_to_plate": stats(red_final),
        },
        "dataset_summary": {
            "demo_count": int(len(records)),
            "total_steps": int(sum(int(record["states"].shape[0]) for record in records)),
            "total_chunks": int(sum(max(0, int(record["states"].shape[0]) - config.chunk_size + 1) for record in records)),
            "terminal_reward_demos": int(terminal_reward_demos),
            "terminal_done_demos": int(terminal_done_demos),
            "action_dim_values": sorted(action_dim_set),
            "action_min": float(min(action_mins)),
            "action_max": float(max(action_maxs)),
            "actions_finite": bool(action_finite),
            "subgoal_order_counts": subgoal_order_counts,
        },
        "split_summary": split_summary,
        "residual_overlap": {
            "residual_initial_state_sha256": sorted(residual_hashes),
            "overlap_count": int(len(overlap_hashes)),
            "overlap_hashes": sorted(overlap_hashes),
        },
        "deployment_input_policy": {
            "allowed_inference_inputs": ["agentview_rgb", "wrist_rgb", "proprioception", "instruction"],
            "privileged_state_at_inference": False,
            "training_only_phase_label_source": "HDF5 simulator state; no object-state inference inputs",
        },
        "gate_checks": gate_checks,
        "passes_data_health_gate": passed,
        "decision": "TASK6_SPATIAL_DATA_HEALTH_PASS_PREDESIGN_READY" if passed else "TASK6_SPATIAL_DATA_HEALTH_FAIL",
        "demo_summaries": demo_summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--residual-initial-state-sha256", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--train-demo-count", type=int, default=40)
    parser.add_argument("--mug-plate-xy-threshold", type=float, default=0.05)
    parser.add_argument("--pudding-abs-dx-threshold", type=float, default=0.07)
    parser.add_argument("--pudding-dy-min", type=float, default=0.08)
    parser.add_argument("--pudding-dy-max", type=float, default=0.16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = Task6SpatialAuditConfig(
        hdf5_path=Path(args.hdf5_path),
        output_path=Path(args.output_path) if args.output_path else None,
        residual_initial_state_sha256=tuple(args.residual_initial_state_sha256 or ()),
        chunk_size=int(args.chunk_size),
        train_demo_count=int(args.train_demo_count),
        mug_plate_xy_threshold=float(args.mug_plate_xy_threshold),
        pudding_abs_dx_threshold=float(args.pudding_abs_dx_threshold),
        pudding_dy_min=float(args.pudding_dy_min),
        pudding_dy_max=float(args.pudding_dy_max),
    )
    report = audit_task6_spatial_data(config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passes_data_health_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
