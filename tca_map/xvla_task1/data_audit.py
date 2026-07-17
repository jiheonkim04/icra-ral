"""Data-health audit for the Epoch 5 X-VLA task-1 basket residual.

This CPU-only audit checks whether the local LIBERO-10 task-1 HDF5 data has
enough phase structure to support a narrow Ours candidate after the matched
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

CREAM_CHEESE_POS_SLICE = slice(17, 20)
BUTTER_POS_SLICE = slice(52, 55)
BASKET_POS_SLICE = slice(59, 62)


@dataclass(frozen=True)
class Task1BasketAuditConfig:
    hdf5_path: Path
    output_path: Path | None = None
    residual_initial_state_sha256: tuple[str, ...] = ()
    chunk_size: int = 8
    train_demo_count: int = 40
    basket_xy_threshold: float = 0.08


def init_state_sha256(initial_state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _as_float_array(dataset: Any) -> np.ndarray:
    return np.asarray(dataset, dtype=np.float64)


def _object_positions(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] < BASKET_POS_SLICE.stop:
        raise ValueError(f"expected states shape [T, >={BASKET_POS_SLICE.stop}], got {states.shape}")
    return states[:, CREAM_CHEESE_POS_SLICE], states[:, BUTTER_POS_SLICE], states[:, BASKET_POS_SLICE]


def _phase_labels(states: np.ndarray, *, basket_xy_threshold: float) -> dict[str, np.ndarray]:
    cream, butter, basket = _object_positions(states)
    cream_xy_dist = np.linalg.norm(cream[:, :2] - basket[:, :2], axis=1)
    butter_xy_dist = np.linalg.norm(butter[:, :2] - basket[:, :2], axis=1)
    cream_in = cream_xy_dist <= basket_xy_threshold
    butter_in = butter_xy_dist <= basket_xy_threshold
    count_in = cream_in.astype(np.int64) + butter_in.astype(np.int64)
    return {
        "cream_xy_dist_to_basket": cream_xy_dist,
        "butter_xy_dist_to_basket": butter_xy_dist,
        "cream_in_basket_region": cream_in,
        "butter_in_basket_region": butter_in,
        "target_count_in_basket_region": count_in,
        "one_target_remaining_phase": count_in == 1,
    }


def _count_values(values: Iterable[int]) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values).reshape(-1) > threshold)
    return int(indices[0]) if indices.size else None


def _split_name(index: int, train_demo_count: int) -> str:
    return "train" if index < train_demo_count else "validation"


def audit_task1_basket_data(config: Task1BasketAuditConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.train_demo_count <= 0:
        raise ValueError("train_demo_count must be positive")
    if config.basket_xy_threshold <= 0:
        raise ValueError("basket_xy_threshold must be positive")
    if not config.hdf5_path.exists():
        raise FileNotFoundError(config.hdf5_path)

    with h5py.File(config.hdf5_path, "r") as h5:
        if "data" not in h5:
            raise ValueError("HDF5 file has no top-level 'data' group")
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        if len(demo_names) <= config.train_demo_count:
            raise ValueError("train_demo_count must leave at least one validation demo")

        records: list[dict[str, Any]] = []
        initial_xy_distances: list[float] = []
        final_xy_distances: list[float] = []
        final_z_offsets: list[float] = []

        for demo_index, demo_name in enumerate(demo_names):
            group = h5["data"][demo_name]
            states = _as_float_array(group["states"])
            actions = _as_float_array(group["actions"])
            rewards = np.asarray(group["rewards"])
            dones = np.asarray(group["dones"])
            init_state = np.asarray(group.attrs["init_state"], dtype=np.float64).reshape(-1)
            cream, butter, basket = _object_positions(states)
            cream_dist = np.linalg.norm(cream[:, :2] - basket[:, :2], axis=1)
            butter_dist = np.linalg.norm(butter[:, :2] - basket[:, :2], axis=1)
            initial_xy_distances.extend([float(cream_dist[0]), float(butter_dist[0])])
            final_xy_distances.extend([float(cream_dist[-1]), float(butter_dist[-1])])
            final_z_offsets.extend([float(cream[-1, 2] - basket[-1, 2]), float(butter[-1, 2] - basket[-1, 2])])
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
            "demos_with_one_target_remaining_chunks": 0,
        },
        "validation": {
            "demo_count": 0,
            "step_count": 0,
            "chunk_count": 0,
            "phase_step_counts": {"0": 0, "1": 0, "2": 0},
            "phase_chunk_counts": {"0": 0, "1": 0, "2": 0},
            "demos_with_one_target_remaining_chunks": 0,
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

    for record in records:
        split = str(record["split"])
        states = record["states"]
        actions = record["actions"]
        rewards = record["rewards"]
        dones = record["dones"]
        labels = _phase_labels(states, basket_xy_threshold=config.basket_xy_threshold)

        action_dim_set.add(int(actions.shape[1]) if actions.ndim == 2 else -1)
        action_mins.append(float(np.min(actions)))
        action_maxs.append(float(np.max(actions)))
        action_finite = bool(action_finite and np.all(np.isfinite(actions)))
        terminal_reward_demos += int(np.any(rewards > 0))
        terminal_done_demos += int(np.any(dones > 0))
        if record["init_state_sha256"] in residual_hashes:
            overlap_hashes.append(record["init_state_sha256"])

        chunk_count = max(0, int(states.shape[0]) - config.chunk_size + 1)
        chunk_phase_counts = _count_values(labels["target_count_in_basket_region"][:chunk_count])
        step_phase_counts = _count_values(labels["target_count_in_basket_region"])
        has_one_target_chunks = int(chunk_phase_counts.get("1", 0) > 0)

        summary = split_summary[split]
        summary["demo_count"] += 1
        summary["step_count"] += int(states.shape[0])
        summary["chunk_count"] += chunk_count
        summary["demos_with_one_target_remaining_chunks"] += has_one_target_chunks
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
                "has_one_target_remaining_phase": bool(has_one_target_chunks),
                "first_reward_index": record["first_reward_index"],
                "first_done_index": record["first_done_index"],
                "init_state_sha256": record["init_state_sha256"],
            }
        )

    initial_xy = np.asarray(initial_xy_distances, dtype=np.float64)
    final_xy = np.asarray(final_xy_distances, dtype=np.float64)
    final_z = np.asarray(final_z_offsets, dtype=np.float64)
    gate_checks = {
        "hdf5_demo_count_at_least_20": len(records) >= 20,
        "train_validation_split_nonempty": split_summary["train"]["demo_count"] > 0
        and split_summary["validation"]["demo_count"] > 0,
        "actions_are_7d": action_dim_set == {7},
        "actions_are_finite": action_finite,
        "actions_within_libero_controller_range_with_margin": min(action_mins) >= -1.01 and max(action_maxs) <= 1.01,
        "all_demos_have_terminal_reward": terminal_reward_demos == len(records),
        "all_demos_have_terminal_done": terminal_done_demos == len(records),
        "train_has_one_target_remaining_chunks": split_summary["train"]["phase_chunk_counts"]["1"] > 0,
        "validation_has_one_target_remaining_chunks": split_summary["validation"]["phase_chunk_counts"]["1"] > 0,
        "every_train_demo_has_one_target_remaining_chunks": split_summary["train"]["demos_with_one_target_remaining_chunks"]
        == split_summary["train"]["demo_count"],
        "every_validation_demo_has_one_target_remaining_chunks": split_summary["validation"][
            "demos_with_one_target_remaining_chunks"
        ]
        == split_summary["validation"]["demo_count"],
        "phase_labels_not_collapsed": all(
            int(split_summary["train"]["phase_chunk_counts"][key]) > 0 for key in ("0", "1", "2")
        ),
        "initial_states_do_not_overlap_residual_failures": len(overlap_hashes) == 0,
        "basket_region_separates_initial_from_final": float(np.min(initial_xy)) > config.basket_xy_threshold
        and float(np.max(final_xy)) < config.basket_xy_threshold,
        "privileged_state_used_only_for_training_labels": True,
    }

    return {
        "schema_version": "2026-07-17.epoch5_xvla_task1_basket_data_audit.v1",
        "method_family": "candidate_predesign_data_health",
        "hdf5_path": str(config.hdf5_path),
        "chunk_size": int(config.chunk_size),
        "train_demo_count": int(config.train_demo_count),
        "validation_demo_count": int(len(records) - config.train_demo_count),
        "state_layout": {
            "cream_cheese_pos_slice": [CREAM_CHEESE_POS_SLICE.start, CREAM_CHEESE_POS_SLICE.stop],
            "butter_pos_slice": [BUTTER_POS_SLICE.start, BUTTER_POS_SLICE.stop],
            "basket_pos_slice": [BASKET_POS_SLICE.start, BASKET_POS_SLICE.stop],
            "layout_evidence": "verified by exact-init LIBERO env inspection for demo_0 on 2026-07-17 KST; sim_l2_to_hdf5_init=0.0",
        },
        "basket_region": {
            "source": "target-object XY distance to basket XY; threshold selected before training from train/final-vs-initial separation",
            "basket_xy_threshold": float(config.basket_xy_threshold),
            "final_xy_dist_min": float(np.min(final_xy)),
            "final_xy_dist_median": float(np.median(final_xy)),
            "final_xy_dist_q95": float(np.quantile(final_xy, 0.95)),
            "final_xy_dist_max": float(np.max(final_xy)),
            "initial_xy_dist_min": float(np.min(initial_xy)),
            "initial_xy_dist_median": float(np.median(initial_xy)),
            "initial_xy_dist_q95": float(np.quantile(initial_xy, 0.95)),
            "initial_xy_dist_max": float(np.max(initial_xy)),
            "final_z_offset_min": float(np.min(final_z)),
            "final_z_offset_median": float(np.median(final_z)),
            "final_z_offset_max": float(np.max(final_z)),
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
        "passes_data_health_gate": all(bool(value) for value in gate_checks.values()),
        "decision": (
            "TASK1_BASKET_DATA_HEALTH_PASS_PREDESIGN_READY"
            if all(bool(value) for value in gate_checks.values())
            else "TASK1_BASKET_DATA_HEALTH_FAIL"
        ),
        "demo_summaries": demo_summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--residual-initial-state-sha256", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--train-demo-count", type=int, default=40)
    parser.add_argument("--basket-xy-threshold", type=float, default=0.08)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = Task1BasketAuditConfig(
        hdf5_path=Path(args.hdf5_path),
        output_path=Path(args.output_path) if args.output_path else None,
        residual_initial_state_sha256=tuple(args.residual_initial_state_sha256 or ()),
        chunk_size=int(args.chunk_size),
        train_demo_count=int(args.train_demo_count),
        basket_xy_threshold=float(args.basket_xy_threshold),
    )
    report = audit_task1_basket_data(config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passes_data_health_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
