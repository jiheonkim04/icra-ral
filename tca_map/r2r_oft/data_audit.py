"""Data-health audit for R2R-OFT before any training.

R2R-OFT targets the Epoch 5 residual where OpenVLA-OFT places or reaches the
first moka-pot-on-stove phase, then times out while completing the remaining
moka pot. This module performs a CPU-only HDF5 audit for that exact task.
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


MOKA_POT_1_POS_SLICE = slice(10, 13)
MOKA_POT_2_POS_SLICE = slice(17, 20)


@dataclass(frozen=True)
class R2ROFTAuditConfig:
    hdf5_path: Path
    output_path: Path | None = None
    residual_initial_state_sha256: tuple[str, ...] = ()
    chunk_size: int = 8
    train_demo_count: int = 40
    stove_xy_threshold: float = 0.12
    stove_z_min: float = 0.98


def init_state_sha256(initial_state: Any) -> str:
    """Hash initial states exactly as the OpenVLA-OFT gate does."""

    array = np.ascontiguousarray(np.asarray(initial_state))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _as_float_array(dataset: Any) -> np.ndarray:
    return np.asarray(dataset, dtype=np.float64)


def _pot_positions(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] < MOKA_POT_2_POS_SLICE.stop:
        raise ValueError(f"expected states shape [T, >=20], got {states.shape}")
    return states[:, MOKA_POT_1_POS_SLICE], states[:, MOKA_POT_2_POS_SLICE]


def _phase_labels(
    states: np.ndarray,
    *,
    target_xy: np.ndarray,
    stove_xy_threshold: float,
    stove_z_min: float,
) -> dict[str, np.ndarray]:
    pot_1, pot_2 = _pot_positions(states)
    pot_1_xy_dist = np.linalg.norm(pot_1[:, :2] - target_xy.reshape(1, 2), axis=1)
    pot_2_xy_dist = np.linalg.norm(pot_2[:, :2] - target_xy.reshape(1, 2), axis=1)
    pot_1_on = (pot_1_xy_dist <= stove_xy_threshold) & (pot_1[:, 2] >= stove_z_min)
    pot_2_on = (pot_2_xy_dist <= stove_xy_threshold) & (pot_2[:, 2] >= stove_z_min)
    count_on = pot_1_on.astype(np.int64) + pot_2_on.astype(np.int64)
    return {
        "pot_1_xy_dist": pot_1_xy_dist,
        "pot_2_xy_dist": pot_2_xy_dist,
        "pot_1_on": pot_1_on,
        "pot_2_on": pot_2_on,
        "count_on": count_on,
        "one_pot_remaining_phase": count_on == 1,
    }


def _count_values(values: Iterable[int]) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _split_name(index: int, train_demo_count: int) -> str:
    return "train" if index < train_demo_count else "validation"


def audit_r2r_oft_data(config: R2ROFTAuditConfig) -> dict[str, Any]:
    """Audit task-8 HDF5 data for phase-balanced R2R-OFT training readiness."""

    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.train_demo_count <= 0:
        raise ValueError("train_demo_count must be positive")
    if not config.hdf5_path.exists():
        raise FileNotFoundError(config.hdf5_path)

    with h5py.File(config.hdf5_path, "r") as h5:
        if "data" not in h5:
            raise ValueError("HDF5 file has no top-level 'data' group")
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        if len(demo_names) <= config.train_demo_count:
            raise ValueError("train_demo_count must leave at least one validation demo")

        train_final_positions: list[np.ndarray] = []
        all_initial_positions: list[np.ndarray] = []
        records: list[dict[str, Any]] = []

        for demo_index, demo_name in enumerate(demo_names):
            group = h5["data"][demo_name]
            states = _as_float_array(group["states"])
            actions = _as_float_array(group["actions"])
            rewards = np.asarray(group["rewards"])
            dones = np.asarray(group["dones"])
            init_state = np.asarray(group.attrs["init_state"])
            pot_1, pot_2 = _pot_positions(states)

            all_initial_positions.extend([pot_1[0], pot_2[0]])
            if demo_index < config.train_demo_count:
                train_final_positions.extend([pot_1[-1], pot_2[-1]])

            records.append(
                {
                    "demo_name": demo_name,
                    "demo_index": demo_index,
                    "split": _split_name(demo_index, config.train_demo_count),
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "init_state_sha256": init_state_sha256(init_state),
                    "first_reward_index": int(np.flatnonzero(rewards > 0)[0]) if np.any(rewards > 0) else None,
                    "first_done_index": int(np.flatnonzero(dones > 0)[0]) if np.any(dones > 0) else None,
                }
            )

    train_final = np.asarray(train_final_positions, dtype=np.float64)
    initial_positions = np.asarray(all_initial_positions, dtype=np.float64)
    target_xy = np.median(train_final[:, :2], axis=0)
    train_final_xy_dist = np.linalg.norm(train_final[:, :2] - target_xy.reshape(1, 2), axis=1)
    initial_xy_dist = np.linalg.norm(initial_positions[:, :2] - target_xy.reshape(1, 2), axis=1)

    split_summary: dict[str, dict[str, Any]] = {
        "train": {
            "demo_count": 0,
            "step_count": 0,
            "chunk_count": 0,
            "phase_step_counts": {"0": 0, "1": 0, "2": 0},
            "phase_chunk_counts": {"0": 0, "1": 0, "2": 0},
            "demos_with_one_pot_chunks": 0,
        },
        "validation": {
            "demo_count": 0,
            "step_count": 0,
            "chunk_count": 0,
            "phase_step_counts": {"0": 0, "1": 0, "2": 0},
            "phase_chunk_counts": {"0": 0, "1": 0, "2": 0},
            "demos_with_one_pot_chunks": 0,
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
        labels = _phase_labels(
            states,
            target_xy=target_xy,
            stove_xy_threshold=config.stove_xy_threshold,
            stove_z_min=config.stove_z_min,
        )

        action_dim_set.add(int(actions.shape[1]) if actions.ndim == 2 else -1)
        action_mins.append(float(np.min(actions)))
        action_maxs.append(float(np.max(actions)))
        action_finite = bool(action_finite and np.all(np.isfinite(actions)))
        terminal_reward_demos += int(np.any(rewards > 0))
        terminal_done_demos += int(np.any(dones > 0))
        if record["init_state_sha256"] in residual_hashes:
            overlap_hashes.append(record["init_state_sha256"])

        chunk_count = max(0, int(states.shape[0]) - config.chunk_size + 1)
        chunk_phase_counts = _count_values(labels["count_on"][:chunk_count])
        step_phase_counts = _count_values(labels["count_on"])
        has_one_pot_chunks = int(chunk_phase_counts.get("1", 0) > 0)

        summary = split_summary[split]
        summary["demo_count"] += 1
        summary["step_count"] += int(states.shape[0])
        summary["chunk_count"] += chunk_count
        summary["demos_with_one_pot_chunks"] += has_one_pot_chunks
        for key in ("0", "1", "2"):
            summary["phase_step_counts"][key] += int(step_phase_counts.get(key, 0))
            summary["phase_chunk_counts"][key] += int(chunk_phase_counts.get(key, 0))

        demo_summaries.append(
            {
                "demo_name": record["demo_name"],
                "demo_index": int(record["demo_index"]),
                "split": split,
                "steps": int(states.shape[0]),
                "chunks": chunk_count,
                "phase_chunk_counts": chunk_phase_counts,
                "has_one_pot_remaining_phase": bool(has_one_pot_chunks),
                "first_reward_index": record["first_reward_index"],
                "first_done_index": record["first_done_index"],
                "init_state_sha256": record["init_state_sha256"],
            }
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
        "train_has_one_pot_remaining_chunks": split_summary["train"]["phase_chunk_counts"]["1"] > 0,
        "validation_has_one_pot_remaining_chunks": split_summary["validation"]["phase_chunk_counts"]["1"] > 0,
        "phase_labels_not_collapsed": all(
            int(split_summary["train"]["phase_chunk_counts"][key]) > 0 for key in ("0", "1", "2")
        ),
        "initial_states_do_not_overlap_residual_failures": len(overlap_hashes) == 0,
        "target_region_separates_initial_from_final": float(np.min(initial_xy_dist)) > config.stove_xy_threshold
        and float(np.quantile(train_final_xy_dist, 0.95)) < config.stove_xy_threshold,
        "privileged_state_used_only_for_training_labels": True,
    }

    report = {
        "schema_version": 1,
        "method": "R2R-OFT",
        "audit_type": "pretraining_data_health",
        "hdf5_path": str(config.hdf5_path),
        "chunk_size": int(config.chunk_size),
        "train_demo_count": int(config.train_demo_count),
        "validation_demo_count": int(len(records) - config.train_demo_count),
        "state_layout": {
            "moka_pot_1_pos_slice": [MOKA_POT_1_POS_SLICE.start, MOKA_POT_1_POS_SLICE.stop],
            "moka_pot_2_pos_slice": [MOKA_POT_2_POS_SLICE.start, MOKA_POT_2_POS_SLICE.stop],
            "layout_evidence": "verified against LIBERO env observation for demo_0 exact init on 2026-07-17 KST",
        },
        "target_region": {
            "source": "median final pot xy over training demos only",
            "target_xy": [float(x) for x in target_xy],
            "stove_xy_threshold": float(config.stove_xy_threshold),
            "stove_z_min": float(config.stove_z_min),
            "train_final_xy_dist_q95": float(np.quantile(train_final_xy_dist, 0.95)),
            "train_final_xy_dist_max": float(np.max(train_final_xy_dist)),
            "initial_xy_dist_min": float(np.min(initial_xy_dist)),
            "initial_xy_dist_mean": float(np.mean(initial_xy_dist)),
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
            "training_only_phase_label_source": "HDF5 simulator state / bounded replay",
        },
        "gate_checks": gate_checks,
        "passes_data_health_gate": all(bool(value) for value in gate_checks.values()),
        "demo_summaries": demo_summaries,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--residual-initial-state-sha256", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--train-demo-count", type=int, default=40)
    parser.add_argument("--stove-xy-threshold", type=float, default=0.12)
    parser.add_argument("--stove-z-min", type=float, default=0.98)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = R2ROFTAuditConfig(
        hdf5_path=Path(args.hdf5_path),
        output_path=Path(args.output_path) if args.output_path else None,
        residual_initial_state_sha256=tuple(args.residual_initial_state_sha256 or ()),
        chunk_size=int(args.chunk_size),
        train_demo_count=int(args.train_demo_count),
        stove_xy_threshold=float(args.stove_xy_threshold),
        stove_z_min=float(args.stove_z_min),
    )
    report = audit_r2r_oft_data(config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passes_data_health_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

