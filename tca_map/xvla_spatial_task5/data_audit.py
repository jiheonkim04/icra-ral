"""Data-health audit for the LIBERO-spatial task-5 residual.

This CPU-only audit checks whether local HDF5 supervision for
``libero_spatial/task_5`` is usable before any candidate generation or
training. It performs no training, no model loading, no optimizer steps, no
simulator rollout, and no downloads.
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

TARGET_BOWL_POS_SLICE = slice(10, 13)
RAMEKIN_POS_SLICE = slice(31, 34)
PLATE_POS_SLICE = slice(38, 41)


@dataclass(frozen=True)
class SpatialTask5DataAuditConfig:
    hdf5_path: Path
    output_path: Path | None = None
    residual_initial_state_sha256: tuple[str, ...] = ()
    chunk_size: int = 8
    train_demo_count: int = 40
    source_xy_threshold: float = 0.05
    target_xy_threshold: float = 0.05
    source_target_separation_min: float = 0.10
    action_abs_max_gate: float = 1.25


def init_state_sha256(initial_state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values).reshape(-1) > threshold)
    return int(indices[0]) if indices.size else None


def _split_name(index: int, train_demo_count: int) -> str:
    return "train" if index < train_demo_count else "validation"


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _positions(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] < PLATE_POS_SLICE.stop:
        raise ValueError(f"expected states shape [T, >={PLATE_POS_SLICE.stop}], got {states.shape}")
    return (
        states[:, TARGET_BOWL_POS_SLICE],
        states[:, RAMEKIN_POS_SLICE],
        states[:, PLATE_POS_SLICE],
    )


def _phase_labels(states: np.ndarray, config: SpatialTask5DataAuditConfig) -> dict[str, np.ndarray]:
    bowl, ramekin, plate = _positions(states)
    bowl_ramekin_xy = np.linalg.norm(bowl[:, :2] - ramekin[:, :2], axis=1)
    bowl_plate_xy = np.linalg.norm(bowl[:, :2] - plate[:, :2], axis=1)
    source_region = (bowl_ramekin_xy <= config.source_xy_threshold) & (
        bowl_plate_xy >= config.source_target_separation_min
    )
    target_region = bowl_plate_xy <= config.target_xy_threshold
    transit_region = ~(source_region | target_region)
    phase = np.where(source_region, "source_on_ramekin", np.where(target_region, "target_on_plate", "transit"))
    return {
        "bowl_ramekin_xy": bowl_ramekin_xy,
        "bowl_plate_xy": bowl_plate_xy,
        "source_region": source_region,
        "target_region": target_region,
        "transit_region": transit_region,
        "phase": phase,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def audit_spatial_task5_data(config: SpatialTask5DataAuditConfig) -> dict[str, Any]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.train_demo_count <= 0:
        raise ValueError("train_demo_count must be positive")
    if config.source_xy_threshold <= 0 or config.target_xy_threshold <= 0:
        raise ValueError("xy thresholds must be positive")
    if config.source_target_separation_min <= config.target_xy_threshold:
        raise ValueError("source_target_separation_min should exceed target_xy_threshold")
    if not config.hdf5_path.exists():
        raise FileNotFoundError(config.hdf5_path)

    with h5py.File(config.hdf5_path, "r") as h5:
        if "data" not in h5:
            raise ValueError("HDF5 file has no top-level 'data' group")
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        if len(demo_names) <= config.train_demo_count:
            raise ValueError("train_demo_count must leave at least one validation demo")

        records: list[dict[str, Any]] = []
        for demo_index, demo_name in enumerate(demo_names):
            group = h5["data"][demo_name]
            states = np.asarray(group["states"], dtype=np.float64)
            actions = np.asarray(group["actions"], dtype=np.float64)
            rewards = np.asarray(group["rewards"])
            dones = np.asarray(group["dones"])
            init_state = np.asarray(group.attrs["init_state"], dtype=np.float64).reshape(-1)
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
        "train": {"demo_count": 0, "step_count": 0, "chunk_count": 0, "phase_chunk_counts": {}},
        "validation": {"demo_count": 0, "step_count": 0, "chunk_count": 0, "phase_chunk_counts": {}},
    }
    terminal_reward_demos = 0
    terminal_done_demos = 0
    action_dim_set: set[int] = set()
    action_finite = True
    action_mins: list[float] = []
    action_maxs: list[float] = []
    overlap_hashes: list[str] = []
    residual_hashes = set(config.residual_initial_state_sha256)
    demo_summaries: list[dict[str, Any]] = []

    for record in records:
        split = str(record["split"])
        states = record["states"]
        actions = record["actions"]
        labels = _phase_labels(states, config)
        chunk_count = max(0, int(states.shape[0]) - config.chunk_size + 1)
        chunk_phases = labels["phase"][:chunk_count]
        phase_counts = _count_values(str(item) for item in chunk_phases)

        terminal_reward_demos += int(np.any(np.asarray(record["rewards"]) > 0))
        terminal_done_demos += int(np.any(np.asarray(record["dones"]) > 0))
        action_dim_set.add(int(actions.shape[1]) if actions.ndim == 2 else -1)
        action_finite = bool(action_finite and np.all(np.isfinite(actions)))
        action_mins.append(float(np.min(actions)))
        action_maxs.append(float(np.max(actions)))
        if record["init_state_sha256"] in residual_hashes:
            overlap_hashes.append(str(record["init_state_sha256"]))

        summary = split_summary[split]
        summary["demo_count"] += 1
        summary["step_count"] += int(states.shape[0])
        summary["chunk_count"] += int(chunk_count)
        for key, value in phase_counts.items():
            summary["phase_chunk_counts"][key] = int(summary["phase_chunk_counts"].get(key, 0)) + int(value)

        demo_summaries.append(
            {
                "demo_name": str(record["demo_name"]),
                "demo_index": int(record["demo_index"]),
                "split": split,
                "steps": int(states.shape[0]),
                "chunk_count": int(chunk_count),
                "init_state_sha256": str(record["init_state_sha256"]),
                "first_reward_index": record["first_reward_index"],
                "first_done_index": record["first_done_index"],
                "initial_bowl_ramekin_xy": round(float(labels["bowl_ramekin_xy"][0]), 9),
                "initial_bowl_plate_xy": round(float(labels["bowl_plate_xy"][0]), 9),
                "final_bowl_ramekin_xy": round(float(labels["bowl_ramekin_xy"][-1]), 9),
                "final_bowl_plate_xy": round(float(labels["bowl_plate_xy"][-1]), 9),
                "first_target_region_index": _first_index(labels["target_region"].astype(np.int64), 0.5),
                "phase_chunk_counts": phase_counts,
            }
        )

    for summary in split_summary.values():
        for key in ("source_on_ramekin", "transit", "target_on_plate"):
            summary["phase_chunk_counts"].setdefault(key, 0)

    demo_count = len(records)
    final_on_plate = [row["final_bowl_plate_xy"] <= config.target_xy_threshold for row in demo_summaries]
    initial_on_ramekin = [row["initial_bowl_ramekin_xy"] <= config.source_xy_threshold for row in demo_summaries]
    gate_checks = {
        "hdf5_has_50_demos": demo_count >= 50,
        "train_validation_split_nonempty": split_summary["train"]["demo_count"] > 0
        and split_summary["validation"]["demo_count"] > 0,
        "actions_are_7d": action_dim_set == {7},
        "actions_are_finite": action_finite,
        "actions_within_expected_range": max(abs(min(action_mins)), abs(max(action_maxs))) <= config.action_abs_max_gate,
        "all_demos_have_terminal_reward": terminal_reward_demos == demo_count,
        "all_demos_have_terminal_done": terminal_done_demos == demo_count,
        "initial_bowl_on_ramekin_all_demos": all(initial_on_ramekin),
        "final_bowl_on_plate_all_demos": all(final_on_plate),
        "train_has_source_transit_target_chunks": all(
            split_summary["train"]["phase_chunk_counts"].get(key, 0) > 0
            for key in ("source_on_ramekin", "transit", "target_on_plate")
        ),
        "validation_has_source_transit_target_chunks": all(
            split_summary["validation"]["phase_chunk_counts"].get(key, 0) > 0
            for key in ("source_on_ramekin", "transit", "target_on_plate")
        ),
        "initial_states_do_not_overlap_residual_failures": len(overlap_hashes) == 0,
    }
    passes = all(bool(value) for value in gate_checks.values())
    payload: dict[str, Any] = {
        "schema_version": "2026-07-18.xvla_spatial_task5_data_audit.v1",
        "task_suite": "libero_spatial",
        "task_id": 5,
        "task_description": "pick up the black bowl on the ramekin and place it on the plate",
        "inputs": {
            "hdf5_path": str(config.hdf5_path),
            "chunk_size": int(config.chunk_size),
            "train_demo_count": int(config.train_demo_count),
            "residual_initial_state_sha256": list(config.residual_initial_state_sha256),
            "state_layout": {
                "target_black_bowl_pos_slice": [TARGET_BOWL_POS_SLICE.start, TARGET_BOWL_POS_SLICE.stop],
                "ramekin_pos_slice": [RAMEKIN_POS_SLICE.start, RAMEKIN_POS_SLICE.stop],
                "plate_pos_slice": [PLATE_POS_SLICE.start, PLATE_POS_SLICE.stop],
            },
        },
        "policy": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "downloads_performed": False,
            "vla_model_loaded": False,
            "learned_policy_inference_performed": False,
            "ours_design_happened": False,
            "closed_loop_ours_evaluation_happened": False,
        },
        "dataset_summary": {
            "demo_count": int(demo_count),
            "terminal_reward_demo_count": int(terminal_reward_demos),
            "terminal_done_demo_count": int(terminal_done_demos),
            "action_dim_set": sorted(action_dim_set),
            "action_min": round(float(min(action_mins)), 6),
            "action_max": round(float(max(action_maxs)), 6),
            "action_max_abs": round(float(max(abs(min(action_mins)), abs(max(action_maxs)))), 6),
            "action_finite": bool(action_finite),
        },
        "split_summary": split_summary,
        "residual_overlap": {"overlap_hashes": overlap_hashes},
        "gate_checks": gate_checks,
        "passes_data_health_gate": bool(passes),
        "candidate_generation_readiness": bool(passes),
        "candidate_generation_happened": False,
        "training_authorized": False,
        "deployment_input_policy": {
            "privileged_state_at_inference": False,
            "phase_labels_training_only": True,
            "allowed_deployment_inputs": ["RGB", "wrist RGB", "proprioception", "instruction"],
        },
        "demo_summaries": demo_summaries,
    }
    if config.output_path is not None:
        _write_json(config.output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-initial-state-sha256", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--train-demo-count", type=int, default=40)
    args = parser.parse_args(argv)
    report = audit_spatial_task5_data(
        SpatialTask5DataAuditConfig(
            hdf5_path=args.hdf5_path,
            output_path=args.output,
            residual_initial_state_sha256=tuple(str(item) for item in args.residual_initial_state_sha256),
            chunk_size=int(args.chunk_size),
            train_demo_count=int(args.train_demo_count),
        )
    )
    print(json.dumps({"output": str(args.output), "passes_data_health_gate": report["passes_data_health_gate"]}, indent=2))
    return 0 if report["passes_data_health_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
