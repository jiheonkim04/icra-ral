#!/usr/bin/env python3
"""Audit real-demonstration supervision for selective language grounding.

The audit reads metadata and numeric action/state arrays only.  It performs no
model load, training, simulator execution, or outcome evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def balanced_section(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing {marker}")
    depth = 0
    seen_open = False
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
            seen_open = True
        elif text[index] == ")":
            depth -= 1
            if seen_open and depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unbalanced {marker}")


def normalized(section: str) -> str:
    return " ".join(section.lower().split())


def language(text: str) -> str:
    match = re.search(r"\(:language\s+([^\n\r\)]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("missing language")
    return " ".join(match.group(1).strip().lower().split())


def task_stem_from_demo(path: Path) -> str:
    suffix = "_demo"
    if not path.stem.endswith(suffix):
        raise ValueError(f"unexpected demo filename {path.name}")
    return path.stem[: -len(suffix)]


def audit_hdf5(path: Path) -> dict[str, Any]:
    lengths: list[int] = []
    action_chunks: list[np.ndarray] = []
    state_shapes: set[tuple[int, ...]] = set()
    robot_state_shapes: set[tuple[int, ...]] = set()
    finite = True
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"].keys())
        for name in demos:
            demo = handle["data"][name]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            robot_states = np.asarray(demo["robot_states"], dtype=np.float64)
            states = np.asarray(demo["states"], dtype=np.float64)
            lengths.append(int(actions.shape[0]))
            action_chunks.append(actions)
            state_shapes.add(tuple(int(value) for value in states.shape[1:]))
            robot_state_shapes.add(tuple(int(value) for value in robot_states.shape[1:]))
            finite = finite and bool(
                np.isfinite(actions).all()
                and np.isfinite(robot_states).all()
                and np.isfinite(states).all()
            )
        all_actions = np.concatenate(action_chunks, axis=0)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "demonstrations": len(lengths),
        "frames": int(sum(lengths)),
        "min_demo_frames": min(lengths),
        "max_demo_frames": max(lengths),
        "action_shape": [int(all_actions.shape[1])],
        "action_min": np.min(all_actions, axis=0).tolist(),
        "action_max": np.max(all_actions, axis=0).tolist(),
        "action_std": np.std(all_actions, axis=0).tolist(),
        "gripper_positive_fraction": float(np.mean(all_actions[:, -1] > 0)),
        "gripper_negative_fraction": float(np.mean(all_actions[:, -1] < 0)),
        "robot_state_tail_shapes": [list(shape) for shape in sorted(robot_state_shapes)],
        "sim_state_tail_shapes": [list(shape) for shape in sorted(state_shapes)],
        "finite_numeric_arrays": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/c/assets/data/libero"))
    parser.add_argument(
        "--bddl-root",
        type=Path,
        default=Path("/mnt/c/assets/repos/LIBERO-Para/libero/libero/bddl_files/libero_goal"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/mnt/c/assets/repos/LIBERO-Para/metrics/libero_para_metadata.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_files = sorted(args.data_root.rglob("*.hdf5"))
    suite_counts: dict[str, int] = {}
    for path in raw_files:
        suite_counts[path.parent.name] = suite_counts.get(path.parent.name, 0) + 1

    goal_files = sorted((args.data_root / "libero_goal").glob("*_demo.hdf5"))
    if len(goal_files) != 10:
        raise ValueError(f"expected 10 Goal HDF5 files, found {len(goal_files)}")

    records = []
    world_signatures: dict[str, str] = {}
    for demo_path in goal_files:
        stem = task_stem_from_demo(demo_path)
        bddl_path = args.bddl_root / f"{stem}.bddl"
        text = bddl_path.read_text(encoding="utf-8")
        signature_parts = [
            balanced_section(text, marker)
            for marker in ("(:regions", "(:fixtures", "(:objects", "(:init")
        ]
        world_signature = hashlib.sha256(
            normalized("\n".join(signature_parts)).encode("utf-8")
        ).hexdigest()
        world_signatures[stem] = world_signature
        records.append(
            {
                "task_stem": stem,
                "instruction": language(text),
                "bddl_path": str(bddl_path),
                "bddl_sha256": sha256(bddl_path),
                "world_signature": world_signature,
                "objects": normalized(balanced_section(text, "(:objects")),
                "goal": normalized(balanced_section(text, "(:goal")),
                "demonstrations": audit_hdf5(demo_path),
            }
        )

    metadata_rows = list(csv.DictReader(args.metadata.open(encoding="utf-8")))
    metadata_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for row in metadata_rows:
        key = str(int(row["eval"]))
        metadata_counts[key] = metadata_counts.get(key, 0) + 1
        family = str(row["high"])
        family_counts[family] = family_counts.get(family, 0) + 1

    total_frames = sum(record["demonstrations"]["frames"] for record in records)
    all_task_actions_noncollapsed = all(
        sum(float(value) > 1e-6 for value in record["demonstrations"]["action_std"]) >= 6
        for record in records
    )
    all_numeric_finite = all(
        record["demonstrations"]["finite_numeric_arrays"] for record in records
    )
    all_have_50 = all(record["demonstrations"]["demonstrations"] == 50 for record in records)
    shared_world = len(set(world_signatures.values())) == 1

    payload = {
        "schema_version": "epoch7.selectivity_supervision_audit.v1",
        "created_at": timestamp(),
        "execution_type": "OFFLINE_DIAGNOSTIC_NO_MODEL_NO_SIMULATOR_NO_OUTCOMES",
        "data_root": str(args.data_root),
        "raw_hdf5_inventory": {
            "files": len(raw_files),
            "bytes": sum(path.stat().st_size for path in raw_files),
            "suite_counts": dict(sorted(suite_counts.items())),
        },
        "goal_supervision": {
            "tasks": len(records),
            "demonstrations": sum(
                record["demonstrations"]["demonstrations"] for record in records
            ),
            "frames": total_frames,
            "all_tasks_have_50_demonstrations": all_have_50,
            "all_numeric_arrays_finite": all_numeric_finite,
            "all_task_action_arrays_noncollapsed": all_task_actions_noncollapsed,
            "shared_world_signature": shared_world,
            "distinct_goal_count": len({record["goal"] for record in records}),
            "legal_negative_instructions_per_positive": 9 if shared_world else 0,
            "records": records,
        },
        "equivalence_supervision": {
            "metadata_path": str(args.metadata),
            "metadata_sha256": sha256(args.metadata),
            "paraphrases": len(metadata_rows),
            "per_eval_id": metadata_counts,
            "per_family": family_counts,
        },
        "legality": {
            "positive_action_target": "real demonstrated action chunk for the factual instruction",
            "equivalent_positive_target": (
                "the same real demonstrated action chunk paired with a LIBERO-Para instruction "
                "whose metadata declares the same task intent"
            ),
            "negative_use": (
                "rank the factual action chunk as more compatible with its factual/equivalent "
                "instruction than with another feasible Goal instruction in the same declared world"
            ),
            "counterfactual_action_target_created": False,
            "negative_instruction_action_supervised_as_correct": False,
            "privileged_inference_input": False,
            "outcome_used_for_pair_selection": False,
            "required_safety_restriction": (
                "use negative ranking only on an early pre-interaction clip window so alternative "
                "Goal intents remain feasible; freeze the window before Ours outcomes"
            ),
        },
        "policy": {
            "model_loaded": False,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "simulator_episode_count": 0,
            "closed_loop_outcome_read": False,
            "ours_rollout_happened": False,
        },
    }
    passes = bool(
        len(raw_files) == 130
        and all_have_50
        and all_numeric_finite
        and all_task_actions_noncollapsed
        and shared_world
        and payload["goal_supervision"]["distinct_goal_count"] == 10
        and len(metadata_rows) == 4092
    )
    payload["decision"] = (
        "REAL_DEMONSTRATION_SELECTIVITY_SUPERVISION_LEGAL"
        if passes
        else "SELECTIVITY_SUPERVISION_AUDIT_FAIL"
    )
    atomic_write_json(args.output, payload)
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
