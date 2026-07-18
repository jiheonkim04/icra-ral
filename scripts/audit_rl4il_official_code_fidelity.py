#!/usr/bin/env python3
"""Audit whether the released RL4IL scripts are faithful enough to arm a prior rollout.

This is deliberately a no-training/no-rollout audit. It checks the official
release for the specific supervision path needed by the external-prior
comparator gate: demonstration labels/action targets used to train retrieval
and fusion, checkpoint availability, and local LIBERO task loading.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np


PANEL_TASKS = [
    {
        "suite": "libero_goal",
        "hdf5": "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        "language": "open the middle drawer of the cabinet",
    },
    {
        "suite": "libero_object",
        "hdf5": "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
        "language": "pick up the alphabet soup and place it in the basket",
    },
    {
        "suite": "libero_spatial",
        "hdf5": "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "language": "pick up the black bowl on the ramekin and place it on the plate",
    },
]


SCRIPT_FILES = [
    "rl4il-sptl.py",
    "rl4il-obj.py",
    "rl4il-goal.py",
    "rl4il-epoch.py",
    "rl4il-topk.py",
]


def _git_head(repo: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_official_module(repo: pathlib.Path) -> Any:
    module_path = repo / "rl4il-sptl.py"
    spec = importlib.util.spec_from_file_location("rl4il_sptl_fidelity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _static_occurrences(repo: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    patterns = {
        "constant_label_assignment": '"label"',
        "train_policy": "def train_policy",
        "fusion_training_target": "tgt  = torch.tensor(float(tr_labels[i])",
        "rollout_action_replay": 'actions    = train_demos[best_tr_idx]["actions"]',
        "checkpoint_load_flag": "SKIP_TRAINING",
    }
    out: dict[str, list[dict[str, Any]]] = {key: [] for key in patterns}
    for file_name in SCRIPT_FILES:
        path = repo / file_name
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for key, needle in patterns.items():
                if needle in line:
                    out[key].append(
                        {
                            "file": file_name,
                            "line": lineno,
                            "text": line.strip(),
                        }
                    )
    return out


def _checkpoint_files(repo: pathlib.Path) -> list[str]:
    suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    return sorted(
        str(path.relative_to(repo))
        for path in repo.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def run_audit(repo: pathlib.Path, dataset_root: pathlib.Path) -> dict[str, Any]:
    module = _load_official_module(repo)
    static = _static_occurrences(repo)
    labels_by_task = []
    all_unique_values: list[float] = []

    for task in PANEL_TASKS:
        hdf5_path = dataset_root / task["suite"] / task["hdf5"]
        demos = module._load_task_demos(
            str(hdf5_path), task["language"], np.random.RandomState(42)
        )
        labels = [float(demo["label"]) for demo in demos]
        unique = sorted(set(labels))
        all_unique_values.extend(unique)
        labels_by_task.append(
            {
                "suite": task["suite"],
                "hdf5": str(hdf5_path),
                "demo_count": len(demos),
                "unique_label_values": unique,
                "label_count": len(labels),
            }
        )

    all_unique_values = sorted(set(all_unique_values))
    constant_label_supervision = len(all_unique_values) == 1
    checkpoints = _checkpoint_files(repo)
    rollout_not_armed = constant_label_supervision and not checkpoints

    decision = (
        "RL4IL_OFFICIAL_RELEASE_PRIOR_ROLLOUT_NOT_ARMED_CONSTANT_LABEL_SUPERVISION"
        if rollout_not_armed
        else "RL4IL_OFFICIAL_RELEASE_PRIOR_ROLLOUT_ARMABLE"
    )

    return {
        "schema_version": "2026-07-18.epoch5_rl4il_official_code_fidelity_audit.v1",
        "decision": decision,
        "stage": "epoch_5_external_prior_official_code_fidelity_audit",
        "official_repo": str(repo),
        "official_repo_head": _git_head(repo),
        "dataset_root": str(dataset_root),
        "panel_task_label_audit": labels_by_task,
        "all_unique_label_values": all_unique_values,
        "constant_label_supervision": constant_label_supervision,
        "checkpoint_files_found": checkpoints,
        "official_checkpoints_found": bool(checkpoints),
        "static_occurrences": static,
        "interpretation": {
            "paper_prior_status": "RL4IL remains the closest paper-level external prior for missing in-hand camera dropout.",
            "official_release_status": "The cloned official scripts are not armed for comparator rollout as-is because their demo loader assigns label=1 to every demo while retrieval/fusion training uses tr_labels as the supervised signal.",
            "why_it_matters": "With a single label value, validation accuracy and PPO candidate rewards can be degenerate; a rollout could replay actions from retrieved demos but would not establish the paper's learned action-signal retrieval/fusion mechanism.",
            "required_next_step": "Acquire/fix official checkpoints or implement a mechanism-faithful local port with an action-sequence oracle before using RL4IL as the prior comparator.",
        },
        "negative_controls": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "simulator_rollout_happened": False,
            "ours_method_selected": False,
            "ours_training_happened": False,
            "ours_rollout_happened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="/mnt/c/assets/repos/RL4IL-Missing-Camera",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--dataset-root",
        default="/mnt/c/assets/data/libero",
        type=pathlib.Path,
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    result = run_audit(args.repo, args.dataset_root)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
