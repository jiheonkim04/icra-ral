#!/usr/bin/env python3
"""Preflight the action-sequence oracle port for the selected RL4IL prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import h5py
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.rl4il_prior.action_oracle import (
    ActionOracleConfig,
    oracle_index_for_candidates,
    pairwise_action_distance_matrix,
)


IDENTITY_BASE = 20260711
PANEL = [
    {
        "suite": "libero_goal",
        "task_id": 0,
        "instruction": "open the middle drawer of the cabinet",
        "hdf5": "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        "identities": [20260733, 20260734, 20260735],
    },
    {
        "suite": "libero_object",
        "task_id": 0,
        "instruction": "pick up the alphabet soup and place it in the basket",
        "hdf5": "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
        "identities": [20260733, 20260734, 20260735],
    },
    {
        "suite": "libero_spatial",
        "task_id": 5,
        "instruction": "pick up the black bowl on the ramekin and place it on the plate",
        "hdf5": "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "identities": [20260731, 20260732, 20260735],
    },
]


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def _demo_index(key: str) -> int:
    if not key.startswith("demo_"):
        raise ValueError(f"unexpected demo key {key!r}")
    return int(key.split("_", 1)[1])


def _load_actions_and_init_states(path: pathlib.Path) -> tuple[list[str], list[np.ndarray], dict[int, np.ndarray]]:
    keys: list[str] = []
    actions: list[np.ndarray] = []
    init_by_index: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as h:
        for key in sorted(h["data"].keys(), key=_demo_index):
            group = h["data"][key]
            keys.append(key)
            actions.append(np.asarray(group["actions"], dtype=np.float32))
            init_by_index[_demo_index(key)] = np.asarray(group.attrs["init_state"]).reshape(-1)
    return keys, actions, init_by_index


def run_preflight(dataset_root: pathlib.Path, config: ActionOracleConfig) -> dict[str, Any]:
    from libero.libero.benchmark import get_benchmark_dict

    benchmark_dict = get_benchmark_dict()
    tasks: list[dict[str, Any]] = []
    for item in PANEL:
        path = dataset_root / item["suite"] / item["hdf5"]
        keys, actions, hdf5_init_by_index = _load_actions_and_init_states(path)
        matrix = pairwise_action_distance_matrix(actions, config)
        nearest = []
        for i in range(len(keys)):
            oracle = oracle_index_for_candidates(i, range(len(keys)), matrix)
            nearest.append(
                {
                    "source_demo": keys[i],
                    "oracle_demo": keys[oracle],
                    "distance": float(matrix[i, oracle]),
                }
            )

        suite = benchmark_dict[item["suite"]]()
        official_init_states = suite.get_task_init_states(int(item["task_id"]))
        identity_checks = []
        for identity in item["identities"]:
            index = int(identity) - IDENTITY_BASE
            official = np.asarray(official_init_states[index]).reshape(-1)
            hdf5_init = hdf5_init_by_index.get(index)
            identity_checks.append(
                {
                    "reset_identity": int(identity),
                    "initial_state_index": int(index),
                    "official_init_shape": list(official.shape),
                    "official_init_sha256": _sha256_array(official),
                    "hdf5_demo_same_index_init_shape": None if hdf5_init is None else list(hdf5_init.shape),
                    "hdf5_same_index_l2_to_official": None
                    if hdf5_init is None or hdf5_init.shape != official.shape
                    else float(np.linalg.norm(hdf5_init - official)),
                }
            )

        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        tasks.append(
            {
                "suite": item["suite"],
                "task_id": item["task_id"],
                "instruction": item["instruction"],
                "hdf5": str(path),
                "demo_count": len(keys),
                "action_dim": int(actions[0].shape[1]),
                "action_length_min": int(min(action.shape[0] for action in actions)),
                "action_length_max": int(max(action.shape[0] for action in actions)),
                "pairwise_distance_min_offdiag": float(np.min(off_diag)),
                "pairwise_distance_mean_offdiag": float(np.mean(off_diag)),
                "nearest_oracle_unique_count": int(len({row["oracle_demo"] for row in nearest})),
                "nearest_oracle_preview": nearest[:5],
                "official_init_count": int(len(official_init_states)),
                "identity_checks": identity_checks,
            }
        )

    valid = all(
        task["demo_count"] == 50
        and task["action_dim"] == 7
        and task["nearest_oracle_unique_count"] > 1
        and task["pairwise_distance_min_offdiag"] > 0.0
        and len(task["identity_checks"]) == 3
        for task in tasks
    )

    return {
        "schema_version": "2026-07-18.epoch5_rl4il_action_oracle_port_preflight.v1",
        "decision": "RL4IL_ACTION_ORACLE_PORT_PREFLIGHT_PASS" if valid else "RL4IL_ACTION_ORACLE_PORT_PREFLIGHT_FAIL",
        "dataset_root": str(dataset_root),
        "identity_base": IDENTITY_BASE,
        "action_oracle_config": {
            "resample_steps": int(config.resample_steps),
            "length_penalty_weight": float(config.length_penalty_weight),
        },
        "tasks": tasks,
        "hdf5_demo_index_matches_official_reset_state": False,
        "live_query_required": True,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "simulator_rollout_happened": False,
        "ours_method_selected": False,
        "ours_training_happened": False,
        "ours_rollout_happened": False,
        "next_action": "Implement or launch the bounded RL4IL action-sequence-oracle prior runner with live initial-observation queries.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("/mnt/c/assets/data/libero"))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--resample-steps", type=int, default=64)
    parser.add_argument("--length-penalty-weight", type=float, default=0.01)
    args = parser.parse_args()

    result = run_preflight(
        args.dataset_root,
        ActionOracleConfig(
            resample_steps=int(args.resample_steps),
            length_penalty_weight=float(args.length_penalty_weight),
        ),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
