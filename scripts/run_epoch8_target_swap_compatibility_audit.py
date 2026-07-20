#!/usr/bin/env python3
"""Outcome-suppressed reset audit for frozen Epoch 8 target swaps.

The audit loads no policy and never calls success/reward logic. It verifies that
the source task's frozen simulator state can be installed into the target task's
goal BDDL for every predeclared directed intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")
SOURCE_AUDIT = ROOT / "reports/epoch7_selective_language_grounding/selectivity_supervision_audit.json"
SPLIT_MANIFEST = ROOT / "reports/epoch8_language_splits/confirmation_manifest.json"
DEFAULT_OUTPUT = ROOT / "reports/epoch8_language_splits/target_swap_compatibility_audit.json"
DIRECTED_PAIRS = ((3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3), (7, 8), (8, 7))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--init-index", type=int, default=14)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    if str(PARA_ROOT) not in sys.path:
        sys.path.insert(0, str(PARA_ROOT))
    from libero.libero.envs import OffScreenRenderEnv

    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    records = source_audit["goal_supervision"]["records"]
    if len(records) != 10 or not source_audit["goal_supervision"]["shared_world_signature"]:
        raise ValueError("inherited shared-world audit changed")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    frozen_pairs = {
        (int(row["state_source_eval_id"]), int(row["instruction_eval_id"]))
        for row in split["episodes_per_policy"]
        if row["condition"] == "target_swap"
    }
    if frozen_pairs != set(DIRECTED_PAIRS):
        raise ValueError(f"confirmation target swaps changed: {sorted(frozen_pairs)}")

    bddl_root = PARA_ROOT / "libero/libero/bddl_files/libero_goal"
    init_root = PARA_ROOT / "libero/libero/init_files/libero_goal"
    result: dict[str, Any] = {
        "schema_version": "epoch8.target_swap_compatibility_audit.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classification": "OUTCOME_SUPPRESSED_STATIC_AND_SIMULATOR_RESET_AUDIT",
        "candidate_independent": True,
        "ours_outcomes_observed": False,
        "model_loaded": False,
        "policy_forward_happened": False,
        "training_happened": False,
        "success_check_called": False,
        "reward_or_done_retained": False,
        "simulator_reset_count": 0,
        "one_live_environment_limit": 1,
        "init_index": args.init_index,
        "sources": {
            "inherited_audit": str(SOURCE_AUDIT.relative_to(ROOT)),
            "inherited_audit_sha256": sha256_file(SOURCE_AUDIT),
            "confirmation_manifest": str(SPLIT_MANIFEST.relative_to(ROOT)),
            "confirmation_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "libero_para_root": str(PARA_ROOT),
        },
        "directed_pairs": [],
    }

    for source_eval, target_eval in DIRECTED_PAIRS:
        source = records[source_eval]
        target = records[target_eval]
        source_init = init_root / f"{source['task_stem']}.pruned_init"
        target_bddl = bddl_root / f"{target['task_stem']}.bddl"
        row: dict[str, Any] = {
            "source_eval_id": source_eval,
            "target_eval_id": target_eval,
            "source_task_stem": source["task_stem"],
            "target_task_stem": target["task_stem"],
            "source_instruction": source["instruction"],
            "target_instruction": target["instruction"],
            "source_world_signature": source["world_signature"],
            "target_world_signature": target["world_signature"],
            "shared_world_signature": source["world_signature"] == target["world_signature"],
            "source_init_sha256": sha256_file(source_init),
            "target_bddl_sha256": sha256_file(target_bddl),
            "completed": False,
            "exception": None,
        }
        environment = None
        try:
            initial_states = torch.load(source_init, weights_only=False, map_location="cpu")
            if not 0 <= args.init_index < len(initial_states):
                raise IndexError(args.init_index)
            source_state = np.asarray(initial_states[args.init_index], dtype=np.float64).reshape(-1)
            environment = OffScreenRenderEnv(
                bddl_file_name=str(target_bddl), camera_heights=256, camera_widths=256
            )
            environment.seed(105)
            environment.reset()
            observation = environment.set_init_state(initial_states[args.init_index])
            installed = np.asarray(environment.env.sim.get_state().flatten(), dtype=np.float64).reshape(-1)
            if installed.shape != source_state.shape:
                raise ValueError(f"state shape changed: {source_state.shape} -> {installed.shape}")
            row.update(
                {
                    "completed": True,
                    "available_init_states": len(initial_states),
                    "source_state_shape": list(source_state.shape),
                    "installed_state_shape": list(installed.shape),
                    "source_state_finite": bool(np.isfinite(source_state).all()),
                    "installed_state_finite": bool(np.isfinite(installed).all()),
                    "installation_max_abs_residual": float(np.max(np.abs(installed - source_state))),
                    "agentview_shape": list(observation["agentview_image"].shape),
                    "eye_in_hand_shape": list(observation["robot0_eye_in_hand_image"].shape),
                }
            )
            result["simulator_reset_count"] += 1
        except Exception as exc:
            row["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            if environment is not None:
                environment.close()
        result["directed_pairs"].append(row)

    result["decision"] = (
        "TARGET_SWAP_RESETS_COMPATIBLE"
        if all(
            row["completed"]
            and row["shared_world_signature"]
            and row["source_state_finite"]
            and row["installed_state_finite"]
            for row in result["directed_pairs"]
        )
        else "TARGET_SWAP_RESET_COMPATIBILITY_FAILED"
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "decision": result["decision"], "resets": result["simulator_reset_count"]}, indent=2))
    return 0 if result["decision"] == "TARGET_SWAP_RESETS_COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
