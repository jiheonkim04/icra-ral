#!/usr/bin/env python3
"""Run the frozen standard-only expert selection and paired feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import (  # noqa: E402
    apply_intervention,
    atomic_write_json,
    load_json,
    target_contact_state,
    validate_protocol,
)

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_latent_dynamics_attribution/discovery_protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/epoch7_latent_dynamics_attribution/expert_feasibility.json"
DEFAULT_LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
DEFAULT_PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")
DEFAULT_DATA_ROOT = Path("/mnt/c/assets/data/libero/libero_goal")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def numeric_demo_key(value: str) -> tuple[int, str]:
    try:
        return int(str(value).rsplit("_", 1)[-1]), str(value)
    except ValueError:
        return 10**9, str(value)


def read_demo(path: Path, demo_name: str, max_steps: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64).reshape(-1)
        rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float64).reshape(-1)
        dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.float64).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"invalid expert action array for {path}::{demo_name}: {actions.shape}")
    positive = np.flatnonzero((rewards > 0.0) | (dones > 0.5))
    horizon = min(len(actions), int(max_steps))
    if positive.size:
        horizon = min(horizon, int(positive[0]) + 20)
    horizon = max(1, horizon)
    return {
        "demo_name": demo_name,
        "init_state": init_state,
        "actions": actions[:horizon],
        "full_action_steps": int(len(actions)),
        "replay_steps": int(horizon),
        "hdf5_first_signal_step": int(positive[0]) if positive.size else None,
        "action_sha256": hashlib.sha256(np.ascontiguousarray(actions[:horizon]).tobytes()).hexdigest(),
        "init_state_sha256": hashlib.sha256(np.ascontiguousarray(init_state).tobytes()).hexdigest(),
    }


def replay(
    *,
    env_class: Any,
    bddl_path: Path,
    demo: dict[str, Any],
    task: dict[str, Any],
    condition: str,
    camera_resolution: int,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "condition": condition,
        "completed": False,
        "success": False,
        "error": None,
        "steps": 0,
        "target_contact_any": False,
        "first_target_contact_step": None,
        "first_success_step": None,
        "mutation": None,
    }
    try:
        env = env_class(
            bddl_file_name=str(bddl_path),
            camera_heights=int(camera_resolution),
            camera_widths=int(camera_resolution),
        )
        env.seed(0)
        env.reset()
        env.set_init_state(demo["init_state"])
        if condition == "latent_dynamics_intervention":
            row["mutation"] = apply_intervention(env.sim.model, task["intervention"])
            env.sim.forward()
        for step, action in enumerate(demo["actions"]):
            _, reward, done, _ = env.step(action)
            contact = target_contact_state(env.sim, str(task["target_body"]))
            if contact["target_contact"]:
                row["target_contact_any"] = True
                if row["first_target_contact_step"] is None:
                    row["first_target_contact_step"] = int(step)
            success = bool(env.check_success())
            row["steps"] = int(step) + 1
            if success or bool(done) or float(reward) > 0.0:
                row["success"] = success or bool(done) or float(reward) > 0.0
                row["first_success_step"] = int(step)
                break
        row["completed"] = True
        row["final_official_success"] = bool(env.check_success())
        row["success"] = bool(row["success"] or row["final_official_success"])
    except Exception as error:  # pragma: no cover - runtime defects only
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--para-root", type=Path, default=DEFAULT_PARA_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--camera-resolution", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-standard-attempts", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError(f"invalid frozen protocol: {errors}")

    package_root = str(args.libero_root / "libero")
    sys.path = [path for path in sys.path if path.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(args.libero_root))
    import h5py
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = args.para_root / "libero/libero/bddl_files/libero_goal"
    if args.resume and args.output.exists():
        result = load_json(args.output)
        if result.get("schema_version") != "epoch7.latent_dynamics.expert_feasibility.v1":
            raise ValueError("existing output has incompatible schema")
        if result.get("protocol") != str(args.protocol):
            raise ValueError("existing output uses a different protocol")
        result["resumed_at"] = timestamp()
    else:
        result = {
            "schema_version": "epoch7.latent_dynamics.expert_feasibility.v1",
            "started_at": timestamp(),
            "execution_type": "FROZEN_STANDARD_ONLY_DEMO_SELECTION_PAIRED_EXPERT_REPLAY_NO_POLICY",
            "protocol": str(args.protocol),
            "selection_rule": protocol["expert_feasibility"]["selection_rule"],
            "policy_loaded_or_queried": False,
            "expert_actions_counted_as_policy_success": False,
            "runtime": {"python": platform.python_version(), "camera_resolution": int(args.camera_resolution)},
            "tasks": [],
        }

    completed_eval_ids = {int(row["eval_id"]) for row in result["tasks"]}

    for task in protocol["tasks"]:
        eval_id = int(task["eval_id"])
        if eval_id in completed_eval_ids:
            continue
        stem = Path(str(task["goal_bddl"])).stem
        hdf5_path = args.data_root / f"{stem}_demo.hdf5"
        with h5py.File(hdf5_path, "r") as handle:
            demo_names = sorted(handle["data"].keys(), key=numeric_demo_key)
        task_row: dict[str, Any] = {
            "eval_id": eval_id,
            "family": task["family"],
            "hdf5_path": str(hdf5_path),
            "standard_attempts": [],
            "selected_demo": None,
            "selection_used_intervention_outcomes": False,
            "standard_success": False,
            "intervention_success": False,
        }
        selected_demo: dict[str, Any] | None = None
        for demo_name in demo_names[: int(args.max_standard_attempts)]:
            demo = read_demo(hdf5_path, str(demo_name), int(args.max_steps))
            standard = replay(
                env_class=OffScreenRenderEnv,
                bddl_path=bddl_root / task["goal_bddl"],
                demo=demo,
                task=task,
                condition="standard",
                camera_resolution=int(args.camera_resolution),
            )
            task_row["standard_attempts"].append(
                {
                    "demo_name": demo_name,
                    "success": standard["success"],
                    "completed": standard["completed"],
                    "error": standard["error"],
                    "steps": standard["steps"],
                }
            )
            if standard["completed"] and standard["success"]:
                selected_demo = demo
                task_row["selected_demo"] = {
                    key: value for key, value in demo.items() if key not in {"actions", "init_state"}
                }
                task_row["standard"] = standard
                task_row["standard_success"] = True
                break
        if selected_demo is not None:
            intervention = replay(
                env_class=OffScreenRenderEnv,
                bddl_path=bddl_root / task["goal_bddl"],
                demo=selected_demo,
                task=task,
                condition="latent_dynamics_intervention",
                camera_resolution=int(args.camera_resolution),
            )
            task_row["intervention"] = intervention
            task_row["intervention_success"] = bool(intervention["completed"] and intervention["success"])
        result["tasks"].append(task_row)
        atomic_write_json(args.output, result)

    eligible = [row for row in result["tasks"] if row["standard_success"] and row["intervention_success"]]
    eligible_families = {str(row["family"]) for row in eligible}
    all_families = {
        "articulated" if str(row["family"]).startswith("articulated") else str(row["family"])
        for row in eligible
    }
    result["completed_at"] = timestamp()
    result["summary"] = {
        "tasks": len(result["tasks"]),
        "standard_success_tasks": sum(row["standard_success"] for row in result["tasks"]),
        "intervention_success_tasks": len(eligible),
        "eligible_task_ids": [row["eval_id"] for row in eligible],
        "eligible_families": sorted(eligible_families),
        "eligible_collapsed_families": sorted(all_families),
        "headroom_gate_pass": len(eligible) >= 3 and len(all_families) >= 3,
    }
    result["summary"]["decision"] = (
        "EXPERT_FEASIBILITY_PASS" if result["summary"]["headroom_gate_pass"] else "EXPERT_FEASIBILITY_FAIL"
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["headroom_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
