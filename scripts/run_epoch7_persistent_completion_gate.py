#!/usr/bin/env python3
"""Execute the frozen ten-task persistent-completion expert replay gate."""

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

from tca_map.epoch7_persistent_completion import (  # noqa: E402
    adjudicate_result,
    atomic_write_json,
    branch_signature,
    load_json,
    neutral_action,
    numeric_demo_key,
    persistence_summary,
    validate_protocol,
)

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_persistent_completion/problem_verification_protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/epoch7_persistent_completion/problem_verification_result.json"
DEFAULT_LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
DEFAULT_PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")
DEFAULT_DATA_ROOT = Path("/mnt/c/assets/data/libero/libero_goal")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_demo(path: Path, demo_name: str) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"invalid expert actions in {path.name}::{demo_name}: {actions.shape}")
    return {
        "demo_name": demo_name,
        "actions": actions,
        "init_state": init_state,
        "action_steps": int(len(actions)),
        "action_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "init_state_sha256": hashlib.sha256(np.ascontiguousarray(init_state).tobytes()).hexdigest(),
    }


def make_env(env_class: Any, bddl_path: Path, camera_resolution: int) -> Any:
    env = env_class(
        bddl_file_name=str(bddl_path),
        camera_heights=int(camera_resolution),
        camera_widths=int(camera_resolution),
    )
    env.seed(0)
    env.reset()
    return env


def native_replay(
    env_class: Any, bddl_path: Path, demo: dict[str, Any], camera_resolution: int
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "native_success": False,
        "first_success_step": None,
        "steps": 0,
        "error": None,
    }
    try:
        env = make_env(env_class, bddl_path, camera_resolution)
        env.set_init_state(demo["init_state"])
        for step, action in enumerate(demo["actions"]):
            env.step(action)
            row["steps"] = step + 1
            if bool(env.check_success()):
                row["native_success"] = True
                row["first_success_step"] = step
                break
    except Exception as error:  # pragma: no cover - runtime failures
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def run_branch(
    *,
    env_class: Any,
    bddl_path: Path,
    demo: dict[str, Any],
    camera_resolution: int,
    hold_steps: int,
    branch: str,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "branch": branch,
        "native_success": False,
        "first_success_step": None,
        "expert_steps_executed": 0,
        "hold_action": None,
        "hold_success_trace": [],
        "error": None,
    }
    try:
        env = make_env(env_class, bddl_path, camera_resolution)
        env.set_init_state(demo["init_state"])
        first_success_step: int | None = None
        success_action: np.ndarray | None = None
        final_expert_action: np.ndarray | None = None
        for step, action in enumerate(demo["actions"]):
            env.step(action)
            row["expert_steps_executed"] = step + 1
            final_expert_action = np.asarray(action, dtype=np.float64)
            if first_success_step is None and bool(env.check_success()):
                first_success_step = step
                success_action = final_expert_action.copy()
                row["native_success"] = True
                row["first_success_step"] = step
                if branch in ("immediate_neutral_hold", "last_action_repeat"):
                    break
        if first_success_step is not None:
            if branch == "last_action_repeat":
                hold_action = np.asarray(success_action, dtype=np.float64)
            elif branch == "immediate_neutral_hold":
                hold_action = neutral_action(success_action)
            elif branch == "expert_suffix_then_hold":
                if final_expert_action is None:
                    raise RuntimeError("expert suffix branch has no final action")
                hold_action = neutral_action(final_expert_action)
            else:
                raise ValueError(f"unknown branch: {branch}")
            row["hold_action"] = hold_action.tolist()
            trace: list[bool] = []
            for _ in range(hold_steps):
                env.step(hold_action)
                trace.append(bool(env.check_success()))
            row["hold_success_trace"] = trace
            row.update(persistence_summary(trace, hold_steps))
        else:
            row.update(persistence_summary([], hold_steps))
    except Exception as error:  # pragma: no cover - runtime failures
        row["error"] = f"{type(error).__name__}: {error}"
        row.update(persistence_summary(row.get("hold_success_trace", []), hold_steps))
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
    camera_resolution = int(protocol["benchmark"]["camera_resolution"])
    hold_steps = int(protocol["hold_contract"]["steps"])
    cold_repeat_ids = set(protocol["gates"]["execution"]["cold_repeat_task_ids"])

    if args.resume and args.output.exists():
        result = load_json(args.output)
        if result.get("schema_version") != "epoch7.persistent_completion.problem_result.v1":
            raise ValueError("existing output has incompatible schema")
        result["resumed_at"] = timestamp()
    else:
        result = {
            "schema_version": "epoch7.persistent_completion.problem_result.v1",
            "started_at": timestamp(),
            "execution_type": protocol["execution_type"],
            "protocol": str(args.protocol),
            "policy_loaded_or_queried": False,
            "ours_authorized_or_run": False,
            "dataset_rewards_or_dones_read": False,
            "expert_actions_counted_as_policy_success": False,
            "runtime": {
                "python": platform.python_version(),
                "camera_resolution": camera_resolution,
                "hold_steps": hold_steps,
            },
            "tasks": [],
        }

    completed_ids = {int(row["task_id"]) for row in result["tasks"] if row.get("completed")}
    for task in protocol["tasks"]:
        task_id = int(task["task_id"])
        if task_id in completed_ids:
            continue
        hdf5_path = args.data_root / task["hdf5"]
        bddl_path = bddl_root / f"{task['task_name']}.bddl"
        row: dict[str, Any] = {
            "task_id": task_id,
            "task_name": task["task_name"],
            "mechanism": task["mechanism"],
            "hdf5_path": str(hdf5_path),
            "bddl_path": str(bddl_path),
            "completed": False,
            "finite_actions": False,
            "error": None,
            "selection_attempts": [],
            "selected_demo": None,
            "branches": {},
            "cold_repeat_required": task_id in cold_repeat_ids,
            "cold_repeat_match": None,
        }
        try:
            if not hdf5_path.exists() or not bddl_path.exists():
                raise FileNotFoundError(f"missing artifact: {hdf5_path} or {bddl_path}")
            if sha256_file(hdf5_path) != str(task["hdf5_sha256"]).upper():
                raise ValueError(f"HDF5 hash mismatch for task {task_id}")
            with h5py.File(hdf5_path, "r") as handle:
                demo_names = sorted(handle["data"].keys(), key=numeric_demo_key)
            selected: dict[str, Any] | None = None
            for demo_name in demo_names[: int(protocol["selection"]["max_demo_attempts"])]:
                demo = load_demo(hdf5_path, demo_name)
                attempt = native_replay(OffScreenRenderEnv, bddl_path, demo, camera_resolution)
                row["selection_attempts"].append({"demo_name": demo_name, **attempt})
                if attempt["error"] is None and attempt["native_success"]:
                    selected = demo
                    break
            if selected is None:
                raise RuntimeError("no native-success demonstration found under frozen numeric selection")
            row["selected_demo"] = {
                key: value for key, value in selected.items() if key not in ("actions", "init_state")
            }
            row["finite_actions"] = bool(np.isfinite(selected["actions"]).all())
            for branch in (
                "immediate_neutral_hold",
                "expert_suffix_then_hold",
                "last_action_repeat",
            ):
                row["branches"][branch] = run_branch(
                    env_class=OffScreenRenderEnv,
                    bddl_path=bddl_path,
                    demo=selected,
                    camera_resolution=camera_resolution,
                    hold_steps=hold_steps,
                    branch=branch,
                )
            if row["cold_repeat_required"]:
                repeats = {}
                matches = []
                for branch in ("immediate_neutral_hold", "expert_suffix_then_hold"):
                    repeat = run_branch(
                        env_class=OffScreenRenderEnv,
                        bddl_path=bddl_path,
                        demo=selected,
                        camera_resolution=camera_resolution,
                        hold_steps=hold_steps,
                        branch=branch,
                    )
                    repeats[branch] = repeat
                    matches.append(
                        branch_signature(repeat) == branch_signature(row["branches"][branch])
                    )
                row["cold_repeats"] = repeats
                row["cold_repeat_match"] = bool(all(matches))
            row["completed"] = all(
                branch_row.get("error") is None for branch_row in row["branches"].values()
            ) and (row["cold_repeat_match"] is not False)
        except Exception as error:  # pragma: no cover - runtime failures
            row["error"] = f"{type(error).__name__}: {error}"
        result["tasks"].append(row)
        result["summary"] = adjudicate_result(protocol, result["tasks"])
        atomic_write_json(args.output, result)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "task_name": task["task_name"],
                    "completed": row["completed"],
                    "selected_demo": (row["selected_demo"] or {}).get("demo_name"),
                    "immediate_persistent": row["branches"].get("immediate_neutral_hold", {}).get("persistent_success"),
                    "suffix_persistent": row["branches"].get("expert_suffix_then_hold", {}).get("persistent_success"),
                    "error": row["error"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result["completed_at"] = timestamp()
    result["summary"] = adjudicate_result(protocol, result["tasks"])
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    return 0 if result["summary"]["execution_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
