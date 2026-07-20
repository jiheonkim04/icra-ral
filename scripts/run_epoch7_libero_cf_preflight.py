#!/usr/bin/env python3
"""Verify that the released LIBERO-CF task assets run in the retained runtime.

This is an artifact/runtime preflight only.  It does not load a VLA, execute
Ours, inspect a policy outcome, or create scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


SUITES = (
    "libero_cf_spatial",
    "libero_cf_spatial_focused",
    "libero_cf_object",
    "libero_cf_long",
    "libero_cf_ood",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_conditions(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected condition mapping in {path}")
    return len(payload), sum(len(conditions) for conditions in payload.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-cf-root", type=Path, default=Path("/mnt/c/assets/repos/LIBERO-CF")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_cf_spatial", choices=SUITES)
    parser.add_argument("--task-stem", default="01-pick_up_the_cookie_box_and_place_it_on_the_plate")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")

    root = args.libero_cf_root.resolve()
    # The repository intentionally uses a namespace-package layout:
    # <repo>/libero/libero/__init__.py is imported as ``libero.libero``.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    result: dict[str, Any] = {
        "schema_version": "epoch7.libero_cf_artifact_preflight.v1",
        "created_at": timestamp(),
        "execution_classification": "PRIOR_ARTIFACT_FEASIBILITY_NO_OURS",
        "scientific_outcome_inspected": False,
        "vla_loaded": False,
        "policy_forward_happened": False,
        "training_happened": False,
        "ours_design_happened": False,
        "ours_rollout_happened": False,
        "artifact": {
            "repository": "https://github.com/yuffish/LIBERO-CF",
            "root": str(root),
            "head": git_output(root, "rev-parse", "HEAD"),
            "working_tree_note": (
                "Windows Git reported a clean working tree immediately after acquisition; "
                "WSL Git status is intentionally omitted because the Windows checkout uses "
                "CRLF conversion."
            ),
            "license": "MIT",
            "license_sha256": sha256(root / "LICENSE"),
        },
        "suite_inventory": {},
        "environment_smoke": {
            "suite": args.suite,
            "task_stem": args.task_stem,
            "init_index": args.init_index,
            "settle_steps": args.settle_steps,
            "completed": False,
            "exception": None,
        },
    }

    libero_root = root / "libero" / "libero"
    for suite in SUITES:
        condition_path = libero_root / "conditions" / f"{suite}.json"
        condition_tasks, condition_count = count_conditions(condition_path)
        result["suite_inventory"][suite] = {
            "bddl_files": len(list((libero_root / "bddl_files" / suite).glob("*.bddl"))),
            "pruned_init_files": len(
                list((libero_root / "init_files" / suite).glob("*.pruned_init"))
            ),
            "condition_tasks": condition_tasks,
            "condition_predicates": condition_count,
            "condition_file_sha256": sha256(condition_path),
        }

    environment = None
    try:
        import libero as libero_namespace
        import torch

        source_namespace = str(root / "libero")
        if source_namespace not in libero_namespace.__path__:
            libero_namespace.__path__.insert(0, source_namespace)
        import libero.libero as libero_core

        artifact_paths = {
            "benchmark_root": str(libero_root),
            "bddl_files": str(libero_root / "bddl_files"),
            "init_states": str(libero_root / "init_files"),
            "datasets": str(root / "libero" / "datasets"),
            "assets": str(libero_root / "assets"),
            "conditions": str(libero_root / "conditions"),
        }

        def artifact_get_libero_path(query_key: str) -> str:
            if query_key not in artifact_paths:
                raise KeyError(query_key)
            return artifact_paths[query_key]

        # Patch before importing the environment modules, which bind this
        # function at module import time.  This is a path-routing repair only.
        libero_core.get_libero_path = artifact_get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        imported_libero_core = Path(sys.modules["libero.libero"].__file__).resolve()
        imported_envs = Path(sys.modules["libero.libero.envs"].__file__).resolve()
        bddl = libero_root / "bddl_files" / args.suite / f"{args.task_stem}.bddl"
        init = libero_root / "init_files" / args.suite / f"{args.task_stem}.pruned_init"
        initial_states = torch.load(init, weights_only=False, map_location="cpu")
        if not 0 <= args.init_index < len(initial_states):
            raise IndexError(f"init index {args.init_index} outside 0..{len(initial_states) - 1}")
        environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        environment.seed(7)
        environment.reset()
        observation = environment.set_init_state(initial_states[args.init_index])
        dummy_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(args.settle_steps):
            observation, _, _, _ = environment.step(dummy_action)

        # Do not call check_success(), and do not retain reward/done from the
        # technical settling trace: this preflight is outcome-suppressed.
        result["environment_smoke"].update(
            {
                "completed": True,
                "imported_libero_namespace": str(Path(libero_namespace.__file__).resolve()),
                "imported_libero_core": str(imported_libero_core),
                "imported_envs": str(imported_envs),
                "bddl_sha256": sha256(bddl),
                "init_sha256": sha256(init),
                "available_init_states": len(initial_states),
                "agentview_shape": list(observation["agentview_image"].shape),
                "eye_in_hand_shape": list(observation["robot0_eye_in_hand_image"].shape),
                "robot_count": len(environment.env.robots),
                "dummy_actions_executed": args.settle_steps,
                "sim_state_finite": bool(
                    np.isfinite(np.asarray(environment.env.sim.get_state().flatten())).all()
                ),
            }
        )
    except Exception as exc:
        result["environment_smoke"]["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if environment is not None:
            environment.close()

    result["decision"] = (
        "LIBERO_CF_RETAINED_RUNTIME_PREFLIGHT_PASS"
        if result["environment_smoke"]["completed"]
        else "LIBERO_CF_RETAINED_RUNTIME_PREFLIGHT_FAIL"
    )
    atomic_write_json(args.output, result)
    return 0 if result["environment_smoke"]["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
