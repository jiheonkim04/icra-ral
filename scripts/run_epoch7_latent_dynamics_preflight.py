#!/usr/bin/env python3
"""Outcome-free simulator preflight for the frozen latent-dynamics thesis."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
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
    body_descendant_ids,
    compare_observations,
    load_json,
    target_contact_state,
    validate_protocol,
)

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_latent_dynamics_attribution/discovery_protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/epoch7_latent_dynamics_attribution/outcome_free_simulator_preflight.json"
DEFAULT_LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
DEFAULT_PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def _hash_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _render_model_hashes(model: Any) -> dict[str, str]:
    names = (
        "body_pos",
        "body_quat",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_rgba",
        "geom_matid",
        "cam_pos",
        "cam_quat",
        "light_pos",
        "light_dir",
        "light_diffuse",
        "light_ambient",
        "light_specular",
    )
    return {name: _hash_array(getattr(model, name)) for name in names if hasattr(model, name)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--para-root", type=Path, default=DEFAULT_PARA_ROOT)
    parser.add_argument("--camera-resolution", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise ValueError(f"invalid frozen protocol: {protocol_errors}")

    # The retained simulation environment has an editable LIBERO path. Remove
    # the package-root entry and expose the repository root so LIBERO's own
    # ``libero.libero`` imports resolve exactly as in the frozen evaluators.
    package_root = str(args.libero_root / "libero")
    sys.path = [path for path in sys.path if path.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(args.libero_root))

    import torch
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = args.para_root / "libero/libero/bddl_files/libero_goal"
    init_root = args.para_root / "libero/libero/init_files/libero_para"
    result: dict[str, Any] = {
        "schema_version": "epoch7.latent_dynamics.outcome_free_preflight.v1",
        "started_at": timestamp(),
        "execution_type": "OUTCOME_FREE_NO_MODEL_NO_TASK_OUTCOME",
        "protocol": str(args.protocol),
        "protocol_schema": protocol["schema_version"],
        "ours_authorized": False,
        "model_loaded": False,
        "policy_actions_generated": False,
        "reward_or_success_recorded": False,
        "runtime": {
            "python": platform.python_version(),
            "pid": os.getpid(),
            "torch": torch.__version__,
            "camera_resolution": int(args.camera_resolution),
        },
        "tasks": [],
    }

    dummy_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for task in protocol["tasks"]:
        eval_id = int(task["eval_id"])
        env = None
        row: dict[str, Any] = {
            "eval_id": eval_id,
            "family": task["family"],
            "goal_bddl": task["goal_bddl"],
            "target_body": task["target_body"],
            "intervention": task["intervention"],
            "exception": None,
        }
        try:
            states = torch.load(
                init_root / f"eval{eval_id}.pruned_init",
                weights_only=False,
                map_location="cpu",
            )
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_root / task["goal_bddl"]),
                camera_heights=int(args.camera_resolution),
                camera_widths=int(args.camera_resolution),
            )
            env.seed(100_000 + 1_000 * eval_id)
            env.reset()
            observation = env.set_init_state(states[0])
            for _ in range(int(protocol["benchmark"]["settle_steps_under_standard_dynamics"])):
                observation, _, _, _ = env.step(dummy_action)
            for robot in env.env.robots:
                robot.controller.use_delta = False

            standard_observation = _copy_observation(observation)
            paired_cached_observation = _copy_observation(observation)
            state_before = {
                "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
                "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).copy(),
            }
            render_hashes_before = _render_model_hashes(env.sim.model)
            no_op = apply_intervention(env.sim.model, task["intervention"], factor_override=1.0)
            env.sim.forward()
            no_op_equivalence = compare_observations(standard_observation, paired_cached_observation)

            intervention = apply_intervention(env.sim.model, task["intervention"])
            env.sim.forward()
            intervention_equivalence = compare_observations(standard_observation, paired_cached_observation)
            state_after = {
                "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
                "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).copy(),
            }
            state_equivalence = {
                "qpos_max_abs": float(np.max(np.abs(state_before["qpos"] - state_after["qpos"]))),
                "qvel_max_abs": float(np.max(np.abs(state_before["qvel"] - state_after["qvel"]))),
            }
            render_hashes_after = _render_model_hashes(env.sim.model)
            render_hash_equivalence = {
                "before": render_hashes_before,
                "after": render_hashes_after,
                "unchanged": render_hashes_before == render_hashes_after,
            }
            contact = target_contact_state(env.sim, str(task["target_body"]))
            descendants = body_descendant_ids(env.sim.model, str(task["target_body"]))

            row.update(
                {
                    "state_index": 0,
                    "initial_state_count": len(states),
                    "no_op_mutation": no_op,
                    "no_op_observation_equivalence": no_op_equivalence,
                    "intervention_mutation": intervention,
                    "intervention_observation_equivalence": intervention_equivalence,
                    "sim_state_equivalence": state_equivalence,
                    "render_model_hash_equivalence": render_hash_equivalence,
                    "observation_regenerated_after_mutation": False,
                    "target_body_ids": sorted(descendants),
                    "initial_contact_snapshot": contact,
                    "pass": (
                        no_op["changed_values"] == 0
                        and no_op_equivalence["eligible"]
                        and intervention["changed_values"] > 0
                        and intervention_equivalence["eligible"]
                        and state_equivalence["qpos_max_abs"] == 0.0
                        and state_equivalence["qvel_max_abs"] == 0.0
                        and render_hash_equivalence["unchanged"]
                        and bool(descendants)
                    ),
                }
            )
        except Exception as error:  # pragma: no cover - exercised only by runtime defects
            row["exception"] = f"{type(error).__name__}: {error}"
            row["pass"] = False
        finally:
            if env is not None:
                env.close()
        result["tasks"].append(row)
        atomic_write_json(args.output, result)

    result["completed_at"] = timestamp()
    result["summary"] = {
        "tasks": len(result["tasks"]),
        "passed": sum(row.get("pass") is True for row in result["tasks"]),
        "exceptions": sum(row.get("exception") is not None for row in result["tasks"]),
        "decision": "OUTCOME_FREE_SIMULATOR_PREFLIGHT_PASS"
        if all(row.get("pass") is True for row in result["tasks"])
        else "OUTCOME_FREE_SIMULATOR_PREFLIGHT_FAIL",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["decision"] == "OUTCOME_FREE_SIMULATOR_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
