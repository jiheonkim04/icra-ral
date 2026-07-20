#!/usr/bin/env python3
"""Run the frozen feedback-expert development gate for latent dynamics."""

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
    target_contact_state,
)

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch8_latent_dynamics_feedback_development_protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/epoch8_latent_dynamics_feedback_development.json"
DEFAULT_LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
DEFAULT_PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def action_for_target(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip((desired - current) / 0.05, -1.0, 1.0)
    action[6] = -1.0
    return action


def run_episode(
    *,
    env_class: Any,
    bddl_path: Path,
    init_state: np.ndarray,
    task: dict[str, Any],
    simulator: dict[str, Any],
    controller: dict[str, Any],
    config: dict[str, Any],
    state_index: int,
    condition: str,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "state_index": int(state_index),
        "condition": condition,
        "completed": False,
        "success": False,
        "target_contact_any": False,
        "first_target_contact_step": None,
        "first_success_step": None,
        "finite_actions": True,
        "steps": 0,
        "phase_transitions": [],
        "error": None,
        "mutation": None,
    }
    try:
        env = env_class(
            bddl_file_name=str(bddl_path),
            camera_heights=int(simulator["camera_resolution"]),
            camera_widths=int(simulator["camera_resolution"]),
        )
        env.seed(102_000 + int(state_index))
        env.reset()
        observation = env.set_init_state(init_state)
        dummy = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(int(simulator["settle_steps_under_standard_dynamics"])):
            observation, _, _, _ = env.step(dummy)

        qpos_before = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
        qvel_before = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
        if condition == "intervention":
            row["mutation"] = apply_intervention(env.sim.model, task["intervention"])
            env.sim.forward()
        row["paired_state_residual"] = {
            "qpos_max_abs": float(np.max(np.abs(qpos_before - np.asarray(env.sim.data.qpos)))),
            "qvel_max_abs": float(np.max(np.abs(qvel_before - np.asarray(env.sim.data.qvel)))),
        }

        plate_initial = np.asarray(env.sim.data.get_body_xpos(task["target_body"]), dtype=np.float64).copy()
        goal = np.asarray(env.sim.data.get_site_xpos(task["goal_site"]), dtype=np.float64).copy()
        direction_xy = goal[:2] - plate_initial[:2]
        direction_norm = float(np.linalg.norm(direction_xy))
        if not np.isfinite(direction_norm) or direction_norm < 1e-6:
            raise ValueError("invalid plate-to-goal direction")
        direction_xy /= direction_norm

        behind = plate_initial.copy()
        behind[:2] -= direction_xy * float(config["behind_distance_m"])
        behind[2] = plate_initial[2] + float(controller["approach_height_m"])
        contact = behind.copy()
        contact[2] = plate_initial[2] + float(config["contact_height_m"])
        terminal = goal.copy()
        terminal[:2] += direction_xy * float(config["overshoot_m"])
        terminal[2] = plate_initial[2] + float(config["contact_height_m"])
        waypoints = [behind, contact, terminal]

        phase = 0
        phase_steps = 0
        success_streak = 0
        tolerance = float(controller["waypoint_tolerance_m"])
        min_phase_steps = int(controller["minimum_phase_steps"])
        max_phase_steps = int(controller["maximum_phase_steps"])
        max_steps = int(simulator["max_steps"])
        final_eef = None
        final_plate = None
        for step in range(max_steps):
            eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            desired = waypoints[min(phase, len(waypoints) - 1)]
            action = action_for_target(eef, desired)
            if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.0 + 1e-7:
                row["finite_actions"] = False
                raise ValueError("non-finite or out-of-range action")
            observation, _, _, _ = env.step(action)
            phase_steps += 1
            row["steps"] = int(step) + 1

            contact_state = target_contact_state(env.sim, str(task["target_body"]))
            if bool(contact_state["target_contact"]):
                row["target_contact_any"] = True
                if row["first_target_contact_step"] is None:
                    row["first_target_contact_step"] = int(step)

            success = bool(env.check_success())
            if success:
                if row["first_success_step"] is None:
                    row["first_success_step"] = int(step)
                success_streak += 1
            else:
                success_streak = 0
            if success_streak >= int(controller["success_hold_steps"]):
                row["success"] = True
                break

            new_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            reached = float(np.linalg.norm(new_eef - desired)) <= tolerance
            timed_out = phase_steps >= max_phase_steps
            if phase < len(waypoints) - 1 and phase_steps >= min_phase_steps and (reached or timed_out):
                row["phase_transitions"].append(
                    {
                        "from": int(phase),
                        "to": int(phase + 1),
                        "step": int(step),
                        "reason": "reached" if reached else "phase_timeout",
                        "distance_m": float(np.linalg.norm(new_eef - desired)),
                    }
                )
                phase += 1
                phase_steps = 0

        final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        final_plate = np.asarray(env.sim.data.get_body_xpos(task["target_body"]), dtype=np.float64).copy()
        row["final_official_success"] = bool(env.check_success())
        row["success"] = bool(row["success"] or row["final_official_success"])
        row["completed"] = True
        row["geometry"] = {
            "plate_initial": plate_initial.tolist(),
            "goal": goal.tolist(),
            "direction_xy": direction_xy.tolist(),
            "waypoints": [value.tolist() for value in waypoints],
            "final_eef": final_eef.tolist(),
            "final_plate": final_plate.tolist(),
            "final_plate_goal_xy_distance_m": float(np.linalg.norm(final_plate[:2] - goal[:2])),
        }
    except Exception as error:  # pragma: no cover - runtime failures only
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_NEW_FEEDBACK_EXPERT_OUTCOMES":
        raise ValueError("development protocol is not frozen")
    task = protocol["task"]
    init_path = Path(task["init_file"])
    if sha256_file(init_path) != task["init_file_sha256"]:
        raise ValueError("init file hash mismatch")

    package_root = str(args.libero_root / "libero")
    sys.path = [path for path in sys.path if path.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(args.libero_root))
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    states = torch.load(init_path, weights_only=False, map_location="cpu")
    development_indices = [int(value) for value in protocol["identity_partition"]["development_state_indices"]]
    forbidden = {
        int(value)
        for key in ("sealed_validation_state_indices", "sealed_confirmation_state_indices")
        for value in protocol["identity_partition"][key]
    }
    if set(development_indices) & forbidden:
        raise ValueError("development identity overlaps a sealed identity")

    result: dict[str, Any] = {
        "schema_version": "epoch8.latent_dynamics.feedback_development.v1",
        "started_at": timestamp(),
        "execution_type": "PRIVILEGED_FEEDBACK_FEASIBILITY_ORACLE_NO_POLICY",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "script_sha256": sha256_file(Path(__file__)),
        "policy_loaded_or_queried": False,
        "validation_or_confirmation_accessed": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "camera_resolution": int(protocol["simulator"]["camera_resolution"]),
        },
        "controller_attempts": [],
        "selected_controller": None,
    }

    bddl_path = args.para_root / "libero/libero/bddl_files/libero_goal" / task["goal_bddl"]
    for config in protocol["controller_grid_order"]:
        attempt: dict[str, Any] = {"controller": config, "pairs": [], "pass": False}
        failed = False
        for state_index in development_indices:
            pair: dict[str, Any] = {"state_index": state_index, "episodes": []}
            for condition in ("standard", "intervention"):
                episode = run_episode(
                    env_class=OffScreenRenderEnv,
                    bddl_path=bddl_path,
                    init_state=np.asarray(states[state_index], dtype=np.float64),
                    task=task,
                    simulator=protocol["simulator"],
                    controller=protocol["controller"],
                    config=config,
                    state_index=state_index,
                    condition=condition,
                )
                pair["episodes"].append(episode)
                atomic_write_json(args.output, result)
            pair["pass"] = all(
                episode["completed"]
                and episode["success"]
                and episode["target_contact_any"]
                and episode["finite_actions"]
                and episode["paired_state_residual"]["qpos_max_abs"] == 0.0
                and episode["paired_state_residual"]["qvel_max_abs"] == 0.0
                for episode in pair["episodes"]
            )
            attempt["pairs"].append(pair)
            if not pair["pass"]:
                failed = True
                break
        attempt["pass"] = not failed and len(attempt["pairs"]) == len(development_indices)
        result["controller_attempts"].append(attempt)
        atomic_write_json(args.output, result)
        if attempt["pass"]:
            result["selected_controller"] = config
            break

    result["completed_at"] = timestamp()
    if result["selected_controller"] is not None:
        result["decision"] = protocol["decisions"]["pass"]
    elif any(
        episode.get("error")
        for attempt in result["controller_attempts"]
        for pair in attempt["pairs"]
        for episode in pair["episodes"]
    ):
        result["decision"] = protocol["decisions"]["invalid"]
    else:
        result["decision"] = protocol["decisions"]["fail"]
    result["summary"] = {
        "attempted_controllers": len(result["controller_attempts"]),
        "executed_episodes": sum(
            len(pair["episodes"])
            for attempt in result["controller_attempts"]
            for pair in attempt["pairs"]
        ),
        "selected_controller_id": (
            result["selected_controller"]["id"] if result["selected_controller"] else None
        ),
        "sealed_indices_accessed": [],
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"decision": result["decision"], **result["summary"]}, sort_keys=True))
    return 0 if result["selected_controller"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
