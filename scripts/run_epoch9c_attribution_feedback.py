#!/usr/bin/env python3
"""Execute the frozen Epoch 9C planar-push feasibility oracle."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json, target_contact_state


PROTOCOL = ROOT / "reports/epoch9c_attribution_feedback_protocol.json"
OUTPUT = ROOT / "reports/epoch9c_attribution_feedback_result.json"
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def run_episode(env_class: Any, protocol: dict[str, Any], condition: str, state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    task = protocol["task"]
    control = protocol["controller"]
    env = None
    started = time.monotonic()
    row: dict[str, Any] = {
        "condition": condition,
        "completed": False,
        "error": None,
        "target_contact_any": False,
        "first_target_contact_step": None,
        "official_success": False,
        "finite_bounded_actions": True,
        "actions": [],
        "target_xy_trace_eval_only": [],
        "target_goal_distance_trace_m_eval_only": [],
        "mutation": None,
    }
    try:
        bddl = PARA_ROOT / "libero/libero/bddl_files/libero_goal" / task["goal_bddl"]
        env = env_class(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        observation = env.set_init_state(state)
        row["initial_qpos"] = np.asarray(env.sim.data.qpos, dtype=np.float64).tolist()
        row["initial_qvel"] = np.asarray(env.sim.data.qvel, dtype=np.float64).tolist()
        row["initial_agentview_sha256"] = array_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
        if condition == "latent_dynamics_intervention":
            row["mutation"] = apply_intervention(env.sim.model, task["intervention"])
            env.sim.forward()

        prefix_limit = min(int(control["maximum_prefix_steps"]), len(actions))
        contact_step = None
        for step in range(prefix_limit):
            action = np.asarray(actions[step], dtype=np.float32)
            observation, _, _, _ = env.step(action)
            row["actions"].append(action.tolist())
            contact = bool(target_contact_state(env.sim, task["target_body"])["target_contact"])
            if contact:
                row["target_contact_any"] = True
                row["first_target_contact_step"] = int(step)
                contact_step = int(step)
                break
        if contact_step is None:
            raise RuntimeError("frozen demonstration prefix did not contact target")

        gain = float(control["feedback_translation_gain"])
        for feedback_step in range(int(control["maximum_feedback_steps"])):
            target = np.asarray(env.sim.data.get_body_xpos(task["target_body"]), dtype=np.float64)
            goal = np.asarray(env.sim.data.get_site_xpos(task["goal_site"]), dtype=np.float64)
            delta = goal[:2] - target[:2]
            distance = float(np.linalg.norm(delta))
            row["target_xy_trace_eval_only"].append(target[:2].tolist())
            row["target_goal_distance_trace_m_eval_only"].append(distance)
            action = np.zeros(7, dtype=np.float32)
            if distance > 1e-9:
                action[:2] = (gain * delta / distance).astype(np.float32)
            action[6] = float(control["gripper_action"])
            row["finite_bounded_actions"] = bool(
                row["finite_bounded_actions"]
                and np.isfinite(action).all()
                and np.max(np.abs(action)) <= 1.0
            )
            observation, _, _, _ = env.step(action)
            row["actions"].append(action.tolist())
            if target_contact_state(env.sim, task["target_body"])["target_contact"]:
                row["target_contact_any"] = True
            if env.check_success():
                row["official_success"] = True
                row["first_success_feedback_step"] = int(feedback_step)
                break
        row["official_success"] = bool(row["official_success"] or env.check_success())
        row["completed"] = True
        row["final_target_xyz_eval_only"] = np.asarray(
            env.sim.data.get_body_xpos(task["target_body"]), dtype=np.float64
        ).tolist()
        row["final_goal_xyz_eval_only"] = np.asarray(
            env.sim.data.get_site_xpos(task["goal_site"]), dtype=np.float64
        ).tolist()
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    row["action_count"] = len(row["actions"])
    row.pop("actions", None)
    row["wall_seconds"] = float(time.monotonic() - started)
    return row


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sys.path.insert(0, str(LIBERO_ROOT))
    import h5py
    from libero.libero.envs import OffScreenRenderEnv

    with h5py.File(protocol["task"]["hdf5_path"], "r") as handle:
        demo = handle["data"][protocol["task"]["selected_standard_demo"]["demo_name"]]
        state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float32)
    rows = [
        run_episode(OffScreenRenderEnv, protocol, condition, state, actions)
        for condition in protocol["paired_conditions"]
    ]
    paired_state_exact = bool(
        np.array_equal(np.asarray(rows[0]["initial_qpos"]), np.asarray(rows[1]["initial_qpos"]))
        and np.array_equal(np.asarray(rows[0]["initial_qvel"]), np.asarray(rows[1]["initial_qvel"]))
    )
    first_observation_exact = rows[0]["initial_agentview_sha256"] == rows[1]["initial_agentview_sha256"]
    passed = bool(
        len(rows) == 2
        and all(row["completed"] and row["finite_bounded_actions"] for row in rows)
        and all(row["target_contact_any"] for row in rows)
        and all(row["official_success"] for row in rows)
        and paired_state_exact
        and first_observation_exact
    )
    result = {
        "schema_version": "epoch9c.attribution_feedback_result.v1",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(PROTOCOL),
        "execution_type": "PRIVILEGED_FEASIBILITY_ORACLE_NO_POLICY",
        "policy_loaded_or_queried": False,
        "sealed_validation_or_confirmation_accessed": False,
        "rows": rows,
        "paired_initial_state_exact": paired_state_exact,
        "first_observation_exact": first_observation_exact,
        "decision": (
            "ATTRIBUTION_THIRD_FAMILY_HEADROOM_RESTORED"
            if passed
            else "NO_DEFENSIBLE_LOCAL_PATH_AFTER_EMPIRICAL_ROTATIONS"
        ),
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({"decision": result["decision"], "successes": sum(row["official_success"] for row in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
