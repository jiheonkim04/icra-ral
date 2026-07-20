#!/usr/bin/env python3
"""Develop and evaluate legal active-property probe/return controllers.

Simulator object state and expert demonstrations are used only by explicitly
named development diagnostics and evaluation metrics.  Controller actions are
computed from fixed slot calibration, ordinary RGB/proprioception, elapsed
steps, and the controller's own command history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json, target_contact_state

LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
BDDL_ROOT = LIBERO_ROOT / "libero/libero/bddl_files/libero_90"
DATA_ROOT = Path("/mnt/c/assets/data/libero/libero_90")
OUTPUT_ROOT = REPO_ROOT / "reports/epoch9_controller_development"

TASKS = {
    "front": {
        "body": "akita_black_bowl_1_main",
        "bddl": "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.bddl",
        "hdf5": "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_demo.hdf5",
    },
    "back": {
        "body": "akita_black_bowl_3_main",
        "bddl": "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.bddl",
        "hdf5": "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_demo.hdf5",
    },
}

# Fixed slot calibration is learned only from development demos 30..32.  The
# controller never reads a simulator object pose online.
SLOT_CALIBRATION = {
    "front": {
        "contact_eef": np.asarray([0.0883, 0.1763, 0.9231], dtype=np.float64),
        "push_delta": np.asarray([0.0, -0.018, 0.0], dtype=np.float64),
        "pixel_center_xy": (95, 62),
    },
    "back": {
        "contact_eef": np.asarray([-0.1593, -0.0012, 0.9218], dtype=np.float64),
        "push_delta": np.asarray([0.0, 0.018, 0.0], dtype=np.float64),
        "pixel_center_xy": (70, 57),
    },
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def read_demo(task: dict[str, str], demo_index: int) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(DATA_ROOT / task["hdf5"], "r") as handle:
        demo = handle["data"][f"demo_{demo_index}"]
        return (
            np.asarray(demo.attrs["init_state"], dtype=np.float64),
            np.asarray(demo["actions"], dtype=np.float32),
        )


def load_env_class() -> Any:
    package_root = str(LIBERO_ROOT / "libero")
    sys.path = [value for value in sys.path if value.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(LIBERO_ROOT))
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def make_env(env_class: Any, task: dict[str, str], init_state: np.ndarray, resolution: int = 128) -> tuple[Any, dict[str, Any]]:
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / task["bddl"]),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    env.seed(9009)
    env.reset()
    observation = env.set_init_state(init_state)
    for _ in range(10):
        observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    return env, observation


def expert_contact_diagnostic(demo_indices: list[int]) -> dict[str, Any]:
    env_class = load_env_class()
    result: dict[str, Any] = {
        "schema_version": "epoch9.probe_waypoint_diagnostic.v1",
        "timestamp": timestamp(),
        "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
        "online_policy_authorized": False,
        "simulator_state_role": "offline waypoint calibration and metric evaluation only",
        "rows": [],
    }
    for slot, task in TASKS.items():
        for demo_index in demo_indices:
            env = None
            row: dict[str, Any] = {"slot": slot, "demo_index": int(demo_index), "exception": None}
            try:
                init_state, actions = read_demo(task, demo_index)
                env, observation = make_env(env_class, task, init_state)
                if demo_index == demo_indices[0]:
                    from PIL import Image

                    frame_root = OUTPUT_ROOT / "frames"
                    frame_root.mkdir(parents=True, exist_ok=True)
                    for key, camera in (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "wrist")):
                        if key in observation:
                            destination = frame_root / f"{slot}_demo{demo_index}_{camera}_initial.png"
                            temporary = destination.with_suffix(".tmp.png")
                            Image.fromarray(np.asarray(observation[key], dtype=np.uint8)).save(temporary)
                            temporary.replace(destination)
                initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
                initial_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
                initial_target = np.asarray(env.sim.data.get_body_xpos(task["body"]), dtype=np.float64).copy()
                first_contact = None
                pose_trace = [{"step": -1, "eef_pos": initial_eef.tolist(), "eef_quat": initial_quat.tolist()}]
                for step, action in enumerate(actions):
                    observation, _, _, _ = env.step(action)
                    pose_trace.append(
                        {
                            "step": int(step),
                            "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).tolist(),
                            "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).tolist(),
                        }
                    )
                    if bool(target_contact_state(env.sim, task["body"])["target_contact"]):
                        first_contact = {
                            "step": int(step),
                            "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).tolist(),
                            "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).tolist(),
                            "controller_goal_pos": np.asarray(env.env.robots[0].controller.goal_pos, dtype=np.float64).tolist(),
                            "controller_ee_pos": np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float64).tolist(),
                            "action": np.asarray(action, dtype=np.float64).tolist(),
                        }
                        break
                row.update(
                    {
                        "completed": True,
                        "action_source_sha256": array_sha256(actions),
                        "initial_eef": initial_eef.tolist(),
                        "initial_target_eval_only": initial_target.tolist(),
                        "first_contact": first_contact,
                        "cartesian_pose_waypoints": [
                            value
                            for index, value in enumerate(pose_trace)
                            if index == 0 or index == len(pose_trace) - 1 or index % 5 == 0
                        ],
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime diagnostic
                row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
            finally:
                if env is not None:
                    env.close()
            result["rows"].append(row)
            atomic_write_json(OUTPUT_ROOT / "expert_contact_diagnostic.json", result)
    valid = [row for row in result["rows"] if row.get("completed") and row.get("first_contact")]
    result["summary"] = {
        "requested_rows": len(TASKS) * len(demo_indices),
        "valid_contact_rows": len(valid),
        "all_contact": len(valid) == len(TASKS) * len(demo_indices),
    }
    atomic_write_json(OUTPUT_ROOT / "expert_contact_diagnostic.json", result)
    return result


def _gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def template_shift(initial_frame: np.ndarray, current_frame: np.ndarray, slot: str) -> dict[str, float]:
    """Estimate candidate pixel displacement without simulator segmentation."""

    import cv2

    x, y = SLOT_CALIBRATION[slot]["pixel_center_xy"]
    initial = _gray(initial_frame)
    current = _gray(current_frame)
    radius = 8
    search = 12
    template = initial[y - radius : y + radius + 1, x - radius : x + radius + 1]
    region = current[
        y - radius - search : y + radius + search + 1,
        x - radius - search : x + radius + search + 1,
    ]
    score = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, quality, _, location = cv2.minMaxLoc(score)
    best_x = int(location[0]) + x - search
    best_y = int(location[1]) + y - search
    return {"dx": float(best_x - x), "dy": float(best_y - y), "quality": float(quality)}


def _trace_step(
    trace: dict[str, list[Any]],
    env: Any,
    observation: dict[str, Any],
    action: np.ndarray,
    phase: str,
    initial_gray_small: np.ndarray,
    eval_target_body: str | None = None,
) -> None:
    import cv2

    controller = env.env.robots[0].controller
    frame = _gray(np.asarray(observation["agentview_image"], dtype=np.uint8))
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    trace["phase"].append(phase)
    trace["action"].append(np.asarray(action, dtype=np.float32).copy())
    trace["eef_pos"].append(np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy())
    trace["eef_quat"].append(np.asarray(observation["robot0_eef_quat"], dtype=np.float32).copy())
    trace["controller_goal_pos"].append(np.asarray(controller.goal_pos, dtype=np.float32).copy())
    trace["controller_error"].append(float(np.linalg.norm(np.asarray(controller.goal_pos) - np.asarray(controller.ee_pos))))
    trace["rgb_diff_32"].append((small - initial_gray_small).astype(np.int16))
    trace["target_contact_eval"].append(
        bool(target_contact_state(env.sim, eval_target_body)["target_contact"])
        if eval_target_body is not None
        else False
    )


def feedback_move(
    env: Any,
    observation: dict[str, Any],
    target: np.ndarray,
    *,
    gripper: float,
    phase: str,
    trace: dict[str, list[Any]],
    initial_gray_small: np.ndarray,
    max_steps: int = 70,
    tolerance_m: float = 0.004,
    target_quat: np.ndarray | None = None,
    absolute_pose: bool = False,
    eval_target_body: str | None = None,
) -> tuple[dict[str, Any], bool]:
    stable = 0
    for _ in range(max_steps):
        current = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        error = np.asarray(target, dtype=np.float64) - current
        action = np.zeros(7, dtype=np.float32)
        if absolute_pose:
            from robosuite.utils import transform_utils as transform

            env.env.robots[0].controller.use_delta = False
            action[:3] = np.asarray(target, dtype=np.float32)
            if target_quat is not None:
                action[3:6] = transform.quat2axisangle(np.asarray(target_quat, dtype=np.float64)).astype(np.float32)
        else:
            action[:3] = np.clip(error / 0.04, -1.0, 1.0).astype(np.float32)
        if target_quat is not None and not absolute_pose:
            from robosuite.utils import transform_utils as transform
            from robosuite.utils.control_utils import orientation_error

            current_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
            desired_matrix = transform.quat2mat(np.asarray(target_quat, dtype=np.float64))
            current_matrix = transform.quat2mat(current_quat)
            rotation_error = orientation_error(desired_matrix, current_matrix)
            action[3:6] = np.clip(rotation_error / 0.5, -1.0, 1.0).astype(np.float32)
        action[6] = float(gripper)
        observation, _, _, _ = env.step(action)
        _trace_step(trace, env, observation, action, phase, initial_gray_small, eval_target_body)
        stable = stable + 1 if float(np.linalg.norm(error)) <= tolerance_m else 0
        if stable >= 3:
            return observation, True
    return observation, False


def feedback_hold(
    env: Any,
    observation: dict[str, Any],
    target: np.ndarray,
    *,
    phase: str,
    steps: int,
    trace: dict[str, list[Any]],
    initial_gray_small: np.ndarray,
    target_quat: np.ndarray | None = None,
    eval_target_body: str | None = None,
) -> dict[str, Any]:
    for _ in range(steps):
        current = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((np.asarray(target) - current) / 0.04, -1.0, 1.0).astype(np.float32)
        if target_quat is not None:
            from robosuite.utils import transform_utils as transform
            from robosuite.utils.control_utils import orientation_error

            current_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
            desired_matrix = transform.quat2mat(np.asarray(target_quat, dtype=np.float64))
            current_matrix = transform.quat2mat(current_quat)
            rotation_error = orientation_error(desired_matrix, current_matrix)
            action[3:6] = np.clip(rotation_error / 0.5, -1.0, 1.0).astype(np.float32)
        action[6] = 1.0
        observation, _, _, _ = env.step(action)
        _trace_step(trace, env, observation, action, phase, initial_gray_small, eval_target_body)
    return observation


def persist_trace(path: Path, trace: dict[str, list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            phase=np.asarray(trace["phase"]),
            action=np.asarray(trace["action"], dtype=np.float32),
            eef_pos=np.asarray(trace["eef_pos"], dtype=np.float32),
            eef_quat=np.asarray(trace["eef_quat"], dtype=np.float32),
            controller_goal_pos=np.asarray(trace["controller_goal_pos"], dtype=np.float32),
            controller_error=np.asarray(trace["controller_error"], dtype=np.float32),
            rgb_diff_32=np.asarray(trace["rgb_diff_32"], dtype=np.int16),
            target_contact_eval=np.asarray(trace["target_contact_eval"], dtype=bool),
        )
    temporary.replace(path)


def controller_episode(
    env_class: Any,
    *,
    slot: str,
    demo_index: int,
    factor: float,
    attempt_id: str,
) -> dict[str, Any]:
    task = TASKS[slot]
    env = None
    row: dict[str, Any] = {
        "slot": slot,
        "demo_index": int(demo_index),
        "mass_factor": float(factor),
        "attempt_id": attempt_id,
        "exception": None,
    }
    trace: dict[str, list[Any]] = {
        "phase": [],
        "action": [],
        "eef_pos": [],
        "eef_quat": [],
        "controller_goal_pos": [],
        "controller_error": [],
        "rgb_diff_32": [],
        "target_contact_eval": [],
    }
    try:
        init_state, _ = read_demo(task, demo_index)
        env, observation = make_env(env_class, task, init_state)
        initial_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
        import cv2

        initial_gray_small = cv2.resize(_gray(initial_frame), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
        initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        initial_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
        initial_target = np.asarray(env.sim.data.get_body_xpos(task["body"]), dtype=np.float64).copy()
        if factor != 1.0:
            apply_intervention(
                env.sim.model,
                {"axis": "target_mass", "body_name": task["body"], "arrays": ["body_mass", "body_inertia"], "factor": factor},
            )
            env.sim.forward()

        calibration = SLOT_CALIBRATION[slot]
        contact = np.asarray(calibration["contact_eef"], dtype=np.float64)
        variant_v10 = attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
        if variant_v10 and slot == "front":
            contact = np.asarray([0.050 if attempt_id.startswith(("v11_", "v12_")) else 0.060, 0.169, 0.926], dtype=np.float64)
        above = contact.copy()
        above[2] = 1.02
        variant_v2 = attempt_id.startswith("v2_") or attempt_id.startswith("v3_") or attempt_id.startswith("v4_") or attempt_id.startswith("v5_") or attempt_id.startswith("v6_") or attempt_id.startswith("v7_") or attempt_id.startswith("v8_") or attempt_id.startswith("v9_") or attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
        variant_v3 = attempt_id.startswith("v3_") or attempt_id.startswith("v4_") or attempt_id.startswith("v5_") or attempt_id.startswith("v6_") or attempt_id.startswith("v7_") or attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
        variant_v4 = attempt_id.startswith("v4_")
        variant_v5 = attempt_id.startswith("v5_") or attempt_id.startswith("v6_")
        variant_v6 = attempt_id.startswith("v6_")
        variant_v7 = attempt_id.startswith("v7_")
        variant_v8 = attempt_id.startswith("v8_") or attempt_id.startswith("v9_")
        variant_v9 = attempt_id.startswith("v9_")
        push_scale = 2.0 / 3.0 if variant_v2 else 1.0
        pushed = contact + push_scale * np.asarray(calibration["push_delta"], dtype=np.float64)
        path = [(above, "approach_above"), (contact, "approach_contact"), (pushed, "probe_inward")]
        phase_reached: dict[str, bool] = {}
        contacted = False
        for target, phase in path:
            tolerance = 0.025 if phase == "approach_contact" else 0.012 if variant_v2 else 0.004
            gripper = 1.0 if phase == "approach_above" else 0.0 if variant_v2 else 1.0
            observation, phase_reached[phase] = feedback_move(
                env,
                observation,
                target,
                gripper=gripper,
                phase=phase,
                trace=trace,
                initial_gray_small=initial_gray_small,
                tolerance_m=tolerance,
                target_quat=initial_quat if variant_v4 else None,
                eval_target_body=task["body"],
            )
            contacted = contacted or bool(target_contact_state(env.sim, task["body"])["target_contact"])
        observation = feedback_hold(
            env,
            observation,
            pushed,
            phase="probe_hold",
            steps=8,
            trace=trace,
            initial_gray_small=initial_gray_small,
            target_quat=initial_quat if variant_v4 else None,
            eval_target_body=task["body"],
        )
        contacted = contacted or bool(target_contact_state(env.sim, task["body"])["target_contact"])
        if variant_v8:
            diagnostic = json.loads((OUTPUT_ROOT / "expert_contact_diagnostic.json").read_text(encoding="utf-8"))
            calibration_row = next(
                value
                for value in diagnostic["rows"]
                if value["slot"] == slot and int(value["demo_index"]) == 30
            )
            waypoint_reached: list[bool] = []
            for waypoint_index, waypoint in enumerate(reversed(calibration_row["cartesian_pose_waypoints"])):
                observation, reached = feedback_move(
                    env,
                    observation,
                    np.asarray(waypoint["eef_pos"], dtype=np.float64),
                    gripper=0.0 if variant_v9 and waypoint_index < 3 else 1.0,
                    phase=f"return_waypoint_{waypoint_index:02d}",
                    trace=trace,
                    initial_gray_small=initial_gray_small,
                    max_steps=70 if variant_v9 else 20,
                    tolerance_m=0.015,
                    target_quat=np.asarray(waypoint["eef_quat"], dtype=np.float64),
                    eval_target_body=task["body"],
                )
                waypoint_reached.append(bool(reached))
            observation, phase_reached["return_neutral"] = feedback_move(
                env,
                observation,
                initial_eef,
                gripper=1.0,
                phase="return_neutral",
                trace=trace,
                initial_gray_small=initial_gray_small,
                max_steps=70,
                tolerance_m=0.004,
                target_quat=initial_quat,
                eval_target_body=task["body"],
            )
            phase_reached["return_waypoints_fraction"] = float(np.mean(waypoint_reached)) if waypoint_reached else 0.0
            return_path = []
        elif variant_v3:
            low_above = contact.copy()
            low_above[2] = 0.97
            retreat_side = np.asarray([contact[0], 0.08 if slot == "front" else 0.0, 0.98])
            central = np.asarray([-0.05, 0.02, 0.98])
            prehome = np.asarray([initial_eef[0], initial_eef[1], 1.02])
            return_path = [
                (contact, "withdraw_contact"),
                (low_above, "withdraw_low_above"),
                (retreat_side, "return_retreat_side"),
                (central, "return_central"),
                (prehome, "return_prehome"),
                (initial_eef, "return_neutral"),
            ]
        else:
            return_path = [(contact, "withdraw_contact"), (above, "withdraw_above")]
        if variant_v2 and not variant_v3 and not variant_v8:
            transit = np.asarray([(above[0] + initial_eef[0]) / 2.0, (above[1] + initial_eef[1]) / 2.0, 1.02])
            prehome = np.asarray([initial_eef[0], initial_eef[1], 1.04])
            return_path.extend([(transit, "return_transit"), (prehome, "return_prehome")])
        if not variant_v3 and not variant_v8:
            return_path.append((initial_eef, "return_neutral"))
        for target, phase in return_path:
            observation, phase_reached[phase] = feedback_move(
                env,
                observation,
                target,
                gripper=1.0,
                phase=phase,
                trace=trace,
                initial_gray_small=initial_gray_small,
                tolerance_m=0.012 if variant_v2 and phase != "return_neutral" else 0.004,
                target_quat=initial_quat if variant_v4 or ((variant_v5 or variant_v7) and phase.startswith("return_")) else None,
                absolute_pose=bool(variant_v7 and phase.startswith("return_")),
                eval_target_body=task["body"],
            )
            contacted = contacted or bool(target_contact_state(env.sim, task["body"])["target_contact"])

        final_target = np.asarray(env.sim.data.get_body_xpos(task["body"]), dtype=np.float64).copy()
        final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        final_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
        contacted = bool(any(trace["target_contact_eval"]))
        all_actions = np.asarray(trace["action"], dtype=np.float32)
        phase_array = np.asarray(trace["phase"])
        if variant_v7:
            delta_mask = np.asarray([not value.startswith("return_") for value in phase_array], dtype=bool)
            absolute_mask = ~delta_mask
            delta_legal = not np.any(delta_mask) or bool(np.max(np.abs(all_actions[delta_mask])) <= 1.0 + 1e-7)
            absolute_rows = all_actions[absolute_mask]
            absolute_legal = bool(
                absolute_rows.size
                and np.isfinite(absolute_rows).all()
                and np.all(np.abs(absolute_rows[:, :2]) <= 1.0)
                and np.all((absolute_rows[:, 2] >= 0.7) & (absolute_rows[:, 2] <= 1.3))
                and np.all(np.linalg.norm(absolute_rows[:, 3:6], axis=1) <= np.pi + 1e-5)
                and np.all(np.abs(absolute_rows[:, 6]) <= 1.0)
            )
            actions_legal = bool(np.isfinite(all_actions).all() and delta_legal and absolute_legal)
        else:
            actions_legal = bool(all_actions.size and np.isfinite(all_actions).all() and np.max(np.abs(all_actions)) <= 1.0 + 1e-7)
        trace_path = OUTPUT_ROOT / attempt_id / "traces" / f"{slot}_demo{demo_index}_mass{factor:g}.npz"
        persist_trace(trace_path, trace)
        from PIL import Image

        frame_root = OUTPUT_ROOT / attempt_id / "frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        for label, frame in (("initial", initial_frame), ("final", final_frame)):
            destination = frame_root / f"{slot}_demo{demo_index}_mass{factor:g}_{label}.png"
            temporary = destination.with_suffix(".tmp.png")
            Image.fromarray(frame).save(temporary)
            temporary.replace(destination)
        row.update(
            {
                "completed": True,
                "online_inputs": ["fixed slot calibration", "agentview RGB", "end-effector proprioception", "command history", "elapsed steps"],
                "forbidden_online_inputs_used": [],
                "expert_actions_used_online": False,
                "simulator_state_used_for_actions": False,
                "target_contact_eval_only": bool(contacted),
                "phase_reached": phase_reached,
                "steps": len(all_actions),
                "finite_bounded_actions": actions_legal,
                "action_modes": ["delta OSC_POSE probe", "absolute OSC_POSE return"] if variant_v7 else ["delta OSC_POSE"],
                "initial_eef": initial_eef.tolist(),
                "initial_eef_quat": initial_quat.tolist(),
                "final_eef": final_eef.tolist(),
                "final_eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
                "initial_target_eval_only": initial_target.tolist(),
                "final_target_eval_only": final_target.tolist(),
                "final_target_displacement_m_eval_only": float(np.linalg.norm(final_target - initial_target)),
                "visual_return_estimate": template_shift(initial_frame, final_frame, slot),
                "mean_controller_error_m": float(np.mean(trace["controller_error"])),
                "p90_controller_error_m": float(np.quantile(trace["controller_error"], 0.9)),
                "max_controller_error_m": float(np.max(trace["controller_error"])),
                "trace_path": str(trace_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime development
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    return row


def controller_attempt(attempt_id: str, demo_indices: list[int], mass_factors: list[float]) -> dict[str, Any]:
    if not attempt_id.startswith("v"):
        raise ValueError("attempt id must be versioned, for example v1_proprio_feedback")
    output = OUTPUT_ROOT / attempt_id / "result.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preserved development attempt: {output}")
    env_class = load_env_class()
    variant_v2 = attempt_id.startswith("v2_") or attempt_id.startswith("v3_") or attempt_id.startswith("v4_") or attempt_id.startswith("v5_") or attempt_id.startswith("v6_") or attempt_id.startswith("v7_") or attempt_id.startswith("v8_") or attempt_id.startswith("v9_") or attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
    variant_v3 = attempt_id.startswith("v3_") or attempt_id.startswith("v4_") or attempt_id.startswith("v5_") or attempt_id.startswith("v6_") or attempt_id.startswith("v7_") or attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
    variant_v4 = attempt_id.startswith("v4_")
    variant_v5 = attempt_id.startswith("v5_") or attempt_id.startswith("v6_")
    variant_v6 = attempt_id.startswith("v6_")
    variant_v7 = attempt_id.startswith("v7_")
    variant_v8 = attempt_id.startswith("v8_") or attempt_id.startswith("v9_")
    variant_v9 = attempt_id.startswith("v9_")
    variant_v10 = attempt_id.startswith("v10_") or attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
    variant_v11 = attempt_id.startswith("v11_") or attempt_id.startswith("v12_")
    variant_v12 = attempt_id.startswith("v12_")
    result: dict[str, Any] = {
        "schema_version": "epoch9.probe_controller_development.v1",
        "timestamp": timestamp(),
        "evidence_class": "DEVELOPMENT",
        "attempt_id": attempt_id,
        "controller": (
            "v11 contact-depth controller with development-calibrated five-pixel RGB return tolerance"
            if variant_v12
            else
            "fixed-slot short delta push probe with contact-depth corrected front reach, neutral gripper, low-clearance staged proprioceptive return, and RGB residual verification"
            if variant_v11
            else
            "fixed-slot short delta push probe with reset-calibration front reach correction, neutral gripper, low-clearance staged proprioceptive return, and RGB residual verification"
            if variant_v10
            else
            "fixed-slot short delta push probe with neutral gripper, dense-tolerance feedback tracking of sparse calibrated Cartesian pose waypoints, and RGB residual verification"
            if variant_v9
            else
            "fixed-slot short delta push probe with neutral gripper, feedback tracking of sparse development-calibrated Cartesian pose waypoints, and RGB residual verification"
            if variant_v8
            else
            "fixed-slot short delta push probe with neutral gripper, staged absolute-pose OSC return, and RGB residual verification"
            if variant_v7
            else
            "fixed-slot short push probe with neutral gripper, position-only contact, official OSC orientation-error return feedback, and RGB residual verification"
            if variant_v6
            else
            "fixed-slot short push probe with neutral gripper, position-only contact, full-pose return feedback, and RGB residual verification"
            if variant_v5
            else
            "fixed-slot short push probe with neutral gripper, full pose feedback, low-clearance staged return, and RGB residual verification"
            if variant_v4
            else
            "fixed-slot short push probe with neutral gripper, low-clearance staged proprioceptive return, and RGB residual verification"
            if variant_v3
            else "fixed-slot short push probe with neutral gripper, staged proprioceptive return, and RGB residual verification"
            if variant_v2
            else "fixed-slot push probe with proprioceptive waypoint and neutral-return feedback"
        ),
        "visual_feedback_role": (
            "post-return template residual is a controller completion gate; no action-prefix inversion is used"
            if variant_v2
            else "measures post-return residual; correction is deferred in v1"
        ),
        "demo_indices": demo_indices,
        "mass_factors": mass_factors,
        "rows": [],
    }
    for slot in TASKS:
        for demo_index in demo_indices:
            for factor in mass_factors:
                row = controller_episode(
                    env_class,
                    slot=slot,
                    demo_index=demo_index,
                    factor=factor,
                    attempt_id=attempt_id,
                )
                result["rows"].append(row)
                atomic_write_json(output, result)
    rows = result["rows"]
    valid = all(row.get("completed") and row.get("exception") is None for row in rows)
    target_gate = 0.03
    eef_gate = 0.05
    visual_residuals = [
        float(np.hypot(row["visual_return_estimate"]["dx"], row["visual_return_estimate"]["dy"]))
        for row in rows
        if row.get("visual_return_estimate")
    ]
    visual_limit = 5.0 if attempt_id.startswith("v12_") else 4.0
    visual_gate = bool(
        visual_residuals
        and all(value <= visual_limit for value in visual_residuals)
        and all(row.get("visual_return_estimate", {}).get("quality", 0.0) >= 0.5 for row in rows)
    )
    result["summary"] = {
        "episodes": len(rows),
        "valid": valid,
        "contact_count": sum(bool(row.get("target_contact_eval_only")) for row in rows),
        "bounded_action_count": sum(bool(row.get("finite_bounded_actions")) for row in rows),
        "target_return_count": sum(row.get("final_target_displacement_m_eval_only", np.inf) <= target_gate for row in rows),
        "eef_return_count": sum(row.get("final_eef_displacement_m", np.inf) <= eef_gate for row in rows),
        "max_target_displacement_m": max((row.get("final_target_displacement_m_eval_only", np.inf) for row in rows), default=np.inf),
        "max_eef_displacement_m": max((row.get("final_eef_displacement_m", np.inf) for row in rows), default=np.inf),
        "max_visual_residual_pixels": max(visual_residuals, default=np.inf),
        "visual_residual_limit_pixels": visual_limit,
        "visual_return_gate": visual_gate,
        "development_milestone_pass": bool(
            valid
            and all(row.get("target_contact_eval_only") for row in rows)
            and all(row.get("finite_bounded_actions") for row in rows)
            and all(row.get("final_target_displacement_m_eval_only", np.inf) <= target_gate for row in rows)
            and all(row.get("final_eef_displacement_m", np.inf) <= eef_gate for row in rows)
            and (visual_gate if variant_v2 else True)
        ),
    }
    atomic_write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("expert-contact-diagnostic", "controller-attempt"), required=True)
    parser.add_argument("--demo-indices", default="30,31,32")
    parser.add_argument("--mass-factors", default="1,8")
    parser.add_argument("--attempt-id", default="v1_proprio_feedback")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_indices = [int(value) for value in args.demo_indices.split(",") if value.strip()]
    if any(index < 30 or index > 39 for index in demo_indices):
        raise ValueError("controller development diagnostics are restricted to fresh demo indices 30..39")
    if args.mode == "expert-contact-diagnostic":
        result = expert_contact_diagnostic(demo_indices)
        print(json.dumps(result["summary"], sort_keys=True))
        return 0 if result["summary"]["all_contact"] else 1
    mass_factors = [float(value) for value in args.mass_factors.split(",") if value.strip()]
    result = controller_attempt(args.attempt_id, demo_indices, mass_factors)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
