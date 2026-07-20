#!/usr/bin/env python3
"""Calibrate, develop, freeze, and evaluate the Epoch 9B dynamic nudge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_epoch9_probe_controller_development import TASKS, load_env_class, make_env, read_demo
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json, target_contact_state
from tca_map.epoch9b_metrics import rgb_sha256, template_shift_at_center

PROTOCOL_PATH = ROOT / "reports/epoch9b_v2_task_preservation_protocol.json"
OUTPUT_ROOT = ROOT / "reports/epoch9b_dynamic_nudge"
BASE_CALIBRATION_PATH = OUTPUT_ROOT / "controller_calibration.json"
CALIBRATION_REPAIR1_PATH = OUTPUT_ROOT / "controller_calibration_repair1.json"
CALIBRATION_PATH = CALIBRATION_REPAIR1_PATH if CALIBRATION_REPAIR1_PATH.exists() else BASE_CALIBRATION_PATH
CALIBRATION_FRAME = OUTPUT_ROOT / "controller_calibration_reference.png"
FREEZE_PATH = OUTPUT_ROOT / "controller_freeze.json"
PANEL_PATH = OUTPUT_ROOT / "feasibility_panel_result.json"
SCORE_CALIBRATION_PATH = OUTPUT_ROOT / "back_response_threshold_calibration.json"
AUDITED_CENTERS_128 = {"front": (92, 28), "back": (71, 60)}
BODY_BY_SLOT = {slot: task["body"] for slot, task in TASKS.items()}
PLATE_BODY = "plate_1_main"
FROZEN_BACK_HEAVY_THRESHOLD_M = 0.005219466062047384


@dataclass(frozen=True)
class ControllerConfig:
    approach_start_offset_x_m: float = -0.095
    approach_stop_offset_x_m: float = -0.040
    paddle_y_offset_m: float = 0.018
    front_clear_approach_y_m: float = 0.105
    front_contact_transition_offset_x_m: float = -0.062
    front_centered_contact: bool = False
    front_inward_contact: bool = False
    front_lane_mid_y_m: float = 0.145
    paddle_z_m: float = 0.930
    approach_increment_m: float = 0.003
    approach_control_steps_per_increment: int = 4
    visual_contact_threshold_pixels: float = 0.55
    visual_quality_min: float = 0.50
    contact_verify_retract_m: float = 0.008
    contact_verify_steps: int = 7
    controller_error_contact_m: float = 0.012
    controller_error_over_baseline_m: float = 0.006
    controller_error_consecutive_steps: int = 2
    impulse_action_x: float = 0.65
    impulse_steps: int = 3
    coast_steps: int = 2
    back_heavy_threshold_m: float = FROZEN_BACK_HEAVY_THRESHOLD_M
    close_steps: int = 8
    gripper_closed_command: float = 1.0
    neutral_tolerance_m: float = 0.004
    move_gain_denominator_m: float = 0.04

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def save_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary)
    temporary.replace(path)


def body_qpos_address(sim: Any, body_name: str) -> int:
    body_id = int(sim.model.body_name2id(body_name))
    joint_id = int(sim.model.body_jntadr[body_id])
    if joint_id < 0 or int(sim.model.jnt_type[joint_id]) != 0:
        raise RuntimeError(f"{body_name} is not attached by a free joint")
    return int(sim.model.jnt_qposadr[joint_id])


def body_position(sim: Any, body_name: str) -> np.ndarray:
    return np.asarray(sim.data.get_body_xpos(body_name), dtype=np.float64).copy()


def forced_observation(env: Any) -> dict[str, Any]:
    return env.env._get_observations(force_update=True)


def calibrate() -> dict[str, Any]:
    if BASE_CALIBRATION_PATH.exists() or CALIBRATION_FRAME.exists():
        raise FileExistsError("refusing to overwrite dynamic-nudge calibration")
    env_class = load_env_class()
    env = None
    translation_rows: list[dict[str, Any]] = []
    try:
        init_state, _ = read_demo(TASKS["front"], 37)
        env, observation = make_env(env_class, TASKS["front"], init_state)
        baseline_qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
        reference = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
        reference_world = {slot: body_position(env.sim, body) for slot, body in BODY_BY_SLOT.items()}
        save_rgb(CALIBRATION_FRAME, reference)
        offsets = ((0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02), (0.015, 0.015), (-0.015, 0.015))
        transforms: dict[str, Any] = {}
        for slot, body in BODY_BY_SLOT.items():
            address = body_qpos_address(env.sim, body)
            world_rows = []
            pixel_rows = []
            for dx, dy in offsets:
                env.sim.data.qpos[:] = baseline_qpos
                env.sim.data.qvel[:] = 0.0
                env.sim.data.qpos[address] = baseline_qpos[address] + dx
                env.sim.data.qpos[address + 1] = baseline_qpos[address + 1] + dy
                env.sim.forward()
                frame = np.asarray(forced_observation(env)["agentview_image"], dtype=np.uint8).copy()
                metric = template_shift_at_center(
                    reference, frame, AUDITED_CENTERS_128[slot], radius=7, search=18
                )
                world_rows.append([dx, dy])
                pixel_rows.append([metric["dx"], metric["dy"]])
                translation_rows.append(
                    {
                        "slot": slot,
                        "world_delta_xy_m": [dx, dy],
                        "pixel_delta_xy": [metric["dx"], metric["dy"]],
                        "quality": metric["quality"],
                    }
                )
            world = np.asarray(world_rows, dtype=np.float64)
            pixel = np.asarray(pixel_rows, dtype=np.float64)
            world_to_pixel_row_matrix, _, _, _ = np.linalg.lstsq(world, pixel, rcond=None)
            pixel_to_world = np.linalg.inv(world_to_pixel_row_matrix.T)
            reconstructed = (pixel_to_world @ pixel.T).T
            errors = np.linalg.norm(reconstructed - world, axis=1)
            transforms[slot] = {
                "reference_center_xy_pixels": list(AUDITED_CENTERS_128[slot]),
                "reference_world_xyz_m_eval_calibration_only": reference_world[slot].tolist(),
                "world_row_to_pixel_row_matrix": world_to_pixel_row_matrix.tolist(),
                "pixel_column_to_world_column_matrix": pixel_to_world.tolist(),
                "translation_reconstruction_error_m": {
                    "mean": float(np.mean(errors)),
                    "max": float(np.max(errors)),
                },
            }
    finally:
        if env is not None:
            env.close()

    oracle: dict[str, Any] = {}
    for slot, task in TASKS.items():
        env = None
        try:
            init_state, actions = read_demo(task, 30)
            env, observation = make_env(env_class, task, init_state)
            initial_object = body_position(env.sim, task["body"])
            initial_plate = body_position(env.sim, PLATE_BODY)
            grip = actions[:, 6]
            close_indices = [index for index in range(1, len(grip)) if grip[index - 1] < 0 and grip[index] > 0]
            release_indices = [index for index in range(1, len(grip)) if grip[index - 1] > 0 and grip[index] < 0]
            close_index = close_indices[-1]
            release_index = next(index for index in release_indices if index > close_index)
            close_pose = None
            release_pose = None
            for index, action in enumerate(actions):
                observation, _, _, _ = env.step(action)
                if index == close_index:
                    close_pose = {
                        "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
                        "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy(),
                    }
                if index == release_index:
                    release_pose = {
                        "eef_pos": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
                        "eef_quat": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy(),
                    }
                    break
            if close_pose is None or release_pose is None:
                raise RuntimeError(f"oracle pose calibration failed for {slot}")
            oracle[slot] = {
                "source_demo_index": 30,
                "close_transition_step": int(close_index),
                "release_transition_step": int(release_index),
                "grasp_eef_minus_object_xyz_m": (close_pose["eef_pos"] - initial_object).tolist(),
                "grasp_eef_quat_xyzw": close_pose["eef_quat"].tolist(),
                "release_eef_minus_plate_xyz_m": (release_pose["eef_pos"] - initial_plate).tolist(),
                "release_eef_quat_xyzw": release_pose["eef_quat"].tolist(),
            }
        finally:
            if env is not None:
                env.close()
    result = {
        "schema_version": "epoch9b.dynamic_nudge_calibration.v1",
        "frozen_at": timestamp(),
        "evidence_class": "DEVELOPMENT_CALIBRATION_NO_CONTROLLER_OUTCOME",
        "demo_indices": [30, 37],
        "sealed_identities_accessed": False,
        "reference_frame_path": str(CALIBRATION_FRAME.relative_to(ROOT)).replace("\\", "/"),
        "reference_frame_sha256": sha256(CALIBRATION_FRAME),
        "transforms": transforms,
        "translation_rows": translation_rows,
        "pose_adaptive_oracle_calibration": oracle,
        "information_boundary": "simulator pose is used only to fit the offline RGB calibration and oracle headroom offsets; live probe localization uses RGB only",
    }
    atomic_write_json(BASE_CALIBRATION_PATH, result)
    return result


def load_calibration() -> dict[str, Any]:
    value = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    frame_path = ROOT / value["reference_frame_path"]
    if sha256(frame_path) != value["reference_frame_sha256"]:
        raise RuntimeError("calibration reference hash mismatch")
    return value


def localize_candidate(
    frame: np.ndarray, slot: str, calibration: dict[str, Any]
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    reference = np.asarray(Image.open(ROOT / calibration["reference_frame_path"]).convert("RGB"), dtype=np.uint8)
    transform = calibration["transforms"][slot]
    center = tuple(int(value) for value in transform["reference_center_xy_pixels"])
    metric = template_shift_at_center(reference, frame, center, radius=7, search=18)
    pixel_delta = np.asarray([metric["subpixel_dx"], metric["subpixel_dy"]], dtype=np.float64)
    pixel_to_world = np.asarray(transform["pixel_column_to_world_column_matrix"], dtype=np.float64)
    world = np.asarray(transform["reference_world_xyz_m_eval_calibration_only"], dtype=np.float64).copy()
    world[:2] += pixel_to_world @ pixel_delta
    current_center = (
        int(round(center[0] + metric["subpixel_dx"])),
        int(round(center[1] + metric["subpixel_dy"])),
    )
    return world, current_center, metric


def set_scene_candidates(env: Any, scene: dict[str, Any]) -> None:
    for slot, body in BODY_BY_SLOT.items():
        address = body_qpos_address(env.sim, body)
        xy = scene["candidate_initial_xy_m"][slot]
        env.sim.data.qpos[address] = float(xy[0])
        env.sim.data.qpos[address + 1] = float(xy[1])
        env.sim.data.qvel[:] = 0.0
    env.sim.forward()
    for slot, body in BODY_BY_SLOT.items():
        factor = float(scene["mass_factor"][slot])
        if factor != 1.0:
            apply_intervention(
                env.sim.model,
                {"axis": "target_mass", "body_name": body, "arrays": ["body_mass", "body_inertia"], "factor": factor},
            )
    env.sim.forward()


def candidate_pair_contact_records(env: Any, first_body: str, second_body: str) -> list[dict[str, Any]]:
    model = env.sim.model
    body_ids = {int(model.body_name2id(first_body)), int(model.body_name2id(second_body))}
    records: list[dict[str, Any]] = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        pair = {body1, body2}
        if pair == body_ids:
            records.append(
                {
                    "contact_index": int(index),
                    "geom1": str(model.geom_id2name(geom1)),
                    "geom2": str(model.geom_id2name(geom2)),
                    "body1": str(model.body_id2name(body1)),
                    "body2": str(model.body_id2name(body2)),
                    "distance_m": float(contact.dist),
                }
            )
    return records


def candidate_pair_contact(env: Any, first_body: str, second_body: str) -> bool:
    return bool(candidate_pair_contact_records(env, first_body, second_body))


class EpisodeRecorder:
    def __init__(self, env: Any, initial_frame: np.ndarray, slot: str, center_xy: tuple[int, int], calibration: dict[str, Any]) -> None:
        self.env = env
        self.initial_frame = initial_frame
        self.slot = slot
        self.center_xy = center_xy
        self.calibration = calibration
        self.phase: list[str] = []
        self.action: list[np.ndarray] = []
        self.eef_pos: list[np.ndarray] = []
        self.eef_quat: list[np.ndarray] = []
        self.controller_error: list[float] = []
        self.rgb_displacement_pixels: list[float] = []
        self.rgb_delta_pixels: list[np.ndarray] = []
        self.rgb_quality: list[float] = []
        self.estimated_world_displacement_m: list[float] = []
        self.estimated_world_delta_xy_m: list[np.ndarray] = []
        self.target_contact_eval: list[bool] = []
        self.candidate_positions_eval: list[dict[str, list[float]]] = []
        self.candidate_pair_collision_eval: list[bool] = []
        self.distractor_collision_eval: list[bool] = []
        self.distractor_contact_records_eval: list[dict[str, Any]] = []

    def step(self, observation: dict[str, Any], action: np.ndarray, phase: str) -> dict[str, Any]:
        observation, _, _, _ = self.env.step(np.asarray(action, dtype=np.float32))
        controller = self.env.env.robots[0].controller
        frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
        metric = template_shift_at_center(self.initial_frame, frame, self.center_xy, radius=7, search=18)
        pixel_to_world = np.asarray(
            self.calibration["transforms"][self.slot]["pixel_column_to_world_column_matrix"], dtype=np.float64
        )
        pixel_delta = np.asarray([metric["subpixel_dx"], metric["subpixel_dy"]], dtype=np.float64)
        world_delta = pixel_to_world @ pixel_delta
        positions = {slot: body_position(self.env.sim, body).tolist() for slot, body in BODY_BY_SLOT.items()}
        self.phase.append(phase)
        self.action.append(np.asarray(action, dtype=np.float32).copy())
        self.eef_pos.append(np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy())
        self.eef_quat.append(np.asarray(observation["robot0_eef_quat"], dtype=np.float32).copy())
        self.controller_error.append(
            float(np.linalg.norm(np.asarray(controller.goal_pos) - np.asarray(controller.ee_pos)))
        )
        self.rgb_displacement_pixels.append(float(metric["subpixel_magnitude_pixels"]))
        self.rgb_delta_pixels.append(pixel_delta.copy())
        self.rgb_quality.append(float(metric["quality"]))
        self.estimated_world_displacement_m.append(float(np.linalg.norm(world_delta)))
        self.estimated_world_delta_xy_m.append(world_delta.copy())
        self.target_contact_eval.append(bool(target_contact_state(self.env.sim, BODY_BY_SLOT[self.slot])["target_contact"]))
        self.candidate_positions_eval.append(positions)
        self.candidate_pair_collision_eval.append(
            candidate_pair_contact(self.env, BODY_BY_SLOT["front"], BODY_BY_SLOT["back"])
        )
        distractor_records = []
        for candidate_slot in ("front", "back"):
            for value in candidate_pair_contact_records(
                self.env, BODY_BY_SLOT[candidate_slot], "akita_black_bowl_2_main"
            ):
                distractor_records.append(
                    {
                        **value,
                        "candidate_slot": candidate_slot,
                        "phase": phase,
                        "recorder_step": len(self.phase),
                    }
                )
        self.distractor_collision_eval.append(bool(distractor_records))
        self.distractor_contact_records_eval.extend(distractor_records)
        return observation

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "phase": np.asarray(self.phase),
            "action": np.asarray(self.action, dtype=np.float32),
            "eef_pos": np.asarray(self.eef_pos, dtype=np.float32),
            "eef_quat": np.asarray(self.eef_quat, dtype=np.float32),
            "controller_error": np.asarray(self.controller_error, dtype=np.float32),
            "rgb_displacement_pixels": np.asarray(self.rgb_displacement_pixels, dtype=np.float32),
            "rgb_delta_pixels": np.asarray(self.rgb_delta_pixels, dtype=np.float32),
            "rgb_quality": np.asarray(self.rgb_quality, dtype=np.float32),
            "estimated_world_displacement_m": np.asarray(self.estimated_world_displacement_m, dtype=np.float32),
            "estimated_world_delta_xy_m": np.asarray(self.estimated_world_delta_xy_m, dtype=np.float32),
            "target_contact_eval_only": np.asarray(self.target_contact_eval, dtype=np.bool_),
            "candidate_pair_collision_eval_only": np.asarray(
                self.candidate_pair_collision_eval, dtype=np.bool_
            ),
            "candidate_distractor_collision_eval_only": np.asarray(
                self.distractor_collision_eval, dtype=np.bool_
            ),
            "candidate_positions_eval_only": np.asarray(
                [
                    [positions["front"], positions["back"]]
                    for positions in self.candidate_positions_eval
                ],
                dtype=np.float32,
            ),
        }


def legal_action(action: np.ndarray) -> bool:
    return bool(action.shape == (7,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0 + 1e-7)


def calibrated_rank_scores(raw_response_m: dict[str, float], threshold_m: float) -> dict[str, float]:
    """Map the calibrated back-slot response to complementary mass evidence."""

    if set(raw_response_m) != {"front", "back"}:
        raise ValueError("raw response must contain exactly front and back")
    back_delta = float(raw_response_m["back"]) - float(threshold_m)
    return {"front": -back_delta, "back": back_delta}


def control_action(
    observation: dict[str, Any], target: np.ndarray, config: ControllerConfig, gripper: float, target_quat: np.ndarray | None = None
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    error = np.asarray(target, dtype=np.float64) - np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    action[:3] = np.clip(error / config.move_gain_denominator_m, -1.0, 1.0).astype(np.float32)
    if target_quat is not None:
        from robosuite.utils import transform_utils as transform
        from robosuite.utils.control_utils import orientation_error

        desired = transform.quat2mat(np.asarray(target_quat, dtype=np.float64))
        current = transform.quat2mat(np.asarray(observation["robot0_eef_quat"], dtype=np.float64))
        action[3:6] = np.clip(orientation_error(desired, current) / 0.5, -1.0, 1.0).astype(np.float32)
    action[6] = float(gripper)
    return action


def move_to(
    observation: dict[str, Any], target: np.ndarray, recorder: EpisodeRecorder, config: ControllerConfig, phase: str,
    *, gripper: float, max_steps: int = 70, tolerance: float = 0.006, target_quat: np.ndarray | None = None
) -> tuple[dict[str, Any], bool]:
    stable = 0
    for _ in range(max_steps):
        error = float(np.linalg.norm(np.asarray(target) - np.asarray(observation["robot0_eef_pos"])))
        action = control_action(observation, target, config, gripper, target_quat)
        observation = recorder.step(observation, action, phase)
        stable = stable + 1 if error <= tolerance else 0
        if stable >= 3:
            return observation, True
    return observation, False


def lane_contains(protocol: dict[str, Any], slot: str, xyz: list[float]) -> bool:
    lane = protocol["safe_center_lanes_m"][slot]
    reach = protocol["reachable_center_envelope_m"]
    return bool(
        lane["x"][0] <= xyz[0] <= lane["x"][1]
        and lane["y"][0] <= xyz[1] <= lane["y"][1]
        and reach["z"][0] <= xyz[2] <= reach["z"][1]
    )


def paddle_y_offset(slot: str, config: ControllerConfig) -> float:
    """Mirror the lateral paddle clearance in the two fixture slot frames."""

    if slot not in ("front", "back"):
        raise ValueError(f"unknown candidate slot: {slot}")
    return (-1.0 if slot == "front" else 1.0) * abs(config.paddle_y_offset_m)


def approach_y(slot: str, estimated_target_y: float, offset_x: float, config: ControllerConfig) -> float:
    """Route the front paddle past the central distractor before moving laterally."""

    if slot == "front" and offset_x < config.front_contact_transition_offset_x_m:
        return float(config.front_clear_approach_y_m)
    if slot == "front" and config.front_inward_contact:
        inward_side = 1.0 if estimated_target_y >= config.front_lane_mid_y_m else -1.0
        return float(estimated_target_y + inward_side * abs(config.paddle_y_offset_m))
    if slot == "front" and config.front_centered_contact:
        return float(estimated_target_y)
    return float(estimated_target_y + paddle_y_offset(slot, config))


def persist_trace(path: Path, recorder: EpisodeRecorder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **recorder.arrays())
    temporary.replace(path)


def probe_candidate(
    env: Any,
    observation: dict[str, Any],
    slot: str,
    scene_id: str,
    config: ControllerConfig,
    calibration: dict[str, Any],
    protocol: dict[str, Any],
    trace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
    estimated_target, center_xy, localization = localize_candidate(initial_frame, slot, calibration)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    initial_eef_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    initial_positions = {name: body_position(env.sim, body) for name, body in BODY_BY_SLOT.items()}
    recorder = EpisodeRecorder(env, initial_frame, slot, center_xy, calibration)
    phases_reached: dict[str, bool] = {}
    for _ in range(config.close_steps):
        action = np.zeros(7, dtype=np.float32)
        action[6] = config.gripper_closed_command
        observation = recorder.step(observation, action, "preclose_paddle")

    prehigh = np.asarray(
        [
            estimated_target[0] + config.approach_start_offset_x_m,
            approach_y(slot, float(estimated_target[1]), config.approach_start_offset_x_m, config),
            1.02,
        ]
    )
    precontact = prehigh.copy()
    precontact[2] = config.paddle_z_m
    observation, phases_reached["transit_high"] = move_to(
        observation, prehigh, recorder, config, "transit_high", gripper=config.gripper_closed_command
    )
    observation, phases_reached["descend_paddle"] = move_to(
        observation, precontact, recorder, config, "descend_paddle", gripper=config.gripper_closed_command
    )
    contact_detected = False
    contact_trigger: str | None = None
    contact_verification: list[dict[str, Any]] = []
    contact_target = precontact.copy()
    next_verification_offset = config.approach_start_offset_x_m
    approach_offsets = np.arange(
        config.approach_start_offset_x_m + config.approach_increment_m,
        config.approach_stop_offset_x_m + 0.5 * config.approach_increment_m,
        config.approach_increment_m,
    )
    for offset_index, offset in enumerate(approach_offsets):
        contact_target[0] = estimated_target[0] + float(offset)
        contact_target[1] = approach_y(slot, float(estimated_target[1]), float(offset), config)
        visual_event = False
        visual_at_trigger = 0.0
        quality_at_trigger = 0.0
        for _ in range(config.approach_control_steps_per_increment):
            action = control_action(observation, contact_target, config, config.gripper_closed_command)
            observation = recorder.step(observation, action, "guarded_incremental_approach")
            visual_at_trigger = recorder.rgb_displacement_pixels[-1]
            quality_at_trigger = recorder.rgb_quality[-1]
            if (
                offset >= next_verification_offset
                and visual_at_trigger >= config.visual_contact_threshold_pixels
                and quality_at_trigger >= config.visual_quality_min
            ):
                visual_event = True
                break
        final_offset = offset_index == len(approach_offsets) - 1
        if not visual_event and not final_offset:
            continue

        # A paddle entering the crop can look like target motion. Retracting
        # from the observed EEF pose removes that occluder; only motion that
        # persists after withdrawal is accepted as contact evidence.
        verification_target = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        verification_target[0] -= config.contact_verify_retract_m
        observation, verification_reached = move_to(
            observation,
            verification_target,
            recorder,
            config,
            "contact_verify_retract",
            gripper=config.gripper_closed_command,
            max_steps=24,
            tolerance=0.003,
        )
        for _ in range(config.contact_verify_steps):
            action = control_action(
                observation, verification_target, config, config.gripper_closed_command
            )
            observation = recorder.step(observation, action, "contact_verify_observe")
        verify_count = min(3, config.contact_verify_steps)
        persistent_visual = float(np.median(recorder.rgb_displacement_pixels[-verify_count:]))
        persistent_quality = float(np.median(recorder.rgb_quality[-verify_count:]))
        verified = bool(
            verification_reached
            and persistent_visual >= config.visual_contact_threshold_pixels
            and persistent_quality >= config.visual_quality_min
        )
        contact_verification.append(
            {
                "approach_offset_x_m": float(offset),
                "trigger": "visual_event" if visual_event else "final_offset_fallback",
                "trigger_subpixel_motion_pixels": float(visual_at_trigger),
                "trigger_tracker_quality": float(quality_at_trigger),
                "retract_reached": bool(verification_reached),
                "persistent_subpixel_motion_pixels": persistent_visual,
                "tracker_quality": persistent_quality,
                "verified": verified,
            }
        )
        if verified:
            contact_detected = True
            contact_trigger = "persistent_rgb_after_retract"
            break
        observation, _ = move_to(
            observation,
            contact_target,
            recorder,
            config,
            "resume_guarded_approach",
            gripper=config.gripper_closed_command,
            max_steps=24,
            tolerance=0.003,
        )
        next_verification_offset = float(offset) + config.approach_increment_m - 1e-9

    # The fixed impulse starts from the verified 8 mm clearance. A bounded
    # ballistic re-tap avoids a position-controlled preload that would carry
    # the object a prescribed distance almost independently of mass.
    if contact_detected:
        phases_reached["preimpulse_clearance"] = True

    response_baseline_world = (
        np.median(np.asarray(recorder.estimated_world_delta_xy_m[-3:]), axis=0)
        if recorder.estimated_world_delta_xy_m
        else np.zeros(2, dtype=np.float64)
    )
    response_baseline_eval = {
        name: body_position(env.sim, body) for name, body in BODY_BY_SLOT.items()
    }
    response_start = len(recorder.action)
    if contact_detected:
        for _ in range(config.impulse_steps):
            action = np.zeros(7, dtype=np.float32)
            action[0] = config.impulse_action_x
            action[6] = config.gripper_closed_command
            observation = recorder.step(observation, action, "fixed_micro_impulse")
        for _ in range(config.coast_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = config.gripper_closed_command
            observation = recorder.step(observation, action, "post_impulse_response")
    response_stop = len(recorder.action)

    current = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    retreat = current.copy()
    retreat[0] -= 0.055
    observation, phases_reached["retreat_side"] = move_to(
        observation, retreat, recorder, config, "retreat_side", gripper=config.gripper_closed_command
    )
    retreat_high = retreat.copy()
    retreat_high[2] = 1.04
    observation, phases_reached["retreat_high"] = move_to(
        observation, retreat_high, recorder, config, "retreat_high", gripper=config.gripper_closed_command
    )
    observation, phases_reached["return_neutral"] = move_to(
        observation,
        initial_eef,
        recorder,
        config,
        "return_neutral",
        gripper=config.gripper_closed_command,
        tolerance=config.neutral_tolerance_m,
        target_quat=initial_eef_quat,
    )
    final_positions = {name: body_position(env.sim, body) for name, body in BODY_BY_SLOT.items()}
    final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    actions = np.asarray(recorder.action, dtype=np.float32)
    response_slice = slice(response_start, response_stop)
    response_world = np.asarray(recorder.estimated_world_delta_xy_m[response_slice], dtype=np.float64)
    response_increment_world = response_world - response_baseline_world[None, :] if response_world.size else np.zeros((0, 2))
    response_expected_axis = response_increment_world[:, 0] if response_increment_world.size else np.zeros(0)
    response_estimated = np.linalg.norm(response_increment_world, axis=1) if response_increment_world.size else np.zeros(0)
    response_pixels = recorder.rgb_displacement_pixels[response_slice]
    all_positions = recorder.candidate_positions_eval
    lane_all = all(
        lane_contains(protocol, candidate_slot, positions[candidate_slot])
        for positions in all_positions
        for candidate_slot in ("front", "back")
    )
    minimum_separation = min(
        np.linalg.norm(np.asarray(value["front"])[:2] - np.asarray(value["back"])[:2])
        for value in all_positions
    )
    displacement = {
        name: float(np.linalg.norm(final_positions[name] - initial_positions[name])) for name in ("front", "back")
    }
    trace_path = trace_root / f"{scene_id}_{slot}.npz"
    persist_trace(trace_path, recorder)
    incremental_expected_axis_peak = float(max(np.max(response_expected_axis), 0.0)) if response_expected_axis.size else 0.0
    incremental_expected_axis_settled = (
        float(np.mean(response_expected_axis[-min(2, response_expected_axis.size) :]))
        if response_expected_axis.size
        else 0.0
    )
    response_eval_positions = recorder.candidate_positions_eval[response_slice]
    response_eval_expected_axis = [
        float(np.asarray(value[slot])[0] - response_baseline_eval[slot][0])
        for value in response_eval_positions
    ]
    target_trajectory = [np.asarray(value[slot], dtype=np.float64) for value in all_positions]
    target_excitation_peak = float(
        max((np.linalg.norm(value - initial_positions[slot]) for value in target_trajectory), default=0.0)
    )
    sampled_target_contact = bool(any(recorder.target_contact_eval))
    target_excited = bool(target_excitation_peak >= 0.001)
    result = {
        "slot": slot,
        "online_localization": localization,
        "estimated_initial_target_xyz_m": estimated_target.tolist(),
        "contact_detected_online": contact_detected,
        "contact_trigger": contact_trigger,
        "contact_verification": contact_verification,
        "intended_target_contact_eval_only": sampled_target_contact,
        "intended_target_excited_eval_only": target_excited,
        "intended_target_contact_or_excitation_eval_only": bool(sampled_target_contact or target_excited),
        "intended_target_excitation_peak_m_eval_only": target_excitation_peak,
        "finite_bounded_actions": bool(actions.size and all(legal_action(value) for value in actions)),
        "steps": int(len(actions)),
        "phases_reached": phases_reached,
        "response_estimated_displacement_m": incremental_expected_axis_peak,
        "response_incremental_expected_axis_peak_m": incremental_expected_axis_peak,
        "response_incremental_expected_axis_settled_m": incremental_expected_axis_settled,
        "response_peak_estimated_displacement_m": float(max(response_estimated, default=0.0)),
        "response_visual_displacement_pixels": float(max(response_pixels, default=0.0)),
        "response_tracker_quality_min": float(min(recorder.rgb_quality[response_slice], default=0.0)),
        "response_eval_only_expected_axis_peak_m": float(
            max(max(response_eval_expected_axis, default=0.0), 0.0)
        ),
        "candidate_final_displacement_m_eval_only": displacement,
        "intended_candidate_displacement_m_eval_only": displacement[slot],
        "final_eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
        "final_eef_z_m": float(final_eef[2]),
        "lane_and_reachability_continuous_pass": lane_all,
        "minimum_candidate_center_separation_m_eval_only": float(minimum_separation),
        "candidate_pair_collision_eval_only": bool(any(recorder.candidate_pair_collision_eval)),
        "candidate_distractor_collision_eval_only": bool(any(recorder.distractor_collision_eval)),
        "candidate_distractor_contact_records_eval_only": recorder.distractor_contact_records_eval,
        "maximum_intended_displacement_limit_pass": displacement[slot]
        <= float(protocol["v2_absolute_displacement_rule"]["limit_m"]),
        "trace_path": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
        "forbidden_online_inputs_used": [],
        "simulator_state_used_for_actions": False,
        "mass_or_property_used_for_actions": False,
    }
    return observation, result


def oracle_completion(
    env: Any,
    observation: dict[str, Any],
    target_slot: str,
    calibration: dict[str, Any],
    config: ControllerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # A separate recorder is used only to reuse bounded Cartesian control. Its
    # target slot is an explicit oracle privilege and is never passed to probe_candidate.
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
    _, center, _ = localize_candidate(frame, target_slot, calibration)
    recorder = EpisodeRecorder(env, frame, target_slot, center, calibration)
    oracle = calibration["pose_adaptive_oracle_calibration"][target_slot]
    object_xyz = body_position(env.sim, BODY_BY_SLOT[target_slot])
    plate_xyz = body_position(env.sim, PLATE_BODY)
    grasp = object_xyz + np.asarray(oracle["grasp_eef_minus_object_xyz_m"], dtype=np.float64)
    grasp_quat = np.asarray(oracle["grasp_eef_quat_xyzw"], dtype=np.float64)
    release = plate_xyz + np.asarray(oracle["release_eef_minus_plate_xyz_m"], dtype=np.float64)
    release_quat = np.asarray(oracle["release_eef_quat_xyzw"], dtype=np.float64)
    stages: dict[str, bool] = {}
    for _ in range(10):
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0
        observation = recorder.step(observation, action, "oracle_open")
    above = grasp.copy()
    above[2] += 0.12
    observation, stages["above_grasp"] = move_to(
        observation, above, recorder, config, "oracle_above_grasp", gripper=-1.0, target_quat=grasp_quat
    )
    observation, stages["at_grasp"] = move_to(
        observation, grasp, recorder, config, "oracle_at_grasp", gripper=-1.0, tolerance=0.004, target_quat=grasp_quat
    )
    object_initial = object_xyz.copy()
    for _ in range(18):
        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0
        observation = recorder.step(observation, action, "oracle_close")
    object_after_close = body_position(env.sim, BODY_BY_SLOT[target_slot])
    lift = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    lift[2] += 0.16
    observation, stages["lift"] = move_to(
        observation, lift, recorder, config, "oracle_lift", gripper=1.0, target_quat=grasp_quat
    )
    object_after_lift = body_position(env.sim, BODY_BY_SLOT[target_slot])
    above_plate = release.copy()
    above_plate[2] += 0.14
    observation, stages["above_plate"] = move_to(
        observation, above_plate, recorder, config, "oracle_above_plate", gripper=1.0, target_quat=release_quat
    )
    observation, stages["release_pose"] = move_to(
        observation, release, recorder, config, "oracle_release_pose", gripper=1.0, tolerance=0.005, target_quat=release_quat
    )
    for _ in range(20):
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0
        observation = recorder.step(observation, action, "oracle_release")
    object_after_release = body_position(env.sim, BODY_BY_SLOT[target_slot])
    success = bool(env.check_success())
    return observation, {
        "target_slot_eval_only": target_slot,
        "stages_reached": stages,
        "official_task_success": success,
        "finite_bounded_actions": all(legal_action(value) for value in recorder.action),
        "steps": len(recorder.action),
        "oracle_privileged_pose_used": True,
        "target_object_positions_eval_only": {
            "initial": object_initial.tolist(),
            "after_close": object_after_close.tolist(),
            "after_lift": object_after_lift.tolist(),
            "after_release": object_after_release.tolist(),
            "plate": plate_xyz.tolist(),
        },
    }


def development_manifest(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        heavy = "front" if index % 2 == 0 else "back"
        light = "back" if heavy == "front" else "front"
        rows.append(
            {
                "scene_id": f"epoch9b_controller_dev_{index:03d}",
                "partition": "CONTROLLER_DEVELOPMENT",
                "source_state_demo_index": 31,
                "candidate_initial_xy_m": {
                    "front": [0.080 + 0.009 * ((3 * index + 1) % 5), 0.130 + 0.009 * ((2 * index) % 5)],
                    "back": [-0.170 + 0.009 * ((2 * index + 1) % 5), 0.030 + 0.009 * ((3 * index) % 5)],
                },
                "mass_factor": {heavy: 8.0, light: 1.0},
                "heavy_slot": heavy,
                "probe_order": ["front", "back"] if (index // 2) % 2 == 0 else ["back", "front"],
                "instruction_property": "heaviest",
                "completion_target_slot_eval_only": heavy,
            }
        )
    return rows


def edge_stress_manifest(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        heavy = "front" if index % 2 == 0 else "back"
        light = "back" if heavy == "front" else "front"
        rows.append(
            {
                "scene_id": f"epoch9b_edge_stress_dev_{index:03d}",
                "partition": "CONTROLLER_REPAIR1_EDGE_STRESS_DEVELOPMENT",
                "source_state_demo_index": 32,
                "candidate_initial_xy_m": {
                    "front": [0.084 + 0.008 * (index % 4), 0.174 - 0.001 * (index % 4)],
                    "back": [-0.166 + 0.008 * ((3 * index + 1) % 4), 0.034 + 0.010 * (index % 4)],
                },
                "mass_factor": {heavy: 8.0, light: 1.0},
                "heavy_slot": heavy,
                "probe_order": ["front", "back"] if (index // 2) % 2 == 0 else ["back", "front"],
                "instruction_property": "heaviest",
                "completion_target_slot_eval_only": heavy,
            }
        )
    return rows


def run_scene(
    scene: dict[str, Any], config: ControllerConfig, calibration: dict[str, Any], protocol: dict[str, Any], trace_root: Path
) -> dict[str, Any]:
    target_slot = str(scene["completion_target_slot_eval_only"])
    env_class = load_env_class()
    env = None
    started = time.monotonic()
    row: dict[str, Any] = {"scene_id": scene["scene_id"], "scene": scene, "exception": None}
    try:
        task = TASKS[target_slot]
        init_state, _ = read_demo(task, int(scene["source_state_demo_index"]))
        env, observation = make_env(env_class, task, init_state)
        set_scene_candidates(env, scene)
        observation = forced_observation(env)
        for _ in range(10):
            observation, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
        initial_positions = {slot: body_position(env.sim, body).tolist() for slot, body in BODY_BY_SLOT.items()}
        probes = []
        for slot in scene["probe_order"]:
            observation, probe = probe_candidate(
                env, observation, slot, scene["scene_id"], config, calibration, protocol, trace_root
            )
            probes.append(probe)
        raw_scores = {probe["slot"]: float(probe["response_estimated_displacement_m"]) for probe in probes}
        scores = calibrated_rank_scores(raw_scores, config.back_heavy_threshold_m)
        predicted_heavy = min(scores, key=scores.get)
        pre_oracle_state = env.sim.get_state().flatten().tolist()
        observation, oracle = oracle_completion(env, observation, target_slot, calibration, config)
        row.update(
            {
                "completed": True,
                "initial_candidate_positions_eval_only": initial_positions,
                "probes": probes,
                "response_score_rule": "complementary mass evidence from frozen back-slot response threshold; smaller predicts heavier",
                "response_raw_scores_m": raw_scores,
                "response_scores_m": scores,
                "predicted_heavy_slot": predicted_heavy,
                "heavy_rank_correct": predicted_heavy == scene["heavy_slot"],
                "oracle_completion": oracle,
                "canonical_vla_endpoint": {
                    "status": "NOT_RUN_DURING_CONTROLLER_DEVELOPMENT",
                    "pre_oracle_state_persisted_in_memory_only": True,
                    "note": "same-state canonical endpoint is deferred until controller feasibility GO to avoid resident-VLA cost during tuning",
                },
                "post_probe_state_length": len(pre_oracle_state),
            }
        )
    except Exception as exc:
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = float(time.monotonic() - started)
    return row


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("completed")]
    probes = [probe for row in valid_rows for probe in row["probes"]]
    intended_contacts = sum(
        bool(value["intended_target_contact_or_excitation_eval_only"]) for value in probes
    )
    sampled_contacts = sum(bool(value["intended_target_contact_eval_only"]) for value in probes)
    both_contacted = sum(
        all(value["intended_target_contact_or_excitation_eval_only"] for value in row["probes"])
        for row in valid_rows
    )
    heavy_correct = sum(bool(row["heavy_rank_correct"]) for row in valid_rows)
    oracle_success = sum(bool(row["oracle_completion"]["official_task_success"]) for row in valid_rows)
    by_heavy = {}
    for slot in ("front", "back"):
        subset = [row for row in valid_rows if row["scene"]["heavy_slot"] == slot]
        by_heavy[slot] = {"scenes": len(subset), "rank_correct": sum(row["heavy_rank_correct"] for row in subset)}
    panel = len(rows) == 24
    gate = bool(
        panel
        and len(valid_rows) == 24
        and len(probes) == 48
        and all(value["finite_bounded_actions"] for value in probes)
        and intended_contacts >= 46
        and both_contacted >= 22
        and all(value["lane_and_reachability_continuous_pass"] for value in probes)
        and all(value["maximum_intended_displacement_limit_pass"] for value in probes)
        and all(value["response_tracker_quality_min"] >= 0.50 for value in probes)
        and all(value["final_eef_displacement_m"] <= 0.05 and value["final_eef_z_m"] >= 1.10 for value in probes)
        and all(not value["candidate_pair_collision_eval_only"] for value in probes)
        and all(not value["candidate_distractor_collision_eval_only"] for value in probes)
        and heavy_correct >= 20
        and by_heavy["front"]["rank_correct"] >= 10
        and by_heavy["back"]["rank_correct"] >= 10
        and oracle_success >= 20
    )
    return {
        "scenes_recorded": len(rows),
        "scenes_completed": len(valid_rows),
        "probe_count": len(probes),
        "finite_bounded_action_probes": sum(value["finite_bounded_actions"] for value in probes),
        "intended_contact_probes": intended_contacts,
        "intended_contact_or_excitation_probes": intended_contacts,
        "sampled_instantaneous_contact_probes": sampled_contacts,
        "both_candidates_contacted_scenes": both_contacted,
        "lane_reachability_pass_probes": sum(value["lane_and_reachability_continuous_pass"] for value in probes),
        "distance_limit_pass_probes": sum(value["maximum_intended_displacement_limit_pass"] for value in probes),
        "candidate_pair_collision_probes": sum(value["candidate_pair_collision_eval_only"] for value in probes),
        "candidate_distractor_collision_probes": sum(
            value["candidate_distractor_collision_eval_only"] for value in probes
        ),
        "tracker_quality_pass_probes": sum(value["response_tracker_quality_min"] >= 0.50 for value in probes),
        "neutral_return_pass_probes": sum(
            value["final_eef_displacement_m"] <= 0.05 and value["final_eef_z_m"] >= 1.10
            for value in probes
        ),
        "heavy_rank_correct_scenes": heavy_correct,
        "heavy_rank_by_position": by_heavy,
        "oracle_completion_success_scenes": oracle_success,
        "canonical_vla_endpoint_completed_scenes": 0,
        "minimal_feasibility_panel_go": gate,
        "legacy_3cm_reference_exceedance_count": sum(
            value["intended_candidate_displacement_m_eval_only"] > 0.03 for value in probes
        ),
        "v2_limit_m": protocol["v2_absolute_displacement_rule"]["limit_m"],
    }


def run_collection(
    mode: str,
    attempt_id: str,
    config: ControllerConfig,
    scene_count: int,
    resume: bool,
    development_manifest_kind: str = "balanced",
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    calibration = load_calibration()
    if mode == "development":
        output = OUTPUT_ROOT / "development" / attempt_id / "result.json"
        trace_root = OUTPUT_ROOT / "development" / attempt_id / "traces"
        manifest = (
            edge_stress_manifest(scene_count)
            if development_manifest_kind == "edge-stress"
            else development_manifest(scene_count)
        )
    else:
        output = PANEL_PATH
        trace_root = OUTPUT_ROOT / "feasibility_panel_traces"
        manifest = protocol["feasibility_manifest"]
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["controller_config"] != config.as_dict():
            raise RuntimeError("resume controller config mismatch")
    else:
        result = {
            "schema_version": f"epoch9b.dynamic_nudge.{mode}.v2",
            "started_at": timestamp(),
            "evidence_class": "DEVELOPMENT" if mode == "development" else "FROZEN_DEVELOPMENT_FEASIBILITY",
            "attempt_id": attempt_id,
            "controller_config": config.as_dict(),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "calibration_path": str(CALIBRATION_PATH.relative_to(ROOT)).replace("\\", "/"),
            "calibration_sha256": sha256(CALIBRATION_PATH),
            "manifest": manifest,
            "rows": [],
            "validation_accessed": False,
            "confirmation_accessed": False,
        }
    completed = {row["scene_id"] for row in result["rows"]}
    for scene in manifest:
        if scene["scene_id"] in completed:
            continue
        result["rows"].append(run_scene(scene, config, calibration, protocol, trace_root))
        result["summary"] = summarize(result["rows"], protocol)
        atomic_write_json(output, result)
    result["completed_at"] = timestamp()
    result["summary"] = summarize(result["rows"], protocol)
    atomic_write_json(output, result)
    return result


def freeze_controller(attempt_id: str) -> dict[str, Any]:
    if FREEZE_PATH.exists():
        raise FileExistsError("refusing to overwrite controller freeze")
    result_path = OUTPUT_ROOT / "development" / attempt_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [row for row in result["rows"] if row.get("completed")]
    probes = [probe for row in rows for probe in row["probes"]]
    if len(rows) < 4 or len(probes) < 8:
        raise RuntimeError("at least four complete development scenes are required to freeze")
    if not all(value["finite_bounded_actions"] for value in probes):
        raise RuntimeError("cannot freeze controller with illegal actions")
    required_probe_checks = [
        all(value["intended_target_contact_or_excitation_eval_only"] for value in probes),
        all(value["lane_and_reachability_continuous_pass"] for value in probes),
        all(value["maximum_intended_displacement_limit_pass"] for value in probes),
        all(value["response_tracker_quality_min"] >= 0.50 for value in probes),
        all(value["final_eef_displacement_m"] <= 0.05 and value["final_eef_z_m"] >= 1.10 for value in probes),
        all(not value["candidate_pair_collision_eval_only"] for value in probes),
        all(not value["candidate_distractor_collision_eval_only"] for value in probes),
    ]
    if not all(required_probe_checks):
        raise RuntimeError("cannot freeze controller without complete v2-valid development probes")
    if sum(row["heavy_rank_correct"] for row in rows) < len(rows) - 1:
        raise RuntimeError("cannot freeze controller without repeatable development ranking")
    if sum(row["oracle_completion"]["official_task_success"] for row in rows) < len(rows) - 2:
        raise RuntimeError("cannot freeze controller without post-probe oracle headroom")
    score_calibration = json.loads(SCORE_CALIBRATION_PATH.read_text(encoding="utf-8"))
    if abs(
        float(score_calibration["back_heavy_threshold_m"])
        - float(result["controller_config"]["back_heavy_threshold_m"])
    ) > 1e-12:
        raise RuntimeError("controller threshold does not match frozen score calibration")
    record = {
        "schema_version": "epoch9b.dynamic_nudge.controller_freeze.v1",
        "frozen_at": timestamp(),
        "status": "FROZEN_BEFORE_24_SCENE_FEASIBILITY_PANEL",
        "selected_development_attempt": attempt_id,
        "selected_result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "selected_result_sha256": sha256(result_path),
        "controller_config": result["controller_config"],
        "development_summary": result["summary"],
        "score_calibration_path": str(SCORE_CALIBRATION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "score_calibration_sha256": sha256(SCORE_CALIBRATION_PATH),
        "calibration_path": str(CALIBRATION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "calibration_sha256": sha256(CALIBRATION_PATH),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(FREEZE_PATH, record)
    return record


def config_from_args(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        impulse_action_x=float(args.impulse_action_x),
        impulse_steps=int(args.impulse_steps),
        visual_contact_threshold_pixels=float(args.visual_contact_threshold_pixels),
        controller_error_contact_m=float(args.controller_error_contact_m),
        front_centered_contact=bool(args.front_centered_contact),
        front_inward_contact=bool(args.front_inward_contact),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "development", "freeze", "panel"), required=True)
    parser.add_argument("--attempt-id", default="d1_guarded_lateral_nudge")
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--impulse-action-x", type=float, default=0.65)
    parser.add_argument("--impulse-steps", type=int, default=3)
    parser.add_argument("--visual-contact-threshold-pixels", type=float, default=0.55)
    parser.add_argument("--controller-error-contact-m", type=float, default=0.012)
    parser.add_argument("--front-centered-contact", action="store_true")
    parser.add_argument("--front-inward-contact", action="store_true")
    parser.add_argument("--development-manifest", choices=("balanced", "edge-stress"), default="balanced")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "calibrate":
        result = calibrate()
        summary = {"status": result["evidence_class"], "max_calibration_error_m": max(
            value["translation_reconstruction_error_m"]["max"] for value in result["transforms"].values()
        )}
    elif args.mode == "freeze":
        result = freeze_controller(args.attempt_id)
        summary = {"status": result["status"]}
    elif args.mode == "panel":
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        config = ControllerConfig(**freeze["controller_config"])
        result = run_collection("panel", freeze["selected_development_attempt"], config, 24, args.resume)
        summary = result["summary"]
    else:
        result = run_collection(
            "development",
            args.attempt_id,
            config_from_args(args),
            args.scene_count,
            args.resume,
            args.development_manifest,
        )
        summary = result["summary"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
