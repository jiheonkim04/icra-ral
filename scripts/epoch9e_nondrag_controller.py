#!/usr/bin/env python3
"""Epoch 9E one-shot non-drag disengagement controller primitive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts import run_epoch9b_dynamic_nudge as campaign


def inward_approach_y(
    slot: str,
    estimated_target_y: float,
    offset_x: float,
    config: campaign.ControllerConfig,
    protocol: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Return the frozen RGB/geometry-only inward-facing paddle orientation."""

    if slot == "front" and offset_x < config.front_contact_transition_offset_x_m:
        value = float(config.front_clear_approach_y_m)
        return value, {"mode": "front_high_clear_transit", "approach_y_m": value}
    lane = protocol["safe_center_lanes_m"][slot]["y"]
    lower_margin = float(estimated_target_y - lane[0])
    upper_margin = float(lane[1] - estimated_target_y)
    if lower_margin < upper_margin:
        boundary = "lower"
    elif upper_margin < lower_margin:
        boundary = "upper"
    else:
        boundary = "lower" if slot == "front" else "upper"
    paddle_offset = abs(float(config.paddle_y_offset_m))
    value = (
        float(estimated_target_y - paddle_offset)
        if boundary == "lower"
        else float(estimated_target_y + paddle_offset)
    )
    return value, {
        "mode": "inward_facing_contact",
        "slot": slot,
        "estimated_target_y_m": float(estimated_target_y),
        "lower_lane_margin_m": lower_margin,
        "upper_lane_margin_m": upper_margin,
        "nearest_boundary": boundary,
        "approach_y_m": value,
        "mass_or_property_input": False,
    }


def vertical_liftoff_action(config: campaign.ControllerConfig, lift_action_z: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[2] = float(lift_action_z)
    action[6] = float(config.gripper_closed_command)
    return action


def run_nondrag_probe(
    env: Any,
    observation: dict[str, Any],
    slot: str,
    scene_id: str,
    config: campaign.ControllerConfig,
    calibration: dict[str, Any],
    original_protocol: dict[str, Any],
    controller_contract: dict[str, Any],
    trace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the frozen probe with only contact disengagement replaced."""

    original_approach_y = campaign.approach_y
    original_move_to = campaign.move_to
    orientation_events: list[dict[str, Any]] = []
    disengagement_attempts: list[dict[str, Any]] = []
    skip_resume = False
    lift = controller_contract["nondrag_lift"]
    separation = controller_contract["separation_trackability_verification"]
    reentry = controller_contract["response_reentry"]

    def approach_y(
        candidate_slot: str,
        estimated_target_y: float,
        offset_x: float,
        controller_config: campaign.ControllerConfig,
    ) -> float:
        value, event = inward_approach_y(
            candidate_slot,
            estimated_target_y,
            offset_x,
            controller_config,
            original_protocol,
        )
        if event["mode"] == "inward_facing_contact":
            event = {**event, "approach_offset_x_m": float(offset_x)}
            if not orientation_events or orientation_events[-1] != event:
                orientation_events.append(event)
        return value

    def move_to(
        current_observation: dict[str, Any],
        target: np.ndarray,
        recorder: campaign.EpisodeRecorder,
        controller_config: campaign.ControllerConfig,
        phase: str,
        *,
        gripper: float,
        max_steps: int = 70,
        tolerance: float = 0.006,
        target_quat: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], bool]:
        nonlocal skip_resume
        if phase == "resume_guarded_approach" and skip_resume:
            skip_resume = False
            return current_observation, True
        if phase != "contact_verify_retract":
            return original_move_to(
                current_observation,
                target,
                recorder,
                controller_config,
                phase,
                gripper=gripper,
                max_steps=max_steps,
                tolerance=tolerance,
                target_quat=target_quat,
            )

        start_eef = np.asarray(current_observation["robot0_eef_pos"], dtype=np.float64).copy()
        lift_start_action = len(recorder.action)
        lift_reached = False
        for _ in range(int(lift["maximum_steps"])):
            action = vertical_liftoff_action(controller_config, float(lift["raw_action_xyz"][2]))
            current_observation = recorder.step(
                current_observation, action, "nondrag_vertical_liftoff"
            )
            lifted = float(
                np.asarray(current_observation["robot0_eef_pos"], dtype=np.float64)[2]
                - start_eef[2]
            )
            if lifted >= float(lift["target_proprioceptive_lift_m"]):
                lift_reached = True
                break
        lift_stop_action = len(recorder.action)
        for _ in range(int(separation["hold_steps"])):
            action = np.zeros(7, dtype=np.float32)
            action[6] = float(controller_config.gripper_closed_command)
            current_observation = recorder.step(
                current_observation, action, "nondrag_separation_observe"
            )
        actual_lift = float(
            np.asarray(current_observation["robot0_eef_pos"], dtype=np.float64)[2] - start_eef[2]
        )
        quality_count = min(int(separation["hold_steps"]), len(recorder.rgb_quality))
        quality = float(np.median(recorder.rgb_quality[-quality_count:])) if quality_count else 0.0
        separation_verified = bool(
            lift_reached
            and actual_lift >= float(separation["eef_lift_m_at_least"])
            and quality >= float(separation["rgb_tracker_quality_at_least"])
        )
        high_reached = False
        descend_reached = False
        if separation_verified:
            high_target = np.asarray(target, dtype=np.float64).copy()
            high_target[2] = float(
                np.asarray(current_observation["robot0_eef_pos"], dtype=np.float64)[2]
            )
            current_observation, high_reached = original_move_to(
                current_observation,
                high_target,
                recorder,
                controller_config,
                "nondrag_high_clearance_reposition",
                gripper=gripper,
                max_steps=32,
                tolerance=0.004,
                target_quat=target_quat,
            )
            current_observation, descend_reached = original_move_to(
                current_observation,
                np.asarray(target, dtype=np.float64),
                recorder,
                controller_config,
                "nondrag_response_clearance_descend",
                gripper=gripper,
                max_steps=40,
                tolerance=0.004,
                target_quat=target_quat,
            )
        lift_actions = np.asarray(recorder.action[lift_start_action:lift_stop_action], dtype=np.float32)
        planar_max = (
            float(np.max(np.abs(lift_actions[:, :2]))) if lift_actions.size else float("inf")
        )
        disengagement_attempts.append(
            {
                "start_eef_xyz_m": start_eef.tolist(),
                "commanded_direction_world_xyz": lift["direction_world_xyz"],
                "lift_action_count": int(len(lift_actions)),
                "maximum_absolute_commanded_planar_component": planar_max,
                "actual_proprioceptive_lift_m": actual_lift,
                "tracker_quality_after_lift": quality,
                "lift_target_reached": lift_reached,
                "separation_verified_ordinary_observations": separation_verified,
                "high_clearance_reposition_reached": bool(high_reached),
                "response_clearance_descend_reached": bool(descend_reached),
                "simulator_contact_or_force_used_for_control": False,
                "mass_property_score_reward_success_or_oracle_used_for_control": False,
            }
        )
        skip_resume = True
        return current_observation, bool(separation_verified and high_reached and descend_reached)

    campaign.approach_y = approach_y
    campaign.move_to = move_to
    try:
        observation, probe = campaign.probe_candidate(
            env,
            observation,
            slot,
            scene_id,
            config,
            calibration,
            original_protocol,
            trace_root,
        )
    finally:
        campaign.approach_y = original_approach_y
        campaign.move_to = original_move_to
    probe["epoch9e_nondrag_disengagement"] = {
        "controller_name": controller_contract["name"],
        "orientation_events": orientation_events,
        "attempts": disengagement_attempts,
        "all_liftoff_planar_commands_exact_zero": bool(
            disengagement_attempts
            and all(row["maximum_absolute_commanded_planar_component"] == 0.0 for row in disengagement_attempts)
        ),
        "all_actions_label_blind": True,
        "forbidden_inputs_used": [],
    }
    return observation, probe
