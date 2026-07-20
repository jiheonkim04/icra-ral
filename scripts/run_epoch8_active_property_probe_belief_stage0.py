#!/usr/bin/env python3
"""Run the frozen legal-response mass-belief Stage-0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json, target_contact_state

PROTOCOL_PATH = REPO_ROOT / "reports/epoch8_active_property_probe_belief_protocol.json"
OUTPUT_PATH = REPO_ROOT / "reports/epoch8_active_property_probe_belief_stage0.json"
EXPECTED_PROTOCOL_SHA256 = "729D7632AEB155C08B126CE47544D7A4993C00B1BD4AA3FB268EAA464E9C1A21"
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
BDDL_ROOT = LIBERO_ROOT / "libero/libero/bddl_files/libero_90"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_protocol() -> dict[str, Any]:
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol not hash-frozen: {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_PROBE_RESPONSE_MASS_OUTCOMES":
        raise RuntimeError("protocol is not outcome-blind frozen")
    for task in protocol["data"]["tasks"]:
        path = Path(protocol["data"]["root"]) / task["hdf5"]
        if sha256_file(path) != task["hdf5_sha256"]:
            raise RuntimeError(f"HDF5 hash mismatch: {task['target']}")
    return protocol


def read_demo(path: Path, demo_name: str) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"invalid action array: {path}::{demo_name}")
    return init_state, actions


def temporal_mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def replay_probe(
    *,
    env_class: Any,
    bddl: Path,
    target_body: str,
    init_state: np.ndarray,
    actions: np.ndarray,
    factor: float,
    camera_resolution: int,
    post_contact_steps: int,
    fixed_action_count: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    row: dict[str, Any] = {
        "factor": factor,
        "completed": False,
        "target_contact": False,
        "first_contact_step": None,
        "executed_steps": 0,
        "post_contact_frames": 0,
        "exception": None,
    }
    try:
        env = env_class(
            bddl_file_name=str(bddl),
            camera_heights=int(camera_resolution),
            camera_widths=int(camera_resolution),
        )
        env.seed(0)
        env.reset()
        observation = env.set_init_state(init_state)
        if factor != 1.0:
            row["mutation"] = apply_intervention(
                env.sim.model,
                {
                    "axis": "target_mass",
                    "body_name": target_body,
                    "arrays": ["body_mass", "body_inertia"],
                    "factor": factor,
                },
            )
            env.sim.forward()
        errors: list[float] = []
        agent_mad: list[float] = []
        wrist_mad: list[float] = []
        executed: list[np.ndarray] = []
        previous_agent = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
        previous_wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
        remaining_after_contact = None
        for step, action in enumerate(actions):
            observation, _, _, _ = env.step(action)
            executed.append(np.asarray(action, dtype=np.float32).copy())
            contact = bool(target_contact_state(env.sim, target_body)["target_contact"])
            if contact and row["first_contact_step"] is None:
                row["first_contact_step"] = int(step)
                row["target_contact"] = True
                remaining_after_contact = int(post_contact_steps)
            if remaining_after_contact is not None:
                controller = env.env.robots[0].controller
                error = float(np.linalg.norm(np.asarray(controller.goal_pos) - np.asarray(controller.ee_pos)))
                current_agent = np.asarray(observation["agentview_image"], dtype=np.uint8)
                current_wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8)
                errors.append(error)
                agent_mad.append(temporal_mad(previous_agent, current_agent))
                wrist_mad.append(temporal_mad(previous_wrist, current_wrist))
                remaining_after_contact -= 1
                if remaining_after_contact <= 0 and fixed_action_count is None:
                    break
            previous_agent = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
            previous_wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
            if fixed_action_count is not None and len(executed) >= int(fixed_action_count):
                break
        row["executed_steps"] = len(executed)
        row["post_contact_frames"] = len(errors)
        row["executed_action_sha256"] = array_hash(np.asarray(executed, dtype=np.float32))
        if errors:
            row["features"] = [
                float(np.mean(errors)),
                float(np.quantile(errors, 0.9)),
                float(np.max(errors)),
                float(np.mean(agent_mad)),
                float(np.max(agent_mad)),
                float(np.mean(wrist_mad)),
                float(np.max(wrist_mad)),
            ]
        else:
            row["features"] = None
        row["completed"] = True
    except Exception as exc:  # pragma: no cover
        row["exception"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def fit_discriminant(rows: list[dict[str, Any]], feature_indices: list[int]) -> dict[str, Any]:
    x = np.asarray([[row["features"][index] for index in feature_indices] for row in rows], dtype=np.float64)
    y = np.asarray([int(row["label_heavy"]) for row in rows], dtype=np.int64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-9)
    z = (x - mean) / scale
    direction = z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)
    scores = z @ direction
    threshold = 0.5 * (scores[y == 1].mean() + scores[y == 0].mean())
    return {
        "feature_indices": feature_indices,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "direction": direction.tolist(),
        "threshold": float(threshold),
    }


def score(model: dict[str, Any], features: list[float]) -> float:
    indices = model["feature_indices"]
    x = np.asarray([features[index] for index in indices], dtype=np.float64)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(z @ np.asarray(model["direction"]))


def evaluate(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        value = score(model, row["features"])
        predicted = value >= float(model["threshold"])
        scored.append({**row, "score": value, "predicted_heavy": predicted, "correct": predicted == bool(row["label_heavy"])})
    return {
        "accuracy": float(np.mean([row["correct"] for row in scored])),
        "rows": scored,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    package_root = str(LIBERO_ROOT / "libero")
    sys.path = [value for value in sys.path if value.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(LIBERO_ROOT))
    from libero.libero.envs import OffScreenRenderEnv

    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"result exists; use --resume for missing candidates: {args.output}")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            raise RuntimeError("resume protocol mismatch")
        result["resumed_at"] = timestamp()
    else:
        result = {
            "schema_version": "epoch8.active_property_probe_belief.stage0_result.v1",
            "started_at": timestamp(),
            "execution_type": "EXPERT_ACTION_REPLAY_OFFLINE_MECHANISM_GATE",
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "script_sha256": sha256_file(Path(__file__)),
            "policy_loaded_or_queried": False,
            "simulator_success_or_reward_used": False,
            "confirmation_accessed": False,
            "selected_pairs": [],
        }
    for partition, candidates, required in (
        ("training", protocol["data"]["training_candidate_demo_indices"], int(protocol["data"]["selected_training_demos_per_task"])),
        ("validation", protocol["data"]["validation_candidate_demo_indices"], int(protocol["data"]["selected_validation_demos_per_task"])),
    ):
        for task in protocol["data"]["tasks"]:
            selected = sum(
                pair["partition"] == partition and pair["target"] == task["target"]
                for pair in result["selected_pairs"]
            )
            attempted = {
                row["demo_name"]
                for row in result.get("rejected_light_attempts", [])
                if row["partition"] == partition and row["target"] == task["target"]
            }
            attempted.update(
                pair["demo_name"]
                for pair in result["selected_pairs"]
                if pair["partition"] == partition and pair["target"] == task["target"]
            )
            hdf5_path = Path(protocol["data"]["root"]) / task["hdf5"]
            for demo_index in candidates:
                if selected >= required:
                    break
                demo_name = f"demo_{demo_index}"
                if demo_name in attempted:
                    continue
                init_state, actions = read_demo(hdf5_path, demo_name)
                light = replay_probe(
                    env_class=OffScreenRenderEnv,
                    bddl=BDDL_ROOT / task["bddl"],
                    target_body=task["body"],
                    init_state=init_state,
                    actions=actions,
                    factor=1.0,
                    camera_resolution=int(protocol["probe"]["camera_resolution"]),
                    post_contact_steps=int(protocol["probe"]["stop_rule"].split()[0]),
                )
                if not light["completed"] or not light["target_contact"] or light["features"] is None:
                    result.setdefault("rejected_light_attempts", []).append(
                        {"partition": partition, "target": task["target"], "demo_name": demo_name, "result": light}
                    )
                    atomic_write_json(args.output, result)
                    continue
                heavy = replay_probe(
                    env_class=OffScreenRenderEnv,
                    bddl=BDDL_ROOT / task["bddl"],
                    target_body=task["body"],
                    init_state=init_state,
                    actions=actions,
                    factor=float(protocol["probe"]["heavy_mass_factor"]),
                    camera_resolution=int(protocol["probe"]["camera_resolution"]),
                    post_contact_steps=int(protocol["probe"]["stop_rule"].split()[0]),
                    fixed_action_count=int(light["executed_steps"]),
                )
                pair = {
                    "partition": partition,
                    "target": task["target"],
                    "demo_name": demo_name,
                    "action_source_sha256": array_hash(actions),
                    "light": light,
                    "heavy": heavy,
                    "paired_action_hash_match": light.get("executed_action_sha256") == heavy.get("executed_action_sha256"),
                }
                result["selected_pairs"].append(pair)
                selected += 1
                atomic_write_json(args.output, result)
            if selected != required:
                result["fatal_selection_error"] = f"selected only {selected}/{required} for {partition}/{task['target']}"

    min_frames = int(protocol["probe"]["minimum_post_contact_frames"])
    flat = []
    for pair in result["selected_pairs"]:
        for label, key in ((0, "light"), (1, "heavy")):
            episode = pair[key]
            flat.append(
                {
                    "partition": pair["partition"],
                    "target": pair["target"],
                    "demo_name": pair["demo_name"],
                    "label_heavy": label,
                    "features": episode.get("features"),
                    "valid": bool(
                        episode.get("completed")
                        and episode.get("target_contact")
                        and episode.get("exception") is None
                        and int(episode.get("post_contact_frames", 0)) >= min_frames
                    ),
                }
            )
    train = [row for row in flat if row["partition"] == "training" and row["valid"]]
    validation = [row for row in flat if row["partition"] == "validation" and row["valid"]]
    full_model = fit_discriminant(train, list(range(7)))
    single_model = fit_discriminant(train, [0])
    full_eval = evaluate(full_model, validation)
    single_eval = evaluate(single_model, validation)
    pair_wins = 0
    per_task: dict[str, float] = {}
    for target in ("front", "middle", "back"):
        task_rows = [row for row in full_eval["rows"] if row["target"] == target]
        per_task[target] = float(np.mean([row["correct"] for row in task_rows]))
        for demo_name in sorted({row["demo_name"] for row in task_rows}):
            values = {row["label_heavy"]: row["score"] for row in task_rows if row["demo_name"] == demo_name}
            pair_wins += int(values.get(1, -np.inf) > values.get(0, np.inf))
    required_train_pairs = int(protocol["gates"]["execution"]["selected_training_pairs_required"])
    required_val_pairs = int(protocol["gates"]["execution"]["selected_validation_pairs_required"])
    execution_pass = bool(
        len([p for p in result["selected_pairs"] if p["partition"] == "training"]) == required_train_pairs
        and len([p for p in result["selected_pairs"] if p["partition"] == "validation"]) == required_val_pairs
        and all(row["valid"] for row in flat)
        and all(pair["paired_action_hash_match"] for pair in result["selected_pairs"])
        and not result.get("fatal_selection_error")
    )
    mechanism_pass = bool(
        full_eval["accuracy"] >= float(protocol["gates"]["mechanism"]["validation_accuracy_min"])
        and pair_wins >= int(protocol["gates"]["mechanism"]["validation_paired_heavy_score_win_count_min"])
        and min(per_task.values()) >= float(protocol["gates"]["mechanism"]["validation_task_accuracy_min"])
        and full_eval["accuracy"] > float(protocol["controls"]["balanced_no_probe_prior_accuracy"])
        and full_eval["accuracy"] >= single_eval["accuracy"]
    )
    decision = (
        protocol["decisions"]["invalid"]
        if not execution_pass
        else protocol["decisions"]["positive"]
        if mechanism_pass
        else protocol["decisions"]["negative"]
    )
    result.update(
        {
            "completed_at": timestamp(),
            "classifier": full_model,
            "single_feature_control_classifier": single_model,
            "validation": full_eval,
            "single_feature_control_validation": single_eval,
            "summary": {
                "training_pairs": len([p for p in result["selected_pairs"] if p["partition"] == "training"]),
                "validation_pairs": len([p for p in result["selected_pairs"] if p["partition"] == "validation"]),
                "validation_accuracy": full_eval["accuracy"],
                "single_feature_validation_accuracy": single_eval["accuracy"],
                "paired_heavy_score_win_count": pair_wins,
                "per_task_validation_accuracy": per_task,
                "execution_pass": execution_pass,
                "mechanism_pass": mechanism_pass,
            },
            "decision": decision,
        }
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"decision": decision, **result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
