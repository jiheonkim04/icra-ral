"""Run the frozen Epoch 10 ICAE exact-state mechanics calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight_epoch10_icae_exact_states import _make_env, _sim_state  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _task_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _flatten_joint(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    pieces = []
    for item in value:
        pieces.append(np.asarray(item, dtype=np.float64).reshape(-1))
    return np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.float64)


def _operand_snapshot(task_env: Any, name: str) -> dict[str, Any]:
    state = task_env.object_states_dict[name]
    geom = state.get_geom_state()
    try:
        joints = _flatten_joint(state.get_joint_state())
    except Exception:
        joints = np.empty((0,), dtype=np.float64)
    return {
        "position": np.asarray(geom["pos"], dtype=np.float64).reshape(-1),
        "quaternion": np.asarray(geom["quat"], dtype=np.float64).reshape(-1),
        "joints": joints,
    }


def _goal_signature(task_env: Any) -> dict[str, Any]:
    predicates = [list(row) for row in task_env.parsed_problem.get("goal_state", [])]
    operands = sorted({str(name) for row in predicates for name in row[1:]})
    return {
        "predicates": predicates,
        "operands": {name: _operand_snapshot(task_env, name) for name in operands},
        "native_success": bool(task_env._check_success()),
    }


def _quat_angle(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _predicate_value(task_env: Any, predicate: Sequence[str]) -> bool:
    try:
        return bool(task_env._eval_predicate(list(predicate)))
    except Exception:
        return False


def goal_error(task_env: Any, target: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["task_error"]
    position_scale = float(config["position_scale_m"])
    orientation_scale = float(config["orientation_scale_rad"])
    joint_scale = float(config["joint_scale_native"])
    predicate_penalty = float(config["predicate_violation_penalty"])
    current = {name: _operand_snapshot(task_env, name) for name in target["operands"]}
    components: list[dict[str, Any]] = []
    for raw_predicate in target["predicates"]:
        predicate = [str(value) for value in raw_predicate]
        names = predicate[1:]
        if len(names) == 2:
            cur_rel = current[names[0]]["position"] - current[names[1]]["position"]
            tar_rel = np.asarray(target["operands"][names[0]]["position"]) - np.asarray(
                target["operands"][names[1]]["position"]
            )
            components.append(
                {
                    "kind": "relative_position",
                    "predicate": predicate[0],
                    "value": float(np.linalg.norm(cur_rel - tar_rel) / position_scale),
                }
            )
            components.append(
                {
                    "kind": "object_orientation",
                    "predicate": predicate[0],
                    "value": _quat_angle(
                        current[names[0]]["quaternion"], target["operands"][names[0]]["quaternion"]
                    )
                    / orientation_scale,
                }
            )
        elif len(names) == 1:
            name = names[0]
            current_joint = current[name]["joints"]
            target_joint = np.asarray(target["operands"][name]["joints"], dtype=np.float64)
            if current_joint.size and current_joint.shape == target_joint.shape:
                components.append(
                    {
                        "kind": "joint",
                        "predicate": predicate[0],
                        "value": float(np.mean(np.abs(current_joint - target_joint)) / joint_scale),
                    }
                )
            else:
                components.append(
                    {
                        "kind": "position",
                        "predicate": predicate[0],
                        "value": float(
                            np.linalg.norm(
                                current[name]["position"] - np.asarray(target["operands"][name]["position"])
                            )
                            / position_scale
                        ),
                    }
                )
        satisfied = _predicate_value(task_env, predicate)
        components.append(
            {
                "kind": "native_predicate_violation",
                "predicate": predicate[0],
                "value": 0.0 if satisfied else predicate_penalty,
            }
        )
    values = [float(row["value"]) for row in components]
    return {
        "raw_error": float(np.mean(values)) if values else float(config["maximum_harm_score"]),
        "native_success": bool(task_env._check_success()),
        "components": components,
    }


def _capture_target(env: Any, states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    observation = env.set_init_state(states[-1])
    del observation
    _, _, _, _ = env.step(np.clip(actions[-1], -1.0, 1.0))
    signature = _goal_signature(_task_env(env))
    signature["source_final_state_sha256"] = _array_sha256(states[-1])
    signature["source_final_action_sha256"] = _array_sha256(actions[-1])
    return signature


def _perturbation_direction(state_id: str) -> np.ndarray:
    digest = hashlib.sha256(f"epoch10-icae-perturbation|{state_id}".encode("utf-8")).digest()
    raw = np.frombuffer(digest[:6], dtype=np.int8).astype(np.float64)
    raw = np.where(raw >= 0.0, 1.0, -1.0)
    raw /= max(float(np.max(np.abs(raw))), 1.0)
    return np.concatenate([raw, np.zeros((1,), dtype=np.float64)])


def _branch_order(state_id: str, horizon: int) -> list[str]:
    names = ["nominal_reference", "small_perturbation", "medium_perturbation", "nominal_sham"]
    return sorted(
        names,
        key=lambda name: hashlib.sha256(f"epoch10-order|{state_id}|{horizon}|{name}".encode("utf-8")).hexdigest(),
    )


def execute_branch(
    env: Any,
    *,
    state: np.ndarray,
    first_action: np.ndarray,
    continuation: np.ndarray,
    target: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {
        "valid": False,
        "error": None,
        "requested_control_steps": int(1 + len(continuation)),
        "completed_control_steps": 0,
    }
    try:
        env.set_init_state(state)
        restored = _sim_state(env)
        row["restore_l2"] = float(np.linalg.norm(restored - state))
        raw_actions = np.vstack([np.asarray(first_action).reshape(1, 7), np.asarray(continuation).reshape(-1, 7)])
        executed_actions = np.clip(raw_actions, -1.0, 1.0)
        row["first_action_raw"] = raw_actions[0]
        row["first_action_executed"] = executed_actions[0]
        row["first_action_clipped"] = bool(np.max(np.abs(raw_actions[0] - executed_actions[0])) > 1e-12)
        premature_terminal = False
        for index, action in enumerate(executed_actions):
            _, _, done, _ = env.step(action)
            row["completed_control_steps"] += 1
            if bool(done) and index + 1 < len(executed_actions):
                premature_terminal = True
                break
        final_state = _sim_state(env)
        row["final_state_sha256"] = _array_sha256(final_state)
        row["final_state"] = final_state
        row["final_state_finite"] = bool(np.isfinite(final_state).all())
        row["premature_terminal"] = premature_terminal
        row["goal_error"] = goal_error(_task_env(env), target, protocol)
        row["valid"] = bool(
            row["restore_l2"] <= 1e-8
            and row["completed_control_steps"] == row["requested_control_steps"]
            and row["final_state_finite"]
            and not premature_terminal
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _bootstrap_episode_difference(
    episode_rows: Mapping[str, list[tuple[float, float]]], *, seed: int, replicates: int
) -> dict[str, Any]:
    keys = sorted(episode_rows)
    episode_differences = np.asarray(
        [np.mean([medium - small for small, medium in episode_rows[key]]) for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = rng.integers(0, len(keys), size=len(keys))
        draws[index] = float(np.mean(episode_differences[sampled]))
    return {
        "episode_count": len(keys),
        "episode_mean_differences": episode_differences,
        "mean_difference": float(np.mean(episode_differences)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_95_interval": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.monotonic()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    if preflight["status"] != "PASS" or int(preflight["checkpoint_actions_queried"]) != 0:
        raise RuntimeError("exact-state preflight is not an outcome-blind PASS")
    mechanics_rows = [row for row in preflight["rows"] if row["partition"] == "mechanics_calibration"]
    task_by_key = {(row["suite"], int(row["task_id"])): row for row in preflight["tasks"]}
    rows_by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mechanics_rows:
        rows_by_task[(str(row["suite"]), int(row["task_id"]))].append(row)

    all_rows: list[dict[str, Any]] = []
    target_audits: list[dict[str, Any]] = []
    envs: dict[tuple[str, int], Any] = {}
    hdf5_handles: dict[tuple[str, int], Any] = {}
    try:
        for task_key, registered_rows in rows_by_task.items():
            task = task_by_key[task_key]
            env = _make_env(Path(task["bddl_file"]), int(args.camera_size))
            env.seed(int(args.seed))
            env.reset()
            envs[task_key] = env
            handle = h5py.File(task["hdf5_path"], "r")
            hdf5_handles[task_key] = handle
            targets: dict[str, dict[str, Any]] = {}
            for demo_name in sorted({str(row["demo_name"]) for row in registered_rows}):
                demo = handle["data"][demo_name]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
                target = _capture_target(env, states, actions)
                targets[demo_name] = target
                target_audits.append(
                    {
                        "suite": task_key[0],
                        "task_id": task_key[1],
                        "demo_name": demo_name,
                        "target_native_success": bool(target["native_success"]),
                        "goal_predicates": target["predicates"],
                        "operand_names": sorted(target["operands"]),
                    }
                )
            for registered in sorted(registered_rows, key=lambda row: row["state_id"]):
                demo_name = str(registered["demo_name"])
                demo = handle["data"][demo_name]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
                frame = int(registered["frame"])
                state = states[frame]
                expert = actions[frame]
                direction = _perturbation_direction(str(registered["state_id"]))
                first_actions = {
                    "nominal_reference": expert,
                    "nominal_sham": expert,
                    "small_perturbation": np.clip(
                        expert + float(protocol["mechanics_calibration"]["small_action_perturbation_linf"]) * direction,
                        -1.0,
                        1.0,
                    ),
                    "medium_perturbation": np.clip(
                        expert + float(protocol["mechanics_calibration"]["medium_action_perturbation_linf"]) * direction,
                        -1.0,
                        1.0,
                    ),
                }
                for horizon in protocol["paired_splice"]["horizon_candidates"]:
                    continuation = actions[frame + 1 : frame + 1 + int(horizon)]
                    branch_rows = {}
                    order = _branch_order(str(registered["state_id"]), int(horizon))
                    for branch in order:
                        branch_rows[branch] = execute_branch(
                            env,
                            state=state,
                            first_action=first_actions[branch],
                            continuation=continuation,
                            target=targets[demo_name],
                            protocol=protocol,
                        )
                    nominal = branch_rows["nominal_reference"]
                    sham = branch_rows["nominal_sham"]
                    small = branch_rows["small_perturbation"]
                    medium = branch_rows["medium_perturbation"]
                    nominal_state = np.asarray(nominal.get("final_state", []), dtype=np.float64)
                    state_dim = max(1, nominal_state.size)

                    def state_rms(branch: Mapping[str, Any]) -> float:
                        candidate = np.asarray(branch.get("final_state", []), dtype=np.float64)
                        if candidate.shape != nominal_state.shape or not candidate.size:
                            return float(protocol["task_error"]["maximum_harm_score"])
                        return float(np.linalg.norm(candidate - nominal_state) / math.sqrt(state_dim))

                    all_rows.append(
                        {
                            "state_id": registered["state_id"],
                            "episode_group": f"{registered['suite']}|task_{registered['task_id']}|{demo_name}",
                            "suite": registered["suite"],
                            "task_id": int(registered["task_id"]),
                            "demo_name": demo_name,
                            "frame": frame,
                            "horizon": int(horizon),
                            "branch_order": order,
                            "direction": direction,
                            "nominal_valid": bool(nominal["valid"]),
                            "sham_valid": bool(sham["valid"]),
                            "nominal_sham_state_l2": state_rms(sham) * math.sqrt(state_dim),
                            "nominal_sham_task_error_abs": abs(
                                float(nominal.get("goal_error", {}).get("raw_error", 1e9))
                                - float(sham.get("goal_error", {}).get("raw_error", -1e9))
                            ),
                            "small_state_deterioration_rms": state_rms(small),
                            "medium_state_deterioration_rms": state_rms(medium),
                            "small_goal_deterioration": float(small.get("goal_error", {}).get("raw_error", 10.0))
                            - float(nominal.get("goal_error", {}).get("raw_error", 0.0)),
                            "medium_goal_deterioration": float(medium.get("goal_error", {}).get("raw_error", 10.0))
                            - float(nominal.get("goal_error", {}).get("raw_error", 0.0)),
                            "branches": branch_rows,
                        }
                    )
    finally:
        for handle in hdf5_handles.values():
            handle.close()
        for env in envs.values():
            env.close()

    horizon_audits = []
    selected_horizon = None
    cfg = protocol["mechanics_calibration"]
    for horizon in protocol["paired_splice"]["horizon_candidates"]:
        rows = [row for row in all_rows if int(row["horizon"]) == int(horizon)]
        restore_passes = [
            row["nominal_sham_state_l2"] <= float(cfg["restore_sham_tolerance_state_l2"])
            and row["nominal_sham_task_error_abs"] <= float(cfg["restore_sham_tolerance_task_error"])
            for row in rows
        ]
        episode_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            episode_values[str(row["episode_group"])].append(
                (float(row["small_state_deterioration_rms"]), float(row["medium_state_deterioration_rms"]))
            )
        grouped = _bootstrap_episode_difference(
            episode_values, seed=int(args.seed) + int(horizon), replicates=int(args.bootstrap_replicates)
        )
        restore_rate = float(np.mean(restore_passes))
        nominal_valid_rate = float(np.mean([bool(row["nominal_valid"] and row["sham_valid"]) for row in rows]))
        branch_order_effect = max(float(row["nominal_sham_state_l2"]) for row in rows)
        monotone = bool(
            grouped["mean_difference"] >= 0.0 and float(grouped["bootstrap_95_interval"][0]) >= -1e-6
        )
        passed = bool(
            restore_rate >= float(cfg["minimum_restore_sham_equivalence_rate"])
            and nominal_valid_rate >= float(cfg["minimum_valid_nominal_rate"])
            and branch_order_effect <= float(cfg["branch_order_materiality_tolerance"])
            and monotone
        )
        audit = {
            "horizon": int(horizon),
            "row_count": len(rows),
            "restore_sham_equivalence_rate": restore_rate,
            "valid_nominal_response_rate": nominal_valid_rate,
            "maximum_nominal_sham_state_l2": branch_order_effect,
            "grouped_medium_minus_small": grouped,
            "monotone_response_pass": monotone,
            "pass": passed,
        }
        horizon_audits.append(audit)
        if passed and selected_horizon is None:
            selected_horizon = int(horizon)

    report = {
        "schema_version": 1,
        "campaign": "epoch10_icae_vla",
        "status": "MECHANICS_CALIBRATION_PASS" if selected_horizon is not None else "MECHANICS_CALIBRATION_FAIL",
        "protocol_path": str(Path(args.protocol)),
        "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        "preflight_path": str(Path(args.preflight)),
        "preflight_sha256": hashlib.sha256(Path(args.preflight).read_bytes()).hexdigest(),
        "checkpoint_actions_queried": 0,
        "prospective_checkpoint_outcomes_read": False,
        "official_outcomes_opened": False,
        "selected_horizon": selected_horizon,
        "target_audits": target_audits,
        "all_targets_native_success": all(bool(row["target_native_success"]) for row in target_audits),
        "horizon_audits": horizon_audits,
        "rows": all_rows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="reports/epoch10_icae_intervention_protocol.json")
    parser.add_argument("--preflight", default="reports/epoch10_icae_exact_state_preflight_attempt2.json")
    parser.add_argument("--output", default="reports/epoch10_icae_mechanics_calibration.json")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    _write_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_horizon": report["selected_horizon"],
                "rows": len(report["rows"]),
                "targets_native_success": report["all_targets_native_success"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )
    return 0 if report["status"] == "MECHANICS_CALIBRATION_PASS" and report["all_targets_native_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
