"""Fresh-controller engineering and mechanics certification for Epoch 10B ICAE.

This runner is checkpoint- and outcome-blind.  It uses raw demonstrations only
to reconstruct controller state, execute preregistered mechanics controls, and
certify a deterministic intervention assay before any checkpoint action or
closed-loop label may be opened.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
import weakref
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight_epoch10_icae_exact_states import (  # noqa: E402
    TASK_SPECS,
    _make_env,
    _resolve_bddl_file,
    _sim_state,
    _task_for_stem,
)
from scripts.run_epoch10_icae_mechanics import _capture_target, goal_error  # noqa: E402


CAMPAIGN = "epoch10b_icae_fresh_controller"
SOURCE_COMMIT = "bffbb29d7e638f66d4da5a2c45c84a727da7abec"
ENGINEERING_DEMOS = (0,)
CERTIFICATION_DEMOS = (8, 9, 10, 11)
DEVELOPMENT_RESERVED_DEMOS = tuple(range(12, 20))
HELDOUT_RESERVED_DEMOS = tuple(range(20, 28))
STATE_PHASE_SPECS = (
    (0.10, "free_motion"),
    (0.20, "free_motion"),
    (0.35, "approach"),
    (0.45, "approach"),
    (0.58, "contact_grasp_release"),
    (0.68, "contact_grasp_release"),
    (0.80, "transport_goal"),
    (0.90, "transport_goal"),
)
HORIZONS = (4, 8, 16)
CONTROL_NAMES = (
    "nominal_a",
    "nominal_b",
    "sham",
    "held_noop",
    "harmful_phase_matched",
    "small_plus",
    "small_minus",
    "medium_plus",
    "medium_minus",
)
SMALL_DELTA = 0.05
MEDIUM_DELTA = 0.15
TWIN_SCORE_TOLERANCE = 1e-8
TWIN_STATE_MATERIALITY = 1e-6
PREFIX_FIDELITY_TOLERANCE = 1e-5
ENGINEERING_TWIN_TOLERANCE = 1e-8
PREFIX_ONE_STEP_RMS_TOLERANCE = 0.03
PREFIX_TERMINAL_RMS_TOLERANCE = 0.05
PREFIX_RELATIVE_ONE_STEP_RATIO = 0.20
BOOTSTRAP_REPLICATES = 4000
TASK_ERROR_PROTOCOL = {
    "task_error": {
        "position_scale_m": 0.05,
        "orientation_scale_rad": 0.5,
        "joint_scale_native": 0.5,
        "predicate_violation_penalty": 1.0,
        "maximum_harm_score": 10.0,
    }
}


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") & 0x7FFFFFFF


def seed_runtime(seed: int) -> dict[str, int]:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    return {"python_random_seed": int(seed), "numpy_seed": int(seed) % (2**32 - 1), "environment_seed": int(seed)}


@contextlib.contextmanager
def closing_environment(factory: Callable[[], Any]) -> Iterator[tuple[Any, dict[str, Any]]]:
    env = factory()
    audit: dict[str, Any] = {
        "environment_class": f"{env.__class__.__module__}.{env.__class__.__qualname__}",
        "environment_object_id": int(id(env)),
        "close_called": False,
        "weakref_released_after_close": None,
    }
    reference = weakref.ref(env)
    try:
        yield env, audit
    finally:
        try:
            env.close()
            audit["close_called"] = True
        finally:
            del env
            gc.collect()
            audit["weakref_released_after_close"] = reference() is None


def _task_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _observation_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"{")
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _safe_public_state(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size <= 64 and np.issubdtype(array.dtype, np.number):
            return array.astype(np.float64).tolist()
        return {"shape": list(array.shape), "sha256": _array_sha256(array)}
    if isinstance(value, (list, tuple)) and len(value) <= 16:
        converted = []
        for item in value:
            if isinstance(item, (bool, int, float, str, np.generic)):
                converted.append(_json_default(item) if isinstance(item, np.generic) else item)
            else:
                return None
        return converted
    return None


def controller_snapshot(env: Any) -> dict[str, Any]:
    task = _task_env(env)
    robots = getattr(task, "robots", [])
    controller = getattr(robots[0], "controller", None) if robots else None
    if controller is None:
        return {"available": False, "sha256": _canonical_sha256({"available": False})}
    fields: dict[str, Any] = {}
    for name, value in sorted(vars(controller).items()):
        lowered = name.lower()
        if any(token in lowered for token in ("goal", "integr", "interp", "error", "counter", "step", "queue", "buffer")):
            converted = _safe_public_state(value)
            if converted is not None:
                fields[name] = converted
    snapshot = {
        "available": True,
        "class": f"{controller.__class__.__module__}.{controller.__class__.__qualname__}",
        "public_execution_fields": fields,
    }
    snapshot["sha256"] = _canonical_sha256(snapshot)
    return snapshot


def action_queue_audit(env: Any) -> dict[str, Any]:
    task = _task_env(env)
    objects = [env, task]
    robots = getattr(task, "robots", [])
    if robots:
        objects.extend([robots[0], getattr(robots[0], "controller", None)])
    rows = []
    for obj in objects:
        if obj is None or not hasattr(obj, "__dict__"):
            continue
        for name, value in vars(obj).items():
            if not any(token in name.lower() for token in ("queue", "buffer", "history")):
                continue
            length = None
            try:
                length = len(value)
            except Exception:
                pass
            rows.append({"owner": obj.__class__.__qualname__, "attribute": name, "length": length})
    return {"inspected": True, "queue_like_fields": rows, "policy_queue_present": False}


def _read_wsl_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(raw.strip().split()[0]) * 1024
    except Exception:
        pass
    return {
        "wsl_mem_total_bytes": values.get("MemTotal", 0),
        "wsl_mem_available_bytes": values.get("MemAvailable", 0),
        "wsl_swap_total_bytes": values.get("SwapTotal", 0),
        "wsl_swap_used_bytes": max(0, values.get("SwapTotal", 0) - values.get("SwapFree", 0)),
    }


def _read_host_ram_percent() -> float | None:
    executable = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not executable.is_file():
        return None
    script = (
        "[Console]::Write([math]::Round((1-"
        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/"
        "(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize)*100,4))"
    )
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", f"{executable} -NoProfile -Command '{script}'"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        decoded = result.stdout.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        return float(decoded)
    except Exception:
        return None


def resource_sample() -> dict[str, Any]:
    sample = {"monotonic_seconds": time.monotonic(), "host_ram_percent": _read_host_ram_percent(), **_read_wsl_memory()}
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        used, total, util = [float(part.strip()) for part in raw.split(",")]
        sample.update({"gpu_vram_used_mib": used, "gpu_vram_total_mib": total, "gpu_utilization_percent": util})
    except Exception:
        sample.update({"gpu_vram_used_mib": None, "gpu_vram_total_mib": None, "gpu_utilization_percent": None})
    return sample


def _frame_for_fraction(length: int, fraction: float, horizon_guard: int = 17) -> int:
    last = max(1, int(length) - int(horizon_guard) - 1)
    return min(last, max(1, int(round(float(fraction) * (length - 1)))))


def _perturbation_direction(state_id: str) -> np.ndarray:
    raw = np.frombuffer(hashlib.sha256(f"epoch10b-symmetric|{state_id}".encode("utf-8")).digest()[:6], dtype=np.int8)
    direction = np.where(raw >= 0, 1.0, -1.0)
    return np.concatenate([direction, np.zeros((1,), dtype=np.float64)])


def _harmful_action(expert: np.ndarray, phase: str, state_id: str) -> np.ndarray:
    action = np.asarray(expert, dtype=np.float64).copy()
    if phase == "contact_grasp_release":
        action[6] = -1.0 if float(expert[6]) >= 0.0 else 1.0
        return np.clip(action, -1.0, 1.0)
    translation = np.asarray(expert[:3], dtype=np.float64)
    norm = float(np.linalg.norm(translation))
    if norm <= 1e-9:
        translation = _perturbation_direction(state_id)[:3]
        norm = float(np.linalg.norm(translation))
    action[:3] = -0.5 * translation / max(norm, 1e-12)
    return np.clip(action, -1.0, 1.0)


def control_actions(expert: np.ndarray, phase: str, state_id: str) -> dict[str, np.ndarray]:
    expert = np.clip(np.asarray(expert, dtype=np.float64), -1.0, 1.0)
    direction = _perturbation_direction(state_id)
    held = np.zeros((7,), dtype=np.float64)
    held[6] = expert[6]
    return {
        "nominal_a": expert.copy(),
        "nominal_b": expert.copy(),
        "sham": expert.copy(),
        "held_noop": held,
        "harmful_phase_matched": _harmful_action(expert, phase, state_id),
        "small_plus": np.clip(expert + SMALL_DELTA * direction, -1.0, 1.0),
        "small_minus": np.clip(expert - SMALL_DELTA * direction, -1.0, 1.0),
        "medium_plus": np.clip(expert + MEDIUM_DELTA * direction, -1.0, 1.0),
        "medium_minus": np.clip(expert - MEDIUM_DELTA * direction, -1.0, 1.0),
    }


def _load_task_records(raw_root: Path) -> list[dict[str, Any]]:
    records = []
    for spec in TASK_SPECS:
        suite_name = str(spec["suite"])
        stem = str(spec["hdf5_stem"])
        task_id, task, _suite = _task_for_stem(suite_name, stem)
        bddl = _resolve_bddl_file(task)
        hdf5_path = raw_root / suite_name / f"{stem}_demo.hdf5"
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        records.append(
            {
                **spec,
                "task_id": int(task_id),
                "instruction": str(task.language),
                "bddl_file": str(bddl.resolve()),
                "bddl_sha256": _sha256_file(bddl),
                "hdf5_path": str(hdf5_path.resolve()),
                "hdf5_sha256": _sha256_file(hdf5_path),
            }
        )
    return records


def _state_manifest(tasks: Sequence[Mapping[str, Any]], demo_ids: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        with h5py.File(task["hdf5_path"], "r") as handle:
            for demo_id in demo_ids:
                demo_name = f"demo_{int(demo_id)}"
                demo = handle["data"][demo_name]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
                for phase_index, (fraction, phase) in enumerate(STATE_PHASE_SPECS):
                    frame = _frame_for_fraction(len(actions), fraction)
                    state_id = f"{task['suite']}|task_{task['task_id']}|{demo_name}|frame_{frame}|{phase}"
                    rows.append(
                        {
                            "state_id": state_id,
                            "suite": task["suite"],
                            "task_id": int(task["task_id"]),
                            "demo_name": demo_name,
                            "demo_cluster": f"{task['suite']}|task_{task['task_id']}|{demo_name}",
                            "episode_length": int(len(actions)),
                            "frame": int(frame),
                            "fraction": float(fraction),
                            "phase": phase,
                            "phase_index": int(phase_index),
                            "state_sha256": _array_sha256(states[frame]),
                            "expert_action_sha256": _array_sha256(actions[frame]),
                            "registered_seed": _seed_from_text(f"epoch10b-reset|{state_id}"),
                            "reverse_order_duplicate": bool(demo_id == demo_ids[0] and phase_index % 2 == 0),
                        }
                    )
    return rows


def _target_for_demo(task: Mapping[str, Any], states: np.ndarray, actions: np.ndarray, camera_size: int) -> dict[str, Any]:
    seed = _seed_from_text(f"epoch10b-target|{task['suite']}|{task['task_id']}|{len(actions)}")
    with closing_environment(lambda: _make_env(Path(task["bddl_file"]), int(camera_size))) as (env, _cleanup):
        seed_runtime(seed)
        env.seed(seed)
        env.reset()
        return _capture_target(env, states, actions)


def _prefix_reconstruct(env: Any, states: np.ndarray, actions: np.ndarray, frame: int) -> dict[str, Any]:
    observation = env.set_init_state(states[0])
    initial_l2 = float(np.linalg.norm(_sim_state(env) - states[0]))
    errors = []
    hash_matches = 0
    premature_terminal = False
    for index in range(int(frame)):
        observation, _reward, done, _info = env.step(np.clip(actions[index], -1.0, 1.0))
        expected = states[index + 1]
        actual = _sim_state(env)
        error = float(np.linalg.norm(actual - expected))
        errors.append(error)
        hash_matches += int(_array_sha256(actual) == _array_sha256(expected))
        if bool(done) and index + 1 < int(frame):
            premature_terminal = True
            break
    return {
        "initial_restore_l2": initial_l2,
        "prefix_steps_requested": int(frame),
        "prefix_steps_completed": len(errors),
        "prefix_state_hash_matches": int(hash_matches),
        "prefix_max_state_l2": max(errors, default=0.0),
        "prefix_terminal_state_l2_before_exact_set": errors[-1] if errors else initial_l2,
        "prefix_premature_terminal": premature_terminal,
        "last_observation_hash_before_exact_set": _observation_hash(observation),
    }


def execute_fresh_branch(
    *,
    task: Mapping[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
    target: Mapping[str, Any],
    frame: int,
    first_action: np.ndarray,
    design: str,
    branch_key: str,
    camera_size: int,
    horizon: int = 16,
    registered_seed: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    seed = int(registered_seed) if registered_seed is not None else _seed_from_text(f"epoch10b-branch|{branch_key}")
    before_resource = resource_sample()
    row: dict[str, Any] = {
        "branch_key": branch_key,
        "design": design,
        "registered_seed": int(seed),
        "rng_identity": seed_runtime(seed),
        "valid": False,
        "error": None,
        "termination_initialized_false": True,
        "requested_control_steps": int(1 + horizon),
        "completed_control_steps": 0,
        "prefix": None,
    }
    cleanup: dict[str, Any] = {}
    try:
        with closing_environment(lambda: _make_env(Path(task["bddl_file"]), int(camera_size))) as (env, cleanup):
            env.seed(seed)
            env.reset()
            reset_controller = controller_snapshot(env)
            reset_queue = action_queue_audit(env)
            if design == "prefix_reconstructed_fresh_controller":
                row["prefix"] = _prefix_reconstruct(env, states, actions, int(frame))
            elif design != "fresh_controller_restore":
                raise ValueError(f"unknown constructor design: {design}")
            observation = env.set_init_state(states[frame])
            restored = _sim_state(env)
            row.update(
                {
                    "pre_action_state_l2": float(np.linalg.norm(restored - states[frame])),
                    "pre_action_state_sha256": _array_sha256(restored),
                    "registered_state_sha256": _array_sha256(states[frame]),
                    "pre_action_observation_sha256": _observation_hash(observation),
                    "reset_controller_snapshot": reset_controller,
                    "pre_action_controller_snapshot": controller_snapshot(env),
                    "action_queue_audit": reset_queue,
                }
            )
            sequence = np.vstack(
                [
                    np.asarray(first_action, dtype=np.float64).reshape(1, 7),
                    np.asarray(actions[frame + 1 : frame + 1 + int(horizon)], dtype=np.float64).reshape(-1, 7),
                ]
            )
            sequence = np.clip(sequence, -1.0, 1.0)
            state_trace = []
            goal_trace = []
            expert_state_l2_trace = []
            terminal_trace = []
            for index, action in enumerate(sequence):
                observation, reward, done, info = env.step(action)
                actual = _sim_state(env)
                expected_index = min(int(frame + index + 1), len(states) - 1)
                expected = states[expected_index]
                state_trace.append(actual.copy())
                goal_trace.append(float(goal_error(_task_env(env), target, TASK_ERROR_PROTOCOL)["raw_error"]))
                expert_state_l2_trace.append(float(np.linalg.norm(actual - expected) / math.sqrt(max(1, actual.size))))
                terminal_trace.append({"step": index + 1, "done": bool(done), "reward": float(reward)})
                row["completed_control_steps"] += 1
                if bool(done):
                    break
            absorbing_steps = 0
            if terminal_trace and terminal_trace[-1]["done"] and len(state_trace) < len(sequence):
                terminal_state = state_trace[-1].copy()
                terminal_goal = goal_trace[-1]
                for index in range(len(state_trace), len(sequence)):
                    expected_index = min(int(frame + index + 1), len(states) - 1)
                    expected = states[expected_index]
                    state_trace.append(terminal_state.copy())
                    goal_trace.append(terminal_goal)
                    expert_state_l2_trace.append(
                        float(np.linalg.norm(terminal_state - expected) / math.sqrt(max(1, terminal_state.size)))
                    )
                    absorbing_steps += 1
            state_by_horizon = {}
            goal_auc_by_horizon = {}
            recovery_by_horizon = {}
            for candidate_horizon in HORIZONS:
                used = min(int(candidate_horizon) + 1, len(state_trace))
                if used <= 0:
                    continue
                state_by_horizon[str(candidate_horizon)] = state_trace[used - 1]
                goal_auc_by_horizon[str(candidate_horizon)] = float(np.mean(goal_trace[:used]))
                recovery_by_horizon[str(candidate_horizon)] = float(np.mean(expert_state_l2_trace[:used]))
            row.update(
                {
                    "executed_first_action": sequence[0],
                    "executed_first_action_sha256": _array_sha256(sequence[0]),
                    "one_step_state_l2_to_registered_next": expert_state_l2_trace[0] * math.sqrt(max(1, state_trace[0].size)),
                    "state_dimension": int(state_trace[0].size),
                    "first_step_state": state_trace[0],
                    "state_by_horizon": state_by_horizon,
                    "native_goal_error_auc_by_horizon": goal_auc_by_horizon,
                    "bounded_recovery_cost_by_horizon": recovery_by_horizon,
                    "terminal_trace": terminal_trace,
                    "final_state_finite": bool(state_trace and np.isfinite(state_trace[-1]).all()),
                    "candidate_only_terminal": bool(terminal_trace and terminal_trace[-1]["done"]),
                    "absorbing_terminal_steps_scored_without_additional_env_step": int(absorbing_steps),
                }
            )
            row["valid"] = bool(
                row["pre_action_state_l2"] <= 1e-10
                and row["completed_control_steps"] > 0
                and row["final_state_finite"]
            )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["cleanup"] = cleanup
    row["resource_before"] = before_resource
    row["resource_after"] = resource_sample()
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _state_l2(left: Mapping[str, Any], right: Mapping[str, Any], horizon: int) -> float:
    a = np.asarray(left["state_by_horizon"][str(horizon)], dtype=np.float64)
    b = np.asarray(right["state_by_horizon"][str(horizon)], dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _task_error_abs(left: Mapping[str, Any], right: Mapping[str, Any], horizon: int) -> float:
    return abs(float(left["native_goal_error_auc_by_horizon"][str(horizon)]) - float(right["native_goal_error_auc_by_horizon"][str(horizon)]))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0

    def ranks(values: Sequence[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values), kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return result

    a = ranks(left)
    b = ranks(right)
    if float(np.std(a)) <= 1e-15 and float(np.std(b)) <= 1e-15:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _icc_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    data = np.column_stack([left, right]).astype(np.float64)
    n, k = data.shape
    if n < 2:
        return 0.0
    row_means = np.mean(data, axis=1)
    column_means = np.mean(data, axis=0)
    grand = float(np.mean(data))
    ms_rows = float(k * np.sum((row_means - grand) ** 2) / max(1, n - 1))
    ms_error = float(
        np.sum((data - row_means[:, None] - column_means[None, :] + grand) ** 2) / max(1, (n - 1) * (k - 1))
    )
    denominator = ms_rows + (k - 1) * ms_error
    if denominator <= 1e-30:
        return 1.0 if np.array_equal(data[:, 0], data[:, 1]) else 0.0
    return float((ms_rows - ms_error) / denominator)


def _cluster_bootstrap(values: Mapping[str, Sequence[float]], seed: int) -> dict[str, Any]:
    keys = sorted(values)
    cluster_means = np.asarray([np.mean(values[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((BOOTSTRAP_REPLICATES,), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(keys), size=len(keys))
        draws[index] = float(np.mean(cluster_means[sample]))
    return {
        "cluster_count": len(keys),
        "mean": float(np.mean(cluster_means)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_95_interval": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def run_engineering(args: argparse.Namespace) -> dict[str, Any]:
    existing_output = Path(args.attempts_output)
    prior_attempt = json.loads(existing_output.read_text(encoding="utf-8")) if existing_output.is_file() else None
    tasks = _load_task_records(Path(args.raw_root))
    states_manifest = _state_manifest(tasks, ENGINEERING_DEMOS)
    selected = [row for row in states_manifest if int(row["phase_index"]) == 2]
    attempts = []
    for state_row in selected:
        task = next(row for row in tasks if row["suite"] == state_row["suite"] and int(row["task_id"]) == int(state_row["task_id"]))
        with h5py.File(task["hdf5_path"], "r") as handle:
            demo = handle["data"][state_row["demo_name"]]
            states = np.asarray(demo["states"], dtype=np.float64)
            actions = np.asarray(demo["actions"], dtype=np.float64)
        target = _target_for_demo(task, states, actions, int(args.camera_size))
        design_rows = {}
        for design in ("fresh_controller_restore", "prefix_reconstructed_fresh_controller"):
            twins = []
            for duplicate in ("a", "b"):
                key = f"engineering|{state_row['state_id']}|{design}|{duplicate}"
                twins.append(
                    execute_fresh_branch(
                        task=task,
                        states=states,
                        actions=actions,
                        target=target,
                        frame=int(state_row["frame"]),
                        first_action=actions[int(state_row["frame"])],
                        design=design,
                        branch_key=key,
                        camera_size=int(args.camera_size),
                        horizon=4,
                        registered_seed=int(state_row["registered_seed"]),
                    )
                )
            design_rows[design] = {
                "twins": twins,
                "pre_action_state_l2": max(float(row.get("pre_action_state_l2", math.inf)) for row in twins),
                "pre_action_observations_identical": twins[0].get("pre_action_observation_sha256") == twins[1].get("pre_action_observation_sha256"),
                "pre_action_controller_snapshots_identical": twins[0].get("pre_action_controller_snapshot", {}).get("sha256") == twins[1].get("pre_action_controller_snapshot", {}).get("sha256"),
                "final_state_l2": _state_l2(twins[0], twins[1], 4) if all(row.get("valid") for row in twins) else math.inf,
                "task_error_abs": _task_error_abs(twins[0], twins[1], 4) if all(row.get("valid") for row in twins) else math.inf,
                "one_step_replay_l2_max": max(float(row.get("one_step_state_l2_to_registered_next", math.inf)) for row in twins),
                "one_step_replay_rms_max": max(
                    float(row.get("one_step_state_l2_to_registered_next", math.inf))
                    / math.sqrt(max(1, int(row.get("state_dimension", 1))))
                    for row in twins
                ),
                "prefix_terminal_l2_max": max(
                    float((row.get("prefix") or {}).get("prefix_terminal_state_l2_before_exact_set", 0.0)) for row in twins
                ),
                "prefix_terminal_rms_max": max(
                    float((row.get("prefix") or {}).get("prefix_terminal_state_l2_before_exact_set", 0.0))
                    / math.sqrt(max(1, int(row.get("state_dimension", 1))))
                    for row in twins
                ),
                "all_cleanup_called": all(bool(row.get("cleanup", {}).get("close_called")) for row in twins),
                "all_valid": all(bool(row.get("valid")) for row in twins),
            }
        attempts.append({"state": state_row, "designs": design_rows})
    summaries = {}
    for design in ("fresh_controller_restore", "prefix_reconstructed_fresh_controller"):
        rows = [attempt["designs"][design] for attempt in attempts]
        summaries[design] = {
            "states": len(rows),
            "all_valid": all(row["all_valid"] for row in rows),
            "all_cleanup_called": all(row["all_cleanup_called"] for row in rows),
            "all_pre_action_observations_identical": all(row["pre_action_observations_identical"] for row in rows),
            "all_pre_action_controller_snapshots_identical": all(row["pre_action_controller_snapshots_identical"] for row in rows),
            "maximum_twin_final_state_l2": max(row["final_state_l2"] for row in rows),
            "maximum_twin_task_error_abs": max(row["task_error_abs"] for row in rows),
            "maximum_one_step_replay_l2": max(row["one_step_replay_l2_max"] for row in rows),
            "maximum_one_step_replay_rms": max(row["one_step_replay_rms_max"] for row in rows),
            "maximum_prefix_terminal_l2": max(row["prefix_terminal_l2_max"] for row in rows),
            "maximum_prefix_terminal_rms": max(row["prefix_terminal_rms_max"] for row in rows),
        }
    prefix_relative_ratios = [
        prefix["one_step_replay_l2_max"] / max(direct["one_step_replay_l2_max"], 1e-12)
        for direct, prefix in zip(
            [attempt["designs"]["fresh_controller_restore"] for attempt in attempts],
            [attempt["designs"]["prefix_reconstructed_fresh_controller"] for attempt in attempts],
        )
    ]
    summaries["fresh_controller_restore"]["prefix_fidelity_pass"] = True
    summaries["prefix_reconstructed_fresh_controller"]["maximum_one_step_error_ratio_vs_direct"] = max(prefix_relative_ratios)
    summaries["prefix_reconstructed_fresh_controller"]["prefix_fidelity_pass"] = bool(
        summaries["prefix_reconstructed_fresh_controller"]["maximum_one_step_replay_rms"] <= PREFIX_ONE_STEP_RMS_TOLERANCE
        and summaries["prefix_reconstructed_fresh_controller"]["maximum_prefix_terminal_rms"] <= PREFIX_TERMINAL_RMS_TOLERANCE
        and max(prefix_relative_ratios) <= PREFIX_RELATIVE_ONE_STEP_RATIO
    )
    for design in ("fresh_controller_restore", "prefix_reconstructed_fresh_controller"):
        prefix_ok = bool(summaries[design]["prefix_fidelity_pass"])
        summaries[design]["pass"] = bool(
            summaries[design]["all_valid"]
            and summaries[design]["all_cleanup_called"]
            and summaries[design]["all_pre_action_observations_identical"]
            and summaries[design]["maximum_twin_final_state_l2"] <= ENGINEERING_TWIN_TOLERANCE
            and summaries[design]["maximum_twin_task_error_abs"] <= ENGINEERING_TWIN_TOLERANCE
            and prefix_ok
        )
    if summaries["prefix_reconstructed_fresh_controller"]["pass"]:
        selected_design = "prefix_reconstructed_fresh_controller"
        selection_reason = "Prefix data were available and the preferred prefix-reconstructed design passed deterministic twins and prefix fidelity."
    elif summaries["fresh_controller_restore"]["pass"] and summaries["fresh_controller_restore"]["maximum_one_step_replay_l2"] <= PREFIX_FIDELITY_TOLERANCE:
        selected_design = "fresh_controller_restore"
        selection_reason = "Direct fresh-controller restore passed twins and the registered one-step expert replay tolerance."
    else:
        selected_design = None
        selection_reason = "Neither allowed in-process initialization design passed its frozen engineering requirements; process isolation is required."
    report = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "source_commit": SOURCE_COMMIT,
        "evidence_role": "OLD_EPOCH10_MECHANICS_ROWS_FOR_ENGINEERING_ONLY",
        "checkpoint_actions_queried": 0,
        "closed_loop_outcomes_opened": False,
        "thresholds": {
            "engineering_twin_state_and_task_tolerance": ENGINEERING_TWIN_TOLERANCE,
            "attempt_1_absolute_prefix_tolerance_retained_as_failed": PREFIX_FIDELITY_TOLERANCE,
            "attempt_2_prefix_one_step_rms_tolerance": PREFIX_ONE_STEP_RMS_TOLERANCE,
            "attempt_2_prefix_terminal_rms_tolerance": PREFIX_TERMINAL_RMS_TOLERANCE,
            "attempt_2_maximum_one_step_error_ratio_vs_direct": PREFIX_RELATIVE_ONE_STEP_RATIO,
        },
        "summaries": summaries,
        "selected_design": selected_design,
        "process_isolation_required": selected_design is None,
        "selection_reason": selection_reason,
        "prior_engineering_attempts": [prior_attempt] if prior_attempt else [],
        "attempts": attempts,
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.attempts_output), report)
    lines = [
        "# Epoch 10B branch constructor audit",
        "",
        "Old Epoch 10 mechanics demonstrations were used only for outcome-blind engineering. No checkpoint action or closed-loop outcome was queried.",
        "",
        f"Selected design: `{selected_design or 'NONE_IN_PROCESS'}`",
        "",
        selection_reason,
        "",
        "| Design | Valid | Twin state L2 max | Twin task-error max | One-step replay L2 max | Prefix terminal L2 max | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['all_valid']} | {summary['maximum_twin_final_state_l2']:.12g} | "
            f"{summary['maximum_twin_task_error_abs']:.12g} | {summary['maximum_one_step_replay_l2']:.12g} | "
            f"{summary['maximum_prefix_terminal_l2']:.12g} | {summary['pass']} |"
        )
    lines.extend(
        [
            "",
            "Each branch constructed, reset, restored, executed, closed, and destroyed its own environment/controller. The machine-readable attempt log records RNG identity, observation and state hashes, controller snapshots, queue audits, termination initialization, cleanup, and resource samples.",
        ]
    )
    _write_text(Path(args.audit_output), "\n".join(lines))
    return report


def freeze_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    branch_audit = json.loads(Path(args.attempts_output).read_text(encoding="utf-8"))
    design = branch_audit.get("selected_design")
    if not design:
        raise RuntimeError("branch audit did not select an in-process constructor; process-isolation fallback must be implemented first")
    tasks = _load_task_records(Path(args.raw_root))
    states = _state_manifest(tasks, CERTIFICATION_DEMOS)
    if len(states) != 128:
        raise RuntimeError(f"certification manifest must contain 128 states, got {len(states)}")
    old = json.loads((REPO_ROOT / "reports/epoch10_icae_exact_state_preflight_attempt2.json").read_text(encoding="utf-8"))
    old_mechanics = {row["state_id"] for row in old["rows"] if row["partition"] == "mechanics_calibration"}
    old_mechanics_clusters = {
        f"{row['suite']}|task_{row['task_id']}|{row['demo_name']}"
        for row in old["rows"]
        if row["partition"] == "mechanics_calibration"
    }
    new_mechanics_clusters = {row["demo_cluster"] for row in states}
    if old_mechanics_clusters.intersection(new_mechanics_clusters):
        raise RuntimeError("new mechanics manifest overlaps old mechanics rows")
    report = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_CERTIFICATION",
        "source_commit": SOURCE_COMMIT,
        "constructor": {
            "selected_design": design,
            "branch_audit_path": str(Path(args.attempts_output)),
            "branch_audit_sha256": _sha256_file(Path(args.attempts_output)),
            "branch_constructor_path": str(Path(__file__).resolve()),
            "branch_constructor_sha256": _sha256_file(Path(__file__)),
            "fresh_environment_controller_per_branch": True,
            "process_isolation": False,
        },
        "leakage_boundaries": {
            "checkpoint_actions_queried": 0,
            "checkpoint_comparative_outcomes_opened": False,
            "closed_loop_success_labels_opened": False,
            "mechanics_demo_ids": list(CERTIFICATION_DEMOS),
            "development_reserved_demo_ids": list(DEVELOPMENT_RESERVED_DEMOS),
            "heldout_reserved_demo_ids": list(HELDOUT_RESERVED_DEMOS),
            "whole_demo_sets_pairwise_disjoint": True,
        },
        "tasks": tasks,
        "states": states,
        "state_count": len(states),
        "whole_demo_cluster_count": len({row["demo_cluster"] for row in states}),
        "suite_counts": {suite: sum(row["suite"] == suite for row in states) for suite in sorted({row["suite"] for row in states})},
        "phase_counts": {phase: sum(row["phase"] == phase for row in states) for phase in sorted({row["phase"] for row in states})},
        "horizon_candidates": list(HORIZONS),
        "controls": list(CONTROL_NAMES),
        "symmetric_action_deltas": {"small": SMALL_DELTA, "medium": MEDIUM_DELTA},
        "branch_order": {
            "primary": "SHA-256-deterministic random order per state",
            "reverse_order_duplicate_states": sum(bool(row["reverse_order_duplicate"]) for row in states),
        },
        "terminal_semantics": {
            "identical_nominal_same_early_terminal": "valid_absorbing_pair",
            "candidate_only_early_terminal": "scientific_consequence",
            "infrastructure_incomplete": "recorded_separately_never_scored_as_success",
            "retry_rule": "one identical retry is allowed only for an infrastructure-incomplete branch with no materialized final score; the incomplete attempt remains recorded",
        },
        "endpoint_selection_rule": {
            "candidates_in_order": ["native_goal_error_auc", "bounded_expert_recovery_cost"],
            "native_goal_error_auc": "mean native task goal-error over the first H post-splice control steps",
            "bounded_expert_recovery_cost": "mean flattened-physics RMS distance to the registered expert future states over the first H post-splice control steps",
            "eligibility": "harmful-minus-nominal grouped 95% interval lower bound > 0; response above twin-noise floor on >=25% of states and every suite; >=32 responsive states",
            "tie_break": "prefer native goal-error if eligible; otherwise use bounded recovery cost",
        },
        "horizon_selection_rule": "Among eligible horizons, maximize twin fidelity first, then endpoint dynamic range, then duplicate repeatability, then choose the lowest simulator-step horizon.",
        "twin_gate": {
            "pre_action_state_and_observation_identity_rate": 1.0,
            "minimum_pairs_within_1e_8": 63,
            "pair_count": 64,
            "maximum_state_l2": TWIN_STATE_MATERIALITY,
            "minimum_equivalent_pairs_per_suite": 15,
            "minimum_duplicate_score_spearman": 0.95,
            "minimum_duplicate_score_icc": 0.95,
        },
        "responsiveness_rule": "A state is responsive when harmful-minus-nominal primary score exceeds max(99th percentile absolute nominal-A/nominal-B score difference, 1e-10).",
        "resource_contract": {
            "execution": "serial",
            "maximum_simulator_environments_resident": 1,
            "soft_host_ram_warning_percent": 80,
            "hard_host_ram_stop_percent": 90,
            "checkpoint_model_resident": False,
        },
        "statistics": {
            "bootstrap_unit": "whole_demonstration_cluster",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": 20260721,
        },
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.preregistration_output), report)
    return report


def _branch_orders(state_id: str) -> list[str]:
    return sorted(CONTROL_NAMES, key=lambda name: hashlib.sha256(f"epoch10b-order|{state_id}|{name}".encode("utf-8")).hexdigest())


def _load_completed_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    completed = {}
    invalid_lines = []
    if not path.is_file():
        return completed, invalid_lines
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
            completed[str(row["branch_key"])] = row
        except Exception as exc:
            invalid_lines.append({"line_number": line_number, "error": f"{type(exc).__name__}: {exc}", "raw_sha256": hashlib.sha256(line.encode()).hexdigest()})
    return completed, invalid_lines


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _primary_score(row: Mapping[str, Any], endpoint: str, horizon: int) -> float:
    key = "native_goal_error_auc_by_horizon" if endpoint == "native_goal_error_auc" else "bounded_recovery_cost_by_horizon"
    return float(row[key][str(horizon)])


def _index_manifest_states(states: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index state records without losing a registered reverse-panel flag.

    The original Epoch 10B manifest serialized two phase slots to one state ID.
    A plain dict comprehension retained only the later slot, whose reverse flag
    was false, and therefore hid an already executed reverse panel.  Preserve
    the last record's ordinary metadata while joining the registration flag by
    logical OR.  This changes no state, score, threshold, or raw row.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for state in states:
        state_id = str(state["state_id"])
        registered_reverse = bool(state.get("reverse_order_duplicate"))
        if state_id in indexed:
            registered_reverse = bool(indexed[state_id].get("reverse_order_duplicate")) or registered_reverse
        indexed[state_id] = {**state, "reverse_order_duplicate": registered_reverse}
    return indexed


def adjudicate_certification(branches: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any]) -> dict[str, Any]:
    by_key = {str(row["branch_key"]): row for row in branches}
    state_by_id = _index_manifest_states(prereg["states"])
    audits = []
    for horizon in HORIZONS:
        state_records = []
        for state_id, state in state_by_id.items():
            primary = {
                control: by_key.get(f"certification|primary|{state_id}|{control}")
                for control in CONTROL_NAMES
            }
            if not all(primary.values()):
                state_records.append({"state_id": state_id, "complete": False})
                continue
            nominal_a = primary["nominal_a"]
            nominal_b = primary["nominal_b"]
            sham = primary["sham"]
            twin_l2 = max(_state_l2(nominal_a, nominal_b, horizon), _state_l2(nominal_a, sham, horizon))
            twin_task = max(_task_error_abs(nominal_a, nominal_b, horizon), _task_error_abs(nominal_a, sham, horizon))
            record = {
                "state_id": state_id,
                "suite": state["suite"],
                "demo_cluster": state["demo_cluster"],
                "complete": all(bool(row.get("valid")) for row in primary.values()),
                "pre_action_identity": len({row.get("pre_action_state_sha256") for row in primary.values()}) == 1
                and len({row.get("pre_action_observation_sha256") for row in primary.values()}) == 1,
                "twin_state_l2": twin_l2,
                "twin_task_error_abs": twin_task,
                "scores": {},
                "first_step_response": {},
            }
            for endpoint in ("native_goal_error_auc", "bounded_expert_recovery_cost"):
                record["scores"][endpoint] = {name: _primary_score(row, endpoint, horizon) for name, row in primary.items()}
            nominal_one = np.asarray(nominal_a["first_step_state"], dtype=np.float64)
            for name in ("small_plus", "small_minus", "medium_plus", "medium_minus"):
                candidate_one = np.asarray(primary[name]["first_step_state"], dtype=np.float64)
                record["first_step_response"][name] = float(np.linalg.norm(candidate_one - nominal_one))
            if state["reverse_order_duplicate"]:
                reverse = {
                    control: by_key.get(f"certification|reverse|{state_id}|{control}")
                    for control in CONTROL_NAMES
                }
                if all(reverse.values()) and all(bool(row.get("valid")) for row in reverse.values()):
                    record["reverse_order_max_state_l2"] = max(
                        _state_l2(primary[name], reverse[name], horizon) for name in CONTROL_NAMES
                    )
                else:
                    record["reverse_order_max_state_l2"] = math.inf
            else:
                record["reverse_order_max_state_l2"] = None
            state_records.append(record)
        complete = [row for row in state_records if row.get("complete")]
        subset_ids = [row["state_id"] for row in prereg["states"] if int(row["phase_index"]) % 2 == 0]
        twins = [row for row in complete if row["state_id"] in subset_ids]
        endpoint_audits = {}
        for endpoint in ("native_goal_error_auc", "bounded_expert_recovery_cost"):
            twin_differences = [abs(row["scores"][endpoint]["nominal_a"] - row["scores"][endpoint]["nominal_b"]) for row in twins]
            noise_floor = max(float(np.quantile(twin_differences, 0.99)) if twin_differences else math.inf, 1e-10)
            harmful_by_cluster: dict[str, list[float]] = defaultdict(list)
            medium_small_by_cluster: dict[str, list[float]] = defaultdict(list)
            responsive = []
            for row in complete:
                scores = row["scores"][endpoint]
                harmful = scores["harmful_phase_matched"] - scores["nominal_a"]
                harmful_by_cluster[row["demo_cluster"]].append(harmful)
                small = 0.5 * (scores["small_plus"] + scores["small_minus"]) - scores["nominal_a"]
                medium = 0.5 * (scores["medium_plus"] + scores["medium_minus"]) - scores["nominal_a"]
                medium_small_by_cluster[row["demo_cluster"]].append(medium - small)
                if harmful > noise_floor:
                    responsive.append(row["state_id"])
            harmful_boot = _cluster_bootstrap(harmful_by_cluster, 20260721 + horizon)
            stress_boot = _cluster_bootstrap(medium_small_by_cluster, 20260731 + horizon)
            suites = {
                suite: sum(state_by_id[state_id]["suite"] == suite for state_id in responsive)
                for suite in sorted({row["suite"] for row in prereg["states"]})
            }
            nominal_a_scores = [row["scores"][endpoint]["nominal_a"] for row in twins]
            nominal_b_scores = [row["scores"][endpoint]["nominal_b"] for row in twins]
            eligible = bool(
                harmful_boot["bootstrap_95_interval"][0] > 0.0
                and len(responsive) >= 32
                and len(responsive) >= math.ceil(0.25 * len(complete))
                and all(count > 0 for count in suites.values())
            )
            endpoint_audits[endpoint] = {
                "twin_noise_floor": noise_floor,
                "harmful_minus_nominal_grouped": harmful_boot,
                "medium_minus_small_descriptive_stress": stress_boot,
                "responsive_state_count": len(responsive),
                "responsive_state_ids": responsive,
                "responsive_counts_by_suite": suites,
                "nominal_duplicate_score_spearman": _spearman(nominal_a_scores, nominal_b_scores),
                "nominal_duplicate_score_icc": _icc_agreement(nominal_a_scores, nominal_b_scores),
                "eligible": eligible,
            }
        endpoint = next((name for name in ("native_goal_error_auc", "bounded_expert_recovery_cost") if endpoint_audits[name]["eligible"]), None)
        selected_endpoint_audit = endpoint_audits.get(endpoint or "native_goal_error_auc")
        pairs_within = sum(
            row["twin_state_l2"] <= TWIN_SCORE_TOLERANCE and row["twin_task_error_abs"] <= TWIN_SCORE_TOLERANCE
            for row in twins
        )
        suite_pairs = {
            suite: sum(
                row["suite"] == suite
                and row["twin_state_l2"] <= TWIN_SCORE_TOLERANCE
                and row["twin_task_error_abs"] <= TWIN_SCORE_TOLERANCE
                for row in twins
            )
            for suite in sorted({row["suite"] for row in prereg["states"]})
        }
        reverse_order_rows = [row for row in complete if row["reverse_order_max_state_l2"] is not None]
        reverse_order_max = max((row["reverse_order_max_state_l2"] for row in reverse_order_rows), default=math.inf)
        twin_pass = bool(
            len(twins) == 64
            and all(row["pre_action_identity"] for row in twins)
            and pairs_within >= 63
            and max((row["twin_state_l2"] for row in twins), default=math.inf) <= TWIN_STATE_MATERIALITY
            and all(count >= 15 for count in suite_pairs.values())
            and len(reverse_order_rows) == 16
            and reverse_order_max <= TWIN_SCORE_TOLERANCE
            and selected_endpoint_audit["nominal_duplicate_score_spearman"] >= 0.95
            and selected_endpoint_audit["nominal_duplicate_score_icc"] >= 0.95
        )
        audits.append(
            {
                "horizon": horizon,
                "complete_state_count": len(complete),
                "twin_subset_count": len(twins),
                "pre_action_identity_rate": float(np.mean([row["pre_action_identity"] for row in twins])) if twins else 0.0,
                "pairs_within_1e_8": int(pairs_within),
                "maximum_twin_state_l2": max((row["twin_state_l2"] for row in twins), default=math.inf),
                "suite_pairs_within_1e_8": suite_pairs,
                "reverse_order_duplicate_count": len(reverse_order_rows),
                "reverse_order_max_state_l2": reverse_order_max,
                "twin_gate_pass": twin_pass,
                "endpoint_audits": endpoint_audits,
                "selected_endpoint": endpoint,
                "responsiveness_gate_pass": bool(endpoint),
                "pass": bool(twin_pass and endpoint),
            }
        )
    passing = [row for row in audits if row["pass"]]
    selected = min(passing, key=lambda row: row["horizon"]) if passing else None
    return {
        "horizon_audits": audits,
        "selected_horizon": selected["horizon"] if selected else None,
        "selected_endpoint": selected["selected_endpoint"] if selected else None,
        "responsive_state_ids": selected["endpoint_audits"][selected["selected_endpoint"]]["responsive_state_ids"] if selected else [],
        "certified": selected is not None,
    }


def run_certification(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration_output)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg["status"] != "FROZEN_BEFORE_CERTIFICATION":
        raise RuntimeError("assay preregistration is not frozen")
    tasks = {(row["suite"], int(row["task_id"])): row for row in prereg["tasks"]}
    design = str(prereg["constructor"]["selected_design"])
    run_dir = Path(args.run_dir)
    raw_path = run_dir / "branches.jsonl"
    pending_path = run_dir / "active_branch.json"
    completed, invalid_lines = _load_completed_jsonl(raw_path)
    infrastructure_attempts = []
    if pending_path.is_file():
        pending_bytes = pending_path.read_bytes()
        try:
            pending_record: Any = json.loads(pending_bytes.decode("utf-8"))
        except Exception as exc:
            pending_record = {
                "unparseable_pending_record": True,
                "bytes": len(pending_bytes),
                "sha256": hashlib.sha256(pending_bytes).hexdigest(),
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
        interrupted = {
            "classification": "INCOMPLETE_BRANCH_WITHOUT_MATERIALIZED_SCORE",
            "pending_record": pending_record,
            "retry_authorized_by_frozen_rule": True,
        }
        infrastructure_attempts.append(interrupted)
        attempts_log = run_dir / "infrastructure_attempts.jsonl"
        prior_attempt_hashes = set()
        if attempts_log.is_file():
            for line in attempts_log.read_text(encoding="utf-8").splitlines():
                try:
                    prior_attempt_hashes.add(json.loads(line)["attempt_sha256"])
                except Exception:
                    pass
        interrupted["attempt_sha256"] = _canonical_sha256(interrupted)
        if interrupted["attempt_sha256"] not in prior_attempt_hashes:
            _append_jsonl(attempts_log, interrupted)
    peak_host = 0.0
    peak_wsl = 0
    peak_swap = 0
    peak_vram = 0.0
    started = time.monotonic()
    new_branch_count = 0
    batch_limit_reached = False
    expected_branch_count = sum(
        len(CONTROL_NAMES) * (2 if bool(row["reverse_order_duplicate"]) else 1)
        for row in prereg["states"]
    )
    for state_index, state_row in enumerate(prereg["states"]):
        state_id = str(state_row["state_id"])
        orders = [("primary", _branch_orders(state_id))]
        if state_row["reverse_order_duplicate"]:
            orders.append(("reverse", list(reversed(_branch_orders(state_id)))))
        required_keys = [
            f"certification|{pass_name}|{state_id}|{control_name}"
            for pass_name, order in orders
            for control_name in order
        ]
        if all(key in completed for key in required_keys):
            continue
        task = tasks[(state_row["suite"], int(state_row["task_id"]))]
        with h5py.File(task["hdf5_path"], "r") as handle:
            demo = handle["data"][state_row["demo_name"]]
            states = np.asarray(demo["states"], dtype=np.float64)
            actions = np.asarray(demo["actions"], dtype=np.float64)
        if _array_sha256(states[int(state_row["frame"])]) != state_row["state_sha256"]:
            raise RuntimeError(f"frozen state hash mismatch for {state_id}")
        target = _target_for_demo(task, states, actions, int(args.camera_size))
        controls = control_actions(actions[int(state_row["frame"])], state_row["phase"], state_id)
        for pass_name, order in orders:
            for control_name in order:
                branch_key = f"certification|{pass_name}|{state_id}|{control_name}"
                if branch_key in completed:
                    continue
                pending = {
                    "branch_key": branch_key,
                    "started_at_unix": time.time(),
                    "state_index": state_index,
                    "control": control_name,
                    "pass": pass_name,
                    "retry_rule": prereg["terminal_semantics"]["retry_rule"],
                }
                _write_json(pending_path, pending)
                row = execute_fresh_branch(
                    task=task,
                    states=states,
                    actions=actions,
                    target=target,
                    frame=int(state_row["frame"]),
                    first_action=controls[control_name],
                    design=design,
                    branch_key=branch_key,
                    camera_size=int(args.camera_size),
                    horizon=max(HORIZONS),
                    registered_seed=int(state_row["registered_seed"]),
                )
                row.update(
                    {
                        "state_id": state_id,
                        "suite": state_row["suite"],
                        "task_id": int(state_row["task_id"]),
                        "demo_name": state_row["demo_name"],
                        "demo_cluster": state_row["demo_cluster"],
                        "phase": state_row["phase"],
                        "control": control_name,
                        "execution_pass": pass_name,
                        "nominal_first_action": controls["nominal_a"],
                        "delivered_action_delta_l2": float(np.linalg.norm(controls[control_name] - controls["nominal_a"])),
                        "delivered_action_delta_linf": float(np.max(np.abs(controls[control_name] - controls["nominal_a"]))),
                    }
                )
                _append_jsonl(raw_path, row)
                completed[branch_key] = row
                new_branch_count += 1
                if pending_path.exists():
                    pending_path.unlink()
                for sample in (row.get("resource_before", {}), row.get("resource_after", {})):
                    peak_host = max(peak_host, float(sample.get("host_ram_percent") or 0.0))
                    total = int(sample.get("wsl_mem_total_bytes") or 0)
                    available = int(sample.get("wsl_mem_available_bytes") or 0)
                    peak_wsl = max(peak_wsl, max(0, total - available))
                    peak_swap = max(peak_swap, int(sample.get("wsl_swap_used_bytes") or 0))
                    peak_vram = max(peak_vram, float(sample.get("gpu_vram_used_mib") or 0.0))
                if peak_host >= 90.0:
                    raise RuntimeError(f"hard host RAM stop reached at {peak_host:.3f}%")
                if len(completed) % 10 == 0:
                    print(json.dumps({"completed_branches": len(completed), "branch_key": branch_key, "valid": row["valid"], "host_ram_peak": peak_host}), flush=True)
                if int(args.max_new_branches) > 0 and new_branch_count >= int(args.max_new_branches):
                    batch_limit_reached = True
                    break
            if batch_limit_reached:
                break
        if batch_limit_reached:
            break
    if batch_limit_reached and len(completed) < expected_branch_count:
        batch_state = {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "status": "BATCH_COMPLETE_RESUMABLE",
            "completed_branch_count": len(completed),
            "expected_branch_count": expected_branch_count,
            "new_branch_count": new_branch_count,
            "raw_branch_log": str(raw_path),
            "raw_branch_log_sha256": _sha256_file(raw_path),
            "active_branch_present": pending_path.is_file(),
        }
        _write_json(run_dir / "batch_state.json", batch_state)
        return batch_state
    branches = list(completed.values())
    adjudication = adjudicate_certification(branches, prereg)
    report = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "ASSAY_CERTIFIED" if adjudication["certified"] else "ASSAY_INVALID",
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": _sha256_file(prereg_path),
        "checkpoint_actions_queried": 0,
        "prospective_checkpoint_outcomes_read": False,
        "closed_loop_success_labels_opened": False,
        "branch_count": len(branches),
        "invalid_branch_count": sum(not bool(row.get("valid")) for row in branches),
        "invalid_jsonl_lines": invalid_lines,
        "infrastructure_attempts": infrastructure_attempts,
        "resource_telemetry": {
            "peak_host_ram_percent": peak_host,
            "peak_wsl_used_bytes": peak_wsl,
            "peak_wsl_swap_used_bytes": peak_swap,
            "peak_gpu_vram_used_mib": peak_vram,
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "process_exit": 0,
        },
        **adjudication,
        "raw_branch_log": str(raw_path),
        "raw_branch_log_sha256": _sha256_file(raw_path),
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.certification_output), report)
    return report


def freeze_certified_assay(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.preregistration_output)
    certification_path = Path(args.certification_output)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    if not certification.get("certified"):
        adjudication = {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "status": "TERMINAL",
            "terminal_state": "EPOCH10B_ICAE_ASSAY_INVALID_ROUTE_CLOSED",
            "reason": "No allowed constructor/horizon/endpoint combination passed the prospectively frozen twin and responsiveness gates.",
            "preregistration_sha256": _sha256_file(prereg_path),
            "mechanics_certification_sha256": _sha256_file(certification_path),
            "checkpoint_actions_queried": 0,
            "closed_loop_outcomes_opened": False,
        }
        adjudication["canonical_payload_sha256"] = _canonical_sha256(adjudication)
        _write_json(Path(args.adjudication_output), adjudication)
        return adjudication
    selected_horizon = int(certification["selected_horizon"])
    selected_endpoint = str(certification["selected_endpoint"])
    freeze = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "ASSAY_FROZEN_AFTER_CERTIFICATION",
        "constructor": prereg["constructor"],
        "action_unit": "one deployment-equivalent 7-D action followed by the registered expert continuation",
        "selected_horizon": selected_horizon,
        "continuation": "registered expert actions from frame t+1 through t+H",
        "primary_endpoint": selected_endpoint,
        "responsive_state_selector": prereg["responsiveness_rule"],
        "responsive_state_ids": certification["responsive_state_ids"],
        "checkpoint_split": {
            "development_reserved_demo_ids": list(DEVELOPMENT_RESERVED_DEMOS),
            "heldout_reserved_demo_ids": list(HELDOUT_RESERVED_DEMOS),
            "split_by_whole_seed_lineage": True,
        },
        "baselines": [
            "raw_mse",
            "raw_mae",
            "action_dimension_normalized_mse",
            "arm_gripper_weighted_mse",
            "phase_state_only_criticality",
            "faithful_ci_mse_when_compatible",
            "unpaired_icae",
            "state_shuffled_icae",
            "held_noop_response",
            "response_magnitude_only",
        ],
        "statistics": prereg["statistics"],
        "terminal_semantics": prereg["terminal_semantics"],
        "preregistration_sha256": _sha256_file(prereg_path),
        "mechanics_certification_sha256": _sha256_file(certification_path),
        "branch_constructor_sha256": _sha256_file(Path(__file__)),
        "post_freeze_scientific_change_allowed": False,
        "checkpoint_actions_queried_at_freeze": 0,
        "closed_loop_outcomes_opened_at_freeze": False,
    }
    freeze["canonical_payload_sha256"] = _canonical_sha256(freeze)
    _write_json(Path(args.freeze_output), freeze)
    adjudication = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "CONTINUE_TO_CHECKPOINT_LINEAGE_GATE",
        "terminal_state": None,
        "selected_constructor": prereg["constructor"]["selected_design"],
        "selected_horizon": selected_horizon,
        "selected_endpoint": selected_endpoint,
        "responsive_state_count": len(certification["responsive_state_ids"]),
        "assay_freeze_sha256": _sha256_file(Path(args.freeze_output)),
    }
    adjudication["canonical_payload_sha256"] = _canonical_sha256(adjudication)
    _write_json(Path(args.adjudication_output), adjudication)
    return adjudication


def integrate_host_monitor(args: argparse.Namespace) -> dict[str, Any]:
    certification_path = Path(args.certification_output)
    monitor_path = Path(args.host_monitor)
    report = json.loads(certification_path.read_text(encoding="utf-8"))
    monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
    report["resource_telemetry"]["peak_host_ram_percent"] = float(monitor["peak_host_ram_percent"])
    report["resource_telemetry"]["host_soft_warning_crossed"] = bool(monitor["soft_warning_crossed"])
    report["resource_telemetry"]["host_hard_stop_crossed"] = bool(monitor["hard_stop_crossed"])
    report["resource_telemetry"]["host_monitor_path"] = str(monitor_path)
    report["resource_telemetry"]["host_monitor_sha256"] = _sha256_file(monitor_path)
    report.pop("canonical_payload_sha256", None)
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(certification_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("engineering", "preregister", "certify", "integrate-monitor", "freeze"))
    parser.add_argument("--raw-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--attempts-output", default="reports/epoch10b_branch_constructor_attempts.json")
    parser.add_argument("--audit-output", default="reports/epoch10b_branch_constructor_audit.md")
    parser.add_argument("--preregistration-output", default="reports/epoch10b_assay_preregistration.json")
    parser.add_argument("--certification-output", default="reports/epoch10b_mechanics_certification.json")
    parser.add_argument("--adjudication-output", default="reports/epoch10b_assay_adjudication.json")
    parser.add_argument("--freeze-output", default="reports/epoch10b_assay_freeze.json")
    parser.add_argument("--run-dir", default="runs/epoch10b_mechanics_certification")
    parser.add_argument("--host-monitor", default="runs/epoch10b_mechanics_certification/host_monitor.json")
    parser.add_argument("--max-new-branches", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "engineering":
        result = run_engineering(args)
        code = 0 if result.get("selected_design") else 3
    elif args.mode == "preregister":
        result = freeze_preregistration(args)
        code = 0
    elif args.mode == "certify":
        result = run_certification(args)
        code = 0 if result.get("certified") or result.get("status") == "BATCH_COMPLETE_RESUMABLE" else 4
    elif args.mode == "integrate-monitor":
        result = integrate_host_monitor(args)
        code = 0
    else:
        result = freeze_certified_assay(args)
        code = 0
    print(json.dumps({key: result.get(key) for key in ("status", "selected_design", "selected_horizon", "selected_endpoint", "terminal_state")}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
