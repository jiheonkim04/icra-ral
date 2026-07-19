"""Run the frozen Epoch 6 schedule-dependence closed-loop problem gate.

The two schedules execute in fresh processes and share exactly one pinned
X-VLA model per process.  Simulator workers submit policy requests to one FIFO
queue, making the model-process service ordinal the observable global noise
position.  Every served request and worker state is durably checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import run_epoch6_schedule_invariance_stage0 as stage0
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_epoch6_schedule_invariance_stage0 as stage0


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reports" / "epoch6_schedule_invariant_evaluation" / "problem_verification_protocol.json"
EXECUTION_MANIFEST_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch6_schedule_invariant_evaluation"
    / "closed_loop_execution_manifest.json"
)
STAGE0_RESULT_PATH = REPO_ROOT / "reports" / "epoch6_schedule_invariant_evaluation" / "stage0_result.json"
RESOURCE_AMENDMENT_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch6_schedule_invariant_evaluation"
    / "resource_governance_amendment_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = "E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946"
EXPECTED_EXECUTION_MANIFEST_SHA256 = "93DEA8F3A9EAAE864A0A691B11312492E9E2D585606859597228E6F23D0FB242"
EXPECTED_STAGE0_RESULT_SHA256 = "2CE2222184C9AE2D0EB49DCDD4A3D3BAED1D76F1BC1B79EFC3AE0BDA0D2BD617"
EXPECTED_RESOURCE_AMENDMENT_SHA256 = "E98AED352765CEDA55607A2182ECE7B6E44B499DCC0253DA66313C72A6F3C601"
EXPECTED_CLOSED_LOOP_MONITOR_SHA256 = "B4383593900DCAD45D76C2BF3E7C8172E790DD7021249F4D5D81A59A15A0A9F5"

ROOT_SEED = 620260719
ENV_SEED = 7
TASK_ID = 4
SETTLE_STEPS = 10
RAW_CHUNK_SHAPE = (30, 20)
PROCESSED_CHUNK_SHAPE = (30, 7)
SCHEDULES = ("single_lane_canonical_serial", "four_shards_predeclared_reversed_launch_offsets")
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SERIAL_ASSIGNMENTS = {0: list(range(20))}
SHARDED_ASSIGNMENTS = {
    0: [0, 4, 8, 12, 16],
    1: [1, 5, 9, 13, 17],
    2: [2, 6, 10, 14, 18],
    3: [3, 7, 11, 15, 19],
}
SERIAL_OFFSETS = {0: 0.0}
SHARDED_OFFSETS = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
REQUIRED_SHARDED_FIRST_ARRIVAL = [3, 2, 1, 0]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def identity_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for suite in SUITES:
        for initial_state_index in range(5):
            rows.append(
                {
                    "global_index": global_index,
                    "suite": suite,
                    "task_id": TASK_ID,
                    "initial_state_index": initial_state_index,
                }
            )
            global_index += 1
    return rows


IDENTITIES = identity_manifest()
IDENTITY_BY_INDEX = {int(row["global_index"]): row for row in IDENTITIES}


def validate_manifest() -> dict[str, Any]:
    if stage0.sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("scientific protocol hash mismatch")
    if stage0.sha256_file(EXECUTION_MANIFEST_PATH) != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise RuntimeError("closed-loop execution-manifest hash mismatch")
    if stage0.sha256_file(STAGE0_RESULT_PATH) != EXPECTED_STAGE0_RESULT_SHA256:
        raise RuntimeError("Stage 0 result-binding hash mismatch")
    if stage0.sha256_file(RESOURCE_AMENDMENT_PATH) != EXPECTED_RESOURCE_AMENDMENT_SHA256:
        raise RuntimeError("resource-amendment hash mismatch")
    manifest = json.loads(EXECUTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["scientific_outcomes_exposed_at_freeze"]:
        raise RuntimeError("closed-loop manifest was not frozen outcome-blind")
    if manifest["identity_manifest"] != IDENTITIES:
        raise RuntimeError("closed-loop identity manifest mismatch")
    stage0_result = json.loads(STAGE0_RESULT_PATH.read_text(encoding="utf-8"))
    if stage0_result["final_decision"] != "ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO":
        raise RuntimeError("Stage 0 did not authorize closed-loop execution")
    return manifest


def exact_array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def policy_input_hash(
    agentview: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, language: str
) -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("agentview", np.asarray(agentview, dtype=np.uint8)),
        ("wrist", np.asarray(wrist, dtype=np.uint8)),
        ("proprio", np.asarray(proprio, dtype="<f4")),
    ):
        array = np.ascontiguousarray(value)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    digest.update(b"language\0" + language.encode("utf-8"))
    return digest.hexdigest().upper()


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def task_assets(suite_name: str) -> tuple[Any, Path, Path, np.ndarray, str]:
    benchmark, OffScreenRenderEnv, _executed = stage0.import_pinned_libero()
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(TASK_ID)
    asset_root = stage0.LIBERO_ROOT / "libero" / "libero"
    bddl = (asset_root / "bddl_files" / task.problem_folder / task.bddl_file).resolve(strict=True)
    init_path = (asset_root / "init_files" / task.problem_folder / task.init_states_file).resolve(strict=True)
    import torch

    initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
    initial_states = np.asarray(initial_states, dtype=np.float64)
    if initial_states.shape[0] < 5:
        raise RuntimeError(f"{suite_name}/task_4 has fewer than five initial states")
    return OffScreenRenderEnv, bddl, init_path, initial_states, str(task.language)


def simulator_state(env: Any) -> np.ndarray:
    return np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64).copy()


def regenerate_observation(env: Any) -> dict[str, Any]:
    env.env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def controller_proprio(env: Any) -> np.ndarray:
    controller = env.env.robots[0].controller
    position = np.asarray(controller.ee_pos, dtype=np.float32)
    rotation = np.asarray(controller.ee_ori_mat, dtype=np.float32)
    first_arm = np.concatenate(
        [position, rotation[:3, 0], rotation[:3, 1], np.zeros(1, dtype=np.float32)]
    ).astype(np.float32)
    return np.concatenate([first_arm, np.zeros_like(first_arm)]).astype(np.float32)


def make_policy_request_arrays(
    env: Any, observation: Mapping[str, Any], last_raw_chunk: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agentview = np.flip(np.asarray(observation["agentview_image"], dtype=np.uint8), axis=(0, 1)).copy()
    wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
    if last_raw_chunk is None:
        proprio = controller_proprio(env)
    else:
        proprio = np.zeros(20, dtype=np.float32)
        proprio[:10] = np.asarray(last_raw_chunk[-1, :10], dtype=np.float32)
    if agentview.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected policy image shapes: {agentview.shape}, {wrist.shape}")
    return agentview, wrist, proprio


def checkpoint_paths(schedule_dir: Path, identity_index: int) -> tuple[Path, Path]:
    root = schedule_dir / "checkpoints"
    return root / f"identity_{identity_index:03d}.json", root / f"identity_{identity_index:03d}.npz"


def episode_paths(schedule_dir: Path, identity_index: int) -> tuple[Path, Path]:
    root = schedule_dir / "episodes"
    return root / f"identity_{identity_index:03d}.json", root / f"identity_{identity_index:03d}.npz"


def write_worker_checkpoint(
    schedule_dir: Path,
    context: Mapping[str, Any],
    status: str,
) -> None:
    identity_index = int(context["identity"]["global_index"])
    metadata_path, arrays_path = checkpoint_paths(schedule_dir, identity_index)
    executed = np.asarray(context["executed_actions"], dtype=np.float32).reshape(-1, 7)
    last_raw = (
        np.asarray(context["last_raw_chunk"], dtype=np.float32)
        if context["last_raw_chunk"] is not None
        else np.empty((0, 20), dtype=np.float32)
    )
    write_npz_atomic(
        arrays_path,
        simulator_state=np.asarray(context["simulator_state"], dtype=np.float64),
        initial_state=np.asarray(context["initial_state"], dtype=np.float64),
        post_settle_state=np.asarray(context["post_settle_state"], dtype=np.float64),
        executed_actions=executed,
        last_raw_chunk=last_raw,
    )
    payload = {
        "schema_version": "epoch6.schedule_closed_loop.worker_checkpoint.v1",
        "updated_at": utc_now(),
        "status": status,
        "schedule": context["schedule"],
        "shard_id": int(context["shard_id"]),
        "identity": context["identity"],
        "language": context["language"],
        "horizon": int(context["horizon"]),
        "step_count": int(context["step_count"]),
        "policy_call_count": int(context["policy_call_count"]),
        "done": bool(context["done"]),
        "success": bool(context["success"]),
        "final_reward": float(context["final_reward"]),
        "initial_state_sha256": exact_array_hash(context["initial_state"]),
        "post_settle_state_sha256": exact_array_hash(context["post_settle_state"]),
        "current_simulator_state_sha256": exact_array_hash(context["simulator_state"]),
        "executed_action_sha256": stage0.hash_array(executed),
        "query_rows": list(context["query_rows"]),
        "arrays_path": str(arrays_path),
        "arrays_sha256": stage0.sha256_file(arrays_path),
    }
    stage0.write_json(metadata_path, payload)


def load_worker_checkpoint(schedule_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    identity_index = int(identity["global_index"])
    metadata_path, arrays_path = checkpoint_paths(schedule_dir, identity_index)
    if not metadata_path.is_file() and not arrays_path.is_file():
        return None
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise RuntimeError(f"incomplete worker checkpoint for identity {identity_index}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["identity"] != dict(identity):
        raise RuntimeError(f"worker checkpoint identity mismatch for {identity_index}")
    if metadata["arrays_sha256"] != stage0.sha256_file(arrays_path):
        raise RuntimeError(f"worker checkpoint array hash mismatch for {identity_index}")
    arrays = read_npz(arrays_path)
    if exact_array_hash(arrays["initial_state"]) != metadata["initial_state_sha256"]:
        raise RuntimeError(f"worker checkpoint initial-state hash mismatch for {identity_index}")
    if exact_array_hash(arrays["simulator_state"]) != metadata["current_simulator_state_sha256"]:
        raise RuntimeError(f"worker checkpoint simulator-state hash mismatch for {identity_index}")
    if stage0.hash_array(arrays["executed_actions"]) != metadata["executed_action_sha256"]:
        raise RuntimeError(f"worker checkpoint action hash mismatch for {identity_index}")
    return {"metadata": metadata, "arrays": arrays}


def finalize_episode(schedule_dir: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    identity_index = int(context["identity"]["global_index"])
    metadata_path, arrays_path = episode_paths(schedule_dir, identity_index)
    if metadata_path.exists() or arrays_path.exists():
        raise RuntimeError(f"refusing to overwrite final episode {identity_index}")
    executed = np.asarray(context["executed_actions"], dtype=np.float32).reshape(-1, 7)
    final_state = np.asarray(context["simulator_state"], dtype=np.float64)
    write_npz_atomic(
        arrays_path,
        initial_state=np.asarray(context["initial_state"], dtype=np.float64),
        post_settle_state=np.asarray(context["post_settle_state"], dtype=np.float64),
        final_state=final_state,
        executed_actions=executed,
    )
    timeout = bool(not context["success"] and int(context["step_count"]) >= int(context["horizon"]))
    result = {
        "schema_version": "epoch6.schedule_closed_loop.episode.v1",
        "completed_at": utc_now(),
        "schedule": context["schedule"],
        "shard_id": int(context["shard_id"]),
        "identity": context["identity"],
        "language": context["language"],
        "horizon": int(context["horizon"]),
        "steps": int(context["step_count"]),
        "policy_call_count": int(context["policy_call_count"]),
        "success": bool(context["success"]),
        "timeout": timeout,
        "done": bool(context["done"]),
        "final_reward": float(context["final_reward"]),
        "exception": None,
        "initial_state_sha256": exact_array_hash(context["initial_state"]),
        "post_settle_state_sha256": exact_array_hash(context["post_settle_state"]),
        "final_state_sha256": exact_array_hash(final_state),
        "executed_action_sha256": stage0.hash_array(executed),
        "executed_action_count": int(len(executed)),
        "query_rows": list(context["query_rows"]),
        "arrays_path": str(arrays_path),
        "arrays_sha256": stage0.sha256_file(arrays_path),
    }
    stage0.write_json(metadata_path, result)
    write_worker_checkpoint(schedule_dir, context, "completed")
    return result


def new_episode_context(
    schedule: str,
    shard_id: int,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    suite_name = str(identity["suite"])
    OffScreenRenderEnv, bddl, _init_path, initial_states, language = task_assets(suite_name)
    initial_state = np.asarray(initial_states[int(identity["initial_state_index"])], dtype=np.float64).copy()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(ENV_SEED)
    env.reset()
    observation = env.set_init_state(initial_state)
    settle_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        observation, _reward, _done, _info = env.step(settle_action)
    for robot in env.env.robots:
        robot.controller.use_delta = False
    post_settle_state = simulator_state(env)
    context = {
        "schedule": schedule,
        "shard_id": shard_id,
        "identity": dict(identity),
        "language": language,
        "horizon": HORIZONS[suite_name],
        "step_count": 0,
        "policy_call_count": 0,
        "done": False,
        "success": False,
        "final_reward": 0.0,
        "initial_state": initial_state,
        "post_settle_state": post_settle_state,
        "simulator_state": post_settle_state.copy(),
        "executed_actions": [],
        "last_raw_chunk": None,
        "query_rows": [],
    }
    return context, env, observation


def restored_episode_context(
    schedule: str,
    shard_id: int,
    identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    metadata = checkpoint["metadata"]
    arrays = checkpoint["arrays"]
    suite_name = str(identity["suite"])
    OffScreenRenderEnv, bddl, _init_path, _initial_states, language = task_assets(suite_name)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(ENV_SEED)
    env.reset()
    env.set_state(arrays["simulator_state"])
    observation = regenerate_observation(env)
    for robot in env.env.robots:
        robot.controller.use_delta = False
    last_raw = arrays["last_raw_chunk"] if arrays["last_raw_chunk"].size else None
    context = {
        "schedule": schedule,
        "shard_id": shard_id,
        "identity": dict(identity),
        "language": language,
        "horizon": int(metadata["horizon"]),
        "step_count": int(metadata["step_count"]),
        "policy_call_count": int(metadata["policy_call_count"]),
        "done": bool(metadata["done"]),
        "success": bool(metadata["success"]),
        "final_reward": float(metadata["final_reward"]),
        "initial_state": arrays["initial_state"],
        "post_settle_state": arrays["post_settle_state"],
        "simulator_state": arrays["simulator_state"],
        "executed_actions": arrays["executed_actions"].tolist(),
        "last_raw_chunk": last_raw,
        "query_rows": list(metadata["query_rows"]),
    }
    return context, env, observation


def simulator_worker(
    schedule: str,
    shard_id: int,
    identity_indices: list[int],
    launch_offset_seconds: float,
    run_dir_text: str,
    request_queue: Any,
    response_queue: Any,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    schedule_dir = Path(run_dir_text) / schedule
    env = None
    try:
        time.sleep(float(launch_offset_seconds))
        for identity_index in identity_indices:
            identity = IDENTITY_BY_INDEX[int(identity_index)]
            final_metadata, final_arrays = episode_paths(schedule_dir, identity_index)
            if final_metadata.is_file() and final_arrays.is_file():
                continue
            checkpoint = load_worker_checkpoint(schedule_dir, identity)
            if checkpoint is None:
                context, env, observation = new_episode_context(schedule, shard_id, identity)
                write_worker_checkpoint(schedule_dir, context, "active")
            else:
                if checkpoint["metadata"]["status"] == "completed":
                    raise RuntimeError(f"completed checkpoint exists without final episode {identity_index}")
                context, env, observation = restored_episode_context(
                    schedule, shard_id, identity, checkpoint
                )
            while not context["done"] and int(context["step_count"]) < int(context["horizon"]):
                call_index = int(context["policy_call_count"])
                agentview, wrist, proprio = make_policy_request_arrays(
                    env, observation, context["last_raw_chunk"]
                )
                pre_state = simulator_state(env)
                request_id = f"identity_{identity_index:03d}_query_{call_index:04d}"
                input_sha = policy_input_hash(agentview, wrist, proprio, str(context["language"]))
                sent_ns = time.monotonic_ns()
                request_queue.put(
                    {
                        "type": "request",
                        "schedule": schedule,
                        "shard_id": shard_id,
                        "identity": dict(identity),
                        "episode_policy_call_index": call_index,
                        "request_id": request_id,
                        "request_sent_monotonic_ns": sent_ns,
                        "pre_query_sim_state_sha256": exact_array_hash(pre_state),
                        "policy_input_sha256": input_sha,
                        "agentview": agentview,
                        "wrist": wrist,
                        "proprio": proprio,
                        "language": str(context["language"]),
                    }
                )
                response = response_queue.get(timeout=600)
                if response.get("request_id") != request_id:
                    raise RuntimeError(
                        f"response mismatch: expected {request_id}, got {response.get('request_id')}"
                    )
                raw_chunk = np.asarray(response["raw_chunk"], dtype=np.float32)
                processed_chunk = np.asarray(response["processed_chunk"], dtype=np.float32)
                if raw_chunk.shape != RAW_CHUNK_SHAPE or processed_chunk.shape != PROCESSED_CHUNK_SHAPE:
                    raise RuntimeError(f"unexpected response shapes for {request_id}")
                start_step = int(context["step_count"])
                executed_count = 0
                for action in processed_chunk:
                    if int(context["step_count"]) >= int(context["horizon"]) or context["done"]:
                        break
                    observation, reward, done, _info = env.step(np.asarray(action, dtype=np.float32))
                    context["executed_actions"].append(np.asarray(action, dtype=np.float32).tolist())
                    context["step_count"] = int(context["step_count"]) + 1
                    context["final_reward"] = float(reward)
                    context["done"] = bool(done)
                    context["success"] = bool(done)
                    executed_count += 1
                context["last_raw_chunk"] = raw_chunk.copy()
                context["policy_call_count"] = call_index + 1
                context["simulator_state"] = simulator_state(env)
                context["query_rows"].append(
                    {
                        "request_id": request_id,
                        "identity_global_index": identity_index,
                        "episode_policy_call_index": call_index,
                        "global_noise_position": int(response["global_noise_position"]),
                        "request_sent_monotonic_ns": sent_ns,
                        "request_arrived_monotonic_ns": int(response["request_arrived_monotonic_ns"]),
                        "service_started_monotonic_ns": int(response["service_started_monotonic_ns"]),
                        "service_completed_monotonic_ns": int(response["service_completed_monotonic_ns"]),
                        "pre_query_sim_state_sha256": exact_array_hash(pre_state),
                        "policy_input_sha256": input_sha,
                        "prepared_input_sha256": str(response["prepared_input_sha256"]),
                        "raw_chunk_sha256": stage0.hash_array(raw_chunk),
                        "processed_chunk_sha256": stage0.hash_array(processed_chunk),
                        "executed_step_start": start_step,
                        "executed_step_end_exclusive": int(context["step_count"]),
                        "executed_action_count": executed_count,
                        "done_after_chunk": bool(context["done"]),
                        "transaction_sha256": str(response["transaction_sha256"]),
                        "replayed_from_transaction": bool(response["replayed_from_transaction"]),
                    }
                )
                write_worker_checkpoint(schedule_dir, context, "active")
            result = finalize_episode(schedule_dir, context)
            request_queue.put(
                {
                    "type": "episode_complete",
                    "schedule": schedule,
                    "shard_id": shard_id,
                    "identity_global_index": identity_index,
                    "success": result["success"],
                    "timeout": result["timeout"],
                }
            )
            env.close()
            env = None
        request_queue.put({"type": "worker_done", "schedule": schedule, "shard_id": shard_id})
    except Exception as exc:
        failure = {
            "schema_version": "epoch6.schedule_closed_loop.worker_failure.v1",
            "failed_at": utc_now(),
            "schedule": schedule,
            "shard_id": shard_id,
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        failure_path = schedule_dir / f"worker_{shard_id}_failure_{int(time.time())}.json"
        stage0.write_json(failure_path, failure)
        request_queue.put(
            {
                "type": "worker_error",
                "schedule": schedule,
                "shard_id": shard_id,
                "failure_path": str(failure_path),
                "exception": failure["exception"],
            }
        )
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def transaction_paths(schedule_dir: Path, request_id: str) -> tuple[Path, Path]:
    root = schedule_dir / "server_transactions"
    return root / f"{request_id}.json", root / f"{request_id}.npz"


def rng_arrays_and_metadata(rng_state: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {
        "rng_numpy_keys": np.asarray(rng_state["numpy_keys"], dtype=np.uint32),
        "rng_torch_cpu": np.asarray(rng_state["torch_cpu"], dtype=np.uint8),
    }
    for index, value in enumerate(rng_state["torch_cuda"]):
        arrays[f"rng_torch_cuda_{index}"] = np.asarray(value, dtype=np.uint8)
    metadata = {
        "python_random_state": rng_state["python_random_state"],
        "numpy_algorithm": rng_state["numpy_algorithm"],
        "numpy_position": int(rng_state["numpy_position"]),
        "numpy_has_gauss": int(rng_state["numpy_has_gauss"]),
        "numpy_cached_gaussian": float(rng_state["numpy_cached_gaussian"]),
        "cuda_device_count": len(rng_state["torch_cuda"]),
    }
    return arrays, metadata


def rng_state_from_transaction(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rng = metadata["rng_state_after_service"]
    count = int(rng["cuda_device_count"])
    return {
        "python_random_state": rng["python_random_state"],
        "numpy_algorithm": rng["numpy_algorithm"],
        "numpy_keys": arrays["rng_numpy_keys"],
        "numpy_position": int(rng["numpy_position"]),
        "numpy_has_gauss": int(rng["numpy_has_gauss"]),
        "numpy_cached_gaussian": float(rng["numpy_cached_gaussian"]),
        "torch_cpu": arrays["rng_torch_cpu"],
        "torch_cuda": [arrays[f"rng_torch_cuda_{index}"] for index in range(count)],
    }


def load_transaction(schedule_dir: Path, request_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    metadata_path, arrays_path = transaction_paths(schedule_dir, request_id)
    if not metadata_path.exists() and not arrays_path.exists():
        return None
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise RuntimeError(f"incomplete server transaction {request_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["arrays_sha256"] != stage0.sha256_file(arrays_path):
        raise RuntimeError(f"server transaction array hash mismatch for {request_id}")
    arrays = read_npz(arrays_path)
    if stage0.hash_array(arrays["raw_chunk"]) != metadata["raw_chunk_sha256"]:
        raise RuntimeError(f"server transaction raw hash mismatch for {request_id}")
    if stage0.hash_array(arrays["processed_chunk"]) != metadata["processed_chunk_sha256"]:
        raise RuntimeError(f"server transaction processed hash mismatch for {request_id}")
    return metadata, arrays


def existing_transactions(schedule_dir: Path) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    rows: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    root = schedule_dir / "server_transactions"
    for metadata_path in root.glob("*.json") if root.is_dir() else []:
        loaded = load_transaction(schedule_dir, metadata_path.stem)
        if loaded is not None:
            rows.append(loaded)
    rows.sort(key=lambda item: int(item[0]["global_noise_position"]))
    if [int(item[0]["global_noise_position"]) for item in rows] != list(range(len(rows))):
        raise RuntimeError("server transaction noise positions are not a contiguous prefix")
    return rows


def prepare_model_inputs(
    request: Mapping[str, Any], processor: Any, model: Any, torch_module: Any
) -> tuple[dict[str, Any], str]:
    images = [np.asarray(request["agentview"], dtype=np.uint8), np.asarray(request["wrist"], dtype=np.uint8)]
    inputs = processor(images, str(request["language"]))
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch_module.Tensor):
            if value.is_floating_point():
                model_inputs[key] = value.to(device="cuda:0", dtype=torch_module.float32)
            else:
                model_inputs[key] = value.to(device="cuda:0")
        else:
            model_inputs[key] = value
    model_inputs["proprio"] = torch_module.as_tensor(
        np.asarray(request["proprio"], dtype=np.float32), device="cuda:0", dtype=torch_module.float32
    ).unsqueeze(0)
    model_inputs["domain_id"] = torch_module.tensor([3], dtype=torch_module.long, device="cuda:0")
    return model_inputs, stage0.tensor_mapping_hash(model_inputs, torch_module)


def serve_request(
    schedule_dir: Path,
    request: Mapping[str, Any],
    global_noise_position: int,
    model: Any,
    processor: Any,
    torch_module: Any,
) -> dict[str, Any]:
    request_id = str(request["request_id"])
    existing = load_transaction(schedule_dir, request_id)
    if existing is not None:
        metadata, arrays = existing
        if metadata["policy_input_sha256"] != request["policy_input_sha256"]:
            raise RuntimeError(f"transaction input mismatch on resume for {request_id}")
        return {
            "request_id": request_id,
            "global_noise_position": int(metadata["global_noise_position"]),
            "request_arrived_monotonic_ns": int(metadata["request_arrived_monotonic_ns"]),
            "service_started_monotonic_ns": int(metadata["service_started_monotonic_ns"]),
            "service_completed_monotonic_ns": int(metadata["service_completed_monotonic_ns"]),
            "prepared_input_sha256": metadata["prepared_input_sha256"],
            "raw_chunk": arrays["raw_chunk"],
            "processed_chunk": arrays["processed_chunk"],
            "transaction_sha256": stage0.sha256_file(transaction_paths(schedule_dir, request_id)[0]),
            "replayed_from_transaction": True,
        }
    arrived_ns = time.monotonic_ns()
    model_inputs, prepared_hash = prepare_model_inputs(request, processor, model, torch_module)
    service_started_ns = time.monotonic_ns()
    with torch_module.no_grad():
        action = model.generate_actions(**model_inputs, steps=10)
    torch_module.cuda.synchronize()
    service_completed_ns = time.monotonic_ns()
    raw_chunk = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32)
    processed_chunk = stage0.raw_to_processed_7d(raw_chunk).astype(np.float32)
    if raw_chunk.shape != RAW_CHUNK_SHAPE or not np.isfinite(raw_chunk).all():
        raise RuntimeError(f"nonfinite or malformed raw action chunk for {request_id}")
    rng_state = stage0.capture_rng_state(torch_module)
    rng_arrays, rng_metadata = rng_arrays_and_metadata(rng_state)
    metadata_path, arrays_path = transaction_paths(schedule_dir, request_id)
    if metadata_path.exists() or arrays_path.exists():
        raise RuntimeError(f"refusing to overwrite transaction {request_id}")
    write_npz_atomic(
        arrays_path,
        raw_chunk=raw_chunk,
        processed_chunk=processed_chunk,
        **rng_arrays,
    )
    metadata = {
        "schema_version": "epoch6.schedule_closed_loop.server_transaction.v1",
        "served_at": utc_now(),
        "request_id": request_id,
        "schedule": request["schedule"],
        "shard_id": int(request["shard_id"]),
        "identity": request["identity"],
        "episode_policy_call_index": int(request["episode_policy_call_index"]),
        "global_noise_position": int(global_noise_position),
        "request_sent_monotonic_ns": int(request["request_sent_monotonic_ns"]),
        "request_arrived_monotonic_ns": int(arrived_ns),
        "service_started_monotonic_ns": int(service_started_ns),
        "service_completed_monotonic_ns": int(service_completed_ns),
        "pre_query_sim_state_sha256": request["pre_query_sim_state_sha256"],
        "policy_input_sha256": request["policy_input_sha256"],
        "prepared_input_sha256": prepared_hash,
        "raw_chunk_sha256": stage0.hash_array(raw_chunk),
        "processed_chunk_sha256": stage0.hash_array(processed_chunk),
        "rng_state_after_service": rng_metadata,
        "arrays_path": str(arrays_path),
        "arrays_sha256": stage0.sha256_file(arrays_path),
    }
    stage0.write_json(metadata_path, metadata)
    return {
        "request_id": request_id,
        "global_noise_position": global_noise_position,
        "request_arrived_monotonic_ns": arrived_ns,
        "service_started_monotonic_ns": service_started_ns,
        "service_completed_monotonic_ns": service_completed_ns,
        "prepared_input_sha256": prepared_hash,
        "raw_chunk": raw_chunk,
        "processed_chunk": processed_chunk,
        "transaction_sha256": stage0.sha256_file(metadata_path),
        "replayed_from_transaction": False,
    }


def run_schedule(run_dir: Path, schedule: str) -> int:
    validate_manifest()
    stage0.require_parent_run_lock(run_dir)
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule}")
    schedule_dir = run_dir / schedule
    schedule_dir.mkdir(parents=True, exist_ok=True)
    final_path = schedule_dir / "schedule_result.json"
    if final_path.exists():
        raise RuntimeError(f"refusing to overwrite completed schedule {schedule}")
    before = stage0.resource_snapshot()
    stage0.require_safe_resources(before)
    assignments = SERIAL_ASSIGNMENTS if schedule == SCHEDULES[0] else SHARDED_ASSIGNMENTS
    offsets = SERIAL_OFFSETS if schedule == SCHEDULES[0] else SHARDED_OFFSETS
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    response_queues = {shard_id: context.Queue() for shard_id in assignments}
    workers = []
    for shard_id, identities in assignments.items():
        worker = context.Process(
            target=simulator_worker,
            args=(
                schedule,
                shard_id,
                identities,
                offsets[shard_id],
                str(run_dir),
                request_queue,
                response_queues[shard_id],
            ),
            name=f"epoch6-{schedule}-shard-{shard_id}",
        )
        worker.start()
        workers.append(worker)
    torch_module = stage0.seed_process_once(ROOT_SEED)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    monitor = stage0.ResourceMonitor(
        torch_module,
        schedule_dir / "resource_heartbeat.json",
    )
    monitor.start()
    model = processor = None
    errors: list[dict[str, Any]] = []
    done_workers: set[int] = set()
    completed_identities: set[int] = set()
    result: dict[str, Any] = {
        "schema_version": "epoch6.schedule_closed_loop.schedule.v1",
        "started_at": utc_now(),
        "schedule": schedule,
        "root_seed": ROOT_SEED,
        "env_seed": ENV_SEED,
        "assignments": {str(key): value for key, value in assignments.items()},
        "launch_offset_seconds": {str(key): value for key, value in offsets.items()},
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "execution_manifest_sha256": EXPECTED_EXECUTION_MANIFEST_SHA256,
        "stage0_result_sha256": EXPECTED_STAGE0_RESULT_SHA256,
        "resources_before": before,
        "errors": errors,
    }
    exit_code = 1
    try:
        model, processor, runtime = stage0.load_xvla(torch_module)
        result["runtime"] = runtime
        transactions = existing_transactions(schedule_dir)
        if transactions:
            latest_metadata, latest_arrays = transactions[-1]
            stage0.restore_rng_state(
                rng_state_from_transaction(latest_metadata, latest_arrays), torch_module
            )
        next_noise_position = len(transactions)
        last_message_at = time.monotonic()
        while len(done_workers) < len(workers):
            try:
                message = request_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [
                    worker
                    for worker in workers
                    if worker.exitcode is not None and worker.exitcode != 0
                ]
                if failed:
                    raise RuntimeError(
                        "one or more simulator workers exited nonzero: "
                        + ", ".join(f"{worker.name}={worker.exitcode}" for worker in failed)
                    )
                if time.monotonic() - last_message_at > 600:
                    raise RuntimeError("no simulator-worker message for 600 seconds")
                continue
            last_message_at = time.monotonic()
            message_type = message.get("type")
            if message_type == "request":
                existing = load_transaction(schedule_dir, str(message["request_id"]))
                service_position = (
                    int(existing[0]["global_noise_position"])
                    if existing is not None
                    else next_noise_position
                )
                response = serve_request(
                    schedule_dir,
                    message,
                    service_position,
                    model,
                    processor,
                    torch_module,
                )
                if existing is None:
                    next_noise_position += 1
                response_queues[int(message["shard_id"])].put(response)
            elif message_type == "episode_complete":
                completed_identities.add(int(message["identity_global_index"]))
            elif message_type == "worker_done":
                done_workers.add(int(message["shard_id"]))
            elif message_type == "worker_error":
                errors.append(dict(message))
                raise RuntimeError(message["exception"])
            else:
                raise RuntimeError(f"unknown simulator-worker message: {message_type!r}")
            stage0.write_json(
                schedule_dir / "heartbeat.json",
                {
                    "status": "running",
                    "updated_at": utc_now(),
                    "schedule": schedule,
                    "served_query_count": next_noise_position,
                    "completed_identity_count": len(completed_identities),
                    "done_worker_count": len(done_workers),
                },
            )
        for worker in workers:
            worker.join(timeout=60)
            if worker.exitcode != 0:
                raise RuntimeError(f"simulator worker {worker.name} exited {worker.exitcode}")
        episode_files = sorted((schedule_dir / "episodes").glob("identity_*.json"))
        if len(episode_files) != 20:
            raise RuntimeError(f"schedule {schedule} completed only {len(episode_files)}/20 episodes")
        episodes = [json.loads(path.read_text(encoding="utf-8")) for path in episode_files]
        if any(row["exception"] is not None for row in episodes):
            raise RuntimeError(f"schedule {schedule} has episode exceptions")
        transaction_rows = existing_transactions(schedule_dir)
        result.update(
            {
                "completed_at": utc_now(),
                "completed_episode_count": len(episodes),
                "successful_episode_count": sum(bool(row["success"]) for row in episodes),
                "timeout_episode_count": sum(bool(row["timeout"]) for row in episodes),
                "served_query_count": len(transaction_rows),
                "episode_metadata_sha256": {
                    path.stem: stage0.sha256_file(path) for path in episode_files
                },
                "transaction_metadata_sha256": {
                    item[0]["request_id"]: stage0.sha256_file(
                        transaction_paths(schedule_dir, item[0]["request_id"])[0]
                    )
                    for item in transaction_rows
                },
                "errors": errors,
                "status": "SCHEDULE_COMPLETE",
            }
        )
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "failed_at": utc_now(),
                "status": "SCHEDULE_FAILED",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        model = processor = None
        try:
            import gc

            gc.collect()
            torch_module.cuda.empty_cache()
        except Exception:
            pass
        result["resource_monitor"] = monitor.stop()
        result["resources_after"] = stage0.resource_snapshot(torch_module)
        if (
            result["resource_monitor"]["maximum_swap_used_bytes"] != 0
            or result["resource_monitor"]["exceptions"]
        ):
            result["status"] = "SCHEDULE_FAILED_RESOURCE_TELEMETRY"
            exit_code = 1
        stage0.write_json(final_path if exit_code == 0 else schedule_dir / f"schedule_failure_{int(time.time())}.json", result)
        stage0.write_text(schedule_dir / "exit_code.txt", f"{exit_code}\n")
    return exit_code


def static_preflight(run_dir: Path) -> dict[str, Any]:
    manifest = validate_manifest()
    if not (run_dir / "static_preflight.json").is_file():
        stage0.static_preflight(run_dir)
    provenance = stage0.validate_execution_provenance(run_dir)
    source_checks = {
        "harness_libero_benchmark_sha256": stage0.sha256_file(
            stage0.HARNESS_ROOT / "src" / "vla_eval" / "benchmarks" / "libero" / "benchmark.py"
        ),
        "harness_xvla_server_sha256": stage0.sha256_file(
            stage0.HARNESS_ROOT / "src" / "vla_eval" / "model_servers" / "xvla.py"
        ),
        "xvla_modeling_source_sha256": stage0.sha256_file(
            stage0.XVLA_ROOT / "models" / "modeling_xvla.py"
        ),
        "harness_xvla_libero_config_sha256": stage0.sha256_file(
            stage0.HARNESS_ROOT / "configs" / "model_servers" / "xvla" / "libero.yaml"
        ),
        "harness_suite_config_sha256": {
            "libero_spatial": stage0.sha256_file(
                stage0.HARNESS_ROOT / "configs" / "benchmarks" / "libero" / "spatial.yaml"
            ),
            "libero_object": stage0.sha256_file(
                stage0.HARNESS_ROOT / "configs" / "benchmarks" / "libero" / "object.yaml"
            ),
            "libero_goal": stage0.sha256_file(
                stage0.HARNESS_ROOT / "configs" / "benchmarks" / "libero" / "goal.yaml"
            ),
            "libero_10": stage0.sha256_file(
                stage0.HARNESS_ROOT / "configs" / "benchmarks" / "libero" / "10.yaml"
            ),
        },
    }
    pinned = manifest["pinned_sources"]
    for key, value in source_checks.items():
        if value != pinned[key]:
            raise RuntimeError(f"closed-loop pinned source mismatch for {key}")
    implementation_checks = {
        "runner_sha256": stage0.sha256_file(Path(__file__).resolve()),
        "monitor_sha256": stage0.sha256_file(
            REPO_ROOT / "scripts" / "monitor_epoch6_schedule_closed_loop_smoke.ps1"
        ),
        "smoke_wrapper_sha256": stage0.sha256_file(
            REPO_ROOT / "scripts" / "run_epoch6_schedule_closed_loop_smoke_wsl.sh"
        ),
    }
    if implementation_checks["monitor_sha256"] != EXPECTED_CLOSED_LOOP_MONITOR_SHA256:
        raise RuntimeError("closed-loop monitor implementation hash mismatch")
    task_rows = []
    for suite_name in SUITES:
        _env_class, bddl, init_path, initial_states, language = task_assets(suite_name)
        task_rows.append(
            {
                "suite": suite_name,
                "task_id": TASK_ID,
                "language": language,
                "bddl_path": str(bddl),
                "bddl_sha256": stage0.sha256_file(bddl),
                "init_states_path": str(init_path),
                "init_states_sha256": stage0.sha256_file(init_path),
                "selected_initial_state_sha256": {
                    str(index): exact_array_hash(initial_states[index]) for index in range(5)
                },
                "horizon": HORIZONS[suite_name],
            }
        )
    result = {
        "schema_version": "epoch6.schedule_closed_loop.static_preflight.v1",
        "completed_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "execution_manifest_sha256": EXPECTED_EXECUTION_MANIFEST_SHA256,
        "stage0_result_sha256": EXPECTED_STAGE0_RESULT_SHA256,
        "resource_amendment_sha256": EXPECTED_RESOURCE_AMENDMENT_SHA256,
        "provenance": provenance,
        "source_checks": source_checks,
        "implementation_checks": implementation_checks,
        "identity_manifest": IDENTITIES,
        "task_rows": task_rows,
        "scientific_outcomes_read": False,
        "simulator_actions_executed": 0,
        "status": "CLOSED_LOOP_PREFLIGHT_PASS",
    }
    stage0.write_json(run_dir / "closed_loop_static_preflight.json", result)
    return result


def resource_env_holder(shard_id: int, ready_queue: Any, release_event: Any) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    env = None
    try:
        benchmark, OffScreenRenderEnv, _executed = stage0.import_pinned_libero()
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        asset_root = stage0.LIBERO_ROOT / "libero" / "libero"
        bddl = (asset_root / "bddl_files" / task.problem_folder / task.bddl_file).resolve(strict=True)
        init_path = (asset_root / "init_files" / task.problem_folder / task.init_states_file).resolve(strict=True)
        import torch

        initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
        success_check_calls = 0

        def forbidden_success_check(*_args: Any, **_kwargs: Any) -> None:
            nonlocal success_check_calls
            success_check_calls += 1
            raise RuntimeError("success checking is forbidden during the closed-loop resource smoke")

        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        original_wrapper_check = env.check_success
        original_inner_check = env.env._check_success
        env.check_success = forbidden_success_check
        env.env._check_success = forbidden_success_check
        env.seed(ENV_SEED)
        env.reset()
        env.set_state(initial_states[0])
        observation = regenerate_observation(env)
        env.check_success = original_wrapper_check
        env.env._check_success = original_inner_check
        agentview, wrist, proprio = make_policy_request_arrays(env, observation, None)
        ready_queue.put(
            {
                "status": "ready",
                "shard_id": shard_id,
                "pid": os.getpid(),
                "success_check_calls": success_check_calls,
                "agentview": agentview,
                "wrist": wrist,
                "proprio": proprio,
                "language": str(task.language),
            }
        )
        if not release_event.wait(timeout=900):
            raise RuntimeError("resource-smoke environment holder timed out waiting for release")
    except Exception as exc:
        ready_queue.put(
            {
                "status": "error",
                "shard_id": shard_id,
                "pid": os.getpid(),
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def resource_smoke(run_dir: Path) -> int:
    validate_manifest()
    stage0.require_host_smoke_lock(run_dir)
    provenance = stage0.validate_execution_provenance(run_dir)
    before = stage0.resource_snapshot()
    stage0.require_safe_resources(before)
    result: dict[str, Any] = {
        "schema_version": "epoch6.schedule_closed_loop.resource_smoke.v1",
        "started_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "execution_manifest_sha256": EXPECTED_EXECUTION_MANIFEST_SHA256,
        "provenance": provenance,
        "resources_before": before,
        "simultaneous_env_instances": 0,
        "simultaneous_env_processes": 0,
        "model_inference_calls": 0,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "status": "CLOSED_LOOP_RESOURCE_SMOKE_FAILED",
    }
    holder_context = mp.get_context("spawn")
    ready_queue = holder_context.Queue()
    release_event = holder_context.Event()
    holders: list[Any] = []
    model = processor = torch_module = None
    monitor = None
    exit_code = 1
    try:
        for shard_id in range(4):
            holder = holder_context.Process(
                target=resource_env_holder,
                args=(shard_id, ready_queue, release_event),
                name=f"epoch6-closed-loop-smoke-env-{shard_id}",
            )
            holder.start()
            holders.append(holder)
        torch_module = stage0.seed_process_once(ROOT_SEED)
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        monitor = stage0.ResourceMonitor(torch_module, run_dir / "closed_loop_resource_smoke_heartbeat.json")
        monitor.start()
        ready_rows = []
        while len(ready_rows) < 4:
            message = ready_queue.get(timeout=300)
            if message.get("status") != "ready":
                raise RuntimeError(
                    f"resource-smoke env holder {message.get('shard_id')} failed: {message.get('exception')}"
                )
            ready_rows.append(message)
        ready_rows.sort(key=lambda row: int(row["shard_id"]))
        success_check_calls = sum(int(row["success_check_calls"]) for row in ready_rows)
        result["simultaneous_env_instances"] = len(ready_rows)
        result["simultaneous_env_processes"] = len(holders)
        result["env_holder_pids"] = [int(row["pid"]) for row in ready_rows]
        if success_check_calls != 0:
            raise RuntimeError("closed-loop resource smoke called success logic")
        model, processor, runtime = stage0.load_xvla(torch_module)
        result["runtime"] = runtime
        first = ready_rows[0]
        request = {
            "agentview": first["agentview"],
            "wrist": first["wrist"],
            "proprio": first["proprio"],
            "language": first["language"],
        }
        model_inputs, prepared_hash = prepare_model_inputs(request, processor, model, torch_module)
        started = time.monotonic()
        with torch_module.no_grad():
            action = model.generate_actions(**model_inputs, steps=10)
        torch_module.cuda.synchronize()
        raw = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32)
        result.update(
            {
                "forward_seconds": time.monotonic() - started,
                "prepared_input_sha256": prepared_hash,
                "raw_chunk_shape": list(raw.shape),
                "raw_chunk_finite": bool(np.isfinite(raw).all()),
                "raw_chunk_sha256": stage0.hash_array(raw),
                "model_inference_calls": 1,
                "success_check_calls": success_check_calls,
                "status": "CLOSED_LOOP_ACTUAL_PATH_RESOURCE_SMOKE_PASS",
            }
        )
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        release_event.set()
        for holder in holders:
            holder.join(timeout=60)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=10)
            if holder.exitcode not in (0, None):
                result.setdefault("env_holder_exit_errors", []).append(
                    {"name": holder.name, "exit_code": holder.exitcode}
                )
                result["status"] = "CLOSED_LOOP_RESOURCE_SMOKE_FAILED_ENV_HOLDER_EXIT"
                exit_code = 1
        model = processor = None
        if torch_module is not None:
            import gc

            gc.collect()
            torch_module.cuda.empty_cache()
        if monitor is not None:
            result["resource_monitor"] = monitor.stop()
            if (
                result["resource_monitor"]["maximum_swap_used_bytes"] != 0
                or result["resource_monitor"]["exceptions"]
            ):
                result["status"] = "CLOSED_LOOP_RESOURCE_SMOKE_FAILED_TELEMETRY_OR_SWAP"
                exit_code = 1
        result["resources_after"] = stage0.resource_snapshot(torch_module)
        result["completed_at"] = utc_now()
        stage0.write_json(run_dir / "closed_loop_resource_smoke.json", result)
        stage0.write_text(run_dir / "closed_loop_resource_smoke_exit_code.txt", f"{exit_code}\n")
    return exit_code


def valid_resource_smoke(run_dir: Path) -> bool:
    try:
        internal_path = run_dir / "closed_loop_resource_smoke.json"
        host_path = run_dir / "closed_loop_resource_smoke_host.json"
        internal = json.loads(internal_path.read_text(encoding="utf-8"))
        host = json.loads(host_path.read_text(encoding="utf-8-sig"))
        return bool(
            internal["status"] == "CLOSED_LOOP_ACTUAL_PATH_RESOURCE_SMOKE_PASS"
            and internal["simultaneous_env_instances"] == 4
            and internal["simultaneous_env_processes"] == 4
            and internal["model_inference_calls"] == 1
            and internal["raw_chunk_shape"] == list(RAW_CHUNK_SHAPE)
            and internal["raw_chunk_finite"]
            and internal["success_check_calls"] == 0
            and internal["simulator_actions_executed"] == 0
            and not internal["reward_success_done_read"]
            and internal["resource_monitor"]["maximum_swap_used_bytes"] == 0
            and not internal["resource_monitor"]["exceptions"]
            and internal["runtime"]["parameter_devices"] == ["cuda:0"]
            and not internal["runtime"]["cpu_or_disk_model_offload"]
            and host["final_decision"] == "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_PASS_CALIBRATED"
            and host["idle_control_valid"]
            and not host["sustained_paging_detected"]
            and host["clean_state_restored"]
            and not host["oom_or_kill_signature_detected"]
            and host["internal_report_sha256"] == stage0.sha256_file(internal_path)
            and host["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
            and host["execution_manifest_sha256"] == EXPECTED_EXECUTION_MANIFEST_SHA256
            and host["resource_amendment_sha256"] == EXPECTED_RESOURCE_AMENDMENT_SHA256
            and host["monitor_script_sha256"] == EXPECTED_CLOSED_LOOP_MONITOR_SHA256
        )
    except Exception:
        return False


def valid_schedule(run_dir: Path, schedule: str) -> bool:
    try:
        schedule_dir = run_dir / schedule
        result = json.loads((schedule_dir / "schedule_result.json").read_text(encoding="utf-8"))
        episodes = list((schedule_dir / "episodes").glob("identity_*.json"))
        return bool(
            result["status"] == "SCHEDULE_COMPLETE"
            and result["completed_episode_count"] == 20
            and len(episodes) == 20
            and result["resource_monitor"]["maximum_swap_used_bytes"] == 0
            and not result["resource_monitor"]["exceptions"]
            and not result["errors"]
        )
    except Exception:
        return False


def adjudicate(run_dir: Path) -> dict[str, Any]:
    validate_manifest()
    if not all(valid_schedule(run_dir, schedule) for schedule in SCHEDULES):
        raise RuntimeError("both complete valid schedules are required for adjudication")
    schedule_episodes: dict[str, dict[int, dict[str, Any]]] = {}
    for schedule in SCHEDULES:
        rows = {}
        for path in sorted((run_dir / schedule / "episodes").glob("identity_*.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            identity_index = int(episode["identity"]["global_index"])
            rows[identity_index] = episode
        schedule_episodes[schedule] = rows
    integrity = {
        "identity_sets_match": set(schedule_episodes[SCHEDULES[0]]) == set(range(20))
        and set(schedule_episodes[SCHEDULES[1]]) == set(range(20)),
        "initial_states_match": True,
        "post_settle_states_match": True,
        "zero_exceptions": True,
        "zero_swap_use": True,
        "sharded_first_arrival_order_matches": False,
    }
    serial = schedule_episodes[SCHEDULES[0]]
    sharded = schedule_episodes[SCHEDULES[1]]
    paired_rows = []
    discordant_count = 0
    trace_tied_count = 0
    for identity_index in range(20):
        left = serial[identity_index]
        right = sharded[identity_index]
        if left["initial_state_sha256"] != right["initial_state_sha256"]:
            integrity["initial_states_match"] = False
        if left["post_settle_state_sha256"] != right["post_settle_state_sha256"]:
            integrity["post_settle_states_match"] = False
        if left["exception"] is not None or right["exception"] is not None:
            integrity["zero_exceptions"] = False
        discordant = bool(left["success"] != right["success"] or left["timeout"] != right["timeout"])
        discordant_count += int(discordant)
        left_queries = {int(row["episode_policy_call_index"]): row for row in left["query_rows"]}
        right_queries = {int(row["episode_policy_call_index"]): row for row in right["query_rows"]}
        remapped_call = None
        trace_tied = False
        for call_index in sorted(set(left_queries) & set(right_queries)):
            left_query = left_queries[call_index]
            right_query = right_queries[call_index]
            if int(left_query["global_noise_position"]) != int(right_query["global_noise_position"]):
                remapped_call = call_index
                trace_tied = bool(
                    left_query["pre_query_sim_state_sha256"]
                    == right_query["pre_query_sim_state_sha256"]
                    and left_query["raw_chunk_sha256"] != right_query["raw_chunk_sha256"]
                )
                break
        if discordant and trace_tied:
            trace_tied_count += 1
        paired_rows.append(
            {
                "identity": IDENTITY_BY_INDEX[identity_index],
                "serial_success": bool(left["success"]),
                "serial_timeout": bool(left["timeout"]),
                "sharded_success": bool(right["success"]),
                "sharded_timeout": bool(right["timeout"]),
                "discordant": discordant,
                "first_schedule_remapped_policy_call": remapped_call,
                "trace_divergence_tied_at_first_remap": trace_tied,
            }
        )
    sharded_transactions = existing_transactions(run_dir / SCHEDULES[1])
    first_by_shard: dict[int, tuple[int, int]] = {}
    for metadata, _arrays in sharded_transactions:
        shard_id = int(metadata["shard_id"])
        pair = (int(metadata["request_arrived_monotonic_ns"]), int(metadata["global_noise_position"]))
        first_by_shard.setdefault(shard_id, pair)
    actual_first_order = [
        shard_id for shard_id, _pair in sorted(first_by_shard.items(), key=lambda item: item[1])
    ]
    integrity["sharded_first_arrival_order_matches"] = actual_first_order == REQUIRED_SHARDED_FIRST_ARRIVAL
    for schedule in SCHEDULES:
        schedule_result = json.loads(
            (run_dir / schedule / "schedule_result.json").read_text(encoding="utf-8")
        )
        if schedule_result["resource_monitor"]["maximum_swap_used_bytes"] != 0:
            integrity["zero_swap_use"] = False
    serial_successes = sum(bool(row["success"]) for row in serial.values())
    sharded_successes = sum(bool(row["success"]) for row in sharded.values())
    disagreement_fraction = discordant_count / 20.0
    success_rate_spread = abs(serial_successes - sharded_successes) / 20.0
    trace_fraction = trace_tied_count / discordant_count if discordant_count else 0.0
    gates = {
        "paired_disagreement": disagreement_fraction >= 0.20,
        "success_rate_spread": success_rate_spread >= 0.10,
        "trace_tied": trace_fraction >= 0.80,
        "integrity": all(integrity.values()),
    }
    if not all(integrity.values()):
        decision = "EVALUATION_INVALID"
    elif not all(gates.values()):
        decision = "NO_REPEATABLE_PROBLEM"
    else:
        decision = "PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED"
    result = {
        "schema_version": "epoch6.schedule_closed_loop.adjudication.v1",
        "completed_at": utc_now(),
        "execution_type": "DISCOVERY_PROBLEM_VERIFICATION_CLOSED_LOOP",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "execution_manifest_sha256": EXPECTED_EXECUTION_MANIFEST_SHA256,
        "stage0_result_sha256": EXPECTED_STAGE0_RESULT_SHA256,
        "episode_execution_count": 40,
        "serial_success_count": serial_successes,
        "sharded_success_count": sharded_successes,
        "paired_disagreement_count": discordant_count,
        "paired_disagreement_fraction": disagreement_fraction,
        "success_rate_spread": success_rate_spread,
        "trace_tied_discordant_count": trace_tied_count,
        "trace_tied_fraction_among_discordant": trace_fraction,
        "required_sharded_first_arrival_order": REQUIRED_SHARDED_FIRST_ARRIVAL,
        "actual_sharded_first_arrival_order": actual_first_order,
        "integrity": integrity,
        "gates_passed": gates,
        "paired_rows": paired_rows,
        "final_decision": decision,
        "method_design_authorized": decision == "PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED",
        "ours_executed": False,
        "training_happened": False,
        "paper_generation_authorized": False,
    }
    stage0.write_json(run_dir / "closed_loop_result.json", result)
    return result


def launch_child(run_dir: Path, schedule: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "schedule",
        "--schedule",
        schedule,
        "--run-dir",
        str(run_dir),
        "--child",
    ]
    attempt = 1
    while (run_dir / f"launch_{schedule}_{attempt:03d}.json").exists():
        attempt += 1
    launch_path = run_dir / f"launch_{schedule}_{attempt:03d}.json"
    stdout_path = run_dir / f"launch_{schedule}_{attempt:03d}.stdout.log"
    stderr_path = run_dir / f"launch_{schedule}_{attempt:03d}.stderr.log"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(ROOT_SEED)
    environment["EPOCH6_PARENT_RUN_LOCK"] = str((run_dir / "run.lock.json").resolve())
    started_at = utc_now()
    stage0.write_json(
        launch_path,
        {
            "status": "running",
            "started_at": started_at,
            "command": command,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    stage0.write_json(
        launch_path,
        {
            "status": "completed" if completed.returncode == 0 else "failed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "command": command,
            "exit_code": completed.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"closed-loop schedule {schedule} failed; see {launch_path}")


def run_all(run_dir: Path, resume: bool) -> int:
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory exists; use --resume or a fresh path: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stage0.acquire_run_lock(run_dir)
    exit_code = 1
    try:
        if not (resume and (run_dir / "closed_loop_static_preflight.json").is_file()):
            static_preflight(run_dir)
        if not valid_resource_smoke(run_dir):
            raise RuntimeError("valid host-qualified closed-loop resource smoke is required")
        for schedule in SCHEDULES:
            if resume and valid_schedule(run_dir, schedule):
                continue
            launch_child(run_dir, schedule)
        result = adjudicate(run_dir)
        stage0.write_json(
            run_dir / "heartbeat.json",
            {
                "status": "completed",
                "updated_at": utc_now(),
                "final_decision": result["final_decision"],
            },
        )
        stage0.write_text(run_dir / "exit_code.txt", "0\n")
        exit_code = 0
    except Exception as exc:
        stage0.write_json(
            run_dir / f"parent_failure_{int(time.time())}.json",
            {
                "status": "CLOSED_LOOP_PARENT_FAILED",
                "failed_at": utc_now(),
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        stage0.release_run_lock(run_dir, lock_path)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preflight", "resource-smoke", "schedule", "adjudicate", "run-all"],
        default="run-all",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--schedule", choices=SCHEDULES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        static_preflight(run_dir)
        return 0
    if args.mode == "resource-smoke":
        if not args.child:
            raise RuntimeError("resource-smoke is host-monitor child-only")
        return resource_smoke(run_dir)
    if args.mode == "schedule":
        if not args.child or args.schedule is None:
            raise RuntimeError("schedule mode requires parent child launch and --schedule")
        return run_schedule(run_dir, args.schedule)
    if args.mode == "adjudicate":
        adjudicate(run_dir)
        return 0
    return run_all(run_dir, bool(args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
