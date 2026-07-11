"""Official SmolVLA/LIBERO closed-loop scaleup and failure-mining runner.

The runner keeps the official LeRobot/LIBERO policy, preprocessing,
postprocessing, and action-queue path.  It adds only experiment bookkeeping:
predeclared task/reset manifests, per-episode tracing, paired statistics, and
failure-review records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_wsl_libero_rollout import (
    POLICIES,
    RENAME_MAP,
    _cuda_memory,
    _json_default,
    _load_policy_and_processors,
    _make_env_cfg,
    _round,
    _set_runtime_env,
    static_mix_duplicate_records,
)


SCALEUP_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASKS_PER_SUITE = 5
RESET_SEEDS = [20260711, 20260712, 20260713, 20260714, 20260715]
OFFLINE_ACTION_L2 = {
    "frozen_base": 0.085579125,
    "rank4_lora_seed_11": 0.086743582,
    "rank4_lora_seed_22": 0.086474081,
    "rank4_lora_seed_33": 0.086918872,
}
FAILURE_CATEGORIES = [
    "target_or_object_selection",
    "initial_reach",
    "orientation_or_rotation",
    "grasp_approach",
    "gripper_timing_or_contact",
    "object_transport",
    "placement_or_release",
    "collision_or_workspace_violation",
    "action_chunk_drift",
    "long_horizon_compounding",
    "stochastic_policy_variation",
    "environment_or_infrastructure_failure",
    "ambiguous_or_unclassified",
]
FINAL_DECISIONS = {
    "CLOSED_LOOP_STRUCTURED_FAILURE_READY_FOR_NOVELTY_REVIEW",
    "OFFICIAL_BASELINE_SCALEUP_READY",
    "LORA_TRAINING_SEED_INSTABILITY_IS_PRIMARY_GAP",
    "SPECIFIC_CONTROL_PHASE_FAILURE_FOUND",
    "OFFLINE_ONLINE_MISMATCH_CONFIRMED",
    "NO_METHOD_WORTHY_CLOSED_LOOP_GAP",
    "ROLLOUT_RESULTS_TOO_NOISY",
    "ROLLOUT_INFRASTRUCTURE_FAILURE",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str], timeout: int = 30) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # pragma: no cover - host boundary
        return f"{type(exc).__name__}: {exc}"
    output = (completed.stdout or completed.stderr or "").strip()
    return output[:4000]


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:
        return None


def select_evenly_spaced_task_ids(n_tasks: int, count: int = TASKS_PER_SUITE) -> list[int]:
    if n_tasks < count:
        raise ValueError(f"need at least {count} tasks, got {n_tasks}")
    return [int((index * n_tasks) // count) for index in range(count)]


def build_task_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from libero.libero.benchmark import get_benchmark

    tasks = []
    for suite in SCALEUP_SUITES:
        benchmark = get_benchmark(suite)()
        task_ids = select_evenly_spaced_task_ids(int(benchmark.n_tasks), TASKS_PER_SUITE)
        for task_id in task_ids:
            task = benchmark.get_task(int(task_id))
            instruction = str(getattr(task, "language", "") or getattr(task, "problem_name", "") or task)
            tasks.append(
                {
                    "suite": suite,
                    "task_id": int(task_id),
                    "instruction": instruction,
                    "selection_rule": f"floor(k * {benchmark.n_tasks} / {TASKS_PER_SUITE})",
                    "suite_task_count": int(benchmark.n_tasks),
                }
            )
    payload = {
        "schema_version": 1,
        "date": args.date,
        "suites": list(SCALEUP_SUITES),
        "tasks_per_suite": TASKS_PER_SUITE,
        "task_selection": "deterministic evenly spaced ids: floor(k * n_tasks / 5)",
        "tasks": tasks,
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_json({k: v for k, v in payload.items() if k != "canonical_payload_sha256"}))
    return payload


def build_episode_manifest(args: argparse.Namespace, task_manifest: dict[str, Any]) -> dict[str, Any]:
    policies = [spec.name for spec in POLICIES]
    episodes = []
    index = 0
    for policy in policies:
        for task in task_manifest["tasks"]:
            for seed in RESET_SEEDS:
                episodes.append(
                    {
                        "planned_episode_index": index,
                        "episode_id": f"{policy}|{task['suite']}|task_{task['task_id']}|seed_{seed}",
                        "policy": policy,
                        "suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "instruction": task["instruction"],
                        "reset_seed": int(seed),
                    }
                )
                index += 1
    payload = {
        "schema_version": 1,
        "date": args.date,
        "policies": policies,
        "reset_seeds": list(RESET_SEEDS),
        "batch_size": 1,
        "max_parallel_tasks": 1,
        "control_mode": "relative",
        "static_mix_duplicate_runs_skipped": True,
        "static_mix_duplicate_records": static_mix_duplicate_records(),
        "planned_episode_count": len(episodes),
        "episodes": episodes,
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_json({k: v for k, v in payload.items() if k != "canonical_payload_sha256"}))
    return payload


def write_plan_reports(args: argparse.Namespace, task_manifest: dict[str, Any], episode_manifest: dict[str, Any]) -> None:
    report_dir = Path(args.report_dir)
    _write_json(report_dir / "official_closed_loop_task_manifest.json", task_manifest)
    _write_json(report_dir / "official_closed_loop_episode_manifest.json", episode_manifest)
    lines = [
        "# Official Closed-Loop Scaleup Plan",
        "",
        f"Date: {args.date} KST",
        "",
        "## Frozen Design",
        "",
        f"- suites: `{', '.join(SCALEUP_SUITES)}`",
        f"- task ids per suite: `{[task['task_id'] for task in task_manifest['tasks'][:TASKS_PER_SUITE]]}`",
        f"- reset seeds: `{RESET_SEEDS}`",
        "- policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`",
        f"- planned episodes: `{episode_manifest['planned_episode_count']}`",
        "- batch size: `1`",
        "- max parallel tasks: `1`",
        "- control mode: official relative control",
        "- static-mix duplicate rollouts: skipped because alpha is exactly `0.0`",
        "",
        "## Manifest Hashes",
        "",
        f"- task manifest canonical payload sha256: `{task_manifest['canonical_payload_sha256']}`",
        f"- episode manifest canonical payload sha256: `{episode_manifest['canonical_payload_sha256']}`",
        "",
        "## Guardrails",
        "",
        "No retraining, no seed selection after results, no static-mix duplicates, no old custom `LIBERO_7D` route, no exact-init replay bridge, no OpenVLA-OFT, and no rollout-outcome tuning.",
    ]
    _write_md(report_dir / "official_closed_loop_scaleup_plan.md", lines)


def _checkpoint_hashes(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_path)
    lora_root = Path(args.lora_root)
    return {
        "frozen_base": {
            "path": str(base_path),
            "model.safetensors": _sha256_file(base_path / "model.safetensors"),
            "config.json": _sha256_file(base_path / "config.json"),
        },
        "rank4_lora_seed_11": {
            "path": str(lora_root / "seed_11"),
            "adapter_model.safetensors": _sha256_file(lora_root / "seed_11" / "adapter_model.safetensors"),
            "adapter_config.json": _sha256_file(lora_root / "seed_11" / "adapter_config.json"),
        },
        "rank4_lora_seed_22": {
            "path": str(lora_root / "seed_22"),
            "adapter_model.safetensors": _sha256_file(lora_root / "seed_22" / "adapter_model.safetensors"),
            "adapter_config.json": _sha256_file(lora_root / "seed_22" / "adapter_config.json"),
        },
        "rank4_lora_seed_33": {
            "path": str(lora_root / "seed_33"),
            "adapter_model.safetensors": _sha256_file(lora_root / "seed_33" / "adapter_model.safetensors"),
            "adapter_config.json": _sha256_file(lora_root / "seed_33" / "adapter_config.json"),
        },
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    asset_path = Path("/home/jiheon/miniconda3-official/envs/official-smolvla-libero/lib/python3.10/site-packages/libero/libero/assets")
    try:
        asset_target = str(asset_path.resolve())
    except Exception:
        asset_target = None
    env_lock_path = Path(args.report_dir) / "wsl_official_rollout_environment_lock.md"
    task_manifest_path = Path(args.report_dir) / "official_closed_loop_task_manifest.json"
    episode_manifest_path = Path(args.report_dir) / "official_closed_loop_episode_manifest.json"
    return {
        "wsl": {
            "uname": _run_text(["uname", "-a"]),
            "os_release": _run_text(["bash", "-lc", "cat /etc/os-release"]),
            "free_h": _run_text(["free", "-h"]),
            "df_h": _run_text(["df", "-h", "/home/jiheon", "/mnt/c"]),
        },
        "cuda": {
            "nvidia_smi": _run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA",
            "memory": _cuda_memory(torch),
        },
        "packages": {
            "lerobot": _package_version("lerobot"),
            "hf_libero": _package_version("hf-libero") or _package_version("hf_libero"),
            "robosuite": _package_version("robosuite"),
            "mujoco": _package_version("mujoco"),
            "peft": _package_version("peft"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        "runtime_env": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
        "hashes": {
            "checkpoints": _checkpoint_hashes(args),
            "task_manifest_file": _sha256_file(task_manifest_path),
            "episode_manifest_file": _sha256_file(episode_manifest_path),
            "wsl_environment_lock_file": _sha256_file(env_lock_path),
        },
        "libero_assets": {
            "package_local_assets_path": str(asset_path),
            "package_local_assets_resolved": asset_target,
            "copied_asset_path": "/home/jiheon/assets/repos/LIBERO/libero/libero/assets",
            "is_symlink": asset_path.is_symlink(),
        },
        "official_schema": {
            "rename_map": dict(RENAME_MAP),
            "state_dim": 8,
            "action_dim": 7,
            "control_mode": "relative",
            "old_custom_libero_7d_route_used": False,
        },
    }


def _rss_mb() -> float | None:
    try:
        import psutil

        return _round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def _queue_owner(policy: Any) -> Any | None:
    candidates = [policy]
    for name in ("base_model", "model"):
        value = getattr(policy, name, None)
        if value is not None:
            candidates.append(value)
            nested = getattr(value, "model", None)
            if nested is not None:
                candidates.append(nested)
    for candidate in candidates:
        if hasattr(candidate, "_queues"):
            return candidate
    return None


def _action_queue_len(policy: Any, action_key: Any) -> int | None:
    owner = _queue_owner(policy)
    if owner is None:
        return None
    queue = getattr(owner, "_queues", {}).get(action_key)
    if queue is None:
        return None
    try:
        return int(len(queue))
    except Exception:
        return None


def _successes_from_info(info: dict[str, Any], n_envs: int) -> list[bool]:
    if "final_info" not in info:
        return [False] * n_envs
    final_info = info["final_info"]
    try:
        return [bool(x) for x in final_info["is_success"].tolist()]
    except Exception:
        return [False] * n_envs


def trace_one_episode(
    *,
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    seed: int,
    video_path: Path | None,
) -> dict[str, Any]:
    import torch
    from lerobot.scripts.lerobot_eval import (
        ACTION,
        add_envs_task,
        check_env_attributes_and_types,
        preprocess_observation,
        write_video,
    )

    if env.num_envs != 1:
        raise ValueError("scaleup trace expects batch size 1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    done = np.array([False])
    rewards: list[float] = []
    successes: list[bool] = []
    action_finite = True
    action_shape_ok = True
    action_max_abs = 0.0
    policy_latencies: list[float] = []
    env_latencies: list[float] = []
    chunks_generated = 0
    queue_observable = True
    terminated_last = False
    truncated_last = False
    frames = []

    capture_video = video_path is not None
    if capture_video:
        frames.append(env.envs[0].render())

    check_env_attributes_and_types(env)
    step = 0
    while not np.all(done) and step < max_steps:
        lerobot_observation = preprocess_observation(observation)
        lerobot_observation = add_envs_task(env, lerobot_observation)
        lerobot_observation = env_preprocessor(lerobot_observation)
        batch = preprocessor(lerobot_observation)

        queue_len_before = _action_queue_len(policy, ACTION)
        if queue_len_before is None:
            queue_observable = False
        elif queue_len_before == 0:
            chunks_generated += 1

        start_policy = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        policy_latencies.append(time.perf_counter() - start_policy)

        action = postprocessor(action)
        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]
        action_numpy = action.to("cpu").numpy()
        action_finite = action_finite and bool(np.isfinite(action_numpy).all())
        action_shape_ok = action_shape_ok and action_numpy.shape == (1, 7)
        action_max_abs = max(action_max_abs, float(np.max(np.abs(action_numpy))))

        start_env = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        env_latencies.append(time.perf_counter() - start_env)
        if capture_video:
            frames.append(env.envs[0].render())

        step_successes = _successes_from_info(info, env.num_envs)
        successes.append(bool(step_successes[0]))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_last = bool(np.asarray(truncated).reshape(-1)[0])

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        step += 1

    success = any(successes)
    sum_reward = float(np.sum(rewards)) if rewards else 0.0
    max_reward = float(np.max(rewards)) if rewards else 0.0
    if success:
        termination_reason = "success"
    elif terminated_last:
        termination_reason = "terminated_without_success"
    elif truncated_last or step >= max_steps:
        termination_reason = "max_steps_or_truncated_without_success"
    else:
        termination_reason = "done_without_success"

    saved_video_path = None
    if capture_video and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), np.stack(frames), env.unwrapped.metadata["render_fps"])
        saved_video_path = str(video_path)

    return {
        "success": bool(success),
        "sum_reward": _round(sum_reward, 6),
        "max_reward": _round(max_reward, 6),
        "episode_length": int(step),
        "termination_reason": termination_reason,
        "failure_status": "success" if success else "unsuccessful",
        "exception": None,
        "action_validity": {
            "finite": bool(action_finite),
            "shape_ok": bool(action_shape_ok),
            "max_abs": _round(action_max_abs, 6),
        },
        "action_chunks_generated": int(chunks_generated) if queue_observable else None,
        "env_steps": int(step),
        "policy_latency_mean_s": _round(float(np.mean(policy_latencies)), 6) if policy_latencies else None,
        "policy_latency_max_s": _round(float(np.max(policy_latencies)), 6) if policy_latencies else None,
        "env_step_latency_mean_s": _round(float(np.mean(env_latencies)), 6) if env_latencies else None,
        "env_step_latency_max_s": _round(float(np.max(env_latencies)), 6) if env_latencies else None,
        "peak_vram": _cuda_memory(torch),
        "rss_mb": _rss_mb(),
        "video_path": saved_video_path,
    }


def _extract_single_env(envs: Any, suite: str, task_id: int) -> Any:
    if hasattr(envs, "num_envs"):
        return envs
    try:
        return envs[suite][int(task_id)]
    except Exception as exc:
        raise TypeError(f"Could not extract single env for {suite}/task_{task_id} from {type(envs).__name__}") from exc


def _episode_base_record(policy_name: str, task: dict[str, Any], seed: int, planned_index: int) -> dict[str, Any]:
    return {
        "planned_episode_index": int(planned_index),
        "episode_id": f"{policy_name}|{task['suite']}|task_{task['task_id']}|seed_{seed}",
        "policy": policy_name,
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "instruction": task["instruction"],
        "reset_seed": int(seed),
    }


def run_scaleup(args: argparse.Namespace, task_manifest: dict[str, Any], episode_manifest: dict[str, Any]) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    policy_audits: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    planned_lookup = {item["episode_id"]: item["planned_episode_index"] for item in episode_manifest["episodes"]}
    policy_specs = POLICIES[: args.limit_policies] if args.limit_policies else POLICIES
    tasks = task_manifest["tasks"][: args.limit_tasks] if args.limit_tasks else task_manifest["tasks"]

    for spec in policy_specs:
        print(f"[closed-loop-scaleup] policy {spec.name}", flush=True)
        loaded = _load_policy_and_processors(args, spec)
        policy_audits[spec.name] = loaded["audit"]
        for task in tasks:
            print(f"[closed-loop-scaleup] {spec.name} {task['suite']} task_{task['task_id']}", flush=True)
            env = None
            try:
                env_cfg = _make_env_cfg(task["suite"], [int(task["task_id"])])
                env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), task["suite"], int(task["task_id"]))
                for seed in RESET_SEEDS:
                    episode_id = f"{spec.name}|{task['suite']}|task_{task['task_id']}|seed_{seed}"
                    row = _episode_base_record(spec.name, task, seed, planned_lookup[episode_id])
                    video_path = None
                    if args.capture_failure_videos:
                        video_path = Path(args.video_dir) / spec.name / task["suite"] / f"task_{task['task_id']}_seed_{seed}.mp4"
                    try:
                        trace = trace_one_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            seed=int(seed),
                            video_path=video_path if args.capture_failure_videos else None,
                        )
                        if args.capture_failure_videos and trace["success"] and trace.get("video_path"):
                            # The current trace captures before success is known.  Avoid keeping success videos.
                            Path(trace["video_path"]).unlink(missing_ok=True)
                            trace["video_path"] = None
                        row.update(trace)
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "sum_reward": None,
                                "max_reward": None,
                                "episode_length": None,
                                "termination_reason": "exception",
                                "failure_status": "exception",
                                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-24:]},
                                "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                "action_chunks_generated": None,
                                "env_steps": None,
                                "policy_latency_mean_s": None,
                                "policy_latency_max_s": None,
                                "env_step_latency_mean_s": None,
                                "env_step_latency_max_s": None,
                                "peak_vram": _cuda_memory(torch),
                                "rss_mb": _rss_mb(),
                                "video_path": None,
                            }
                        )
                        errors.append({"episode_id": episode_id, **row["exception"]})
                    rows.append(row)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded
        torch.cuda.empty_cache()

    return {
        "executed": True,
        "planned_episode_count": len(policy_specs) * len(tasks) * len(RESET_SEEDS),
        "completed_episode_count": sum(1 for row in rows if row.get("failure_status") != "exception"),
        "successful_episode_count": sum(1 for row in rows if row.get("success")),
        "infrastructure_failure_count": sum(1 for row in rows if row.get("failure_status") == "exception"),
        "episodes": rows,
        "policy_load_audits": policy_audits,
        "errors": errors,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _mean(values: list[float]) -> float | None:
    return _round(float(np.mean(values)), 6) if values else None


def _std(values: list[float]) -> float | None:
    return _round(float(np.std(values)), 6) if values else None


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return _round(float(np.corrcoef(xs, ys)[0, 1]), 6)


def _rank(values: list[float]) -> list[float]:
    order = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and order[end + 1][0] == order[cursor][0]:
            end += 1
        rank = (cursor + end) / 2.0
        for _, index in order[cursor : end + 1]:
            ranks[index] = rank
        cursor = end + 1
    return ranks


def summarize_scaleup(scaleup: dict[str, Any]) -> dict[str, Any]:
    rows = scaleup.get("episodes", [])
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy_suite: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_policy_task: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_policy[row["policy"]].append(row)
        by_policy_suite[(row["policy"], row["suite"])].append(row)
        by_policy_task[(row["policy"], row["suite"], int(row["task_id"]))].append(row)
        by_pair[(row["suite"], int(row["task_id"]), int(row["reset_seed"]))][row["policy"]] = row

    policies = [spec.name for spec in POLICIES if spec.name in by_policy]
    policy_summary = {}
    for policy in policies:
        policy_rows = by_policy[policy]
        successes = sum(1 for row in policy_rows if row.get("success"))
        total = len(policy_rows)
        task_rates = []
        for (p, _suite, _task_id), task_rows in by_policy_task.items():
            if p == policy:
                task_rates.append(sum(1 for row in task_rows if row.get("success")) / max(1, len(task_rows)))
        policy_summary[policy] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / total, 6) if total else None,
            "success_percent": _round(100 * successes / total, 3) if total else None,
            "wilson_95": wilson_interval(successes, total),
            "task_balanced_success_rate": _mean(task_rates),
            "avg_episode_length": _mean([float(row["episode_length"]) for row in policy_rows if row.get("episode_length") is not None]),
            "avg_policy_latency_s": _mean([float(row["policy_latency_mean_s"]) for row in policy_rows if row.get("policy_latency_mean_s") is not None]),
            "avg_env_step_latency_s": _mean([float(row["env_step_latency_mean_s"]) for row in policy_rows if row.get("env_step_latency_mean_s") is not None]),
            "peak_vram_mb": _round(max([float((row.get("peak_vram") or {}).get("max_allocated_mb") or 0.0) for row in policy_rows] or [0.0]), 3),
        }

    per_suite = {}
    for (policy, suite), suite_rows in sorted(by_policy_suite.items()):
        successes = sum(1 for row in suite_rows if row.get("success"))
        per_suite.setdefault(policy, {})[suite] = {
            "successes": successes,
            "total": len(suite_rows),
            "success_percent": _round(100 * successes / len(suite_rows), 3) if suite_rows else None,
        }

    per_task = {}
    for (policy, suite, task_id), task_rows in sorted(by_policy_task.items()):
        successes = sum(1 for row in task_rows if row.get("success"))
        per_task.setdefault(policy, {})[f"{suite}/task_{task_id}"] = {
            "successes": successes,
            "total": len(task_rows),
            "success_percent": _round(100 * successes / len(task_rows), 3) if task_rows else None,
        }

    reset_paired = {}
    task_paired = {}
    baseline = "frozen_base"
    for policy in policies:
        if policy == baseline:
            continue
        counts = Counter()
        for pair_rows in by_pair.values():
            base = pair_rows.get(baseline)
            other = pair_rows.get(policy)
            if not base or not other:
                continue
            base_success = bool(base.get("success"))
            other_success = bool(other.get("success"))
            if other_success and not base_success:
                counts["win"] += 1
            elif base_success and not other_success:
                counts["loss"] += 1
            else:
                counts["tie"] += 1
        reset_paired[policy] = dict(counts)

        task_counts = Counter()
        task_keys = {(suite, task_id) for (p, suite, task_id) in by_policy_task if p == baseline}
        for suite, task_id in sorted(task_keys):
            base_rows = by_policy_task.get((baseline, suite, task_id), [])
            other_rows = by_policy_task.get((policy, suite, task_id), [])
            base_successes = sum(1 for row in base_rows if row.get("success"))
            other_successes = sum(1 for row in other_rows if row.get("success"))
            if other_successes > base_successes:
                task_counts["win"] += 1
            elif other_successes < base_successes:
                task_counts["loss"] += 1
            else:
                task_counts["tie"] += 1
        task_paired[policy] = dict(task_counts)

    lora_names = [policy for policy in policies if policy.startswith("rank4_lora")]
    lora_task_spreads = []
    for suite_task in sorted({(suite, task_id) for (_p, suite, task_id) in by_policy_task}):
        rates = []
        for policy in lora_names:
            task_rows = by_policy_task.get((policy, suite_task[0], suite_task[1]), [])
            if task_rows:
                rates.append(sum(1 for row in task_rows if row.get("success")) / len(task_rows))
        if rates:
            lora_task_spreads.append(max(rates) - min(rates))

    policy_success_rates = [policy_summary[name]["success_rate"] for name in policies if policy_summary[name]["success_rate"] is not None]
    l2_values = [OFFLINE_ACTION_L2[name] for name in policies if policy_summary[name]["success_rate"] is not None]
    success_values = [float(policy_summary[name]["success_rate"]) for name in policies if policy_summary[name]["success_rate"] is not None]
    offline_online = {
        "pearson_l2_vs_success": _safe_corr(l2_values, success_values),
        "spearman_l2_vs_success": _safe_corr(_rank(l2_values), _rank(success_values)),
        "offline_l2": {name: OFFLINE_ACTION_L2[name] for name in policies},
        "success_rate": {name: policy_summary[name]["success_rate"] for name in policies},
    }
    return {
        "policy_summary": policy_summary,
        "per_suite": per_suite,
        "per_task": per_task,
        "paired_vs_frozen_base": {
            "reset_level": reset_paired,
            "task_level": task_paired,
        },
        "training_seed_variance": {
            "lora_policy_success_rates": {name: policy_summary[name]["success_rate"] for name in lora_names},
            "lora_success_rate_std": _std([float(policy_summary[name]["success_rate"]) for name in lora_names if policy_summary[name]["success_rate"] is not None]),
            "mean_task_success_spread": _mean(lora_task_spreads),
            "max_task_success_spread": _round(max(lora_task_spreads), 6) if lora_task_spreads else None,
        },
        "offline_online": offline_online,
        "overall_policy_success_std": _std([float(value) for value in policy_success_rates]),
    }


def annotate_failures(scaleup: dict[str, Any]) -> dict[str, Any]:
    rows = scaleup.get("episodes", [])
    by_pair: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(row["suite"], int(row["task_id"]), int(row["reset_seed"]))][row["policy"]] = row

    annotations = []
    review_queue = []
    for row in rows:
        if row.get("success"):
            continue
        if row.get("failure_status") == "exception":
            category = "environment_or_infrastructure_failure"
            evidence = f"Episode raised {((row.get('exception') or {}).get('type'))}: {((row.get('exception') or {}).get('message'))}"
            first_timestep = None
        elif not (row.get("action_validity") or {}).get("finite", True) or not (row.get("action_validity") or {}).get("shape_ok", True):
            category = "action_chunk_drift"
            evidence = "Action validity check failed: non-finite action or unexpected action shape."
            first_timestep = 0
        else:
            category = "ambiguous_or_unclassified"
            evidence = (
                "Official rollout recorded no success before termination, but no reliable visual or semantic phase "
                "trace is available for automatic phase labeling."
            )
            first_timestep = None
        pair_rows = by_pair[(row["suite"], int(row["task_id"]), int(row["reset_seed"]))]
        failed_policies = [policy for policy, item in pair_rows.items() if not item.get("success")]
        task_policy_rows = [
            item
            for item in rows
            if item["policy"] == row["policy"] and item["suite"] == row["suite"] and int(item["task_id"]) == int(row["task_id"])
        ]
        failed_seeds = [int(item["reset_seed"]) for item in task_policy_rows if not item.get("success")]
        base = pair_rows.get("frozen_base")
        if row["policy"] == "frozen_base" or base is None:
            lora_change = None
        elif base.get("success") and not row.get("success"):
            lora_change = "lora_introduced_failure_relative_to_frozen_base"
        elif not base.get("success") and row.get("success"):
            lora_change = "lora_resolved_failure_relative_to_frozen_base"
        elif not base.get("success") and not row.get("success"):
            lora_change = "same_pair_failed_for_frozen_base_and_lora"
        else:
            lora_change = "no_failure_change"
        annotation = {
            "episode_id": row["episode_id"],
            "policy": row["policy"],
            "suite": row["suite"],
            "task_id": int(row["task_id"]),
            "reset_seed": int(row["reset_seed"]),
            "dominant_failure_phase": category,
            "first_visible_failure_timestep": first_timestep,
            "evidence": evidence,
            "repeats_across_policies": len(failed_policies) > 1,
            "failed_policies_same_task_seed": failed_policies,
            "repeats_across_reset_seeds": len(failed_seeds) > 1,
            "failed_reset_seeds_same_policy_task": failed_seeds,
            "lora_changes_failure_phase_relative_to_frozen_base": lora_change,
            "video_path": row.get("video_path"),
            "needs_human_review": category == "ambiguous_or_unclassified",
        }
        annotations.append(annotation)
        if annotation["needs_human_review"]:
            review_queue.append(
                {
                    "episode_id": row["episode_id"],
                    "suite": row["suite"],
                    "task_id": int(row["task_id"]),
                    "policy": row["policy"],
                    "reset_seed": int(row["reset_seed"]),
                    "video_path": row.get("video_path"),
                    "reason": "automatic phase label would be unreliable",
                }
            )
    counts = Counter(item["dominant_failure_phase"] for item in annotations)
    return {
        "schema_version": 1,
        "failure_categories": list(FAILURE_CATEGORIES),
        "annotation_method": "rule-based from official rollout traces; ambiguous when no visual/semantic evidence is available",
        "failure_count": len(annotations),
        "category_counts": dict(counts),
        "annotations": annotations,
        "bounded_review_queue": review_queue,
    }


def choose_final_decision(scaleup: dict[str, Any], summary: dict[str, Any], annotations: dict[str, Any]) -> str:
    planned = int(scaleup.get("planned_episode_count") or 0)
    completed = int(scaleup.get("completed_episode_count") or 0)
    if scaleup.get("infrastructure_failure_count"):
        return "ROLLOUT_INFRASTRUCTURE_FAILURE"
    if planned and completed < planned:
        return "ROLLOUT_RESULTS_TOO_NOISY"

    category_counts = annotations.get("category_counts") or {}
    non_ambiguous = {
        key: value
        for key, value in category_counts.items()
        if key not in {"ambiguous_or_unclassified", "environment_or_infrastructure_failure"}
    }
    if non_ambiguous:
        category, count = max(non_ambiguous.items(), key=lambda item: item[1])
        if count >= max(5, 0.2 * max(1, annotations.get("failure_count", 0))):
            return "SPECIFIC_CONTROL_PHASE_FAILURE_FOUND"

    variance = summary.get("training_seed_variance") or {}
    if (variance.get("max_task_success_spread") or 0) >= 0.8 and (variance.get("mean_task_success_spread") or 0) >= 0.25:
        return "LORA_TRAINING_SEED_INSTABILITY_IS_PRIMARY_GAP"

    offline = summary.get("offline_online") or {}
    spearman = offline.get("spearman_l2_vs_success")
    if spearman is not None and spearman <= 0.0:
        return "OFFLINE_ONLINE_MISMATCH_CONFIRMED"

    if annotations.get("failure_count", 0) == 0:
        return "NO_METHOD_WORTHY_CLOSED_LOOP_GAP"
    return "OFFICIAL_BASELINE_SCALEUP_READY"


def build_markdown_reports(report: dict[str, Any], annotations: dict[str, Any], summary: dict[str, Any], args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    decision = report["final_decision"]
    policy_summary = summary["policy_summary"]
    result_lines = [
        "# Official Closed-Loop Scaleup Result",
        "",
        f"Date: {args.date} KST",
        "",
        f"- final decision: `{decision}`",
        f"- planned episodes: `{report['scaleup']['planned_episode_count']}`",
        f"- completed episodes: `{report['scaleup']['completed_episode_count']}`",
        f"- infrastructure failures: `{report['scaleup']['infrastructure_failure_count']}`",
        "",
        "## Policy Success",
        "",
        "| Policy | Successes | Total | Rate | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for policy, stats in policy_summary.items():
        result_lines.append(
            f"| `{policy}` | `{stats['successes']}` | `{stats['total']}` | `{stats['success_percent']}%` | `{stats['wilson_95']}` |"
        )
    result_lines += [
        "",
        "## Paired Difference Versus Frozen Base",
        "",
        "```json",
        json.dumps(summary["paired_vs_frozen_base"], indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Suite Success",
        "",
        "```json",
        json.dumps(summary["per_suite"], indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Task Success",
        "",
        "```json",
        json.dumps(summary["per_task"], indent=2, sort_keys=True),
        "```",
    ]
    _write_md(report_dir / "official_closed_loop_scaleup_result.md", result_lines)

    taxonomy_lines = [
        "# Official Closed-Loop Failure Taxonomy",
        "",
        f"Date: {args.date} KST",
        "",
        "## Category Counts",
        "",
        "```json",
        json.dumps(annotations.get("category_counts") or {}, indent=2, sort_keys=True),
        "```",
        "",
        "Automatic phase labels are intentionally conservative. Episodes without visual/semantic evidence are marked `ambiguous_or_unclassified` and placed in the bounded review queue.",
    ]
    _write_md(report_dir / "official_closed_loop_failure_taxonomy.md", taxonomy_lines)

    seed_lines = [
        "# Official Closed-Loop Seed Robustness",
        "",
        f"Date: {args.date} KST",
        "",
        "```json",
        json.dumps(summary["training_seed_variance"], indent=2, sort_keys=True),
        "```",
    ]
    _write_md(report_dir / "official_closed_loop_seed_robustness.md", seed_lines)

    offline_lines = [
        "# Official Closed-Loop Offline-Online Analysis",
        "",
        f"Date: {args.date} KST",
        "",
        "```json",
        json.dumps(summary["offline_online"], indent=2, sort_keys=True),
        "```",
        "",
        "Closed-loop task success is the primary evidence. Offline action L2 is reported only as a diagnostic comparison and is not used to select a LoRA seed.",
    ]
    _write_md(report_dir / "official_closed_loop_offline_online_analysis.md", offline_lines)

    method_lines = [
        "# Official Closed-Loop Method Gap Decision",
        "",
        f"Date: {args.date} KST",
        "",
        f"Final decision: `{decision}`",
        "",
        "No method is implemented in this run. Candidate directions are emitted only if a repeated, success-critical, mechanism-linked failure survives frozen-base, LoRA-seed, task, and reset explanations.",
        "",
        "## Evidence Summary",
        "",
        f"- completed episodes: `{report['scaleup']['completed_episode_count']}/{report['scaleup']['planned_episode_count']}`",
        f"- failure annotations: `{annotations.get('failure_count')}`",
        f"- category counts: `{annotations.get('category_counts')}`",
        f"- paired differences: `{summary['paired_vs_frozen_base']}`",
        f"- offline-online: `{summary['offline_online']}`",
    ]
    if decision in {
        "CLOSED_LOOP_STRUCTURED_FAILURE_READY_FOR_NOVELTY_REVIEW",
        "SPECIFIC_CONTROL_PHASE_FAILURE_FOUND",
        "LORA_TRAINING_SEED_INSTABILITY_IS_PRIMARY_GAP",
        "OFFLINE_ONLINE_MISMATCH_CONFIRMED",
        "OFFICIAL_BASELINE_SCALEUP_READY",
    }:
        method_lines += [
            "",
            "## Exact Next Step",
            "",
            "Use the bounded review queue to inspect failed episodes with official videos enabled for the strongest repeated task/reset failures, then run a novelty/method-design gate only if the visual phase evidence supports it.",
        ]
    _write_md(report_dir / "official_closed_loop_method_gap_decision.md", method_lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_env(args)
    task_manifest = build_task_manifest(args)
    episode_manifest = build_episode_manifest(args, task_manifest)
    write_plan_reports(args, task_manifest, episode_manifest)
    if args.mode == "plan":
        return {
            "schema_version": 1,
            "date": args.date,
            "mode": args.mode,
            "task_manifest": task_manifest,
            "episode_manifest": {
                key: value for key, value in episode_manifest.items() if key != "episodes"
            },
            "final_decision": None,
        }

    preflight = build_preflight(args)
    scaleup = run_scaleup(args, task_manifest, episode_manifest)
    summary = summarize_scaleup(scaleup)
    annotations = annotate_failures(scaleup)
    final_decision = choose_final_decision(scaleup, summary, annotations)
    report = {
        "schema_version": 1,
        "date": args.date,
        "mode": args.mode,
        "branch": "codex/official-smolvla-closed-loop-failure-mining",
        "preflight": preflight,
        "task_manifest_summary": {key: value for key, value in task_manifest.items() if key != "tasks"},
        "episode_manifest_summary": {key: value for key, value in episode_manifest.items() if key != "episodes"},
        "scaleup": scaleup,
        "summary": summary,
        "static_mix_duplicate_records": static_mix_duplicate_records(),
        "static_mix_duplicate_runs_skipped": True,
        "old_custom_libero_7d_route_used": False,
        "final_decision": final_decision,
    }
    report_dir = Path(args.report_dir)
    _write_json(report_dir / "official_closed_loop_scaleup_result.json", report)
    _write_json(report_dir / "official_closed_loop_failure_annotations.json", annotations)
    build_markdown_reports(report, annotations, summary, args)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "scaleup"], default="scaleup")
    parser.add_argument("--date", default="2026-07-11")
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--video-dir", default="runs/official_closed_loop_failure_videos")
    parser.add_argument("--capture-failure-videos", action="store_true")
    parser.add_argument("--limit-policies", type=int, default=0)
    parser.add_argument("--limit-tasks", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report.get("final_decision"),
                "planned": ((report.get("scaleup") or {}).get("planned_episode_count")),
                "completed": ((report.get("scaleup") or {}).get("completed_episode_count")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if args.mode == "plan" or report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
