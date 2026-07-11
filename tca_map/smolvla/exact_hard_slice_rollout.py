"""Exact-init SmolVLA frozen-base hard-slice runner.

This runner preserves the official LeRobot/SmolVLA policy, preprocessors,
postprocessors, and LIBERO action path. It only replaces the ambiguous reset
sequence with a fresh one-env wrapper whose episode_index is the exact frozen
LIBERO initial-state index for the current episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_closed_loop_scaleup import (
    _json_default,
    _set_runtime_env,
    trace_one_episode,
)
from tca_map.smolvla.official_wsl_libero_rollout import (
    POLICIES,
    _cuda_memory,
    _load_policy_and_processors,
)


RESET_IDENTITIES = [20260711, 20260712, 20260713, 20260714, 20260715]
HARD_SLICE_TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "hard_slice_stable_grasp"},
    {"suite": "libero_10", "task_id": 4, "role": "hard_slice_long_horizon"},
    {"suite": "libero_spatial", "task_id": 2, "role": "matched_control_spatial"},
    {"suite": "libero_10", "task_id": 2, "role": "matched_control_libero10"},
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _rss_mib() -> float | None:
    try:
        import psutil

        return _round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def _init_state_sha256(initial_state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(initial_state))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _identity_to_initial_state_index(identity: int) -> int:
    if int(identity) not in RESET_IDENTITIES:
        raise ValueError(f"unknown reset identity: {identity}")
    return RESET_IDENTITIES.index(int(identity))


def build_manifest() -> dict[str, Any]:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    episodes = []
    tasks = []
    planned_index = 0
    for task_spec in HARD_SLICE_TASKS:
        suite = str(task_spec["suite"])
        task_id = int(task_spec["task_id"])
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        tasks.append(
            {
                **task_spec,
                "instruction": str(getattr(task, "language", "")),
                "bddl_file": getattr(task, "bddl_file", None),
                "problem_folder": getattr(task, "problem_folder", None),
                "initial_state_count": int(len(initial_states)),
            }
        )
        for identity in RESET_IDENTITIES:
            init_index = _identity_to_initial_state_index(identity)
            initial_state = initial_states[init_index]
            initial_array = np.asarray(initial_state)
            episodes.append(
                {
                    "planned_episode_index": int(planned_index),
                    "episode_id": f"frozen_base|{suite}|task_{task_id}|identity_{identity}",
                    "policy": "frozen_base",
                    "suite": suite,
                    "task_id": task_id,
                    "role": str(task_spec["role"]),
                    "instruction": str(getattr(task, "language", "")),
                    "reset_identity": int(identity),
                    "initial_state_index": int(init_index),
                    "initial_state_shape": [int(dim) for dim in initial_array.shape],
                    "initial_state_dtype": str(initial_array.dtype),
                    "initial_state_sha256": _init_state_sha256(initial_state),
                }
            )
            planned_index += 1
    payload = {
        "schema_version": 1,
        "date_kst": "2026-07-11",
        "policy": "frozen_base",
        "reset_identities": list(RESET_IDENTITIES),
        "identity_mapping_rule": "reset identity labels 20260711..20260715 map to official LIBERO initial_state indices 0..4 per task",
        "exact_init_control": "fresh LiberoEnv per episode with episode_index set to initial_state_index before reset",
        "tasks": tasks,
        "planned_episode_count": len(episodes),
        "episodes": episodes,
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "canonical_payload_sha256"}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _make_exact_vector_env(suite: str, task_id: int, initial_state_index: int) -> Any:
    from gymnasium.vector import SyncVectorEnv
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()

    def make_env() -> LiberoEnv:
        return LiberoEnv(
            task_suite=task_suite,
            task_id=int(task_id),
            task_suite_name=str(suite),
            episode_index=int(initial_state_index),
            n_envs=1,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=256,
            observation_height=256,
            init_states=True,
            control_mode="relative",
        )

    return SyncVectorEnv([make_env])


def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    started = time.monotonic()
    manifest = build_manifest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "date_kst": "2026-07-11",
        "policy": "frozen_base",
        "official_path": "LeRobot SmolVLA policy, processors, postprocessors, and LIBERO env wrapper",
        "training_happened": False,
        "manifest": manifest,
        "episodes": [],
        "errors": [],
        "success": False,
    }
    loaded = None
    env = None
    identical_error_counts: dict[str, int] = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CPU_FALLBACK_BUG: CUDA unavailable before SmolVLA exact rerun")
        spec = next(item for item in POLICIES if item.name == "frozen_base")
        loaded = _load_policy_and_processors(args, spec)
        report["policy_load_audit"] = loaded["audit"]
        for planned in manifest["episodes"]:
            episode_started = time.monotonic()
            row: dict[str, Any] = {
                **planned,
                "success": False,
                "exception": None,
                "video_path": None,
                "exact_init_reset_proof": {
                    "fresh_env_per_episode": True,
                    "episode_index": int(planned["initial_state_index"]),
                    "n_envs": 1,
                    "auto_reset_after_terminal_cannot_affect_next_episode": True,
                },
            }
            try:
                env = _make_exact_vector_env(
                    str(planned["suite"]),
                    int(planned["task_id"]),
                    int(planned["initial_state_index"]),
                )
                video_path = (
                    Path(args.video_dir)
                    / "frozen_base"
                    / str(planned["suite"])
                    / f"task_{planned['task_id']}_identity_{planned['reset_identity']}.mp4"
                )
                trace = trace_one_episode(
                    env=env,
                    policy=loaded["policy"],
                    env_preprocessor=loaded["env_preprocessor"],
                    env_postprocessor=loaded["env_postprocessor"],
                    preprocessor=loaded["preprocessor"],
                    postprocessor=loaded["postprocessor"],
                    seed=int(planned["reset_identity"]),
                    video_path=video_path,
                )
                row.update(trace)
                row["cuda_memory"] = _cuda_memory(torch)
                row["rss_mib"] = _rss_mib()
                row["elapsed_seconds"] = _round(time.monotonic() - episode_started, 3)
            except Exception as exc:  # pragma: no cover - simulator boundary
                error_key = f"{type(exc).__name__}:{str(exc)[:160]}"
                identical_error_counts[error_key] = identical_error_counts.get(error_key, 0) + 1
                row["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
                row["cuda_memory"] = _cuda_memory(torch)
                row["rss_mib"] = _rss_mib()
                report["errors"].append({"episode_id": row["episode_id"], **row["exception"]})
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
                report["episodes"].append(row)
            if identical_error_counts and max(identical_error_counts.values()) >= 2:
                report["stopped_early_reason"] = "two_identical_infrastructure_failures"
                break
        report["success"] = len(report["episodes"]) == manifest["planned_episode_count"] and not report["errors"]
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        report["completed_episode_count"] = len(report.get("episodes", []))
        report["successful_episode_count"] = sum(1 for item in report.get("episodes", []) if item.get("success"))
        report["infrastructure_failure_count"] = len(report.get("errors", []))
        report["cuda_memory"] = _cuda_memory(torch)
        report["rss_mib"] = _rss_mib()
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del loaded
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest", "rollout"])
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--video-dir", default="runs/openvla_oft_int4/smolvla_exact_videos")
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "manifest":
        payload = build_manifest()
    elif args.command == "rollout":
        payload = run_rollout(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    out = Path(args.out) if args.out else Path("runs/openvla_oft_int4") / f"smolvla_exact_{args.command}.json"
    _write_json(out, payload)
    print(json.dumps({"command": args.command, "out": str(out), "success": payload.get("success")}, indent=2))
    if payload.get("exception") or payload.get("errors"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
