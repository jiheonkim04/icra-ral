"""Run an Epoch 5 X-VLA prior diagnostic on LIBERO-10 task 8.

This is an official-prior execution harness, not an Ours method.  It follows
the official X-VLA LIBERO protocol closely: direct OffScreenRenderEnv,
absolute controller mode (`use_delta = False`), two camera views, controller
EEF pose as proprio, X-VLA-Libero checkpoint, and 6D-rotation action conversion.
The only narrowing is the pre-registered current residual task/reset set:
`libero_10/task_8` identities 20260716..20260723 mapped to official initial
state indices 5..12.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.xvla_task1.train_lora import XVLA_CACHE_DIR, _prepare_xvla_imports

XVLA_ROOT = "/mnt/c/assets/repos/X-VLA"
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
SOURCE_HEAD = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
IDENTITIES = list(range(20260716, 20260724))
INITIAL_STATE_INDICES = {identity: index for identity, index in zip(IDENTITIES, range(5, 13))}
DEFAULT_TASK_SUITE_NAME = "libero_10"
DEFAULT_TASK_ID = 8
DEFAULT_TASK_DESCRIPTION = "put both moka pots on the stove"
DEFAULT_IDENTITY_BASE = 20260711
EPS = 1e-6


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def beat(path: Path, stage: str) -> None:
    path.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {stage}\n", encoding="utf-8")


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - runtime boundary
        return f"nvidia_smi_failed: {type(exc).__name__}: {exc}"


def cuda_memory(torch_mod: Any) -> dict[str, Any]:
    if not torch_mod.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_bytes": int(torch_mod.cuda.memory_allocated()),
        "max_allocated_bytes": int(torch_mod.cuda.max_memory_allocated()),
        "allocated_mib": round(torch_mod.cuda.memory_allocated() / (1024**2), 3),
        "max_allocated_mib": round(torch_mod.cuda.max_memory_allocated() / (1024**2), 3),
    }


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def flip_agentview(img: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(img, 0), 1)


class LiberoAbsActionProcessor:
    def __init__(self) -> None:
        import robosuite.utils.transform_utils as transform_utils

        self.T = transform_utils

    def rotate6d_to_axisangle(self, r6d: np.ndarray) -> np.ndarray:
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True
        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)
        b3 = np.cross(b1, b2, axis=-1)
        rotation = np.stack([b1, b2, b3], axis=-1)
        axis_angles = []
        for i in range(rotation.shape[0]):
            quat = self.T.mat2quat(rotation[i])
            axis_angles.append(self.T.quat2axisangle(quat))
        out = np.stack(axis_angles, axis=0)
        return out[0] if single else out

    @staticmethod
    def mat_to_rotate6d(rotation: np.ndarray) -> np.ndarray:
        if rotation.ndim != 2:
            raise ValueError("rotation matrix must be rank-2")
        return np.concatenate([rotation[:3, 0], rotation[:3, 1]], axis=-1)


class DirectXVLAPolicy:
    def __init__(self, denoise_steps: int) -> None:
        import torch
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        self.torch = torch
        self.denoise_steps = int(denoise_steps)
        self.processor = XVLAProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=XVLA_CACHE_DIR,
        )
        self.model = XVLA.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
            cache_dir=XVLA_CACHE_DIR,
        )
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).to(torch.float32)
        self.action_processor = LiberoAbsActionProcessor()
        self.proprio: np.ndarray | None = None
        self.action_plan: collections.deque[list[float]] = collections.deque()
        self.policy_latencies: list[float] = []
        self.chunk_shapes: list[list[int]] = []
        self.chunk_ranges: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.proprio = None
        self.action_plan.clear()
        self.policy_latencies.clear()
        self.chunk_shapes.clear()
        self.chunk_ranges.clear()

    def _to_model(self, tensor: Any) -> Any:
        if tensor.is_floating_point():
            return tensor.to(device=self.device, dtype=self.torch.float32)
        return tensor.to(device=self.device)

    def _current_proprio(self, obs: dict[str, Any]) -> np.ndarray:
        pos = np.asarray(obs["robo_pos"], dtype=np.float32)
        ori6d = np.asarray(obs["robo_ori"], dtype=np.float32)
        current = np.concatenate([pos, ori6d, np.array([0.0], dtype=np.float32)], axis=-1)
        return np.concatenate([current, np.zeros_like(current)], axis=-1).astype(np.float32)

    def _query(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        images = [
            flip_agentview(np.asarray(obs["agentview_image"])),
            np.asarray(obs["robot0_eye_in_hand_image"]),
        ]
        if self.proprio is None:
            self.proprio = self._current_proprio(obs)
        inputs = self.processor(images, instruction)
        proprio = self.torch.as_tensor(self.proprio).unsqueeze(0)
        domain_id = self.torch.tensor([3], dtype=self.torch.long)
        model_inputs = {key: self._to_model(value) for key, value in inputs.items()}
        model_inputs["proprio"] = self._to_model(proprio)
        model_inputs["domain_id"] = domain_id.to(self.device)

        started = time.monotonic()
        with self.torch.no_grad():
            action = self.model.generate_actions(**model_inputs, steps=self.denoise_steps)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.policy_latencies.append(time.monotonic() - started)
        arr = action.float().detach().cpu().numpy().squeeze(0)
        self.chunk_shapes.append([int(dim) for dim in arr.shape])
        self.chunk_ranges.append(
            {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "finite": bool(np.isfinite(arr).all()),
            }
        )
        return arr

    def step(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        if not self.action_plan:
            action = self._query(obs, instruction)
            self.proprio[:9] = action[-1, :9].copy()
            target_eef = action[:, :3]
            target_axis = self.action_processor.rotate6d_to_axisangle(action[:, 3:9])
            target_grip = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_grip], axis=-1)
            for row in final_action.tolist():
                self.action_plan.append(row)
        env_action = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        env_action[-1] = 1.0 if env_action[-1] > 0.5 else -1.0
        return env_action


def parse_identities(raw: str) -> list[int]:
    if not raw:
        return list(IDENTITIES)
    out = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return out


def identity_to_initial_state_index(identity: int, identity_base: int = DEFAULT_IDENTITY_BASE) -> int:
    if identity in INITIAL_STATE_INDICES:
        return int(INITIAL_STATE_INDICES[identity])
    index = int(identity) - int(identity_base)
    if index < 0:
        raise ValueError(f"identity {identity} maps to negative initial-state index {index}")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--identities", default="")
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE_NAME)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--identity-base", type=int, default=DEFAULT_IDENTITY_BASE)
    parser.add_argument("--eval-horizon", type=int, default=900)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=10)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = run_dir / "heartbeat.txt"
    partial = run_dir / "partial.json"
    result_path = run_dir / "result.json"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("HF_HOME", "/home/jiheon/assets/checkpoints/xvla_hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    xvla_import_prepare = _prepare_xvla_imports(Path(XVLA_ROOT))

    started = time.monotonic()
    identities = parse_identities(args.identities)
    initial_state_indices = {identity: identity_to_initial_state_index(identity, args.identity_base) for identity in identities}
    task_suite_name = str(args.task_suite)
    task_id = int(args.task_id)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "third_pass_official_prior_diagnostic",
        "policy": "X-VLA-Libero",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_repo": "C:\\assets\\repos\\X-VLA",
        "source_repo_head": SOURCE_HEAD,
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task_description": args.task_description or None,
        "reset_identities": identities,
        "initial_state_indices": {str(k): int(v) for k, v in initial_state_indices.items()},
        "identity_mapping_rule": f"initial_state_index = reset_identity - {int(args.identity_base)} unless identity is in the frozen task-8 map",
        "official_protocol": {
            "environment": "libero.libero.envs.OffScreenRenderEnv",
            "camera_heights": 256,
            "camera_widths": 256,
            "absolute_controller": True,
            "controller_flag": "robot.controller.use_delta = False",
            "settle_steps": int(args.settle_steps),
            "eval_horizon": int(args.eval_horizon),
            "domain_id": 3,
            "denoise_steps": int(args.denoise_steps),
            "action_mode": "ee6d absolute EEF converted to axis-angle for LIBERO env.step",
        },
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_design_happened": False,
        "closed_loop_rollout_happened": True,
        "episodes": [],
        "errors": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "nvidia_smi_before": nvidia_smi(),
        "xvla_import_prepare": xvla_import_prepare,
    }
    env = None
    policy = None
    try:
        beat(heartbeat, "import")
        import imageio.v2 as imageio  # noqa: F401
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        report["torch"] = torch.__version__
        report["cuda_name"] = torch.cuda.get_device_name(0)
        report["cuda_memory_after_import"] = cuda_memory(torch)

        beat(heartbeat, "load_policy")
        load_started = time.monotonic()
        policy = DirectXVLAPolicy(denoise_steps=int(args.denoise_steps))
        report["policy_load_seconds"] = round_or_none(time.monotonic() - load_started, 3)
        report["model_parameter_count"] = int(sum(p.numel() for p in policy.model.parameters()))
        report["processor_type"] = type(policy.processor).__name__
        report["model_type"] = type(policy.model).__name__
        report["cuda_memory_after_load"] = cuda_memory(torch)

        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_description = args.task_description or str(getattr(task, "language", ""))
        report["task_description"] = task_description
        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        initial_states = task_suite.get_task_init_states(task_id)

        for identity in identities:
            beat(heartbeat, f"episode_{identity}")
            row_started = time.monotonic()
            index = initial_state_indices[identity]
            row: dict[str, Any] = {
                "reset_identity": int(identity),
                "initial_state_index": int(index),
                "success": False,
                "completed": False,
                "exception": None,
            }
            try:
                policy.reset()
                env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
                env.seed(int(identity))
                env.reset()
                obs = env.set_init_state(initial_states[index])
                for _ in range(int(args.settle_steps)):
                    obs, reward, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                for robot in env.env.robots:
                    robot.controller.use_delta = False

                final_reward = 0.0
                done_flag = False
                env_latencies: list[float] = []
                step_count = 0
                for step in range(int(args.eval_horizon)):
                    obs["robo_ori"] = policy.action_processor.mat_to_rotate6d(env.env.robots[0].controller.ee_ori_mat)
                    obs["robo_pos"] = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float32)
                    action = policy.step(obs, task_description)
                    env_started = time.monotonic()
                    obs, reward, done, info = env.step(action)
                    env_latencies.append(time.monotonic() - env_started)
                    final_reward = float(reward)
                    step_count = step + 1
                    if done:
                        done_flag = True
                        break

                row.update(
                    {
                        "completed": True,
                        "success": bool(done_flag),
                        "done": bool(done_flag),
                        "steps": int(step_count),
                        "final_reward": float(final_reward),
                        "action_chunk_count": int(len(policy.chunk_shapes)),
                        "action_chunk_shapes": policy.chunk_shapes,
                        "action_chunk_ranges": policy.chunk_ranges,
                        "policy_latency_seconds": {
                            "count": len(policy.policy_latencies),
                            "mean": round_or_none(float(np.mean(policy.policy_latencies)), 6)
                            if policy.policy_latencies
                            else None,
                            "max": round_or_none(float(np.max(policy.policy_latencies)), 6)
                            if policy.policy_latencies
                            else None,
                        },
                        "environment_latency_seconds": {
                            "count": len(env_latencies),
                            "mean": round_or_none(float(np.mean(env_latencies)), 6) if env_latencies else None,
                            "max": round_or_none(float(np.max(env_latencies)), 6) if env_latencies else None,
                        },
                        "elapsed_seconds": round_or_none(time.monotonic() - row_started, 3),
                        "cuda_memory": cuda_memory(torch),
                    }
                )
            except Exception as exc:  # pragma: no cover - simulator boundary
                row["completed"] = False
                row["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                }
                report["errors"].append({"reset_identity": int(identity), **row["exception"]})
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
                report["episodes"].append(row)
                write_json(partial, report)

        report["completed_episode_count"] = sum(1 for row in report["episodes"] if row.get("completed"))
        report["successful_episode_count"] = sum(1 for row in report["episodes"] if row.get("success"))
        report["infrastructure_failure_count"] = len(report["errors"])
        report["failures"] = [
            row["reset_identity"] for row in report["episodes"] if row.get("completed") and not row.get("success")
        ]
        report["completed"] = report["completed_episode_count"] == len(identities) and not report["errors"]
        report["decision"] = (
            "XVLA_PRIOR_DIAGNOSTIC_COMPLETE"
            if report["completed"]
            else "XVLA_PRIOR_DIAGNOSTIC_INFRASTRUCTURE_BLOCKED"
        )
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["completed"] = False
        report["decision"] = "XVLA_PRIOR_DIAGNOSTIC_INFRASTRUCTURE_BLOCKED"
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-60:],
        }
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del policy
            gc.collect()
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
        except Exception:
            pass
        report["elapsed_seconds"] = round_or_none(time.monotonic() - started, 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        report["nvidia_smi_after"] = nvidia_smi()
        beat(heartbeat, "finished")
        write_json(result_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
