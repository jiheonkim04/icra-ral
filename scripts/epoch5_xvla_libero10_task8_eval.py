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
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_rgb_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Return a compact RGB uint8 copy for trace-only observability analysis."""

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {arr.shape}")
    arr = np.asarray(arr, dtype=np.uint8)
    if int(size) <= 0:
        return arr.copy()
    height, width = arr.shape[:2]
    if height == int(size) and width == int(size):
        return arr.copy()
    y_idx = np.linspace(0, height - 1, int(size)).round().astype(np.int32)
    x_idx = np.linspace(0, width - 1, int(size)).round().astype(np.int32)
    return arr[np.ix_(y_idx, x_idx)].copy()


def center_blackout_rgb(img: np.ndarray, fraction: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8).copy()
    frac = max(0.0, min(1.0, float(fraction)))
    if frac <= 0.0:
        return arr
    height, width = arr.shape[:2]
    box_h = max(1, int(round(height * frac)))
    box_w = max(1, int(round(width * frac)))
    y0 = max(0, (height - box_h) // 2)
    x0 = max(0, (width - box_w) // 2)
    arr[y0 : y0 + box_h, x0 : x0 + box_w, :] = 0
    return arr


def apply_rgb_input_perturbation(obs: dict[str, Any], mode: str, fraction: float) -> dict[str, Any]:
    """Apply legal RGB-only deployment perturbations to the policy input copy."""

    if mode == "none":
        return obs
    out = dict(obs)
    if mode == "wrist_blackout":
        out["robot0_eye_in_hand_image"] = np.zeros_like(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
    elif mode == "wrist_center_blackout":
        out["robot0_eye_in_hand_image"] = center_blackout_rgb(obs["robot0_eye_in_hand_image"], fraction)
    elif mode == "agentview_center_blackout":
        out["agentview_image"] = center_blackout_rgb(obs["agentview_image"], fraction)
    elif mode == "dual_center_blackout":
        out["agentview_image"] = center_blackout_rgb(obs["agentview_image"], fraction)
        out["robot0_eye_in_hand_image"] = center_blackout_rgb(obs["robot0_eye_in_hand_image"], fraction)
    else:
        raise ValueError(f"unknown rgb input perturbation: {mode}")
    return out


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
    parser.add_argument(
        "--rgb-input-perturbation",
        choices=["none", "wrist_blackout", "wrist_center_blackout", "agentview_center_blackout", "dual_center_blackout"],
        default="none",
        help="Optional legal RGB-only perturbation applied to policy inputs before action generation.",
    )
    parser.add_argument(
        "--rgb-input-perturbation-fraction",
        type=float,
        default=0.5,
        help="Center-box fraction for *_center_blackout modes.",
    )
    parser.add_argument(
        "--trace-dir",
        default="",
        help="Optional directory for no-training per-step legal traces. No reward/done/success is written to trace npz.",
    )
    parser.add_argument(
        "--trace-rgb-size",
        type=int,
        default=64,
        help="Nearest-neighbor RGB trace size. Use 0 to keep native resolution.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = run_dir / "heartbeat.txt"
    partial = run_dir / "partial.json"
    result_path = run_dir / "result.json"
    trace_dir = Path(args.trace_dir) if str(args.trace_dir).strip() else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

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
        "input_perturbation": {
            "mode": str(args.rgb_input_perturbation),
            "fraction": float(args.rgb_input_perturbation_fraction),
            "applied_to_policy_input_only": True,
            "simulator_state_unchanged": True,
            "privileged_state_used": False,
        },
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_design_happened": False,
        "closed_loop_rollout_happened": True,
        "trace_acquisition": {
            "enabled": trace_dir is not None,
            "trace_dir": str(trace_dir) if trace_dir is not None else None,
            "rgb_size": int(args.trace_rgb_size),
            "legal_trace_fields": [
                "step_index",
                "seconds_since_episode_start",
                "chunk_index",
                "action_index_in_chunk",
                "new_chunk_started",
                "eef_position",
                "eef_orientation_6d",
                "executed_env_action_7d",
                "policy_input_agentview_rgb",
                "wrist_rgb",
            ],
            "forbidden_inference_fields_absent_from_trace_npz": [
                "reward",
                "done",
                "success",
                "simulator_object_state",
                "simulator_contact_state",
                "privileged_object_pose",
                "future_observation",
            ],
            "metadata_fields_not_for_inference": [
                "task_suite",
                "task_id",
                "reset_identity",
                "initial_state_index",
                "frozen_prior_identity",
            ],
        },
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
                trace_step_index: list[int] = []
                trace_seconds: list[float] = []
                trace_chunk_index: list[int] = []
                trace_action_index_in_chunk: list[int] = []
                trace_new_chunk_started: list[bool] = []
                trace_eef_pos: list[np.ndarray] = []
                trace_eef_ori6d: list[np.ndarray] = []
                trace_actions: list[np.ndarray] = []
                trace_agentview: list[np.ndarray] = []
                trace_wrist: list[np.ndarray] = []
                trace_episode_start = time.monotonic()
                for step in range(int(args.eval_horizon)):
                    obs["robo_ori"] = policy.action_processor.mat_to_rotate6d(env.env.robots[0].controller.ee_ori_mat)
                    obs["robo_pos"] = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float32)
                    policy_obs = apply_rgb_input_perturbation(
                        obs,
                        str(args.rgb_input_perturbation),
                        float(args.rgb_input_perturbation_fraction),
                    )
                    new_chunk_started = not policy.action_plan
                    action = policy.step(policy_obs, task_description)
                    chunk_index = int(len(policy.chunk_shapes) - 1) if policy.chunk_shapes else -1
                    chunk_len = (
                        int(policy.chunk_shapes[chunk_index][0])
                        if chunk_index >= 0 and policy.chunk_shapes[chunk_index]
                        else 0
                    )
                    remaining_after_pop = int(len(policy.action_plan))
                    action_index_in_chunk = int(chunk_len - remaining_after_pop - 1) if chunk_len else -1
                    if trace_dir is not None:
                        trace_step_index.append(int(step))
                        trace_seconds.append(float(time.monotonic() - trace_episode_start))
                        trace_chunk_index.append(chunk_index)
                        trace_action_index_in_chunk.append(action_index_in_chunk)
                        trace_new_chunk_started.append(bool(new_chunk_started))
                        trace_eef_pos.append(np.asarray(policy_obs["robo_pos"], dtype=np.float32).copy())
                        trace_eef_ori6d.append(np.asarray(policy_obs["robo_ori"], dtype=np.float32).copy())
                        trace_actions.append(np.asarray(action, dtype=np.float32).copy())
                        trace_agentview.append(
                            resize_rgb_nearest(
                                flip_agentview(np.asarray(policy_obs["agentview_image"])),
                                int(args.trace_rgb_size),
                            )
                        )
                        trace_wrist.append(
                            resize_rgb_nearest(
                                np.asarray(policy_obs["robot0_eye_in_hand_image"]),
                                int(args.trace_rgb_size),
                            )
                        )
                    env_started = time.monotonic()
                    obs, reward, done, info = env.step(action)
                    env_latencies.append(time.monotonic() - env_started)
                    final_reward = float(reward)
                    step_count = step + 1
                    if done:
                        done_flag = True
                        break

                trace_artifact: dict[str, Any] | None = None
                if trace_dir is not None:
                    trace_path = trace_dir / f"{task_suite_name}_task{task_id}_identity{identity}_trace.npz"
                    trace_meta_path = trace_dir / f"{task_suite_name}_task{task_id}_identity{identity}_trace_manifest.json"
                    np.savez_compressed(
                        trace_path,
                        step_index=np.asarray(trace_step_index, dtype=np.int32),
                        seconds_since_episode_start=np.asarray(trace_seconds, dtype=np.float32),
                        chunk_index=np.asarray(trace_chunk_index, dtype=np.int32),
                        action_index_in_chunk=np.asarray(trace_action_index_in_chunk, dtype=np.int32),
                        new_chunk_started=np.asarray(trace_new_chunk_started, dtype=np.bool_),
                        eef_position=np.stack(trace_eef_pos).astype(np.float32),
                        eef_orientation_6d=np.stack(trace_eef_ori6d).astype(np.float32),
                        executed_env_action_7d=np.stack(trace_actions).astype(np.float32),
                        policy_input_agentview_rgb=np.stack(trace_agentview).astype(np.uint8),
                        wrist_rgb=np.stack(trace_wrist).astype(np.uint8),
                    )
                    trace_sha = sha256_file(trace_path)
                    trace_artifact = {
                        "schema_version": "2026-07-18.epoch5_xvla_legal_trace.v1",
                        "trace_npz": str(trace_path),
                        "trace_sha256": trace_sha,
                        "trace_manifest": str(trace_meta_path),
                        "trace_step_count": int(len(trace_step_index)),
                        "rgb_size": int(args.trace_rgb_size),
                        "task_suite": task_suite_name,
                        "task_id": int(task_id),
                        "reset_identity": int(identity),
                        "initial_state_index": int(index),
                        "frozen_prior_identity": "X-VLA-Libero",
                        "allowed_feature_fields": list(report["trace_acquisition"]["legal_trace_fields"]),
                        "forbidden_inference_fields_absent_from_trace_npz": list(
                            report["trace_acquisition"]["forbidden_inference_fields_absent_from_trace_npz"]
                        ),
                        "metadata_fields_not_for_inference": list(
                            report["trace_acquisition"]["metadata_fields_not_for_inference"]
                        ),
                    }
                    write_json(trace_meta_path, trace_artifact)

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
                        "trace_artifact": trace_artifact,
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
