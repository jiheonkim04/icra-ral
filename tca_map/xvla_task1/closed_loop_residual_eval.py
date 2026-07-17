"""Frozen BR-XVLA closed-loop residual-manifest screen.

This module is deliberately narrow.  It evaluates the already-trained BR-XVLA
adapter on the single predeclared shared residual identity, `20260727`, for
LIBERO-10 task 1.  It may also rerun the X-VLA prior and the uniform-weight
ablation on the same identity, but it performs no training, no optimizer step,
no checkpoint writing, and no configuration search.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.xvla_task1.data_adapter_smoke import DEFAULT_XVLA_ROOT, TASK_DESCRIPTION
from tca_map.xvla_task1.gradient_smoke import cuda_memory, nvidia_smi
from tca_map.xvla_task1.offline_validate import _default_adapter_dirs, _generate_actions
from tca_map.xvla_task1.train_lora import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_TRAINING_OUTPUT_ROOT,
    XVLA_CACHE_DIR,
    _git_commit,
    _json_default,
    _load_spec,
    _prepare_xvla_imports,
    _write_json,
)
from tca_map.xvla_task1.training_spec import MODEL_ID, MODEL_REVISION, SPEC_ARTIFACT

DEFAULT_OUTPUT_ROOT = Path("runs/xvla_prior/epoch5_br_xvla_closed_loop_residual_20260727")
DEFAULT_TASK_SUITE_NAME = "libero_10"
DEFAULT_TASK_ID = 1
DEFAULT_IDENTITY_BASE = 20260711
DEFAULT_IDENTITIES = (20260727,)
DEFAULT_POLICY_LABELS = ("xvla_prior_base", "br_xvla_primary", "uniform_xvla_ablation")
EXPECTED_INITIAL_STATE_INDICES = {20260727: 16}
PRIOR_TASK1_RESULT = Path(
    "runs/xvla_prior/diagnostic_xvla_libero10_task1_id20260724_20260731_20260717T1729KST/result.json"
)
SMOLVLA_BASE_TASK1_RESULT = Path(
    "runs/xvla_prior/diagnostic_smolvla_base_libero10_task1_id20260724_20260731_officialenv_20260717T1739KST/result.json"
)
EPS = 1e-6


@dataclass(frozen=True)
class ClosedLoopResidualConfig:
    spec_path: Path = SPEC_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    training_output_root: Path = DEFAULT_TRAINING_OUTPUT_ROOT
    primary_adapter_dir: Path | None = None
    ablation_adapter_dir: Path | None = None
    xvla_root: Path = DEFAULT_XVLA_ROOT
    task_suite: str = DEFAULT_TASK_SUITE_NAME
    task_id: int = DEFAULT_TASK_ID
    task_description: str = TASK_DESCRIPTION
    identities: tuple[int, ...] = DEFAULT_IDENTITIES
    identity_base: int = DEFAULT_IDENTITY_BASE
    policy_labels: tuple[str, ...] = DEFAULT_POLICY_LABELS
    eval_horizon: int = 900
    settle_steps: int = 10
    denoise_steps: int = 10
    device_index: int = 0
    local_files_only: bool = True


@dataclass(frozen=True)
class PolicySpec:
    label: str
    role: str
    adapter_dir: Path | None


def _sha256_or_none(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(img, 0), 1)


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_policy_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(labels) - set(DEFAULT_POLICY_LABELS))
    if unknown:
        raise ValueError(f"unknown BR-XVLA closed-loop policy labels: {unknown}")
    if not labels:
        raise ValueError("at least one policy label is required")
    return labels


def identity_to_initial_state_index(identity: int, identity_base: int = DEFAULT_IDENTITY_BASE) -> int:
    if int(identity) in EXPECTED_INITIAL_STATE_INDICES:
        return int(EXPECTED_INITIAL_STATE_INDICES[int(identity)])
    index = int(identity) - int(identity_base)
    if index < 0:
        raise ValueError(f"identity {identity} maps to negative initial-state index {index}")
    return index


def _policy_specs_from_labels(config: ClosedLoopResidualConfig, spec: dict[str, Any]) -> list[PolicySpec]:
    primary_adapter, ablation_adapter = _default_adapter_dirs(
        # _default_adapter_dirs only reads these three attributes; using config
        # directly keeps the path derivation identical to offline validation.
        config,  # type: ignore[arg-type]
        spec,
    )
    by_label = {
        "xvla_prior_base": PolicySpec("xvla_prior_base", "same_run_xvla_prior_check", None),
        "br_xvla_primary": PolicySpec("br_xvla_primary", "primary_selected_method", primary_adapter),
        "uniform_xvla_ablation": PolicySpec("uniform_xvla_ablation", "uniform_weight_ablation", ablation_adapter),
    }
    return [by_label[label] for label in config.policy_labels]


def _build_frozen_manifest(
    config: ClosedLoopResidualConfig,
    spec: dict[str, Any],
    policies: list[PolicySpec],
) -> dict[str, Any]:
    identities = [int(identity) for identity in config.identities]
    initial_state_indices = {
        str(identity): int(identity_to_initial_state_index(identity, config.identity_base)) for identity in identities
    }
    return {
        "schema_version": "2026-07-17.epoch5_br_xvla_closed_loop_residual_manifest.v1",
        "status": "FROZEN_BEFORE_RESULT_INSPECTION",
        "method": "BR-XVLA",
        "stage": "epoch_5_br_xvla_closed_loop_residual_manifest",
        "git_commit": _git_commit(),
        "created_unix": time.time(),
        "spec_path": str(config.spec_path),
        "spec_sha256": _sha256_or_none(config.spec_path),
        "xvla_root": str(config.xvla_root),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task_suite": str(config.task_suite),
        "task_id": int(config.task_id),
        "task_description": str(config.task_description),
        "reset_identities": identities,
        "initial_state_indices": initial_state_indices,
        "identity_mapping_rule": f"initial_state_index = reset_identity - {int(config.identity_base)} except frozen overrides",
        "policy_labels": [policy.label for policy in policies],
        "policy_specs": [
            {"label": policy.label, "role": policy.role, "adapter_dir": str(policy.adapter_dir) if policy.adapter_dir else None}
            for policy in policies
        ],
        "reference_artifacts": {
            "xvla_prior_task1_result": str(PRIOR_TASK1_RESULT),
            "xvla_prior_task1_result_sha256": _sha256_or_none(PRIOR_TASK1_RESULT),
            "smolvla_base_task1_result": str(SMOLVLA_BASE_TASK1_RESULT),
            "smolvla_base_task1_result_sha256": _sha256_or_none(SMOLVLA_BASE_TASK1_RESULT),
        },
        "official_protocol": {
            "environment": "libero.libero.envs.OffScreenRenderEnv",
            "camera_heights": 256,
            "camera_widths": 256,
            "absolute_controller": True,
            "controller_flag": "robot.controller.use_delta = False",
            "settle_steps": int(config.settle_steps),
            "eval_horizon": int(config.eval_horizon),
            "domain_id": 3,
            "denoise_steps": int(config.denoise_steps),
            "action_mode": "ee6d absolute EEF converted to axis-angle for LIBERO env.step",
        },
        "selection_rules": {
            "this_is_a_single_residual_manifest_screen": True,
            "identity_for_screen": 20260727,
            "retuning_from_this_result_allowed": False,
            "broader_confirmatory_evaluation_allowed_by_this_manifest": False,
            "success_decision_requires_prior_failure_reproduced": "if xvla_prior_base is included",
            "primary_beats_ablation_definition": "primary succeeds and uniform ablation fails on the same identity",
        },
        "training_happened_at_manifest_write": False,
        "optimizer_step_happened_at_manifest_write": False,
        "checkpoint_written_at_manifest_write": False,
        "closed_loop_ours_evaluation_happened_at_manifest_write": False,
    }


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
        for index in range(rotation.shape[0]):
            quat = self.T.mat2quat(rotation[index])
            axis_angles.append(self.T.quat2axisangle(quat))
        out = np.stack(axis_angles, axis=0)
        return out[0] if single else out

    @staticmethod
    def mat_to_rotate6d(rotation: np.ndarray) -> np.ndarray:
        if rotation.ndim != 2:
            raise ValueError("rotation matrix must be rank-2")
        return np.concatenate([rotation[:3, 0], rotation[:3, 1]], axis=-1)


class DirectBRXVLAPolicy:
    def __init__(self, *, config: ClosedLoopResidualConfig, policy: PolicySpec, import_report: dict[str, Any]) -> None:
        import torch
        from peft import PeftModel
        from models.modeling_xvla import XVLA  # type: ignore
        from models.processing_xvla import XVLAProcessor  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for BR-XVLA closed-loop residual screen")
        torch.cuda.set_device(int(config.device_index))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
        self.torch = torch
        self.policy = policy
        self.denoise_steps = int(config.denoise_steps)
        self.device = torch.device(f"cuda:{int(config.device_index)}")
        self.processor = XVLAProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=bool(config.local_files_only),
            cache_dir=XVLA_CACHE_DIR,
        )
        model = XVLA.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=bool(config.local_files_only),
            cache_dir=XVLA_CACHE_DIR,
        )
        if policy.adapter_dir is not None:
            model = PeftModel.from_pretrained(model, str(policy.adapter_dir), is_trainable=False)
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.action_processor = LiberoAbsActionProcessor()
        self.proprio: np.ndarray | None = None
        self.action_plan: collections.deque[list[float]] = collections.deque()
        self.policy_latencies: list[float] = []
        self.chunk_shapes: list[list[int]] = []
        self.chunk_ranges: list[dict[str, Any]] = []
        import_report["model_type"] = type(self.model).__name__
        import_report["processor_type"] = type(self.processor).__name__
        import_report["adapter_dir"] = str(policy.adapter_dir) if policy.adapter_dir else None
        import_report["cuda_memory_after_load"] = cuda_memory()

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
            _flip_agentview(np.asarray(obs["agentview_image"])),
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
            action = _generate_actions(self.model, model_inputs, steps=self.denoise_steps)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.policy_latencies.append(time.monotonic() - started)
        arr = action.float().detach().cpu().numpy().squeeze(0)
        self.chunk_shapes.append([int(dim) for dim in arr.shape])
        self.chunk_ranges.append({"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)), "finite": bool(np.isfinite(arr).all())})
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


def _evaluate_policy(
    *,
    config: ClosedLoopResidualConfig,
    policy_spec: PolicySpec,
    initial_state_indices: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    import_report = _prepare_xvla_imports(config.xvla_root)
    started = time.monotonic()
    load_started = time.monotonic()
    policy = DirectBRXVLAPolicy(config=config, policy=policy_spec, import_report=import_report)
    import_report["policy_load_seconds"] = _round_or_none(time.monotonic() - load_started, 3)

    task_suite = benchmark.get_benchmark_dict()[str(config.task_suite)]()
    task = task_suite.get_task(int(config.task_id))
    task_description = config.task_description or str(getattr(task, "language", ""))
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(int(config.task_id))
    rows: list[dict[str, Any]] = []
    env = None
    try:
        for identity in config.identities:
            row_started = time.monotonic()
            row: dict[str, Any] = {
                "policy": policy_spec.label,
                "role": policy_spec.role,
                "adapter_dir": str(policy_spec.adapter_dir) if policy_spec.adapter_dir else None,
                "reset_identity": int(identity),
                "initial_state_index": int(initial_state_indices[int(identity)]),
                "success": False,
                "completed": False,
                "exception": None,
            }
            try:
                policy.reset()
                env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
                env.seed(int(identity))
                env.reset()
                obs = env.set_init_state(initial_states[int(initial_state_indices[int(identity)])])
                for _ in range(int(config.settle_steps)):
                    obs, reward, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                for robot in env.env.robots:
                    robot.controller.use_delta = False

                final_reward = 0.0
                done_flag = False
                env_latencies: list[float] = []
                step_count = 0
                for step in range(int(config.eval_horizon)):
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
                            "mean": _round_or_none(float(np.mean(policy.policy_latencies)) if policy.policy_latencies else None),
                            "max": _round_or_none(float(np.max(policy.policy_latencies)) if policy.policy_latencies else None),
                        },
                        "environment_latency_seconds": {
                            "count": len(env_latencies),
                            "mean": _round_or_none(float(np.mean(env_latencies)) if env_latencies else None),
                            "max": _round_or_none(float(np.max(env_latencies)) if env_latencies else None),
                        },
                        "elapsed_seconds": _round_or_none(time.monotonic() - row_started, 3),
                        "cuda_memory": cuda_memory(),
                    }
                )
            except Exception as exc:  # pragma: no cover - simulator boundary
                row["completed"] = False
                row["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-40:],
                }
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
                rows.append(row)
    finally:
        try:
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    runtime = {
        **import_report,
        "elapsed_seconds": float(time.monotonic() - started),
        "cuda_memory_after_eval": cuda_memory(),
    }
    return rows, runtime


def _decide(rows_by_policy: dict[str, list[dict[str, Any]]], errors: list[dict[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    successes = {
        label: all(bool(row.get("completed")) and bool(row.get("success")) for row in rows)
        for label, rows in rows_by_policy.items()
    }
    completed = {
        label: all(bool(row.get("completed")) for row in rows)
        for label, rows in rows_by_policy.items()
    }
    prior_failure_reproduced = None
    if "xvla_prior_base" in rows_by_policy and completed.get("xvla_prior_base", False):
        prior_failure_reproduced = not successes["xvla_prior_base"]
    primary_success = successes.get("br_xvla_primary", False)
    uniform_success = successes.get("uniform_xvla_ablation")
    primary_beats_uniform = bool(primary_success and uniform_success is False)
    summary = {
        "policy_successes": successes,
        "policy_completed": completed,
        "xvla_prior_failure_reproduced": prior_failure_reproduced,
        "primary_success": bool(primary_success),
        "uniform_success": uniform_success,
        "primary_beats_uniform_ablation": primary_beats_uniform,
    }
    if errors or not all(completed.values()):
        return False, "BR_XVLA_CLOSED_LOOP_RESIDUAL_INFRASTRUCTURE_BLOCKED", summary
    if prior_failure_reproduced is False:
        return False, "BR_XVLA_CLOSED_LOOP_RESIDUAL_PRIOR_FAILURE_NOT_REPRODUCED", summary
    if primary_beats_uniform:
        return True, "BR_XVLA_CLOSED_LOOP_RESIDUAL_PASS_BEATS_ABLATION", summary
    if primary_success:
        return True, "BR_XVLA_CLOSED_LOOP_RESIDUAL_PASS_NOT_ABLATION_DECISIVE", summary
    return False, "BR_XVLA_CLOSED_LOOP_RESIDUAL_NOT_PASSED", summary


def run_closed_loop_residual_eval(config: ClosedLoopResidualConfig) -> dict[str, Any]:
    started = time.monotonic()
    config.output_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = config.output_root / "closed_loop_heartbeat.json"
    status_path = config.output_root / "closed_loop_status.json"
    manifest_path = config.output_root / "closed_loop_manifest.json"
    result_path = config.output_root / "closed_loop_result.json"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    spec = _load_spec(config.spec_path)
    policies = _policy_specs_from_labels(config, spec)
    manifest = _build_frozen_manifest(config, spec, policies)
    _write_json(manifest_path, manifest)
    initial_state_indices = {
        int(identity): int(identity_to_initial_state_index(identity, config.identity_base)) for identity in config.identities
    }
    result: dict[str, Any] = {
        "schema_version": "2026-07-17.epoch5_br_xvla_closed_loop_residual_eval.v1",
        "method": "BR-XVLA",
        "stage": "epoch_5_br_xvla_closed_loop_residual_screen",
        "status": "RUNNING",
        "success": False,
        "decision": "BR_XVLA_CLOSED_LOOP_RESIDUAL_RUNNING",
        "git_commit": _git_commit(),
        "worker_pid": os.getpid(),
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
        "spec_path": str(config.spec_path),
        "output_root": str(config.output_root),
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "closed_loop_ours_evaluation_happened": any(policy.adapter_dir is not None for policy in policies),
        "local_files_only": bool(config.local_files_only),
        "nvidia_smi_before": nvidia_smi(),
        "started_unix": time.time(),
    }
    _write_json(status_path, result)
    _write_json(heartbeat_path, {"status": "manifest_frozen", "pid": os.getpid(), "time_unix": time.time()})
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    try:
        for policy in policies:
            _write_json(
                heartbeat_path,
                {
                    "status": f"evaluating_{policy.label}",
                    "pid": os.getpid(),
                    "time_unix": time.time(),
                    "completed_policies": sorted(rows_by_policy),
                },
            )
            rows, runtime = _evaluate_policy(config=config, policy_spec=policy, initial_state_indices=initial_state_indices)
            rows_by_policy[policy.label] = rows
            runtimes[policy.label] = runtime
            errors.extend(
                {
                    "policy": policy.label,
                    "reset_identity": row.get("reset_identity"),
                    **row["exception"],
                }
                for row in rows
                if row.get("exception")
            )
            _write_json(
                status_path,
                {
                    **result,
                    "status": "RUNNING",
                    "completed_policies": sorted(rows_by_policy),
                    "episodes": rows_by_policy,
                    "errors": errors,
                },
            )
        success, decision, decision_summary = _decide(rows_by_policy, errors)
        result.update(
            {
                "status": "COMPLETE" if not errors else "FAILED",
                "success": bool(success),
                "decision": decision,
                "decision_summary": decision_summary,
                "episodes": rows_by_policy,
                "runtimes": runtimes,
                "errors": errors,
                "completed_policy_count": len(rows_by_policy),
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime boundary
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "decision": "BR_XVLA_CLOSED_LOOP_RESIDUAL_INFRASTRUCTURE_BLOCKED",
                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "episodes": rows_by_policy,
                "errors": errors,
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    finally:
        _write_json(result_path, result)
        _write_json(status_path, result)
        _write_json(
            heartbeat_path,
            {
                "status": str(result["status"]).lower(),
                "pid": os.getpid(),
                "success": bool(result.get("success", False)),
                "decision": result.get("decision"),
                "result_path": str(result_path),
                "time_unix": time.time(),
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--training-output-root", type=Path, default=DEFAULT_TRAINING_OUTPUT_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE_NAME)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--task-description", default=TASK_DESCRIPTION)
    parser.add_argument("--identities", default=",".join(str(identity) for identity in DEFAULT_IDENTITIES))
    parser.add_argument("--identity-base", type=int, default=DEFAULT_IDENTITY_BASE)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICY_LABELS))
    parser.add_argument("--eval-horizon", type=int, default=900)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    report = run_closed_loop_residual_eval(
        ClosedLoopResidualConfig(
            spec_path=args.spec,
            output_root=args.output_root,
            training_output_root=args.training_output_root,
            xvla_root=args.xvla_root,
            task_suite=str(args.task_suite),
            task_id=int(args.task_id),
            task_description=str(args.task_description),
            identities=parse_csv_ints(str(args.identities)),
            identity_base=int(args.identity_base),
            policy_labels=parse_policy_labels(str(args.policies)),
            eval_horizon=int(args.eval_horizon),
            settle_steps=int(args.settle_steps),
            denoise_steps=int(args.denoise_steps),
            device_index=int(args.device_index),
            local_files_only=not bool(args.allow_download),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
