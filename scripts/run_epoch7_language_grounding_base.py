#!/usr/bin/env python3
"""Run the frozen Epoch 7 Base/control language-grounding panel serially.

This runner is intentionally incapable of evaluating or training Ours. It
loads one X-VLA model, creates one LIBERO environment per episode, and writes
an atomic result transaction after every completed episode.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
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

from scripts.epoch5_xvla_libero10_task8_eval import (  # noqa: E402
    DirectXVLAPolicy,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_HEAD,
)
from tca_map.epoch7_selective_language_grounding import (  # noqa: E402
    atomic_write_json,
    canonicalize_instruction,
    load_json,
    parse_bddl_instruction,
    select_pair_specs,
    summarize_episodes,
    validate_protocol,
)
from tca_map.xvla_task1.train_lora import _prepare_xvla_imports  # noqa: E402

DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_selective_language_grounding/problem_verification_protocol.json"
DEFAULT_PARA_ROOT = Path("/mnt/c/assets/repos/LIBERO-Para")
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")


class CAGDirectXVLAPolicy(DirectXVLAPolicy):
    """Training-free Counterfactual Action Guidance for the direct X-VLA path."""

    def __init__(self, denoise_steps: int, omega: float) -> None:
        super().__init__(denoise_steps=denoise_steps)
        self.omega = float(omega)
        self.guided_chunk_shapes: list[list[int]] = []
        self.guided_chunk_ranges: list[dict[str, Any]] = []
        self.cag_branch_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.guided_chunk_shapes.clear()
        self.guided_chunk_ranges.clear()
        self.cag_branch_records.clear()

    def _capture_rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": self.torch.get_rng_state(),
            "torch_cuda": self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        self.torch.set_rng_state(state["torch_cpu"])
        if self.torch.cuda.is_available() and state["torch_cuda"] is not None:
            self.torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _query_guided(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        common_state = self._capture_rng_state()
        conditional = super()._query(obs, instruction)
        self._restore_rng_state(common_state)
        unconditional = super()._query(obs, "")
        if conditional.shape != unconditional.shape:
            raise ValueError(
                f"CAG branch shape mismatch: conditional={conditional.shape}, unconditional={unconditional.shape}"
            )
        guided = conditional + self.omega * (conditional - unconditional)
        if not np.isfinite(guided).all():
            raise ValueError("CAG generated a nonfinite guided action chunk")
        self.guided_chunk_shapes.append([int(dimension) for dimension in guided.shape])
        self.guided_chunk_ranges.append(
            {
                "min": float(np.min(guided)),
                "max": float(np.max(guided)),
                "finite": True,
            }
        )
        self.cag_branch_records.append(
            {
                "conditional_min": float(np.min(conditional)),
                "conditional_max": float(np.max(conditional)),
                "unconditional_min": float(np.min(unconditional)),
                "unconditional_max": float(np.max(unconditional)),
                "mean_absolute_language_delta": float(np.mean(np.abs(conditional - unconditional))),
                "max_absolute_language_delta": float(np.max(np.abs(conditional - unconditional))),
                "guided_min": float(np.min(guided)),
                "guided_max": float(np.max(guided)),
            }
        )
        return guided

    def step(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        if not self.action_plan:
            action = self._query_guided(obs, instruction)
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


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def heartbeat(path: Path, stage: str) -> None:
    path.write_text(f"{timestamp()} {stage}\n", encoding="utf-8")


def nvidia_snapshot() -> str:
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
        return f"unavailable: {type(exc).__name__}: {exc}"


def memory_snapshot(torch_module: Any) -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0])
    except Exception:  # pragma: no cover - runtime boundary
        pass
    cuda: dict[str, Any] = {"available": bool(torch_module.cuda.is_available())}
    if cuda["available"]:
        cuda.update(
            {
                "allocated_bytes": int(torch_module.cuda.memory_allocated()),
                "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            }
        )
    return {
        "mem_total_kib": meminfo.get("MemTotal"),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "swap_total_kib": meminfo.get("SwapTotal"),
        "swap_free_kib": meminfo.get("SwapFree"),
        "swap_used_kib": (
            meminfo["SwapTotal"] - meminfo["SwapFree"]
            if "SwapTotal" in meminfo and "SwapFree" in meminfo
            else None
        ),
        "cuda": cuda,
        "nvidia_smi": nvidia_snapshot(),
    }


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def load_existing_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if payload.get("schema_version") != "epoch7.language_grounding_base.v1":
        raise ValueError(f"unexpected existing result schema in {path}")
    return payload


def build_episode_plan(
    specs: list[dict[str, Any]], role: str, para_bddl_dir: Path
) -> list[dict[str, Any]]:
    canonical_catalog = {int(spec["eval_id"]): str(spec["canonical_instruction"]) for spec in specs}
    # A restricted pair selection must still retrieve against all ten canonical
    # tasks. The caller replaces this catalog before using the control role.
    plan: list[dict[str, Any]] = []
    for spec in specs:
        paraphrase_instruction = parse_bddl_instruction(para_bddl_dir / spec["paraphrase_bddl"])
        if role in {"base", "cag"}:
            conditions = (
                ("canonical", spec["canonical_instruction"], None),
                ("paraphrase", paraphrase_instruction, None),
            )
        elif role == "control":
            retrieval = canonicalize_instruction(paraphrase_instruction, canonical_catalog)
            conditions = (("canonicalizer_control", retrieval["selected_instruction"], retrieval),)
        else:  # defensive: argparse also constrains this
            raise ValueError(f"unsupported role {role}")
        for condition, instruction, retrieval in conditions:
            plan.append(
                {
                    **spec,
                    "condition": condition,
                    "instruction": instruction,
                    "source_paraphrase_instruction": paraphrase_instruction,
                    "canonicalizer": retrieval,
                    "episode_id": f"{role}_{spec['pair_id']}_{condition}",
                }
            )
    return plan


def run_episode(
    *,
    episode: dict[str, Any],
    policy: DirectXVLAPolicy,
    torch_module: Any,
    env_class: Any,
    goal_bddl_dir: Path,
    init_dir: Path,
    resolution: int,
    settle_steps: int,
    horizon: int,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    record = dict(episode)
    record.update(
        {
            "completed": False,
            "success": False,
            "exception": None,
            "actions_executed": 0,
            "action_chunks_generated": 0,
        }
    )
    try:
        seed_everything(int(episode["model_seed"]), torch_module)
        policy.reset()
        init_states = torch_module.load(
            init_dir / f"eval{int(episode['eval_id'])}.pruned_init",
            weights_only=False,
            map_location="cpu",
        )
        state_index = int(episode["initial_state_index"])
        if state_index < 0 or state_index >= len(init_states):
            raise IndexError(f"initial-state index {state_index} outside 0..{len(init_states)-1}")
        env = env_class(
            bddl_file_name=str(goal_bddl_dir / episode["goal_bddl"]),
            camera_heights=int(resolution),
            camera_widths=int(resolution),
        )
        env.seed(int(episode["seed"]))
        env.reset()
        observation = env.set_init_state(init_states[state_index])
        dummy_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(int(settle_steps)):
            observation, _, _, _ = env.step(dummy_action)
        for robot in env.env.robots:
            robot.controller.use_delta = False

        done_seen = False
        final_reward = 0.0
        for step in range(int(horizon)):
            controller = env.env.robots[0].controller
            observation["robo_ori"] = policy.action_processor.mat_to_rotate6d(controller.ee_ori_mat)
            observation["robo_pos"] = np.asarray(controller.ee_pos, dtype=np.float32)
            action = policy.step(observation, str(episode["instruction"]))
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"invalid environment action shape/values: {action.shape}")
            observation, reward, done, _ = env.step(action)
            record["actions_executed"] = step + 1
            final_reward = float(reward)
            done_seen = done_seen or bool(done)
            if bool(env.check_success()):
                record["success"] = True
                break
        action_chunk_shapes = getattr(policy, "guided_chunk_shapes", policy.chunk_shapes)
        action_chunk_ranges = getattr(policy, "guided_chunk_ranges", policy.chunk_ranges)
        record.update(
            {
                "completed": True,
                "done_seen": done_seen,
                "final_reward": final_reward,
                "action_chunks_generated": len(action_chunk_shapes),
                "action_chunk_shapes": list(action_chunk_shapes),
                "action_chunk_ranges": list(action_chunk_ranges),
                "policy_query_seconds": {
                    "count": len(policy.policy_latencies),
                    "mean": float(np.mean(policy.policy_latencies)) if policy.policy_latencies else None,
                    "max": float(np.max(policy.policy_latencies)) if policy.policy_latencies else None,
                },
            }
        )
        if isinstance(policy, CAGDirectXVLAPolicy):
            record["cag"] = {
                "omega": policy.omega,
                "conditional_instruction": str(episode["instruction"]),
                "unconditional_instruction": "",
                "branch_queries": len(policy.chunk_shapes),
                "guided_chunks": len(policy.guided_chunk_shapes),
                "shared_rng_state_per_chunk": True,
                "branch_records": list(policy.cag_branch_records),
            }
    except Exception as exc:  # pragma: no cover - simulator boundary
        record["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        record["elapsed_seconds"] = time.monotonic() - started
        record["resource_after_episode"] = memory_snapshot(torch_module)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--para-root", type=Path, default=DEFAULT_PARA_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--role", choices=("base", "control", "cag"), default="base")
    parser.add_argument("--pair", action="append", default=[], help="Frozen pair ID such as eval0_act; repeatable.")
    parser.add_argument(
        "--condition",
        action="append",
        choices=("canonical", "paraphrase", "canonicalizer_control"),
        default=[],
        help="Limit the episode conditions for a bounded smoke; repeatable.",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

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

    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir / "result.json"
    heartbeat_path = args.run_dir / "heartbeat.txt"
    exit_code_path = args.run_dir / "exit_code.txt"
    heartbeat(heartbeat_path, "validate_protocol")

    protocol = load_json(args.protocol)
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("invalid frozen protocol: " + "; ".join(errors))
    all_specs = select_pair_specs(protocol, [], None)
    selected_specs = select_pair_specs(protocol, args.pair, args.max_pairs)
    para_bddl_dir = args.para_root / "libero/libero/bddl_files/libero_para"
    goal_bddl_dir = args.para_root / "libero/libero/bddl_files/libero_goal"
    init_dir = args.para_root / "libero/libero/init_files/libero_para"
    plan = build_episode_plan(selected_specs, args.role, para_bddl_dir)
    if args.role == "control":
        full_catalog = {int(spec["eval_id"]): str(spec["canonical_instruction"]) for spec in all_specs}
        for episode in plan:
            retrieval = canonicalize_instruction(episode["source_paraphrase_instruction"], full_catalog)
            episode["canonicalizer"] = retrieval
            episode["instruction"] = retrieval["selected_instruction"]
    if args.condition:
        requested_conditions = set(args.condition)
        legal_conditions = (
            {"canonical", "paraphrase"}
            if args.role in {"base", "cag"}
            else {"canonicalizer_control"}
        )
        if not requested_conditions <= legal_conditions:
            raise ValueError(
                f"conditions {sorted(requested_conditions)} are invalid for role {args.role}; "
                f"expected a subset of {sorted(legal_conditions)}"
            )
        plan = [episode for episode in plan if episode["condition"] in requested_conditions]

    existing = load_existing_result(result_path)
    if existing is None:
        result: dict[str, Any] = {
            "schema_version": "epoch7.language_grounding_base.v1",
            "created_at": timestamp(),
            "last_updated_at": timestamp(),
            "execution_classification": (
                "PRIOR_PROBLEM_VERIFICATION_NO_OURS"
                if args.role == "cag"
                else "BASE_PROBLEM_VERIFICATION_NO_OURS"
            ),
            "role": args.role,
            "protocol_path": str(args.protocol),
            "protocol_schema": protocol["schema_version"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_HEAD,
            "benchmark_revision": protocol["benchmark"]["revision"],
            "one_live_environment_limit": 1,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "ours_design_happened": False,
            "ours_rollout_happened": False,
            "closed_loop_rollout_happened": False,
            "selected_pair_ids": [spec["pair_id"] for spec in selected_specs],
            "episode_plan": [episode["episode_id"] for episode in plan],
            "episodes": [],
            "resource_before_load": None,
            "resource_after_load": None,
            "summary": {},
        }
        if args.role == "cag":
            result["prior"] = {
                "name": protocol["prior"]["name"],
                "reference": protocol["prior"]["reference"],
                "formula": protocol["prior"]["formula"],
                "omega": float(protocol["prior"]["omega"]),
                "shared_rng_state": True,
                "one_model_resident": True,
                "implementation": "local mechanism-faithful port; not claimed as official author code",
            }
    else:
        result = existing
        expected_plan = [episode["episode_id"] for episode in plan]
        if result.get("role") != args.role or result.get("episode_plan") != expected_plan:
            raise ValueError("existing result role/plan does not match this invocation")

    if args.dry_run:
        result["dry_run"] = True
        result["last_updated_at"] = timestamp()
        result["summary"] = summarize_episodes(result["episodes"])
        atomic_write_json(result_path, result)
        heartbeat(heartbeat_path, "dry_run_complete")
        exit_code_path.write_text("0\n", encoding="utf-8")
        return 0

    import torch

    result["resource_before_load"] = memory_snapshot(torch)
    heartbeat(heartbeat_path, "load_xvla")
    _prepare_xvla_imports(args.xvla_root)
    if args.role == "cag":
        policy = CAGDirectXVLAPolicy(
            denoise_steps=int(protocol["base"]["denoise_steps"]),
            omega=float(protocol["prior"]["omega"]),
        )
    else:
        policy = DirectXVLAPolicy(denoise_steps=int(protocol["base"]["denoise_steps"]))
    result["resource_after_load"] = memory_snapshot(torch)
    result["model_parameter_count"] = int(sum(parameter.numel() for parameter in policy.model.parameters()))
    result["processor_type"] = type(policy.processor).__name__
    result["model_type"] = type(policy.model).__name__
    result["last_updated_at"] = timestamp()
    atomic_write_json(result_path, result)

    if str(args.para_root) not in sys.path:
        sys.path.insert(0, str(args.para_root))
    from libero.libero.envs import OffScreenRenderEnv

    completed_ids = {
        str(episode["episode_id"])
        for episode in result["episodes"]
        if episode.get("completed") is True
    }
    exit_code = 0
    for episode in plan:
        if episode["episode_id"] in completed_ids:
            continue
        heartbeat(heartbeat_path, f"episode {episode['episode_id']}")
        record = run_episode(
            episode=episode,
            policy=policy,
            torch_module=torch,
            env_class=OffScreenRenderEnv,
            goal_bddl_dir=goal_bddl_dir,
            init_dir=init_dir,
            resolution=int(protocol["benchmark"]["camera_resolution"]),
            settle_steps=int(protocol["benchmark"]["settle_steps"]),
            horizon=int(protocol["benchmark"]["horizon"]),
        )
        result["episodes"].append(record)
        result["closed_loop_rollout_happened"] = True
        result["last_updated_at"] = timestamp()
        result["summary"] = summarize_episodes(result["episodes"])
        atomic_write_json(result_path, result)
        if not record["completed"]:
            exit_code = 1
            if not args.continue_on_error:
                break

    result["last_updated_at"] = timestamp()
    result["summary"] = summarize_episodes(result["episodes"])
    result["resource_at_exit"] = memory_snapshot(torch)
    result["run_complete"] = all(
        episode_id in {str(row["episode_id"]) for row in result["episodes"] if row.get("completed") is True}
        for episode_id in result["episode_plan"]
    )
    atomic_write_json(result_path, result)
    heartbeat(heartbeat_path, "complete" if result["run_complete"] else "incomplete")
    exit_code_path.write_text(f"{exit_code}\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
