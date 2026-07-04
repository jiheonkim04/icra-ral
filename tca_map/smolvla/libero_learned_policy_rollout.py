"""Bounded WSL SmolVLA policy rollout in a tiny LIBERO diagnostic envelope.

This entrypoint is intentionally small and conservative. It loads the local
SmolVLA policy on CPU, creates a bounded LIBERO/RoboSuite environment, runs a
few policy-controlled steps, and writes an evidence-labeled diagnostic report.
It is not benchmark evaluation, training, SOTA evidence, or a paper-grade
result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.interface_adapters import (
    ACTION_STRATEGY_GRIPPER_CLOSE,
    ACTION_STRATEGY_GRIPPER_OPEN,
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    DEFAULT_IMAGE_ALIASES,
    DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS,
    adapt_observation_state,
    adapt_policy_action_to_env_action,
    select_image_source,
)
from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)
from tca_map.smolvla.single_sample_interface_smoke import _load_policy


FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
]
MAX_RUNTIME_SECONDS = 1800
MAX_TASK_COUNT = 5
MAX_STEPS_PER_TASK = 10
PROMPT_STRATEGY_STEM_SPACES = "stem_spaces"
PROMPT_STRATEGY_BDDL_LANGUAGE = "bddl_language"
PROMPT_STRATEGY_BDDL_LANGUAGE_PERIOD = "bddl_language_period"
PROMPT_STRATEGIES = [
    PROMPT_STRATEGY_STEM_SPACES,
    PROMPT_STRATEGY_BDDL_LANGUAGE,
    PROMPT_STRATEGY_BDDL_LANGUAGE_PERIOD,
]
CAMERA_ALIAS_STRATEGY_CURRENT = "current_aliases"
CAMERA_ALIAS_STRATEGY_CAMERA3_EYE_IN_HAND = "camera3_eye_in_hand"
CAMERA_ALIAS_STRATEGY_ALL_AGENTVIEW = "all_agentview"
CAMERA_ALIAS_STRATEGIES = [
    CAMERA_ALIAS_STRATEGY_CURRENT,
    CAMERA_ALIAS_STRATEGY_CAMERA3_EYE_IN_HAND,
    CAMERA_ALIAS_STRATEGY_ALL_AGENTVIEW,
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _ensure_paths(libero_root: Path, robosuite_root: Path) -> None:
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]
    sys.path = [path for path in sys.path if not path.startswith(str(libero_root))]
    sys.path.insert(0, str(robosuite_root))
    sys.path.insert(0, str(libero_root))


def _task_files(libero_root: Path, suite: str) -> list[Path]:
    pattern = libero_root / "libero" / "libero" / "bddl_files" / suite / "*.bddl"
    return [Path(path) for path in sorted(glob.glob(str(pattern)))]


def _read_bddl_language(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            value = stripped[len("(:language") :].strip()
            if value.endswith(")"):
                value = value[:-1].strip()
            return " ".join(value.split())
    return None


def _task_language(path: Path, strategy: str = PROMPT_STRATEGY_STEM_SPACES) -> str:
    if strategy == PROMPT_STRATEGY_STEM_SPACES:
        return path.stem.replace("_", " ")
    if strategy in {PROMPT_STRATEGY_BDDL_LANGUAGE, PROMPT_STRATEGY_BDDL_LANGUAGE_PERIOD}:
        language = _read_bddl_language(path) or path.stem.replace("_", " ")
        if strategy == PROMPT_STRATEGY_BDDL_LANGUAGE_PERIOD and language and language[-1] not in ".!?":
            return language + "."
        return language
    raise ValueError(f"unsupported prompt strategy: {strategy}")


def _camera_aliases(strategy: str) -> dict[str, tuple[str, ...]]:
    aliases = {key: tuple(value) for key, value in DEFAULT_IMAGE_ALIASES.items()}
    if strategy == CAMERA_ALIAS_STRATEGY_CURRENT:
        return aliases
    if strategy == CAMERA_ALIAS_STRATEGY_CAMERA3_EYE_IN_HAND:
        aliases["observation.images.camera3"] = ("robot0_eye_in_hand_image", "agentview_image")
        aliases["observation.image3"] = ("robot0_eye_in_hand_image", "agentview_image")
        return aliases
    if strategy == CAMERA_ALIAS_STRATEGY_ALL_AGENTVIEW:
        aliases["observation.images.camera2"] = ("agentview_image", "robot0_eye_in_hand_image")
        aliases["observation.images.camera3"] = ("agentview_image", "robot0_eye_in_hand_image")
        aliases["observation.image2"] = ("agentview_image", "robot0_eye_in_hand_image")
        aliases["observation.image3"] = ("agentview_image", "robot0_eye_in_hand_image")
        return aliases
    raise ValueError(f"unsupported camera alias strategy: {strategy}")


def _state_tensor(obs: dict[str, Any], dim: int, device: str):
    import torch

    state_adapter = adapt_observation_state(
        obs,
        DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS,
        dim,
        adapter_name="diagnostic_eef_pos_quat_xyz_6d_state_adapter",
    )
    tensor = torch.tensor([state_adapter.values], dtype=torch.float32, device=device)
    return tensor, state_adapter.metadata


def _image_tensor(
    obs: dict[str, Any],
    feature_key: str,
    feature: Any,
    device: str,
    aliases: dict[str, tuple[str, ...]] | None = None,
):
    import torch
    import torch.nn.functional as F

    channels, height, width = [int(x) for x in feature.shape]
    selected = select_image_source(obs, feature_key, aliases=aliases)
    array = np.asarray(selected.value)
    source_key = selected.metadata["source_key"]

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(
            f"image source {source_key} for {feature_key} must be rank 2 or 3, got shape {list(array.shape)}"
        )

    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        chw = array.astype(np.float32)
    else:
        chw = np.transpose(array, (2, 0, 1)).astype(np.float32)
    if chw.max(initial=0.0) > 1.5:
        chw = chw / 255.0
    channel_padding_performed = chw.shape[0] < channels
    channel_truncation_performed = chw.shape[0] > channels
    if chw.shape[0] < channels:
        pad = np.zeros((channels - chw.shape[0], chw.shape[1], chw.shape[2]), dtype=np.float32)
        chw = np.concatenate([chw, pad], axis=0)
    chw = chw[:channels]
    tensor = torch.from_numpy(chw).unsqueeze(0).to(dtype=torch.float32, device=device)
    resized = tensor.shape[-2:] != (height, width)
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    metadata = dict(selected.metadata)
    metadata.update(
        {
            "input_shape": list(array.shape),
            "tensor_shape": list(tensor.shape),
            "target_shape": [1, channels, height, width],
            "resized": bool(resized),
            "channel_padding_performed": bool(channel_padding_performed),
            "channel_truncation_performed": bool(channel_truncation_performed),
            "zero_image_fallback_performed": False,
        }
    )
    return tensor, source_key, metadata


def _build_batch(
    config: Any,
    tokenizer_root: Path,
    obs: dict[str, Any],
    task: str,
    device: str,
    camera_alias_strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(getattr(config, "tokenizer_max_length", 48)),
    )

    state_dim = int(config.input_features["observation.state"].shape[0])
    state_tensor, state_metadata = _state_tensor(obs, state_dim, device)
    batch: dict[str, Any] = {
        "observation.state": state_tensor,
        "observation.language.tokens": encoded["input_ids"].to(dtype=torch.long, device=device),
        "observation.language.attention_mask": encoded["attention_mask"].to(dtype=torch.bool, device=device),
    }
    image_sources: dict[str, str | None] = {}
    image_adapters: dict[str, dict[str, Any]] = {}
    aliases = _camera_aliases(camera_alias_strategy)
    for key, feature in config.image_features.items():
        tensor, source, image_metadata = _image_tensor(obs, key, feature, device, aliases=aliases)
        batch[key] = tensor
        image_sources[key] = source
        image_adapters[key] = image_metadata

    metadata = {
        "batch_keys": sorted(batch.keys()),
        "image_sources": image_sources,
        "image_adapters": image_adapters,
        "state_dim": state_dim,
        "state_adapter": state_metadata,
        "camera_alias_strategy": camera_alias_strategy,
    }
    return batch, metadata


def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    smolvla_ckpt = Path(args.smolvla_ckpt)
    checkpoint_root = Path(args.checkpoint_root)
    hf_home = Path(args.hf_home)
    libero_root = Path(args.libero_root)
    robosuite_root = Path(args.robosuite_root)

    config_files = _find_files(smolvla_ckpt, ["config.json"])
    weight_files = _find_files(
        smolvla_ckpt,
        ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
        ["*.safetensors", "*.bin"],
    )
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    deps = _runtime_dependencies()

    report: dict[str, Any] = {
        "policy": {
            "tiny_learned_policy_libero_rollout": True,
            "diagnostic_only": True,
            "downloads_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "learned_policy_inference_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "diagnostic_rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "multi_seed_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "benchmark_claims_made": False,
            "sota_claims_made": False,
            "paper_grade_claims_made": False,
            "task_local_gate_set": (
                _env_flag("ALLOW_TINY_LEARNED_POLICY_ROLLOUT")
                or _env_flag("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX")
                or _env_flag("ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC")
                or _env_flag("ALLOW_ACTION_SCALE_DIAGNOSTIC")
                or _env_flag("ALLOW_PROMPT_FORMAT_DIAGNOSTIC")
                or _env_flag("ALLOW_CAMERA_SOURCE_DIAGNOSTIC")
            ),
            "task_local_gates_set": [
                name
                for name in [
                    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
                    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
                    "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
                    "ALLOW_ACTION_SCALE_DIAGNOSTIC",
                    "ALLOW_PROMPT_FORMAT_DIAGNOSTIC",
                    "ALLOW_CAMERA_SOURCE_DIAGNOSTIC",
                ]
                if _env_flag(name)
            ],
            "forbidden_gates_set": [name for name in FORBIDDEN_GATES if _env_flag(name)],
        },
        "risk_limits": {
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_task_count": MAX_TASK_COUNT,
            "max_steps_per_task": MAX_STEPS_PER_TASK,
            "device": args.device,
            "action_adapter_strategy": args.action_adapter_strategy,
            "action_scale": args.action_scale,
            "prompt_strategy": args.prompt_strategy,
            "camera_alias_strategy": args.camera_alias_strategy,
        },
        "paths": {
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "libero_data_root": str(args.libero_data_root),
        },
        "files": {
            "config_found": config_files,
            "weights_found": weight_files,
            "external_tokenizer_dependency": external_dependency,
            "files_ready": bool(config_files and weight_files and external_dependency["found"]),
        },
        "runtime_dependencies": deps,
        "gpu": _nvidia_smi(),
        "runtime": {
            "rss_before_mb": _rss_mb(),
            "rss_after_mb": None,
            "elapsed_sec": None,
        },
        "tasks": [],
        "result": {
            "passed": False,
            "blocked": True,
            "blocked_reason": None,
        },
        "recommended_next_step": None,
    }

    if not report["policy"]["task_local_gate_set"]:
        report["result"]["blocked_reason"] = (
            "ALLOW_TINY_LEARNED_POLICY_ROLLOUT=1 or "
            "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1 or "
            "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC=1 or "
            "ALLOW_ACTION_SCALE_DIAGNOSTIC=1 or "
            "ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1 or "
            "ALLOW_CAMERA_SOURCE_DIAGNOSTIC=1 is required for this bounded task."
        )
        return report
    if report["policy"]["forbidden_gates_set"]:
        report["result"]["blocked_reason"] = "Forbidden gate(s) set: " + ", ".join(report["policy"]["forbidden_gates_set"])
        return report
    if args.task_count < 1 or args.task_count > MAX_TASK_COUNT:
        report["result"]["blocked_reason"] = f"task_count must be between 1 and {MAX_TASK_COUNT}."
        return report
    if args.max_steps_per_task < 1 or args.max_steps_per_task > MAX_STEPS_PER_TASK:
        report["result"]["blocked_reason"] = f"max_steps_per_task must be between 1 and {MAX_STEPS_PER_TASK}."
        return report
    if args.device != "cpu":
        report["result"]["blocked_reason"] = "The first tiny learned-policy LIBERO rollout is CPU-only."
        return report
    if not report["files"]["files_ready"]:
        report["result"]["blocked_reason"] = "SmolVLA local files or tokenizer dependency are incomplete."
        return report
    if not all(deps.values()):
        missing = [name for name, present in deps.items() if not present]
        report["result"]["blocked_reason"] = "Missing runtime dependencies: " + ", ".join(missing)
        return report

    try:
        _ensure_paths(libero_root, robosuite_root)
        from libero.libero.envs import OffScreenRenderEnv

        task_files = _task_files(libero_root, args.task_suite)
        if not task_files:
            raise FileNotFoundError(f"no BDDL files found for suite {args.task_suite}")
        if args.start_task_id < 0 or args.start_task_id + args.task_count > len(task_files):
            raise ValueError("requested task range exceeds available task files")

        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = _load_policy(smolvla_ckpt, hf_home, external_dependency, args.device)
        report["policy"]["model_load_performed"] = True
        tokenizer_root = Path(external_dependency["root"])

        import torch

        for offset in range(args.task_count):
            task_started = time.monotonic()
            task_id = args.start_task_id + offset
            bddl_file = task_files[task_id]
            task_language = _task_language(bddl_file, args.prompt_strategy)
            summary: dict[str, Any] = {
                "task_id": task_id,
                "task_name": bddl_file.stem,
                "prompt_strategy": args.prompt_strategy,
                "language": task_language,
                "bddl_file": str(bddl_file),
                "env_created": False,
                "reset_ok": False,
                "steps_performed": 0,
                "policy_calls": 0,
                "reward_sum": 0.0,
                "done_seen": False,
                "success_check": None,
                "action_dim": None,
                "last_policy_action_shape": None,
                "last_env_action_preview": None,
                "last_action_adapter_metadata": None,
                "last_batch_metadata": None,
                "last_adapter_metadata": None,
                "agentview_image_shape": None,
                "agentview_image_mean": None,
                "elapsed_sec": None,
                "error": None,
            }
            env = None
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=str(bddl_file),
                    camera_heights=args.camera_size,
                    camera_widths=args.camera_size,
                )
                report["policy"]["simulator_environment_created"] = True
                summary["env_created"] = True
                env.seed(0)
                obs = env.reset()
                summary["reset_ok"] = True
                action_dim = int(getattr(env, "action_dim", 7) or 7)
                summary["action_dim"] = action_dim
                policy.reset()
                for _step in range(args.max_steps_per_task):
                    batch, batch_metadata = _build_batch(
                        config,
                        tokenizer_root,
                        obs,
                        task_language,
                        args.device,
                        args.camera_alias_strategy,
                    )
                    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=args.device)
                    infer_started = time.monotonic()
                    with torch.inference_mode():
                        policy_action = policy.select_action(batch, noise=noise)
                    report["policy"]["learned_policy_inference_performed"] = True
                    report["policy"]["model_inference_performed"] = True
                    action_adapter = adapt_policy_action_to_env_action(
                        policy_action,
                        action_dim,
                        strategy=args.action_adapter_strategy,
                        action_scale=args.action_scale,
                    )
                    env_action = action_adapter.values
                    obs, reward, done, _info = env.step(env_action)
                    summary["steps_performed"] += 1
                    summary["policy_calls"] += 1
                    summary["last_policy_action_shape"] = list(policy_action.detach().cpu().shape)
                    summary["last_env_action_preview"] = [round(float(x), 6) for x in env_action[: min(7, len(env_action))]]
                    summary["last_action_adapter_metadata"] = action_adapter.metadata
                    summary["last_batch_metadata"] = batch_metadata
                    summary["last_adapter_metadata"] = {
                        "action_adapter": action_adapter.metadata,
                        "state_adapter": batch_metadata.get("state_adapter"),
                        "image_adapters": batch_metadata.get("image_adapters"),
                    }
                    summary["last_inference_sec"] = round(time.monotonic() - infer_started, 6)
                    try:
                        summary["reward_sum"] += float(reward)
                    except Exception:
                        pass
                    summary["done_seen"] = bool(summary["done_seen"] or done)
                try:
                    summary["success_check"] = bool(env.check_success())
                except Exception:
                    summary["success_check"] = None
                image = obs.get("agentview_image") if isinstance(obs, dict) else None
                if image is not None:
                    arr = np.asarray(image)
                    summary["agentview_image_shape"] = list(arr.shape)
                    summary["agentview_image_mean"] = float(arr.mean())
            except Exception as exc:  # noqa: BLE001 - keep exact diagnostic error.
                summary["error"] = _compact_error(exc)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                summary["elapsed_sec"] = round(time.monotonic() - task_started, 3)
                report["tasks"].append(summary)

        completed = [task for task in report["tasks"] if task["error"] is None and task["steps_performed"] == args.max_steps_per_task]
        total_steps = sum(int(task["steps_performed"]) for task in report["tasks"])
        passed = (
            len(completed) == args.task_count
            and total_steps == args.task_count * args.max_steps_per_task
            and report["policy"]["model_load_performed"]
            and report["policy"]["learned_policy_inference_performed"]
            and report["policy"]["simulator_environment_created"]
            and not report["policy"]["benchmark_rollouts_performed"]
            and not report["policy"]["training_performed"]
            and not report["policy"]["openvla_oft_executed"]
            and not report["policy"]["paper_grade_claims_made"]
        )
        report["policy"]["diagnostic_rollouts_performed"] = passed
        report["result"]["passed"] = passed
        report["result"]["blocked"] = not passed
        if passed:
            report["result"]["blocked_reason"] = None
        else:
            first_error = next((task["error"] for task in report["tasks"] if task["error"] is not None), None)
            report["result"]["blocked_reason"] = f"tiny learned-policy rollout did not pass: {first_error}"
        report["result"]["tasks_completed"] = len(completed)
        report["result"]["total_steps_performed"] = total_steps
    except Exception as exc:  # noqa: BLE001 - report exact blocker.
        report["result"]["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        report["result"]["error"] = _compact_error(exc)

    report["runtime"]["rss_after_mb"] = _rss_mb()
    report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
    if report["runtime"]["elapsed_sec"] > MAX_RUNTIME_SECONDS:
        report["result"]["passed"] = False
        report["result"]["blocked"] = True
        report["result"]["blocked_reason"] = "runtime exceeded the bounded 30 minute budget"

    report["recommended_next_step"] = (
        "Plan tiny benchmark rollout metrics with strict evidence labels; do not make paper-grade claims."
        if report["result"]["passed"]
        else "Fix the reported tiny learned-policy rollout blocker before benchmark rollout or claims."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-ckpt", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--robosuite-root", required=True)
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--start-task-id", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--max-steps-per-task", type=int, default=3)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--prompt-strategy", default=PROMPT_STRATEGY_STEM_SPACES, choices=PROMPT_STRATEGIES)
    parser.add_argument(
        "--camera-alias-strategy",
        default=CAMERA_ALIAS_STRATEGY_CURRENT,
        choices=CAMERA_ALIAS_STRATEGIES,
    )
    parser.add_argument(
        "--action-adapter-strategy",
        default=ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
        choices=[
            ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
            ACTION_STRATEGY_GRIPPER_OPEN,
            ACTION_STRATEGY_GRIPPER_CLOSE,
        ],
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args(argv)

    report = run_rollout(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["result"]["passed"] else 8


if __name__ == "__main__":
    sys.exit(main())
