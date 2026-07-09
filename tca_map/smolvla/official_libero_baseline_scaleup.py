"""Bounded official SmolVLA-LIBERO LoRA baseline scaleup runner.

This runner uses only the downloaded official LeRobot assets:

- lerobot/smolvla_libero
- lerobot/libero

It performs a small offline supervised LoRA baseline check. It does not run a
simulator, rollout, full benchmark, OpenVLA-OFT, or any method variant.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


FINAL_DECISIONS = {
    "READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA",
    "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO",
    "NEEDS_OFFICIAL_EVAL_OR_ROLLOUT_SETUP",
    "CPU_FALLBACK_BUG",
    "ACTION_OR_SCHEMA_MISMATCH",
    "TOO_HEAVY_LOCAL",
    "TRAINING_UNSTABLE",
}

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_CLOUD_HANDOFF",
]

DEFAULT_STEPS = 100
MAX_STEPS = 200
MAX_RUNTIME_SECONDS = 45 * 60
MAX_VRAM_MB = 14 * 1024
DEFAULT_TRAIN_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DEFAULT_EVAL_INDICES = [0, 10, 20, 30, 40]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _finite_round_list(values: Any, limit: int | None = None) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if limit is not None:
        array = array[:limit]
    return [round(float(x), 6) for x in array]


def _stat_vector(stats: dict[str, Any], key: str, field: str) -> list[float] | None:
    value = (stats.get(key) or {}).get(field)
    if value is None:
        return None
    return _finite_round_list(value)


def _safe_autocast_status(torch_mod: Any) -> dict[str, bool]:
    def enabled(device_type: str) -> bool:
        try:
            return bool(torch_mod.is_autocast_enabled(device_type))
        except TypeError:
            if device_type == "cuda":
                return bool(torch_mod.is_autocast_enabled())
            return False

    return {
        "cuda": enabled("cuda"),
        "cpu": enabled("cpu"),
    }


def _cuda_memory(torch_mod: Any) -> dict[str, float | None]:
    if not torch_mod.cuda.is_available():
        return {"allocated_mb": None, "max_allocated_mb": None}
    return {
        "allocated_mb": round(float(torch_mod.cuda.memory_allocated()) / (1024 * 1024), 3),
        "max_allocated_mb": round(float(torch_mod.cuda.max_memory_allocated()) / (1024 * 1024), 3),
    }


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024), 3)
    except Exception:
        return None


def _loss_from_output(output: Any) -> Any:
    if isinstance(output, dict):
        if "loss" in output:
            return _loss_from_output(output["loss"])
        for value in output.values():
            candidate = _loss_from_output(value)
            if _looks_scalar(candidate):
                return candidate
        return None
    if isinstance(output, (tuple, list)):
        for value in output:
            candidate = _loss_from_output(value)
            if _looks_scalar(candidate):
                return candidate
        return output[0] if output else None
    return getattr(output, "loss", output)


def _looks_scalar(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.number)):
        return True
    if hasattr(value, "numel"):
        try:
            return int(value.numel()) == 1
        except Exception:
            return False
    return False


def _to_float(value: Any) -> float:
    if value is None:
        raise TypeError("loss value is None")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _gradient_summary(policy: Any) -> dict[str, Any]:
    total_sq = 0.0
    trainable_tensors = 0
    grad_tensors = 0
    nonzero_tensors = 0
    max_abs = 0.0
    for param in policy.parameters():
        if not param.requires_grad:
            continue
        trainable_tensors += 1
        if param.grad is None:
            continue
        grad_tensors += 1
        grad = param.grad.detach().float()
        norm = float(grad.norm(2).item())
        total_sq += norm * norm
        grad_abs = float(grad.abs().max().item()) if grad.numel() else 0.0
        max_abs = max(max_abs, grad_abs)
        if grad_abs > 0:
            nonzero_tensors += 1
    return {
        "trainable_tensors": trainable_tensors,
        "grad_tensors": grad_tensors,
        "nonzero_grad_tensors": nonzero_tensors,
        "grad_norm": round(math.sqrt(total_sq), 9),
        "max_abs_grad": round(max_abs, 9),
    }


def _parameter_summary(policy: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    first_device = None
    first_dtype = None
    for param in policy.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        if first_device is None:
            first_device = str(param.device)
            first_dtype = str(param.dtype)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percent": round(100.0 * trainable / total, 6) if total else 0.0,
        "first_parameter_device": first_device,
        "first_parameter_dtype": first_dtype,
    }


def _tensor_devices(batch: dict[str, Any]) -> dict[str, str]:
    devices = {}
    for key, value in batch.items():
        if hasattr(value, "device"):
            devices[key] = str(value.device)
    return devices


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    shapes = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            shapes[key] = [int(x) for x in value.shape]
    return shapes


def _add_training_batch_dims(batch: dict[str, Any]) -> dict[str, Any]:
    action = batch.get("action")
    if action is not None and hasattr(action, "ndim") and int(action.ndim) == 2:
        batch["action"] = action.unsqueeze(0)
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is not None and hasattr(action_is_pad, "ndim") and int(action_is_pad.ndim) == 1:
        batch["action_is_pad"] = action_is_pad.unsqueeze(0)
    if "actions_id_pad" not in batch and "action_is_pad" in batch:
        batch["actions_id_pad"] = batch["action_is_pad"].clone()
    return batch


def _raw_current_action(sample: dict[str, Any]) -> np.ndarray:
    action = sample["action"]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 2:
        array = array[0]
    return array.reshape(-1)


def _postprocess_action(action: Any, postprocessor: Any) -> np.ndarray:
    processed = postprocessor(action)
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    return np.asarray(processed, dtype=np.float32).reshape(-1)


def choose_final_decision(report: dict[str, Any]) -> str:
    if report.get("cpu_fallback_occurred"):
        return "CPU_FALLBACK_BUG"
    if report.get("schema_mismatch"):
        return "ACTION_OR_SCHEMA_MISMATCH"
    if report.get("too_heavy_local"):
        return "TOO_HEAVY_LOCAL"
    if report.get("training_unstable"):
        return "TRAINING_UNSTABLE"

    training = report.get("training") or {}
    eval_after = ((report.get("evaluation") or {}).get("lora_after_training") or {})
    eval_base = ((report.get("evaluation") or {}).get("frozen_base") or {})
    metrics_available = bool(eval_after.get("action_l2_mean") is not None and eval_after.get("eval_loss_mean") is not None)
    loss_decrease = training.get("loss_decrease_fraction")
    steps_completed = int(training.get("completed_steps") or 0)
    vram_peak = (((report.get("runtime") or {}).get("cuda") or {}).get("max_allocated_mb"))
    runtime_sec = ((report.get("runtime") or {}).get("total_elapsed_sec"))

    if not training.get("training_completed"):
        return "TRAINING_UNSTABLE"
    if loss_decrease is None or loss_decrease < 0.10:
        return "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO"
    if not metrics_available:
        return "NEEDS_OFFICIAL_EVAL_OR_ROLLOUT_SETUP"
    if vram_peak is not None and float(vram_peak) > MAX_VRAM_MB:
        return "TOO_HEAVY_LOCAL"
    if runtime_sec is not None and float(runtime_sec) > MAX_RUNTIME_SECONDS:
        return "TOO_HEAVY_LOCAL"

    base_l2 = eval_base.get("action_l2_mean")
    lora_l2 = eval_after.get("action_l2_mean")
    if base_l2 is not None and lora_l2 is not None and float(lora_l2) > float(base_l2) * 1.05:
        return "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO"
    if steps_completed < 50:
        return "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO"
    return "READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    audit = report.get("dataset_audit") or {}
    training = report.get("training") or {}
    evaluation = report.get("evaluation") or {}
    runtime = report.get("runtime") or {}
    cuda = runtime.get("cuda") or {}
    policy = report.get("policy") or {}
    lines = [
        "# Official SmolVLA-LIBERO Baseline Scaleup Result",
        "",
        f"- decision: `{report.get('final_decision')}`",
        f"- status: `{report.get('status')}`",
        f"- downloads performed: `{policy.get('downloads_performed')}`",
        f"- training performed: `{policy.get('training_performed')}`",
        f"- rollouts performed: `{policy.get('rollouts_performed')}`",
        f"- OpenVLA-OFT executed: `{policy.get('openvla_oft_executed')}`",
        f"- CPU fallback occurred: `{report.get('cpu_fallback_occurred')}`",
        f"- schema mismatch: `{report.get('schema_mismatch')}`",
        "",
        "## Dataset Audit",
        "",
        f"- total episodes: `{audit.get('total_episodes')}`",
        f"- total frames: `{audit.get('total_frames')}`",
        f"- total tasks: `{audit.get('total_tasks')}`",
        f"- splits: `{audit.get('splits')}`",
        f"- action dim: `{audit.get('action_dim')}`",
        f"- state dim: `{audit.get('state_dim')}`",
        f"- image streams: `{audit.get('image_streams')}`",
        f"- data deterministic: `{audit.get('data_loading_deterministic')}`",
        f"- labels/action stats loaded: `{audit.get('labels_and_action_stats_loaded')}`",
        "",
        "## Training",
        "",
        f"- LoRA rank: `{training.get('lora_rank')}`",
        f"- batch size: `{training.get('batch_size')}`",
        f"- requested/completed steps: `{training.get('requested_steps')}` / `{training.get('completed_steps')}`",
        f"- trainable params: `{training.get('trainable_params')}`",
        f"- loss before/after: `{training.get('loss_before')}` / `{training.get('loss_after')}`",
        f"- loss decrease fraction: `{training.get('loss_decrease_fraction')}`",
        f"- last grad norm: `{training.get('last_grad_norm')}`",
        f"- steps/sec: `{training.get('steps_per_sec')}`",
        "",
        "## Evaluation",
        "",
        f"- frozen/base action L2 mean: `{(evaluation.get('frozen_base') or {}).get('action_l2_mean')}`",
        f"- LoRA action L2 mean: `{(evaluation.get('lora_after_training') or {}).get('action_l2_mean')}`",
        f"- frozen/base eval loss mean: `{(evaluation.get('frozen_base') or {}).get('eval_loss_mean')}`",
        f"- LoRA eval loss mean: `{(evaluation.get('lora_after_training') or {}).get('eval_loss_mean')}`",
        f"- LoRA action finite: `{(evaluation.get('lora_after_training') or {}).get('finite_all')}`",
        "",
        "## Runtime",
        "",
        f"- total elapsed sec: `{runtime.get('total_elapsed_sec')}`",
        f"- training elapsed sec: `{training.get('training_elapsed_sec')}`",
        f"- CUDA available: `{cuda.get('available')}`",
        f"- CUDA device: `{cuda.get('device_name')}`",
        f"- CUDA max allocated MB: `{cuda.get('max_allocated_mb')}`",
        f"- RSS final MB: `{runtime.get('rss_final_mb')}`",
        "",
        f"Exact next step: {report.get('exact_next_step')}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_policy(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    indices: list[int],
    label: str,
) -> dict[str, Any]:
    import torch

    rows = []
    losses = []
    action_l2 = []
    translation_l2 = []
    rotation_l2 = []
    gripper_abs = []
    gripper_sign = []
    finite_all = True
    action_values = []

    policy.eval()
    with torch.no_grad():
        for index in indices:
            sample = dataset[int(index)]
            batch = preprocessor(sample)
            batch = _add_training_batch_dims(batch)
            output = policy.forward(batch)
            loss = _loss_from_output(output)
            loss_float = _to_float(loss)
            losses.append(loss_float)

            if hasattr(policy, "reset"):
                policy.reset()
            selected = policy.select_action(batch)
            pred = _postprocess_action(selected, postprocessor)
            target = _raw_current_action(sample)
            min_dim = min(int(pred.shape[0]), int(target.shape[0]))
            pred = pred[:min_dim]
            target = target[:min_dim]
            finite = bool(np.isfinite(pred).all() and np.isfinite(target).all() and math.isfinite(loss_float))
            finite_all = finite_all and finite
            diff = pred - target
            row_action_l2 = float(np.linalg.norm(diff))
            row_translation_l2 = float(np.linalg.norm(diff[:3])) if min_dim >= 3 else None
            row_rotation_l2 = float(np.linalg.norm(diff[3:6])) if min_dim >= 6 else None
            row_gripper_abs = float(abs(diff[6])) if min_dim >= 7 else None
            row_gripper_sign = bool(np.sign(pred[6]) == np.sign(target[6])) if min_dim >= 7 else None
            action_l2.append(row_action_l2)
            if row_translation_l2 is not None:
                translation_l2.append(row_translation_l2)
            if row_rotation_l2 is not None:
                rotation_l2.append(row_rotation_l2)
            if row_gripper_abs is not None:
                gripper_abs.append(row_gripper_abs)
            if row_gripper_sign is not None:
                gripper_sign.append(row_gripper_sign)
            action_values.extend([float(x) for x in pred])
            rows.append(
                {
                    "index": int(index),
                    "loss": round(loss_float, 9),
                    "action_l2": round(row_action_l2, 9),
                    "translation_l2": round(row_translation_l2, 9) if row_translation_l2 is not None else None,
                    "rotation_l2": round(row_rotation_l2, 9) if row_rotation_l2 is not None else None,
                    "gripper_abs": round(row_gripper_abs, 9) if row_gripper_abs is not None else None,
                    "gripper_sign_match": row_gripper_sign,
                    "pred_preview": _finite_round_list(pred, 7),
                    "target_preview": _finite_round_list(target, 7),
                    "finite": finite,
                }
            )

    def mean_or_none(values: list[Any]) -> float | None:
        if not values:
            return None
        return round(float(np.mean(values)), 9)

    return {
        "label": label,
        "sample_count": len(rows),
        "indices": [int(i) for i in indices],
        "eval_loss_mean": mean_or_none(losses),
        "eval_loss_before_after_available": True,
        "action_l2_mean": mean_or_none(action_l2),
        "translation_l2_mean": mean_or_none(translation_l2),
        "rotation_l2_mean": mean_or_none(rotation_l2),
        "gripper_abs_mean": mean_or_none(gripper_abs),
        "gripper_sign_accuracy": mean_or_none([1.0 if x else 0.0 for x in gripper_sign]),
        "action_range": [
            round(float(np.min(action_values)), 9) if action_values else None,
            round(float(np.max(action_values)), 9) if action_values else None,
        ],
        "finite_all": finite_all,
        "rows": rows,
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    report: dict[str, Any] = {
        "status": "started",
        "final_decision": None,
        "policy": {
            "downloads_performed": False,
            "installs_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_claims_made": False,
            "custom_libero_7d_route_used": False,
            "method_variant_run": False,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "hf_home": str(Path(args.hf_home)),
            "vlm_root": str(Path(args.vlm_root)),
        },
        "risk_assessment": {
            "task": "bounded official SmolVLA-LIBERO rank-4 LoRA baseline scaleup",
            "source": "already downloaded official Hugging Face assets only",
            "new_download_expected_bytes": 0,
            "target_paths": [str(Path(args.checkpoint_path)), str(Path(args.dataset_root))],
            "expected_runtime_minutes": "under 30 preferred, 45 hard cap",
            "expected_vram_mb": "under 14336",
            "batch_size": 1,
            "rank": int(args.rank),
            "requested_steps": int(args.steps),
            "decision": "proceed if gates, CUDA, paths, and offline assets are present",
        },
        "cpu_fallback_occurred": False,
        "schema_mismatch": False,
        "too_heavy_local": False,
        "training_unstable": False,
        "errors": [],
    }

    def block(decision: str, message: str, code: int) -> tuple[dict[str, Any], int]:
        report["status"] = "blocked"
        report["final_decision"] = decision
        report["errors"].append({"message": message})
        report["runtime"] = {
            "total_elapsed_sec": round(time.monotonic() - started, 3),
        }
        return report, code

    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        return block("TRAINING_UNSTABLE", f"Forbidden gate(s) set: {', '.join(forbidden)}", 2)
    if not _env_flag("ALLOW_HEAVY_IMPORT") or not _env_flag("ALLOW_GPU_TRAINING"):
        return block("TRAINING_UNSTABLE", "Requires ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1.", 3)
    if int(args.rank) != 4:
        return block("TRAINING_UNSTABLE", "This official baseline runner is fixed to rank-4 LoRA.", 4)
    if int(args.steps) < 1 or int(args.steps) > MAX_STEPS:
        return block("TOO_HEAVY_LOCAL", f"Requested steps must be in [1, {MAX_STEPS}].", 5)

    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    for path, label in [(checkpoint_path, "checkpoint"), (dataset_root, "dataset"), (hf_home, "hf_home"), (vlm_root, "vlm_root")]:
        if not path.exists():
            return block("ACTION_OR_SCHEMA_MISMATCH", f"Missing {label} path: {path}", 6)

    try:
        import torch

        if not torch.cuda.is_available():
            report["cpu_fallback_occurred"] = True
            return block("CPU_FALLBACK_BUG", "CUDA is not available to PyTorch.", 7)
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

        import pandas as pd
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        info = _read_json(dataset_root / "meta" / "info.json")
        stats = _read_json(dataset_root / "meta" / "stats.json")
        tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
        episodes_df = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        metadata = LeRobotDatasetMetadata("lerobot/libero", root=dataset_root)

        features = info.get("features") or {}
        image_streams = {
            key: value.get("shape")
            for key, value in features.items()
            if str(value.get("dtype")) == "video" or key.startswith("observation.images.")
        }
        action_dim = int((features.get("action") or {}).get("shape", [0])[0])
        state_dim = int((features.get("observation.state") or {}).get("shape", [0])[0])
        split = info.get("splits") or {}
        train_episode = 0
        eval_episode = 1 if int(info.get("total_episodes", 0)) > 1 else 0
        delta_timestamps = {"action": [i / float(info.get("fps", 10.0)) for i in range(int(args.chunk_size))]}

        plain_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=[train_episode],
            video_backend=args.video_backend,
        )
        train_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=[train_episode],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )
        eval_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=[eval_episode],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )

        sample_a = plain_dataset[0]
        sample_b = plain_dataset[0]
        deterministic_checks = {}
        for key in ["observation.state", "action", "observation.images.image", "observation.images.image2"]:
            first = sample_a.get(key)
            second = sample_b.get(key)
            if hasattr(first, "detach") and hasattr(second, "detach"):
                deterministic_checks[key] = round(float((first.detach().cpu() - second.detach().cpu()).abs().max().item()), 9)
        deterministic = all(value == 0 for value in deterministic_checks.values())

        cfg = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            cache_dir=hf_home,
        )
        cfg.device = device
        cfg.load_vlm_weights = True
        cfg.compile_model = False
        cfg.push_to_hub = False
        cfg.vlm_model_name = str(vlm_root)
        if hasattr(cfg, "chunk_size"):
            cfg.chunk_size = int(args.chunk_size)

        policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path,
            config=cfg,
            local_files_only=True,
            cache_dir=hf_home,
            token=False,
            strict=False,
        )
        policy.to(device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
                "device_processor": {"device": device},
            },
            postprocessor_overrides={"device_processor": {"device": device}},
        )

        processor_probe_sample = train_dataset[0]
        processor_probe = _add_training_batch_dims(preprocessor(processor_probe_sample))
        processor_shapes = _tensor_shapes(processor_probe)
        processor_devices = _tensor_devices(processor_probe)

        param_summary = _parameter_summary(policy)
        cuda_device_name = torch.cuda.get_device_name(0)
        input_devices_ok = all(device_name.startswith("cuda") for device_name in processor_devices.values())
        model_on_cuda = str(param_summary["first_parameter_device"]).startswith("cuda")
        if not input_devices_ok or not model_on_cuda:
            report["cpu_fallback_occurred"] = True
            report["dataset_audit"] = {
                "processor_output_devices": processor_devices,
                "model_parameter_device": param_summary["first_parameter_device"],
            }
            return block("CPU_FALLBACK_BUG", "CUDA available but model parameters or inputs are on CPU.", 8)

        schema_mismatch = action_dim != 7 or state_dim != 8 or processor_shapes.get("action", [0, 0, 0])[-1] != 7
        labels_and_action_stats_loaded = (
            len(_stat_vector(stats, "action", "mean") or []) == 7
            and len(_stat_vector(stats, "observation.state", "mean") or []) == 8
            and bool(_stat_vector(stats, "action", "min"))
            and bool(_stat_vector(stats, "action", "max"))
        )
        report["schema_mismatch"] = bool(schema_mismatch)
        report["dataset_audit"] = {
            "metadata_loaded": isinstance(metadata, LeRobotDatasetMetadata),
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "total_tasks": int(info.get("total_tasks", 0)),
            "available_episode_rows": int(len(episodes_df)),
            "available_task_rows": int(len(tasks_df)),
            "splits": split,
            "official_eval_split_present": bool("eval" in split or "test" in split or "validation" in split),
            "mini_holdout_episode": int(eval_episode),
            "train_episode": int(train_episode),
            "episode_0_sample_count": int(len(train_dataset)),
            "eval_episode_sample_count": int(len(eval_dataset)),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "action_min_stats": _stat_vector(stats, "action", "min"),
            "action_max_stats": _stat_vector(stats, "action", "max"),
            "state_min_stats": _stat_vector(stats, "observation.state", "min"),
            "state_max_stats": _stat_vector(stats, "observation.state", "max"),
            "image_streams": image_streams,
            "task_examples": [str(value) for value in tasks_df.index[: min(5, len(tasks_df))].tolist()],
            "processor_output_shapes": processor_shapes,
            "processor_output_devices": processor_devices,
            "labels_and_action_stats_loaded": labels_and_action_stats_loaded,
            "data_loading_deterministic": deterministic,
            "determinism_max_abs_diff": deterministic_checks,
            "schema_mismatch_signs": {
                "dataset_action_not_7d": action_dim != 7,
                "dataset_state_not_8d": state_dim != 8,
                "processor_action_not_7d": processor_shapes.get("action", [0, 0, 0])[-1] != 7,
                "checkpoint_config_state_shape": list(getattr((cfg.input_features or {}).get("observation.state"), "shape", [])),
                "checkpoint_config_action_shape": list(getattr((cfg.output_features or {}).get("action"), "shape", [])),
            },
        }
        if schema_mismatch:
            return block("ACTION_OR_SCHEMA_MISMATCH", "Official dataset or processor action/state schema did not match 8D/7D expectations.", 9)

        eval_indices = [i for i in DEFAULT_EVAL_INDICES if i < len(eval_dataset)]
        train_indices = [i for i in DEFAULT_TRAIN_INDICES if i < len(train_dataset)]
        if not eval_indices or not train_indices:
            return block("ACTION_OR_SCHEMA_MISMATCH", "Not enough official samples for train/eval mini baseline.", 10)

        base_eval = _evaluate_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            indices=eval_indices,
            label="frozen_base",
        )

        policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": int(args.rank)})
        policy.to(device)
        policy.train()
        lora_param_summary = _parameter_summary(policy)
        optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))

        loss_curve = []
        grad_curve = []
        training_started = time.monotonic()
        report["policy"]["training_performed"] = True
        for step in range(int(args.steps)):
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                report["too_heavy_local"] = True
                break
            sample = train_dataset[train_indices[step % len(train_indices)]]
            batch = _add_training_batch_dims(preprocessor(sample))
            input_devices = _tensor_devices(batch)
            if not all(device_name.startswith("cuda") for device_name in input_devices.values()):
                report["cpu_fallback_occurred"] = True
                return block("CPU_FALLBACK_BUG", f"CPU fallback in training input devices: {input_devices}", 11)

            optimizer.zero_grad(set_to_none=True)
            output = policy.forward(batch)
            loss = _loss_from_output(output)
            loss_value = _to_float(loss)
            if not math.isfinite(loss_value):
                report["training_unstable"] = True
                return block("TRAINING_UNSTABLE", f"Non-finite loss at step {step}: {loss_value}", 12)
            loss.backward()
            grad_summary = _gradient_summary(policy)
            if grad_summary["grad_tensors"] == 0 or grad_summary["nonzero_grad_tensors"] == 0:
                report["training_unstable"] = True
                return block("TRAINING_UNSTABLE", f"No nonzero gradients at step {step}.", 13)
            optimizer.step()
            cuda_now = _cuda_memory(torch)
            loss_curve.append(
                {
                    "step": int(step),
                    "loss": round(loss_value, 9),
                    "cuda_allocated_mb": cuda_now["allocated_mb"],
                    "cuda_max_allocated_mb": cuda_now["max_allocated_mb"],
                }
            )
            grad_curve.append({"step": int(step), **grad_summary})

        training_elapsed = time.monotonic() - training_started
        completed_steps = len(loss_curve)
        if completed_steps == 0:
            report["training_unstable"] = True
            return block("TRAINING_UNSTABLE", "No optimizer steps completed.", 14)

        policy.eval()
        lora_eval = _evaluate_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            indices=eval_indices,
            label="lora_after_training",
        )

        loss_before = float(loss_curve[0]["loss"])
        loss_after = float(loss_curve[-1]["loss"])
        loss_decrease_fraction = (loss_before - loss_after) / max(abs(loss_before), 1e-12)
        report["training"] = {
            "lora_rank": int(args.rank),
            "batch_size": 1,
            "requested_steps": int(args.steps),
            "completed_steps": completed_steps,
            "training_completed": completed_steps == int(args.steps),
            "train_episode": int(train_episode),
            "train_indices": train_indices,
            "optimizer": "AdamW",
            "learning_rate": float(args.lr),
            "total_params": lora_param_summary["total_params"],
            "trainable_params": lora_param_summary["trainable_params"],
            "trainable_percent": lora_param_summary["trainable_percent"],
            "parameter_device": lora_param_summary["first_parameter_device"],
            "parameter_dtype": lora_param_summary["first_parameter_dtype"],
            "input_tensor_devices_first_batch": processor_devices,
            "input_tensor_shapes_first_batch": processor_shapes,
            "autocast_initial": _safe_autocast_status(torch),
            "autocast_final": _safe_autocast_status(torch),
            "loss_before": round(loss_before, 9),
            "loss_after": round(loss_after, 9),
            "loss_decrease_fraction": round(float(loss_decrease_fraction), 9),
            "loss_curve": loss_curve,
            "gradient_curve": grad_curve,
            "last_grad_norm": grad_curve[-1]["grad_norm"],
            "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"],
            "training_elapsed_sec": round(training_elapsed, 3),
            "steps_per_sec": round(completed_steps / training_elapsed, 6) if training_elapsed > 0 else None,
        }
        report["evaluation"] = {
            "eval_label": "offline_mini_holdout_from_official_train_split_not_simulator_success",
            "eval_episode": int(eval_episode),
            "eval_indices": eval_indices,
            "frozen_base": base_eval,
            "lora_after_training": lora_eval,
        }
        report["runtime"] = {
            "total_elapsed_sec": round(time.monotonic() - started, 3),
            "rss_final_mb": _rss_mb(),
            "cuda": {
                "available": bool(torch.cuda.is_available()),
                "device_name": cuda_device_name,
                **_cuda_memory(torch),
            },
        }
        report["model"] = {
            "config_device": cfg.device,
            "load_vlm_weights": cfg.load_vlm_weights,
            "vlm_model_name": cfg.vlm_model_name,
            "frozen_base_parameter_summary": param_summary,
            "lora_parameter_summary": lora_param_summary,
        }
        if (report["runtime"]["cuda"]["max_allocated_mb"] or 0) > MAX_VRAM_MB:
            report["too_heavy_local"] = True
        report["status"] = "completed"
        report["final_decision"] = choose_final_decision(report)
        if report["final_decision"] == "READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA":
            report["exact_next_step"] = (
                "Start method design only on the official SmolVLA-LIBERO path, using this rank-4 baseline "
                "as the minimum frozen/base and LoRA comparison anchor."
            )
        elif report["final_decision"] == "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO":
            report["exact_next_step"] = (
                "Run a longer official baseline reproduction with the same assets, rank-4 LoRA, batch size 1, "
                "and a predeclared mini held-out evaluation before any method work."
            )
        elif report["final_decision"] == "NEEDS_OFFICIAL_EVAL_OR_ROLLOUT_SETUP":
            report["exact_next_step"] = "Set up official WSL/Linux LIBERO eval or a stronger held-out official metric before method work."
        else:
            report["exact_next_step"] = "Fix the reported blocker before any method work."
        return report, 0 if report["final_decision"] in FINAL_DECISIONS else 20
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            report["too_heavy_local"] = True
            decision = "TOO_HEAVY_LOCAL"
        else:
            report["training_unstable"] = True
            decision = "TRAINING_UNSTABLE"
        report["status"] = "failed"
        report["final_decision"] = decision
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": message,
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 30
    except Exception as exc:
        report["training_unstable"] = True
        report["status"] = "failed"
        report["final_decision"] = "TRAINING_UNSTABLE"
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 31


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=r"C:\assets\checkpoints\smolvla_libero")
    parser.add_argument("--dataset-root", default=r"C:\assets\datasets\lerobot_libero")
    parser.add_argument("--hf-home", default=r"C:\assets\hf_home")
    parser.add_argument("--vlm-root", default=r"C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="reports/official_smolvla_libero_baseline_scaleup_result.json")
    parser.add_argument("--report-md", default="reports/official_smolvla_libero_baseline_scaleup_result.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
