"""Bounded spec-locked MPR-XVLA LoRA trainer.

This module is intentionally narrow.  It only accepts arms from the frozen
MPR-XVLA spec, uses the cached X-VLA-Libero model with official PEFT LoRA
attachment, samples task-6 clips through X-VLA's LIBERO reader contract, and
writes durable heartbeat/status/result artifacts.

Optimizer steps are allowed here only because the data-adapter and gradient
smoke gates have already passed and this module is the frozen bounded training
gate.  It performs no closed-loop Ours evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tca_map.xvla_task6.data_adapter_smoke import (
    DEFAULT_XVLA_ROOT,
    TASK_DESCRIPTION,
    _install_mmengine_fileio_shim_if_needed,
    _write_encoded_rgb_frames,
    build_abs_action_6d,
)
from tca_map.xvla_task6.gradient_smoke import (
    LOCAL_MODEL_SNAPSHOT,
    cuda_memory,
    gradient_summary,
    install_optional_server_import_shims,
    install_xvla_transformers_compat_patches,
    nvidia_smi,
    package_version,
    prepare_inputs,
    task6_phase_labels,
)
from tca_map.xvla_task6.training_spec import (
    MODEL_ID,
    MODEL_REVISION,
    SPEC_ARTIFACT,
    TASK6_HDF5_WSL,
    build_mpr_xvla_training_spec,
    validate_mpr_xvla_training_spec,
)

DEFAULT_OUTPUT_ROOT = Path("runs/xvla_prior/epoch5_mpr_xvla_training")
XVLA_CACHE_DIR = "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers"
# A 32-frame materialized clip preserves X-VLA's 30-action horizon while still
# leaving phase-2 starts in the task-6 training/validation splits.  Longer
# 96-frame clips were fine for the mug-done/pudding-remaining gradient smoke but
# can erase phase-2 candidates because completed states occur near demo endings.
DEFAULT_CLIP_STEPS = 32


@dataclass(frozen=True)
class TrainArmConfig:
    spec_path: Path = SPEC_ARTIFACT
    arm_id: str = ""
    output_root: Path = DEFAULT_OUTPUT_ROOT
    xvla_root: Path = DEFAULT_XVLA_ROOT
    hdf5_path: Path = Path(TASK6_HDF5_WSL)
    max_steps_override: int | None = None
    device_index: int = 0
    local_files_only: bool = True
    clip_steps: int = DEFAULT_CLIP_STEPS


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = str(name).rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return (int(suffix), str(name))
    return (10**9, str(name))


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else build_mpr_xvla_training_spec()
    errors = validate_mpr_xvla_training_spec(spec)
    if errors:
        raise ValueError(f"invalid MPR-XVLA training spec: {'; '.join(errors)}")
    return spec


def _arm_by_id(spec: dict[str, Any], arm_id: str) -> dict[str, Any]:
    matches = [arm for arm in spec.get("arms", []) if arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise ValueError(f"arm_id {arm_id!r} is not exactly one frozen MPR-XVLA arm")
    return matches[0]


def _phase_for_step(cycle: list[int], step_index_zero_based: int) -> int:
    if not cycle:
        raise ValueError("phase cycle must not be empty")
    return int(cycle[int(step_index_zero_based) % len(cycle)])


def task6_completed_subgoal_count(
    states: np.ndarray,
    *,
    mug_plate_xy_threshold: float,
    pudding_abs_dx_threshold: float,
    pudding_dy_min: float,
    pudding_dy_max: float,
) -> np.ndarray:
    labels = task6_phase_labels(
        states,
        mug_plate_xy_threshold=float(mug_plate_xy_threshold),
        pudding_abs_dx_threshold=float(pudding_abs_dx_threshold),
        pudding_dy_min=float(pudding_dy_min),
        pudding_dy_max=float(pudding_dy_max),
    )
    return np.asarray(labels["completed_count"], dtype=np.int64)


def build_phase_clip_index(
    hdf5_path: Path,
    *,
    demo_indices: list[int],
    clip_steps: int,
    mug_plate_xy_threshold: float,
    pudding_abs_dx_threshold: float,
    pudding_dy_min: float,
    pudding_dy_max: float,
) -> dict[int, list[dict[str, Any]]]:
    """Build candidate clip starts grouped by task-6 completed-subgoal count."""

    allowed = set(int(index) for index in demo_indices)
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    with h5py.File(hdf5_path, "r") as handle:
        names = sorted([str(name) for name in handle["data"].keys()], key=_demo_sort_key)
        for demo_index, demo_name in enumerate(names):
            if demo_index not in allowed:
                continue
            states = np.asarray(handle["data"][demo_name]["states"], dtype=np.float64)
            counts = task6_completed_subgoal_count(
                states,
                mug_plate_xy_threshold=mug_plate_xy_threshold,
                pudding_abs_dx_threshold=pudding_abs_dx_threshold,
                pudding_dy_min=pudding_dy_min,
                pudding_dy_max=pudding_dy_max,
            )
            max_start = int(states.shape[0]) - int(clip_steps)
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                phase = int(counts[start])
                grouped.setdefault(phase, []).append(
                    {
                        "demo_index": int(demo_index),
                        "demo_name": str(demo_name),
                        "source_start_index": int(start),
                        "source_end_index": int(start + clip_steps),
                        "phase_count_in_basket": int(phase),
                        "phase_count_meaning": "task6 completed-subgoal count; phase 1 is mug-done/pudding-remaining",
                    }
                )
    return grouped


def select_clip_for_step(
    grouped: dict[int, list[dict[str, Any]]],
    *,
    cycle: list[int],
    step_index_zero_based: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    preferred_phase = _phase_for_step(cycle, step_index_zero_based)
    phases = [preferred_phase] + [phase for phase in sorted(grouped) if phase != preferred_phase]
    for phase in phases:
        candidates = grouped.get(int(phase), [])
        if candidates:
            index = int(rng.integers(0, len(candidates)))
            return dict(candidates[index])
    raise RuntimeError("no MPR-XVLA training clips are available")


def materialize_xvla_clip(source_hdf5: Path, output_dir: Path, clip: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize one selected source clip into X-VLA's LIBERO reader format."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    converted = output_dir / "converted_hdf5"
    converted.mkdir(parents=True, exist_ok=True)
    demo_name = str(clip["demo_name"])
    start = int(clip["source_start_index"])
    end = int(clip["source_end_index"])
    out_hdf5 = converted / f"{demo_name}_start{start:04d}_end{end:04d}.hdf5"

    with h5py.File(source_hdf5, "r") as source:
        demo = source["data"][demo_name]
        actions = np.asarray(demo["actions"][start:end], dtype=np.float64)
        robot_states = np.asarray(demo["robot_states"][start:end], dtype=np.float64)
        states = np.asarray(demo["states"][start:end], dtype=np.float64)
        agentview = np.asarray(demo["obs"]["agentview_rgb"][start:end], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"][start:end], dtype=np.uint8)

    abs_action_6d = build_abs_action_6d(robot_states, actions)
    with h5py.File(out_hdf5, "w") as target:
        target.create_dataset("abs_action_6d", data=abs_action_6d, compression="gzip")
        _write_encoded_rgb_frames(target, "agentview_rgb", agentview)
        _write_encoded_rgb_frames(target, "eye_in_hand_rgb", wrist)
        target.create_dataset("language_instruction", data=np.bytes_(TASK_DESCRIPTION))

    meta = {
        "dataset_name": "libero",
        "datalist": [str(out_hdf5)],
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    layout = spec["data"]["phase_state_layout"]
    counts = task6_completed_subgoal_count(
        states,
        mug_plate_xy_threshold=float(layout["mug_plate_xy_threshold"]),
        pudding_abs_dx_threshold=float(layout["pudding_abs_dx_threshold"]),
        pudding_dy_min=float(layout["pudding_dy_min"]),
        pudding_dy_max=float(layout["pudding_dy_max"]),
    )
    return {
        **clip,
        "meta_path": str(meta_path),
        "clip_hdf5": str(out_hdf5),
        "clip_steps": int(end - start),
        "completed_count_first": int(counts[0]) if counts.size else None,
        "mug_done_pudding_remaining_fraction": float(np.mean(counts == 1)) if counts.size else None,
        "abs_action_6d_shape": [int(x) for x in abs_action_6d.shape],
    }


def _first_xvla_reader_sample(xvla_root: Path, meta_path: Path) -> dict[str, Any]:
    _install_mmengine_fileio_shim_if_needed()
    root = str(xvla_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    from datasets.dataset import InfiniteDataReader  # type: ignore

    reader = InfiniteDataReader(str(meta_path), num_actions=30, num_views=3, training=False, action_mode="ee6d")
    return next(iter(reader))


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _prepare_xvla_imports(xvla_root: Path) -> dict[str, Any]:
    os.environ["HF_HOME"] = "/home/jiheon/assets/checkpoints/xvla_hf_cache"
    os.environ["HF_HUB_CACHE"] = XVLA_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = XVLA_CACHE_DIR
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = str(xvla_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return {
        "optional_server_import_shims_used": install_optional_server_import_shims(),
        "runtime_dependency_versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "timm": package_version("timm"),
        },
        "transformers_compat_patches": install_xvla_transformers_compat_patches(),
    }


def _build_official_style_optimizer(model: Any, spec: dict[str, Any], arm: dict[str, Any]) -> Any:
    import torch

    shared = spec["shared_training"]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    vlm_params = list(base.vlm.parameters())
    soft_prompt_params = list(base.transformer.soft_prompt_hub.parameters())
    action_params = list(base.transformer.action_decoder.parameters()) + list(base.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [param for param in model.parameters() if id(param) not in exclude]

    def trainable(params: list[Any]) -> list[Any]:
        return [param for param in params if getattr(param, "requires_grad", False)]

    lr = float(shared["learning_rate"])
    groups = [
        {"name": "vlm", "params": trainable(vlm_params), "lr": 0.0, "weight_decay": float(shared["weight_decay"])},
        {
            "name": "transformer_core",
            "params": trainable(transformer_core_params),
            "lr": 0.0,
            "weight_decay": float(shared["weight_decay"]),
        },
        {
            "name": "soft_prompts",
            "params": trainable(soft_prompt_params),
            "lr": lr * float(shared["learning_coef"]),
            "weight_decay": float(shared["weight_decay"]),
        },
        {
            "name": "action_heads",
            "params": trainable(action_params),
            "lr": lr,
            "weight_decay": float(shared["weight_decay"]),
        },
    ]
    groups = [group for group in groups if group["params"]]
    if not groups:
        raise RuntimeError("no trainable MPR-XVLA optimizer groups found")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    optimizer._MPR_XVLA_phase_weight_lambda = float(arm["phase_weight_lambda"])  # type: ignore[attr-defined]
    return optimizer


def _load_model_processor_optimizer(config: TrainArmConfig, spec: dict[str, Any], arm: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, get_peft_model

    import_report = _prepare_xvla_imports(config.xvla_root)
    from models.modeling_xvla import XVLA  # type: ignore
    from models.processing_xvla import XVLAProcessor  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for MPR-XVLA bounded training")
    torch.cuda.set_device(int(config.device_index))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
    device = torch.device(f"cuda:{int(config.device_index)}")
    load_source = str(LOCAL_MODEL_SNAPSHOT) if LOCAL_MODEL_SNAPSHOT.exists() else MODEL_ID
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,
        "local_files_only": bool(config.local_files_only),
    }
    if load_source == MODEL_ID:
        load_kwargs["revision"] = MODEL_REVISION
        load_kwargs["cache_dir"] = XVLA_CACHE_DIR
    import_report["pretrained_load_source"] = load_source
    import_report["pretrained_load_source_is_local_snapshot"] = load_source != MODEL_ID

    model = XVLA.from_pretrained(load_source, **load_kwargs)
    processor = XVLAProcessor.from_pretrained(load_source, **load_kwargs)
    lora = spec["shared_training"]["official_lora_config"]
    model = get_peft_model(
        model,
        LoraConfig(
            lora_alpha=int(lora["lora_alpha"]),
            r=int(lora["r"]),
            bias=str(lora["bias"]),
            target_modules=str(lora["target_modules"]),
            modules_to_save=list(lora["modules_to_save"]),
        ),
    )
    model.to(device=device, dtype=torch.float32)
    model.train()
    optimizer = _build_official_style_optimizer(model, spec, arm)
    import_report["cuda_memory_after_load"] = cuda_memory()
    import_report["optimizer_group_lrs"] = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
    return model, processor, optimizer, import_report


def run_training_arm(config: TrainArmConfig) -> dict[str, Any]:
    started = time.monotonic()
    spec = _load_spec(config.spec_path)
    arm = _arm_by_id(spec, config.arm_id)
    shared = spec["shared_training"]
    max_steps = int(config.max_steps_override or shared["max_optimizer_steps"])
    if max_steps > int(shared["max_optimizer_steps"]):
        raise ValueError("max_steps_override cannot exceed frozen spec max_optimizer_steps")

    run_dir = config.output_root / config.arm_id
    heartbeat_path = run_dir / "heartbeat.json"
    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_root = run_dir / "checkpoints"
    work_clip_dir = run_dir / "working_clip"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "frozen_spec_snapshot.json", spec)

    result: dict[str, Any] = {
        "schema_version": "2026-07-17.epoch5_MPR_XVLA_bounded_training.v1",
        "method": "MPR-XVLA",
        "arm_id": config.arm_id,
        "role": arm["role"],
        "status": "RUNNING",
        "success": False,
        "decision": "MPR_XVLA_TRAINING_RUNNING",
        "training_happened": False,
        "optimizer_created": False,
        "optimizer_steps_completed": 0,
        "checkpoint_written": False,
        "closed_loop_ours_evaluation_happened": False,
        "phase_weight_lambda": float(arm["phase_weight_lambda"]),
        "max_steps": int(max_steps),
        "spec_path": str(config.spec_path),
        "spec_freeze_id": spec["freeze_id"],
        "local_files_only": bool(config.local_files_only),
        "git_commit": _git_commit(),
        "run_dir": str(run_dir),
        "worker_pid": os.getpid(),
        "started_unix": time.time(),
        "nvidia_smi_before": nvidia_smi(),
        "resume_command": (
            f"wsl -d Ubuntu-22.04 --cd /mnt/c/Users/jiheo/tca_map -- "
            f"/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python "
            f"-m tca_map.xvla_task6.train_lora --spec {config.spec_path.as_posix()} "
            f"--arm-id {config.arm_id} --output-root {config.output_root.as_posix()}"
        ),
    }
    _write_json(status_path, result)
    _write_json(heartbeat_path, {"status": "initializing", "pid": os.getpid(), "arm_id": config.arm_id, "time_unix": time.time()})

    model = None
    try:
        _seed_everything(int(shared["seed"]))
        layout = spec["data"]["phase_state_layout"]
        grouped = build_phase_clip_index(
            config.hdf5_path,
            demo_indices=list(spec["data"]["train_demo_indices"]),
            clip_steps=int(config.clip_steps),
            mug_plate_xy_threshold=float(layout["mug_plate_xy_threshold"]),
            pudding_abs_dx_threshold=float(layout["pudding_abs_dx_threshold"]),
            pudding_dy_min=float(layout["pudding_dy_min"]),
            pudding_dy_max=float(layout["pudding_dy_max"]),
        )
        if not all(grouped.get(phase) for phase in (0, 1, 2)):
            raise RuntimeError("training split must contain phase 0/1/2 clips")
        rng = np.random.default_rng(int(shared["seed"]))
        cycle = list(arm["sampler"]["cycle_phase_counts"])
        selected_clips = [
            select_clip_for_step(grouped, cycle=cycle, step_index_zero_based=step, rng=rng)
            for step in range(max_steps)
        ]
        _write_json(
            run_dir / "selection_manifest.json",
            {
                "arm_id": config.arm_id,
                "cycle": cycle,
                "phase_candidate_counts": {str(phase): len(rows) for phase, rows in grouped.items()},
                "selected_clips": selected_clips,
                "confirmatory_residual_resets_used_for_sampling": False,
            },
        )
        _write_json(heartbeat_path, {"status": "load_model", "pid": os.getpid(), "arm_id": config.arm_id, "time_unix": time.time()})
        model, processor, optimizer, load_report = _load_model_processor_optimizer(config, spec, arm)
        result.update(load_report)
        result["optimizer_created"] = True

        import torch

        device = torch.device(f"cuda:{int(config.device_index)}")
        save_steps = {int(step) for step in shared["save_steps"] if int(step) <= max_steps}
        last_metric: dict[str, Any] | None = None
        for step_index, clip in enumerate(selected_clips):
            step = step_index + 1
            materialized = materialize_xvla_clip(config.hdf5_path, work_clip_dir, clip, spec)
            sample = _first_xvla_reader_sample(config.xvla_root, Path(materialized["meta_path"]))
            inputs = prepare_inputs(sample, processor, device, torch.float32)
            shutil.rmtree(work_clip_dir, ignore_errors=True)

            loss_dict = model(**inputs)
            base_loss = sum(loss_dict.values())
            phase_weight = 1.0 + float(arm["phase_weight_lambda"]) * float(int(clip["phase_count_in_basket"]) == 1)
            weighted_loss = base_loss * phase_weight
            if not torch.isfinite(weighted_loss):
                raise RuntimeError(f"nonfinite weighted loss at step {step}")
            weighted_loss.backward()
            grad = gradient_summary(model)
            if int(grad["finite_grad_tensor_count"]) != int(grad["grad_tensor_count"]) or int(grad["nonzero_grad_tensor_count"]) <= 0:
                raise RuntimeError(f"invalid gradients at step {step}")
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(shared["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            result["training_happened"] = True
            result["optimizer_steps_completed"] = int(step)
            cuda = cuda_memory()
            if float(cuda.get("max_allocated_mib", 0.0)) > float(shared["max_cuda_peak_mib"]):
                raise RuntimeError(f"cuda peak exceeded frozen limit at step {step}: {cuda}")
            losses = {key: float(value.detach().float().item()) for key, value in loss_dict.items()}
            losses["loss_total"] = float(base_loss.detach().float().item())
            losses["phase_weight"] = float(phase_weight)
            losses["weighted_loss"] = float(weighted_loss.detach().float().item())
            last_metric = {
                "step": int(step),
                "clip": clip,
                "materialized": {
                    key: materialized[key]
                    for key in (
                        "clip_steps",
                        "completed_count_first",
                        "mug_done_pudding_remaining_fraction",
                        "abs_action_6d_shape",
                    )
                },
                "losses": losses,
                "gradient_global_norm": float(grad["gradient_global_norm"]),
                "nonzero_grad_tensor_count": int(grad["nonzero_grad_tensor_count"]),
                "clipped_grad_norm": float(clipped_grad_norm.detach().float().item())
                if hasattr(clipped_grad_norm, "detach")
                else float(clipped_grad_norm),
                "cuda_memory": cuda,
                "optimizer_group_lrs": {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups},
                "elapsed_seconds": float(time.monotonic() - started),
            }
            _append_jsonl(metrics_path, last_metric)
            _write_json(status_path, {**result, "status": "RUNNING", "last_metric": last_metric})
            _write_json(
                heartbeat_path,
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "arm_id": config.arm_id,
                    "optimizer_steps_completed": int(step),
                    "training_happened": True,
                    "last_metric": last_metric,
                    "time_unix": time.time(),
                },
            )
            if step in save_steps:
                adapter_dir = checkpoint_root / f"step_{step:04d}" / "adapter"
                adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(adapter_dir), safe_serialization=True)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "step": int(step),
                        "arm_id": config.arm_id,
                        "spec_freeze_id": spec["freeze_id"],
                    },
                    adapter_dir.parent / "optimizer_state.pt",
                )
                shutil.copy2(run_dir / "frozen_spec_snapshot.json", adapter_dir.parent / "frozen_spec_snapshot.json")
                result["checkpoint_written"] = True

        result.update(
            {
                "status": "COMPLETE",
                "success": True,
                "decision": "MPR_XVLA_BOUNDED_TRAINING_ARM_COMPLETE",
                "elapsed_seconds": float(time.monotonic() - started),
                "last_metric": last_metric,
                "metrics_path": str(metrics_path),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
                "result_path": str(result_path),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    except Exception as exc:  # pragma: no cover - real runtime boundary
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "decision": "MPR_XVLA_BOUNDED_TRAINING_ARM_FAILED",
                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "elapsed_seconds": float(time.monotonic() - started),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
                "result_path": str(result_path),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    finally:
        try:
            del model
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(work_clip_dir, ignore_errors=True)
        _write_json(result_path, result)
        _write_json(status_path, result)
        _write_json(
            heartbeat_path,
            {
                "status": str(result["status"]).lower(),
                "pid": os.getpid(),
                "arm_id": config.arm_id,
                "optimizer_steps_completed": int(result.get("optimizer_steps_completed", 0)),
                "training_happened": bool(result.get("training_happened", False)),
                "success": bool(result.get("success", False)),
                "result_path": str(result_path),
                "time_unix": time.time(),
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--hdf5-path", type=Path, default=Path(TASK6_HDF5_WSL))
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--clip-steps", type=int, default=DEFAULT_CLIP_STEPS)
    args = parser.parse_args(argv)
    result = run_training_arm(
        TrainArmConfig(
            spec_path=args.spec,
            arm_id=str(args.arm_id),
            output_root=args.output_root,
            xvla_root=args.xvla_root,
            hdf5_path=args.hdf5_path,
            max_steps_override=args.max_steps_override,
            device_index=int(args.device_index),
            local_files_only=not bool(args.allow_download),
            clip_steps=int(args.clip_steps),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
