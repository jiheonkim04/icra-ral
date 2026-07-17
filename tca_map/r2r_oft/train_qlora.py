"""Spec-locked bounded QLoRA trainer for Epoch 5 R2R-OFT.

The trainer is intentionally narrow: it refuses to run arms not present in the
frozen spec, writes heartbeat/status artifacts before training, trains only VLA
LoRA adapters, and saves adapter checkpoints plus metrics at the frozen steps.
Heavy OpenVLA imports happen inside ``run_training_arm`` so lightweight tests can
exercise the safety logic without model loading.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tca_map.r2r_oft.data_audit import _phase_labels
from tca_map.r2r_oft.qlora_smoke import _normalize_bounds_q99, _target_xy_from_train_finals
from tca_map.r2r_oft.training_spec import (
    build_epoch5_training_spec,
    validate_training_spec,
)


@dataclass(frozen=True)
class TrainArmConfig:
    spec_path: Path
    arm_id: str
    output_root: Path
    openvla_repo: Path
    checkpoint_dir: Path
    hdf5_path: Path
    max_steps_override: int | None = None
    device_index: int = 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


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


def _ensure_rich_logging_handler() -> bool:
    """Install a minimal RichHandler fallback when ``rich`` is absent.

    OpenVLA-OFT's Prismatic logger config references ``rich.logging.RichHandler``.
    The local WSL training env used for the bounded run may not have the optional
    ``rich`` package installed; downloading it would violate the no-new-download
    boundary.  A plain ``logging.StreamHandler`` is sufficient for this detached
    trainer, so we provide only the handler symbol Prismatic resolves.

    Returns:
        True if a fallback was installed, False if the real package was present.
    """

    try:
        from rich.logging import RichHandler as _RichHandler  # noqa: F401

        return False
    except ModuleNotFoundError:
        rich_module = sys.modules.setdefault("rich", types.ModuleType("rich"))
        rich_module.__spec__ = importlib.machinery.ModuleSpec("rich", loader=None)
        logging_module = types.ModuleType("rich.logging")
        logging_module.__spec__ = importlib.machinery.ModuleSpec("rich.logging", loader=None)

        class RichHandler(logging.StreamHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()

        logging_module.RichHandler = RichHandler
        setattr(rich_module, "logging", logging_module)
        sys.modules["rich.logging"] = logging_module
        return True


def _load_spec(path: Path) -> dict[str, Any]:
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
    else:
        spec = build_epoch5_training_spec()
    errors = validate_training_spec(spec)
    if errors:
        raise ValueError(f"invalid training spec: {'; '.join(errors)}")
    return spec


def _arm_by_id(spec: dict[str, Any], arm_id: str) -> dict[str, Any]:
    matches = [arm for arm in spec["arms"] if arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise ValueError(f"arm_id {arm_id!r} is not exactly one frozen spec arm")
    return matches[0]


def _phase_for_step(cycle: list[int], step_index_zero_based: int) -> int:
    if not cycle:
        raise ValueError("phase cycle must not be empty")
    return int(cycle[step_index_zero_based % len(cycle)])


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_")[-1]), name)
    except ValueError:
        return (10**9, name)


def _proprio_from_group(group: h5py.Group, t: int) -> np.ndarray:
    obs = group["obs"]
    return np.concatenate(
        [
            np.asarray(obs["ee_pos"][t], dtype=np.float64).reshape(-1)[:3],
            np.asarray(obs["ee_ori"][t], dtype=np.float64).reshape(-1)[:3],
            np.asarray(obs["gripper_states"][t], dtype=np.float64).reshape(-1)[:2],
        ]
    )


def build_phase_chunk_index(
    hdf5_path: Path,
    *,
    demo_indices: list[int],
    chunk_size: int,
    train_demo_count_for_target_xy: int,
    stove_xy_threshold: float = 0.12,
    stove_z_min: float = 0.98,
) -> dict[int, list[dict[str, Any]]]:
    """Build chunk metadata grouped by phase count 0/1/2."""

    target_xy = _target_xy_from_train_finals(hdf5_path, train_demo_count_for_target_xy)
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    allowed = set(int(index) for index in demo_indices)
    with h5py.File(hdf5_path, "r") as h5:
        names = sorted(h5["data"].keys(), key=_demo_sort_key)
        for demo_index, name in enumerate(names):
            if demo_index not in allowed:
                continue
            group = h5["data"][name]
            states = np.asarray(group["states"], dtype=np.float64)
            labels = _phase_labels(
                states,
                target_xy=target_xy,
                stove_xy_threshold=stove_xy_threshold,
                stove_z_min=stove_z_min,
            )
            max_start = max(0, states.shape[0] - chunk_size + 1)
            for t in range(max_start):
                phase_count = int(labels["count_on"][t])
                grouped.setdefault(phase_count, []).append(
                    {
                        "demo_index": int(demo_index),
                        "demo_name": name,
                        "timestep": int(t),
                        "phase_count_on": phase_count,
                    }
                )
    return grouped


def select_chunk_for_step(
    grouped: dict[int, list[dict[str, Any]]],
    *,
    cycle: list[int],
    step_index_zero_based: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Select a chunk according to the frozen deterministic phase cycle."""

    preferred_phase = _phase_for_step(cycle, step_index_zero_based)
    phases = [preferred_phase] + [phase for phase in sorted(grouped) if phase != preferred_phase]
    for phase in phases:
        candidates = grouped.get(int(phase), [])
        if candidates:
            index = int(rng.integers(0, len(candidates)))
            return dict(candidates[index])
    raise RuntimeError("no chunks available for any phase")


def _load_chunk(hdf5_path: Path, chunk: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as h5:
        group = h5["data"][chunk["demo_name"]]
        t = int(chunk["timestep"])
        return {
            **chunk,
            "agentview_rgb": np.asarray(group["obs"]["agentview_rgb"][t], dtype=np.uint8),
            "wrist_rgb": np.asarray(group["obs"]["eye_in_hand_rgb"][t], dtype=np.uint8),
            "raw_actions": np.asarray(group["actions"][t : t + chunk_size], dtype=np.float64),
            "raw_proprio": _proprio_from_group(group, t),
        }


def _make_instance(
    *,
    sample: dict[str, Any],
    action_stats: dict[str, Any],
    proprio_stats: dict[str, Any],
    batch_transform: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    normalized_actions = _normalize_bounds_q99(sample["raw_actions"], action_stats)
    normalized_proprio = _normalize_bounds_q99(sample["raw_proprio"], proprio_stats)
    rlds_batch = {
        "dataset_name": "r2r_oft_task8_hdf5",
        "observation": {
            "image_primary": np.expand_dims(sample["agentview_rgb"], axis=0),
            "image_wrist": np.expand_dims(sample["wrist_rgb"], axis=0),
            "proprio": np.expand_dims(normalized_proprio, axis=0),
        },
        "task": {"language_instruction": b"put both moka pots on the stove"},
        "action": normalized_actions,
    }
    return batch_transform(rlds_batch), normalized_actions


def run_training_arm(config: TrainArmConfig) -> dict[str, Any]:
    """Run one frozen training arm and return the final result payload."""

    started = time.monotonic()
    spec = _load_spec(config.spec_path)
    arm = _arm_by_id(spec, config.arm_id)
    shared = spec["shared_training"]
    cycle = list(arm["sampler"]["cycle_phase_counts"])
    max_steps = int(config.max_steps_override or shared["max_optimizer_steps"])
    if max_steps > int(shared["max_optimizer_steps"]):
        raise ValueError("max_steps_override cannot exceed frozen spec max_optimizer_steps")

    run_dir = config.output_root / config.arm_id
    heartbeat_path = run_dir / "heartbeat.json"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"
    result_path = run_dir / "result.json"
    checkpoints_dir = run_dir / "checkpoints"
    spec_snapshot_path = run_dir / "frozen_spec_snapshot.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(spec_snapshot_path, spec)

    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "R2R-OFT",
        "arm_id": config.arm_id,
        "role": arm["role"],
        "status": "RUNNING",
        "success": False,
        "training_happened": False,
        "optimizer_steps_completed": 0,
        "checkpoint_written": False,
        "exception": None,
        "git_commit": _git_commit(),
        "run_dir": str(run_dir),
        "spec_path": str(config.spec_path),
        "spec_freeze_id": spec["freeze_id"],
        "phase_weight_lambda": float(arm["phase_weight_lambda"]),
        "lora_rank": int(arm["lora_rank"]),
        "lora_alpha": int(arm["lora_alpha"]),
        "max_steps": max_steps,
        "started_unix": time.time(),
    }
    _write_json(status_path, result)
    _write_json(
        heartbeat_path,
        {
            "status": "initializing",
            "arm_id": config.arm_id,
            "pid": os.getpid(),
            "optimizer_steps_completed": 0,
            "training_happened": False,
            "time_unix": time.time(),
            "git_commit": result["git_commit"],
        },
    )

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        sys.path.insert(0, str(config.openvla_repo))
        rich_logging_fallback_installed = _ensure_rich_logging_handler()
        result["rich_logging_fallback_installed"] = bool(rich_logging_fallback_installed)

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
        from prismatic.vla.datasets import RLDSBatchTransform

        from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(int(config.device_index))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
        device = torch.device(f"cuda:{int(config.device_index)}")

        grouped = build_phase_chunk_index(
            config.hdf5_path,
            demo_indices=list(spec["data"]["train_demo_indices"]),
            chunk_size=int(shared["action_chunk_size"]),
            train_demo_count_for_target_xy=len(spec["data"]["train_demo_indices"]),
        )
        if not all(grouped.get(phase) for phase in (0, 1, 2)):
            raise RuntimeError("training split must contain phase 0/1/2 chunks")

        gen_cfg = GenerateConfig(
            pretrained_checkpoint=str(config.checkpoint_dir),
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            load_in_4bit=True,
            load_in_8bit=False,
            task_suite_name="libero_10",
            num_trials_per_task=1,
        )
        vla = get_vla(gen_cfg)
        check_unnorm_key(gen_cfg, vla)
        processor = get_processor(gen_cfg)
        action_head = get_action_head(gen_cfg, vla.llm_dim).to(device)
        proprio_projector = get_proprio_projector(gen_cfg, vla.llm_dim, proprio_dim=8).to(device)
        action_head.eval().requires_grad_(False)
        proprio_projector.eval().requires_grad_(False)

        vla = prepare_model_for_kbit_training(vla, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=int(arm["lora_rank"]),
            lora_alpha=int(arm["lora_alpha"]),
            lora_dropout=float(shared["lora_dropout"]),
            target_modules=shared["lora_target_modules"],
            bias="none",
        )
        vla = get_peft_model(vla, lora_config)
        if hasattr(vla, "config"):
            vla.config.use_cache = False
        vla.train()

        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(shared["learning_rate"]),
            weight_decay=float(shared["weight_decay"]),
        )

        action_stats = vla.norm_stats[gen_cfg.unnorm_key]["action"]
        proprio_stats = vla.norm_stats[gen_cfg.unnorm_key]["proprio"]
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=True,
            use_proprio=True,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        num_patches += 1
        rng = np.random.default_rng(int(shared["seed"]))
        save_steps = set(int(step) for step in shared["save_steps"] if int(step) <= max_steps)

        for step_index in range(max_steps):
            chunk_meta = select_chunk_for_step(grouped, cycle=cycle, step_index_zero_based=step_index, rng=rng)
            sample = _load_chunk(config.hdf5_path, chunk_meta, int(shared["action_chunk_size"]))
            instance, normalized_actions = _make_instance(
                sample=sample,
                action_stats=action_stats,
                proprio_stats=proprio_stats,
                batch_transform=batch_transform,
            )
            batch = collator([instance])
            labels_for_mask = batch["labels"][:, 1:].to(device)
            current_action_mask = get_current_action_mask(labels_for_mask)
            next_actions_mask = get_next_actions_mask(labels_for_mask)

            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=batch["proprio"].to(torch.bfloat16).to(device),
                proprio_projector=proprio_projector,
                use_film=False,
            )
            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )
            predicted_actions = action_head.predict_action(actions_hidden_states)
            target_actions = batch["actions"].to(torch.bfloat16).to(device)
            base_l1 = torch.nn.functional.l1_loss(predicted_actions, target_actions)
            phase_weight = 1.0 + float(arm["phase_weight_lambda"]) * float(sample["phase_count_on"] == 1)
            loss = base_l1 * phase_weight
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step_index + 1}")

            loss.backward()
            grad_norm_sq = 0.0
            nonzero_grad_tensors = 0
            for param in trainable_params:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"nonfinite gradient at step {step_index + 1}")
                norm = float(torch.linalg.vector_norm(grad).item())
                if norm > 0:
                    nonzero_grad_tensors += 1
                grad_norm_sq += norm * norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step = step_index + 1
            result["training_happened"] = True
            result["optimizer_steps_completed"] = step
            cuda_peak_mib = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
            metric = {
                "step": step,
                "demo_name": sample["demo_name"],
                "timestep": int(sample["timestep"]),
                "phase_count_on": int(sample["phase_count_on"]),
                "base_l1": float(base_l1.detach().float().item()),
                "phase_weight": float(phase_weight),
                "weighted_loss": float(loss.detach().float().item()),
                "normalized_action_min": float(np.min(normalized_actions)),
                "normalized_action_max": float(np.max(normalized_actions)),
                "nonzero_grad_tensors": int(nonzero_grad_tensors),
                "gradient_global_norm": float(grad_norm_sq ** 0.5),
                "cuda_peak_mib": cuda_peak_mib,
                "elapsed_seconds": float(time.monotonic() - started),
            }
            _append_jsonl(metrics_path, metric)
            _write_json(
                heartbeat_path,
                {
                    "status": "running",
                    "arm_id": config.arm_id,
                    "pid": os.getpid(),
                    "optimizer_steps_completed": step,
                    "training_happened": True,
                    "last_metric": metric,
                    "time_unix": time.time(),
                    "git_commit": result["git_commit"],
                },
            )
            _write_json(status_path, {**result, "status": "RUNNING", "last_metric": metric})

            if cuda_peak_mib > float(shared["max_cuda_peak_mib"]):
                raise RuntimeError(f"cuda peak {cuda_peak_mib:.3f} MiB exceeds frozen limit")

            if step in save_steps:
                adapter_dir = checkpoints_dir / f"step_{step:04d}" / "adapter"
                adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                vla.save_pretrained(str(adapter_dir))
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "arm_id": config.arm_id,
                        "spec_freeze_id": spec["freeze_id"],
                    },
                    adapter_dir.parent / "optimizer_state.pt",
                )
                shutil.copy2(spec_snapshot_path, adapter_dir.parent / "frozen_spec_snapshot.json")
                result["checkpoint_written"] = True

        result.update(
            {
                "status": "COMPLETE",
                "success": True,
                "elapsed_seconds": float(time.monotonic() - started),
                "cuda_peak_mib": float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)),
                "metrics_path": str(metrics_path),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
                "result_path": str(result_path),
            }
        )
    except Exception as exc:  # pragma: no cover - covered by real run artifacts
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.monotonic() - started),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
                "result_path": str(result_path),
            }
        )
    finally:
        _write_json(result_path, result)
        _write_json(status_path, result)
        _write_json(
            heartbeat_path,
            {
                "status": result["status"].lower(),
                "arm_id": config.arm_id,
                "pid": os.getpid(),
                "optimizer_steps_completed": int(result.get("optimizer_steps_completed", 0)),
                "training_happened": bool(result.get("training_happened", False)),
                "success": bool(result.get("success", False)),
                "time_unix": time.time(),
                "git_commit": result.get("git_commit"),
                "result_path": str(result_path),
            },
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json"))
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/openvla_oft_int4/epoch5_r2r_oft_training"))
    parser.add_argument("--openvla-repo", type=Path, default=Path("/mnt/c/assets/repos/openvla-oft"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/home/jiheon/assets/checkpoints/openvla-oft/"
            "moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
        ),
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("/mnt/c/assets/data/libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"),
    )
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    result = run_training_arm(
        TrainArmConfig(
            spec_path=args.spec,
            arm_id=args.arm_id,
            output_root=args.output_root,
            openvla_repo=args.openvla_repo,
            checkpoint_dir=args.checkpoint_dir,
            hdf5_path=args.hdf5_path,
            max_steps_override=args.max_steps_override,
            device_index=args.device_index,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
