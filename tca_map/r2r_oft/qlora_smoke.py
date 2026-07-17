"""One-batch QLoRA gradient smoke for R2R-OFT.

This is a mechanism/feasibility smoke, not a full training run. It loads the
validated quantized OpenVLA-OFT checkpoint, attaches LoRA adapters, builds one
normalized HDF5 action chunk from the audited one-pot-remaining phase, computes
a weighted L1 loss, and verifies that LoRA parameters receive finite nonzero
gradients within local VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tca_map.r2r_oft.data_audit import (
    MOKA_POT_1_POS_SLICE,
    MOKA_POT_2_POS_SLICE,
    _phase_labels,
)


@dataclass(frozen=True)
class QLoRASmokeConfig:
    openvla_repo: Path
    checkpoint_dir: Path
    hdf5_path: Path
    output_path: Path
    lora_rank: int = 4
    lora_alpha: int = 8
    phase_weight_lambda: float = 2.0
    train_demo_count: int = 40
    stove_xy_threshold: float = 0.12
    stove_z_min: float = 0.98
    sample_demo_index: int | None = None


def _normalize_bounds_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _target_xy_from_train_finals(hdf5_path: Path, train_demo_count: int) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5:
        names = sorted(h5["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        finals = []
        for name in names[:train_demo_count]:
            states = np.asarray(h5["data"][name]["states"], dtype=np.float64)
            finals.extend([states[-1, MOKA_POT_1_POS_SLICE], states[-1, MOKA_POT_2_POS_SLICE]])
    return np.median(np.asarray(finals)[:, :2], axis=0)


def _select_one_pot_sample(config: QLoRASmokeConfig, chunk_size: int) -> dict[str, Any]:
    target_xy = _target_xy_from_train_finals(config.hdf5_path, config.train_demo_count)
    with h5py.File(config.hdf5_path, "r") as h5:
        names = sorted(h5["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        candidate_names = names[: config.train_demo_count]
        if config.sample_demo_index is not None:
            candidate_names = [f"demo_{int(config.sample_demo_index)}"]
        for name in candidate_names:
            group = h5["data"][name]
            states = np.asarray(group["states"], dtype=np.float64)
            labels = _phase_labels(
                states,
                target_xy=target_xy,
                stove_xy_threshold=config.stove_xy_threshold,
                stove_z_min=config.stove_z_min,
            )
            valid = np.flatnonzero(labels["one_pot_remaining_phase"][: max(0, states.shape[0] - chunk_size + 1)])
            if valid.size == 0:
                continue
            t = int(valid[0])
            proprio = np.concatenate(
                [
                    np.asarray(group["obs"]["ee_pos"][t], dtype=np.float64).reshape(-1)[:3],
                    np.asarray(group["obs"]["ee_ori"][t], dtype=np.float64).reshape(-1)[:3],
                    np.asarray(group["obs"]["gripper_states"][t], dtype=np.float64).reshape(-1)[:2],
                ]
            )
            return {
                "demo_name": name,
                "timestep": t,
                "target_xy": target_xy,
                "agentview_rgb": np.asarray(group["obs"]["agentview_rgb"][t], dtype=np.uint8),
                "wrist_rgb": np.asarray(group["obs"]["eye_in_hand_rgb"][t], dtype=np.uint8),
                "raw_actions": np.asarray(group["actions"][t : t + chunk_size], dtype=np.float64),
                "raw_proprio": proprio,
                "phase_count_on": int(labels["count_on"][t]),
            }
    raise RuntimeError("No one-pot-remaining sample found in selected HDF5 split")


def run_qlora_gradient_smoke(config: QLoRASmokeConfig) -> dict[str, Any]:
    started = time.monotonic()
    sys.path.insert(0, str(config.openvla_repo))

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

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "R2R-OFT",
        "smoke_type": "one_batch_qlora_gradient",
        "success": False,
        "exception": None,
        "training_run_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "config": {
            "openvla_repo": str(config.openvla_repo),
            "checkpoint_dir": str(config.checkpoint_dir),
            "hdf5_path": str(config.hdf5_path),
            "lora_rank": int(config.lora_rank),
            "lora_alpha": int(config.lora_alpha),
            "phase_weight_lambda": float(config.phase_weight_lambda),
        },
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")

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

        sample = _select_one_pot_sample(config, NUM_ACTIONS_CHUNK)
        action_stats = vla.norm_stats[gen_cfg.unnorm_key]["action"]
        proprio_stats = vla.norm_stats[gen_cfg.unnorm_key]["proprio"]
        normalized_actions = _normalize_bounds_q99(sample["raw_actions"], action_stats)
        normalized_proprio = _normalize_bounds_q99(sample["raw_proprio"], proprio_stats)

        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=True,
            use_proprio=True,
        )
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
        instance = batch_transform(rlds_batch)
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        batch = collator([instance])

        vla = prepare_model_for_kbit_training(vla, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=int(config.lora_rank),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        )
        vla = get_peft_model(vla, lora_config)
        if hasattr(vla, "config"):
            vla.config.use_cache = False
        vla.train()

        num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        num_patches += 1
        labels_for_mask = batch["labels"][:, 1:].to(device)
        current_action_mask = get_current_action_mask(labels_for_mask)
        next_actions_mask = get_next_actions_mask(labels_for_mask)

        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            # The validated INT4 checkpoint can leave vision-layer bias terms in
            # fp32. Keep pixels fp32 for the quantized QLoRA smoke to avoid a
            # conv input/bias dtype mismatch during gradient-enabled forward.
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
        phase_weight = 1.0 + float(config.phase_weight_lambda)
        loss = base_l1 * phase_weight
        loss.backward()

        lora_grad_norm_sq = 0.0
        lora_grad_params = 0
        lora_trainable_params = 0
        for name, param in vla.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_trainable_params += int(param.numel())
                if param.grad is not None:
                    grad = param.grad.detach().float()
                    if torch.isfinite(grad).all():
                        norm = float(torch.linalg.vector_norm(grad).item())
                        if norm > 0:
                            lora_grad_params += 1
                        lora_grad_norm_sq += norm * norm
                    else:
                        report.setdefault("nonfinite_gradient_params", []).append(name)

        report.update(
            {
                "success": bool(lora_grad_params > 0 and not report.get("nonfinite_gradient_params")),
                "sample": {
                    "demo_name": sample["demo_name"],
                    "timestep": int(sample["timestep"]),
                    "phase_count_on": int(sample["phase_count_on"]),
                    "raw_action_min": float(np.min(sample["raw_actions"])),
                    "raw_action_max": float(np.max(sample["raw_actions"])),
                    "normalized_action_min": float(np.min(normalized_actions)),
                    "normalized_action_max": float(np.max(normalized_actions)),
                },
                "unnorm_key": gen_cfg.unnorm_key,
                "loss": {
                    "base_l1": float(base_l1.detach().float().cpu().item()),
                    "phase_weight": float(phase_weight),
                    "weighted_loss": float(loss.detach().float().cpu().item()),
                },
                "lora": {
                    "trainable_params": int(lora_trainable_params),
                    "gradient_nonzero_param_tensors": int(lora_grad_params),
                    "gradient_global_norm": float(lora_grad_norm_sq**0.5),
                },
                "cuda_memory": {
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in real WSL smoke
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            import torch

            if torch.cuda.is_available():
                report["cuda_memory"] = {
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                }
        except Exception:
            pass
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--phase-weight-lambda", type=float, default=2.0)
    parser.add_argument("--train-demo-count", type=int, default=40)
    parser.add_argument("--sample-demo-index", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_qlora_gradient_smoke(
        QLoRASmokeConfig(
            openvla_repo=Path(args.openvla_repo),
            checkpoint_dir=Path(args.checkpoint_dir),
            hdf5_path=Path(args.hdf5_path),
            output_path=Path(args.output_path),
            lora_rank=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            phase_weight_lambda=float(args.phase_weight_lambda),
            train_demo_count=int(args.train_demo_count),
            sample_demo_index=args.sample_demo_index,
        )
    )
    print(json.dumps({k: report.get(k) for k in ("success", "exception", "elapsed_seconds", "cuda_memory", "loss", "lora")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
