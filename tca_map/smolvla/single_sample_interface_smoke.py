"""Bounded SmolVLA single-sample interface smoke.

This entrypoint is intentionally narrow: it constructs one synthetic observation,
runs one local-only CPU action selection, and records interface/memory metadata.
It is not a benchmark, not training, and not rollout evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.interface_adapters import (
    DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS,
    adapt_observation_state,
    adapt_policy_action_to_env_action,
    select_image_source,
)
from tca_map.smolvla.load_only_smoke import (
    MAX_LOAD_ONLY_SECONDS,
    RUNTIME_DEPENDENCIES,
    _env_flag,
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)


FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_CLOUD_HANDOFF",
]
MAX_INTERFACE_SECONDS = 600
MAX_VRAM_MB = 14336
DIAGNOSTIC_LIBERO_ENV_ACTION_DIM = 7


def _load_policy(smolvla_ckpt: Path, hf_home: Path, external_dependency: dict[str, Any], device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(
        smolvla_ckpt,
        local_files_only=True,
        cache_dir=hf_home,
    )
    config.device = device
    config.load_vlm_weights = False
    config.compile_model = False
    config.push_to_hub = False
    config.num_steps = min(int(getattr(config, "num_steps", 1)), 1)
    if external_dependency.get("found") and external_dependency.get("root"):
        config.vlm_model_name = external_dependency["root"]

    policy = SmolVLAPolicy.from_pretrained(
        smolvla_ckpt,
        config=config,
        local_files_only=True,
        cache_dir=hf_home,
        token=False,
        strict=False,
    )
    policy.eval()
    policy.reset()
    return policy, config


def _build_synthetic_batch(config, tokenizer_root: Path, task: str, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
    synthetic_state_obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    state_adapter = adapt_observation_state(
        synthetic_state_obs,
        DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS,
        state_dim,
        adapter_name="diagnostic_eef_pos_quat_xyz_6d_state_adapter",
    )

    batch: dict[str, Any] = {
        "observation.state": torch.tensor([state_adapter.values], dtype=torch.float32),
        "observation.language.tokens": encoded["input_ids"].to(dtype=torch.long),
        "observation.language.attention_mask": encoded["attention_mask"].to(dtype=torch.bool),
    }
    image_adapter_metadata: dict[str, Any] = {}
    for key, feature in config.image_features.items():
        channels, height, width = feature.shape
        fill = min(0.75, 0.1 * (len(batch) + 1))
        synthetic_images = {
            "agentview_image": np.full((height, width, channels), fill, dtype=np.float32),
            "robot0_eye_in_hand_image": np.full((height, width, channels), fill / 2.0, dtype=np.float32),
        }
        selected = select_image_source(synthetic_images, key)
        array = np.asarray(selected.value, dtype=np.float32)
        image = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
        batch[key] = image
        image_adapter_metadata[key] = selected.metadata

    metadata = {
        "state_adapter": state_adapter.metadata,
        "image_adapters": image_adapter_metadata,
        "synthetic_only": True,
    }
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}, metadata


def _run_single_sample_interface(
    smolvla_ckpt: Path,
    hf_home: Path,
    external_dependency: dict[str, Any],
    task: str,
    device: str,
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    rss_before = _rss_mb()
    gpu_before = _nvidia_smi()

    policy, config = _load_policy(smolvla_ckpt, hf_home, external_dependency, device)
    tokenizer_root = Path(external_dependency["root"])
    batch, adapter_metadata = _build_synthetic_batch(config, tokenizer_root, task, device)
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)

    infer_started = time.monotonic()
    with torch.inference_mode():
        action = policy.select_action(batch, noise=noise)
    inference_elapsed = time.monotonic() - infer_started

    action_cpu = action.detach().cpu()
    finite = bool(torch.isfinite(action_cpu).all().item())
    action_shape = list(action_cpu.shape)
    action_preview = [round(float(x), 6) for x in action_cpu.flatten()[: min(6, action_cpu.numel())]]
    action_adapter = adapt_policy_action_to_env_action(action_cpu, DIAGNOSTIC_LIBERO_ENV_ACTION_DIM)
    adapter_metadata["action_adapter"] = action_adapter.metadata
    adapter_metadata["diagnostic_env_action_dim"] = DIAGNOSTIC_LIBERO_ENV_ACTION_DIM
    adapter_metadata["adapted_action_preview"] = [
        round(float(x), 6) for x in action_adapter.values[: min(DIAGNOSTIC_LIBERO_ENV_ACTION_DIM, len(action_adapter.values))]
    ]

    cuda_max_allocated_mb = None
    if torch.cuda.is_available():
        cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "load_and_interface_elapsed_sec": round(time.monotonic() - started, 3),
        "single_sample_inference_elapsed_sec": round(inference_elapsed, 3),
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(),
        "gpu_before": gpu_before,
        "gpu_after": _nvidia_smi(),
        "cuda_max_allocated_mb": cuda_max_allocated_mb,
        "device": device,
        "config_device": config.device,
        "load_vlm_weights": config.load_vlm_weights,
        "num_steps": config.num_steps,
        "task": task,
        "batch_keys": sorted(batch.keys()),
        "action_shape": action_shape,
        "action_finite": finite,
        "action_preview": action_preview,
        "adapter_metadata": adapter_metadata,
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)

    config_files = _find_files(smolvla_ckpt, ["config.json"])
    tokenizer_files = _find_files(
        smolvla_ckpt,
        [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "sentencepiece.bpe.model",
        ],
    )
    weight_files = _find_files(
        smolvla_ckpt,
        ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
        ["*.safetensors", "*.bin"],
    )
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    deps = _runtime_dependencies()
    gpu = _nvidia_smi()

    heavy_gate = _env_flag("ALLOW_HEAVY_IMPORT")
    interface_gate = _env_flag("ALLOW_SINGLE_SAMPLE_INFERENCE")
    forbidden_gates_set = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    files_ready = bool(config_files and weight_files and (tokenizer_files or external_dependency["found"]))
    deps_ready = all(deps.values())
    gpu_memory_ok = bool(gpu["available"] and gpu["memory_total_mb"] and gpu["memory_total_mb"] >= 14048)

    exit_code = 0
    blocked_reason = None
    interface_attempted = False
    interface_result: dict[str, Any] | None = None
    interface_error: dict[str, Any] | None = None

    if not heavy_gate:
        exit_code = 2
        blocked_reason = "ALLOW_HEAVY_IMPORT=1 is required only inside bounded SmolVLA interface smoke."
    elif not interface_gate:
        exit_code = 3
        blocked_reason = "ALLOW_SINGLE_SAMPLE_INFERENCE=1 is required only for one synthetic interface smoke."
    elif forbidden_gates_set:
        exit_code = 4
        blocked_reason = f"Forbidden gate(s) set: {', '.join(forbidden_gates_set)}"
    elif not files_ready:
        exit_code = 5
        blocked_reason = "SmolVLA checkpoint/tokenizer/weights readiness is incomplete."
    elif not deps_ready:
        exit_code = 6
        missing = [name for name, present in deps.items() if not present]
        blocked_reason = f"Missing runtime dependencies: {', '.join(missing)}"
    elif not gpu_memory_ok:
        exit_code = 7
        blocked_reason = "GPU memory policy check failed before interface smoke."
    else:
        try:
            interface_attempted = True
            interface_result = _run_single_sample_interface(
                smolvla_ckpt=smolvla_ckpt,
                hf_home=hf_home,
                external_dependency=external_dependency,
                task=args.task,
                device=args.device,
            )
            if not interface_result["action_finite"]:
                exit_code = 8
                blocked_reason = "Single-sample action output contains non-finite values."
            elif interface_result["load_and_interface_elapsed_sec"] > MAX_INTERFACE_SECONDS:
                exit_code = 9
                blocked_reason = "Single-sample interface smoke exceeded the 10 minute runtime budget."
            elif (interface_result.get("cuda_max_allocated_mb") or 0) > MAX_VRAM_MB:
                exit_code = 10
                blocked_reason = "Single-sample interface smoke exceeded the 14GB VRAM budget."
        except Exception as exc:  # noqa: BLE001 - report exact interface failure.
            exit_code = 8
            blocked_reason = f"SmolVLA single-sample interface smoke failed: {type(exc).__name__}: {exc}"
            interface_error = {"type": type(exc).__name__, "message": str(exc)}

    report = {
        "policy": {
            "single_sample_interface_smoke": True,
            "synthetic_input_only": True,
            "downloads_performed": False,
            "heavy_model_imports_performed": interface_attempted,
            "model_load_performed": bool(interface_result),
            "single_sample_model_inference_performed": bool(interface_result),
            "model_inference_performed": bool(interface_result),
            "adapter_metadata_recorded": bool(interface_result and interface_result.get("adapter_metadata")),
            "gpu_training_performed": False,
            "training_performed": False,
            "real_rollouts_performed": False,
            "simulator_executed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": heavy_gate,
            "single_sample_inference_gate_set": interface_gate,
            "forbidden_gates_set": forbidden_gates_set,
            "max_vram_policy_mb": MAX_VRAM_MB,
            "max_runtime_sec": MAX_INTERFACE_SECONDS,
        },
        "paths": {
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
        },
        "files": {
            "config_found": config_files,
            "tokenizer_found": tokenizer_files,
            "weights_found": weight_files,
            "external_tokenizer_dependency": external_dependency,
            "files_ready": files_ready,
        },
        "runtime_dependencies": deps,
        "gpu": gpu,
        "interface": interface_result,
        "interface_error": interface_error,
        "result": {
            "passed": exit_code == 0,
            "blocked": exit_code != 0,
            "blocked_reason": blocked_reason,
            "elapsed_sec": round(time.monotonic() - started, 3),
        },
        "recommended_next_step": (
            "Continue to tiny feature-cache/interface validation; do not train or rollout."
            if exit_code == 0
            else "Fix the reported interface blocker before any feature-cache or head-only smoke."
        ),
    }
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--report-path", default="reports/smolvla_single_sample_interface_report.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--task", default="pick up the object")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
