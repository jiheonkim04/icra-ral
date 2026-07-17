"""RTX 5080-only quantized OpenVLA-OFT feasibility helpers.

The helpers in this module are intentionally narrow wrappers around the
official OpenVLA-OFT evaluation stack.  They do not train, fine-tune, use CPU
offload, or download any benchmark datasets.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ID = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
HF_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
CHECKPOINT_VISIBLE_BYTES = 15_939_168_050
CHECKPOINT_VISIBLE_GIB = round(CHECKPOINT_VISIBLE_BYTES / (1024**3), 3)
DEFAULT_CHECKPOINT_DIR = Path(
    "/home/jiheon/assets/checkpoints/openvla-oft/"
    "moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
)
DEFAULT_OPENVLA_REPO = Path("/mnt/c/assets/repos/openvla-oft")
DEFAULT_LIBERO_REPO = Path("/home/jiheon/assets/repos/LIBERO")
DEFAULT_RUN_DIR = Path("runs/openvla_oft_int4")
RESET_IDENTITIES = [20260711, 20260712, 20260713, 20260714, 20260715]
RESET_IDENTITY_BASE = 20260711
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
HARD_SLICE_TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "hard_slice_stable_grasp"},
    {"suite": "libero_10", "task_id": 4, "role": "hard_slice_long_horizon"},
    {"suite": "libero_spatial", "task_id": 2, "role": "matched_control_spatial"},
    {"suite": "libero_10", "task_id": 2, "role": "matched_control_libero10"},
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _git_commit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    head = _run(["git", "rev-parse", "HEAD"], cwd=path)
    status = _run(["git", "status", "--short", "--branch"], cwd=path)
    remote = _run(["git", "remote", "-v"], cwd=path)
    return {
        "path": str(path),
        "exists": True,
        "head": head.get("stdout"),
        "status": status.get("stdout"),
        "remote": remote.get("stdout"),
        "head_returncode": head.get("returncode"),
    }


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception as exc:
        return f"IMPORT_OR_METADATA_ERROR:{type(exc).__name__}:{exc}"


def _torch_info() -> dict[str, Any]:
    try:
        import torch

        info: dict[str, Any] = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "cuda_mem_free_bytes": int(free),
                    "cuda_mem_total_bytes": int(total),
                    "cuda_mem_free_mib": round(free / (1024**2), 3),
                    "cuda_mem_total_mib": round(total / (1024**2), 3),
                    "device_capability": list(torch.cuda.get_device_capability(0)),
                    "device_total_memory_bytes": int(props.total_memory),
                }
            )
        return info
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _memory_info() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        payload["process_view"] = {
            "ram_total_bytes": int(vm.total),
            "ram_available_bytes": int(vm.available),
            "ram_used_bytes": int(vm.used),
            "ram_available_gib": round(vm.available / (1024**3), 3),
            "ram_total_gib": round(vm.total / (1024**3), 3),
            "swap_total_bytes": int(swap.total),
            "swap_used_bytes": int(swap.used),
        }
    except Exception as exc:
        payload["process_view_error"] = f"{type(exc).__name__}: {exc}"
    for path in [Path("/"), Path("/home/jiheon/assets"), Path("/mnt/c")]:
        try:
            usage = shutil.disk_usage(path)
            payload.setdefault("disk", {})[str(path)] = {
                "total_bytes": int(usage.total),
                "free_bytes": int(usage.free),
                "used_bytes": int(usage.used),
                "free_gib": round(usage.free / (1024**3), 3),
            }
        except Exception as exc:
            payload.setdefault("disk", {})[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
    return payload


def _rss_mib() -> float | None:
    try:
        import psutil

        return _round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def _cuda_memory_payload(torch_mod: Any) -> dict[str, Any]:
    if not torch_mod.cuda.is_available():
        return {"allocated_bytes": None, "max_allocated_bytes": None}
    return {
        "allocated_bytes": int(torch_mod.cuda.memory_allocated()),
        "max_allocated_bytes": int(torch_mod.cuda.max_memory_allocated()),
        "allocated_mib": _round(torch_mod.cuda.memory_allocated() / (1024**2), 3),
        "max_allocated_mib": _round(torch_mod.cuda.max_memory_allocated() / (1024**2), 3),
    }


def _autocast_payload(torch_mod: Any) -> dict[str, Any]:
    return {
        "cuda": bool(torch_mod.is_autocast_enabled("cuda")),
        "cpu": bool(torch_mod.is_autocast_enabled("cpu")),
        "fp16_or_bf16_active": bool(torch_mod.is_autocast_enabled("cuda")),
    }


def _init_state_sha256(initial_state: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(initial_state))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _identity_to_initial_state_index(identity: int) -> int:
    index = int(identity) - RESET_IDENTITY_BASE
    if index < 0 or index >= MAX_OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(f"identity {identity} maps to invalid official initial state index {index}")
    return index


def _parse_reset_identities(raw: str | None) -> list[int]:
    if not raw:
        return list(RESET_IDENTITIES)
    identities = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not identities:
        raise ValueError("reset identity list is empty")
    for identity in identities:
        _identity_to_initial_state_index(identity)
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate reset identities: {identities}")
    return identities


def _parse_task_specs(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return [dict(item) for item in HARD_SLICE_TASKS]
    specs: list[dict[str, Any]] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("task specs must be comma-separated suite:task_id[:role] entries")
        suite, task_id_text = parts[:2]
        role = parts[2] if len(parts) == 3 else f"{suite}_task_{task_id_text}"
        specs.append({"suite": suite.strip(), "task_id": int(task_id_text), "role": role.strip()})
    if not specs:
        raise ValueError("task spec list is empty")
    seen = {(str(item["suite"]), int(item["task_id"])) for item in specs}
    if len(seen) != len(specs):
        raise ValueError(f"duplicate task specs: {raw}")
    return specs


def _manifest_label(args: argparse.Namespace) -> str:
    return str(getattr(args, "manifest_label", "") or "hard_slice")


def build_environment_lock(args: argparse.Namespace) -> dict[str, Any]:
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "tokenizers",
        "bitsandbytes",
        "accelerate",
        "peft",
        "huggingface-hub",
        "safetensors",
        "draccus",
        "json-numpy",
        "tensorflow",
        "tensorflow-datasets",
        "tensorflow-graphics",
        "dlimp",
        "robosuite",
        "mujoco",
        "bddl",
        "libero",
        "timm",
        "sentencepiece",
        "protobuf",
    ]
    return {
        "schema_version": 1,
        "date_kst": "2026-07-11",
        "machine_scope": "single_local_rtx5080_only",
        "forbidden": {
            "full_bf16_loading": True,
            "cpu_offload": True,
            "disk_offload": True,
            "training": True,
            "rlds_dataset_download": True,
            "libero_pro_download": True,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "torch": _torch_info(),
        "memory": _memory_info(),
        "nvidia_smi": _run(["nvidia-smi"]),
        "packages": {name: _package_version(name) for name in packages},
        "sources": {
            "openvla_oft": _git_commit(Path(args.openvla_repo)),
            "libero": _git_commit(Path(args.libero_repo)),
            "transformers_openvla_oft": {
                "repo": "https://github.com/moojink/transformers-openvla-oft.git",
                "pinned_commit": "bc339d9ad707454c0c115970db43c260067c61ab",
            },
            "dlimp_openvla": {
                "repo": "https://github.com/moojink/dlimp_openvla.git",
                "pinned_commit": "040105d256bd28866cc6620621a3d5f7b6b91b46",
            },
        },
        "checkpoint": {
            "repo_id": REPO_ID,
            "revision": HF_REVISION,
            "visible_size_bytes": CHECKPOINT_VISIBLE_BYTES,
            "visible_size_gib": CHECKPOINT_VISIBLE_GIB,
            "local_dir": str(Path(args.checkpoint_dir)),
        },
        "compatibility_deviations": [
            "PyTorch remains 2.10.0+cu128 rather than official 2.2.0 because RTX 5080 requires a newer CUDA stack.",
            "protobuf is locked to the newest version that lets TensorFlow Datasets and dlimp import in this venv; TensorFlow 2.15.0 declares an older protobuf range.",
            "peft remains inherited from the SmolVLA env because OpenVLA-OFT eval path does not use PEFT when use_film=False.",
        ],
    }


def download_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO_ID, revision=HF_REVISION, files_metadata=True)
    metadata = {
        "repo_id": REPO_ID,
        "requested_revision": HF_REVISION,
        "resolved_sha": info.sha,
        "license": (info.cardData or {}).get("license") if isinstance(info.cardData, dict) else None,
        "gated": bool(getattr(info, "gated", False)),
        "siblings": [
            {
                "rfilename": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "lfs": getattr(sibling, "lfs", None),
            }
            for sibling in info.siblings
        ],
        "visible_size_bytes": sum((getattr(sibling, "size", None) or 0) for sibling in info.siblings),
        "approved_download_scope": "exact_selected_checkpoint_only",
        "local_dir": str(checkpoint_dir),
    }
    started = time.monotonic()
    local_path = snapshot_download(
        repo_id=REPO_ID,
        revision=HF_REVISION,
        local_dir=str(checkpoint_dir),
        max_workers=int(args.max_workers),
    )
    metadata["download"] = {
        "local_path": str(local_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed": True,
    }
    return metadata


def checksum_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir)
    files = []
    total = 0
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(checkpoint_dir).as_posix()
        if rel.startswith(".cache/"):
            continue
        size = path.stat().st_size
        total += size
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        files.append({"path": rel, "size_bytes": int(size), "sha256": h.hexdigest()})
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "file_count": len(files),
        "visible_bytes": int(total),
        "visible_gib": round(total / (1024**3), 3),
        "files": files,
    }


def memory_preflight(args: argparse.Namespace) -> dict[str, Any]:
    env = build_environment_lock(args)
    torch_info = env.get("torch", {})
    mem = env.get("memory", {}).get("process_view", {})
    cuda_total = int(torch_info.get("cuda_mem_total_bytes") or 0)
    cuda_free = int(torch_info.get("cuda_mem_free_bytes") or 0)
    ram_available = int(mem.get("ram_available_bytes") or 0)
    expected = {
        "int4_expected_model_vram_gib": "4.0-7.5 plus bf16 vision/projector/action-head overhead",
        "int8_expected_model_vram_gib": "8.0-11.5 plus bf16 vision/projector/action-head overhead",
        "full_bf16_expected_vram_gib": ">14.8 plus overhead; forbidden",
    }
    decision = "INT4_PREFLIGHT_OK"
    reasons: list[str] = []
    if not torch_info.get("cuda_available"):
        decision = "RTX5080_MEMORY_BLOCKED"
        reasons.append("CUDA is unavailable.")
    if cuda_total and cuda_total < 15 * 1024**3:
        decision = "RTX5080_MEMORY_BLOCKED"
        reasons.append("Total VRAM is below the expected 16GB RTX 5080 envelope.")
    if cuda_free and cuda_free < 10 * 1024**3:
        decision = "RTX5080_MEMORY_BLOCKED"
        reasons.append("Free VRAM is below 10GiB before loading.")
    if ram_available and ram_available < 8 * 1024**3:
        decision = "RTX5080_MEMORY_BLOCKED"
        reasons.append("Available WSL RAM is below 8GiB before loading.")
    return {
        "schema_version": 1,
        "decision": decision,
        "reasons": reasons,
        "hard_rules": {
            "full_bf16_attempt_allowed": False,
            "cpu_offload_allowed": False,
            "disk_offload_allowed": False,
            "swap_or_pagefile_model_execution_allowed": False,
        },
        "environment": env,
        "expected_memory": expected,
    }


def _first_parameter_device(module: Any) -> dict[str, Any]:
    try:
        for name, parameter in module.named_parameters():
            return {"name": name, "device": str(parameter.device), "dtype": str(parameter.dtype)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"name": None, "device": None, "dtype": None}


def _hf_device_map(module: Any) -> Any:
    for attr in ["hf_device_map", "device_map"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def _assert_no_offload(device_map: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(device_map, dict):
        for name, device in device_map.items():
            text = str(device).lower()
            if "cpu" in text or "disk" in text or "offload" in text:
                errors.append(f"{name} -> {device}")
    return errors


def load_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(args.openvla_repo)))
    import experiments.robot.openvla_utils as official_openvla_utils

    # Preserve the exact downloaded checkpoint files and checksums. The
    # official loader syncs local checkpoint modeling/config files to the
    # current source tree when they differ; the HF checkpoint already ships
    # trusted remote-code files, so bypass only that mutating local-path sync.
    official_openvla_utils.update_auto_map = lambda pretrained_checkpoint: None
    official_openvla_utils.check_model_logic_mismatch = lambda pretrained_checkpoint: None

    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.openvla_utils import get_vla_action

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(Path(args.checkpoint_dir)),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        task_suite_name=str(args.task_suite_name),
        num_trials_per_task=1,
        seed=int(args.seed),
    )
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant": "int4" if args.load_in_4bit else "int8" if args.load_in_8bit else "forbidden_full_precision",
        "config": cfg.__dict__.copy(),
        "hard_rules": {
            "cpu_offload_allowed": False,
            "disk_offload_allowed": False,
            "full_bf16_attempt_allowed": False,
        },
        "success": False,
        "exception": None,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CPU_FALLBACK_BUG: CUDA unavailable before model load")
        if not (cfg.load_in_4bit ^ cfg.load_in_8bit):
            raise RuntimeError("Exactly one quantized mode must be enabled")
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        device_map = _hf_device_map(model)
        offload_errors = _assert_no_offload(device_map)
        if offload_errors:
            raise RuntimeError("OFFLOAD_FORBIDDEN: " + "; ".join(offload_errors))
        obs = {
            "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "state": np.zeros((8,), dtype=np.float32),
        }
        action_started = time.monotonic()
        actions = get_vla_action(
            cfg,
            model,
            processor,
            obs,
            "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=False,
        )
        action_array = np.asarray(actions)
        report.update(
            {
                "success": True,
                "load_seconds": round(action_started - started, 3),
                "one_chunk_seconds": round(time.monotonic() - action_started, 3),
                "model_parameter": _first_parameter_device(model),
                "action_head_parameter": _first_parameter_device(action_head),
                "proprio_projector_parameter": _first_parameter_device(proprio_projector),
                "model_hf_device_map": device_map,
                "offload_status": "NO_CPU_OR_DISK_OFFLOAD_DETECTED",
                "action_chunk_shape": list(action_array.shape),
                "action_chunk": action_array.tolist(),
                "action_dtype": str(action_array.dtype),
                "action_range": {
                    "min": float(np.nanmin(action_array)),
                    "max": float(np.nanmax(action_array)),
                    "finite": bool(np.isfinite(action_array).all()),
                },
                "unnormalization_key": cfg.unnorm_key,
                "cuda_memory": {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                },
                "autocast": {
                    "cuda": bool(torch.is_autocast_enabled("cuda")),
                    "cpu": bool(torch.is_autocast_enabled("cpu")),
                },
            }
        )
        if not str(report["model_parameter"].get("device", "")).startswith("cuda"):
            raise RuntimeError("CPU_FALLBACK_BUG: first model parameter is not on CUDA")
        if action_array.shape[-1] != 7:
            raise RuntimeError(f"ROLLOUT_SCHEMA_OR_ACTION_MISMATCH: expected 7D actions, got {action_array.shape}")
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["success"] = False
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if torch.cuda.is_available():
            report.setdefault(
                "cuda_memory",
                {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                },
            )
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


def episode_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from libero.libero import benchmark

    sys.path.insert(0, str(Path(args.openvla_repo)))
    import experiments.robot.openvla_utils as official_openvla_utils

    official_openvla_utils.update_auto_map = lambda pretrained_checkpoint: None
    official_openvla_utils.check_model_logic_mismatch = lambda pretrained_checkpoint: None

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        save_rollout_video,
    )
    from experiments.robot.libero.run_libero_eval import (
        TASK_MAX_STEPS,
        GenerateConfig,
        initialize_model,
        prepare_observation,
        process_action,
    )
    from experiments.robot.robot_utils import get_action, get_image_resize_size

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(Path(args.checkpoint_dir)),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        task_suite_name=str(args.task_suite_name),
        num_trials_per_task=1,
        seed=int(args.seed),
    )
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant": "int4" if args.load_in_4bit else "int8" if args.load_in_8bit else "forbidden_full_precision",
        "config": cfg.__dict__.copy(),
        "task_suite_name": str(args.task_suite_name),
        "task_id": int(args.task_id),
        "initial_state_index": int(args.initial_state_index),
        "success": False,
        "exception": None,
        "offload_status": None,
    }
    env = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CPU_FALLBACK_BUG: CUDA unavailable before model load")
        if not (cfg.load_in_4bit ^ cfg.load_in_8bit):
            raise RuntimeError("Exactly one quantized mode must be enabled")
        load_started = time.monotonic()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)
        device_map = _hf_device_map(model)
        offload_errors = _assert_no_offload(device_map)
        if offload_errors:
            raise RuntimeError("OFFLOAD_FORBIDDEN: " + "; ".join(offload_errors))
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        task = task_suite.get_task(int(args.task_id))
        initial_states = task_suite.get_task_init_states(int(args.task_id))
        if int(args.initial_state_index) >= len(initial_states):
            raise IndexError(f"initial_state_index {args.initial_state_index} >= {len(initial_states)}")
        initial_state = initial_states[int(args.initial_state_index)]
        env_started = time.monotonic()
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        env_creation_seconds = time.monotonic() - env_started
        env.reset()
        obs = env.set_init_state(initial_state)
        action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
        max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
        replay_images: list[Any] = []
        policy_latencies: list[float] = []
        env_latencies: list[float] = []
        chunk_shapes: list[list[int]] = []
        chunk_ranges: list[dict[str, Any]] = []
        action_chunks = 0
        final_reward = 0.0
        success = False
        done = False
        t = 0
        while t < max_steps + cfg.num_steps_wait:
            step_started = time.monotonic()
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                env_latencies.append(time.monotonic() - step_started)
                final_reward = float(reward)
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            if len(action_queue) == 0:
                policy_started = time.monotonic()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=False,
                )
                policy_latencies.append(time.monotonic() - policy_started)
                action_array = np.asarray(actions)
                action_chunks += 1
                chunk_shapes.append(list(action_array.shape))
                chunk_ranges.append(
                    {
                        "min": float(np.nanmin(action_array)),
                        "max": float(np.nanmax(action_array)),
                        "finite": bool(np.isfinite(action_array).all()),
                    }
                )
                action_queue.extend(actions)
            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            env_latencies.append(time.monotonic() - step_started)
            final_reward = float(reward)
            if done:
                success = True
                break
            t += 1
        video_path = save_rollout_video(
            replay_images,
            idx=int(args.video_index),
            success=success,
            task_description=task_description,
            log_file=None,
        )
        report.update(
            {
                "success": bool(success),
                "done": bool(done),
                "task_description": task_description,
                "task_bddl_file": getattr(task, "bddl_file", None),
                "task_problem_folder": getattr(task, "problem_folder", None),
                "initial_state_shape": list(np.asarray(initial_state).shape),
                "load_seconds": round(time.monotonic() - load_started, 3),
                "env_creation_seconds": round(env_creation_seconds, 3),
                "steps": int(t),
                "max_steps": int(max_steps),
                "num_steps_wait": int(cfg.num_steps_wait),
                "final_reward": final_reward,
                "action_chunk_count": int(action_chunks),
                "action_chunk_shapes": chunk_shapes,
                "action_chunk_ranges": chunk_ranges,
                "policy_latency_seconds": {
                    "count": len(policy_latencies),
                    "mean": round(float(np.mean(policy_latencies)), 6) if policy_latencies else None,
                    "max": round(float(np.max(policy_latencies)), 6) if policy_latencies else None,
                },
                "environment_latency_seconds": {
                    "count": len(env_latencies),
                    "mean": round(float(np.mean(env_latencies)), 6) if env_latencies else None,
                    "max": round(float(np.max(env_latencies)), 6) if env_latencies else None,
                },
                "model_parameter": _first_parameter_device(model),
                "action_head_parameter": _first_parameter_device(action_head),
                "proprio_projector_parameter": _first_parameter_device(proprio_projector),
                "model_hf_device_map": device_map,
                "offload_status": "NO_CPU_OR_DISK_OFFLOAD_DETECTED",
                "unnormalization_key": cfg.unnorm_key,
                "video_path": str(video_path),
                "cuda_memory": {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                },
                "autocast": {
                    "cuda": bool(torch.is_autocast_enabled("cuda")),
                    "cpu": bool(torch.is_autocast_enabled("cpu")),
                },
            }
        )
        if not str(report["model_parameter"].get("device", "")).startswith("cuda"):
            raise RuntimeError("CPU_FALLBACK_BUG: first model parameter is not on CUDA")
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["success"] = False
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if torch.cuda.is_available():
            report.setdefault(
                "cuda_memory",
                {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 3),
                    "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
                },
            )
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


def build_hard_slice_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_specs = _parse_task_specs(getattr(args, "task_specs", ""))
    reset_identities = _parse_reset_identities(getattr(args, "reset_identities", ""))
    manifest_label = _manifest_label(args)
    episodes = []
    tasks = []
    episode_index = 0
    for task_spec in task_specs:
        suite = str(task_spec["suite"])
        task_id = int(task_spec["task_id"])
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_record = {
            **task_spec,
            "instruction": str(getattr(task, "language", "")),
            "bddl_file": getattr(task, "bddl_file", None),
            "problem_folder": getattr(task, "problem_folder", None),
            "initial_state_count": int(len(initial_states)),
        }
        tasks.append(task_record)
        for identity in reset_identities:
            init_index = _identity_to_initial_state_index(identity)
            initial_state = initial_states[init_index]
            initial_array = np.asarray(initial_state)
            episodes.append(
                {
                    "planned_episode_index": int(episode_index),
                    "episode_id": f"quantized_openvla_oft_int4|{suite}|task_{task_id}|identity_{identity}",
                    "policy": "quantized_openvla_oft_int4",
                    "suite": suite,
                    "task_id": task_id,
                    "role": str(task_spec["role"]),
                    "instruction": str(getattr(task, "language", "")),
                    "reset_identity": int(identity),
                    "initial_state_index": int(init_index),
                    "initial_state_shape": [int(dim) for dim in initial_array.shape],
                    "initial_state_dtype": str(initial_array.dtype),
                    "initial_state_sha256": _init_state_sha256(initial_state),
                }
            )
            episode_index += 1
    payload = {
        "schema_version": 1,
        "date_kst": "2026-07-11",
        "manifest_label": manifest_label,
        "policy": "quantized_openvla_oft_int4",
        "quantized": True,
        "full_precision_claim": False,
        "reset_identities": list(reset_identities),
        "identity_mapping_rule": "reset identity label 20260711 + n maps to official LIBERO initial_state index n per task",
        "prior_smolvla_reuse_status": "not_reused_without_matched_exact_init_rerun",
        "prior_smolvla_identity_caveat": (
            "Earlier LeRobot runs used reset_seed labels, but LiberoEnv increments init_state_id on explicit reset "
            "and again after success auto-reset; exact video-backed init-state-array identity therefore requires "
            "the matched exact-init rerun."
        ),
        "tasks": tasks,
        "planned_episode_count": len(episodes),
        "max_total_episode_budget_with_smolvla": 2 * len(episodes),
        "episodes": episodes,
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "canonical_payload_sha256"}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def hard_slice_rollout(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from libero.libero import benchmark

    sys.path.insert(0, str(Path(args.openvla_repo)))
    import experiments.robot.openvla_utils as official_openvla_utils

    official_openvla_utils.update_auto_map = lambda pretrained_checkpoint: None
    official_openvla_utils.check_model_logic_mismatch = lambda pretrained_checkpoint: None

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        save_rollout_video,
    )
    from experiments.robot.libero.run_libero_eval import (
        TASK_MAX_STEPS,
        GenerateConfig,
        check_unnorm_key,
        initialize_model,
        prepare_observation,
        process_action,
    )
    from experiments.robot.openvla_utils import DEVICE as OPENVLA_INPUT_DEVICE
    from experiments.robot.robot_utils import get_action, get_image_resize_size

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(Path(args.checkpoint_dir)),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        task_suite_name="libero_spatial",
        num_trials_per_task=1,
        seed=int(args.seed),
    )
    started = time.monotonic()
    manifest = build_hard_slice_manifest(args)
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant": "int4" if args.load_in_4bit else "int8" if args.load_in_8bit else "forbidden_full_precision",
        "policy": "quantized_openvla_oft_int4" if args.load_in_4bit else "quantized_openvla_oft_int8",
        "quantized": True,
        "full_precision_claim": False,
        "config": cfg.__dict__.copy(),
        "manifest": manifest,
        "episodes": [],
        "errors": [],
        "success": False,
        "exception": None,
        "offload_status": None,
    }
    model = None
    env = None
    identical_error_counts: dict[str, int] = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CPU_FALLBACK_BUG: CUDA unavailable before model load")
        if not (cfg.load_in_4bit ^ cfg.load_in_8bit):
            raise RuntimeError("Exactly one quantized mode must be enabled")
        load_started = time.monotonic()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)
        device_map = _hf_device_map(model)
        offload_errors = _assert_no_offload(device_map)
        if offload_errors:
            raise RuntimeError("OFFLOAD_FORBIDDEN: " + "; ".join(offload_errors))
        report.update(
            {
                "load_seconds": _round(time.monotonic() - load_started, 3),
                "model_parameter": _first_parameter_device(model),
                "action_head_parameter": _first_parameter_device(action_head),
                "proprio_projector_parameter": _first_parameter_device(proprio_projector),
                "model_hf_device_map": device_map,
                "offload_status": "NO_CPU_OR_DISK_OFFLOAD_DETECTED",
                "openvla_input_device_constant": str(OPENVLA_INPUT_DEVICE),
            }
        )
        if not str(report["model_parameter"].get("device", "")).startswith("cuda"):
            raise RuntimeError("CPU_FALLBACK_BUG: first model parameter is not on CUDA")

        benchmark_dict = benchmark.get_benchmark_dict()
        suite_cache: dict[str, Any] = {}
        for planned in manifest["episodes"]:
            episode_started = time.monotonic()
            suite = str(planned["suite"])
            task_id = int(planned["task_id"])
            identity = int(planned["reset_identity"])
            initial_state_index = int(planned["initial_state_index"])
            row: dict[str, Any] = {
                **planned,
                "success": False,
                "done": False,
                "exception": None,
                "video_path": None,
                "offload_status": "NO_CPU_OR_DISK_OFFLOAD_DETECTED",
            }
            try:
                cfg.task_suite_name = suite
                check_unnorm_key(cfg, model)
                if suite not in suite_cache:
                    suite_cache[suite] = benchmark_dict[suite]()
                task_suite = suite_cache[suite]
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                initial_state = initial_states[initial_state_index]
                row["unnormalization_key"] = cfg.unnorm_key
                row["initial_state_sha256_runtime"] = _init_state_sha256(initial_state)
                if row["initial_state_sha256_runtime"] != row["initial_state_sha256"]:
                    raise RuntimeError("INITIAL_STATE_MANIFEST_MISMATCH")
                env_started = time.monotonic()
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
                row["env_creation_seconds"] = _round(time.monotonic() - env_started, 3)
                env.reset()
                obs = env.set_init_state(initial_state)
                action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
                replay_images: list[Any] = []
                policy_latencies: list[float] = []
                env_latencies: list[float] = []
                chunk_shapes: list[list[int]] = []
                chunk_ranges: list[dict[str, Any]] = []
                action_chunks = 0
                final_reward = 0.0
                done = False
                episode_success = False
                t = 0
                first_policy_input_device = str(OPENVLA_INPUT_DEVICE)
                while t < max_steps + cfg.num_steps_wait:
                    step_started = time.monotonic()
                    if t < cfg.num_steps_wait:
                        obs, reward, done, _info = env.step(get_libero_dummy_action(cfg.model_family))
                        env_latencies.append(time.monotonic() - step_started)
                        final_reward = float(reward)
                        t += 1
                        continue

                    observation, img = prepare_observation(obs, resize_size)
                    replay_images.append(img)
                    if len(action_queue) == 0:
                        policy_started = time.monotonic()
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=False,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        policy_latencies.append(time.monotonic() - policy_started)
                        action_array = np.asarray(actions)
                        action_chunks += 1
                        chunk_shapes.append(list(action_array.shape))
                        chunk_ranges.append(
                            {
                                "min": float(np.nanmin(action_array)),
                                "max": float(np.nanmax(action_array)),
                                "finite": bool(np.isfinite(action_array).all()),
                            }
                        )
                        action_queue.extend(actions)
                    action = process_action(action_queue.popleft(), cfg.model_family)
                    obs, reward, done, _info = env.step(action.tolist())
                    env_latencies.append(time.monotonic() - step_started)
                    final_reward = float(reward)
                    if done:
                        episode_success = True
                        break
                    t += 1
                video_path = save_rollout_video(
                    replay_images,
                    idx=110000 + int(planned["planned_episode_index"]),
                    success=episode_success,
                    task_description=task_description,
                    log_file=None,
                )
                row.update(
                    {
                        "success": bool(episode_success),
                        "done": bool(done),
                        "task_description": task_description,
                        "task_bddl_file": getattr(task, "bddl_file", None),
                        "task_problem_folder": getattr(task, "problem_folder", None),
                        "steps": int(t),
                        "max_steps": int(max_steps),
                        "num_steps_wait": int(cfg.num_steps_wait),
                        "final_reward": float(final_reward),
                        "action_chunk_count": int(action_chunks),
                        "action_chunk_shapes": chunk_shapes,
                        "action_chunk_ranges": chunk_ranges,
                        "input_tensor_devices": {
                            "official_openvla_utils_DEVICE": first_policy_input_device,
                            "note": "official get_vla_action moves processor inputs to this DEVICE before predict_action",
                        },
                        "policy_latency_seconds": {
                            "count": len(policy_latencies),
                            "mean": _round(float(np.mean(policy_latencies)), 6) if policy_latencies else None,
                            "max": _round(float(np.max(policy_latencies)), 6) if policy_latencies else None,
                        },
                        "environment_latency_seconds": {
                            "count": len(env_latencies),
                            "mean": _round(float(np.mean(env_latencies)), 6) if env_latencies else None,
                            "max": _round(float(np.max(env_latencies)), 6) if env_latencies else None,
                        },
                        "video_path": str(video_path),
                        "cuda_memory": _cuda_memory_payload(torch),
                        "rss_mib": _rss_mib(),
                        "autocast": _autocast_payload(torch),
                        "elapsed_seconds": _round(time.monotonic() - episode_started, 3),
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime boundary
                error_key = f"{type(exc).__name__}:{str(exc)[:160]}"
                identical_error_counts[error_key] = identical_error_counts.get(error_key, 0) + 1
                row["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
                row["cuda_memory"] = _cuda_memory_payload(torch)
                row["rss_mib"] = _rss_mib()
                report["errors"].append({"episode_id": row["episode_id"], **row["exception"]})
            finally:
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
                report["episodes"].append(row)
            if identical_error_counts and max(identical_error_counts.values()) >= 2:
                report["stopped_early_reason"] = "two_identical_infrastructure_failures"
                break
        report["success"] = len(report["episodes"]) == manifest["planned_episode_count"] and not report["errors"]
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["success"] = False
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        report["completed_episode_count"] = len(report.get("episodes", []))
        report["successful_episode_count"] = sum(1 for item in report.get("episodes", []) if item.get("success"))
        report["infrastructure_failure_count"] = len(report.get("errors", []))
        report["cuda_memory"] = _cuda_memory_payload(torch)
        report["rss_mib"] = _rss_mib()
        report["autocast"] = _autocast_payload(torch)
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "env-lock",
            "download",
            "checksum",
            "memory-preflight",
            "load-smoke",
            "episode-smoke",
            "hard-slice-manifest",
            "hard-slice-rollout",
        ],
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--openvla-repo", default=str(DEFAULT_OPENVLA_REPO))
    parser.add_argument("--libero-repo", default=str(DEFAULT_LIBERO_REPO))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument(
        "--task-specs",
        default="",
        help="Optional comma-separated suite:task_id[:role] entries for manifest-controlled rollouts.",
    )
    parser.add_argument(
        "--reset-identities",
        default="",
        help="Optional comma-separated reset labels; 20260711+n maps to official LIBERO initial_state index n.",
    )
    parser.add_argument("--manifest-label", default="")
    parser.add_argument("--video-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(argv)

    if args.command == "env-lock":
        payload = build_environment_lock(args)
    elif args.command == "download":
        payload = download_checkpoint(args)
    elif args.command == "checksum":
        payload = checksum_checkpoint(args)
    elif args.command == "memory-preflight":
        payload = memory_preflight(args)
    elif args.command == "load-smoke":
        payload = load_smoke(args)
    elif args.command == "episode-smoke":
        payload = episode_smoke(args)
    elif args.command == "hard-slice-manifest":
        payload = build_hard_slice_manifest(args)
    elif args.command == "hard-slice-rollout":
        payload = hard_slice_rollout(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)

    out = Path(args.out) if args.out else Path(args.run_dir) / f"{args.command}.json"
    _write_json(out, payload)
    print(json.dumps({"command": args.command, "out": str(out), "success": payload.get("success")}, indent=2))
    if payload.get("decision") in {"RTX5080_MEMORY_BLOCKED"}:
        return 2
    if args.command in {"load-smoke", "episode-smoke", "hard-slice-rollout"} and payload.get("exception"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
