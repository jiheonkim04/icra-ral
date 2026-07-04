"""Bounded SmolVLA load-only smoke guard.

This module is intentionally conservative. It checks gates, local files, runtime
dependencies, and memory before any heavy import or model load can happen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RUNTIME_DEPENDENCIES = [
    "torch",
    "transformers",
    "lerobot",
    "safetensors",
    "num2words",
    "draccus",
    "datasets",
    "imageio",
    "diffusers",
    "serial",
    "deepdiff",
    "av",
    "einops",
]
FORBIDDEN_GATES = ["ALLOW_GPU_TRAINING", "ALLOW_ROLLOUTS", "ALLOW_CLOUD_HANDOFF"]
MAX_LOAD_ONLY_SECONDS = 600


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _find_files(root: Path, names: list[str], patterns: list[str] | None = None) -> list[str]:
    if not root.exists():
        return []
    found: list[str] = []
    for name in names:
        if (root / name).exists():
            found.append(name)
    for pattern in patterns or []:
        found.extend(path.name for path in root.glob(pattern) if path.is_file())
    return sorted(set(found))


def _read_tokenizer_dependency(root: Path) -> str | None:
    preprocessor = root / "policy_preprocessor.json"
    if not preprocessor.exists():
        return None
    try:
        data = json.loads(preprocessor.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for step in data.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            tokenizer_name = step.get("config", {}).get("tokenizer_name")
            if tokenizer_name:
                return str(tokenizer_name)
    return None


def _dependency_roots(dependency_name: str | None, base_roots: list[Path]) -> list[Path]:
    if not dependency_name or "/" not in dependency_name:
        return []
    org, repo = dependency_name.split("/", 1)
    roots: list[Path] = []
    for base in base_roots:
        if not base.exists():
            continue
        plain = base / org / repo
        roots.append(plain)
        hub_root = base / f"models--{org}--{repo}"
        roots.append(hub_root)
        snapshots = hub_root / "snapshots"
        if snapshots.exists():
            roots.extend(path for path in snapshots.iterdir() if path.is_dir())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _external_tokenizer_files(dependency_name: str | None, base_roots: list[Path]) -> dict[str, Any]:
    expected = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "config.json",
    ]
    roots = _dependency_roots(dependency_name, base_roots)
    for root in roots:
        found = _find_files(root, expected)
        if found:
            return {
                "name": dependency_name,
                "found": True,
                "root": str(root),
                "files_found": found,
                "candidate_roots": [str(path) for path in roots],
            }
    return {
        "name": dependency_name,
        "found": False,
        "root": None,
        "files_found": [],
        "candidate_roots": [str(path) for path in roots],
    }


def _runtime_dependencies() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in RUNTIME_DEPENDENCIES}


def _nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "gpu_name": None, "memory_total_mb": None, "memory_used_mb": None}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "gpu_name": None, "memory_total_mb": None, "memory_used_mb": None}
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return {"available": False, "gpu_name": None, "memory_total_mb": None, "memory_used_mb": None}
    return {
        "available": True,
        "gpu_name": parts[0],
        "memory_total_mb": int(parts[1]),
        "memory_used_mb": int(parts[2]),
    }


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def _load_smolvla_policy(
    smolvla_ckpt: Path,
    hf_home: Path,
    external_dependency: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    started = time.monotonic()
    memory_before = _rss_mb()
    gpu_before = _nvidia_smi()

    config = PreTrainedConfig.from_pretrained(
        smolvla_ckpt,
        local_files_only=True,
        cache_dir=hf_home,
    )
    config.device = device
    config.load_vlm_weights = False
    config.compile_model = False
    config.push_to_hub = False
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

    parameter_count = sum(param.numel() for param in policy.parameters())
    trainable_parameter_count = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    model_device = next(policy.parameters()).device.type
    elapsed = time.monotonic() - started

    cuda_max_allocated_mb = None
    if torch.cuda.is_available():
        cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    # Drop the model immediately: this is load-only validation, not an interactive session.
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "load_elapsed_sec": round(elapsed, 3),
        "rss_before_mb": memory_before,
        "rss_after_mb": _rss_mb(),
        "gpu_before": gpu_before,
        "gpu_after": _nvidia_smi(),
        "cuda_max_allocated_mb": cuda_max_allocated_mb,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "device": model_device,
        "config_device": config.device,
        "vlm_model_name": config.vlm_model_name,
        "load_vlm_weights": config.load_vlm_weights,
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
    forbidden_gates_set = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    files_ready = bool(config_files and weight_files and (tokenizer_files or external_dependency["found"]))
    deps_ready = all(deps.values())
    gpu_memory_ok = bool(gpu["available"] and gpu["memory_total_mb"] and gpu["memory_total_mb"] <= 16384)
    vram_policy_ok = bool(gpu["available"] and gpu["memory_total_mb"] and gpu["memory_total_mb"] >= 14048)

    exit_code = 0
    blocked_reason = None
    load_attempted = False
    load_result: dict[str, Any] | None = None
    load_error: dict[str, Any] | None = None
    if not heavy_gate:
        exit_code = 2
        blocked_reason = "ALLOW_HEAVY_IMPORT=1 is required and may be set only inside the bounded SmolVLA autonomous load-only task."
    elif forbidden_gates_set:
        exit_code = 3
        blocked_reason = f"Forbidden gate(s) set: {', '.join(forbidden_gates_set)}"
    elif not files_ready:
        exit_code = 4
        blocked_reason = "SmolVLA checkpoint/tokenizer/weights readiness is incomplete."
    elif not deps_ready:
        exit_code = 5
        missing = [name for name, present in deps.items() if not present]
        blocked_reason = f"Missing runtime dependencies: {', '.join(missing)}"
    elif not gpu_memory_ok or not vram_policy_ok:
        exit_code = 6
        blocked_reason = "GPU memory policy check failed before load."
    else:
        try:
            load_attempted = True
            load_result = _load_smolvla_policy(
                smolvla_ckpt=smolvla_ckpt,
                hf_home=hf_home,
                external_dependency=external_dependency,
                device=args.device,
            )
            if load_result["load_elapsed_sec"] > MAX_LOAD_ONLY_SECONDS:
                exit_code = 8
                blocked_reason = "Load-only smoke exceeded the 10 minute runtime budget."
            elif (load_result.get("cuda_max_allocated_mb") or 0) > 14336:
                exit_code = 9
                blocked_reason = "Load-only smoke exceeded the 14GB VRAM budget."
        except Exception as exc:  # noqa: BLE001 - report exact load-only smoke failure.
            exit_code = 8
            blocked_reason = f"SmolVLA load-only construction failed: {type(exc).__name__}: {exc}"
            load_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    report = {
        "policy": {
            "load_only": True,
            "downloads_performed": False,
            "heavy_model_imports_performed": load_attempted,
            "model_load_performed": bool(load_result),
            "model_inference_performed": False,
            "gpu_training_performed": False,
            "training_performed": False,
            "real_rollouts_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "heavy_import_gate_set": heavy_gate,
            "forbidden_gates_set": forbidden_gates_set,
            "max_vram_policy_mb": 14336,
            "max_runtime_sec": MAX_LOAD_ONLY_SECONDS,
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
        "load": load_result,
        "load_error": load_error,
        "result": {
            "passed": exit_code == 0,
            "blocked": exit_code != 0,
            "blocked_reason": blocked_reason,
            "elapsed_sec": round(time.monotonic() - started, 3),
        },
    }
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--report-path", default="reports/smolvla_load_only_smoke_report.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

