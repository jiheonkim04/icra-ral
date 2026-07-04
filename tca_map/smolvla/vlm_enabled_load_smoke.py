"""Bounded SmolVLA load-only smoke with VLM weights enabled.

This entrypoint is intentionally narrower than inference smoke. It constructs a
local SmolVLA policy with ``load_vlm_weights=True`` on CPU, records memory and
configuration metadata, then immediately releases the model. It must not run
``select_action``, train, rollout, use GPU jobs, download assets, execute
OpenVLA-OFT, access tokens, or make paper claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
VLM_LOAD_GATE = "ALLOW_VLM_ENABLED_LOAD_SMOKE"
MAX_RUNTIME_SECONDS = 900
MAX_VRAM_MB = 14336
MIN_TOTAL_RAM_GB = 20.0

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
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
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _ram_info() -> dict[str, float | None]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_ram_gb": round(mem.total / (1024**3), 3),
            "available_ram_gb": round(mem.available / (1024**3), 3),
        }
    except Exception:
        return {"total_ram_gb": None, "available_ram_gb": None}


def _external_vlm_weight_files(root: Path) -> list[str]:
    return _find_files(root, ["model.safetensors", "pytorch_model.bin"], ["model-*.safetensors", "*.bin"])


def _load_policy_with_vlm(
    *,
    smolvla_ckpt: Path,
    hf_home: Path,
    external_dependency: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    started = time.monotonic()
    rss_before = _rss_mb()
    gpu_before = _nvidia_smi()

    config = PreTrainedConfig.from_pretrained(
        smolvla_ckpt,
        local_files_only=True,
        cache_dir=hf_home,
    )
    config.device = device
    config.load_vlm_weights = True
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
    first_device = next(policy.parameters()).device.type
    cuda_max_allocated_mb = None
    if torch.cuda.is_available():
        cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    result = {
        "load_elapsed_sec": round(time.monotonic() - started, 3),
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(),
        "gpu_before": gpu_before,
        "gpu_after": _nvidia_smi(),
        "cuda_max_allocated_mb": cuda_max_allocated_mb,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "device": first_device,
        "config_device": config.device,
        "vlm_model_name": config.vlm_model_name,
        "load_vlm_weights": bool(config.load_vlm_weights),
    }

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


Loader = Callable[..., dict[str, Any]]


def build_report(args: argparse.Namespace, loader: Loader = _load_policy_with_vlm) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)
    plan_path = Path(args.plan_report)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    external_root = Path(external_dependency["root"]) if external_dependency.get("root") else None
    config_files = _find_files(smolvla_ckpt, ["config.json"])
    smolvla_weight_files = _find_files(
        smolvla_ckpt,
        ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
        ["*.safetensors", "*.bin"],
    )
    vlm_weight_files = _external_vlm_weight_files(external_root) if external_root else []
    deps = _runtime_dependencies()
    ram = _ram_info()
    gpu = _nvidia_smi()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]

    report: dict[str, Any] = {
        "evidence_label": "vlm_enabled_load_smoke",
        "vlm_enabled_load_smoke_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "load_only": True,
            "vlm_enabled": True,
            "device": args.device,
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": _env_flag(HEAVY_IMPORT_GATE),
            "vlm_load_gate_set": _env_flag(VLM_LOAD_GATE),
            "forbidden_gates_set": forbidden,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_vram_mb": MAX_VRAM_MB,
        },
        "claims": {
            "standard_success_claimed": False,
            "benchmark_success_claimed": False,
            "counterfactual_robustness_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "paths": {
            "plan_report": str(plan_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
            "external_dependency_root": str(external_root) if external_root else None,
        },
        "files": {
            "config_found": config_files,
            "smolvla_weights_found": smolvla_weight_files,
            "external_tokenizer_dependency": external_dependency,
            "external_vlm_weights_found": vlm_weight_files,
            "files_ready": bool(config_files and smolvla_weight_files and external_dependency["found"] and vlm_weight_files),
        },
        "runtime_dependencies": deps,
        "ram": ram,
        "gpu": gpu,
        "load": None,
        "error": None,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        report["runtime"] = {"elapsed_sec": round(time.monotonic() - started, 3)}
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return block(f"{HEAVY_IMPORT_GATE}=1 is required only inside this bounded load-only task.", 2)
    if not report["policy"]["vlm_load_gate_set"]:
        return block(f"{VLM_LOAD_GATE}=1 is required only inside this bounded VLM-enabled load-only task.", 3)
    if forbidden:
        return block("Forbidden gate(s) set: " + ", ".join(forbidden), 4)
    if args.device != "cpu":
        return block("The first VLM-enabled load smoke must be CPU-only.", 5)
    if not plan_path.exists():
        return block(f"VLM-enabled load-smoke plan report is missing: {plan_path}", 6)

    try:
        plan = _read_json(plan_path)
        report["plan_summary"] = {
            "decision": plan.get("decision"),
            "ready_for_bounded_vlm_enabled_load_smoke_runner": plan.get(
                "ready_for_bounded_vlm_enabled_load_smoke_runner"
            ),
        }
        if plan.get("decision") != "proceed" or not plan.get("ready_for_bounded_vlm_enabled_load_smoke_runner"):
            return block("VLM-enabled load-smoke plan did not authorize execution.", 7)
        if not report["files"]["files_ready"]:
            return block("SmolVLA or external VLM dependency files are incomplete.", 8)
        if not all(deps.values()):
            missing = [name for name, present in deps.items() if not present]
            return block("Missing runtime dependencies: " + ", ".join(missing), 9)
        total_ram = ram.get("total_ram_gb")
        if total_ram is not None and total_ram < MIN_TOTAL_RAM_GB:
            return block(f"Total RAM is below the bounded VLM load minimum: {total_ram}GB", 10)

        report["policy"]["heavy_model_imports_performed"] = True
        load = loader(
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            external_dependency=external_dependency,
            device=args.device,
        )
        report["load"] = load
        report["policy"]["model_load_performed"] = True

        elapsed = time.monotonic() - started
        if not load.get("load_vlm_weights"):
            return block("Loaded config did not keep load_vlm_weights=true.", 11)
        if elapsed > MAX_RUNTIME_SECONDS:
            return block("VLM-enabled load smoke exceeded the 15 minute runtime budget.", 12)
        if (load.get("cuda_max_allocated_mb") or 0) > MAX_VRAM_MB:
            return block("VLM-enabled load smoke exceeded the 14GB VRAM budget.", 13)
        if (load.get("cuda_max_allocated_mb") or 0) > 0:
            return block("CPU-first VLM-enabled load smoke unexpectedly allocated CUDA memory.", 14)

        report["vlm_enabled_load_smoke_passed"] = True
        report["decision"] = "load_smoke_complete"
        report["runtime"] = {"elapsed_sec": round(elapsed, 3)}
        report["recommended_next_step"] = (
            "Summarize VLM-enabled load result, then plan a bounded repeated offline decoding recheck "
            "with load_vlm_weights=true. Do not run rollout scaling yet."
        )
        return report, 0
    except Exception as exc:  # noqa: BLE001 - exact load-smoke failure matters.
        report["error"] = _compact_error(exc)
        report["runtime"] = {"elapsed_sec": round(time.monotonic() - started, 3)}
        report["recommended_next_step"] = "Fix the VLM-enabled load-smoke blocker or keep load_vlm_weights=false diagnostics."
        return report, 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/vlm_enabled_load_smoke_plan_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--report-path", default="reports/vlm_enabled_load_smoke_report.json")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
