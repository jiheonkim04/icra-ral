"""Bounded repeated offline action decoding with VLM weights enabled."""

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

import numpy as np

from tca_map.smolvla.interface_adapters import ACTION_STRATEGY_GRIPPER_CLOSE, adapt_policy_action_to_env_action
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
)
from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)
from tca_map.smolvla.offline_demo_action_decoding import _load_first_hdf5_sample


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
VLM_RECHECK_GATE = "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING"
MAX_POLICY_CALLS = 3
MAX_RUNTIME_SECONDS = 1200
MAX_VRAM_MB = 14336
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


def _finite_list(values: np.ndarray, limit: int | None = None) -> list[float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [round(float(x), 6) for x in flat]


def _alignment_signal(mean_l1: float | None, valid: bool) -> str:
    if not valid or mean_l1 is None:
        return "invalid"
    if mean_l1 <= 0.1:
        return "strong"
    if mean_l1 <= 0.25:
        return "moderate"
    return "weak"


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    metrics = report.get("metrics") or {}
    policy = report.get("policy") or {}
    lines = [
        "# VLM-Enabled Repeated Offline Decoding Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{report.get('vlm_enabled_repeated_offline_decoding_passed')}`",
        f"- evidence label: `{report.get('evidence_label')}`",
        f"- sample count: `{metrics.get('sample_count')}`",
        f"- timesteps: `{metrics.get('timesteps')}`",
        f"- load_vlm_weights: `{metrics.get('load_vlm_weights')}`",
        f"- mean action L1 to expert: `{metrics.get('mean_action_l1_to_expert')}`",
        f"- mean action MSE to expert: `{metrics.get('mean_action_mse_to_expert')}`",
        f"- mean action L1 delta vs previous no-VLM: `{metrics.get('mean_action_l1_delta_vs_previous')}`",
        f"- mean action MSE delta vs previous no-VLM: `{metrics.get('mean_action_mse_delta_vs_previous')}`",
        f"- offline alignment signal: `{metrics.get('offline_alignment_signal')}`",
        f"- clipped values total: `{metrics.get('clipped_values_total')}`",
        f"- CUDA max allocated MB: `{metrics.get('cuda_max_allocated_mb')}`",
        "",
        "Policy flags:",
        "",
        f"- downloads performed: `{policy.get('downloads_performed')}`",
        f"- training performed: `{policy.get('training_performed')}`",
        f"- rollouts performed: `{policy.get('rollouts_performed')}`",
        f"- GPU jobs performed: `{policy.get('gpu_jobs_performed')}`",
        f"- OpenVLA-OFT executed: `{policy.get('openvla_oft_executed')}`",
        f"- paper-grade claims made: `{policy.get('paper_grade_claims_made')}`",
        "",
        f"Recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_policy_with_vlm(smolvla_ckpt: Path, hf_home: Path, external_dependency: dict[str, Any], device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(
        smolvla_ckpt,
        local_files_only=True,
        cache_dir=hf_home,
    )
    config.device = device
    config.load_vlm_weights = True
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


PolicyLoader = Callable[[Path, Path, dict[str, Any], str], Any]


def _decode_one(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timestep: int,
    task_text: str,
    device: str,
) -> dict[str, Any]:
    import torch

    sample = _load_first_hdf5_sample(hdf5_path, demo_name, timestep)
    expert_action = np.asarray(sample["expert_action"], dtype=np.float32).reshape(-1)
    batch, batch_metadata = _build_batch(
        config,
        tokenizer_root,
        sample["obs"],
        task_text,
        device,
        CAMERA_ALIAS_STRATEGY_CURRENT,
        STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    )
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)
    inference_started = time.monotonic()
    with torch.inference_mode():
        policy_action = policy.select_action(batch, noise=noise)
    inference_elapsed = time.monotonic() - inference_started

    adapter = adapt_policy_action_to_env_action(
        policy_action,
        int(expert_action.shape[0]),
        strategy=ACTION_STRATEGY_GRIPPER_CLOSE,
        action_scale=1.0,
    )
    adapted = np.asarray(adapter.values, dtype=np.float32)
    policy_np = policy_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
    prefix = min(6, expert_action.shape[0], policy_np.shape[0])
    return {
        "demo_name": sample["metadata"]["demo_name"],
        "timestep": int(timestep),
        "task": task_text,
        "inference_elapsed_sec": round(inference_elapsed, 6),
        "action_l1_to_expert": round(float(np.mean(np.abs(adapted - expert_action))), 6),
        "action_mse_to_expert": round(float(np.mean((adapted - expert_action) ** 2)), 6),
        "policy6_l1_to_expert_first6": round(float(np.mean(np.abs(policy_np[:prefix] - expert_action[:prefix]))), 6),
        "action_finite": bool(np.isfinite(adapted).all() and np.isfinite(policy_np).all()),
        "policy_action_shape": list(policy_np.shape),
        "expert_action_shape": list(expert_action.shape),
        "policy_action_preview": _finite_list(policy_np, 6),
        "adapted_action_preview": _finite_list(adapted, int(expert_action.shape[0])),
        "expert_action_preview": _finite_list(expert_action, int(expert_action.shape[0])),
        "action_adapter_metadata": adapter.metadata,
        "batch_metadata": batch_metadata,
        "sample_metadata": sample["metadata"],
    }


def _plan_timesteps(plan: dict[str, Any]) -> list[int]:
    items = (((plan.get("inputs") or {}).get("selected_timesteps")) or [])
    if not items:
        items = (((plan.get("planned_sample") or {}).get("hdf5") or {}).get("selected_timesteps") or [])
    cleaned = []
    for item in items:
        value = int(item)
        if value >= 0 and value not in cleaned:
            cleaned.append(value)
    return cleaned[:MAX_POLICY_CALLS]


def build_report(args: argparse.Namespace, loader: PolicyLoader = _load_policy_with_vlm) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)
    plan_path = Path(args.plan_report)
    previous_path = Path(args.previous_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    deps = _runtime_dependencies()
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])

    report: dict[str, Any] = {
        "evidence_label": "vlm_enabled_repeated_offline_decoding",
        "vlm_enabled_repeated_offline_decoding_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "bounded_vlm_enabled_repeated_offline_decoding": True,
            "task_local_gates_required": [f"{HEAVY_IMPORT_GATE}=1", f"{VLM_RECHECK_GATE}=1"],
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
            "vlm_recheck_gate_set": _env_flag(VLM_RECHECK_GATE),
            "forbidden_gates_set": forbidden,
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
            "previous_report": str(previous_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "hf_home": str(hf_home),
            "checkpoint_root": str(checkpoint_root),
        },
        "files": {
            "config_found": _find_files(smolvla_ckpt, ["config.json"]),
            "weights_found": _find_files(smolvla_ckpt, ["model.safetensors"], ["*.safetensors"]),
            "external_tokenizer_dependency": external_dependency,
        },
        "runtime_dependencies": deps,
        "gpu": _nvidia_smi(),
        "runtime": {"rss_before_mb": _rss_mb(), "rss_after_mb": None, "elapsed_sec": None},
        "baseline_to_compare": {},
        "samples": [],
        "metrics": {},
        "error": None,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return block(f"{HEAVY_IMPORT_GATE}=1 is required only inside this bounded offline recheck.", 2)
    if not report["policy"]["vlm_recheck_gate_set"]:
        return block(f"{VLM_RECHECK_GATE}=1 is required only inside this bounded offline recheck.", 3)
    if forbidden:
        return block("Forbidden gate(s) set: " + ", ".join(forbidden), 4)
    if args.device != "cpu":
        return block("The VLM-enabled repeated offline decoding recheck is CPU-only.", 5)
    if not plan_path.exists():
        return block(f"VLM-enabled repeated offline plan report is missing: {plan_path}", 6)
    if not previous_path.exists():
        return block(f"Previous repeated offline report is missing: {previous_path}", 7)

    try:
        plan = _read_json(plan_path)
        previous = _read_json(previous_path)
        if not plan.get("ready_for_bounded_vlm_enabled_repeated_offline_decoding_runner"):
            return block("VLM-enabled repeated offline plan did not authorize execution.", 8)
        timesteps = _plan_timesteps(plan)
        if not timesteps:
            return block("Plan did not provide selected timesteps.", 9)
        hdf5_path = Path((plan.get("inputs") or {}).get("hdf5_path") or args.hdf5_path)
        if not hdf5_path.exists():
            return block(f"Selected HDF5 file is missing: {hdf5_path}", 10)
        if not all(deps.values()):
            missing = [name for name, present in deps.items() if not present]
            return block("Missing runtime dependencies: " + ", ".join(missing), 11)
        if not external_dependency.get("found"):
            return block("External tokenizer/VLM dependency root is missing.", 12)

        report["baseline_to_compare"] = (plan.get("baseline_to_compare") or previous.get("metrics") or {})
        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = loader(smolvla_ckpt, hf_home, external_dependency, args.device)
        report["policy"]["model_load_performed"] = True
        tokenizer_root = Path(external_dependency["root"])
        task_text = (((plan.get("planned_sample") or {}).get("selected_task_text")) or args.task)
        demo_name = (((plan.get("planned_sample") or {}).get("hdf5") or {}).get("demo_name"))

        total_inference = 0.0
        samples = []
        for timestep in timesteps[:MAX_POLICY_CALLS]:
            item = _decode_one(
                policy=policy,
                config=config,
                tokenizer_root=tokenizer_root,
                hdf5_path=hdf5_path,
                demo_name=demo_name,
                timestep=timestep,
                task_text=task_text,
                device=args.device,
            )
            total_inference += float(item["inference_elapsed_sec"])
            samples.append(item)
        report["policy"]["model_inference_performed"] = True
        report["samples"] = samples

        import torch

        cuda_max = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3) if torch.cuda.is_available() else 0.0
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        finite = all(bool(item["action_finite"]) for item in samples)
        l1 = [float(item["action_l1_to_expert"]) for item in samples]
        mse = [float(item["action_mse_to_expert"]) for item in samples]
        policy6 = [float(item["policy6_l1_to_expert_first6"]) for item in samples]
        clipped = int(sum(int((item.get("action_adapter_metadata") or {}).get("clipped_values") or 0) for item in samples))
        mean_l1 = float(np.mean(l1)) if l1 else None
        mean_mse = float(np.mean(mse)) if mse else None
        signal = _alignment_signal(mean_l1, finite)
        prev_l1 = (previous.get("metrics") or {}).get("mean_action_l1_to_expert")
        prev_mse = (previous.get("metrics") or {}).get("mean_action_mse_to_expert")

        report["metrics"] = {
            "sample_count": len(samples),
            "timesteps": timesteps,
            "mean_action_l1_to_expert": round(mean_l1, 6) if mean_l1 is not None else None,
            "mean_action_mse_to_expert": round(mean_mse, 6) if mean_mse is not None else None,
            "mean_policy6_l1_to_expert_first6": round(float(np.mean(policy6)), 6) if policy6 else None,
            "all_actions_finite": finite,
            "clipped_values_total": clipped,
            "offline_alignment_signal": signal,
            "load_vlm_weights": bool(getattr(config, "load_vlm_weights", False)),
            "device": args.device,
            "cuda_max_allocated_mb": cuda_max,
            "total_policy_inference_elapsed_sec": round(total_inference, 6),
            "mean_action_l1_delta_vs_previous": round(mean_l1 - float(prev_l1), 6) if mean_l1 is not None and prev_l1 is not None else None,
            "mean_action_mse_delta_vs_previous": round(mean_mse - float(prev_mse), 6) if mean_mse is not None and prev_mse is not None else None,
            "evidence_level": "tiny_vlm_enabled_repeated_offline_action_decoding_diagnostic",
        }

        elapsed = time.monotonic() - started
        if not finite:
            return block("At least one decoded action contained non-finite values.", 13)
        if not report["metrics"]["load_vlm_weights"]:
            return block("Loaded config did not keep load_vlm_weights=true.", 14)
        if elapsed > MAX_RUNTIME_SECONDS:
            return block("VLM-enabled repeated offline decoding exceeded the 20 minute runtime budget.", 15)
        if cuda_max > MAX_VRAM_MB:
            return block("VLM-enabled repeated offline decoding exceeded the 14GB VRAM budget.", 16)
        if cuda_max > 0:
            return block("CPU-first VLM-enabled repeated offline decoding unexpectedly allocated CUDA memory.", 17)

        report["vlm_enabled_repeated_offline_decoding_passed"] = True
        report["decision"] = "diagnostic_complete"
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(elapsed, 3)
        report["recommended_next_step"] = (
            "Summarize VLM-enabled versus no-VLM offline decoding. Keep rollout scaling blocked unless alignment improves enough to justify a new risk gate."
        )
        return report, 0
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        report["recommended_next_step"] = "Fix the VLM-enabled repeated offline decoding blocker before any rollout scaling."
        return report, 18


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/vlm_enabled_repeated_offline_decoding_plan_report.json")
    parser.add_argument("--previous-report", default="reports/repeated_offline_demo_action_decoding_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--hdf5-path", default="")
    parser.add_argument("--task", default="perform the task")
    parser.add_argument("--report-path", default="reports/vlm_enabled_repeated_offline_decoding_report.json")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path.suffix == ".json":
        _write_markdown_report(report, report_path.with_suffix(".md"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
