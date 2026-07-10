"""Official WSL LeRobot/LIBERO rollout runner for locked SmolVLA policies.

This module is intentionally narrow: it uses the official LeRobot policy
factory, processors, LIBERO environment, and eval loop.  The only integration
around the official path is resolving persisted PEFT adapters whose metadata
contains Windows-local paths.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

STATIC_MIX_CLASSIFICATION = "DEGENERATE_EQUIVALENT_TO_FROZEN_BASE"
FINAL_DECISIONS = {
    "OFFICIAL_ROLLOUT_BASELINE_READY",
    "CLOSED_LOOP_METHOD_WORTHY_GAP_FOUND",
    "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT",
    "WSL_INSTALL_REBOOT_REQUIRED",
    "WSL_CUDA_BLOCKED",
    "OFFICIAL_LIBERO_ENV_BLOCKED",
    "PEFT_ROLLOUT_LOADING_BLOCKED",
    "ROLLOUT_SCHEMA_OR_ACTION_MISMATCH",
    "ROLLOUT_RUNTIME_TOO_HEAVY",
    "CPU_FALLBACK_BUG",
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    adapter_dir: str | None = None


POLICIES = [
    PolicySpec("frozen_base"),
    PolicySpec("rank4_lora_seed_11", "seed_11"),
    PolicySpec("rank4_lora_seed_22", "seed_22"),
    PolicySpec("rank4_lora_seed_33", "seed_33"),
]


def alpha_zero_static_mix(base_action: Any, lora_action: Any, alpha: float = 0.0) -> np.ndarray:
    """Return action-space static mix; alpha=0 must be exactly frozen base."""
    base = np.asarray(base_action)
    lora = np.asarray(lora_action)
    return alpha * lora + (1.0 - alpha) * base


def static_mix_duplicate_records() -> dict[str, str]:
    return {
        "static_mix_seed_11": STATIC_MIX_CLASSIFICATION,
        "static_mix_seed_22": STATIC_MIX_CLASSIFICATION,
        "static_mix_seed_33": STATIC_MIX_CLASSIFICATION,
    }


def choose_final_decision(report: dict[str, Any]) -> str:
    if report.get("cpu_fallback_bug"):
        return "CPU_FALLBACK_BUG"
    if report.get("wsl_cuda_blocked"):
        return "WSL_CUDA_BLOCKED"
    if report.get("schema_or_action_mismatch"):
        return "ROLLOUT_SCHEMA_OR_ACTION_MISMATCH"
    if report.get("peft_loading_blocked"):
        return "PEFT_ROLLOUT_LOADING_BLOCKED"
    if report.get("official_env_blocked"):
        return "OFFICIAL_LIBERO_ENV_BLOCKED"
    if report.get("runtime_too_heavy"):
        return "ROLLOUT_RUNTIME_TOO_HEAVY"

    smoke = report.get("smoke") or {}
    pilot = report.get("pilot") or {}
    if pilot.get("executed") and pilot.get("planned_episodes"):
        if pilot.get("completed_episodes") == pilot.get("planned_episodes"):
            return "OFFICIAL_ROLLOUT_BASELINE_READY"
        return "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT"
    if not smoke.get("all_policies_executed"):
        return "OFFICIAL_LIBERO_ENV_BLOCKED"
    if not pilot.get("executed"):
        return "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT"
    return "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _set_runtime_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["LIBERO_CONFIG_PATH"] = str(Path(args.libero_config_dir))


def _policy_adapter_path(args: argparse.Namespace, spec: PolicySpec) -> Path | None:
    if spec.adapter_dir is None:
        return None
    return Path(args.lora_root) / spec.adapter_dir


def _torch_autocast(torch_mod: Any) -> dict[str, bool]:
    def enabled(device_type: str) -> bool:
        try:
            return bool(torch_mod.is_autocast_enabled(device_type))
        except TypeError:
            return bool(torch_mod.is_autocast_enabled()) if device_type == "cuda" else False

    return {"cuda": enabled("cuda"), "cpu": enabled("cpu")}


def _cuda_memory(torch_mod: Any) -> dict[str, Any]:
    if not torch_mod.cuda.is_available():
        return {"allocated_bytes": None, "max_allocated_bytes": None}
    return {
        "allocated_bytes": int(torch_mod.cuda.memory_allocated()),
        "max_allocated_bytes": int(torch_mod.cuda.max_memory_allocated()),
        "allocated_mb": _round(torch_mod.cuda.memory_allocated() / (1024 * 1024), 3),
        "max_allocated_mb": _round(torch_mod.cuda.max_memory_allocated() / (1024 * 1024), 3),
    }


def _tensor_devices(batch: dict[str, Any]) -> dict[str, str]:
    return {key: str(value.device) for key, value in batch.items() if hasattr(value, "device")}


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {key: [int(dim) for dim in value.shape] for key, value in batch.items() if hasattr(value, "shape")}


def _first_parameter_summary(policy: Any) -> dict[str, Any]:
    for param in policy.parameters():
        return {"device": str(param.device), "dtype": str(param.dtype), "numel": int(param.numel())}
    return {"device": None, "dtype": None, "numel": 0}


def _dummy_observation(torch_mod: Any) -> dict[str, Any]:
    return {
        "observation.images.image": torch_mod.zeros((1, 3, 256, 256), dtype=torch_mod.float32),
        "observation.images.image2": torch_mod.zeros((1, 3, 256, 256), dtype=torch_mod.float32),
        "observation.robot_state": {
            "eef": {
                "pos": torch_mod.zeros((1, 3), dtype=torch_mod.float32),
                "quat": torch_mod.tensor([[0, 0, 0, 1]], dtype=torch_mod.float32),
            },
            "gripper": {"qpos": torch_mod.zeros((1, 2), dtype=torch_mod.float32)},
        },
        "task": ["pick up the object"],
    }


def _make_env_cfg(task: str, task_ids: list[int] | None = None) -> Any:
    from lerobot.envs.configs import LiberoEnv

    return LiberoEnv(
        task=task,
        task_ids=task_ids or [0],
        observation_width=256,
        observation_height=256,
        control_mode="relative",
    )


def _load_policy_and_processors(args: argparse.Namespace, spec: PolicySpec) -> dict[str, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.scripts.lerobot_eval import make_env_pre_post_processors
    from peft import PeftConfig, PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("WSL_CUDA_BLOCKED: torch.cuda.is_available() is false")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_path = Path(args.base_path)
    cfg = PreTrainedConfig.from_pretrained(str(base_path))
    cfg.pretrained_path = str(base_path)
    cfg.device = "cuda"
    cfg.use_amp = False
    cfg.use_peft = False
    cfg.empty_cameras = 1
    cfg.compile_model = False

    env_cfg = _make_env_cfg("libero_spatial", [0])
    load_started = time.monotonic()
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    peft_audit: dict[str, Any] = {"used": False}
    adapter_path = _policy_adapter_path(args, spec)
    if adapter_path is not None:
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        original_base = peft_config.base_model_name_or_path
        peft_config.base_model_name_or_path = str(base_path)
        policy = PeftModel.from_pretrained(policy, str(adapter_path), config=peft_config)
        policy.to("cuda")
        peft_audit = {
            "used": True,
            "adapter_path": str(adapter_path),
            "original_base_model_name_or_path": original_base,
            "resolved_base_model_name_or_path": str(base_path),
        }
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(base_path),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    dummy = env_preprocessor(_dummy_observation(torch))
    batch = preprocessor(dummy)
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)

    param = _first_parameter_summary(policy)
    devices = _tensor_devices(batch)
    if param["device"] != "cuda:0" or any(device != "cuda:0" for device in devices.values()):
        raise RuntimeError("CPU_FALLBACK_BUG: CUDA available but model or preprocessed inputs are on CPU")

    audit = {
        "policy_name": spec.name,
        "policy_class": type(policy).__name__,
        "load_seconds": _round(time.monotonic() - load_started, 3),
        "parameter": param,
        "input_tensor_devices": devices,
        "input_tensor_shapes": _tensor_shapes(batch),
        "action_chunk_shape": [int(dim) for dim in action_chunk.shape],
        "action_chunk_device": str(action_chunk.device),
        "action_chunk_dtype": str(action_chunk.dtype),
        "action_chunk_finite": bool(torch.isfinite(action_chunk).all().item()),
        "cuda_memory": _cuda_memory(torch),
        "autocast": _torch_autocast(torch),
        "amp_fp16_or_bf16_active": bool(_torch_autocast(torch)["cuda"]),
        "peft": peft_audit,
        "rename_map": dict(RENAME_MAP),
        "empty_cameras": int(cfg.empty_cameras),
        "control_mode": "relative",
        "old_custom_libero_7d_route_used": False,
    }
    return {
        "cfg": cfg,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
        "audit": audit,
    }


def _run_policy_rollout(
    args: argparse.Namespace,
    spec: PolicySpec,
    suites: list[str],
    episodes_per_task: int,
    start_seed: int,
) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env
    from lerobot.scripts.lerobot_eval import eval_policy_all

    loaded = _load_policy_and_processors(args, spec)
    env_cfg = _make_env_cfg(",".join(suites), [0])
    env_started = time.monotonic()
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env_seconds = time.monotonic() - env_started
    run_started = time.monotonic()
    metrics = eval_policy_all(
        envs=envs,
        policy=loaded["policy"],
        env_preprocessor=loaded["env_preprocessor"],
        env_postprocessor=loaded["env_postprocessor"],
        preprocessor=loaded["preprocessor"],
        postprocessor=loaded["postprocessor"],
        n_episodes=int(episodes_per_task),
        start_seed=int(start_seed),
        max_parallel_tasks=1,
        max_episodes_rendered=0,
    )
    run_seconds = time.monotonic() - run_started
    completed = _completed_episodes(metrics)
    result = {
        "policy": spec.name,
        "suites": suites,
        "task_ids": [0],
        "episodes_per_task": int(episodes_per_task),
        "start_seed": int(start_seed),
        "env_creation_seconds": _round(env_seconds, 3),
        "rollout_seconds": _round(run_seconds, 3),
        "completed_episodes": completed,
        "metrics": _json_sanitize(metrics),
        "policy_load_audit": loaded["audit"],
        "cuda_memory_after_rollout": _cuda_memory(torch),
    }
    del loaded
    del envs
    torch.cuda.empty_cache()
    return result


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _completed_episodes(metrics: dict[str, Any]) -> int:
    if "n_episodes" in metrics and metrics["n_episodes"] is not None:
        return int(metrics["n_episodes"])
    overall = metrics.get("overall") or {}
    if overall.get("n_episodes") is not None:
        return int(overall["n_episodes"])
    total = 0
    for item in metrics.get("per_task_infos") or []:
        task_metrics = item.get("metrics") or {}
        if task_metrics.get("n_episodes") is not None:
            total += int(task_metrics["n_episodes"])
        elif task_metrics.get("successes") is not None:
            successes = task_metrics["successes"]
            total += len(successes) if isinstance(successes, list) else 1
    groups = metrics.get("groups") or metrics.get("groups_aggregated") or {}
    if total == 0 and isinstance(groups, dict):
        total = sum(int((group or {}).get("n_episodes") or 0) for group in groups.values())
    return int(total)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    suites = ["libero_spatial"]
    results = []
    errors = []
    started = time.monotonic()
    for spec in POLICIES:
        try:
            print(f"[official-rollout] smoke {spec.name}", flush=True)
            results.append(_run_policy_rollout(args, spec, suites, 1, int(args.smoke_seed)))
        except Exception as exc:  # pragma: no cover - runtime boundary
            errors.append(
                {
                    "policy": spec.name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
            )
            break
    return {
        "executed": bool(results),
        "all_policies_executed": len(results) == len(POLICIES) and not errors,
        "planned_episodes": len(POLICIES),
        "completed_episodes": sum(int(item.get("completed_episodes") or 0) for item in results),
        "suites": suites,
        "task_ids": [0],
        "episodes_per_policy": 1,
        "same_reset_seed_for_all_policies": int(args.smoke_seed),
        "results": results,
        "errors": errors,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    results = []
    errors = []
    started = time.monotonic()
    for spec in POLICIES:
        try:
            print(f"[official-rollout] pilot {spec.name}", flush=True)
            results.append(_run_policy_rollout(args, spec, suites, int(args.pilot_episodes), int(args.pilot_seed)))
        except Exception as exc:  # pragma: no cover - runtime boundary
            errors.append(
                {
                    "policy": spec.name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
            )
            break
    return {
        "executed": bool(results),
        "all_policies_executed": len(results) == len(POLICIES) and not errors,
        "planned_episodes": len(POLICIES) * len(suites) * int(args.pilot_episodes),
        "completed_episodes": sum(int(item.get("completed_episodes") or 0) for item in results),
        "suites": suites,
        "task_ids": [0],
        "episodes_per_task_per_policy": int(args.pilot_episodes),
        "same_reset_seed_for_all_policies": int(args.pilot_seed),
        "results": results,
        "errors": errors,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_env(args)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "date": "2026-07-10",
        "base_path": str(Path(args.base_path)),
        "lora_root": str(Path(args.lora_root)),
        "libero_config_dir": str(Path(args.libero_config_dir)),
        "policies": [spec.__dict__ for spec in POLICIES],
        "rename_map": dict(RENAME_MAP),
        "static_mix_duplicate_records": static_mix_duplicate_records(),
        "static_mix_duplicate_runs_skipped": True,
        "old_custom_libero_7d_route_used": False,
        "mode": args.mode,
        "smoke": {},
        "pilot": {"executed": False},
        "errors": [],
    }
    try:
        if args.mode in {"smoke", "all"}:
            report["smoke"] = run_smoke(args)
        if args.mode in {"pilot", "all"}:
            if args.mode == "all" and not (report.get("smoke") or {}).get("all_policies_executed"):
                report["pilot"] = {"executed": False, "skip_reason": "smoke did not execute all policies"}
            else:
                report["pilot"] = run_pilot(args)
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})

    text = json.dumps(report, default=str)
    report["cpu_fallback_bug"] = "CPU_FALLBACK_BUG" in text
    report["wsl_cuda_blocked"] = "WSL_CUDA_BLOCKED" in text or "torch.cuda.is_available() is false" in text
    report["peft_loading_blocked"] = "Peft" in text and bool(report.get("errors"))
    report["schema_or_action_mismatch"] = "Feature mismatch" in text or "ACTION" in text and "mismatch" in text
    report["official_env_blocked"] = "LIBERO" in text and bool(report.get("errors"))
    report["runtime_too_heavy"] = bool(report.get("runtime_too_heavy"))
    report["runtime"] = {"elapsed_seconds": _round(time.monotonic() - started, 3)}
    report["final_decision"] = choose_final_decision(report)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, report: dict[str, Any], section: str | None = None) -> None:
    data = report if section is None else report.get(section, {})
    lines = [
        f"# {title}",
        "",
        f"- final decision: `{report.get('final_decision')}`",
        f"- mode: `{report.get('mode')}`",
        f"- static mix duplicate runs skipped: `{report.get('static_mix_duplicate_runs_skipped')}`",
        f"- old custom LIBERO_7D route used: `{report.get('old_custom_libero_7d_route_used')}`",
        "",
        "```json",
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_rollout_section(section: dict[str, Any] | None) -> bool:
    return bool(section and (section.get("results") or section.get("errors")))


def write_reports(report: dict[str, Any], report_dir: Path) -> None:
    if _has_rollout_section(report.get("smoke")):
        _write_json(report_dir / "official_libero_rollout_smoke_result.json", report.get("smoke") or {})
        _write_md(report_dir / "official_libero_rollout_smoke_result.md", "Official LIBERO Rollout Smoke Result", report, "smoke")
    if _has_rollout_section(report.get("pilot")):
        _write_json(report_dir / "official_libero_rollout_pilot_result.json", report.get("pilot") or {})
        _write_md(report_dir / "official_libero_rollout_pilot_result.md", "Official LIBERO Rollout Pilot Result", report, "pilot")
    _write_md(report_dir / "official_libero_closed_loop_failure_summary.md", "Official LIBERO Closed-Loop Failure Summary", report)
    _write_md(report_dir / "official_libero_rollout_decision.md", "Official LIBERO Rollout Decision", report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "all"], default="all")
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--smoke-seed", type=int, default=20260710)
    parser.add_argument("--pilot-seed", type=int, default=20260710)
    parser.add_argument("--pilot-episodes", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    report_dir = Path(args.report_dir)
    _write_json(report_dir / "official_libero_rollout_full_result.json", report)
    write_reports(report, report_dir)
    print(json.dumps({"final_decision": report.get("final_decision"), "runtime": report.get("runtime")}, indent=2))
    return 0 if report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
