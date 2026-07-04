"""Bounded offline SmolVLA action decoding on one LIBERO demonstration sample.

This module intentionally avoids simulator creation and rollouts. It reads one
local HDF5 observation/action pair, loads local SmolVLA on CPU, runs one
``select_action`` call, and compares the decoded action to the expert action as
diagnostic evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.interface_adapters import (
    ACTION_STRATEGY_GRIPPER_CLOSE,
    ACTION_STRATEGY_GRIPPER_OPEN,
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    adapt_policy_action_to_env_action,
)
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
from tca_map.smolvla.single_sample_interface_smoke import _load_policy


OFFLINE_DECODE_GATE = "ALLOW_OFFLINE_DEMO_ACTION_DECODING"
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
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
]
MAX_RUNTIME_SECONDS = 600
MAX_VRAM_MB = 14336


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


def _load_first_hdf5_sample(path: Path, demo_name: str | None = None, timestep: int = 0) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        selected_demo = demo_name or sorted(handle["data"].keys())[0]
        demo = handle["data"][selected_demo]
        actions = np.asarray(demo["actions"])
        if timestep < 0 or timestep >= actions.shape[0]:
            raise ValueError(f"timestep {timestep} outside action range 0..{actions.shape[0] - 1}")
        obs_group = demo["obs"]
        ee_states = np.asarray(obs_group["ee_states"][timestep], dtype=np.float32)
        if ee_states.shape[0] < 6:
            raise ValueError(f"expected obs/ee_states to contain at least 6 values, got {ee_states.shape}")
        agentview = np.asarray(obs_group["agentview_rgb"][timestep])
        eye_in_hand = np.asarray(obs_group["eye_in_hand_rgb"][timestep])
        expert_action = np.asarray(actions[timestep], dtype=np.float32).reshape(-1)
        return {
            "demo_name": selected_demo,
            "timestep": int(timestep),
            "expert_action": expert_action,
            "obs": {
                "robot0_eef_pos": ee_states[:3],
                "robot0_eef_quat": ee_states[3:6],
                "agentview_image": agentview,
                "agentview_rgb": agentview,
                "robot0_eye_in_hand_image": eye_in_hand,
                "eye_in_hand_rgb": eye_in_hand,
            },
            "metadata": {
                "hdf5_path": str(path),
                "demo_name": selected_demo,
                "timestep": int(timestep),
                "expert_action_shape": list(expert_action.shape),
                "ee_states_shape": list(ee_states.shape),
                "agentview_shape": list(agentview.shape),
                "eye_in_hand_shape": list(eye_in_hand.shape),
                "init_state_attr_present": "init_state" in demo.attrs,
                "states_present": "states" in demo,
            },
        }


def _finite_list(values: np.ndarray, limit: int | None = None) -> list[float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [round(float(x), 6) for x in flat]


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    plan_path = Path(args.plan_report)
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)

    report: dict[str, Any] = {
        "evidence_label": "offline_demo_action_decoding_diagnostic",
        "offline_demo_action_decoding_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_paper_claim": False,
        "policy": {
            "bounded_offline_demo_action_decoding": True,
            "task_local_gate_required": f"{OFFLINE_DECODE_GATE}=1",
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "single_action_inference_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "task_local_gate_set": _env_flag(OFFLINE_DECODE_GATE),
            "forbidden_gates_set": [name for name in FORBIDDEN_GATES if _env_flag(name)],
        },
        "claims": {
            "standard_success_claimed": False,
            "benchmark_success_claimed": False,
            "counterfactual_robustness_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "risk_limits": {
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_vram_mb": MAX_VRAM_MB,
            "device": args.device,
            "max_hdf5_timesteps_to_read": 1,
            "max_policy_inference_calls": 1,
            "simulator_allowed": False,
            "rollout_allowed": False,
            "training_allowed": False,
        },
        "paths": {
            "plan_report": str(plan_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
        },
        "files": {},
        "runtime_dependencies": {},
        "gpu": _nvidia_smi(),
        "runtime": {
            "rss_before_mb": _rss_mb(),
            "rss_after_mb": None,
            "elapsed_sec": None,
            "single_action_inference_elapsed_sec": None,
        },
        "sample": {},
        "metrics": {},
        "error": None,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int = 2) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        return report, code

    if not report["policy"]["task_local_gate_set"]:
        return block(f"{OFFLINE_DECODE_GATE}=1 is required only inside this bounded offline decoding task.", 2)
    if report["policy"]["forbidden_gates_set"]:
        return block("Forbidden gate(s) set: " + ", ".join(report["policy"]["forbidden_gates_set"]), 3)
    if args.device != "cpu":
        return block("The first offline demonstration-conditioned action decoding runner is CPU-only.", 4)
    if not plan_path.exists():
        return block(f"Offline decoding plan report is missing: {plan_path}", 5)

    try:
        plan = _read_json(plan_path)
        report["plan"] = plan
        if not plan.get("ready_for_bounded_offline_demo_action_decoding_runner"):
            return block("Offline decoding plan did not authorize a bounded runner.", 6)

        hdf5_path = Path((plan.get("inputs") or {}).get("hdf5_path") or args.hdf5_path)
        if not hdf5_path.exists():
            return block(f"Selected HDF5 file is missing: {hdf5_path}", 7)

        config_files = _find_files(smolvla_ckpt, ["config.json"])
        weight_files = _find_files(
            smolvla_ckpt,
            ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
            ["*.safetensors", "*.bin"],
        )
        dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
        external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
        deps = _runtime_dependencies()
        try:
            import h5py  # noqa: F401

            h5py_ready = True
        except Exception:
            h5py_ready = False
        deps["h5py"] = h5py_ready
        report["files"] = {
            "config_found": config_files,
            "weights_found": weight_files,
            "external_tokenizer_dependency": external_dependency,
            "files_ready": bool(config_files and weight_files and external_dependency["found"]),
        }
        report["runtime_dependencies"] = deps
        if not report["files"]["files_ready"]:
            return block("SmolVLA local files or tokenizer dependency are incomplete.", 8)
        if not all(deps.values()):
            missing = [name for name, present in deps.items() if not present]
            return block("Missing runtime dependencies: " + ", ".join(missing), 9)

        sample = _load_first_hdf5_sample(hdf5_path, args.demo_name, args.timestep)
        expert_action = np.asarray(sample["expert_action"], dtype=np.float32).reshape(-1)
        report["sample"] = sample["metadata"]
        report["sample"]["task"] = (plan.get("planned_sample") or {}).get("selected_language") or args.task

        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = _load_policy(smolvla_ckpt, hf_home, external_dependency, args.device)
        report["policy"]["model_load_performed"] = True
        tokenizer_root = Path(external_dependency["root"])
        task_text = report["sample"]["task"]
        batch, batch_metadata = _build_batch(
            config,
            tokenizer_root,
            sample["obs"],
            task_text,
            args.device,
            CAMERA_ALIAS_STRATEGY_CURRENT,
            STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
        )

        import torch

        noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=args.device)
        inference_started = time.monotonic()
        with torch.inference_mode():
            policy_action = policy.select_action(batch, noise=noise)
        inference_elapsed = time.monotonic() - inference_started
        report["policy"]["single_action_inference_performed"] = True
        report["policy"]["model_inference_performed"] = True

        action_adapter = adapt_policy_action_to_env_action(
            policy_action,
            int(expert_action.shape[0]),
            strategy=args.action_adapter_strategy,
            action_scale=args.action_scale,
        )
        adapted = np.asarray(action_adapter.values, dtype=np.float32)
        policy_action_np = policy_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        l1 = float(np.mean(np.abs(adapted - expert_action)))
        mse = float(np.mean((adapted - expert_action) ** 2))
        policy6_l1 = float(np.mean(np.abs(policy_action_np[: min(6, expert_action.shape[0])] - expert_action[: min(6, expert_action.shape[0])])))
        finite = bool(np.isfinite(adapted).all() and np.isfinite(policy_action_np).all())

        cuda_max_allocated_mb = None
        if torch.cuda.is_available():
            cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report["runtime"]["single_action_inference_elapsed_sec"] = round(inference_elapsed, 6)
        report["metrics"] = {
            "action_l1_to_expert": round(l1, 6),
            "action_mse_to_expert": round(mse, 6),
            "policy6_l1_to_expert_first6": round(policy6_l1, 6),
            "action_finite": finite,
            "policy_action_shape": list(policy_action_np.shape),
            "expert_action_shape": list(expert_action.shape),
            "policy_action_preview": _finite_list(policy_action_np, 6),
            "adapted_action_preview": _finite_list(adapted, int(expert_action.shape[0])),
            "expert_action_preview": _finite_list(expert_action, int(expert_action.shape[0])),
            "action_adapter_metadata": action_adapter.metadata,
            "batch_metadata": batch_metadata,
            "load_vlm_weights": bool(getattr(config, "load_vlm_weights", False)),
            "device": args.device,
            "cuda_max_allocated_mb": cuda_max_allocated_mb,
            "evidence_level": "one_sample_offline_action_decoding_diagnostic",
        }

        elapsed = time.monotonic() - started
        if not finite:
            return block("Decoded action contained non-finite values.", 10)
        if elapsed > MAX_RUNTIME_SECONDS:
            return block("Offline decoding exceeded the 10 minute runtime budget.", 11)
        if (cuda_max_allocated_mb or 0) > MAX_VRAM_MB:
            return block("Offline decoding exceeded the 14GB VRAM budget.", 12)

        report["offline_demo_action_decoding_passed"] = True
        report["decision"] = "diagnostic_complete"
        report["recommended_next_step"] = (
            "Summarize this one-sample offline decoding diagnostic before any rollout. "
            "If action error is large, inspect VLM loading policy/checkpoint provenance before more rollouts."
        )
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(elapsed, 3)
        return report, 0
    except Exception as exc:  # noqa: BLE001 - report exact local diagnostic failure.
        report["error"] = _compact_error(exc)
        report["recommended_next_step"] = "Fix the offline action-decoding blocker before any further learned-policy rollout."
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        return report, 13


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/offline_demo_conditioned_action_decoding_plan_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--hdf5-path", default="")
    parser.add_argument("--demo-name", default=None)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--task", default="perform the task")
    parser.add_argument(
        "--action-adapter-strategy",
        default=ACTION_STRATEGY_GRIPPER_CLOSE,
        choices=[
            ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
            ACTION_STRATEGY_GRIPPER_OPEN,
            ACTION_STRATEGY_GRIPPER_CLOSE,
        ],
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--report-path", default="reports/offline_demo_action_decoding_report.json")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
