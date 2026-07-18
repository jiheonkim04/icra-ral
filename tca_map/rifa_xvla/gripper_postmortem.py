"""No-training action-semantics postmortem for the frozen RIFA v1 failure.

This module replays only the one already-recorded Stage 0 initial-observation
row with the frozen Base/full/ablation checkpoints.  It never performs
backward, optimizer, checkpoint, or closed-loop simulator action steps.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from .stage0 import (
    ActionHiddenHook,
    RIFAAdapter,
    action_delta,
    cuda_report,
    extract_rl4il_context,
    freeze_module,
    install_optional_xvla_shims,
    install_xvla_transformers_compat_patches,
    load_frozen_contract,
    matching_rl4il_task,
    matrix_to_rotate6d,
    memory_report,
    normalized_context,
    nvidia_smi,
    plan_to_libero_actions,
    prepare_live_inputs,
    sha256_file,
    tensor_context,
    utcish_timestamp,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "configs" / "rifa_xvla_stage0_frozen_contract.json"
DEFAULT_STAGE0_RESULT = REPO_ROOT / "reports" / "rifa_xvla_stage0_result.json"
DEFAULT_FULL_CHECKPOINT = REPO_ROOT / "reports" / "checkpoints" / "rifa_xvla_stage0" / "rifa_xvla_full.pt"
DEFAULT_ABLATION_CHECKPOINT = (
    REPO_ROOT / "reports" / "checkpoints" / "rifa_xvla_stage0" / "rifa_xvla_no_reliability.pt"
)
DEFAULT_REPORT_JSON = REPO_ROOT / "reports" / "rifa_xvla_gripper_postmortem_result.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports" / "rifa_xvla_gripper_postmortem_result.md"

FROZEN_PROTOCOL_DECISION = "RIFA_XVLA_STAGE0_DESIGN_FAILURE"
CALIBRATED_SCIENTIFIC_INTERPRETATION = (
    "RIFA v1 is not Stage-A-ready because one binary gripper flip violated the frozen action-delta gate "
    "and the full-versus-no-reliability action difference was practically negligible despite technically "
    "exceeding the preregistered minimum."
)
TARGET = {
    "suite": "libero_object",
    "task_id": 0,
    "reset_identity": 20260734,
    "condition": "mask_1_in_hand_dropout",
}
GRIPPER_PLAN_INDEX = 9
GRIPPER_THRESHOLD = 0.5


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _as_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def exact_live_row_index(contract: dict[str, Any], target: dict[str, Any] = TARGET) -> int:
    row_index = 0
    for task in contract["panel"]:
        for identity in task["identities"]:
            for condition in contract["conditions"]:
                if (
                    task["suite"] == target["suite"]
                    and int(task["task_id"]) == int(target["task_id"])
                    and int(identity) == int(target["reset_identity"])
                    and condition == target["condition"]
                ):
                    return row_index
                row_index += 1
    raise ValueError(f"target row not present in frozen panel: {target}")


def _policy_gripper_record(plan: np.ndarray, action_index: int) -> dict[str, Any]:
    score = float(np.asarray(plan, dtype=np.float32)[action_index, GRIPPER_PLAN_INDEX])
    discrete = 1.0 if score > GRIPPER_THRESHOLD else -1.0
    return {
        "raw_pre_discretization_score": score,
        "decision_threshold": GRIPPER_THRESHOLD,
        "signed_margin_from_threshold": float(score - GRIPPER_THRESHOLD),
        "absolute_distance_from_threshold": float(abs(score - GRIPPER_THRESHOLD)),
        "threshold_rule": "score > 0.5 => +1.0; otherwise => -1.0",
        "final_discretized_gripper_action": discrete,
    }


def _continuous_delta(left: np.ndarray, right: np.ndarray, action_index: int) -> dict[str, Any]:
    left_actions = plan_to_libero_actions(left)
    right_actions = plan_to_libero_actions(right)
    delta = left_actions - right_actions
    translation = delta[:, :3].astype(np.float64)
    rotation = delta[:, 3:6].astype(np.float64)
    row_translation = translation[action_index]
    row_rotation = rotation[action_index]
    return {
        "at_flip": {
            "translation_delta_xyz": _as_list(row_translation),
            "translation_l2": float(np.linalg.norm(row_translation)),
            "rotation_delta_rotvec": _as_list(row_rotation),
            "rotation_l2": float(np.linalg.norm(row_rotation)),
        },
        "whole_chunk": {
            "translation_rms": float(np.sqrt(np.mean(translation**2))),
            "translation_max_abs": float(np.max(np.abs(translation))),
            "rotation_rms": float(np.sqrt(np.mean(rotation**2))),
            "rotation_max_abs": float(np.max(np.abs(rotation))),
        },
    }


def analyze_action_semantics(base: np.ndarray, full: np.ndarray, ablation: np.ndarray) -> dict[str, Any]:
    plans = {
        "BASE": np.asarray(base, dtype=np.float32),
        "RIFA_XVLA": np.asarray(full, dtype=np.float32),
        "RIFA_XVLA_NO_RELIABILITY": np.asarray(ablation, dtype=np.float32),
    }
    if any(plan.ndim != 2 or plan.shape[1] <= GRIPPER_PLAN_INDEX for plan in plans.values()):
        return {
            "decision": "RIFA_GRIPPER_INTERNAL_SIGNAL_UNAVAILABLE",
            "reason": "one or more X-VLA plans did not expose the raw gripper score at plan column 9",
        }
    shapes = {tuple(plan.shape) for plan in plans.values()}
    if len(shapes) != 1:
        return {
            "decision": "RIFA_GRIPPER_INTERNAL_SIGNAL_UNAVAILABLE",
            "reason": f"Base/full/ablation plan shapes differ: {sorted(shapes)}",
        }

    actions = {name: plan_to_libero_actions(plan) for name, plan in plans.items()}
    base_full_flips = np.flatnonzero(actions["BASE"][:, 6] != actions["RIFA_XVLA"][:, 6]).tolist()
    base_ablation_flips = np.flatnonzero(
        actions["BASE"][:, 6] != actions["RIFA_XVLA_NO_RELIABILITY"][:, 6]
    ).tolist()
    full_ablation_flips = np.flatnonzero(
        actions["RIFA_XVLA"][:, 6] != actions["RIFA_XVLA_NO_RELIABILITY"][:, 6]
    ).tolist()
    if not base_full_flips:
        return {
            "decision": "RIFA_TRUE_DESTRUCTIVE_GRIPPER_SHIFT_CONFIRMED",
            "reason": "the replay did not reproduce the recorded Base-versus-full gripper flip",
            "flip_indices": {
                "base_vs_full": base_full_flips,
                "base_vs_ablation": base_ablation_flips,
                "full_vs_ablation": full_ablation_flips,
            },
        }

    flip_index = int(base_full_flips[0])
    policy_records = {name: _policy_gripper_record(plan, flip_index) for name, plan in plans.items()}
    base_record = policy_records["BASE"]
    full_record = policy_records["RIFA_XVLA"]
    raw_straddles_threshold = bool(
        float(base_record["signed_margin_from_threshold"])
        * float(full_record["signed_margin_from_threshold"])
        <= 0.0
        and float(base_record["final_discretized_gripper_action"])
        != float(full_record["final_discretized_gripper_action"])
    )
    discrete_delta = float(
        full_record["final_discretized_gripper_action"] - base_record["final_discretized_gripper_action"]
    )
    discontinuity = bool(raw_straddles_threshold and abs(discrete_delta) == 2.0)
    decision = (
        "RIFA_GRIPPER_POSTPROCESS_DISCONTINUITY_CONFIRMED"
        if discontinuity
        else "RIFA_TRUE_DESTRUCTIVE_GRIPPER_SHIFT_CONFIRMED"
    )
    return {
        "decision": decision,
        "plan_shape": list(plans["BASE"].shape),
        "chunk_index": 0,
        "action_index_within_chunk": flip_index,
        "flip_indices": {
            "base_vs_full": [int(value) for value in base_full_flips],
            "base_vs_ablation": [int(value) for value in base_ablation_flips],
            "full_vs_ablation": [int(value) for value in full_ablation_flips],
        },
        "policies_at_flip": policy_records,
        "raw_gripper_scores_whole_chunk": {
            name: _as_list(plan[:, GRIPPER_PLAN_INDEX]) for name, plan in plans.items()
        },
        "final_discrete_gripper_whole_chunk": {
            name: _as_list(action[:, 6]) for name, action in actions.items()
        },
        "continuous_deltas": {
            "full_minus_base": _continuous_delta(full, base, flip_index),
            "ablation_minus_base": _continuous_delta(ablation, base, flip_index),
            "full_minus_ablation": _continuous_delta(full, ablation, flip_index),
        },
        "action_delta_summaries": {
            "full_vs_base": action_delta(full, base),
            "ablation_vs_base": action_delta(ablation, base),
            "full_vs_ablation": action_delta(full, ablation),
        },
        "raw_base_full_scores_straddle_threshold": raw_straddles_threshold,
        "discretized_full_minus_base_at_flip": discrete_delta,
        "max_abs_2_caused_by_sign_threshold_discontinuity": discontinuity,
        "full_and_ablation_same_gripper_at_flip": bool(
            policy_records["RIFA_XVLA"]["final_discretized_gripper_action"]
            == policy_records["RIFA_XVLA_NO_RELIABILITY"]["final_discretized_gripper_action"]
        ),
        "full_and_ablation_same_gripper_whole_chunk": bool(not full_ablation_flips),
    }


def _close_enough(left: Any, right: Any, *, atol: float = 1e-8) -> bool:
    return bool(np.allclose(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64), atol=atol, rtol=0.0))


def _write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    analysis = result.get("analysis") or {}
    policies = analysis.get("policies_at_flip") or {}
    lines = [
        "# RIFA-XVLA Gripper Action-Semantics Postmortem",
        "",
        f"- Diagnostic decision: `{result.get('decision')}`",
        f"- Execution valid: `{result.get('execution_valid')}`",
        f"- FROZEN_PROTOCOL_DECISION: `{FROZEN_PROTOCOL_DECISION}`",
        f"- CALIBRATED_SCIENTIFIC_INTERPRETATION: {CALIBRATED_SCIENTIFIC_INTERPRETATION}",
        f"- Chunk / action index: `{analysis.get('chunk_index')} / {analysis.get('action_index_within_chunk')}`",
        "",
        "## Gripper signal at the flip",
        "",
        "| policy | raw score | threshold margin | discrete action |",
        "|---|---:|---:|---:|",
    ]
    for name in ("BASE", "RIFA_XVLA", "RIFA_XVLA_NO_RELIABILITY"):
        row = policies.get(name) or {}
        lines.append(
            f"| `{name}` | `{row.get('raw_pre_discretization_score')}` | "
            f"`{row.get('signed_margin_from_threshold')}` | `{row.get('final_discretized_gripper_action')}` |"
        )
    lines += [
        "",
        f"The `2.0` delta is a sign/threshold discontinuity: "
        f"`{analysis.get('max_abs_2_caused_by_sign_threshold_discontinuity')}`.",
        f"Full and ablation make the same gripper decision: "
        f"`{analysis.get('full_and_ablation_same_gripper_whole_chunk')}`.",
        "",
        "No training, backward pass, optimizer step, checkpoint write, Stage 0 rerun, or closed-loop rollout occurred.",
        "RIFA v1 remains closed and is not Stage-A-ready; this does not rule out the broader reliability-conditioned family.",
    ]
    if result.get("exceptions"):
        lines += ["", "## Exceptions", "", "```json", json.dumps(result["exceptions"], indent=2), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_postmortem(
    report_json: pathlib.Path = DEFAULT_REPORT_JSON,
    report_md: pathlib.Path = DEFAULT_REPORT_MD,
) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    contract = load_frozen_contract(DEFAULT_CONTRACT)
    stage0 = json.loads(DEFAULT_STAGE0_RESULT.read_text(encoding="utf-8"))
    frozen_inputs = [DEFAULT_STAGE0_RESULT, DEFAULT_FULL_CHECKPOINT, DEFAULT_ABLATION_CHECKPOINT, DEFAULT_CONTRACT]
    hashes_before = {str(path): sha256_file(path) for path in frozen_inputs}
    row = next(
        candidate
        for candidate in stage0["validation"]["rows"]
        if candidate.get("suite") == TARGET["suite"]
        and int(candidate.get("task_id")) == TARGET["task_id"]
        and int(candidate.get("reset_identity", -1)) == TARGET["reset_identity"]
        and candidate.get("condition") == TARGET["condition"]
    )
    row_index = exact_live_row_index(contract)
    eval_seed = int(contract["training_budget"]["seed"]) + 3000 + row_index
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_rifa_gripper_postmortem.v1",
        "stage": "one_no_training_action_semantics_postmortem",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": utcish_timestamp(),
        "pid": int(os.getpid()),
        "cuda_pid": int(os.getpid()) if torch.cuda.is_available() else None,
        "target": TARGET,
        "exact_live_row_index": row_index,
        "exact_generation_seed": eval_seed,
        "frozen_protocol_decision": FROZEN_PROTOCOL_DECISION,
        "calibrated_scientific_interpretation": CALIBRATED_SCIENTIFIC_INTERPRETATION,
        "stage0_source_row": row,
        "frozen_input_hashes_before": hashes_before,
        "execution_valid": False,
        "decision": "RIFA_GRIPPER_INTERNAL_SIGNAL_UNAVAILABLE",
        "exceptions": [],
        "training_occurred": False,
        "backward_pass_count": 0,
        "optimizer_step_count": 0,
        "checkpoint_write_count": 0,
        "stage0_rerun": False,
        "closed_loop_rollout": False,
        "simulator_episode_count": 0,
        "official_success_measurement": False,
        "original_outputs_modified": False,
        "downloads_used": False,
        "nvidia_smi_before": nvidia_smi(),
    }
    clip: Any | None = None
    model: torch.nn.Module | None = None
    hook_handle: Any = None
    env: Any | None = None
    try:
        if stage0.get("decision") != FROZEN_PROTOCOL_DECISION or not bool(stage0.get("execution_valid")):
            raise ValueError("authoritative frozen RIFA Stage 0 result drift")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for exact X-VLA forward replay")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")

        from tca_map.rl4il_prior.mechanism_port import FrozenCLIPEncoder, load_task_port

        task = next(
            candidate
            for candidate in contract["panel"]
            if candidate["suite"] == TARGET["suite"] and int(candidate["task_id"]) == TARGET["task_id"]
        )
        clip = FrozenCLIPEncoder(device)
        freeze_module(clip)
        port = load_task_port(
            pathlib.Path(contract["source_prior_training_dir"]), matching_rl4il_task(task), device
        )
        for key in ("clean_policy", "clean_fusion", "mask1_policy", "mask1_fusion", "imp_policy", "soft_imp"):
            freeze_module(port[key])

        xvla_root = pathlib.Path(contract["xvla"]["source_root"])
        if str(xvla_root) in sys.path:
            sys.path.remove(str(xvla_root))
        sys.path.insert(0, str(xvla_root))
        shims = install_optional_xvla_shims()
        patches = install_xvla_transformers_compat_patches()
        from models.modeling_xvla import XVLA  # type: ignore
        from models.processing_xvla import XVLAProcessor  # type: ignore

        xvla = contract["xvla"]
        model = XVLA.from_pretrained(
            xvla["model_id"],
            revision=xvla["model_revision"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
            cache_dir=xvla["cache_dir"],
        )
        processor = XVLAProcessor.from_pretrained(
            xvla["model_id"],
            revision=xvla["model_revision"],
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=xvla["cache_dir"],
        )
        freeze_module(model)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        mechanism = contract["mechanism"]
        adapter_kwargs = {
            "hidden_size": int(model.transformer.hidden_size),
            "imputed_dim": int(mechanism["imputed_feature_dim"]),
            "bottleneck_dim": int(mechanism["adapter_bottleneck_dim"]),
            "residual_scale": float(mechanism["maximum_hidden_residual_scale"]),
        }
        full_adapter = RIFAAdapter(**adapter_kwargs, no_reliability=False).to(device)
        ablation_adapter = RIFAAdapter(**adapter_kwargs, no_reliability=True).to(device)
        full_adapter.load_state_dict(torch.load(DEFAULT_FULL_CHECKPOINT, map_location=device, weights_only=True))
        ablation_adapter.load_state_dict(
            torch.load(DEFAULT_ABLATION_CHECKPOINT, map_location=device, weights_only=True)
        )
        full_adapter.eval()
        ablation_adapter.eval()
        hook = ActionHiddenHook()
        hook_handle = model.transformer.norm.register_forward_hook(hook)

        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[TARGET["suite"]]()
        libero_task = suite.get_task(TARGET["task_id"])
        bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
        initial_states = suite.get_task_init_states(TARGET["task_id"])
        initial_state_index = TARGET["reset_identity"] - 20260711
        if int(row["initial_state_index"]) != initial_state_index:
            raise ValueError("frozen row initial-state index drift")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(TARGET["reset_identity"])
        env.reset()
        obs = env.set_init_state(np.asarray(initial_states[initial_state_index], dtype=np.float64))
        for _ in range(10):
            obs, _reward, _done, _info = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        obs["robo_ori"] = matrix_to_rotate6d(env.env.robots[0].controller.ee_ori_mat)
        obs["robo_pos"] = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float32)

        context_raw = extract_rl4il_context(clip, port, obs, str(task["instruction"]), TARGET["condition"])
        normalizer = {
            "mean": np.asarray(stage0["reliability"]["training_raw_mean"], dtype=np.float32),
            "std": np.asarray(stage0["reliability"]["training_raw_std"], dtype=np.float32),
        }
        context_np = normalized_context(context_raw, normalizer)
        raw_match = _close_enough(context_raw["reliability_raw"], row["reliability_raw"], atol=1e-8)
        normalized_match = _close_enough(context_np["reliability"], row["reliability_normalized"], atol=1e-7)
        if not raw_match or not normalized_match:
            raise RuntimeError("recreated RL4IL context does not match the exact saved Stage 0 row")
        context = tensor_context(context_np, device)
        inputs = prepare_live_inputs(
            obs, str(task["instruction"]), processor, device, condition=TARGET["condition"]
        )

        from .stage0 import generate_plan

        denoise_steps = int(xvla["denoise_steps"])
        base = generate_plan(
            model, hook, inputs, adapter=None, context=None, denoise_steps=denoise_steps, seed=eval_seed
        )
        full = generate_plan(
            model, hook, inputs, adapter=full_adapter, context=context, denoise_steps=denoise_steps, seed=eval_seed
        )
        ablation = generate_plan(
            model,
            hook,
            inputs,
            adapter=ablation_adapter,
            context=context,
            denoise_steps=denoise_steps,
            seed=eval_seed,
        )
        analysis = analyze_action_semantics(base, full, ablation)
        full_metrics_match = all(
            _close_enough(analysis["action_delta_summaries"]["full_vs_base"][key], row["full_vs_base"][key])
            for key in ("rms", "max_abs", "translation_rms", "rotation_rms", "gripper_flip_count")
        )
        ablation_metrics_match = all(
            _close_enough(
                analysis["action_delta_summaries"]["full_vs_ablation"][key],
                row["full_vs_ablation"][key],
            )
            for key in ("rms", "max_abs", "translation_rms", "rotation_rms", "gripper_flip_count")
        )
        if not full_metrics_match or not ablation_metrics_match:
            raise RuntimeError("diagnostic replay action metrics do not match the saved Stage 0 row")
        result.update(
            {
                "execution_valid": True,
                "decision": analysis["decision"],
                "analysis": analysis,
                "exact_row_reproduction": {
                    "reliability_raw_match": raw_match,
                    "reliability_normalized_match": normalized_match,
                    "full_vs_base_metrics_match": full_metrics_match,
                    "full_vs_ablation_metrics_match": ablation_metrics_match,
                },
                "forward_counts": {
                    "xvla_generate_actions_calls": 3,
                    "rifa_action_hidden_hook_calls": int(hook.forward_count),
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                },
                "xvla": {
                    "model_class": type(model).__name__,
                    "device": str(next(model.parameters()).device),
                    "base_trainable_parameter_count": sum(
                        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                    ),
                    "optional_import_shims_used": shims,
                    "transformers_compat_patches": patches,
                },
            }
        )
    except Exception as exc:  # pragma: no cover - empirical boundary
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-100:],
            }
        )
        result["decision"] = "RIFA_GRIPPER_INTERNAL_SIGNAL_UNAVAILABLE"
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if env is not None:
            env.close()
        try:
            del model, clip
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                result["cuda"] = cuda_report()
                torch.cuda.empty_cache()
        except Exception as cleanup_exc:  # pragma: no cover
            result.setdefault("cleanup_exceptions", []).append(str(cleanup_exc))
        hashes_after = {str(path): sha256_file(path) for path in frozen_inputs}
        result["frozen_input_hashes_after"] = hashes_after
        result["frozen_inputs_unchanged"] = hashes_after == hashes_before
        result["original_outputs_modified"] = not bool(result["frozen_inputs_unchanged"])
        result["system_ram"] = memory_report()
        result["nvidia_smi_after"] = nvidia_smi()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at"] = utcish_timestamp()
        atomic_write_json(report_json, result)
        _write_markdown(report_md, result)
    code = 0 if result.get("execution_valid") and result.get("frozen_inputs_unchanged") else 2
    return code, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=pathlib.Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=pathlib.Path, default=DEFAULT_REPORT_MD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, result = run_postmortem(args.report_json, args.report_md)
    print(json.dumps({"decision": result["decision"], "execution_valid": result["execution_valid"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
