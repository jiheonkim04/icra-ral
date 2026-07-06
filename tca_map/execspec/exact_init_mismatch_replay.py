"""Bounded exact-init replay for ExecSpec action-space mismatches."""

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

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import (
    _as_path,
    _compact,
    _load_env_class,
    _load_json,
    _write_json,
)
from tca_map.datasets.libero_full_demo_expert_replay_sanity import (
    _read_demo_full,
    _run_replay_variant,
)
from tca_map.execspec.mismatch_diagnostic import PERTURBATION_ORDER, _perturbations, action_metrics


SCHEMA_VERSION = "2026-07-07.execspec_exact_init_mismatch_replay.v1"
TASK_GATE = "ALLOW_EXECSPEC_MISMATCH_REPLAY"
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
    "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT",
)
DEFAULT_REPLAY_VARIANTS = (
    "correct_7d_expert_action_replay",
    "gripper_sign_flip",
    "translation_scale_mismatch",
)


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_execspec_exact_init_mismatch_replay": True,
        "task_local_gate_required": f"{TASK_GATE}=1",
        "downloads_performed": False,
        "installs_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "lora_training_performed": False,
        "loss_computed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "learned_policy_inference_performed": False,
        "simulator_environment_created": False,
        "replay_or_rollout_performed": False,
        "diagnostic_rollouts_performed": False,
        "benchmark_rollouts_performed": False,
        "multi_seed_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
        "forbidden_gates_set": forbidden,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = report.get("result", {})
    decision = report.get("decision") or {}
    lines = [
        "# ExecSpec Exact-Init Mismatch Replay",
        "",
        "This is bounded exact-init replay evidence only. It is not benchmark success or paper-grade evidence.",
        "",
        f"- replay passed: `{result.get('passed')}`",
        f"- replay/rollout happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- total simulator steps: `{result.get('total_steps_performed')}`",
        f"- expert replay succeeded: `{decision.get('expert_replay_succeeded')}`",
        f"- replay degradation: `{decision.get('replay_degradation')}`",
        f"- strongest degradation variant: `{decision.get('strongest_replay_degradation_variant')}`",
        f"- recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_variants(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    return names or list(DEFAULT_REPLAY_VARIANTS)


def _first_pair(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("counterfactual split manifest has no pairs")
    return pairs[0]


def build_execspec_replay_case(
    manifest_path: Path,
    *,
    max_steps_cap: int = 300,
    post_signal_margin: int = 20,
    replay_variants: str | list[str] | tuple[str, ...] = DEFAULT_REPLAY_VARIANTS,
) -> dict[str, Any]:
    pair = _first_pair(manifest_path)
    positive = _read_demo_full(
        _as_path(pair["positive_demo_file"]),
        max_steps_cap=max_steps_cap,
        post_signal_margin=post_signal_margin,
    )
    actions = np.asarray(positive["actions"], dtype=np.float64)
    perturbations = _perturbations(actions)
    names = _parse_variants(replay_variants)
    unknown = [name for name in names if name not in PERTURBATION_ORDER]
    if unknown:
        raise ValueError("unknown replay variants: " + ", ".join(unknown))
    variants = []
    action_diagnostics: dict[str, Any] = {}
    for name in names:
        raw_actions = np.asarray(perturbations[name]["raw_actions"], dtype=np.float64)
        env_actions = np.clip(raw_actions, -1.0, 1.0)
        action_diagnostics[name] = {
            "description": perturbations[name]["description"],
            "policy_action_dim_before_bridge": perturbations[name]["policy_action_dim_before_bridge"],
            "metrics": action_metrics(actions, raw_actions),
            "env_actions_are_clipped_raw_actions": True,
        }
        variants.append(
            {
                "name": name,
                "claim_role": "exact_init_expert_control" if name == "correct_7d_expert_action_replay" else "execspec_mismatch_replay",
                "actions": env_actions,
                "use_exact_init_state": True,
            }
        )
    return {
        "pair_id": pair["pair_id"],
        "suite": pair.get("suite") or "libero_10",
        "task_id": pair["positive_task_id"],
        "instruction": pair["positive_instruction"],
        "counterfactual_task_id": pair.get("counterfactual_task_id"),
        "counterfactual_instruction": pair.get("counterfactual_instruction"),
        "positive_demo_path": positive["path"],
        "demo_name": positive["demo_name"],
        "init_state": positive["init_state"],
        "target_horizon": int(actions.shape[0]),
        "hdf5_metadata": {
            "full_action_steps": positive["full_action_steps"],
            "num_samples_attr": positive["num_samples_attr"],
            "first_positive_reward_index": positive["first_reward_index"],
            "first_done_index": positive["first_done_index"],
            "first_signal_index": positive["first_signal_index"],
            "target_horizon": positive["target_horizon"],
            "states0_l2_to_init_state": positive["states0_l2_to_init_state"],
            "model_file_available": positive["model_file_available"],
        },
        "action_diagnostics": action_diagnostics,
        "variants": variants,
    }


def _variant_success(variant: dict[str, Any]) -> bool:
    return bool(
        variant.get("final_success")
        or variant.get("done_seen")
        or float(variant.get("reward_sum") or 0.0) > 0.0
    )


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    case = (report.get("cases") or [{}])[0]
    variants = {item.get("variant"): item for item in case.get("replay_results", [])}
    expert = variants.get("correct_7d_expert_action_replay", {})
    expert_success = _variant_success(expert)
    expert_reward = float(expert.get("reward_sum") or 0.0)
    mismatches = [
        item
        for name, item in variants.items()
        if name and name != "correct_7d_expert_action_replay"
    ]
    degraded = []
    for item in mismatches:
        success_drop = expert_success and not _variant_success(item)
        reward_drop = expert_reward - float(item.get("reward_sum") or 0.0)
        if success_drop or reward_drop > 0.0:
            degraded.append((item.get("variant"), success_drop, reward_drop))
    strongest = None
    if degraded:
        strongest = max(degraded, key=lambda entry: (1 if entry[1] else 0, entry[2]))[0]
    return {
        "expert_replay_succeeded": expert_success,
        "expert_reward_sum": round(expert_reward, 6),
        "replay_degradation": bool(degraded),
        "degraded_variants": [entry[0] for entry in degraded],
        "strongest_replay_degradation_variant": strongest,
        "continue_or_kill": "continue" if expert_success and degraded else ("blocker" if not expert_success else "review"),
        "next_state": "STATE 2: replay calibrated repair for strongest mismatch" if expert_success and degraded else "diagnose replay blocker before repair claims",
    }


def run_execspec_exact_init_mismatch_replay(
    *,
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    libero_root: Path,
    robosuite_root: Path,
    max_steps_cap: int = 300,
    post_signal_margin: int = 20,
    camera_size: int = 64,
    replay_variants: str | list[str] | tuple[str, ...] = DEFAULT_REPLAY_VARIANTS,
) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_exact_init_mismatch_replay",
        "policy": _policy(forbidden),
        "inputs": {
            "manifest_path": str(manifest_path),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "max_steps_cap": int(max_steps_cap),
            "post_signal_margin": int(post_signal_margin),
            "camera_size": int(camera_size),
            "replay_variants": _parse_variants(replay_variants),
        },
        "cases": [],
        "result": {"passed": False, "reason": None, "total_steps_performed": 0, "variant_count": 0},
        "decision": None,
        "elapsed_seconds": None,
        "recommended_next_step": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for this bounded exact-init mismatch replay")
    if max_steps_cap < 1 or max_steps_cap > 320:
        stop_reasons.append("max_steps_cap must be between 1 and 320")
    if post_signal_margin < 0 or post_signal_margin > 50:
        stop_reasons.append("post_signal_margin must be between 0 and 50")
    if camera_size < 16 or camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    try:
        case = build_execspec_replay_case(
            manifest_path,
            max_steps_cap=max_steps_cap,
            post_signal_margin=post_signal_margin,
            replay_variants=replay_variants,
        )
    except Exception as exc:
        case = None
        stop_reasons.append(f"failed to build ExecSpec replay case: {type(exc).__name__}: {exc}")
    if stop_reasons:
        report["result"]["reason"] = "; ".join(stop_reasons)
        report["recommended_next_step"] = "Resolve listed blockers before exact-init ExecSpec mismatch replay."
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(report_json, report)
        _write_markdown(report_md, report)
        return report
    try:
        assert case is not None
        env_cls = _load_env_class(libero_root, robosuite_root)
        bddl_file = libero_root / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
        case_summary = {
            "pair_id": case["pair_id"],
            "task_id": case["task_id"],
            "instruction": case["instruction"],
            "counterfactual_task_id": case["counterfactual_task_id"],
            "counterfactual_instruction": case["counterfactual_instruction"],
            "positive_demo_path": case["positive_demo_path"],
            "demo_name": case["demo_name"],
            "bddl_file": str(bddl_file),
            "target_horizon": case["target_horizon"],
            "hdf5_metadata": case["hdf5_metadata"],
            "action_diagnostics": case["action_diagnostics"],
            "replay_results": [],
        }
        total_steps = 0
        for variant in case["variants"]:
            result = _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=camera_size,
                init_state=case["init_state"],
                variant=variant,
                instruction=case["instruction"],
            )
            case_summary["replay_results"].append(result)
            total_steps += int(result.get("steps_performed") or 0)
        report["cases"].append(case_summary)
        report["policy"]["simulator_environment_created"] = True
        report["policy"]["replay_or_rollout_performed"] = total_steps > 0
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = len(case_summary["replay_results"])
        report["result"]["passed"] = all(item.get("passed") for item in case_summary["replay_results"])
        report["result"]["reason"] = "bounded exact-init mismatch replay completed" if report["result"]["passed"] else "one or more replay variants failed"
        report["decision"] = _classify(report)
        report["recommended_next_step"] = (
            "Proceed to STATE 2 with bounded replay of a calibrated repair for the strongest degraded mismatch."
            if report["decision"]["replay_degradation"]
            else "Inspect replay result before claiming mismatch-induced degradation."
        )
    except Exception as exc:
        report["result"]["reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["recommended_next_step"] = "Diagnose simulator, init-state, or mismatch action error before repair replay."
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/execspec_exact_init_mismatch_replay_report.json")
    parser.add_argument("--report-md", default="reports/execspec_exact_init_mismatch_replay_report.md")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-steps-cap", type=int, default=300)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--replay-variants", default=",".join(DEFAULT_REPLAY_VARIANTS))
    args = parser.parse_args()
    report = run_execspec_exact_init_mismatch_replay(
        manifest_path=_as_path(args.manifest),
        report_json=_as_path(args.report_json),
        report_md=_as_path(args.report_md),
        libero_root=_as_path(args.libero_root),
        robosuite_root=_as_path(args.robosuite_root),
        max_steps_cap=args.max_steps_cap,
        post_signal_margin=args.post_signal_margin,
        camera_size=args.camera_size,
        replay_variants=args.replay_variants,
    )
    if os.environ.get(TASK_GATE) == "1":
        summary = {
            "schema_version": report.get("schema_version"),
            "report_json": str(_as_path(args.report_json)),
            "result": report.get("result"),
            "decision": report.get("decision"),
            "recommended_next_step": report.get("recommended_next_step"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
