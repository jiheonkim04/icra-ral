"""Bounded fixed-prior rollout diagnostic over local LIBERO HDF5 actions.

This runner is intentionally small. It uses local HDF5 demonstration actions as
rollout candidate actions after the 7D readiness gate passes. It performs no
training, no model loading, no VLA inference, no downloads, no GPU work, and no
paper-grade benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "2026-07-06.libero_fixed_prior_rollout_diagnostic.v1"
TASK_GATE = "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC"
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
)


def _as_path(value: str | Path) -> Path:
    text = str(value)
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = report.get("result", {})
    lines = [
        "# Fixed-Prior Rollout Diagnostic",
        "",
        "This is a bounded diagnostic only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- diagnostic passed: `{result.get('passed')}`",
        f"- rollout happened: `{report['policy']['diagnostic_rollouts_performed']}`",
        f"- simulator environments created: `{report['policy']['simulator_environment_created']}`",
        f"- total steps: `{result.get('total_steps_performed')}`",
        f"- fixed-prior support label: `{result.get('fixed_prior_support_label')}`",
        f"- recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _policy() -> dict[str, Any]:
    return {
        "bounded_fixed_prior_rollout_diagnostic": True,
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
        "diagnostic_rollouts_performed": False,
        "benchmark_rollouts_performed": False,
        "multi_seed_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "benchmark_claims_made": False,
        "sota_claims_made": False,
        "paper_grade_claims_made": False,
    }


def _read_demo(path: Path, max_steps: int) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        demo_name = sorted(data_group.keys())[0]
        demo = data_group[demo_name]
        if "actions" not in demo:
            raise ValueError(f"{path} demo {demo_name} has no actions dataset")
        if "init_state" not in demo.attrs:
            raise ValueError(f"{path} demo {demo_name} has no init_state attribute")
        actions = np.asarray(demo["actions"][:max_steps], dtype=np.float64)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path} actions must be [T, 7], got {list(actions.shape)}")
    return {
        "path": str(path),
        "demo_name": demo_name,
        "init_state": init_state,
        "actions": actions,
        "action_shape": list(actions.shape),
    }


def _range(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "max_abs": None}
    return {
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
        "max_abs": round(float(np.max(np.abs(values))), 6),
    }


def _action_stats(actions: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "finite": bool(np.all(np.isfinite(actions))),
        "range": _range(actions),
        "translation_range": _range(actions[:, :3]),
        "rotation_range": _range(actions[:, 3:6]),
        "gripper_range": _range(actions[:, 6:7]),
        "clip_rate_if_env_adapter_applied": round(float(np.mean(np.abs(actions) > 1.0)), 6),
    }


def _mean_l1(left: np.ndarray, right: np.ndarray) -> float:
    steps = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    if steps == 0 or width == 0:
        return 0.0
    return float(np.mean(np.abs(left[:steps, :width] - right[:steps, :width])))


def build_fixed_prior_rollout_cases(manifest_path: Path, *, max_tasks: int = 1, max_steps: int = 10) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")
    cases: list[dict[str, Any]] = []
    for pair in manifest.get("counterfactual_pairs", [])[:max_tasks]:
        positive = _read_demo(_as_path(pair["positive_demo_file"]), max_steps)
        counter = _read_demo(_as_path(pair["counterfactual_demo_file"]), max_steps)
        steps = min(int(positive["actions"].shape[0]), int(counter["actions"].shape[0]), max_steps)
        if steps < 1:
            raise ValueError(f"pair {pair.get('pair_id')} has no actions")
        positive_actions = positive["actions"][:steps]
        counter_actions = counter["actions"][:steps]
        actionmap_actions = np.clip((positive_actions + counter_actions) / 2.0, -1.0, 1.0)
        variants = [
            {
                "name": "actionmap_style_target_agnostic_mean",
                "claim_role": "baseline_style_diagnostic",
                "actions": actionmap_actions,
            },
            {
                "name": "fixed_semantic_target_prior_tca",
                "claim_role": "main_fixed_prior_diagnostic",
                "actions": positive_actions,
            },
            {
                "name": "oracle_target_tca_upper_bound",
                "claim_role": "oracle_upper_bound_not_method",
                "actions": positive_actions,
            },
        ]
        cases.append(
            {
                "pair_id": pair["pair_id"],
                "suite": pair.get("suite") or "libero_10",
                "task_id": pair["positive_task_id"],
                "instruction": pair["positive_instruction"],
                "counterfactual_task_id": pair["counterfactual_task_id"],
                "counterfactual_instruction": pair["counterfactual_instruction"],
                "positive_demo_path": positive["path"],
                "counterfactual_demo_path": counter["path"],
                "demo_name": positive["demo_name"],
                "init_state": positive["init_state"],
                "max_steps": steps,
                "variants": variants,
                "action_diagnostics": {
                    "positive_demo": _action_stats(positive_actions),
                    "counterfactual_demo": _action_stats(counter_actions),
                    "actionmap_style_target_agnostic_mean": _action_stats(actionmap_actions),
                    "candidate_positive_vs_counter_l1": round(_mean_l1(positive_actions, counter_actions), 6),
                    "actionmap_l1_to_positive": round(_mean_l1(actionmap_actions, positive_actions), 6),
                    "fixed_prior_l1_to_positive": 0.0,
                },
            }
        )
    return cases


def _compact(value: Any, limit: int = 800) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _extract_eef_pos(obs: Any) -> list[float] | None:
    if not isinstance(obs, dict):
        return None
    for key in ("robot0_eef_pos", "ee_pos", "eef_pos"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            return [float(value) for value in arr[:3]]
    return None


def _run_case_variant(*, env_cls: Any, bddl_file: Path, camera_size: int, init_state: np.ndarray, variant: dict[str, Any]) -> dict[str, Any]:
    actions = np.asarray(variant["actions"], dtype=np.float64)
    summary: dict[str, Any] = {
        "variant": variant["name"],
        "claim_role": variant["claim_role"],
        "bddl_file": str(bddl_file),
        "action_shape": list(actions.shape),
        "env_action_shape": 7,
        "action_stats": _action_stats(actions),
        "env_created": False,
        "reset_ok": False,
        "set_init_state_ok": False,
        "steps_performed": 0,
        "reward_sum": 0.0,
        "success_check": None,
        "done_seen": False,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "available_obs_keys": [],
        "target_directed_movement_score": "not_available_no_object_position_labels",
        "wrong_target_movement_proxy": "not_available_no_object_position_labels",
        "error": None,
    }
    env = None
    try:
        env = env_cls(
            bddl_file_name=str(bddl_file),
            camera_heights=camera_size,
            camera_widths=camera_size,
        )
        summary["env_created"] = True
        env.seed(0)
        env.reset()
        summary["reset_ok"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["eef_start"] = _extract_eef_pos(obs)
        if isinstance(obs, dict):
            summary["available_obs_keys"] = sorted(str(key) for key in obs.keys())[:40]
        for action in actions:
            obs, reward, done, info = env.step(action)
            summary["steps_performed"] += 1
            summary["reward_sum"] += float(reward)
            summary["done_seen"] = bool(summary["done_seen"] or done)
        summary["eef_final"] = _extract_eef_pos(obs)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            start = np.asarray(summary["eef_start"], dtype=np.float64)
            final = np.asarray(summary["eef_final"], dtype=np.float64)
            summary["eef_displacement_l2"] = round(float(np.linalg.norm(final - start)), 6)
        try:
            summary["success_check"] = bool(env.check_success())
        except Exception:
            summary["success_check"] = None
    except Exception as exc:
        summary["error"] = _compact(f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    summary["passed"] = bool(
        summary["env_created"]
        and summary["reset_ok"]
        and summary["set_init_state_ok"]
        and summary["steps_performed"] == int(actions.shape[0])
        and summary["error"] is None
    )
    return summary


def _load_env_class(libero_root: Path, robosuite_root: Path) -> Any:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]
    sys.path = [path for path in sys.path if not str(path).startswith(str(libero_root))]
    for path in (robosuite_root, libero_root):
        if str(path):
            sys.path.insert(0, str(path))
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def run_fixed_prior_rollout_diagnostic(
    *,
    manifest_path: Path,
    readiness_report_path: Path,
    report_json: Path,
    report_md: Path,
    libero_root: Path,
    robosuite_root: Path,
    max_tasks: int = 1,
    max_steps: int = 10,
    camera_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    readiness = _load_json(readiness_report_path) if readiness_report_path.exists() else {}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(),
        "inputs": {
            "manifest_path": str(manifest_path),
            "readiness_report_path": str(readiness_report_path),
            "libero_root": str(libero_root),
            "robosuite_root": str(robosuite_root),
            "max_tasks": max_tasks,
            "max_steps": max_steps,
            "camera_size": camera_size,
        },
        "readiness_gate": {
            "risk_gate_status": readiness.get("risk_gate_status"),
            "rollout_diagnostic_authorized": bool(readiness.get("rollout_diagnostic_authorized")),
        },
        "cases": [],
        "result": {
            "passed": False,
            "total_steps_performed": 0,
            "variant_count": 0,
            "fixed_prior_support_label": "not_run",
            "reason": None,
        },
        "forbidden_gates_set": forbidden,
        "elapsed_seconds": None,
        "recommended_next_step": None,
    }
    stop_reasons: list[str] = []
    if forbidden:
        stop_reasons.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if os.environ.get(TASK_GATE) != "1":
        stop_reasons.append(f"{TASK_GATE}=1 is required for this bounded rollout diagnostic")
    if readiness.get("risk_gate_status") != "green" or not readiness.get("rollout_diagnostic_authorized"):
        stop_reasons.append("fixed-prior rollout readiness gate is not green/authorized")
    if max_tasks < 1 or max_tasks > 3:
        stop_reasons.append("max_tasks must be between 1 and 3")
    if max_steps < 1 or max_steps > 25:
        stop_reasons.append("max_steps must be between 1 and 25")
    if camera_size < 16 or camera_size > 128:
        stop_reasons.append("camera_size must be between 16 and 128")
    try:
        cases = build_fixed_prior_rollout_cases(manifest_path, max_tasks=max_tasks, max_steps=max_steps)
    except Exception as exc:
        cases = []
        stop_reasons.append(f"failed to build fixed-prior rollout cases: {type(exc).__name__}: {exc}")
    if stop_reasons:
        report["result"]["reason"] = "; ".join(stop_reasons)
        report["recommended_next_step"] = "Resolve listed blockers before running fixed-prior rollout."
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(report_json, report)
        _write_markdown(report_md, report)
        return report

    try:
        env_cls = _load_env_class(libero_root, robosuite_root)
        report["policy"]["simulator_environment_created"] = True
        total_steps = 0
        variant_count = 0
        for case in cases:
            bddl_file = libero_root / "libero" / "libero" / "bddl_files" / case["suite"] / f"{case['task_id']}.bddl"
            case_summary = {
                "pair_id": case["pair_id"],
                "task_id": case["task_id"],
                "instruction": case["instruction"],
                "counterfactual_task_id": case["counterfactual_task_id"],
                "counterfactual_instruction": case["counterfactual_instruction"],
                "demo_name": case["demo_name"],
                "max_steps": case["max_steps"],
                "bddl_file": str(bddl_file),
                "action_diagnostics": case["action_diagnostics"],
                "variants": [],
            }
            for variant in case["variants"]:
                variant_summary = _run_case_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=camera_size,
                    init_state=case["init_state"],
                    variant=variant,
                )
                case_summary["variants"].append(variant_summary)
                total_steps += int(variant_summary["steps_performed"])
                variant_count += 1
            report["cases"].append(case_summary)
        report["policy"]["diagnostic_rollouts_performed"] = total_steps > 0
        report["result"]["total_steps_performed"] = total_steps
        report["result"]["variant_count"] = variant_count
        all_passed = all(
            variant.get("passed")
            for case in report["cases"]
            for variant in case.get("variants", [])
        )
        actionmap = None
        fixed = None
        for case in report["cases"]:
            for variant in case.get("variants", []):
                if variant["variant"] == "actionmap_style_target_agnostic_mean":
                    actionmap = variant
                if variant["variant"] == "fixed_semantic_target_prior_tca":
                    fixed = variant
        if actionmap and fixed:
            reward_delta = float(fixed["reward_sum"]) - float(actionmap["reward_sum"])
            support = "partial_action_bridge_support_no_success_gain"
            if bool(fixed.get("success_check")) and not bool(actionmap.get("success_check")):
                support = "limited_rollout_success_support"
            elif reward_delta > 0:
                support = "limited_reward_support"
            report["result"]["fixed_prior_reward_delta_vs_actionmap"] = round(reward_delta, 6)
            report["result"]["fixed_prior_support_label"] = support
        report["result"]["passed"] = bool(all_passed)
        report["result"]["reason"] = "bounded fixed-prior rollout diagnostic completed" if all_passed else "one or more variants failed"
        report["recommended_next_step"] = (
            "Use this as limited diagnostic evidence only; next inspect reward/success/movement deltas before any larger rollout."
            if all_passed
            else "Diagnose variant failure before any larger rollout."
        )
    except Exception as exc:
        report["result"]["reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["recommended_next_step"] = "Diagnose simulator or bridge error before any larger rollout."
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(report_json, report)
    _write_markdown(report_md, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--readiness-report", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--report-json", default="reports/fixed_prior_rollout_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/fixed_prior_rollout_diagnostic_report.md")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args()
    report = run_fixed_prior_rollout_diagnostic(
        manifest_path=_as_path(args.manifest),
        readiness_report_path=_as_path(args.readiness_report),
        report_json=_as_path(args.report_json),
        report_md=_as_path(args.report_md),
        libero_root=_as_path(args.libero_root),
        robosuite_root=_as_path(args.robosuite_root),
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        camera_size=args.camera_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if os.environ.get(TASK_GATE) == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
