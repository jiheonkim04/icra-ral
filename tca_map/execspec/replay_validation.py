"""STATE 3 multi-demo replay validation for ExecSpec-Repair.

This module extends the calibrated repair diagnostic beyond one held-out demo.
It keeps calibration and evaluation HDF5 actions separated, evaluates all
configured executable-spec mismatches on held-out actions, and optionally runs
bounded exact-init replay behind a task-local gate.
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

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _as_path, _compact, _load_env_class
from tca_map.datasets.libero_full_demo_expert_replay_sanity import _read_demo_full, _run_replay_variant
from tca_map.execspec import repair
from tca_map.execspec.mismatch_diagnostic import _load_json, _round, action_metrics


SCHEMA_VERSION = "2026-07-07.execspec_state3_replay_validation.v1"
TASK_GATE = "ALLOW_EXECSPEC_STATE3_REPLAY_VALIDATION"
FORBIDDEN_GATES = tuple(
    dict.fromkeys(
        list(repair.FORBIDDEN_GATES)
        + [
            repair.TASK_GATE,
            "ALLOW_EXECSPEC_MISMATCH_REPLAY",
            "ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY",
        ]
    )
)
DEFAULT_REPLAY_METHODS = (
    "wrong_executable_spec_replay",
    "clipping_only",
    "global_affine_calibration",
    "diagonal_affine_calibration",
    "gripper_only_calibration",
    "full_execspec_repair",
)
ACTION_METRIC_KEYS = (
    "action_l2_mean",
    "translation_drift_mean",
    "rotation_drift_mean",
    "gripper_mismatch_rate",
    "clip_rate_step",
    "controller_valid_action_rate",
)


def _parse_names(value: str | list[str] | tuple[str, ...], allowed: tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError("unknown names: " + ", ".join(unknown))
    return names or list(allowed)


def _task_id_from_demo_path(path: Path) -> str:
    stem = Path(path).stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem


def _instruction_from_task_id(task_id: str) -> str:
    return task_id.replace("_", " ")


def _norm(path: Path) -> str:
    return repair._norm_path(path)


def _manifest_eval_path(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("manifest has no counterfactual pairs")
    pair = pairs[0]
    return _as_path(pair["positive_demo_file"]), pair


def build_validation_split(
    *,
    manifest_path: Path,
    data_root: Path,
    max_calibration_demos: int,
    max_eval_demos: int,
) -> dict[str, Any]:
    manifest_eval_path, manifest_pair = _manifest_eval_path(manifest_path)
    all_paths = sorted(_as_path(path) for path in data_root.rglob("*.hdf5"))
    if not all_paths:
        raise ValueError(f"no HDF5 demos found under {data_root}")
    manifest_eval_norm = _norm(manifest_eval_path)

    calibration_paths: list[Path] = []
    for path in all_paths:
        if len(calibration_paths) >= max_calibration_demos:
            break
        if _norm(path) != manifest_eval_norm:
            calibration_paths.append(path)

    calibration_norm = {_norm(path) for path in calibration_paths}
    eval_paths = [manifest_eval_path]
    eval_norm = {manifest_eval_norm}
    for path in all_paths:
        if len(eval_paths) >= max_eval_demos:
            break
        norm = _norm(path)
        if norm not in calibration_norm and norm not in eval_norm:
            eval_paths.append(path)
            eval_norm.add(norm)

    overlap = sorted(calibration_norm & {_norm(path) for path in eval_paths})
    if overlap:
        raise ValueError("calibration/eval leakage detected: " + ", ".join(overlap))
    if not calibration_paths:
        raise ValueError("no calibration demos available")
    if not eval_paths:
        raise ValueError("no held-out eval demos available")

    return {
        "manifest_pair": manifest_pair,
        "calibration_paths": calibration_paths,
        "eval_paths": eval_paths,
        "leakage_detected": False,
    }


def _read_eval_items(paths: list[Path], max_steps_per_demo: int) -> list[dict[str, Any]]:
    items = []
    for path in paths:
        actions, meta = repair._read_actions(path, max_steps_per_demo)
        task_id = _task_id_from_demo_path(path)
        items.append(
            {
                "path": path,
                "suite": Path(path).parent.name,
                "task_id": task_id,
                "instruction": _instruction_from_task_id(task_id),
                "actions": actions,
                "metadata": meta,
            }
        )
    return items


def _fit_params_by_mismatch(calibration_actions: np.ndarray) -> dict[str, Any]:
    params = {}
    for mismatch in repair.MISMATCH_TYPES:
        source = repair.apply_mismatch(calibration_actions, mismatch)
        params[mismatch] = repair.fit_repair_parameters(source, calibration_actions)
    return params


def _recovery_fraction(wrong: float, repaired: float, correct: float = 0.0) -> float | None:
    return repair._recovery_fraction(wrong, repaired, correct)


def evaluate_action_cases(
    *,
    eval_items: list[dict[str, Any]],
    fitted_params: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = []
    for item in eval_items:
        reference = np.asarray(item["actions"], dtype=np.float64)
        correct_metrics = action_metrics(reference, reference)
        for mismatch in repair.MISMATCH_TYPES:
            source = repair.apply_mismatch(reference, mismatch)
            wrong_metrics = action_metrics(reference, source)
            method_reports: dict[str, Any] = {}
            for method in repair.REPAIR_METHODS:
                repaired = repair.apply_repair(source, method, fitted_params[mismatch])
                metrics = action_metrics(reference, repaired)
                method_reports[method] = {
                    "metrics": metrics,
                    "recovery_fraction": _recovery_fraction(
                        float(wrong_metrics["action_l2_mean"]),
                        float(metrics["action_l2_mean"]),
                        float(correct_metrics["action_l2_mean"]),
                    ),
                }
            identity_l2 = float(method_reports["identity_no_repair"]["metrics"]["action_l2_mean"])
            clipping_l2 = float(method_reports["clipping_only"]["metrics"]["action_l2_mean"])
            global_l2 = float(method_reports["global_affine_calibration"]["metrics"]["action_l2_mean"])
            for payload in method_reports.values():
                l2 = float(payload["metrics"]["action_l2_mean"])
                payload["beats_identity"] = bool(l2 + 1e-9 < identity_l2)
                payload["beats_clipping_only"] = bool(l2 + 1e-9 < clipping_l2)
                payload["beats_global_affine"] = bool(l2 + 1e-9 < global_l2)
            cases.append(
                {
                    "eval_demo_path": str(item["path"]),
                    "suite": item["suite"],
                    "task_id": item["task_id"],
                    "instruction": item["instruction"],
                    "mismatch_type": mismatch,
                    **repair._mismatch_metadata(mismatch),
                    "correct_spec_metrics": correct_metrics,
                    "wrong_spec_metrics": wrong_metrics,
                    "repair_methods": method_reports,
                }
            )
    return cases


def _mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": _round(float(np.mean(arr)), 9),
        "std": _round(float(np.std(arr)), 9),
        "count": int(arr.size),
    }


def _method_metric_stats(cases: list[dict[str, Any]], method: str) -> dict[str, Any]:
    stats = {}
    for key in ACTION_METRIC_KEYS:
        stats[key] = _mean_std(
            [
                float(case["repair_methods"][method]["metrics"][key])
                for case in cases
                if key in case["repair_methods"][method]["metrics"]
            ]
        )
    stats["recovery_fraction"] = _mean_std(
        [
            float(case["repair_methods"][method]["recovery_fraction"])
            for case in cases
            if case["repair_methods"][method]["recovery_fraction"] is not None
        ]
    )
    return stats


def aggregate_action_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {method: _method_metric_stats(cases, method) for method in repair.REPAIR_METHODS}
    by_mismatch: dict[str, Any] = {}
    for mismatch in repair.MISMATCH_TYPES:
        mismatch_cases = [case for case in cases if case["mismatch_type"] == mismatch]
        by_mismatch[mismatch] = {
            "case_count": len(mismatch_cases),
            "wrong_spec": {
                key: _mean_std([float(case["wrong_spec_metrics"][key]) for case in mismatch_cases])
                for key in ACTION_METRIC_KEYS
                if mismatch_cases and key in mismatch_cases[0]["wrong_spec_metrics"]
            },
            "methods": {method: _method_metric_stats(mismatch_cases, method) for method in repair.REPAIR_METHODS},
        }
    by_task: dict[str, Any] = {}
    for task_id in sorted({case["task_id"] for case in cases}):
        task_cases = [case for case in cases if case["task_id"] == task_id]
        by_task[task_id] = {
            "case_count": len(task_cases),
            "full_action_l2_mean": _method_metric_stats(task_cases, "full_execspec_repair")["action_l2_mean"],
            "full_recovery_fraction": _method_metric_stats(task_cases, "full_execspec_repair")["recovery_fraction"],
        }

    def mean_l2(method: str) -> float:
        return float(by_method[method]["action_l2_mean"]["mean"] or 0.0)

    full_beats_identity = mean_l2("full_execspec_repair") + 1e-9 < mean_l2("identity_no_repair")
    full_beats_clipping = mean_l2("full_execspec_repair") + 1e-9 < mean_l2("clipping_only")
    full_beats_global = mean_l2("full_execspec_repair") + 1e-9 < mean_l2("global_affine_calibration")
    beat_counts = {"identity": 0, "clipping_only": 0, "global_affine": 0, "total": len(repair.MISMATCH_TYPES)}
    for mismatch, payload in by_mismatch.items():
        full_l2 = float(payload["methods"]["full_execspec_repair"]["action_l2_mean"]["mean"] or 0.0)
        if full_l2 + 1e-9 < float(payload["methods"]["identity_no_repair"]["action_l2_mean"]["mean"] or 0.0):
            beat_counts["identity"] += 1
        if full_l2 + 1e-9 < float(payload["methods"]["clipping_only"]["action_l2_mean"]["mean"] or 0.0):
            beat_counts["clipping_only"] += 1
        if full_l2 + 1e-9 < float(payload["methods"]["global_affine_calibration"]["action_l2_mean"]["mean"] or 0.0):
            beat_counts["global_affine"] += 1

    method_recovery = {
        method: float(by_method[method]["recovery_fraction"]["mean"] or 0.0)
        for method in repair.REPAIR_METHODS
    }
    best_method = max(method_recovery, key=method_recovery.get)
    return {
        "overall_by_method": by_method,
        "per_mismatch": by_mismatch,
        "per_task": by_task,
        "mean_action_l2": {
            "identity_no_repair": _round(mean_l2("identity_no_repair"), 9),
            "clipping_only": _round(mean_l2("clipping_only"), 9),
            "global_affine_calibration": _round(mean_l2("global_affine_calibration"), 9),
            "diagonal_affine_calibration": _round(mean_l2("diagonal_affine_calibration"), 9),
            "full_execspec_repair": _round(mean_l2("full_execspec_repair"), 9),
        },
        "full_repair_beats_identity_on_action_drift": bool(full_beats_identity),
        "full_repair_beats_clipping_only_on_action_drift": bool(full_beats_clipping),
        "full_repair_beats_global_affine_on_action_drift": bool(full_beats_global),
        "per_mismatch_full_beat_counts": beat_counts,
        "full_repair_mean_recovery_fraction": _round(method_recovery["full_execspec_repair"], 9),
        "best_repair_method_by_mean_recovery": best_method,
        "best_repair_method_mean_recovery_fraction": _round(method_recovery[best_method], 9),
    }


def calibration_sensitivity(
    *,
    split: dict[str, Any],
    eval_items: list[dict[str, Any]],
    max_actions_per_demo: int,
    sizes: list[int],
) -> list[dict[str, Any]]:
    results = []
    available = len(split["calibration_paths"])
    for size in sizes:
        if size < 1 or size > available:
            continue
        calibration_actions, _ = repair._concat_actions(split["calibration_paths"][:size], max_actions_per_demo)
        params = _fit_params_by_mismatch(calibration_actions)
        cases = evaluate_action_cases(eval_items=eval_items, fitted_params=params)
        aggregate = aggregate_action_cases(cases)
        results.append(
            {
                "calibration_demo_count": size,
                "calibration_action_samples": int(calibration_actions.shape[0]),
                "full_repair_mean_action_l2": aggregate["mean_action_l2"]["full_execspec_repair"],
                "full_repair_mean_recovery_fraction": aggregate["full_repair_mean_recovery_fraction"],
                "full_beats_identity": aggregate["full_repair_beats_identity_on_action_drift"],
                "full_beats_clipping_only": aggregate["full_repair_beats_clipping_only_on_action_drift"],
                "full_beats_global_affine": aggregate["full_repair_beats_global_affine_on_action_drift"],
                "success_recovery_rate": None,
                "success_replay_evaluated": False,
                "limitation": "cheap sensitivity uses held-out action metrics only; replay sensitivity is not rerun for each calibration size",
            }
        )
    return results


def _success(result: dict[str, Any]) -> bool:
    return repair._variant_success(result)


def _reward(result: dict[str, Any]) -> float:
    return float(result.get("reward_sum") or 0.0)


def _done_index(result: dict[str, Any]) -> int | None:
    value = result.get("first_done_index")
    return None if value is None else int(value)


def _method_actions(reference: np.ndarray, mismatch: str, method: str, params: dict[str, Any]) -> np.ndarray:
    source = repair.apply_mismatch(reference, mismatch)
    repair_method = "identity_no_repair" if method == "wrong_executable_spec_replay" else method
    return repair.apply_repair(source, repair_method, params)


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return repair._compact_replay_result(result)


def _replay_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    variants = {item["variant"]: item for item in case.get("replay_results", [])}
    expert = variants.get("correct_7d_expert_action_replay", {})
    wrong = variants.get("wrong_executable_spec_replay", {})
    clipping = variants.get("clipping_only", {})
    global_affine = variants.get("global_affine_calibration", {})
    full = variants.get("full_execspec_repair", {})
    expert_success = _success(expert)
    wrong_success = _success(wrong)
    full_success = _success(full)
    expert_reward = _reward(expert)
    wrong_reward = _reward(wrong)
    full_reward = _reward(full)
    degraded_success = bool(expert_success and not wrong_success)
    degraded_reward = bool(expert_reward > wrong_reward)
    recovered_success = bool(degraded_success and full_success)
    recovered_reward = bool(degraded_reward and full_reward > wrong_reward)
    expert_done = _done_index(expert)
    wrong_done = _done_index(wrong)
    full_done = _done_index(full)
    done_recovered = bool(expert_done is not None and full_done is not None and (wrong_done is None or abs(full_done - expert_done) < abs(wrong_done - expert_done)))
    baseline_match = bool(
        (degraded_success and (_success(clipping) or _success(global_affine)))
        or (degraded_reward and (_reward(clipping) >= full_reward and full_reward > wrong_reward))
        or (degraded_reward and (_reward(global_affine) >= full_reward and full_reward > wrong_reward))
    )
    failure_reasons = []
    if not expert_success and expert_reward <= 0.0:
        failure_reasons.append("expert_upper_bound_failed")
    if expert_success and not degraded_success and not degraded_reward:
        failure_reasons.append("wrong_spec_did_not_degrade")
    if (degraded_success or degraded_reward) and not (recovered_success or recovered_reward or done_recovered):
        failure_reasons.append("full_repair_did_not_recover_replay")
    if baseline_match:
        failure_reasons.append("clipping_or_global_matches_full_replay_recovery")
    return {
        "expert_replay_succeeded": expert_success,
        "wrong_spec_succeeded": wrong_success,
        "full_repair_succeeded": full_success,
        "expert_reward_sum": _round(expert_reward, 6),
        "wrong_spec_reward_sum": _round(wrong_reward, 6),
        "full_repair_reward_sum": _round(full_reward, 6),
        "success_degraded": degraded_success,
        "reward_degraded": degraded_reward,
        "success_recovered": recovered_success,
        "reward_recovered": recovered_reward,
        "done_index_recovered": done_recovered,
        "simple_baseline_matches_full": baseline_match,
        "failure_reasons": failure_reasons,
    }


def run_exact_init_replay(
    *,
    split: dict[str, Any],
    fitted_params: dict[str, Any],
    replay_mismatches: list[str],
    replay_methods: list[str],
    max_replay_eval_demos: int,
    libero_root: Path,
    robosuite_root: Path,
    max_steps_cap: int,
    post_signal_margin: int,
    camera_size: int,
    include_default_reset_sanity: bool,
) -> dict[str, Any]:
    env_cls = _load_env_class(libero_root, robosuite_root)
    cases = []
    total_steps = 0
    default_reset_sanity: dict[str, Any] = {"performed": False, "cases": []}
    for demo_index, eval_path in enumerate(split["eval_paths"][:max_replay_eval_demos]):
        eval_path = _as_path(eval_path)
        task_id = _task_id_from_demo_path(eval_path)
        suite = Path(eval_path).parent.name
        instruction = _instruction_from_task_id(task_id)
        bddl_file = libero_root / "libero" / "libero" / "bddl_files" / suite / f"{task_id}.bddl"
        demo = _read_demo_full(eval_path, max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
        reference = np.asarray(demo["actions"], dtype=np.float64)
        expert_variant = {
            "name": "correct_7d_expert_action_replay",
            "claim_role": "expert_upper_bound",
            "actions": reference,
            "use_exact_init_state": True,
        }
        expert_result_raw = _run_replay_variant(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=camera_size,
            init_state=demo["init_state"],
            variant=expert_variant,
            instruction=instruction,
        )
        total_steps += int(expert_result_raw.get("steps_performed") or 0)
        expert_result = _compact_result(expert_result_raw)
        for mismatch in replay_mismatches:
            action_diagnostics = {
                "correct_7d_expert_action_replay": {"metrics": action_metrics(reference, reference)}
            }
            replay_results = [expert_result]
            source = repair.apply_mismatch(reference, mismatch)
            for method in replay_methods:
                method_actions = _method_actions(reference, mismatch, method, fitted_params[mismatch])
                action_diagnostics[method] = {
                    "metrics": action_metrics(reference, method_actions),
                    "env_actions_are_clipped_for_bounded_replay": True,
                }
                variant = {
                    "name": method,
                    "claim_role": "identity_no_repair" if method == "wrong_executable_spec_replay" else method,
                    "actions": np.clip(method_actions, -1.0, 1.0),
                    "use_exact_init_state": True,
                }
                result_raw = _run_replay_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=camera_size,
                    init_state=demo["init_state"],
                    variant=variant,
                    instruction=instruction,
                )
                total_steps += int(result_raw.get("steps_performed") or 0)
                result = _compact_result(result_raw)
                replay_results.append(result)
                if method == "wrong_executable_spec_replay":
                    alias = dict(result)
                    alias["variant"] = "identity_no_repair"
                    alias["alias_of"] = "wrong_executable_spec_replay"
                    action_diagnostics["identity_no_repair"] = action_diagnostics[method]
                    replay_results.append(alias)
            case = {
                "eval_demo_index": demo_index,
                "eval_demo_path": str(eval_path),
                "suite": suite,
                "task_id": task_id,
                "instruction": instruction,
                "mismatch_type": mismatch,
                **repair._mismatch_metadata(mismatch),
                "bddl_file": str(bddl_file),
                "target_horizon": int(reference.shape[0]),
                "hdf5_metadata": {
                    "first_positive_reward_index": demo["first_reward_index"],
                    "first_done_index": demo["first_done_index"],
                    "first_signal_index": demo["first_signal_index"],
                    "full_action_steps": demo["full_action_steps"],
                },
                "action_diagnostics": action_diagnostics,
                "replay_results": replay_results,
            }
            case["summary"] = _replay_case_summary(case)
            cases.append(case)
            if include_default_reset_sanity and not default_reset_sanity["performed"] and mismatch == replay_mismatches[0]:
                full_actions = _method_actions(reference, mismatch, "full_execspec_repair", fitted_params[mismatch])
                sanity_variants = [
                    {
                        "name": "default_reset_expert_replay",
                        "claim_role": "default_reset_sanity_expert",
                        "actions": reference,
                        "use_exact_init_state": False,
                    },
                    {
                        "name": "default_reset_full_execspec_repair",
                        "claim_role": "default_reset_sanity_full_repair",
                        "actions": np.clip(full_actions, -1.0, 1.0),
                        "use_exact_init_state": False,
                    },
                ]
                sanity_results = []
                for sanity_variant in sanity_variants:
                    sanity_raw = _run_replay_variant(
                        env_cls=env_cls,
                        bddl_file=bddl_file,
                        camera_size=camera_size,
                        init_state=demo["init_state"],
                        variant=sanity_variant,
                        instruction=instruction,
                    )
                    total_steps += int(sanity_raw.get("steps_performed") or 0)
                    sanity_results.append(_compact_result(sanity_raw))
                default_reset_sanity = {
                    "performed": True,
                    "scope": "one demo and one mismatch only; non-primary sanity check",
                    "eval_demo_path": str(eval_path),
                    "mismatch_type": mismatch,
                    "results": sanity_results,
                    "expert_default_reset_succeeded": _success(sanity_results[0]),
                    "full_default_reset_succeeded": _success(sanity_results[1]),
                }
    return {
        "cases": cases,
        "aggregate": aggregate_replay_cases(cases),
        "default_reset_sanity": default_reset_sanity,
        "total_steps_performed": total_steps,
    }


def aggregate_replay_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    degraded = [case for case in cases if case.get("summary", {}).get("success_degraded") or case.get("summary", {}).get("reward_degraded")]
    success_recovered = [case for case in degraded if case.get("summary", {}).get("success_recovered")]
    reward_recovered = [case for case in degraded if case.get("summary", {}).get("reward_recovered")]
    done_recovered = [case for case in degraded if case.get("summary", {}).get("done_index_recovered")]
    baseline_matches = [case for case in degraded if case.get("summary", {}).get("simple_baseline_matches_full")]
    failures = [case for case in cases if case.get("summary", {}).get("failure_reasons")]

    def rate(num: int, denom: int) -> float | None:
        if denom <= 0:
            return None
        return _round(num / denom, 9)

    per_mismatch: dict[str, Any] = {}
    for mismatch in repair.MISMATCH_TYPES:
        mismatch_cases = [case for case in cases if case["mismatch_type"] == mismatch]
        mismatch_degraded = [case for case in mismatch_cases if case.get("summary", {}).get("success_degraded") or case.get("summary", {}).get("reward_degraded")]
        per_mismatch[mismatch] = {
            "case_count": len(mismatch_cases),
            "degraded_case_count": len(mismatch_degraded),
            "success_recovered_count": sum(bool(case.get("summary", {}).get("success_recovered")) for case in mismatch_degraded),
            "reward_recovered_count": sum(bool(case.get("summary", {}).get("reward_recovered")) for case in mismatch_degraded),
            "done_index_recovered_count": sum(bool(case.get("summary", {}).get("done_index_recovered")) for case in mismatch_degraded),
        }
    per_task: dict[str, Any] = {}
    for task_id in sorted({case["task_id"] for case in cases}):
        task_cases = [case for case in cases if case["task_id"] == task_id]
        task_degraded = [case for case in task_cases if case.get("summary", {}).get("success_degraded") or case.get("summary", {}).get("reward_degraded")]
        per_task[task_id] = {
            "case_count": len(task_cases),
            "degraded_case_count": len(task_degraded),
            "success_recovered_count": sum(bool(case.get("summary", {}).get("success_recovered")) for case in task_degraded),
            "reward_recovered_count": sum(bool(case.get("summary", {}).get("reward_recovered")) for case in task_degraded),
        }
    return {
        "case_count": len(cases),
        "degraded_case_count": len(degraded),
        "success_recovered_count": len(success_recovered),
        "reward_recovered_count": len(reward_recovered),
        "done_index_recovered_count": len(done_recovered),
        "success_recovery_rate": rate(len(success_recovered), len(degraded)),
        "reward_recovery_rate": rate(len(reward_recovered), len(degraded)),
        "done_index_recovery_rate": rate(len(done_recovered), len(degraded)),
        "eval_demos_with_success_recovery": len({case["eval_demo_path"] for case in success_recovered}),
        "mismatches_with_success_recovery": len({case["mismatch_type"] for case in success_recovered}),
        "simple_baseline_match_count": len(baseline_matches),
        "failure_count": len(failures),
        "failure_reasons": sorted({reason for case in failures for reason in case.get("summary", {}).get("failure_reasons", [])}),
        "per_mismatch": per_mismatch,
        "per_task": per_task,
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_execspec_state3": True,
        "downloads_performed": False,
        "installs_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "lora_training_performed": False,
        "loss_computed": False,
        "supervised_calibration_metric_computed": True,
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
        "task_local_replay_gate": f"{TASK_GATE}=1",
        "task_local_replay_gate_set": os.environ.get(TASK_GATE) == "1",
    }


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    action = report.get("heldout_action_metrics", {}).get("aggregate", {})
    replay = report.get("exact_init_replay", {}).get("aggregate", {})
    replay_happened = bool(report.get("policy", {}).get("replay_or_rollout_performed"))
    replay_improved = bool(
        replay_happened
        and (
            int(replay.get("success_recovered_count") or 0) > 0
            or int(replay.get("reward_recovered_count") or 0) > 0
            or int(replay.get("done_index_recovered_count") or 0) > 0
        )
    )
    multiple_demos_or_mismatches = bool(
        int(replay.get("eval_demos_with_success_recovery") or 0) >= 2
        or int(replay.get("mismatches_with_success_recovery") or 0) >= 2
        or int(replay.get("reward_recovered_count") or 0) >= 2
    )
    action_pass = bool(
        action.get("full_repair_beats_identity_on_action_drift")
        and action.get("full_repair_beats_clipping_only_on_action_drift")
        and action.get("full_repair_beats_global_affine_on_action_drift")
    )
    no_leakage = not bool(report.get("split", {}).get("leakage_detected"))
    simple_baselines_do_not_match = int(replay.get("simple_baseline_match_count") or 0) == 0 if replay_happened else None
    decision = "continue" if action_pass and replay_improved and multiple_demos_or_mismatches and no_leakage and simple_baselines_do_not_match else "kill_or_reframe"
    if not replay_happened:
        decision = "needs_replay_validation"
    return {
        "continue_or_kill": decision,
        "next_state": "STATE 4 generalization and broader mismatch validation" if decision == "continue" else "kill/reframe",
        "best_repair_method": action.get("best_repair_method_by_mean_recovery"),
        "best_repair_method_mean_recovery_fraction": action.get("best_repair_method_mean_recovery_fraction"),
        "full_repair_beats_identity_on_action_drift": action.get("full_repair_beats_identity_on_action_drift"),
        "full_repair_beats_clipping_only_on_action_drift": action.get("full_repair_beats_clipping_only_on_action_drift"),
        "full_repair_beats_global_affine_on_action_drift": action.get("full_repair_beats_global_affine_on_action_drift"),
        "mean_action_l2": action.get("mean_action_l2"),
        "per_mismatch_full_beat_counts": action.get("per_mismatch_full_beat_counts"),
        "full_repair_mean_recovery_fraction": action.get("full_repair_mean_recovery_fraction"),
        "replay_improves_reward_success_or_done": replay_improved,
        "success_recovery_rate": replay.get("success_recovery_rate"),
        "reward_recovery_rate": replay.get("reward_recovery_rate"),
        "done_index_recovery_rate": replay.get("done_index_recovery_rate"),
        "replay_degraded_case_count": replay.get("degraded_case_count"),
        "replay_success_recovered_count": replay.get("success_recovered_count"),
        "replay_reward_recovered_count": replay.get("reward_recovered_count"),
        "simple_baseline_match_count": replay.get("simple_baseline_match_count"),
    }


def _md(value: Any) -> str:
    return repair._md_value(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    split = report.get("split", {})
    action = report.get("heldout_action_metrics", {}).get("aggregate", {})
    replay = report.get("exact_init_replay", {}).get("aggregate", {})
    lines = [
        "# ExecSpec STATE 3 Replay Validation",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success or paper-grade evidence.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- calibration demos/action samples: `{split.get('calibration_demo_count')}` / `{split.get('calibration_action_samples')}`",
        f"- held-out eval demos/action samples: `{split.get('eval_demo_count')}` / `{split.get('eval_action_samples')}`",
        f"- task count: `{split.get('task_count')}`",
        f"- eval leakage detected: `{split.get('leakage_detected')}`",
        f"- replay/rollout happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- best repair method: `{summary.get('best_repair_method')}`",
        f"- full beats identity/clipping/global: `{summary.get('full_repair_beats_identity_on_action_drift')}` / `{summary.get('full_repair_beats_clipping_only_on_action_drift')}` / `{summary.get('full_repair_beats_global_affine_on_action_drift')}`",
        f"- full repair mean recovery fraction: `{summary.get('full_repair_mean_recovery_fraction')}`",
        f"- success recovery rate: `{summary.get('success_recovery_rate')}`",
        f"- reward recovery rate: `{summary.get('reward_recovery_rate')}`",
        f"- next state: `{summary.get('next_state')}`",
        "",
        "## Split",
        "",
        f"- calibration paths: `{'; '.join(Path(path).name for path in split.get('calibration_paths', []))}`",
        f"- eval paths: `{'; '.join(Path(path).name for path in split.get('eval_paths', []))}`",
        f"- tasks: `{'; '.join(split.get('tasks', []))}`",
        f"- suite coverage: `{json.dumps(split.get('suite_coverage', {}), sort_keys=True)}`",
        "",
        "## Held-Out Action Aggregate",
        "",
        "| mismatch | wrong L2 | full L2 | full recovery | full beats id/clip/global |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mismatch in repair.MISMATCH_TYPES:
        payload = (action.get("per_mismatch") or {}).get(mismatch, {})
        wrong_l2 = (((payload.get("wrong_spec") or {}).get("action_l2_mean") or {}).get("mean"))
        methods = payload.get("methods") or {}
        full = (methods.get("full_execspec_repair") or {})
        full_l2 = ((full.get("action_l2_mean") or {}).get("mean"))
        recovery = ((full.get("recovery_fraction") or {}).get("mean"))
        def beats(method: str) -> bool | None:
            baseline = (((methods.get(method) or {}).get("action_l2_mean") or {}).get("mean"))
            if full_l2 is None or baseline is None:
                return None
            return bool(float(full_l2) + 1e-9 < float(baseline))
        lines.append(
            f"| {mismatch} | {_md(wrong_l2)} | {_md(full_l2)} | {_md(recovery)} | {_md(beats('identity_no_repair'))}/{_md(beats('clipping_only'))}/{_md(beats('global_affine_calibration'))} |"
        )
    lines.extend(["", "## Exact-Init Replay Aggregate", ""])
    if not report.get("policy", {}).get("replay_or_rollout_performed"):
        lines.append(f"Exact-init replay skipped: `{report.get('replay_skip_reason')}`")
    else:
        lines.extend(
            [
                f"- cases: `{replay.get('case_count')}`",
                f"- degraded cases: `{replay.get('degraded_case_count')}`",
                f"- success recovered: `{replay.get('success_recovered_count')}`",
                f"- reward recovered: `{replay.get('reward_recovered_count')}`",
                f"- done-index recovered: `{replay.get('done_index_recovered_count')}`",
                f"- simple baseline match count: `{replay.get('simple_baseline_match_count')}`",
                f"- failure count: `{replay.get('failure_count')}`",
                f"- failure reasons: `{', '.join(replay.get('failure_reasons') or [])}`",
                "",
                "| mismatch | cases | degraded | success recovered | reward recovered | done recovered |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mismatch in repair.MISMATCH_TYPES:
            payload = (replay.get("per_mismatch") or {}).get(mismatch, {})
            lines.append(
                f"| {mismatch} | {_md(payload.get('case_count'))} | {_md(payload.get('degraded_case_count'))} | {_md(payload.get('success_recovered_count'))} | {_md(payload.get('reward_recovered_count'))} | {_md(payload.get('done_index_recovered_count'))} |"
            )
        lines.extend(["", "## Replay Cases", ""])
        lines.append("| demo | mismatch | expert | wrong | clipping | global | diagonal | gripper | full |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for case in report.get("exact_init_replay", {}).get("cases", []):
            variants = {item.get("variant"): item for item in case.get("replay_results", [])}
            def cell(name: str) -> str:
                item = variants.get(name) or {}
                return f"{_md(item.get('reward_sum'))}/{_md(_success(item))}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        Path(case.get("eval_demo_path", "")).stem,
                        case.get("mismatch_type", ""),
                        cell("correct_7d_expert_action_replay"),
                        cell("wrong_executable_spec_replay"),
                        cell("clipping_only"),
                        cell("global_affine_calibration"),
                        cell("diagonal_affine_calibration"),
                        cell("gripper_only_calibration"),
                        cell("full_execspec_repair"),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Calibration Data-Size Sensitivity", ""])
    lines.append("| calibration demos | samples | full L2 | full recovery | success replay evaluated |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for item in report.get("calibration_sensitivity", []):
        lines.append(
            f"| {_md(item.get('calibration_demo_count'))} | {_md(item.get('calibration_action_samples'))} | {_md(item.get('full_repair_mean_action_l2'))} | {_md(item.get('full_repair_mean_recovery_fraction'))} | {_md(item.get('success_replay_evaluated'))} |"
        )
    lines.extend(["", "## Exact-Init vs Default Reset", ""])
    default_reset = report.get("exact_init_replay", {}).get("default_reset_sanity", {})
    if default_reset.get("performed"):
        lines.append(f"- default-reset sanity scope: `{default_reset.get('scope')}`")
        lines.append(f"- default-reset expert succeeded: `{default_reset.get('expert_default_reset_succeeded')}`")
        lines.append(f"- default-reset full repair succeeded: `{default_reset.get('full_default_reset_succeeded')}`")
    else:
        lines.append("- default-reset sanity: `not tested`")
    lines.append("- primary claim boundary: exact-init executable-spec repair under matched replay conditions.")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_state3_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_state3_replay_validation",
        "policy": _policy(forbidden),
        "inputs": vars(args).copy(),
        "split": {},
        "mismatch_types_tested": list(repair.MISMATCH_TYPES),
        "repair_methods_tested": list(repair.REPAIR_METHODS),
        "replay_methods_tested": _parse_names(args.replay_methods, DEFAULT_REPLAY_METHODS),
        "heldout_action_metrics": {},
        "calibration_sensitivity": [],
        "exact_init_replay": {"cases": [], "aggregate": {}, "default_reset_sanity": {"performed": False}},
        "summary": {},
        "result": {"passed": False, "blocked_reason": None},
        "elapsed_seconds": None,
    }
    if forbidden:
        report["result"]["blocked_reason"] = "forbidden gates set: " + ", ".join(forbidden)
        report["summary"] = {"continue_or_kill": "blocked", "next_state": "resolve_state3_blocker"}
        return report
    try:
        split = build_validation_split(
            manifest_path=_as_path(args.manifest),
            data_root=_as_path(args.data_root),
            max_calibration_demos=args.max_calibration_demos,
            max_eval_demos=args.max_eval_demos,
        )
        calibration_actions, calibration_meta = repair._concat_actions(split["calibration_paths"], args.max_actions_per_demo)
        eval_items = _read_eval_items(split["eval_paths"], args.max_actions_per_demo)
        eval_action_samples = int(sum(item["actions"].shape[0] for item in eval_items))
        tasks = sorted({item["task_id"] for item in eval_items} | {Path(meta["path"]).stem.replace("_demo", "") for meta in calibration_meta})
        suite_coverage: dict[str, int] = {}
        for path in split["calibration_paths"] + split["eval_paths"]:
            suite_coverage[Path(path).parent.name] = suite_coverage.get(Path(path).parent.name, 0) + 1
        report["split"] = {
            "calibration_demo_count": len(split["calibration_paths"]),
            "calibration_action_samples": int(calibration_actions.shape[0]),
            "eval_demo_count": len(split["eval_paths"]),
            "eval_action_samples": eval_action_samples,
            "task_count": len(tasks),
            "tasks": tasks,
            "suite_coverage": suite_coverage,
            "calibration_paths": [str(path) for path in split["calibration_paths"]],
            "eval_paths": [str(path) for path in split["eval_paths"]],
            "leakage_detected": False,
            "eval_demo_contributed_to_calibration": False,
            "limitation": "bounded local LIBERO exact-init validation; not default-reset deployment evidence",
        }
        fitted_params = _fit_params_by_mismatch(calibration_actions)
        action_cases = evaluate_action_cases(eval_items=eval_items, fitted_params=fitted_params)
        report["heldout_action_metrics"] = {
            "case_count": len(action_cases),
            "cases": action_cases,
            "aggregate": aggregate_action_cases(action_cases),
        }
        sizes = [int(item.strip()) for item in str(args.calibration_sensitivity_sizes).split(",") if item.strip()]
        report["calibration_sensitivity"] = calibration_sensitivity(
            split=split,
            eval_items=eval_items,
            max_actions_per_demo=args.max_actions_per_demo,
            sizes=sizes,
        )
        if os.environ.get(TASK_GATE) == "1":
            replay_mismatches = _parse_names(args.replay_mismatches, repair.MISMATCH_TYPES)
            replay_methods = _parse_names(args.replay_methods, DEFAULT_REPLAY_METHODS)
            replay = run_exact_init_replay(
                split=split,
                fitted_params=fitted_params,
                replay_mismatches=replay_mismatches,
                replay_methods=replay_methods,
                max_replay_eval_demos=args.max_replay_eval_demos,
                libero_root=_as_path(args.libero_root),
                robosuite_root=_as_path(args.robosuite_root),
                max_steps_cap=args.max_steps_cap,
                post_signal_margin=args.post_signal_margin,
                camera_size=args.camera_size,
                include_default_reset_sanity=args.include_default_reset_sanity,
            )
            report["exact_init_replay"] = replay
            report["policy"]["simulator_environment_created"] = True
            report["policy"]["replay_or_rollout_performed"] = int(replay.get("total_steps_performed") or 0) > 0
            report["policy"]["diagnostic_rollouts_performed"] = report["policy"]["replay_or_rollout_performed"]
        else:
            report["replay_skip_reason"] = f"{TASK_GATE}=1 not set; held-out action validation only"
        report["summary"] = summarize_report(report)
        report["result"]["passed"] = True
    except Exception as exc:
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "next_state": "resolve_state3_blocker"}
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--data-root", default="C:/assets/data/libero")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-calibration-demos", type=int, default=5)
    parser.add_argument("--max-eval-demos", type=int, default=3)
    parser.add_argument("--max-replay-eval-demos", type=int, default=3)
    parser.add_argument("--max-actions-per-demo", type=int, default=300)
    parser.add_argument("--max-steps-cap", type=int, default=300)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--replay-mismatches", default=",".join(repair.MISMATCH_TYPES))
    parser.add_argument("--replay-methods", default=",".join(DEFAULT_REPLAY_METHODS))
    parser.add_argument("--calibration-sensitivity-sizes", default="1,3,5")
    parser.add_argument("--include-default-reset-sanity", action="store_true")
    parser.add_argument("--report-json", default="reports/execspec_state3_replay_validation.json")
    parser.add_argument("--report-md", default="reports/execspec_state3_replay_validation.md")
    args = parser.parse_args(argv)
    report = build_state3_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    console_report = {
        "result": report["result"],
        "summary": report["summary"],
        "split": report["split"],
        "exact_init_replay_aggregate": report.get("exact_init_replay", {}).get("aggregate"),
        "report_json": str(report_json),
    }
    print(json.dumps(console_report, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
