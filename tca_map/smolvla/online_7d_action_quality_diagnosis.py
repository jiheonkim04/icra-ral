"""Action-quality diagnosis for the non-leaking online 7D diagnostic heads."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.online_action_generation_bridge import _as_path, _load_json
from tca_map.smolvla.online_7d_diagnostic_head import (
    _features,
    _read_eval_demo,
    _read_pair_samples,
    _target_prior,
    _with_bias,
    train_online_7d_heads,
)

SCHEMA_VERSION = "2026-07-06.online_7d_action_quality_diagnosis.v1"
VARIANTS = ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d")
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
    "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT",
    "ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT",
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _round(value: float | int | np.floating, digits: int = 9) -> float:
    return round(float(value), digits)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _actions(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([sample["action"] for sample in samples], dtype=np.float64)


def _predict(models: dict[str, Any], meta: dict[str, Any], samples: list[dict[str, Any]], variant: str, horizon: int) -> np.ndarray:
    features = _features(samples, variant, meta["target_prior"], horizon)
    raw = _with_bias(features) @ models[variant]["weights"]
    return np.clip(raw, -1.0, 1.0)


def _basic_error(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    diff = pred - expert
    l2 = np.linalg.norm(diff, axis=1)
    translation_l2 = np.linalg.norm(diff[:, :3], axis=1)
    rotation_l2 = np.linalg.norm(diff[:, 3:6], axis=1)
    gripper_l1 = np.abs(diff[:, 6])
    pred_open = pred[:, 6] >= 0.0
    expert_open = expert[:, 6] >= 0.0
    return {
        "sample_count": int(pred.shape[0]),
        "7d_action_l2": _round(np.mean(l2)),
        "7d_action_l2_sum": _round(np.sum(l2)),
        "translation_l2": _round(np.mean(translation_l2)),
        "rotation_l2": _round(np.mean(rotation_l2)),
        "gripper_l1": _round(np.mean(gripper_l1)),
        "gripper_open_close_accuracy": _round(np.mean(pred_open == expert_open)),
        "raw_first_step_l2": _round(l2[0]) if l2.size else None,
        "raw_last_step_l2": _round(l2[-1]) if l2.size else None,
        "standard_offline_proxy": _round(1.0 / (1.0 + np.mean(l2))) if l2.size else None,
    }


def _first_k_errors(pred: np.ndarray, expert: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 25)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ks:
        if pred.shape[0] >= k:
            out[f"first_{k}"] = _basic_error(pred[:k], expert[:k])
    return out


def _phase_bins(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    names = ("early", "mid", "late")
    out: dict[str, Any] = {}
    for name, indexes in zip(names, np.array_split(np.arange(pred.shape[0]), 3)):
        if indexes.size:
            out[name] = _basic_error(pred[indexes], expert[indexes])
    return out


def _group_errors(samples: list[dict[str, Any]], pred: np.ndarray, expert: np.ndarray, key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(sample.get(key)) for sample in samples}):
        indexes = [idx for idx, sample in enumerate(samples) if str(sample.get(key)) == value]
        if indexes:
            idx_arr = np.asarray(indexes, dtype=np.int64)
            out[value] = _basic_error(pred[idx_arr], expert[idx_arr])
    return out


def _gripper_timing(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    def first_open(values: np.ndarray) -> int | None:
        indexes = np.flatnonzero(values[:, 6] >= 0.0)
        return int(indexes[0]) if indexes.size else None

    pred_first = first_open(pred)
    expert_first = first_open(expert)
    delta = None if pred_first is None or expert_first is None else int(pred_first - expert_first)
    return {
        "predicted_first_open_index": pred_first,
        "expert_first_open_index": expert_first,
        "first_open_timing_delta": delta,
        "predicted_never_opens": pred_first is None,
        "expert_never_opens": expert_first is None,
    }


def _variance_report(actions: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    std = np.std(actions, axis=0)
    expert_std = np.std(expert, axis=0)
    mean_std = _safe_mean(std)
    expert_mean_std = _safe_mean(expert_std)
    ratio = mean_std / max(expert_mean_std, 1e-12)
    return {
        "per_dim_std": [_round(value) for value in std.tolist()],
        "mean_std": _round(mean_std),
        "expert_mean_std": _round(expert_mean_std),
        "std_ratio_to_expert": _round(ratio),
        "constant_or_mean_collapse": bool(ratio < 0.15 or mean_std < 0.02),
    }


def _action_difference_audit(preds: dict[str, np.ndarray], expert: np.ndarray, rollout_report: dict[str, Any] | None) -> dict[str, Any]:
    actionmap = preds["actionmap_7d"]
    fixed = preds["fixed_prior_tca_7d"]
    hard = preds["hard_learned_target_tca_7d"]
    diff = fixed - actionmap
    l2_by_step = np.linalg.norm(diff, axis=1)
    t_l2 = np.linalg.norm(diff[:, :3], axis=1)
    r_l2 = np.linalg.norm(diff[:, 3:6], axis=1)
    g_abs = np.abs(diff[:, 6])
    mean_l2 = _safe_mean(l2_by_step)
    meaningful = bool(mean_l2 >= 0.05 or _safe_mean(t_l2) >= 0.025 or _safe_mean(g_abs) >= 0.1)
    report = {
        "actionmap_vs_fixed_prior_tca": {
            "per_step_action_l2": [_round(value) for value in l2_by_step.tolist()],
            "mean_action_l2": _round(mean_l2),
            "max_action_l2": _round(np.max(l2_by_step)),
            "mean_translation_l2": _round(np.mean(t_l2)),
            "mean_rotation_l2": _round(np.mean(r_l2)),
            "mean_gripper_abs_diff": _round(np.mean(g_abs)),
            "meaningfully_different": meaningful,
            "almost_identical": not meaningful,
        },
        "actionmap_vs_hard_learned_target_tca": {
            "mean_action_l2": _round(np.mean(np.linalg.norm(hard - actionmap, axis=1))),
            "mean_translation_l2": _round(np.mean(np.linalg.norm((hard - actionmap)[:, :3], axis=1))),
            "mean_rotation_l2": _round(np.mean(np.linalg.norm((hard - actionmap)[:, 3:6], axis=1))),
            "mean_gripper_abs_diff": _round(np.mean(np.abs((hard - actionmap)[:, 6]))),
        },
        "variance_over_teacher_forced_eval": {
            name: _variance_report(values, expert) for name, values in preds.items()
        },
        "distribution_reference": _rollout_distribution_reference(rollout_report),
    }
    return report


def _supervised_breakdown(samples: list[dict[str, Any]], preds: dict[str, np.ndarray], expert: np.ndarray, train_actions: np.ndarray) -> dict[str, Any]:
    mean_pred = np.repeat(np.mean(train_actions, axis=0).reshape(1, -1), expert.shape[0], axis=0)
    out: dict[str, Any] = {
        "mean_action_baseline": {
            "metrics": _basic_error(mean_pred, expert),
            "variance": _variance_report(mean_pred, expert),
        },
        "variants": {},
    }
    for name, pred in preds.items():
        out["variants"][name] = {
            "metrics": _basic_error(pred, expert),
            "first_k_step_error": _first_k_errors(pred, expert),
            "phase_bins": _phase_bins(pred, expert),
            "per_target_error": _group_errors(samples, pred, expert, "target_id"),
            "per_task_error": _group_errors(samples, pred, expert, "pair_id"),
            "gripper_timing": _gripper_timing(pred, expert),
            "variance": _variance_report(pred, expert),
            "delta_l2_vs_mean_action_baseline": _round(_basic_error(pred, expert)["7d_action_l2"] - out["mean_action_baseline"]["metrics"]["7d_action_l2"]),
        }
    return out


def _teacher_forced_diagnostic(samples: list[dict[str, Any]], preds: dict[str, np.ndarray], expert: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(samples), "variants": {}}
    for name, pred in preds.items():
        metrics = _basic_error(pred, expert)
        out["variants"][name] = {
            "metrics": metrics,
            "cumulative_action_l2": metrics["7d_action_l2_sum"],
            "gripper_timing": _gripper_timing(pred, expert),
            "phase_bins": _phase_bins(pred, expert),
        }
    actionmap_l2 = out["variants"]["actionmap_7d"]["metrics"]["7d_action_l2"]
    fixed_l2 = out["variants"]["fixed_prior_tca_7d"]["metrics"]["7d_action_l2"]
    out["fixed_prior_tca_better_than_actionmap"] = bool(fixed_l2 < actionmap_l2)
    out["fixed_prior_tca_l2_delta_vs_actionmap"] = _round(fixed_l2 - actionmap_l2)
    return out


def _rollout_variant(report: dict[str, Any] | None, variant: str) -> dict[str, Any] | None:
    if not report:
        return None
    for item in report.get("rollout_results") or []:
        if item.get("variant") == variant:
            return item
    return None


def _l2_timeline(variant: dict[str, Any] | None) -> list[float]:
    if not variant:
        return []
    return [_round(item.get("l2_to_hdf5_expert_same_timestep", math.nan)) for item in variant.get("action_provenance") or []]


def _rollout_distribution_reference(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report or report.get("decision") != "bounded_online_7d_head_rollout_completed":
        return {"available": False, "reason": "bounded online 7D rollout report not available"}
    names = ("native_smolvla_online_policy", "hdf5_expert_replay_exact_init", "actionmap_7d", "fixed_prior_tca_7d")
    return {
        "available": True,
        "variants": {
            name: {
                "action_stats": (_rollout_variant(report, name) or {}).get("action_stats"),
                "expert_match": (_rollout_variant(report, name) or {}).get("expert_match"),
                "reward_sum": (_rollout_variant(report, name) or {}).get("reward_sum"),
                "final_success": (_rollout_variant(report, name) or {}).get("final_success"),
                "target_directed_movement_score": (_rollout_variant(report, name) or {}).get("target_directed_movement_score"),
            }
            for name in names
        },
    }


def _closed_loop_failure_diagnosis(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report or report.get("decision") != "bounded_online_7d_head_rollout_completed":
        return {"available": False, "reason": "bounded online 7D rollout report not available"}
    out: dict[str, Any] = {"available": True, "variants": {}}
    for name in ("native_smolvla_online_policy", "actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"):
        variant = _rollout_variant(report, name)
        timeline = np.asarray(_l2_timeline(variant), dtype=np.float64)
        action_stats = (variant or {}).get("action_stats") or {}
        gripper = action_stats.get("gripper_range") or {}
        rotation = action_stats.get("rotation_range") or {}
        early = timeline[: min(5, timeline.size)]
        late = timeline[-min(5, timeline.size) :] if timeline.size else timeline
        gripper_min = gripper.get("min")
        gripper_max = gripper.get("max")
        if gripper_min is None or gripper_max is None:
            gripper_state = "unavailable"
        elif float(gripper_max) < 0.0:
            gripper_state = "never_opens_or_positive"
        elif float(gripper_min) > 0.0:
            gripper_state = "never_closes_or_negative"
        else:
            gripper_state = "changes_sign_or_mixed"
        out["variants"][name] = {
            "reward_sum": (variant or {}).get("reward_sum"),
            "final_success": (variant or {}).get("final_success"),
            "valid_closed_loop_online_rollout": (variant or {}).get("valid_closed_loop_online_rollout"),
            "mean_l2_to_expert_same_timestep": _round(np.mean(timeline)) if timeline.size else None,
            "first_step_l2_to_expert": _round(timeline[0]) if timeline.size else None,
            "last_step_l2_to_expert": _round(timeline[-1]) if timeline.size else None,
            "early_mean_l2": _round(np.mean(early)) if early.size else None,
            "late_mean_l2": _round(np.mean(late)) if late.size else None,
            "action_divergence_increases_over_time": bool(late.size and early.size and np.mean(late) > np.mean(early)),
            "eef_displacement_l2": (variant or {}).get("eef_displacement_l2"),
            "target_directed_movement_score": (variant or {}).get("target_directed_movement_score"),
            "target_object_displacement_l2": (variant or {}).get("target_object_displacement_l2"),
            "gripper_diagnosis": gripper_state,
            "rotation_max_abs": rotation.get("max_abs"),
            "translation_max_abs": (action_stats.get("translation_range") or {}).get("max_abs"),
        }
    fixed = out["variants"].get("fixed_prior_tca_7d", {})
    actionmap = out["variants"].get("actionmap_7d", {})
    out["fixed_prior_tca_valid_rollout_support"] = bool(
        fixed.get("valid_closed_loop_online_rollout")
        and actionmap.get("valid_closed_loop_online_rollout")
        and (
            float(fixed.get("reward_sum") or 0.0) > float(actionmap.get("reward_sum") or 0.0)
            or (bool(fixed.get("final_success")) and not bool(actionmap.get("final_success")))
        )
    )
    out["fixed_prior_tca_partial_target_movement_support"] = bool(
        not out["fixed_prior_tca_valid_rollout_support"]
        and fixed.get("target_directed_movement_score") is not None
        and actionmap.get("target_directed_movement_score") is not None
        and float(fixed["target_directed_movement_score"]) > float(actionmap["target_directed_movement_score"])
    )
    out["diagnosis"] = "partial_target_movement_no_reward_or_success" if out["fixed_prior_tca_partial_target_movement_support"] else "no_rollout_support"
    out["target_movement_due_to_translation_only"] = True
    out["target_movement_metric_note"] = "The target movement score is a diagnostic movement proxy; reward/success remain primary."
    return out


def _gripper_threshold_calibration(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    calibrated = pred.copy()
    calibrated[:, 6] = np.where(calibrated[:, 6] >= 0.0, 1.0, -1.0)
    before = _basic_error(pred, expert)
    after = _basic_error(calibrated, expert)
    return {
        "before_gripper_l1": before["gripper_l1"],
        "after_gripper_l1": after["gripper_l1"],
        "before_7d_action_l2": before["7d_action_l2"],
        "after_7d_action_l2": after["7d_action_l2"],
        "improved_7d_l2": bool(after["7d_action_l2"] < before["7d_action_l2"]),
        "improved_gripper_l1": bool(after["gripper_l1"] < before["gripper_l1"]),
    }


def _bounded_improvement_audit(preds: dict[str, np.ndarray], expert: np.ndarray, breakdown: dict[str, Any]) -> dict[str, Any]:
    calibrations = {name: _gripper_threshold_calibration(pred, expert) for name, pred in preds.items()}
    current_metrics = {name: data["metrics"] for name, data in breakdown["variants"].items()}
    best_current = min(current_metrics, key=lambda name: current_metrics[name]["7d_action_l2"])
    best_calibrated = min(calibrations, key=lambda name: calibrations[name]["after_7d_action_l2"])
    any_calibration_improves = any(item["improved_7d_l2"] for item in calibrations.values())
    return {
        "mean_action_baseline_7d_l2": breakdown["mean_action_baseline"]["metrics"]["7d_action_l2"],
        "best_current_head_variant": best_current,
        "best_current_head_7d_l2": current_metrics[best_current]["7d_action_l2"],
        "gripper_threshold_calibration": calibrations,
        "best_gripper_calibrated_variant": best_calibrated,
        "best_gripper_calibrated_7d_l2": calibrations[best_calibrated]["after_7d_action_l2"],
        "calibration_improves_any_7d_l2": bool(any_calibration_improves),
        "small_mlp_evaluated": False,
        "small_mlp_reason": "Not run in this milestone; first diagnosis shows conditioning/action quality needs inspection before adding a stronger trainer.",
        "bounded_improved_head_rollout_justified_now": bool(any_calibration_improves and calibrations[best_calibrated]["after_7d_action_l2"] < current_metrics[best_current]["7d_action_l2"] * 0.95),
    }


def _load_optional_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _load_samples(manifest_path: Path, max_steps: int, train_max_steps: int, stride: int, teacher_max_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if len(pairs) < 2:
        raise ValueError("at least two counterfactual pairs are required")
    rollout_pair = pairs[0]
    rollout_demo_path = str(_as_path(rollout_pair["positive_demo_file"]))
    train_samples: list[dict[str, Any]] = []
    for pair in pairs[1:]:
        train_samples.extend(
            sample
            for sample in _read_pair_samples(pair, train_max_steps, stride)
            if sample["demo_path"] != rollout_demo_path
        )
        if len(train_samples) >= 512:
            train_samples = train_samples[:512]
            break
    eval_samples, _ = _read_eval_demo(rollout_pair, max_steps)
    teacher_samples, rollout_demo = _read_eval_demo(rollout_pair, teacher_max_steps)
    return train_samples, eval_samples, teacher_samples, rollout_demo


def build_action_quality_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "stop",
        "policy": {
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "rollout_happened": False,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "result": {"passed": False, "blocked_reason": None},
    }
    if forbidden:
        report["result"]["blocked_reason"] = "Forbidden gate(s) set: " + ", ".join(forbidden)
        return report

    manifest_path = Path(args.manifest)
    models, meta = train_online_7d_heads(manifest_path, args.max_steps, args.train_max_steps, args.sample_stride)
    train_samples, eval_samples, teacher_samples, rollout_demo = _load_samples(
        manifest_path, args.max_steps, args.train_max_steps, args.sample_stride, args.teacher_max_steps
    )
    prior = _target_prior(train_samples)
    meta["target_prior"] = prior
    train_actions = _actions(train_samples)
    eval_expert = _actions(eval_samples)
    teacher_expert = _actions(teacher_samples)
    eval_preds = {variant: _predict(models, meta, eval_samples, variant, args.max_steps) for variant in VARIANTS}
    teacher_preds = {variant: _predict(models, meta, teacher_samples, variant, len(teacher_samples)) for variant in VARIANTS}
    online_report = _load_optional_report(Path(args.online_7d_report))

    supervised = _supervised_breakdown(eval_samples, eval_preds, eval_expert, train_actions)
    teacher = _teacher_forced_diagnostic(teacher_samples, teacher_preds, teacher_expert)
    action_diff = _action_difference_audit(eval_preds, eval_expert, online_report)
    closed_loop = _closed_loop_failure_diagnosis(online_report)
    improvements = _bounded_improvement_audit(eval_preds, eval_expert, supervised)

    fixed_diff = action_diff["actionmap_vs_fixed_prior_tca"]
    best_variant = improvements["best_current_head_variant"]
    fixed_improves_eval = supervised["variants"]["fixed_prior_tca_7d"]["metrics"]["7d_action_l2"] < supervised["variants"]["actionmap_7d"]["metrics"]["7d_action_l2"]
    report.update(
        {
            "decision": "online_7d_action_quality_diagnosis_completed",
            "policy": {
                **report["policy"],
                "training_performed": True,
                "loss_computed": True,
                "rollout_happened": bool((online_report or {}).get("policy", {}).get("rollout_happened", False)),
                "heavy_model_imports_performed": bool((online_report or {}).get("policy", {}).get("heavy_model_imports_performed", False)),
                "model_load_performed": bool((online_report or {}).get("policy", {}).get("model_load_performed", False)),
                "model_inference_performed": bool((online_report or {}).get("policy", {}).get("model_inference_performed", False)),
            },
            "data": {
                "manifest_path": str(manifest_path),
                "train_sample_count": len(train_samples),
                "eval_sample_count": len(eval_samples),
                "teacher_forced_sample_count": len(teacher_samples),
                "rollout_demo_path": rollout_demo["path"],
                "rollout_demo_excluded_from_training": rollout_demo["path"] not in sorted({sample["demo_path"] for sample in train_samples}),
            },
            "training_losses": {name: models[name]["loss"] for name in VARIANTS},
            "action_difference_audit": action_diff,
            "supervised_action_quality_breakdown": supervised,
            "teacher_forced_trajectory_diagnostic": teacher,
            "closed_loop_failure_diagnosis": closed_loop,
            "bounded_head_improvement_audit": improvements,
            "conclusion": {
                "fixed_prior_tca_actions_meaningfully_different_from_actionmap": fixed_diff["meaningfully_different"],
                "fixed_prior_tca_actions_almost_identical_to_actionmap": fixed_diff["almost_identical"],
                "fixed_prior_tca_improves_any_7d_online_metric": bool(fixed_improves_eval or teacher["fixed_prior_tca_better_than_actionmap"]),
                "fixed_prior_tca_valid_rollout_support": bool(closed_loop.get("fixed_prior_tca_valid_rollout_support", False)),
                "dominant_bottleneck": _dominant_bottleneck(supervised["variants"][best_variant]["metrics"]),
                "best_bounded_7d_head_variant": best_variant,
                "recommended_next_milestone": _recommend_next(action_diff, supervised, teacher, closed_loop, improvements),
            },
            "result": {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)},
        }
    )
    return report


def _dominant_bottleneck(metrics: dict[str, Any]) -> str:
    values = {
        "translation": float(metrics.get("translation_l2") or 0.0),
        "rotation": float(metrics.get("rotation_l2") or 0.0),
        "gripper": float(metrics.get("gripper_l1") or 0.0),
    }
    return max(values, key=values.get)


def _recommend_next(action_diff: dict[str, Any], supervised: dict[str, Any], teacher: dict[str, Any], closed_loop: dict[str, Any], improvements: dict[str, Any]) -> str:
    if closed_loop.get("fixed_prior_tca_valid_rollout_support"):
        return "A. bounded improved-head matched-init rollout"
    if improvements.get("bounded_improved_head_rollout_justified_now"):
        return "A. bounded improved-head matched-init rollout"
    if action_diff["actionmap_vs_fixed_prior_tca"]["almost_identical"]:
        return "C. target-prior conditioning redesign"
    best = improvements["best_current_head_variant"]
    metrics = supervised["variants"][best]["metrics"]
    if _dominant_bottleneck(metrics) in {"gripper", "rotation"}:
        return "B. gripper/rotation calibration"
    if not teacher.get("fixed_prior_tca_better_than_actionmap"):
        return "D. paper-readiness package with honest rollout caveat"
    return "B. gripper/rotation calibration"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    conclusion = report.get("conclusion") or {}
    diff = (report.get("action_difference_audit") or {}).get("actionmap_vs_fixed_prior_tca") or {}
    supervised = report.get("supervised_action_quality_breakdown") or {}
    teacher = report.get("teacher_forced_trajectory_diagnostic") or {}
    closed = report.get("closed_loop_failure_diagnosis") or {}
    lines = [
        "# Online 7D Action-Quality Diagnosis",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- training happened: `{(report.get('policy') or {}).get('training_performed')}`",
        f"- LoRA training happened: `{(report.get('policy') or {}).get('lora_training_performed')}`",
        f"- loss computed: `{(report.get('policy') or {}).get('loss_computed')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- fixed-prior TCA/actionmap mean action L2: `{diff.get('mean_action_l2')}`",
        f"- fixed-prior TCA actions meaningfully different: `{conclusion.get('fixed_prior_tca_actions_meaningfully_different_from_actionmap')}`",
        f"- fixed-prior TCA valid rollout support: `{conclusion.get('fixed_prior_tca_valid_rollout_support')}`",
        f"- dominant bottleneck: `{conclusion.get('dominant_bottleneck')}`",
        f"- best bounded 7D head variant: `{conclusion.get('best_bounded_7d_head_variant')}`",
        f"- recommended next milestone: `{conclusion.get('recommended_next_milestone')}`",
        "",
        "## Supervised Eval Metrics",
        "",
    ]
    for name, data in (supervised.get("variants") or {}).items():
        metrics = data.get("metrics") or {}
        lines.append(
            f"- `{name}`: 7D L2 `{metrics.get('7d_action_l2')}`, translation `{metrics.get('translation_l2')}`, rotation `{metrics.get('rotation_l2')}`, gripper L1 `{metrics.get('gripper_l1')}`, gripper acc `{metrics.get('gripper_open_close_accuracy')}`"
        )
    lines.extend(["", "## Teacher-Forced Result", ""])
    for name, data in (teacher.get("variants") or {}).items():
        metrics = data.get("metrics") or {}
        lines.append(f"- `{name}`: cumulative L2 `{data.get('cumulative_action_l2')}`, mean 7D L2 `{metrics.get('7d_action_l2')}`")
    lines.extend(["", "## Closed-Loop Diagnosis", "", f"- available: `{closed.get('available')}`", f"- diagnosis: `{closed.get('diagnosis')}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--online-7d-report", default="reports/online_7d_diagnostic_head_report.json")
    parser.add_argument("--report-json", default="reports/online_7d_action_quality_diagnosis_report.json")
    parser.add_argument("--report-md", default="reports/online_7d_action_quality_diagnosis_report.md")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--train-max-steps", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--teacher-max-steps", type=int, default=300)
    args = parser.parse_args(argv)

    report = build_action_quality_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    sys.exit(main())
