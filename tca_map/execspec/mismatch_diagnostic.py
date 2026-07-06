"""Offline executable-spec mismatch diagnostic for LIBERO HDF5 actions.

This diagnostic reads local expert actions and applies plausible action-space
metadata/controller perturbations. It is report-only unless a later runner adds
simulator replay: no model loading, training, GPU work, downloads, OpenVLA-OFT,
or paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "2026-07-07.execspec_mismatch_diagnostic.v1"
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
PERTURBATION_ORDER = (
    "correct_7d_expert_action_replay",
    "global_action_scale_mismatch",
    "per_dimension_scale_mismatch",
    "gripper_sign_flip",
    "gripper_threshold_0_1_mismatch",
    "translation_scale_mismatch",
    "rotation_scale_mismatch",
    "clipping_only",
    "sixd_to_sevend_zero_gripper_bridge",
)


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _round(value: float | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _range(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "max_abs": None}
    return {
        "min": _round(float(arr.min()), 6),
        "max": _round(float(arr.max()), 6),
        "mean": _round(float(arr.mean()), 6),
        "std": _round(float(arr.std()), 6),
        "max_abs": _round(float(np.max(np.abs(arr))), 6),
    }


def _component_slices() -> dict[str, slice]:
    return {
        "translation": slice(0, 3),
        "rotation": slice(3, 6),
        "gripper": slice(6, 7),
    }


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _discover_demo_path(manifest_path: Path) -> Path:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError(f"manifest has no counterfactual pairs: {manifest_path}")
    candidate = pairs[0].get("positive_demo_file")
    if not candidate:
        raise ValueError(f"first manifest pair has no positive_demo_file: {manifest_path}")
    path = _as_path(candidate)
    if path is None:
        raise ValueError("could not resolve positive demo path")
    return path


def read_hdf5_demo(demo_path: Path, max_steps: int) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(demo_path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{demo_path} has no data group")
        demo_name = sorted(data_group.keys())[0]
        demo = data_group[demo_name]
        if "actions" not in demo:
            raise ValueError(f"{demo_path} demo {demo_name} has no actions dataset")
        actions = np.asarray(demo["actions"], dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"{demo_path} actions must be rank 2, got {list(actions.shape)}")
        rewards = np.asarray(demo["rewards"], dtype=np.float64).reshape(-1) if "rewards" in demo else np.zeros((actions.shape[0],))
        dones = np.asarray(demo["dones"], dtype=np.float64).reshape(-1) if "dones" in demo else np.zeros((actions.shape[0],))
        init_state_available = "init_state" in demo.attrs
        model_file_available = "model_file" in demo.attrs
        num_samples = int(demo.attrs.get("num_samples", actions.shape[0]))
    if actions.shape[1] != 7:
        raise ValueError(f"ExecSpec State 1 expects 7D LIBERO actions, got {list(actions.shape)}")
    selected = actions[: max(1, min(int(max_steps), int(actions.shape[0])))]
    return {
        "path": str(demo_path),
        "demo_name": str(demo_name),
        "actions": selected,
        "full_action_steps": int(actions.shape[0]),
        "selected_steps": int(selected.shape[0]),
        "action_dim": int(selected.shape[1]),
        "num_samples_attr": num_samples,
        "first_positive_reward_index": _first_index(rewards, 0.0),
        "first_done_index": _first_index(dones, 0.5),
        "init_state_available": bool(init_state_available),
        "model_file_available": bool(model_file_available),
    }


def _perturbations(actions: np.ndarray) -> dict[str, dict[str, Any]]:
    per_dim_scale = np.asarray([0.55, 1.6, 0.7, 1.45, 0.5, 1.25, 1.0], dtype=np.float64)
    raw: dict[str, np.ndarray] = {}
    raw["correct_7d_expert_action_replay"] = actions.copy()
    raw["global_action_scale_mismatch"] = actions * 1.5
    raw["per_dimension_scale_mismatch"] = actions * per_dim_scale.reshape(1, -1)
    raw["gripper_sign_flip"] = actions.copy()
    raw["gripper_sign_flip"][:, 6] *= -1.0
    raw["gripper_threshold_0_1_mismatch"] = actions.copy()
    raw["gripper_threshold_0_1_mismatch"][:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, 0.0)
    raw["translation_scale_mismatch"] = actions.copy()
    raw["translation_scale_mismatch"][:, :3] *= 2.0
    raw["rotation_scale_mismatch"] = actions.copy()
    raw["rotation_scale_mismatch"][:, 3:6] *= 0.25
    raw["clipping_only"] = actions.copy()
    raw["sixd_to_sevend_zero_gripper_bridge"] = actions.copy()
    raw["sixd_to_sevend_zero_gripper_bridge"][:, 6] = 0.0
    metadata = {
        "correct_7d_expert_action_replay": {
            "policy_action_dim_before_bridge": 7,
            "description": "Identity 7D HDF5 expert action stream.",
        },
        "global_action_scale_mismatch": {
            "policy_action_dim_before_bridge": 7,
            "description": "All action dimensions are multiplied by an incorrect global unnormalization scale.",
        },
        "per_dimension_scale_mismatch": {
            "policy_action_dim_before_bridge": 7,
            "description": "Translation, rotation, and gripper dimensions use inconsistent metadata scales.",
        },
        "gripper_sign_flip": {
            "policy_action_dim_before_bridge": 7,
            "description": "Closed/open gripper convention is inverted.",
        },
        "gripper_threshold_0_1_mismatch": {
            "policy_action_dim_before_bridge": 7,
            "description": "A 0/1 binary gripper convention is sent to a -1/1 controller interface.",
        },
        "translation_scale_mismatch": {
            "policy_action_dim_before_bridge": 7,
            "description": "Only Cartesian translation dimensions are over-scaled.",
        },
        "rotation_scale_mismatch": {
            "policy_action_dim_before_bridge": 7,
            "description": "Only rotation dimensions are under-scaled.",
        },
        "clipping_only": {
            "policy_action_dim_before_bridge": 7,
            "description": "No repair except environment-range clipping.",
        },
        "sixd_to_sevend_zero_gripper_bridge": {
            "policy_action_dim_before_bridge": 6,
            "description": "A 6D pose action is bridged to 7D with a zero gripper command.",
        },
    }
    return {name: {"raw_actions": raw[name], **metadata[name]} for name in PERTURBATION_ORDER}


def _row_l2(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def action_metrics(reference: np.ndarray, raw_actions: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw_actions, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    steps = min(raw.shape[0], ref.shape[0])
    width = min(raw.shape[1], ref.shape[1]) if raw.ndim == 2 and ref.ndim == 2 else 0
    raw = raw[:steps, :width]
    ref = ref[:steps, :width]
    effective = np.clip(raw, -1.0, 1.0)
    diff = effective - ref
    raw_out_of_range = np.abs(raw) > 1.0
    step_clip = np.any(raw_out_of_range, axis=1) if raw_out_of_range.size else np.zeros((steps,), dtype=bool)
    valid_raw = np.all(np.isfinite(raw), axis=1) & np.all(np.abs(raw) <= 1.0, axis=1) if raw.size else np.zeros((steps,), dtype=bool)
    metrics: dict[str, Any] = {
        "steps": int(steps),
        "action_dim": int(width),
        "raw_action_range": _range(raw),
        "effective_action_range": _range(effective),
        "finite_raw": bool(np.all(np.isfinite(raw))),
        "controller_valid_action_rate": _round(float(np.mean(valid_raw)) if valid_raw.size else 0.0, 6),
        "post_clip_controller_valid_action_rate": 1.0 if steps > 0 and width == 7 and np.all(np.isfinite(effective)) else 0.0,
        "clip_rate_element": _round(float(np.mean(raw_out_of_range)) if raw_out_of_range.size else 0.0, 6),
        "clip_rate_step": _round(float(np.mean(step_clip)) if step_clip.size else 0.0, 6),
        "action_l2_mean": _round(float(np.mean(_row_l2(diff))) if diff.size else 0.0),
        "action_l2_max": _round(float(np.max(_row_l2(diff))) if diff.size else 0.0),
        "action_l1_mean": _round(float(np.mean(np.abs(diff))) if diff.size else 0.0),
    }
    for label, slc in _component_slices().items():
        if width >= slc.stop:
            component_diff = diff[:, slc]
            metrics[f"{label}_drift_mean"] = _round(float(np.mean(_row_l2(component_diff))) if component_diff.size else 0.0)
            metrics[f"{label}_drift_max"] = _round(float(np.max(_row_l2(component_diff))) if component_diff.size else 0.0)
            metrics[f"{label}_range"] = _range(effective[:, slc])
    if width >= 7 and steps > 0:
        ref_open = ref[:, 6] >= 0.0
        eff_open = effective[:, 6] >= 0.0
        metrics["gripper_mismatch_rate"] = _round(float(np.mean(ref_open != eff_open)), 6)
        metrics["gripper_mean_abs_error"] = _round(float(np.mean(np.abs(effective[:, 6] - ref[:, 6]))), 9)
    if width >= 6 and steps > 0:
        trans_integral = np.cumsum(diff[:, :3], axis=0)
        rot_integral = np.cumsum(diff[:, 3:6], axis=0)
        metrics["trajectory_proxy_translation_integral_final_l2"] = _round(float(np.linalg.norm(trans_integral[-1])))
        metrics["trajectory_proxy_rotation_integral_final_l2"] = _round(float(np.linalg.norm(rot_integral[-1])))
    return metrics


def _split_train_eval(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = min(source.shape[0], target.shape[0])
    split = max(1, steps // 2)
    if split >= steps:
        split = max(1, steps - 1)
    if steps < 2:
        return source, source, target, target
    return source[:split], source[split:], target[:split], target[split:]


def _fit_diagonal_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    width = source.shape[1]
    scale = np.ones((width,), dtype=np.float64)
    bias = np.zeros((width,), dtype=np.float64)
    for index in range(width):
        x = source[:, index]
        y = target[:, index]
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom <= 1e-12:
            scale[index] = 0.0
            bias[index] = y_mean
        else:
            scale[index] = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            bias[index] = y_mean - scale[index] * x_mean
    return scale, bias


def _fit_global_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-12:
        return 0.0, y_mean
    scale = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return scale, y_mean - scale * x_mean


def repair_metrics(reference: np.ndarray, raw_actions: np.ndarray) -> dict[str, Any]:
    source = np.clip(np.asarray(raw_actions, dtype=np.float64), -1.0, 1.0)
    target = np.asarray(reference, dtype=np.float64)
    train_source, eval_source, train_target, eval_target = _split_train_eval(source, target)
    diag_scale, diag_bias = _fit_diagonal_affine(train_source, train_target)
    global_scale, global_bias = _fit_global_affine(train_source, train_target)
    identity_eval = action_metrics(eval_target, eval_source)
    clipping_only_eval = identity_eval
    global_repaired = np.clip(eval_source * global_scale + global_bias, -1.0, 1.0)
    diagonal_repaired = np.clip(eval_source * diag_scale.reshape(1, -1) + diag_bias.reshape(1, -1), -1.0, 1.0)
    global_eval = action_metrics(eval_target, global_repaired)
    diagonal_eval = action_metrics(eval_target, diagonal_repaired)
    simple_best = min(
        float(identity_eval["action_l2_mean"]),
        float(clipping_only_eval["action_l2_mean"]),
        float(global_eval["action_l2_mean"]),
    )
    diag_l2 = float(diagonal_eval["action_l2_mean"])
    return {
        "supervised_calibration_label": "train_first_half_eval_second_half_hdf5_actions",
        "uses_hdf5_expert_actions_as_supervision": True,
        "not_a_rollout_policy": True,
        "train_steps": int(train_source.shape[0]),
        "eval_steps": int(eval_source.shape[0]),
        "identity_eval_action_l2_mean": identity_eval["action_l2_mean"],
        "clipping_only_eval_action_l2_mean": clipping_only_eval["action_l2_mean"],
        "naive_global_affine_eval_action_l2_mean": global_eval["action_l2_mean"],
        "diagonal_affine_eval_action_l2_mean": diagonal_eval["action_l2_mean"],
        "diagonal_affine_improvement_vs_identity": _round(float(identity_eval["action_l2_mean"]) - diag_l2),
        "diagonal_affine_improvement_vs_global": _round(float(global_eval["action_l2_mean"]) - diag_l2),
        "diagonal_affine_beats_simple_baselines": bool(diag_l2 + 1e-9 < simple_best),
        "naive_global_scale": _round(global_scale, 9),
        "naive_global_bias": _round(global_bias, 9),
        "diagonal_scale": [_round(float(value), 9) for value in diag_scale],
        "diagonal_bias": [_round(float(value), 9) for value in diag_bias],
    }


def _reference_stats(actions: np.ndarray) -> dict[str, Any]:
    stats = action_metrics(actions, actions)
    stats["component_ranges"] = {
        label: _range(actions[:, slc])
        for label, slc in _component_slices().items()
    }
    return stats


def _summarize_variants(variants: dict[str, Any], drift_threshold: float, gripper_threshold: float) -> dict[str, Any]:
    candidate_names = [name for name in PERTURBATION_ORDER if name not in {"correct_7d_expert_action_replay", "clipping_only"}]
    strongest = max(candidate_names, key=lambda name: float(variants[name]["metrics"]["action_l2_mean"]))
    strongest_repair = max(candidate_names, key=lambda name: float(variants[name]["repair"]["diagonal_affine_improvement_vs_identity"]))
    substantial = [
        name
        for name in candidate_names
        if float(variants[name]["metrics"]["action_l2_mean"]) >= drift_threshold
        or float(variants[name]["metrics"].get("gripper_mismatch_rate") or 0.0) >= gripper_threshold
    ]
    repair_beaten = [
        name
        for name in candidate_names
        if bool(variants[name]["repair"]["diagonal_affine_beats_simple_baselines"])
    ]
    return {
        "mismatch_reproduced": bool(substantial),
        "substantial_mismatch_variants": substantial,
        "drift_threshold": drift_threshold,
        "gripper_mismatch_threshold": gripper_threshold,
        "strongest_mismatch_variant": strongest,
        "strongest_mismatch_action_l2_mean": variants[strongest]["metrics"]["action_l2_mean"],
        "strongest_mismatch_gripper_mismatch_rate": variants[strongest]["metrics"].get("gripper_mismatch_rate"),
        "simple_repair_baseline_beaten": bool(repair_beaten),
        "repair_baseline_beaten_variants": repair_beaten,
        "strongest_repair_variant": strongest_repair,
        "strongest_repair_improvement_vs_identity": variants[strongest_repair]["repair"]["diagonal_affine_improvement_vs_identity"],
        "continue_or_kill": "continue" if substantial else "kill",
        "next_state": "STATE 2: evaluate minimal calibration on exact-init replay" if substantial else "kill_or_replace_topic",
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "report_only_hdf5_metric": True,
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
        "simulator_environment_created": False,
        "replay_or_rollout_performed": False,
        "benchmark_rollouts_performed": False,
        "multi_seed_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
        "forbidden_gates_set": forbidden,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary") or {}
    demo = report.get("demo") or {}
    lines = [
        "# ExecSpec Mismatch Diagnostic",
        "",
        "This is report-only HDF5 action evidence. It is not rollout success, benchmark success, or paper-grade evidence.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- mismatch reproduced: `{summary.get('mismatch_reproduced')}`",
        f"- strongest mismatch: `{summary.get('strongest_mismatch_variant')}`",
        f"- strongest action L2 mean: `{summary.get('strongest_mismatch_action_l2_mean')}`",
        f"- simple repair baseline beaten: `{summary.get('simple_repair_baseline_beaten')}`",
        f"- strongest repair variant: `{summary.get('strongest_repair_variant')}`",
        f"- strongest repair improvement vs identity: `{summary.get('strongest_repair_improvement_vs_identity')}`",
        f"- replay/rollout happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- demo path: `{demo.get('path')}`",
        f"- selected steps: `{demo.get('selected_steps')}`",
        "",
        f"Recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    manifest_path = _as_path(args.manifest)
    demo_path = _as_path(args.demo_path) if getattr(args, "demo_path", "") else None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_hdf5_action_mismatch_metric",
        "policy": _policy(forbidden),
        "inputs": {
            "manifest": str(manifest_path) if manifest_path else None,
            "demo_path": str(demo_path) if demo_path else None,
            "max_steps": int(args.max_steps),
            "substantial_drift_threshold": float(args.substantial_drift_threshold),
            "gripper_mismatch_threshold": float(args.gripper_mismatch_threshold),
        },
        "demo": {},
        "reference_action_stats": {},
        "variants": {},
        "summary": {},
        "recommended_next_step": None,
        "result": {"passed": False, "blocked_reason": None},
    }
    if forbidden:
        report["result"]["blocked_reason"] = "forbidden execution gates are set for report-only ExecSpec diagnostic: " + ", ".join(forbidden)
        report["summary"] = {"continue_or_kill": "blocked", "mismatch_reproduced": False}
        report["recommended_next_step"] = "Clear forbidden gates and rerun the report-only HDF5 mismatch diagnostic."
        return report
    if args.max_steps < 2 or args.max_steps > 512:
        report["result"]["blocked_reason"] = "max_steps must be between 2 and 512"
        report["summary"] = {"continue_or_kill": "blocked", "mismatch_reproduced": False}
        report["recommended_next_step"] = "Rerun with a bounded max_steps value."
        return report
    try:
        if demo_path is None:
            if manifest_path is None:
                raise ValueError("either --manifest or --demo-path is required")
            demo_path = _discover_demo_path(manifest_path)
        if not demo_path.exists():
            raise FileNotFoundError(f"HDF5 demo not found: {demo_path}")
        demo = read_hdf5_demo(demo_path, args.max_steps)
        actions = np.asarray(demo.pop("actions"), dtype=np.float64)
        report["demo"] = demo
        report["reference_action_stats"] = _reference_stats(actions)
        variants = {}
        for name, payload in _perturbations(actions).items():
            raw_actions = np.asarray(payload.pop("raw_actions"), dtype=np.float64)
            variants[name] = {
                **payload,
                "metrics": action_metrics(actions, raw_actions),
                "repair": repair_metrics(actions, raw_actions),
            }
        report["variants"] = variants
        report["summary"] = _summarize_variants(
            variants,
            drift_threshold=float(args.substantial_drift_threshold),
            gripper_threshold=float(args.gripper_mismatch_threshold),
        )
        report["result"]["passed"] = True
        report["recommended_next_step"] = (
            "Proceed to STATE 2 with bounded exact-init replay of the strongest mismatch and the minimal diagonal calibration layer."
            if report["summary"]["mismatch_reproduced"]
            else "Kill or replace the topic unless another local asset can reproduce executable mismatch."
        )
    except Exception as exc:
        report["result"]["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        report["summary"] = {"continue_or_kill": "blocked", "mismatch_reproduced": False}
        report["recommended_next_step"] = "Resolve the concrete HDF5/report blocker before claiming ExecSpec mismatch evidence."
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--demo-path", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--substantial-drift-threshold", type=float, default=0.10)
    parser.add_argument("--gripper-mismatch-threshold", type=float, default=0.25)
    parser.add_argument("--report-json", default="reports/execspec_mismatch_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/execspec_mismatch_diagnostic_report.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    assert report_json is not None and report_md is not None
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report_md, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report.get("result", {}).get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
