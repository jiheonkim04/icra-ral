"""STATE 2 calibrated repair diagnostics for ExecSpec-Repair.

The module fits small interpretable action-space repair layers on calibration
HDF5 demos and evaluates held-out action recovery. When the task-local replay
gate is set, it also runs bounded exact-init replay on one held-out demo.
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

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import (
    _as_path,
    _compact,
    _load_env_class,
    _write_json,
)
from tca_map.datasets.libero_full_demo_expert_replay_sanity import _read_demo_full, _run_replay_variant
from tca_map.execspec.mismatch_diagnostic import _load_json, _round, action_metrics


SCHEMA_VERSION = "2026-07-07.execspec_state2_calibrated_repair.v1"
TASK_GATE = "ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY"
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
    "ALLOW_EXECSPEC_MISMATCH_REPLAY",
)
MISMATCH_TYPES = (
    "gripper_sign_flip",
    "translation_scale_mismatch",
    "rotation_scale_mismatch",
    "global_action_scale_mismatch",
    "per_dimension_scale_mismatch",
    "gripper_threshold_0_1_mismatch",
    "range_clipping_mismatch",
)
REPAIR_METHODS = (
    "identity_no_repair",
    "clipping_only",
    "global_affine_calibration",
    "diagonal_affine_calibration",
    "gripper_only_calibration",
    "split_trg_calibration",
    "full_execspec_repair",
)
REPLAY_METHODS = (
    "wrong_executable_spec_replay",
    "clipping_only",
    "global_affine_calibration",
    "diagonal_affine_calibration",
    "gripper_only_calibration",
    "split_trg_calibration",
    "full_execspec_repair",
)
REPLAY_RESULT_KEYS = (
    "variant",
    "claim_role",
    "passed",
    "env_created",
    "reset_ok",
    "set_init_state_used",
    "set_init_state_ok",
    "use_exact_init_state",
    "steps_requested",
    "steps_performed",
    "reward_sum",
    "final_reward",
    "final_success",
    "done_seen",
    "first_done_index",
    "first_positive_reward_index",
    "first_success_index",
    "eef_displacement_l2",
    "env_action_shape",
    "delta_vs_absolute_action_convention_evidence",
    "error",
)


def _mismatch_metadata(name: str) -> dict[str, Any]:
    descriptions = {
        "gripper_sign_flip": "Closed/open gripper sign convention is inverted.",
        "translation_scale_mismatch": "Cartesian translation dimensions are over-scaled before the controller.",
        "rotation_scale_mismatch": "Rotation dimensions are under-scaled before the controller.",
        "global_action_scale_mismatch": "All action dimensions use an incorrect global unnormalization scale.",
        "per_dimension_scale_mismatch": "Each action dimension uses a different incorrect metadata scale.",
        "gripper_threshold_0_1_mismatch": "A 0/1 gripper convention is sent to a -1/1 controller interface.",
        "range_clipping_mismatch": "Action range metadata is too wide, causing controller clipping.",
    }
    plausible = {
        "gripper_sign_flip": "common binary gripper open/close convention mismatch",
        "translation_scale_mismatch": "policy/controller unit or normalization scale mismatch",
        "rotation_scale_mismatch": "axis-angle / delta-rotation normalization mismatch",
        "global_action_scale_mismatch": "incorrect action unnormalizer scale",
        "per_dimension_scale_mismatch": "stale per-dimension action statistics",
        "gripper_threshold_0_1_mismatch": "binary gripper threshold exported with wrong range",
        "range_clipping_mismatch": "controller range mismatch or missing action validity certificate",
    }
    return {"description": descriptions[name], "plausibility": plausible[name]}


def apply_mismatch(actions: np.ndarray, name: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    out = actions.copy()
    if name == "gripper_sign_flip":
        out[:, 6] *= -1.0
    elif name == "translation_scale_mismatch":
        out[:, :3] *= 2.0
    elif name == "rotation_scale_mismatch":
        out[:, 3:6] *= 0.25
    elif name == "global_action_scale_mismatch":
        out *= 1.5
    elif name == "per_dimension_scale_mismatch":
        scale = np.asarray([0.55, 1.6, 0.7, 1.45, 0.5, 1.25, 1.0], dtype=np.float64)
        out *= scale.reshape(1, -1)
    elif name == "gripper_threshold_0_1_mismatch":
        out[:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, 0.0)
    elif name == "range_clipping_mismatch":
        out *= 2.5
    else:
        raise ValueError(f"unknown mismatch type: {name}")
    return out


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-12:
        return 0.0, y_mean
    scale = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return scale, y_mean - scale * x_mean


def _fit_diagonal(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = np.ones((source.shape[1],), dtype=np.float64)
    bias = np.zeros((source.shape[1],), dtype=np.float64)
    for index in range(source.shape[1]):
        scale[index], bias[index] = _fit_affine(source[:, index], target[:, index])
    return scale, bias


def _fit_group(source: np.ndarray, target: np.ndarray, slc: slice) -> tuple[float, float]:
    return _fit_affine(source[:, slc], target[:, slc])


def _gripper_candidates(source: np.ndarray) -> dict[str, np.ndarray]:
    grip = np.asarray(source, dtype=np.float64).reshape(-1)
    affine_scale, affine_bias = _fit_affine(grip, grip)
    return {
        "identity": grip,
        "sign_flip": -grip,
        "threshold_0_1_to_minus1_1": np.where(grip >= 0.5, 1.0, -1.0),
        "sign_threshold": np.where(grip >= 0.0, 1.0, -1.0),
        "affine_self": affine_scale * grip + affine_bias,
    }


def _fit_gripper(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    grip_source = np.asarray(source[:, 6], dtype=np.float64)
    grip_target = np.asarray(target[:, 6], dtype=np.float64)
    candidates = _gripper_candidates(grip_source)
    affine_scale, affine_bias = _fit_affine(grip_source, grip_target)
    candidates["affine"] = affine_scale * grip_source + affine_bias
    best_name = min(
        candidates,
        key=lambda name: (
            float(np.mean((candidates[name] >= 0.0) != (grip_target >= 0.0))),
            float(np.mean(np.abs(np.clip(candidates[name], -1.0, 1.0) - grip_target))),
        ),
    )
    return {
        "kind": best_name,
        "affine_scale": affine_scale,
        "affine_bias": affine_bias,
        "train_sign_mismatch_rate": _round(float(np.mean((candidates[best_name] >= 0.0) != (grip_target >= 0.0))), 6),
        "train_mae": _round(float(np.mean(np.abs(np.clip(candidates[best_name], -1.0, 1.0) - grip_target))), 9),
    }


def _apply_gripper(values: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    grip = np.asarray(values, dtype=np.float64).reshape(-1)
    kind = params.get("kind")
    if kind == "sign_flip":
        return -grip
    if kind == "threshold_0_1_to_minus1_1":
        return np.where(grip >= 0.5, 1.0, -1.0)
    if kind == "sign_threshold":
        return np.where(grip >= 0.0, 1.0, -1.0)
    if kind == "affine":
        return float(params.get("affine_scale", 1.0)) * grip + float(params.get("affine_bias", 0.0))
    return grip


def fit_repair_parameters(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    global_scale, global_bias = _fit_affine(source, target)
    diag_scale, diag_bias = _fit_diagonal(source, target)
    trans_scale, trans_bias = _fit_group(source, target, slice(0, 3))
    rot_scale, rot_bias = _fit_group(source, target, slice(3, 6))
    gripper = _fit_gripper(source, target)
    train_drift = action_metrics(target, source)
    return {
        "global": {"scale": global_scale, "bias": global_bias},
        "diagonal": {"scale": diag_scale.tolist(), "bias": diag_bias.tolist()},
        "split_trg": {
            "translation": {"scale": trans_scale, "bias": trans_bias},
            "rotation": {"scale": rot_scale, "bias": rot_bias},
            "gripper": gripper,
        },
        "gripper": gripper,
        "identity_guard": {
            "applies": bool(float(train_drift["action_l2_mean"]) <= 1e-9),
            "train_action_l2_mean": train_drift["action_l2_mean"],
        },
    }


def apply_repair(actions: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(actions, dtype=np.float64)
    if method in {"identity_no_repair", "wrong_executable_spec_replay"}:
        return raw.copy()
    if method == "clipping_only":
        return np.clip(raw, -1.0, 1.0)
    if method == "global_affine_calibration":
        cfg = params["global"]
        return np.clip(raw * float(cfg["scale"]) + float(cfg["bias"]), -1.0, 1.0)
    if method == "diagonal_affine_calibration":
        cfg = params["diagonal"]
        scale = np.asarray(cfg["scale"], dtype=np.float64).reshape(1, -1)
        bias = np.asarray(cfg["bias"], dtype=np.float64).reshape(1, -1)
        return np.clip(raw * scale + bias, -1.0, 1.0)
    if method == "gripper_only_calibration":
        repaired = np.clip(raw.copy(), -1.0, 1.0)
        repaired[:, 6] = np.clip(_apply_gripper(raw[:, 6], params["gripper"]), -1.0, 1.0)
        return repaired
    if method == "split_trg_calibration":
        cfg = params["split_trg"]
        repaired = raw.copy()
        repaired[:, :3] = raw[:, :3] * float(cfg["translation"]["scale"]) + float(cfg["translation"]["bias"])
        repaired[:, 3:6] = raw[:, 3:6] * float(cfg["rotation"]["scale"]) + float(cfg["rotation"]["bias"])
        repaired[:, 6] = _apply_gripper(raw[:, 6], cfg["gripper"])
        return np.clip(repaired, -1.0, 1.0)
    if method == "full_execspec_repair":
        if params["identity_guard"]["applies"]:
            return np.clip(raw.copy(), -1.0, 1.0)
        cfg = params["diagonal"]
        scale = np.asarray(cfg["scale"], dtype=np.float64).reshape(1, -1)
        bias = np.asarray(cfg["bias"], dtype=np.float64).reshape(1, -1)
        repaired = raw * scale + bias
        repaired[:, 6] = np.where(_apply_gripper(raw[:, 6], params["gripper"]) >= 0.0, 1.0, -1.0)
        return np.clip(repaired, -1.0, 1.0)
    raise ValueError(f"unknown repair method: {method}")


def _read_actions(path: Path, max_steps: int) -> tuple[np.ndarray, dict[str, Any]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        demo_name = sorted(data_group.keys())[0]
        demo = data_group[demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"{path} actions must be [T, 7], got {list(actions.shape)}")
        rewards = np.asarray(demo["rewards"], dtype=np.float64).reshape(-1) if "rewards" in demo else np.zeros((actions.shape[0],))
        dones = np.asarray(demo["dones"], dtype=np.float64).reshape(-1) if "dones" in demo else np.zeros((actions.shape[0],))
    selected = actions[: max(1, min(int(max_steps), int(actions.shape[0])))]
    metadata = {
        "path": str(path),
        "demo_name": str(demo_name),
        "full_action_steps": int(actions.shape[0]),
        "selected_steps": int(selected.shape[0]),
        "first_positive_reward_index": _first_index(rewards, 0.0),
        "first_done_index": _first_index(dones, 0.5),
    }
    return selected, metadata


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _norm_path(path: Path) -> str:
    try:
        return str(path.resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path).replace("\\", "/").lower()


def _manifest_eval_path(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("manifest has no counterfactual pairs")
    pair = pairs[0]
    return _as_path(pair["positive_demo_file"]), pair


def build_data_split(
    *,
    manifest_path: Path,
    data_root: Path,
    max_calibration_demos: int,
    max_eval_demos: int,
) -> dict[str, Any]:
    eval_path, pair = _manifest_eval_path(manifest_path)
    all_paths = sorted(_as_path(path) for path in data_root.rglob("*.hdf5"))
    eval_paths = [eval_path]
    eval_norm = {_norm_path(eval_path)}
    for path in all_paths:
        if len(eval_paths) >= max_eval_demos:
            break
        if _norm_path(path) not in eval_norm:
            eval_paths.append(path)
            eval_norm.add(_norm_path(path))
    calibration_paths = []
    for path in all_paths:
        if len(calibration_paths) >= max_calibration_demos:
            break
        if _norm_path(path) not in eval_norm:
            calibration_paths.append(path)
    overlap = sorted({_norm_path(path) for path in calibration_paths} & {_norm_path(path) for path in eval_paths})
    if overlap:
        raise ValueError("calibration/eval leakage detected: " + ", ".join(overlap))
    if not calibration_paths:
        raise ValueError("no calibration demos available after excluding held-out eval demos")
    if not eval_paths:
        raise ValueError("no eval demos available")
    return {
        "manifest_pair": pair,
        "calibration_paths": calibration_paths,
        "eval_paths": eval_paths,
        "leakage_detected": False,
    }


def _concat_actions(paths: list[Path], max_steps_per_demo: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks = []
    metadata = []
    for path in paths:
        actions, meta = _read_actions(path, max_steps_per_demo)
        chunks.append(actions)
        metadata.append(meta)
    return np.concatenate(chunks, axis=0), metadata


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, default=lambda value: float(value) if isinstance(value, np.floating) else str(value)))


def _recovery_fraction(wrong: float, repaired: float, correct: float = 0.0) -> float | None:
    denom = wrong - correct
    if abs(denom) <= 1e-12:
        return None
    return _round((wrong - repaired) / denom, 9)


def _evaluate_action_repairs(
    *,
    calibration_actions: np.ndarray,
    eval_actions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mismatch_reports: dict[str, Any] = {}
    fitted_params: dict[str, Any] = {}
    for mismatch in MISMATCH_TYPES:
        train_source = apply_mismatch(calibration_actions, mismatch)
        eval_source = apply_mismatch(eval_actions, mismatch)
        params = fit_repair_parameters(train_source, calibration_actions)
        fitted_params[mismatch] = params
        correct_metrics = action_metrics(eval_actions, eval_actions)
        wrong_metrics = action_metrics(eval_actions, eval_source)
        method_reports: dict[str, Any] = {}
        for method in REPAIR_METHODS:
            repaired = apply_repair(eval_source, method, params)
            metrics = action_metrics(eval_actions, repaired)
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
        for method, payload in method_reports.items():
            l2 = float(payload["metrics"]["action_l2_mean"])
            payload["beats_identity"] = bool(l2 + 1e-9 < identity_l2)
            payload["beats_clipping_only"] = bool(l2 + 1e-9 < clipping_l2)
            payload["beats_global_affine"] = bool(l2 + 1e-9 < global_l2)
        mismatch_reports[mismatch] = {
            **_mismatch_metadata(mismatch),
            "correct_spec_metrics": correct_metrics,
            "wrong_spec_metrics": wrong_metrics,
            "repair_methods": method_reports,
            "fit_summary": _sanitize_params(params),
        }
    return mismatch_reports, fitted_params


def _variant_success(variant: dict[str, Any]) -> bool:
    return bool(variant.get("final_success") or variant.get("done_seen") or float(variant.get("reward_sum") or 0.0) > 0.0)


def _compact_replay_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {key: result.get(key) for key in REPLAY_RESULT_KEYS if key in result}
    for key in (
        "action_stats",
        "controller",
        "gripper_timing",
        "object_movement",
        "reward_trajectory_summary",
        "target_directed_movement",
    ):
        if key in result:
            compact[key] = result[key]
    target_audit = result.get("target_key_audit") or {}
    if target_audit:
        compact["target_key_audit"] = {
            "instruction": target_audit.get("instruction"),
            "best_key": target_audit.get("best_key"),
            "best_score": target_audit.get("best_score"),
            "best_overlap": target_audit.get("best_overlap"),
        }
    return compact


def _run_replay(
    *,
    split: dict[str, Any],
    fitted_params: dict[str, Any],
    replay_mismatches: list[str],
    report: dict[str, Any],
    libero_root: Path,
    robosuite_root: Path,
    max_steps_cap: int,
    post_signal_margin: int,
    camera_size: int,
) -> list[dict[str, Any]]:
    pair = split["manifest_pair"]
    eval_path = split["eval_paths"][0]
    demo = _read_demo_full(_as_path(eval_path), max_steps_cap=max_steps_cap, post_signal_margin=post_signal_margin)
    reference = np.asarray(demo["actions"], dtype=np.float64)
    env_cls = _load_env_class(libero_root, robosuite_root)
    bddl_file = libero_root / "libero" / "libero" / "bddl_files" / (pair.get("suite") or "libero_10") / f"{pair['positive_task_id']}.bddl"
    replay_cases = []
    for mismatch in replay_mismatches:
        source = apply_mismatch(reference, mismatch)
        variants = [
            {
                "name": "correct_7d_expert_action_replay",
                "claim_role": "expert_upper_bound",
                "actions": reference,
                "use_exact_init_state": True,
            }
        ]
        action_diagnostics = {
            "correct_7d_expert_action_replay": {"metrics": action_metrics(reference, reference)}
        }
        for method in REPLAY_METHODS:
            method_actions = apply_repair(source, "identity_no_repair" if method == "wrong_executable_spec_replay" else method, fitted_params[mismatch])
            env_actions = np.clip(method_actions, -1.0, 1.0)
            variants.append(
                {
                    "name": method,
                    "claim_role": "identity_no_repair" if method == "wrong_executable_spec_replay" else method,
                    "actions": env_actions,
                    "use_exact_init_state": True,
                }
            )
            action_diagnostics[method] = {
                "metrics": action_metrics(reference, method_actions),
                "env_actions_are_clipped_for_bounded_replay": True,
            }
        results = []
        total_steps = 0
        for variant in variants:
            result = _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=camera_size,
                init_state=demo["init_state"],
                variant=variant,
                instruction=pair["positive_instruction"],
            )
            results.append(_compact_replay_result(result))
            total_steps += int(result.get("steps_performed") or 0)
        expert = next(item for item in results if item["variant"] == "correct_7d_expert_action_replay")
        wrong = next(item for item in results if item["variant"] == "wrong_executable_spec_replay")
        full = next(item for item in results if item["variant"] == "full_execspec_repair")
        replay_cases.append(
            {
                "mismatch_type": mismatch,
                **_mismatch_metadata(mismatch),
                "eval_demo_path": str(eval_path),
                "bddl_file": str(bddl_file),
                "target_horizon": int(reference.shape[0]),
                "hdf5_metadata": {
                    "first_positive_reward_index": demo["first_reward_index"],
                    "first_done_index": demo["first_done_index"],
                    "first_signal_index": demo["first_signal_index"],
                },
                "action_diagnostics": action_diagnostics,
                "replay_results": results,
                "summary": {
                    "expert_replay_succeeded": _variant_success(expert),
                    "wrong_spec_succeeded": _variant_success(wrong),
                    "full_repair_succeeded": _variant_success(full),
                    "expert_reward_sum": _round(float(expert.get("reward_sum") or 0.0), 6),
                    "wrong_spec_reward_sum": _round(float(wrong.get("reward_sum") or 0.0), 6),
                    "full_repair_reward_sum": _round(float(full.get("reward_sum") or 0.0), 6),
                    "replay_degradation_recovered": bool(_variant_success(expert) and not _variant_success(wrong) and _variant_success(full)),
                    "total_steps_performed": total_steps,
                },
            }
        )
    report["policy"]["simulator_environment_created"] = True
    report["policy"]["replay_or_rollout_performed"] = True
    report["policy"]["diagnostic_rollouts_performed"] = True
    return replay_cases


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    action = report["heldout_action_metrics"]
    full_l2 = [
        float(action[name]["repair_methods"]["full_execspec_repair"]["metrics"]["action_l2_mean"])
        for name in MISMATCH_TYPES
    ]
    identity_l2 = [
        float(action[name]["repair_methods"]["identity_no_repair"]["metrics"]["action_l2_mean"])
        for name in MISMATCH_TYPES
    ]
    clipping_l2 = [
        float(action[name]["repair_methods"]["clipping_only"]["metrics"]["action_l2_mean"])
        for name in MISMATCH_TYPES
    ]
    global_l2 = [
        float(action[name]["repair_methods"]["global_affine_calibration"]["metrics"]["action_l2_mean"])
        for name in MISMATCH_TYPES
    ]
    full_beats_identity = float(np.mean(full_l2)) + 1e-9 < float(np.mean(identity_l2))
    full_beats_clipping = float(np.mean(full_l2)) + 1e-9 < float(np.mean(clipping_l2))
    full_beats_global = float(np.mean(full_l2)) + 1e-9 < float(np.mean(global_l2))
    full_recoveries = [
        float(action[name]["repair_methods"]["full_execspec_repair"]["recovery_fraction"] or 0.0)
        for name in MISMATCH_TYPES
    ]
    best_method_scores = {}
    for method in REPAIR_METHODS:
        scores = [
            float(action[name]["repair_methods"][method]["recovery_fraction"] or 0.0)
            for name in MISMATCH_TYPES
        ]
        best_method_scores[method] = float(np.mean(scores))
    best_method = max(best_method_scores, key=best_method_scores.get)
    replay_cases = report.get("exact_init_replay") or []
    replay_recovered = any(case.get("summary", {}).get("replay_degradation_recovered") for case in replay_cases)
    decision = "continue" if full_beats_identity and full_beats_clipping and full_beats_global and replay_recovered else "kill_or_reframe"
    return {
        "full_repair_beats_identity_on_action_drift": bool(full_beats_identity),
        "full_repair_beats_clipping_only_on_action_drift": bool(full_beats_clipping),
        "full_repair_beats_global_affine_on_action_drift": bool(full_beats_global),
        "mean_action_l2": {
            "identity_no_repair": _round(float(np.mean(identity_l2)), 9),
            "clipping_only": _round(float(np.mean(clipping_l2)), 9),
            "global_affine_calibration": _round(float(np.mean(global_l2)), 9),
            "full_execspec_repair": _round(float(np.mean(full_l2)), 9),
        },
        "per_mismatch_full_beat_counts": {
            "identity": int(sum(action[name]["repair_methods"]["full_execspec_repair"]["beats_identity"] for name in MISMATCH_TYPES)),
            "clipping_only": int(sum(action[name]["repair_methods"]["full_execspec_repair"]["beats_clipping_only"] for name in MISMATCH_TYPES)),
            "global_affine": int(sum(action[name]["repair_methods"]["full_execspec_repair"]["beats_global_affine"] for name in MISMATCH_TYPES)),
            "total": len(MISMATCH_TYPES),
        },
        "full_repair_mean_recovery_fraction": _round(float(np.mean(full_recoveries)), 9),
        "best_repair_method_by_mean_recovery": best_method,
        "best_repair_method_mean_recovery_fraction": _round(best_method_scores[best_method], 9),
        "repair_improves_replay_reward_or_success": bool(replay_recovered),
        "continue_or_kill": decision,
        "next_state": "STATE 3 replay/rollout validation" if decision == "continue" else "kill/reframe",
    }


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_execspec_state2": True,
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


def _md_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(_round(float(value), digits))
    return str(value)


def _metric(metrics: dict[str, Any], key: str) -> str:
    return _md_value(metrics.get(key))


def _replay_degradation_label(report: dict[str, Any], mismatch: str) -> str:
    for case in report.get("exact_init_replay") or []:
        if case.get("mismatch_type") != mismatch:
            continue
        summary = case.get("summary") or {}
        expert = f"{_md_value(summary.get('expert_reward_sum'))}/{_md_value(summary.get('expert_replay_succeeded'))}"
        wrong = f"{_md_value(summary.get('wrong_spec_reward_sum'))}/{_md_value(summary.get('wrong_spec_succeeded'))}"
        return f"{_md_value(summary.get('replay_degradation_recovered'))}; expert {expert}, wrong {wrong}"
    return "not replayed in STATE 2"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary") or {}
    split = report.get("split") or {}
    mean_l2 = summary.get("mean_action_l2") or {}
    beat_counts = summary.get("per_mismatch_full_beat_counts") or {}
    lines = [
        "# ExecSpec STATE 2 Calibrated Repair",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success or paper-grade evidence.",
        "",
        f"- decision: `{summary.get('continue_or_kill')}`",
        f"- calibration demos: `{split.get('calibration_demo_count')}`",
        f"- eval demos: `{split.get('eval_demo_count')}`",
        f"- calibration action samples: `{split.get('calibration_action_samples')}`",
        f"- eval action samples: `{split.get('eval_action_samples')}`",
        f"- eval leakage detected: `{split.get('leakage_detected')}`",
        f"- task count: `{split.get('task_count')}`",
        f"- mismatch types: `{', '.join(report.get('mismatch_types_tested', []))}`",
        f"- best repair method: `{summary.get('best_repair_method_by_mean_recovery')}`",
        f"- full beats identity: `{summary.get('full_repair_beats_identity_on_action_drift')}`",
        f"- full beats clipping-only: `{summary.get('full_repair_beats_clipping_only_on_action_drift')}`",
        f"- full beats global affine: `{summary.get('full_repair_beats_global_affine_on_action_drift')}`",
        f"- full mean recovery fraction: `{summary.get('full_repair_mean_recovery_fraction')}`",
        f"- mean action L2 identity/clipping/global/full: `{mean_l2.get('identity_no_repair')}` / `{mean_l2.get('clipping_only')}` / `{mean_l2.get('global_affine_calibration')}` / `{mean_l2.get('full_execspec_repair')}`",
        f"- full repair per-mismatch beat counts identity/clipping/global: `{beat_counts.get('identity')}/{beat_counts.get('total')}` / `{beat_counts.get('clipping_only')}/{beat_counts.get('total')}` / `{beat_counts.get('global_affine')}/{beat_counts.get('total')}`",
        f"- replay/rollout happened: `{report.get('policy', {}).get('replay_or_rollout_performed')}`",
        f"- replay improved reward/success: `{summary.get('repair_improves_replay_reward_or_success')}`",
        f"- next state: `{summary.get('next_state')}`",
        "",
        "## Held-Out Action Metrics",
        "",
        "| mismatch | generated as | plausible source | wrong L2 | wrong T/R/G | wrong clip/valid | full L2 | full T/R/G | full clip/valid | recovery | full beats id/clip/global | replay degradation before repair |",
        "| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for mismatch in report.get("mismatch_types_tested", []):
        payload = report.get("heldout_action_metrics", {}).get(mismatch, {})
        wrong = payload.get("wrong_spec_metrics") or {}
        full_payload = (payload.get("repair_methods") or {}).get("full_execspec_repair", {})
        full = full_payload.get("metrics") or {}
        wrong_trg = "/".join(
            [
                _metric(wrong, "translation_drift_mean"),
                _metric(wrong, "rotation_drift_mean"),
                _metric(wrong, "gripper_mismatch_rate"),
            ]
        )
        full_trg = "/".join(
            [
                _metric(full, "translation_drift_mean"),
                _metric(full, "rotation_drift_mean"),
                _metric(full, "gripper_mismatch_rate"),
            ]
        )
        wrong_clip_valid = f"{_metric(wrong, 'clip_rate_step')}/{_metric(wrong, 'controller_valid_action_rate')}"
        full_clip_valid = f"{_metric(full, 'clip_rate_step')}/{_metric(full, 'controller_valid_action_rate')}"
        beats = "/".join(
            [
                _md_value(full_payload.get("beats_identity")),
                _md_value(full_payload.get("beats_clipping_only")),
                _md_value(full_payload.get("beats_global_affine")),
            ]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    mismatch,
                    payload.get("description", ""),
                    payload.get("plausibility", ""),
                    _metric(wrong, "action_l2_mean"),
                    wrong_trg,
                    wrong_clip_valid,
                    _metric(full, "action_l2_mean"),
                    full_trg,
                    full_clip_valid,
                    _md_value(full_payload.get("recovery_fraction")),
                    beats,
                    _replay_degradation_label(report, mismatch),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Exact-Init Replay Metrics", ""])
    replay_cases = report.get("exact_init_replay") or []
    if not replay_cases:
        lines.append(f"Exact-init replay skipped: `{report.get('replay_skip_reason')}`")
    for case in replay_cases:
        summary_case = case.get("summary") or {}
        lines.extend(
            [
                f"### {case.get('mismatch_type')}",
                "",
                f"- degradation recovered: `{summary_case.get('replay_degradation_recovered')}`",
                f"- total simulator steps performed: `{summary_case.get('total_steps_performed')}`",
                f"- HDF5 first reward/done/signal index: `{(case.get('hdf5_metadata') or {}).get('first_positive_reward_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_done_index')}` / `{(case.get('hdf5_metadata') or {}).get('first_signal_index')}`",
                "",
                "| variant | reward | success | done index | trajectory length | valid | clip | action L2 | gripper error | T/R drift |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        diagnostics = case.get("action_diagnostics") or {}
        for result in case.get("replay_results") or []:
            variant = result.get("variant")
            metrics = (diagnostics.get(variant) or {}).get("metrics") or {}
            success = _variant_success(result)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(variant),
                        _md_value(result.get("reward_sum")),
                        _md_value(success),
                        _md_value(result.get("first_done_index")),
                        _md_value(result.get("steps_performed")),
                        _metric(metrics, "controller_valid_action_rate"),
                        _metric(metrics, "clip_rate_step"),
                        _metric(metrics, "action_l2_mean"),
                        _metric(metrics, "gripper_mismatch_rate"),
                        f"{_metric(metrics, 'translation_drift_mean')}/{_metric(metrics, 'rotation_drift_mean')}",
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_state2_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_state2_calibrated_repair",
        "policy": _policy(forbidden),
        "inputs": vars(args).copy(),
        "split": {},
        "mismatch_types_tested": list(MISMATCH_TYPES),
        "repair_methods_tested": list(REPAIR_METHODS),
        "heldout_action_metrics": {},
        "exact_init_replay": [],
        "summary": {},
        "result": {"passed": False, "blocked_reason": None},
        "elapsed_seconds": None,
    }
    if forbidden:
        report["result"]["blocked_reason"] = "forbidden gates set: " + ", ".join(forbidden)
        return report
    try:
        split = build_data_split(
            manifest_path=_as_path(args.manifest),
            data_root=_as_path(args.data_root),
            max_calibration_demos=args.max_calibration_demos,
            max_eval_demos=args.max_eval_demos,
        )
        calibration_actions, calibration_meta = _concat_actions(split["calibration_paths"], args.max_actions_per_demo)
        eval_actions, eval_meta = _concat_actions(split["eval_paths"], args.max_actions_per_demo)
        tasks = sorted({Path(meta["path"]).stem.replace("_demo", "") for meta in calibration_meta + eval_meta})
        report["split"] = {
            "calibration_demo_count": len(split["calibration_paths"]),
            "eval_demo_count": len(split["eval_paths"]),
            "task_count": len(tasks),
            "tasks": tasks,
            "calibration_action_samples": int(calibration_actions.shape[0]),
            "eval_action_samples": int(eval_actions.shape[0]),
            "calibration_paths": [str(path) for path in split["calibration_paths"]],
            "eval_paths": [str(path) for path in split["eval_paths"]],
            "leakage_detected": False,
            "limitation": "small held-out replay set by design; action metrics cover all configured mismatches",
        }
        action_metrics_report, fitted_params = _evaluate_action_repairs(
            calibration_actions=calibration_actions,
            eval_actions=eval_actions,
        )
        report["heldout_action_metrics"] = action_metrics_report
        if os.environ.get(TASK_GATE) == "1":
            replay_mismatches = [name.strip() for name in args.replay_mismatches.split(",") if name.strip()]
            report["exact_init_replay"] = _run_replay(
                split=split,
                fitted_params=fitted_params,
                replay_mismatches=replay_mismatches,
                report=report,
                libero_root=_as_path(args.libero_root),
                robosuite_root=_as_path(args.robosuite_root),
                max_steps_cap=args.max_steps_cap,
                post_signal_margin=args.post_signal_margin,
                camera_size=args.camera_size,
            )
        else:
            report["exact_init_replay"] = []
            report["replay_skip_reason"] = f"{TASK_GATE}=1 not set; action-repair metrics only"
        report["summary"] = _summarize(report)
        report["result"]["passed"] = True
    except Exception as exc:
        report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
        report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["summary"] = {"continue_or_kill": "blocked", "next_state": "resolve_state2_blocker"}
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--data-root", default="C:/assets/data/libero")
    parser.add_argument("--libero-root", default=os.environ.get("TCA_MAP_LIBERO_ROOT_WSL", "/mnt/c/assets/repos/LIBERO"))
    parser.add_argument("--robosuite-root", default=os.environ.get("TCA_MAP_ROBOSUITE_ROOT_WSL", "/mnt/c/assets/repos/robosuite"))
    parser.add_argument("--max-calibration-demos", type=int, default=5)
    parser.add_argument("--max-eval-demos", type=int, default=1)
    parser.add_argument("--max-actions-per-demo", type=int, default=300)
    parser.add_argument("--max-steps-cap", type=int, default=300)
    parser.add_argument("--post-signal-margin", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--replay-mismatches", default="gripper_sign_flip,translation_scale_mismatch")
    parser.add_argument("--report-json", default="reports/execspec_state2_calibrated_repair.json")
    parser.add_argument("--report-md", default="reports/execspec_state2_calibrated_repair.md")
    args = parser.parse_args(argv)
    report = build_state2_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    console_report = {
        "result": report["result"],
        "summary": report["summary"],
        "split": report["split"],
        "report_json": str(report_json),
    }
    print(json.dumps(console_report, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
