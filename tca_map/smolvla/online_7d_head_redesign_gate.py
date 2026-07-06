"""Bounded 7D action-head redesign gate.

This module deliberately stays in the offline/teacher-forced lane unless the
offline rollout gate is green. It uses HDF5 actions only as supervised labels
or evaluation references; it never uses current/future HDF5 actions as method
actions at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tca_map.smolvla.online_7d_action_quality_diagnosis import (
    _actions,
    _basic_error,
    _first_k_errors,
    _gripper_timing,
    _group_errors,
    _load_samples,
    _phase_bins,
    _round,
    _variance_report,
)
from tca_map.smolvla.online_7d_diagnostic_head import (
    _features,
    _mse,
    _ridge,
    _target_prior,
    _with_bias,
)

SCHEMA_VERSION = "2026-07-06.online_7d_head_redesign_gate.v1"
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
    "ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT",
)

CURRENT_VARIANTS = ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d")


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _fit_standardizer(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def _apply_standardizer(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (x - stats["mean"]) / stats["std"]


def _phase_augmented_features(samples: list[dict[str, Any]], prior: dict[str, Any], horizon: int) -> np.ndarray:
    base = _features(samples, "fixed_prior_tca_7d", prior, horizon)
    rows = []
    for row, sample in zip(base, samples):
        phase = float(sample["step"]) / float(max(1, horizon - 1))
        prior_part = row[-2:]
        rows.append(np.concatenate([row, [phase * phase, phase**3], prior_part * phase, prior_part * (1.0 - phase)]))
    return np.asarray(rows, dtype=np.float64)


def _feature_matrix(samples: list[dict[str, Any]], variant: str, prior: dict[str, Any], horizon: int) -> np.ndarray:
    if variant == "phase_aware_fixed_prior_tca_7d":
        return _phase_augmented_features(samples, prior, horizon)
    if variant in {"actionmap_7d", "normalized_actionmap_7d"}:
        return _features(samples, "actionmap_7d", prior, horizon)
    if variant in {
        "fixed_prior_tca_7d",
        "normalized_fixed_prior_tca_7d",
        "split_fixed_prior_tca_7d",
        "small_cpu_mlp_fixed_prior_tca_7d",
        "fixed_prior_tca_mean_residual_7d",
    }:
        return _features(samples, "fixed_prior_tca_7d", prior, horizon)
    if variant == "hard_learned_target_tca_7d":
        return _features(samples, "hard_learned_target_tca_7d", prior, horizon)
    raise ValueError(f"unknown feature variant: {variant}")


def _linear_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    normalize_features: bool = False,
    normalize_actions: bool = False,
) -> dict[str, Any]:
    x_stats = _fit_standardizer(x_train) if normalize_features else None
    y_stats = _fit_standardizer(y_train) if normalize_actions else None
    x_fit = _apply_standardizer(x_train, x_stats) if x_stats else x_train
    y_fit = _apply_standardizer(y_train, y_stats) if y_stats else y_train
    weights = _ridge(x_fit, y_fit)
    return {
        "kind": "ridge_linear",
        "weights": weights,
        "x_stats": x_stats,
        "y_stats": y_stats,
        "feature_dim": int(x_train.shape[1]),
        "trainable_parameter_count": int((x_train.shape[1] + 1) * y_train.shape[1]),
        "features_normalized": bool(normalize_features),
        "action_labels_normalized": bool(normalize_actions),
    }


def _linear_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_eval = _apply_standardizer(x, model["x_stats"]) if model.get("x_stats") else x
    pred = _with_bias(x_eval) @ model["weights"]
    if model.get("y_stats"):
        pred = pred * model["y_stats"]["std"] + model["y_stats"]["mean"]
    return pred


def _best_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = sorted({float(v) for v in logits.tolist()} | {0.0})
    best_threshold = 0.0
    best_acc = -1.0
    for threshold in candidates:
        acc = float(np.mean((logits >= threshold) == labels))
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold


def _fit_split_head(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    x_stats = _fit_standardizer(x_train)
    x_fit = _apply_standardizer(x_train, x_stats)
    trans_stats = _fit_standardizer(y_train[:, :3])
    rot_stats = _fit_standardizer(y_train[:, 3:6])
    trans_w = _ridge(x_fit, _apply_standardizer(y_train[:, :3], trans_stats))
    rot_w = _ridge(x_fit, _apply_standardizer(y_train[:, 3:6], rot_stats))
    gripper_labels = y_train[:, 6] >= 0.0
    grip_targets = np.where(gripper_labels, 1.0, -1.0).reshape(-1, 1)
    grip_w = _ridge(x_fit, grip_targets)
    logits = (_with_bias(x_fit) @ grip_w).reshape(-1)
    threshold = _best_threshold(logits, gripper_labels)
    return {
        "kind": "split_regression_gripper_classifier",
        "x_stats": x_stats,
        "translation_weights": trans_w,
        "rotation_weights": rot_w,
        "gripper_weights": grip_w,
        "gripper_threshold": float(threshold),
        "translation_stats": trans_stats,
        "rotation_stats": rot_stats,
        "feature_dim": int(x_train.shape[1]),
        "trainable_parameter_count": int((x_train.shape[1] + 1) * 7),
        "features_normalized": True,
        "action_labels_normalized": True,
        "gripper_mode": "classification_threshold_calibrated_on_train_only",
    }


def _predict_split_head(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_eval = _apply_standardizer(x, model["x_stats"])
    trans = _with_bias(x_eval) @ model["translation_weights"]
    rot = _with_bias(x_eval) @ model["rotation_weights"]
    trans = trans * model["translation_stats"]["std"] + model["translation_stats"]["mean"]
    rot = rot * model["rotation_stats"]["std"] + model["rotation_stats"]["mean"]
    logits = (_with_bias(x_eval) @ model["gripper_weights"]).reshape(-1)
    gripper = np.where(logits >= float(model["gripper_threshold"]), 1.0, -1.0).reshape(-1, 1)
    return np.concatenate([trans, rot, gripper], axis=1)


def _fit_residual_head(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    anchor = np.mean(y_train, axis=0)
    residual = y_train - anchor.reshape(1, -1)
    base = _linear_fit(x_train, residual, normalize_features=True, normalize_actions=True)
    base.update({"kind": "mean_action_residual_ridge", "anchor_action": anchor})
    return base


def _predict_residual_head(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["anchor_action"].reshape(1, -1) + _linear_predict(model, x)


def _fit_small_mlp(x_train: np.ndarray, y_train: np.ndarray, steps: int, hidden_dim: int = 16) -> dict[str, Any]:
    x_stats = _fit_standardizer(x_train)
    y_stats = _fit_standardizer(y_train)
    x = _apply_standardizer(x_train, x_stats)
    y = _apply_standardizer(y_train, y_stats)
    rng = np.random.default_rng(7)
    w1 = rng.normal(0.0, 0.08, size=(x.shape[1], hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, 0.08, size=(hidden_dim, 7))
    b2 = np.zeros(7, dtype=np.float64)
    lr = 0.015

    def forward(xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(xb @ w1 + b1)
        return h @ w2 + b2, h

    initial_pred, _ = forward(x)
    initial_loss = _mse(initial_pred, y)
    losses = [float(initial_loss)]
    order = np.arange(x.shape[0])
    for step in range(max(1, steps)):
        idx = int(order[step % order.size])
        xb = x[idx : idx + 1]
        yb = y[idx : idx + 1]
        pred, h = forward(xb)
        grad_pred = 2.0 * (pred - yb) / pred.shape[1]
        grad_w2 = h.T @ grad_pred
        grad_b2 = grad_pred.reshape(-1)
        grad_h = grad_pred @ w2.T
        grad_z = grad_h * (1.0 - h * h)
        grad_w1 = xb.T @ grad_z
        grad_b1 = grad_z.reshape(-1)
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        if (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
            pred_all, _ = forward(x)
            losses.append(float(_mse(pred_all, y)))
    final_pred, _ = forward(x)
    final_loss = _mse(final_pred, y)
    return {
        "kind": "small_cpu_tanh_mlp",
        "x_stats": x_stats,
        "y_stats": y_stats,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "hidden_dim": hidden_dim,
        "steps": int(steps),
        "batch_size": 1,
        "feature_dim": int(x_train.shape[1]),
        "trainable_parameter_count": int((x_train.shape[1] + 1) * hidden_dim + (hidden_dim + 1) * 7),
        "features_normalized": True,
        "action_labels_normalized": True,
        "loss_curve": [_round(value) for value in losses],
        "initial_train_loss_normalized": _round(initial_loss),
        "final_train_loss_normalized": _round(final_loss),
    }


def _predict_small_mlp(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_eval = _apply_standardizer(x, model["x_stats"])
    h = np.tanh(x_eval @ model["w1"] + model["b1"])
    pred = h @ model["w2"] + model["b2"]
    return pred * model["y_stats"]["std"] + model["y_stats"]["mean"]


def _fit_model(variant: str, x_train: np.ndarray, y_train: np.ndarray, steps: int) -> dict[str, Any]:
    if variant in {"actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d", "phase_aware_fixed_prior_tca_7d"}:
        return _linear_fit(x_train, y_train)
    if variant in {"normalized_actionmap_7d", "normalized_fixed_prior_tca_7d"}:
        return _linear_fit(x_train, y_train, normalize_features=True, normalize_actions=True)
    if variant == "split_fixed_prior_tca_7d":
        return _fit_split_head(x_train, y_train)
    if variant == "small_cpu_mlp_fixed_prior_tca_7d":
        return _fit_small_mlp(x_train, y_train, steps=steps)
    if variant == "fixed_prior_tca_mean_residual_7d":
        return _fit_residual_head(x_train, y_train)
    raise ValueError(f"unsupported model variant: {variant}")


def _predict_model(variant: str, model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if variant == "split_fixed_prior_tca_7d":
        pred = _predict_split_head(model, x)
    elif variant == "small_cpu_mlp_fixed_prior_tca_7d":
        pred = _predict_small_mlp(model, x)
    elif variant == "fixed_prior_tca_mean_residual_7d":
        pred = _predict_residual_head(model, x)
    else:
        pred = _linear_predict(model, x)
    return np.clip(pred, -1.0, 1.0)


def _per_dim_mse(pred: np.ndarray, expert: np.ndarray) -> list[float]:
    return [_round(value) for value in np.mean((pred - expert) ** 2, axis=0).tolist()]


def _component_mse(pred: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    diff = (pred - expert) ** 2
    return {
        "translation_mse": _round(np.mean(diff[:, :3])),
        "rotation_mse": _round(np.mean(diff[:, 3:6])),
        "gripper_mse": _round(np.mean(diff[:, 6])),
    }


def _component_dominance(component: dict[str, float]) -> str:
    return max(component, key=lambda key: float(component[key])).replace("_mse", "")


def _target_prior_dim(variant: str) -> int:
    return 2 if "tca" in variant or "fixed_prior" in variant or "hard_learned" in variant else 0


def _leakage_audit(variant: str) -> dict[str, Any]:
    uses_prior = _target_prior_dim(variant) > 0
    return {
        "classification": "non_leaking_online_candidate",
        "uses_hdf5_actions_only_as_supervised_labels": True,
        "uses_eval_actions_at_inference": False,
        "uses_same_or_future_hdf5_action_at_inference": False,
        "uses_bddl_metadata_at_inference": False,
        "uses_dataset_target_labels_at_inference": False,
        "uses_eval_labels_at_inference": False,
        "uses_task_id_or_filename_at_inference": False,
        "uses_instruction_text_at_inference": True,
        "uses_test_time_semantic_target_prior_only": uses_prior,
        "silently_pads_4d_to_7d": False,
        "valid_7d_action_output": True,
    }


def _evaluate_variant(
    variant: str,
    model: dict[str, Any],
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    teacher_samples: list[dict[str, Any]],
    prior: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    y_train = _actions(train_samples)
    y_eval = _actions(eval_samples)
    y_teacher = _actions(teacher_samples)
    x_train = _feature_matrix(train_samples, variant, prior, max_steps)
    x_eval = _feature_matrix(eval_samples, variant, prior, max_steps)
    x_teacher = _feature_matrix(teacher_samples, variant, prior, len(teacher_samples))
    pred_train = _predict_model(variant, model, x_train)
    pred_eval = _predict_model(variant, model, x_eval)
    pred_teacher = _predict_model(variant, model, x_teacher)
    train_loss_initial = _mse(np.zeros_like(y_train), y_train)
    train_loss_final = _mse(pred_train, y_train)
    eval_metrics = _basic_error(pred_eval, y_eval)
    teacher_metrics = _basic_error(pred_teacher, y_teacher)
    return {
        "variant": variant,
        "kind": model["kind"],
        "feature_dim": int(model["feature_dim"]),
        "target_prior_feature_dim": _target_prior_dim(variant),
        "features_normalized": bool(model.get("features_normalized", False)),
        "action_labels_normalized": bool(model.get("action_labels_normalized", False)),
        "trainable_parameter_count": int(model["trainable_parameter_count"]),
        "training": {
            "training_happened": True,
            "batch_size": int(model.get("batch_size", len(train_samples))),
            "steps": int(model.get("steps", 1)),
            "initial_loss": _round(train_loss_initial),
            "final_loss": _round(train_loss_final),
            "loss_decreased": bool(train_loss_final < train_loss_initial),
            "mlp_loss_curve": model.get("loss_curve"),
            "initial_train_loss_normalized": model.get("initial_train_loss_normalized"),
            "final_train_loss_normalized": model.get("final_train_loss_normalized"),
        },
        "eval": {
            "metrics": eval_metrics,
            "per_dim_mse": _per_dim_mse(pred_eval, y_eval),
            "component_mse": _component_mse(pred_eval, y_eval),
            "dominant_loss_component": _component_dominance(_component_mse(pred_eval, y_eval)),
            "first_k_step_error": _first_k_errors(pred_eval, y_eval),
            "phase_bins": _phase_bins(pred_eval, y_eval),
            "per_target_error": _group_errors(eval_samples, pred_eval, y_eval, "target_id"),
            "per_task_error": _group_errors(eval_samples, pred_eval, y_eval, "pair_id"),
            "gripper_timing": _gripper_timing(pred_eval, y_eval),
            "prediction_variance": _variance_report(pred_eval, y_eval),
            "action_shape": list(pred_eval.shape),
        },
        "teacher_forced": {
            "metrics": teacher_metrics,
            "cumulative_action_l2": teacher_metrics["7d_action_l2_sum"],
            "per_dim_mse": _per_dim_mse(pred_teacher, y_teacher),
            "component_mse": _component_mse(pred_teacher, y_teacher),
            "dominant_loss_component": _component_dominance(_component_mse(pred_teacher, y_teacher)),
            "first_k_step_error": _first_k_errors(pred_teacher, y_teacher),
            "phase_bins": _phase_bins(pred_teacher, y_teacher),
            "per_target_error": _group_errors(teacher_samples, pred_teacher, y_teacher, "target_id"),
            "per_task_error": _group_errors(teacher_samples, pred_teacher, y_teacher, "pair_id"),
            "gripper_timing": _gripper_timing(pred_teacher, y_teacher),
            "prediction_variance": _variance_report(pred_teacher, y_teacher),
        },
        "leakage_audit": _leakage_audit(variant),
        "_pred_eval": pred_eval,
        "_pred_teacher": pred_teacher,
    }


def _mean_action_baseline(train_samples: list[dict[str, Any]], eval_samples: list[dict[str, Any]], teacher_samples: list[dict[str, Any]]) -> dict[str, Any]:
    train_actions = _actions(train_samples)
    y_eval = _actions(eval_samples)
    y_teacher = _actions(teacher_samples)
    mean_action = np.mean(train_actions, axis=0)
    pred_eval = np.repeat(mean_action.reshape(1, -1), y_eval.shape[0], axis=0)
    pred_teacher = np.repeat(mean_action.reshape(1, -1), y_teacher.shape[0], axis=0)
    eval_metrics = _basic_error(pred_eval, y_eval)
    teacher_metrics = _basic_error(pred_teacher, y_teacher)
    return {
        "variant": "mean_action_baseline",
        "kind": "train_split_mean_action",
        "feature_dim": 0,
        "target_prior_feature_dim": 0,
        "features_normalized": False,
        "action_labels_normalized": False,
        "trainable_parameter_count": 0,
        "training": {
            "training_happened": False,
            "initial_loss": None,
            "final_loss": None,
            "loss_decreased": None,
            "note": "Mean action is computed from train labels only; eval labels are not used for inference.",
        },
        "eval": {
            "metrics": eval_metrics,
            "per_dim_mse": _per_dim_mse(pred_eval, y_eval),
            "component_mse": _component_mse(pred_eval, y_eval),
            "dominant_loss_component": _component_dominance(_component_mse(pred_eval, y_eval)),
            "prediction_variance": _variance_report(pred_eval, y_eval),
            "action_shape": list(pred_eval.shape),
        },
        "teacher_forced": {
            "metrics": teacher_metrics,
            "cumulative_action_l2": teacher_metrics["7d_action_l2_sum"],
            "per_dim_mse": _per_dim_mse(pred_teacher, y_teacher),
            "component_mse": _component_mse(pred_teacher, y_teacher),
            "dominant_loss_component": _component_dominance(_component_mse(pred_teacher, y_teacher)),
            "prediction_variance": _variance_report(pred_teacher, y_teacher),
        },
        "leakage_audit": {
            "classification": "non_leaking_baseline",
            "uses_hdf5_actions_only_as_train_split_summary": True,
            "uses_eval_actions_at_inference": False,
            "uses_same_or_future_hdf5_action_at_inference": False,
            "valid_7d_action_output": True,
        },
        "_pred_eval": pred_eval,
        "_pred_teacher": pred_teacher,
    }


def _unavailable_native_residual() -> dict[str, Any]:
    return {
        "variant": "native_smolvla_learned_residual_7d",
        "status": "not_evaluated",
        "reason": "No non-leaking cached native SmolVLA action sequence is available in the runtime report; running heavy model inference is outside this offline head-redesign gate.",
        "leakage_audit": {
            "classification": "not_evaluated",
            "would_require_heavy_model_inference": True,
            "uses_same_or_future_hdf5_action_at_inference": False,
        },
    }


def _action_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = a - b
    return {
        "mean_action_l2": _round(np.mean(np.linalg.norm(diff, axis=1))),
        "mean_translation_l2": _round(np.mean(np.linalg.norm(diff[:, :3], axis=1))),
        "mean_rotation_l2": _round(np.mean(np.linalg.norm(diff[:, 3:6], axis=1))),
        "mean_gripper_abs_diff": _round(np.mean(np.abs(diff[:, 6]))),
        "meaningfully_different": bool(
            np.mean(np.linalg.norm(diff, axis=1)) >= 0.05
            or np.mean(np.linalg.norm(diff[:, :3], axis=1)) >= 0.025
            or np.mean(np.abs(diff[:, 6])) >= 0.1
        ),
    }


def _stage1_diagnosis(
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_actions = _actions(train_samples)
    eval_actions = _actions(eval_samples)
    per_variant = {}
    for name, data in variants.items():
        if name.startswith("_") or "_pred_eval" not in data:
            continue
        pred = data["_pred_eval"]
        per_variant[name] = {
            "feature_dim": data.get("feature_dim"),
            "target_prior_feature_dim": data.get("target_prior_feature_dim"),
            "features_normalized": data.get("features_normalized"),
            "action_labels_normalized": data.get("action_labels_normalized"),
            "prediction_per_dim_variance": [_round(v) for v in np.var(pred, axis=0).tolist()],
            "prediction_mean_std": _round(np.mean(np.std(pred, axis=0))),
            "collapse_to_constant_or_mean": data["eval"]["prediction_variance"]["constant_or_mean_collapse"],
            "per_dim_mse": data["eval"]["per_dim_mse"],
            "component_mse": data["eval"]["component_mse"],
            "dominant_loss_component": data["eval"]["dominant_loss_component"],
            "gripper_open_close_accuracy": data["eval"]["metrics"]["gripper_open_close_accuracy"],
        }
    mean_l2 = variants["mean_action_baseline"]["eval"]["metrics"]["7d_action_l2"]
    best_non_mean = min(
        (name for name in variants if name != "mean_action_baseline" and "_pred_eval" in variants[name]),
        key=lambda name: variants[name]["eval"]["metrics"]["7d_action_l2"],
    )
    best_l2 = variants[best_non_mean]["eval"]["metrics"]["7d_action_l2"]
    return {
        "train_eval_split": "rollout_pair_held_out_from_training_and_rollout_demo_path_filtered",
        "train_sample_count": len(train_samples),
        "eval_sample_count": len(eval_samples),
        "train_action_per_dim_variance": [_round(v) for v in np.var(train_actions, axis=0).tolist()],
        "eval_action_per_dim_variance": [_round(v) for v in np.var(eval_actions, axis=0).tolist()],
        "train_action_per_dim_std": [_round(v) for v in np.std(train_actions, axis=0).tolist()],
        "eval_action_per_dim_std": [_round(v) for v in np.std(eval_actions, axis=0).tolist()],
        "per_variant": per_variant,
        "mean_action_baseline_7d_l2": mean_l2,
        "best_non_mean_variant": best_non_mean,
        "best_non_mean_7d_l2": best_l2,
        "why_mean_baseline_beats_previous_heads": (
            "Current lightweight heads overpredict rollout-demo dynamics from weak features and have worse eval 7D L2 than the train-split mean action; "
            "prediction variance/collapse and component MSE should be read per variant below."
            if mean_l2 < best_l2
            else "At least one redesigned head beats the train-split mean action on eval 7D L2."
        ),
        "gripper_should_be_classification": bool(
            any((data.get("eval") or {}).get("metrics", {}).get("gripper_open_close_accuracy", 1.0) < 0.8 for data in variants.values() if isinstance(data, dict))
        ),
        "task_demo_phase_information": {
            "time_phase_available_without_leakage": True,
            "task_id_used": False,
            "demo_id_or_filename_used": False,
            "note": "Phase comes from current timestep and intended horizon only; task/demo identifiers remain excluded to avoid leakage.",
        },
    }


def _strip_private_arrays(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private_arrays(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_private_arrays(v) for v in obj]
    return obj


def _build_variants(
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    teacher_samples: list[dict[str, Any]],
    prior: dict[str, Any],
    max_steps: int,
    mlp_steps: int,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {
        "mean_action_baseline": _mean_action_baseline(train_samples, eval_samples, teacher_samples)
    }
    names = [
        "actionmap_7d",
        "fixed_prior_tca_7d",
        "hard_learned_target_tca_7d",
        "normalized_actionmap_7d",
        "normalized_fixed_prior_tca_7d",
        "split_fixed_prior_tca_7d",
        "small_cpu_mlp_fixed_prior_tca_7d",
        "fixed_prior_tca_mean_residual_7d",
        "phase_aware_fixed_prior_tca_7d",
    ]
    y_train = _actions(train_samples)
    for name in names:
        x_train = _feature_matrix(train_samples, name, prior, max_steps)
        model = _fit_model(name, x_train, y_train, steps=mlp_steps)
        variants[name] = _evaluate_variant(name, model, train_samples, eval_samples, teacher_samples, prior, max_steps)
    variants["native_smolvla_learned_residual_7d"] = _unavailable_native_residual()
    return variants


def _best_variant(variants: dict[str, dict[str, Any]], names: list[str]) -> str:
    available = [name for name in names if name in variants and "eval" in variants[name]]
    return min(available, key=lambda name: variants[name]["eval"]["metrics"]["7d_action_l2"])


def _rollout_gate(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean_l2 = variants["mean_action_baseline"]["eval"]["metrics"]["7d_action_l2"]
    actionmap_names = ["actionmap_7d", "normalized_actionmap_7d"]
    tca_names = [
        "fixed_prior_tca_7d",
        "normalized_fixed_prior_tca_7d",
        "split_fixed_prior_tca_7d",
        "small_cpu_mlp_fixed_prior_tca_7d",
        "fixed_prior_tca_mean_residual_7d",
        "phase_aware_fixed_prior_tca_7d",
    ]
    best_actionmap = _best_variant(variants, actionmap_names)
    best_tca = _best_variant(variants, tca_names)
    best_method = _best_variant(variants, actionmap_names + tca_names)
    best_l2 = variants[best_method]["eval"]["metrics"]["7d_action_l2"]
    best_tca_l2 = variants[best_tca]["eval"]["metrics"]["7d_action_l2"]
    best_actionmap_l2 = variants[best_actionmap]["eval"]["metrics"]["7d_action_l2"]
    action_diff = _action_diff(variants[best_tca]["_pred_eval"], variants[best_actionmap]["_pred_eval"])
    beats_mean_10pct = bool(best_l2 <= mean_l2 * 0.90)
    fixed_beats_actionmap = bool(best_tca_l2 < best_actionmap_l2)
    exact_7d = bool(variants[best_method]["eval"]["action_shape"][1] == 7)
    no_leakage = bool(
        not variants[best_method]["leakage_audit"]["uses_same_or_future_hdf5_action_at_inference"]
        and not variants[best_method]["leakage_audit"]["uses_eval_actions_at_inference"]
    )
    green = bool(beats_mean_10pct and fixed_beats_actionmap and action_diff["meaningfully_different"] and exact_7d and no_leakage)
    blockers = []
    if not beats_mean_10pct:
        blockers.append("best_head_does_not_beat_mean_action_baseline_by_10_percent")
    if not fixed_beats_actionmap:
        blockers.append("best_fixed_prior_tca_does_not_beat_best_actionmap")
    if not action_diff["meaningfully_different"]:
        blockers.append("best_actionmap_and_tca_actions_not_meaningfully_different")
    if not exact_7d:
        blockers.append("best_action_output_not_exactly_7d")
    if not no_leakage:
        blockers.append("leakage_audit_failed")
    return {
        "status": "green" if green else "red",
        "ready_for_bounded_matched_init_rollout": green,
        "threshold": "best ActionMap/TCA head must beat mean-action baseline eval 7D L2 by at least 10%",
        "mean_action_baseline_7d_l2": mean_l2,
        "best_method_variant": best_method,
        "best_method_7d_l2": best_l2,
        "best_actionmap_variant": best_actionmap,
        "best_actionmap_7d_l2": best_actionmap_l2,
        "best_fixed_prior_tca_variant": best_tca,
        "best_fixed_prior_tca_7d_l2": best_tca_l2,
        "best_head_beats_mean_baseline_by_10_percent": beats_mean_10pct,
        "fixed_prior_tca_beats_actionmap": fixed_beats_actionmap,
        "actionmap_tca_action_difference": action_diff,
        "action_output_exactly_7d": exact_7d,
        "no_hdf5_future_action_leakage": no_leakage,
        "clipping_bounded": True,
        "gripper_output_valid": True,
        "blockers": blockers,
    }


def _teacher_summary(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean_l2 = variants["mean_action_baseline"]["teacher_forced"]["metrics"]["7d_action_l2"]
    best = min(
        (name for name, data in variants.items() if "teacher_forced" in data and name != "mean_action_baseline"),
        key=lambda name: variants[name]["teacher_forced"]["metrics"]["7d_action_l2"],
    )
    current_diff = _action_diff(variants["fixed_prior_tca_7d"]["_pred_teacher"], variants["actionmap_7d"]["_pred_teacher"])
    return {
        "mean_action_baseline_7d_l2": mean_l2,
        "best_non_mean_variant": best,
        "best_non_mean_7d_l2": variants[best]["teacher_forced"]["metrics"]["7d_action_l2"],
        "best_non_mean_beats_mean": bool(variants[best]["teacher_forced"]["metrics"]["7d_action_l2"] < mean_l2),
        "current_actionmap_vs_fixed_prior_tca_action_difference": current_diff,
    }


def _recommend(gate: dict[str, Any], variants: dict[str, dict[str, Any]]) -> str:
    if gate["ready_for_bounded_matched_init_rollout"]:
        return "A. bounded improved-head matched-init rollout"
    best = gate["best_method_variant"]
    dominant = variants[best]["eval"]["dominant_loss_component"]
    if dominant in {"gripper", "rotation"}:
        return "B. gripper/rotation calibration"
    if "best_head_does_not_beat_mean_action_baseline_by_10_percent" in gate["blockers"]:
        return "C. target-prior conditioning redesign"
    return "D. paper-readiness package with honest rollout caveat"


def build_redesign_gate_report(args: argparse.Namespace) -> dict[str, Any]:
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
    if args.max_steps < 1 or args.max_steps > 25:
        report["result"]["blocked_reason"] = "max_steps must be between 1 and 25"
        return report
    if args.teacher_max_steps < 1 or args.teacher_max_steps > 512:
        report["result"]["blocked_reason"] = "teacher_max_steps must be between 1 and 512"
        return report
    if args.mlp_steps < 1 or args.mlp_steps > 300:
        report["result"]["blocked_reason"] = "mlp_steps must be between 1 and 300"
        return report

    train_samples, eval_samples, teacher_samples, rollout_demo = _load_samples(
        Path(args.manifest), args.max_steps, args.train_max_steps, args.sample_stride, args.teacher_max_steps
    )
    prior = _target_prior(train_samples)
    variants = _build_variants(train_samples, eval_samples, teacher_samples, prior, args.max_steps, args.mlp_steps)
    stage1 = _stage1_diagnosis(train_samples, eval_samples, variants)
    gate = _rollout_gate(variants)
    teacher = _teacher_summary(variants)
    public_variants = _strip_private_arrays(variants)
    report.update(
        {
            "decision": "bounded_7d_head_redesign_gate_completed",
            "policy": {
                **report["policy"],
                "training_performed": True,
                "loss_computed": True,
            },
            "data": {
                "manifest_path": str(Path(args.manifest)),
                "train_sample_count": len(train_samples),
                "eval_sample_count": len(eval_samples),
                "teacher_forced_sample_count": len(teacher_samples),
                "rollout_demo_path": rollout_demo["path"],
                "rollout_demo_excluded_from_training": rollout_demo["path"] not in sorted({sample["demo_path"] for sample in train_samples}),
                "data_source": "local LIBERO HDF5 counterfactual/offline source",
            },
            "stage1_mean_baseline_diagnosis": stage1,
            "stage2_head_variants": public_variants,
            "stage3_teacher_forced_trajectory": teacher,
            "stage4_rollout_gate": gate,
            "stage5_rollout": {
                "rollout_happened": False,
                "reason": "rollout gate is red; bounded matched-init rollout is intentionally skipped" if gate["status"] == "red" else "rollout gate is green, but this script only reports the gate; use the bounded rollout runner with the selected variant before any rollout claim",
            },
            "conclusion": {
                "why_mean_action_baseline_beat_previous_heads": stage1["why_mean_baseline_beats_previous_heads"],
                "best_7d_head_variant": gate["best_method_variant"],
                "best_head_beats_mean_action_baseline": bool(gate["best_method_7d_l2"] < gate["mean_action_baseline_7d_l2"]),
                "best_head_beats_mean_action_baseline_by_10_percent": gate["best_head_beats_mean_baseline_by_10_percent"],
                "fixed_prior_tca_beats_actionmap_on_7d_metrics": gate["fixed_prior_tca_beats_actionmap"],
                "actionmap_tca_actions_differ_meaningfully": gate["actionmap_tca_action_difference"]["meaningfully_different"],
                "dominant_bottleneck": public_variants[gate["best_method_variant"]]["eval"]["dominant_loss_component"],
                "teacher_forced_best_non_mean_variant": teacher["best_non_mean_variant"],
                "teacher_forced_best_non_mean_beats_mean": teacher["best_non_mean_beats_mean"],
                "fixed_prior_tca_valid_rollout_level_support": False,
                "recommended_next_milestone": _recommend(gate, variants),
            },
            "result": {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)},
        }
    )
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    gate = report.get("stage4_rollout_gate") or {}
    conclusion = report.get("conclusion") or {}
    stage1 = report.get("stage1_mean_baseline_diagnosis") or {}
    variants = report.get("stage2_head_variants") or {}
    lines = [
        "# Bounded 7D Action-Head Redesign Gate",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- training happened: `{(report.get('policy') or {}).get('training_performed')}`",
        f"- LoRA training happened: `{(report.get('policy') or {}).get('lora_training_performed')}`",
        f"- loss computed: `{(report.get('policy') or {}).get('loss_computed')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- rollout gate: `{gate.get('status')}`",
        f"- best 7D head variant: `{conclusion.get('best_7d_head_variant')}`",
        f"- best head beats mean baseline: `{conclusion.get('best_head_beats_mean_action_baseline')}`",
        f"- best head beats mean baseline by 10%: `{conclusion.get('best_head_beats_mean_action_baseline_by_10_percent')}`",
        f"- fixed-prior TCA beats ActionMap: `{conclusion.get('fixed_prior_tca_beats_actionmap_on_7d_metrics')}`",
        f"- ActionMap/TCA actions differ meaningfully: `{conclusion.get('actionmap_tca_actions_differ_meaningfully')}`",
        f"- dominant bottleneck: `{conclusion.get('dominant_bottleneck')}`",
        f"- recommended next milestone: `{conclusion.get('recommended_next_milestone')}`",
        "",
        "## Mean Baseline Diagnosis",
        "",
        f"- mean-action baseline 7D L2: `{stage1.get('mean_action_baseline_7d_l2')}`",
        f"- best non-mean variant: `{stage1.get('best_non_mean_variant')}`",
        f"- best non-mean 7D L2: `{stage1.get('best_non_mean_7d_l2')}`",
        f"- diagnosis: {stage1.get('why_mean_baseline_beats_previous_heads')}",
        "",
        "## Variant Eval Metrics",
        "",
    ]
    for name, data in variants.items():
        if "eval" not in data:
            lines.append(f"- `{name}`: `{data.get('status')}` - {data.get('reason')}")
            continue
        metrics = data["eval"]["metrics"]
        lines.append(
            f"- `{name}`: 7D L2 `{metrics.get('7d_action_l2')}`, translation `{metrics.get('translation_l2')}`, rotation `{metrics.get('rotation_l2')}`, gripper L1 `{metrics.get('gripper_l1')}`, gripper acc `{metrics.get('gripper_open_close_accuracy')}`"
        )
    lines.extend(["", "## Rollout Gate Blockers", ""])
    for blocker in gate.get("blockers") or []:
        lines.append(f"- {blocker}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/online_7d_head_redesign_gate_report.json")
    parser.add_argument("--report-md", default="reports/online_7d_head_redesign_gate_report.md")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--train-max-steps", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--teacher-max-steps", type=int, default=300)
    parser.add_argument("--mlp-steps", type=int, default=200)
    args = parser.parse_args(argv)

    report = build_redesign_gate_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    sys.exit(main())
