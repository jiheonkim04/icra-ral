"""Scaled fixed-prior ActionMap/TCA-Map offline proxy comparison.

This runner executes the smallest larger deterministic LIBERO HDF5 split
available after the fixed 8-sample diagnostics. It trains only lightweight
CPU NumPy heads and LoRA matrices. It does not load VLA models, use GPU,
run simulators, run rollouts, download assets, execute OpenVLA-OFT, or make
paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.adapters.tiny_lora_smoke import (
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_RUNTIME_SECONDS,
    DEFAULT_MAX_STEPS,
    TinyLoraSmokeError,
    _candidate_count,
    _expert_actions,
    _feature_matrix,
    _lora_param_count,
    _metric_records,
    _one_hot,
    _predict,
    _softmax,
    _target_ids,
    _train_lora_classifier,
    _train_lora_regressor,
    ensure_safe_environment,
    validate_smoke_bounds,
)
from tca_map.datasets.libero_fixed_prior_lora_attribution import (
    _fixed_fusion_probs,
    _instruction_text_probs,
    _select_ablation,
    _select_from_target_probs,
)
from tca_map.datasets.libero_offline_lora_comparison import (
    ACTION_PREFIX_DIM,
    _augment_metrics,
    _combined_losses,
    _loss_curve,
    _split_records,
    build_libero_lora_records,
)


SCHEMA_VERSION = "2026-07-05.fixed_prior_offline_scale_comparison.v1"
SCALE_SAMPLE_CHOICES = (16, 24, 32, 64)
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
)


def _dangerous_gates() -> list[str]:
    return [name for name in FORBIDDEN_GATES if os.environ.get(name)]


def _prior_source_audit(target_prior_variant: str) -> dict[str, Any]:
    """Describe inference-time target-prior information sources.

    The learned component is trained with train-split target labels, but eval
    labels are never used for non-oracle priors.
    """
    base = {
        "target_prior_variant": target_prior_variant,
        "uses_only_natural_language_instruction_text": False,
        "uses_bddl_metadata": False,
        "uses_dataset_target_labels": False,
        "uses_eval_labels": False,
        "available_at_test_time": True,
        "training_uses_dataset_target_labels": False,
        "note": "",
    }
    if target_prior_variant == "none_actionmap_baseline":
        return {
            **base,
            "uses_only_natural_language_instruction_text": True,
            "note": "ActionMap baseline uses instruction-derived features but no target prior.",
        }
    if target_prior_variant == "hard_learned_target":
        return {
            **base,
            "uses_only_natural_language_instruction_text": True,
            "uses_dataset_target_labels": True,
            "training_uses_dataset_target_labels": True,
            "note": "Inference uses learned target logits from instruction-derived features; train-split target labels train the target head.",
        }
    if target_prior_variant in {"fixed_learned_text_fusion", "fixed_learned_text_fusion_select_ablation"}:
        return {
            **base,
            "uses_dataset_target_labels": True,
            "training_uses_dataset_target_labels": True,
            "note": "Fusion uses instruction-text target prior plus learned target logits; eval labels and BDDL metadata are not used at inference.",
        }
    if target_prior_variant == "oracle_target_upper_bound":
        return {
            **base,
            "uses_dataset_target_labels": True,
            "uses_eval_labels": True,
            "available_at_test_time": False,
            "note": "Oracle upper bound uses the ground-truth target label and is not a valid method result.",
        }
    return base


def _select_scaled_sample_count(available_records: int, requested: int | None = None) -> int:
    if requested is not None:
        if requested <= 8:
            raise TinyLoraSmokeError("scaled comparison must use more than the fixed 8-sample split")
        if requested > available_records:
            raise TinyLoraSmokeError(f"requested {requested} samples but only {available_records} are available")
        return requested
    feasible = [count for count in SCALE_SAMPLE_CHOICES if count <= available_records and count > 8]
    if not feasible:
        raise TinyLoraSmokeError("no deterministic expanded split is available beyond 8 samples")
    return feasible[0]


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def _training_order(count: int, steps: int, seed: int | None) -> list[int]:
    if seed is None:
        return [step % count for step in range(steps)]
    rng = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < steps:
        order.extend(int(index) for index in rng.permutation(count).tolist())
    return order[:steps]


def _regression_loss(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    pred = _with_bias(features) @ weights
    return float(np.mean((pred - targets) ** 2))


def _classifier_loss(features: np.ndarray, target_ids: np.ndarray, weights: np.ndarray, num_targets: int) -> float:
    logits = _with_bias(features) @ weights
    probs = _softmax(logits)
    labels = _one_hot(target_ids, num_targets)
    return float(-np.sum(labels * np.log(probs + 1e-12)) / labels.shape[0])


def _train_linear_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    weights = np.zeros((x.shape[1], targets.shape[1]), dtype=np.float64)
    losses = [_regression_loss(features, targets, weights)]
    for index in _training_order(targets.shape[0], steps, seed):
        xi = x[index : index + 1]
        yi = targets[index : index + 1]
        diff = xi @ weights - yi
        grad = (2.0 / max(1, targets.shape[1])) * (xi.T @ diff)
        weights -= lr * grad
        losses.append(_regression_loss(features, targets, weights))
    return weights, losses


def _train_linear_classifier(
    features: np.ndarray,
    target_ids: np.ndarray,
    *,
    num_targets: int,
    steps: int,
    lr: float,
    seed: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    labels = _one_hot(target_ids, num_targets)
    weights = np.zeros((x.shape[1], num_targets), dtype=np.float64)
    losses = [_classifier_loss(features, target_ids, weights, num_targets)]
    for index in _training_order(target_ids.shape[0], steps, seed):
        xi = x[index : index + 1]
        yi = labels[index : index + 1]
        probs = _softmax(xi @ weights)
        weights -= lr * (xi.T @ (probs - yi))
        losses.append(_classifier_loss(features, target_ids, weights, num_targets))
    return weights, losses


def _linear_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(_with_bias(features) @ weights, -1.0, 1.0)


def _linear_logits(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _with_bias(features) @ weights


def _conditioned_features(features: np.ndarray, target_ids: np.ndarray, num_targets: int) -> np.ndarray:
    return np.concatenate([features, _one_hot(target_ids, num_targets)], axis=1)


def _metrics_for(records: list[dict[str, Any]], actions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    return _augment_metrics(records, _metric_records(records, actions, targets, grid_size=8))


def _target_topk_contains(target_probs: np.ndarray, records: list[dict[str, Any]], k: int = 2) -> float:
    probs = np.asarray(target_probs, dtype=np.float64)
    target_ids = _target_ids(records)
    if len(target_ids) == 0:
        return 0.0
    topk = np.argsort(-probs, axis=1)[:, : min(k, probs.shape[1])]
    return round(float(np.mean([int(target) in row.tolist() for target, row in zip(target_ids, topk)])), 6)


def _add_gap(metrics: dict[str, Any], oracle_score: float, suffix: str) -> dict[str, Any]:
    out = dict(metrics)
    out[f"gap_to_oracle_target_tca_{suffix}_standard_proxy"] = round(
        float(oracle_score) - float(out["standard_proxy_score"]),
        6,
    )
    return out


def _train_actionmap_head(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    seed: int | None = None,
) -> dict[str, Any]:
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    weights, losses = _train_linear_regressor(train_features, _expert_actions(train_records), steps=steps, lr=lr, seed=seed)
    return {
        "weights": weights,
        "losses": losses,
        "train_actions": _linear_predict(train_features, weights),
        "eval_actions": _linear_predict(eval_features, weights),
        "train_targets": np.zeros(len(train_records), dtype=np.int64),
        "eval_targets": np.zeros(len(eval_records), dtype=np.int64),
        "trainable_params": int(weights.size),
    }


def _train_tca_head(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    seed: int | None = None,
) -> dict[str, Any]:
    all_records = train_records + eval_records
    num_targets = _candidate_count(all_records)
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    train_targets = _target_ids(train_records)
    target_weights, target_losses = _train_linear_classifier(
        train_features,
        train_targets,
        num_targets=num_targets,
        steps=steps,
        lr=lr,
        seed=None if seed is None else seed + 101,
    )
    train_logits = _linear_logits(train_features, target_weights)
    eval_logits = _linear_logits(eval_features, target_weights)
    conditioned = _conditioned_features(train_features, train_targets, num_targets)
    action_weights, action_losses = _train_linear_regressor(
        conditioned,
        _expert_actions(train_records),
        steps=steps,
        lr=lr,
        seed=None if seed is None else seed + 202,
    )
    return {
        "num_targets": int(num_targets),
        "target_weights": target_weights,
        "target_losses": target_losses,
        "train_logits": train_logits,
        "eval_logits": eval_logits,
        "action_weights": action_weights,
        "action_losses": action_losses,
        "trainable_params": int(target_weights.size + action_weights.size),
    }


def _select_head_from_probs(
    features: np.ndarray,
    action_weights: np.ndarray,
    target_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probs = target_probs / np.maximum(target_probs.sum(axis=1, keepdims=True), 1e-12)
    selected_targets = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    actions = _linear_predict(_conditioned_features(features, selected_targets, probs.shape[1]), action_weights)
    diagnostics = [
        {
            "target_probs": [round(float(value), 6) for value in row.tolist()],
            "selected_target": int(target_id),
        }
        for row, target_id in zip(probs, selected_targets)
    ]
    return actions, selected_targets, diagnostics


def _head_arm(
    *,
    arm: str,
    target_prior_variant: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_actions: np.ndarray,
    train_targets: np.ndarray,
    eval_actions: np.ndarray,
    eval_targets: np.ndarray,
    losses: list[float],
    action_losses: list[float],
    target_losses: list[float],
    trainable_params: int,
    target_probs: np.ndarray | None = None,
    oracle: bool = False,
) -> dict[str, Any]:
    train_metric_records = _metric_records(train_records, train_actions, train_targets, grid_size=8)
    eval_metric_records = _metric_records(eval_records, eval_actions, eval_targets, grid_size=8)
    metrics = _augment_metrics(eval_records, eval_metric_records)
    if target_probs is not None:
        metrics["target_topk_contains_correct"] = _target_topk_contains(target_probs, eval_records, k=2)
    return {
        "arm": arm,
        "family": "head_only",
        "target_prior_variant": target_prior_variant,
        "prior_source_audit": _prior_source_audit(target_prior_variant),
        "oracle": bool(oracle),
        "tca_select_ablation": False,
        "training_performed": True,
        "lora_training_performed": False,
        "initial_loss": round(float(losses[0]), 6),
        "final_loss": round(float(losses[-1]), 6),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "loss_curve": _loss_curve(losses),
        "action_loss_curve": _loss_curve(action_losses),
        "target_loss_curve": _loss_curve(target_losses) if target_losses else [],
        "trainable_parameter_count": int(trainable_params),
        "batch_size": 1,
        "steps": len(losses) - 1,
        "train_metrics": _augment_metrics(train_records, train_metric_records),
        "evaluation_metrics": metrics,
        "eval_metric_records": eval_metric_records,
    }


def _train_actionmap_lora(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    rank: int,
    seed: int = 53,
) -> dict[str, Any]:
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    base, a, b, losses = _train_lora_regressor(
        features=train_features,
        targets=_expert_actions(train_records),
        max_steps=steps,
        lr=lr,
        rank=rank,
        seed=seed,
    )
    return {
        "base": base,
        "a": a,
        "b": b,
        "losses": losses,
        "train_actions": np.clip(_predict(train_features, base, a, b), -1.0, 1.0),
        "eval_actions": np.clip(_predict(eval_features, base, a, b), -1.0, 1.0),
        "train_targets": np.zeros(len(train_records), dtype=np.int64),
        "eval_targets": np.zeros(len(eval_records), dtype=np.int64),
        "trainable_params": int(_lora_param_count(a, b)),
        "frozen_params": int(base.size),
    }


def _train_tca_lora(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    rank: int,
    seed: int = 37,
) -> dict[str, Any]:
    all_records = train_records + eval_records
    num_targets = _candidate_count(all_records)
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    train_targets = _target_ids(train_records)
    target_base, target_a, target_b, target_losses = _train_lora_classifier(
        features=train_features,
        target_ids=train_targets,
        num_targets=num_targets,
        max_steps=steps,
        lr=lr,
        rank=rank,
        seed=seed,
    )
    train_logits = _predict(train_features, target_base, target_a, target_b)
    eval_logits = _predict(eval_features, target_base, target_a, target_b)
    action_base, action_a, action_b, action_losses = _train_lora_regressor(
        features=_conditioned_features(train_features, train_targets, num_targets),
        targets=_expert_actions(train_records),
        max_steps=steps,
        lr=lr,
        rank=rank,
        seed=seed + 16,
    )
    return {
        "num_targets": int(num_targets),
        "target_base": target_base,
        "target_a": target_a,
        "target_b": target_b,
        "target_losses": target_losses,
        "train_logits": train_logits,
        "eval_logits": eval_logits,
        "action_base": action_base,
        "action_a": action_a,
        "action_b": action_b,
        "action_losses": action_losses,
        "trainable_params": int(_lora_param_count(target_a, target_b) + _lora_param_count(action_a, action_b)),
        "frozen_params": int(target_base.size + action_base.size),
    }


def _lora_arm(
    *,
    arm: str,
    target_prior_variant: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_actions: np.ndarray,
    train_targets: np.ndarray,
    eval_actions: np.ndarray,
    eval_targets: np.ndarray,
    losses: list[float],
    action_losses: list[float],
    target_losses: list[float],
    trainable_params: int,
    frozen_params: int,
    rank: int,
    target_probs: np.ndarray | None = None,
    oracle: bool = False,
    select_ablation: bool = False,
    selection_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    train_metric_records = _metric_records(train_records, train_actions, train_targets, grid_size=8)
    eval_metric_records = _metric_records(eval_records, eval_actions, eval_targets, grid_size=8)
    metrics = _augment_metrics(eval_records, eval_metric_records)
    if target_probs is not None:
        metrics["target_topk_contains_correct"] = _target_topk_contains(target_probs, eval_records, k=2)
    return {
        "arm": arm,
        "family": "lora",
        "target_prior_variant": target_prior_variant,
        "prior_source_audit": _prior_source_audit(target_prior_variant),
        "oracle": bool(oracle),
        "tca_select_ablation": bool(select_ablation),
        "lora_target_modules": ["action_head_projection"]
        if arm == "actionmap_lora"
        else ["target_fusion_layers", "target_classifier", "action_head_projection"],
        "training_performed": True,
        "lora_training_performed": True,
        "initial_loss": round(float(losses[0]), 6),
        "final_loss": round(float(losses[-1]), 6),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "loss_curve": _loss_curve(losses),
        "action_loss_curve": _loss_curve(action_losses),
        "target_loss_curve": _loss_curve(target_losses) if target_losses else [],
        "trainable_lora_parameter_count": int(trainable_params),
        "frozen_base_parameter_count": int(frozen_params),
        "lora_rank": int(rank),
        "batch_size": 1,
        "steps": len(losses) - 1,
        "finite_losses": all(math.isfinite(loss) for loss in action_losses + target_losses),
        "train_metrics": _augment_metrics(train_records, train_metric_records),
        "evaluation_metrics": metrics,
        "eval_metric_records": eval_metric_records,
        "selection_diagnostics": selection_diagnostics or [],
    }


def _build_head_arms(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    actionmap = _train_actionmap_head(train_records, eval_records, steps, lr, seed=None if seed is None else seed + 11)
    tca = _train_tca_head(train_records, eval_records, steps, lr, seed=None if seed is None else seed + 29)
    learned_train_probs = _softmax(tca["train_logits"])
    learned_eval_probs = _softmax(tca["eval_logits"])
    text_train_probs = _instruction_text_probs(train_records, tca["num_targets"])
    text_eval_probs = _instruction_text_probs(eval_records, tca["num_targets"])
    fixed_train_probs, _ = _fixed_fusion_probs(tca["train_logits"], learned_train_probs, text_train_probs)
    fixed_eval_probs, _ = _fixed_fusion_probs(tca["eval_logits"], learned_eval_probs, text_eval_probs)
    oracle_train_probs = _one_hot(_target_ids(train_records), tca["num_targets"])
    oracle_eval_probs = _one_hot(_target_ids(eval_records), tca["num_targets"])

    def tca_variant(name: str, prior: str, train_probs: np.ndarray, eval_probs: np.ndarray, oracle: bool = False) -> dict[str, Any]:
        train_actions, train_targets, _ = _select_head_from_probs(train_features, tca["action_weights"], train_probs)
        eval_actions, eval_targets, _ = _select_head_from_probs(eval_features, tca["action_weights"], eval_probs)
        losses = _combined_losses(tca["action_losses"], tca["target_losses"])
        return _head_arm(
            arm=name,
            target_prior_variant=prior,
            train_records=train_records,
            eval_records=eval_records,
            train_actions=train_actions,
            train_targets=train_targets,
            eval_actions=eval_actions,
            eval_targets=eval_targets,
            losses=losses,
            action_losses=tca["action_losses"],
            target_losses=tca["target_losses"],
            trainable_params=tca["trainable_params"],
            target_probs=eval_probs,
            oracle=oracle,
        )

    return [
        _head_arm(
            arm="actionmap_head_only",
            target_prior_variant="none_actionmap_baseline",
            train_records=train_records,
            eval_records=eval_records,
            train_actions=actionmap["train_actions"],
            train_targets=actionmap["train_targets"],
            eval_actions=actionmap["eval_actions"],
            eval_targets=actionmap["eval_targets"],
            losses=actionmap["losses"],
            action_losses=actionmap["losses"],
            target_losses=[],
            trainable_params=actionmap["trainable_params"],
        ),
        tca_variant("tca_map_hard_learned_target_head_only", "hard_learned_target", learned_train_probs, learned_eval_probs),
        tca_variant("tca_map_fixed_learned_text_fusion_head_only", "fixed_learned_text_fusion", fixed_train_probs, fixed_eval_probs),
        tca_variant("oracle_target_tca_head_only_upper_bound", "oracle_target_upper_bound", oracle_train_probs, oracle_eval_probs, True),
    ]


def _build_lora_arms(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    rank: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    actionmap = _train_actionmap_lora(train_records, eval_records, steps, lr, rank, seed=seed + 53)
    tca = _train_tca_lora(train_records, eval_records, steps, lr, rank, seed=seed + 37)
    learned_train_probs = _softmax(tca["train_logits"])
    learned_eval_probs = _softmax(tca["eval_logits"])
    text_train_probs = _instruction_text_probs(train_records, tca["num_targets"])
    text_eval_probs = _instruction_text_probs(eval_records, tca["num_targets"])
    fixed_train_probs, _ = _fixed_fusion_probs(tca["train_logits"], learned_train_probs, text_train_probs)
    fixed_eval_probs, _ = _fixed_fusion_probs(tca["eval_logits"], learned_eval_probs, text_eval_probs)
    oracle_train_probs = _one_hot(_target_ids(train_records), tca["num_targets"])
    oracle_eval_probs = _one_hot(_target_ids(eval_records), tca["num_targets"])
    losses = _combined_losses(tca["action_losses"], tca["target_losses"])

    def tca_variant(name: str, prior: str, train_probs: np.ndarray, eval_probs: np.ndarray, oracle: bool = False, select: bool = False) -> dict[str, Any]:
        if select:
            eval_actions, eval_targets, selection = _select_ablation(
                eval_records,
                eval_features,
                tca["action_base"],
                tca["action_a"],
                tca["action_b"],
                eval_probs,
            )
            train_actions, train_targets, _ = _select_from_target_probs(
                train_features,
                tca["action_base"],
                tca["action_a"],
                tca["action_b"],
                train_probs,
            )
        else:
            train_actions, train_targets, _ = _select_from_target_probs(
                train_features,
                tca["action_base"],
                tca["action_a"],
                tca["action_b"],
                train_probs,
            )
            eval_actions, eval_targets, selection = _select_from_target_probs(
                eval_features,
                tca["action_base"],
                tca["action_a"],
                tca["action_b"],
                eval_probs,
            )
        return _lora_arm(
            arm=name,
            target_prior_variant=prior,
            train_records=train_records,
            eval_records=eval_records,
            train_actions=train_actions,
            train_targets=train_targets,
            eval_actions=eval_actions,
            eval_targets=eval_targets,
            losses=losses,
            action_losses=tca["action_losses"],
            target_losses=tca["target_losses"],
            trainable_params=tca["trainable_params"],
            frozen_params=tca["frozen_params"],
            rank=rank,
            target_probs=eval_probs,
            oracle=oracle,
            select_ablation=select,
            selection_diagnostics=selection if select else [],
        )

    return [
        _lora_arm(
            arm="actionmap_lora",
            target_prior_variant="none_actionmap_baseline",
            train_records=train_records,
            eval_records=eval_records,
            train_actions=actionmap["train_actions"],
            train_targets=actionmap["train_targets"],
            eval_actions=actionmap["eval_actions"],
            eval_targets=actionmap["eval_targets"],
            losses=_combined_losses(actionmap["losses"], []),
            action_losses=actionmap["losses"],
            target_losses=[],
            trainable_params=actionmap["trainable_params"],
            frozen_params=actionmap["frozen_params"],
            rank=rank,
        ),
        tca_variant("tca_map_lora_hard_learned_target", "hard_learned_target", learned_train_probs, learned_eval_probs),
        tca_variant("tca_map_lora_fixed_learned_text_fusion", "fixed_learned_text_fusion", fixed_train_probs, fixed_eval_probs),
        tca_variant("oracle_target_tca_lora_upper_bound", "oracle_target_upper_bound", oracle_train_probs, oracle_eval_probs, True),
        tca_variant(
            "tca_map_lora_fixed_fusion_tca_select_ablation",
            "fixed_learned_text_fusion_select_ablation",
            fixed_train_probs,
            fixed_eval_probs,
            select=True,
        ),
    ]


def _target_balance(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(int(record["target"]["object_id"])) for record in records)
    return dict(sorted(counts.items()))


def _task_count(records: list[dict[str, Any]]) -> int:
    instructions = set()
    for record in records:
        instructions.add(str(record.get("target", {}).get("instruction", "")))
    return len(instructions)


def _task_record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("target", {}).get("instruction", "")) for record in records)
    return dict(sorted(counts.items()))


def _per_pair_breakdown(records: list[dict[str, Any]], arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_records = records
    for pair_id in sorted({record["pair_id"] for record in eval_records}):
        pair_indices = [index for index, record in enumerate(eval_records) if record["pair_id"] == pair_id]
        pair_records = [eval_records[index] for index in pair_indices]
        row: dict[str, Any] = {"pair_id": pair_id, "eval_sample_count": len(pair_indices)}
        for arm_name, arm in arms.items():
            metric_rows = [arm["eval_metric_records"][index] for index in pair_indices]
            metrics = _augment_metrics(pair_records, metric_rows)
            row[f"{arm_name}_standard_proxy_score"] = metrics.get("standard_proxy_score")
            row[f"{arm_name}_wrong_target_proxy_rate"] = metrics.get("wrong_target_proxy_rate")
        rows.append(row)
    return rows


def _add_oracle_gaps(arms: list[dict[str, Any]]) -> None:
    head_oracle = next(arm for arm in arms if arm["arm"] == "oracle_target_tca_head_only_upper_bound")
    lora_oracle = next(arm for arm in arms if arm["arm"] == "oracle_target_tca_lora_upper_bound")
    head_score = float(head_oracle["evaluation_metrics"]["standard_proxy_score"])
    lora_score = float(lora_oracle["evaluation_metrics"]["standard_proxy_score"])
    for arm in arms:
        suffix = "head_only" if arm["family"] == "head_only" else "lora"
        score = head_score if suffix == "head_only" else lora_score
        arm["evaluation_metrics"] = _add_gap(arm["evaluation_metrics"], score, suffix)


def _delta(arms: dict[str, dict[str, Any]], left: str, right: str, metric: str) -> float:
    return round(
        float(arms[left]["evaluation_metrics"][metric]) - float(arms[right]["evaluation_metrics"][metric]),
        6,
    )


def _comparison(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    head_delta = {
        "standard_proxy_score_delta": _delta(arms, "tca_map_fixed_learned_text_fusion_head_only", "actionmap_head_only", "standard_proxy_score"),
        "wrong_target_proxy_rate_delta": _delta(arms, "tca_map_fixed_learned_text_fusion_head_only", "actionmap_head_only", "wrong_target_proxy_rate"),
        "action_target_consistency_score_delta": _delta(arms, "tca_map_fixed_learned_text_fusion_head_only", "actionmap_head_only", "action_target_consistency_score"),
        "counterfactual_margin_delta": _delta(arms, "tca_map_fixed_learned_text_fusion_head_only", "actionmap_head_only", "counterfactual_separation_margin"),
    }
    lora_delta = {
        "standard_proxy_score_delta": _delta(arms, "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "standard_proxy_score"),
        "wrong_target_proxy_rate_delta": _delta(arms, "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "wrong_target_proxy_rate"),
        "action_target_consistency_score_delta": _delta(arms, "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "action_target_consistency_score"),
        "counterfactual_margin_delta": _delta(arms, "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "counterfactual_separation_margin"),
    }
    select_delta = {
        "standard_proxy_score_delta": _delta(arms, "tca_map_lora_fixed_fusion_tca_select_ablation", "tca_map_lora_fixed_learned_text_fusion", "standard_proxy_score"),
        "wrong_target_proxy_rate_delta": _delta(arms, "tca_map_lora_fixed_fusion_tca_select_ablation", "tca_map_lora_fixed_learned_text_fusion", "wrong_target_proxy_rate"),
        "action_target_consistency_score_delta": _delta(arms, "tca_map_lora_fixed_fusion_tca_select_ablation", "tca_map_lora_fixed_learned_text_fusion", "action_target_consistency_score"),
        "counterfactual_margin_delta": _delta(arms, "tca_map_lora_fixed_fusion_tca_select_ablation", "tca_map_lora_fixed_learned_text_fusion", "counterfactual_separation_margin"),
    }
    fixed_lora_beats = lora_delta["standard_proxy_score_delta"] > 0.0 and lora_delta["wrong_target_proxy_rate_delta"] <= 0.0
    fixed_head_beats = head_delta["standard_proxy_score_delta"] > 0.0 and head_delta["wrong_target_proxy_rate_delta"] <= 0.0
    select_helps = (
        select_delta["standard_proxy_score_delta"] >= 0.01
        or select_delta["wrong_target_proxy_rate_delta"] < 0.0
        or (
            select_delta["action_target_consistency_score_delta"] > 0.0
            and select_delta["standard_proxy_score_delta"] >= 0.0
        )
    )
    hard_lora_bad = (
        float(arms["tca_map_lora_hard_learned_target"]["evaluation_metrics"]["standard_proxy_score"])
        < float(arms["tca_map_lora_fixed_learned_text_fusion"]["evaluation_metrics"]["standard_proxy_score"])
    )
    if fixed_lora_beats:
        conclusion = "fixed_prior_tca_lora_advantage_survives_scaled_split"
        recommendation = "A_larger_offline_split_or_B_multi_seed_validation"
    elif float(arms["oracle_target_tca_lora_upper_bound"]["evaluation_metrics"]["standard_proxy_score"]) > float(
        arms["tca_map_lora_fixed_learned_text_fusion"]["evaluation_metrics"]["standard_proxy_score"]
    ):
        conclusion = "fixed_prior_fails_but_oracle_remains_strong"
        recommendation = "C_learned_target_head_redesign"
    else:
        conclusion = "fixed_prior_tca_advantage_disappears_on_scaled_split"
        recommendation = "C_learned_target_head_redesign"
    selector_recommendation = "E_deemphasize_or_kill_TCA_Select" if not select_helps else "keep_TCA_Select_as_secondary_ablation"
    return {
        "conclusion": conclusion,
        "recommended_next_milestone": recommendation,
        "selector_recommendation": selector_recommendation,
        "fixed_prior_tca_head_only_vs_actionmap_head_only": head_delta,
        "fixed_prior_tca_lora_vs_actionmap_lora": lora_delta,
        "tca_select_ablation_vs_fixed_prior_tca_lora": select_delta,
        "fixed_prior_tca_head_only_beats_actionmap_head_only": bool(fixed_head_beats),
        "fixed_prior_tca_lora_beats_actionmap_lora": bool(fixed_lora_beats),
        "hard_learned_target_remains_bottleneck": bool(hard_lora_bad),
        "tca_select_meaningful_gain": bool(select_helps),
    }


def _policy() -> dict[str, Any]:
    return {
        "scaled_fixed_prior_offline_proxy": True,
        "local_libero_hdf5_used": True,
        "real_dataset_used": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": True,
        "lora_training_performed": True,
        "full_finetuning_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Scaled Fixed-Prior Offline Comparison",
        "",
        "Exploratory offline proxy only. This is not standard success, rollout evidence, or paper-grade evidence.",
        "",
        f"- passed: `{report['fixed_prior_offline_scale_comparison_passed']}`",
        f"- conclusion: `{report['comparison']['conclusion']}`",
        f"- recommended next milestone: `{report['comparison']['recommended_next_milestone']}`",
        f"- selector recommendation: `{report['comparison']['selector_recommendation']}`",
        f"- records: `{report['record_count']}`",
        f"- train/eval records: `{report['train_record_count']}` / `{report['eval_record_count']}`",
        f"- task count: `{report['task_count']}`",
        f"- target balance: `{report['target_balance']}`",
        "",
        "## Arms",
    ]
    for arm in report["arms"]:
        metrics = arm["evaluation_metrics"]
        lines.extend(
            [
                f"### `{arm['arm']}`",
                f"- family: `{arm['family']}`",
                f"- target prior: `{arm['target_prior_variant']}`",
                f"- loss: `{arm['initial_loss']} -> {arm['final_loss']}`",
                f"- standard proxy: `{metrics['standard_proxy_score']}`",
                f"- wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- action-target consistency: `{metrics['action_target_consistency_score']}`",
                f"- counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
                "",
            ]
        )
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_fixed_prior_offline_scale_comparison(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    *,
    max_pairs: int = 8,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    max_samples: int | None = None,
    rank: int = DEFAULT_LORA_RANK,
    seed: int | None = None,
    require_training_gate: bool = True,
) -> dict[str, Any]:
    dangerous = _dangerous_gates()
    if dangerous:
        raise TinyLoraSmokeError("dangerous gates are set: " + ", ".join(dangerous))
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds, max_samples=max_samples or 64, rank=rank)
    if max_steps > 300:
        raise TinyLoraSmokeError("max_steps must not exceed 300")
    if max_pairs < 5 or max_pairs > 32:
        raise TinyLoraSmokeError("max_pairs must be between 5 and 32 for scaled comparison")

    started = time.perf_counter()
    available_records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)
    selected_count = _select_scaled_sample_count(len(available_records), max_samples)
    records = available_records[:selected_count]
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise TinyLoraSmokeError("deterministic scaled split did not produce train/eval records")
    lr = 0.05
    arms = _build_head_arms(train_records, eval_records, max_steps, lr, seed=seed)
    arms.extend(_build_lora_arms(train_records, eval_records, max_steps, lr, rank, seed=seed or 0))
    _add_oracle_gaps(arms)
    arm_map = {arm["arm"]: arm for arm in arms}
    comparison = _comparison(arm_map)
    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and all(arm["loss_decreased"] for arm in arms)
        and not any(
            item.get("external_verifier_used") or item.get("privileged_inference_used")
            for item in arm_map["tca_map_lora_fixed_fusion_tca_select_ablation"].get("selection_diagnostics", [])
        )
    )
    interpretation = (
        "Fixed-prior TCA + LoRA still beats ActionMap + LoRA on the scaled exploratory offline split. "
        "This supports cautious offline scaling, but it is not paper-grade and still depends on target-prior correctness."
        if comparison["fixed_prior_tca_lora_beats_actionmap_lora"]
        else "Fixed-prior TCA + LoRA does not beat ActionMap + LoRA on the scaled exploratory split. Diagnose target prior, action-conditioning, and metric stability before further scaling."
    )
    if not comparison["tca_select_meaningful_gain"]:
        interpretation += " TCA-Select again shows no meaningful gain and should be de-emphasized as a core contribution."
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(),
        "source_manifest": str(manifest_path),
        "prior_source_audit": {
            "scope": "inference-time target-prior source audit; training labels are called out separately",
            "variants": {
                variant: _prior_source_audit(variant)
                for variant in [
                    "none_actionmap_baseline",
                    "hard_learned_target",
                    "fixed_learned_text_fusion",
                    "fixed_learned_text_fusion_select_ablation",
                    "oracle_target_upper_bound",
                ]
            },
        },
        "sample_selection": {
            "available_record_count": len(available_records),
            "chosen_record_count": selected_count,
            "sample_count_choices": list(SCALE_SAMPLE_CHOICES),
            "selection_rule": "smallest available deterministic split larger than 8 samples",
            "max_pairs": max_pairs,
            "max_action_steps": max_action_steps,
        },
        "record_count": len(records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "task_count": _task_count(records),
        "per_task_record_counts": _task_record_counts(records),
        "train_per_task_record_counts": _task_record_counts(train_records),
        "eval_per_task_record_counts": _task_record_counts(eval_records),
        "target_class_count": _candidate_count(records),
        "target_balance": _target_balance(records),
        "train_target_balance": _target_balance(train_records),
        "eval_target_balance": _target_balance(eval_records),
        "split": split,
        "max_steps": max_steps,
        "batch_size": 1,
        "lora_rank": rank,
        "seed": seed,
        "seed_policy": "fixed split; seed controls only head-only SGD order and LoRA low-rank initialization",
        "action_prefix_dim": ACTION_PREFIX_DIM,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "arms": arms,
        "comparison": comparison,
        "per_task_breakdown": _per_pair_breakdown(eval_records, arm_map),
        "fixed_prior_offline_scale_comparison_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": interpretation,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_fixed_prior_offline_scale_comparison_report.json")
    parser.add_argument("--report-md", default="reports/libero_fixed_prior_offline_scale_comparison_report.md")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        report = run_fixed_prior_offline_scale_comparison(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            seed=args.seed,
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
