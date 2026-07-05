"""Tiny target-prior rescue diagnostic for TCA-Map on the fixed LIBERO split.

This execution-first diagnostic tests whether TCA-Map recovers when target
prediction is improved, softened, marginalized, or replaced by a stronger
instruction-derived prior. It uses the same tiny local LIBERO HDF5 split as the
head-only and LoRA diagnostics. It does not load VLA models, use GPU, run
rollouts, import simulators, download assets, execute OpenVLA-OFT, or make
paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_offline_head_comparison import (
    BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_RUNTIME_SECONDS,
    DEFAULT_MAX_STEPS,
    MAX_TRAINING_STEPS,
    TARGET_COUNT,
    _candidate_actions,
    _classifier_loss,
    _instruction_features,
    _metric_records,
    _one_hot,
    _pair_features,
    _predict_regressor,
    _predict_targets,
    _softmax,
    _target_ids,
    validate_bounds,
)
from tca_map.datasets.libero_tca_label_conditioning_audit import (
    TcaLabelConditioningAuditError,
    _l1,
    _metrics,
    _policy,
    _prepare_records,
    _predict_tca,
    _round_list,
    _train_models,
)

SCHEMA_VERSION = "tca-map-libero-tca-target-prior-rescue-v1"


class TcaTargetPriorRescueError(RuntimeError):
    """Raised when the bounded target-prior rescue diagnostic cannot run safely."""


def _safe_softmax(row: np.ndarray) -> np.ndarray:
    return _softmax(row.reshape(1, -1))[0]


def _topk_contains(target_probs: np.ndarray, target_ids: np.ndarray, k: int) -> list[bool]:
    topk = np.argsort(-target_probs, axis=1)[:, :k]
    return [int(target_ids[index]) in {int(item) for item in topk[index].tolist()} for index in range(len(target_ids))]


def _target_accuracy(pred_targets: np.ndarray, target_ids: np.ndarray) -> float:
    if target_ids.size == 0:
        return 0.0
    return float(np.mean(pred_targets == target_ids))


def _target_topk_accuracy(target_probs: np.ndarray, target_ids: np.ndarray, k: int) -> float:
    contains = _topk_contains(target_probs, target_ids, k)
    return float(np.mean(contains)) if contains else 0.0


def _confusion_table(pred_targets: np.ndarray, target_ids: np.ndarray) -> dict[str, int]:
    table: dict[str, int] = {}
    for truth, pred in zip(target_ids, pred_targets):
        key = f"true_{int(truth)}__pred_{int(pred)}"
        table[key] = table.get(key, 0) + 1
    return table


def _target_ce_loss(features: np.ndarray, target_ids: np.ndarray, target_weights: np.ndarray) -> float:
    return float(_classifier_loss(features, target_ids, target_weights))


def _target_head_training_sanity(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    models: dict[str, Any],
    steps: int,
) -> dict[str, Any]:
    train_targets = _target_ids(train_records)
    eval_targets = _target_ids(eval_records)
    train_pred, train_logits = _predict_targets(_instruction_features(train_records), models["target_weights"])
    eval_pred, eval_logits = _predict_targets(_instruction_features(eval_records), models["target_weights"])
    train_probs = _softmax(train_logits)
    eval_probs = _softmax(eval_logits)
    topk = min(2, TARGET_COUNT)
    return {
        "training": True,
        "data_source": "local LIBERO HDF5 action snippets from counterfactual split manifest",
        "samples": len(train_records) + len(eval_records),
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "trainable_parameter_count": int(models["target_weights"].size),
        "initial_target_ce_loss": round(float(models["target_losses"][0]), 6),
        "final_target_ce_loss": round(float(models["target_losses"][-1]), 6),
        "loss_decreased": bool(models["target_losses"][-1] < models["target_losses"][0]),
        "train_target_ce_loss": round(_target_ce_loss(_instruction_features(train_records), train_targets, models["target_weights"]), 6),
        "eval_target_ce_loss": round(_target_ce_loss(_instruction_features(eval_records), eval_targets, models["target_weights"]), 6),
        "train_target_top1_accuracy": round(_target_accuracy(train_pred, train_targets), 6),
        "eval_target_top1_accuracy": round(_target_accuracy(eval_pred, eval_targets), 6),
        "train_target_topk_accuracy": round(_target_topk_accuracy(train_probs, train_targets, topk), 6),
        "eval_target_topk_accuracy": round(_target_topk_accuracy(eval_probs, eval_targets, topk), 6),
        "topk": topk,
        "train_confusion_table": _confusion_table(train_pred, train_targets),
        "eval_confusion_table": _confusion_table(eval_pred, eval_targets),
    }


def _variant_metrics(records: list[dict[str, Any]], pred: dict[str, Any], grid_size: int) -> dict[str, Any]:
    metrics = _metrics(records, pred, grid_size)
    target_probs = pred.get("target_probs")
    target_ids = _target_ids(records)
    if target_probs is not None:
        probs = np.asarray(target_probs, dtype=np.float64)
        topk = min(2, probs.shape[1])
        metrics["target_topk_contains_correct"] = round(_target_topk_accuracy(probs, target_ids, topk), 6)
        metrics["target_topk_k"] = topk
    return metrics


def _action_target_scores(
    record: dict[str, Any],
    pair_feature: np.ndarray,
    action_weights: np.ndarray,
    target_probs: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id, action in enumerate(_candidate_actions(pair_feature, action_weights)):
        action_distance = _l1(action, record["candidate_actions"][target_id])
        probability = float(target_probs[target_id])
        rows.append(
            {
                "target_id": target_id,
                "action": action,
                "target_probability": probability,
                "action_target_score": probability * max(0.0, 1.0 - action_distance),
                "action_distance_to_candidate": action_distance,
            }
        )
    return rows


def _soft_marginalized_pred(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    pred_targets, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    probs = _softmax(logits)
    actions: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, target_probs in zip(records, _pair_features(records), probs):
        candidates = _action_target_scores(record, pair_feature, models["tca_action_weights"], target_probs)
        weighted = np.zeros(len(candidates[0]["action"]), dtype=np.float64)
        for candidate in candidates:
            weighted += float(candidate["target_probability"]) * np.asarray(candidate["action"], dtype=np.float64)
        actions.append([float(value) for value in weighted.tolist()])
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "target_probs": _round_list(target_probs),
                "top1_target": int(np.argmax(target_probs)),
                "correct_target_in_top2": bool(int(record["target_id"]) in np.argsort(-target_probs)[: min(2, TARGET_COUNT)].tolist()),
                "marginalized_candidate_scores": [
                    {
                        "target_id": int(item["target_id"]),
                        "target_probability": round(float(item["target_probability"]), 6),
                        "action_target_score": round(float(item["action_target_score"]), 6),
                    }
                    for item in candidates
                ],
            }
        )
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "targets": pred_targets,
        "logits": logits,
        "target_probs": probs,
        "diagnostics": diagnostics,
    }


def _soft_select_pred(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    _, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    probs = _softmax(logits)
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, target_probs, row_logits in zip(records, _pair_features(records), probs, logits):
        scored = _action_target_scores(record, pair_feature, models["tca_action_weights"], target_probs)
        selected = max(scored, key=lambda item: float(item["action_target_score"]))
        selected_actions.append([float(value) for value in selected["action"]])
        selected_targets.append(int(selected["target_id"]))
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "target_probs": _round_list(target_probs),
                "target_logits": _round_list(row_logits),
                "correct_target_in_top2": bool(int(record["target_id"]) in np.argsort(-target_probs)[: min(2, TARGET_COUNT)].tolist()),
                "selected_target": int(selected["target_id"]),
                "scores": [
                    {
                        "target_id": int(item["target_id"]),
                        "target_probability": round(float(item["target_probability"]), 6),
                        "action_target_score": round(float(item["action_target_score"]), 6),
                    }
                    for item in scored
                ],
            }
        )
    return {
        "actions": np.asarray(selected_actions, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "logits": logits,
        "target_probs": probs,
        "diagnostics": diagnostics,
    }


def _tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "both",
        "in",
        "it",
        "of",
        "on",
        "put",
        "the",
        "to",
    }
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token and token not in stop}


def _candidate_texts(record: dict[str, Any]) -> list[str]:
    pair_id = str(record["pair_id"])
    body = pair_id.split(":", 1)[-1]
    if "__vs__" not in body:
        return ["target 0", "target 1"]
    left, right = body.split("__vs__", 1)
    return [left.replace("_", " "), right.replace("_", " ")]


def _instruction_text_prior_pred(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    actions: list[list[float]] = []
    targets: list[int] = []
    probs: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature in zip(records, _pair_features(records)):
        instruction_tokens = _tokens(record["instruction"])
        candidate_texts = _candidate_texts(record)
        raw_scores = []
        for candidate_text in candidate_texts:
            candidate_tokens = _tokens(candidate_text)
            overlap = len(instruction_tokens & candidate_tokens)
            union = max(1, len(instruction_tokens | candidate_tokens))
            raw_scores.append(overlap / union)
        if len(set(raw_scores)) == 1:
            raw_scores = [score + 1e-3 * (len(raw_scores) - idx) for idx, score in enumerate(raw_scores)]
        target_probs = _safe_softmax(np.asarray(raw_scores, dtype=np.float64))
        target_id = int(np.argmax(target_probs))
        conditioned = np.concatenate([pair_feature, _one_hot(np.asarray([target_id]))[0]], axis=0).reshape(1, -1)
        action = _predict_regressor(conditioned, models["tca_action_weights"])[0]
        actions.append([float(value) for value in action.tolist()])
        targets.append(target_id)
        probs.append([float(value) for value in target_probs.tolist()])
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "instruction_tokens": sorted(instruction_tokens),
                "candidate_texts": candidate_texts,
                "raw_overlap_scores": _round_list(raw_scores),
                "target_probs": _round_list(target_probs),
                "selected_target": target_id,
                "metadata_oracle": False,
                "paper_grade": False,
            }
        )
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.int64),
        "target_probs": np.asarray(probs, dtype=np.float64),
        "diagnostics": diagnostics,
    }


def _oracle_pred(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    actions: list[list[float]] = []
    target_ids = _target_ids(records)
    for pair_feature, target_id in zip(_pair_features(records), target_ids):
        conditioned = np.concatenate([pair_feature, _one_hot(np.asarray([target_id]))[0]], axis=0).reshape(1, -1)
        action = _predict_regressor(conditioned, models["tca_action_weights"])[0]
        actions.append([float(value) for value in action.tolist()])
    probs = _one_hot(target_ids)
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "targets": target_ids,
        "target_probs": probs,
        "diagnostics": [
            {
                "sample_id": record["sample_id"],
                "oracle_target_id": int(target_id),
                "oracle": True,
                "paper_grade": False,
            }
            for record, target_id in zip(records, target_ids)
        ],
    }


def _constant_prior_pred(records: list[dict[str, Any]], models: dict[str, Any], constant_target: int = 0) -> dict[str, Any]:
    targets = np.full(len(records), int(constant_target), dtype=np.int64)
    actions: list[list[float]] = []
    for pair_feature in _pair_features(records):
        conditioned = np.concatenate([pair_feature, _one_hot(np.asarray([constant_target]))[0]], axis=0).reshape(1, -1)
        action = _predict_regressor(conditioned, models["tca_action_weights"])[0]
        actions.append([float(value) for value in action.tolist()])
    probs = _one_hot(targets)
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "targets": targets,
        "target_probs": probs,
        "diagnostics": [{"sample_id": record["sample_id"], "constant_target": int(constant_target)} for record in records],
    }


def _learned_hard_pred(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    pred = _predict_tca(records, models)
    logits = np.asarray(pred["logits"], dtype=np.float64)
    pred["target_probs"] = _softmax(logits)
    pred["diagnostics"] = [
        {
            "sample_id": record["sample_id"],
            "target_logits": _round_list(row_logits),
            "target_probs": _round_list(row_probs),
            "selected_target": int(target_id),
            "hard_top1_target": True,
        }
        for record, row_logits, row_probs, target_id in zip(records, logits, pred["target_probs"], pred["targets"])
    ]
    return pred


def _arm_report(
    name: str,
    pred: dict[str, Any],
    eval_records: list[dict[str, Any]],
    grid_size: int,
    models: dict[str, Any],
    oracle: bool = False,
) -> dict[str, Any]:
    metrics = _variant_metrics(eval_records, pred, grid_size)
    return {
        "arm": name,
        "oracle": oracle,
        "not_paper_grade": True,
        "training_performed": True,
        "lora_training_performed": False,
        "trainable_parameter_count": int(models["target_weights"].size + models["tca_action_weights"].size),
        "target_probs_available": "target_probs" in pred,
        "evaluation_metrics": metrics,
        "diagnostics": pred.get("diagnostics", []),
    }


def _comparison(arms: dict[str, dict[str, Any]], target_sanity: dict[str, Any]) -> dict[str, Any]:
    hard = arms["learned_target_hard_tca"]["evaluation_metrics"]
    oracle = arms["oracle_target_tca_upper_bound"]["evaluation_metrics"]
    soft = arms["soft_target_marginalized_tca"]["evaluation_metrics"]
    soft_select = arms["soft_target_distributional_select"]["evaluation_metrics"]
    text_prior = arms["instruction_text_prior_tca"]["evaluation_metrics"]
    constant = arms["constant_majority_target_tca"]["evaluation_metrics"]

    def delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
        return round(float(left[key]) - float(right[key]), 6)

    correct_in_topk = float(soft.get("target_topk_contains_correct", 0.0)) > 0.0
    soft_improves = (
        delta(soft, hard, "standard_proxy_score") > 0.0
        or delta(soft, hard, "action_l1") < 0.0
        or delta(soft_select, hard, "standard_proxy_score") > 0.0
        or delta(soft_select, hard, "wrong_target_proxy_rate") < 0.0
    )
    best_non_oracle_name = max(
        ["learned_target_hard_tca", "soft_target_marginalized_tca", "soft_target_distributional_select", "instruction_text_prior_tca", "constant_majority_target_tca"],
        key=lambda item: float(arms[item]["evaluation_metrics"]["standard_proxy_score"]),
    )
    best_non_oracle_metrics = arms[best_non_oracle_name]["evaluation_metrics"]
    if correct_in_topk and soft_improves:
        conclusion = "soft_target_marginalization_supports_tca_select_redesign"
        recommendation = "redesign_distributional_tca_select_with_soft_target_marginalization"
    elif float(target_sanity["eval_target_top1_accuracy"]) == 0.0 and float(target_sanity["eval_target_topk_accuracy"]) > 0.0:
        conclusion = "target_topk_contains_correct_but_top1_prior_fails"
        recommendation = "improve_target_prior_classifier_then_rerun_head_only_tca"
    elif float(target_sanity["eval_target_topk_accuracy"]) == 0.0:
        conclusion = "target_prior_does_not_cover_correct_target"
        recommendation = "prioritize_target_prior_redesign"
    elif float(text_prior["standard_proxy_score"]) > float(hard["standard_proxy_score"]):
        conclusion = "instruction_text_prior_rescues_tca_proxy"
        recommendation = "improve_target_prior_classifier_then_rerun_head_only_tca"
    else:
        conclusion = "no_target_prior_variant_rescues_tca"
        recommendation = "redesign_or_kill_pivot_current_tca_target_head"
    return {
        "conclusion": conclusion,
        "recommended_next_milestone": recommendation,
        "target_topk_contains_correct": bool(correct_in_topk),
        "soft_marginalization_helps_over_hard": bool(soft_improves),
        "best_non_oracle_arm": best_non_oracle_name,
        "best_non_oracle_standard_proxy": best_non_oracle_metrics["standard_proxy_score"],
        "gap_to_oracle_standard_proxy": round(float(oracle["standard_proxy_score"]) - float(best_non_oracle_metrics["standard_proxy_score"]), 6),
        "hard_to_oracle": {
            "standard_proxy_delta": delta(oracle, hard, "standard_proxy_score"),
            "wrong_target_delta": delta(oracle, hard, "wrong_target_proxy_rate"),
            "action_target_consistency_delta": delta(oracle, hard, "action_target_consistency_score"),
        },
        "soft_vs_hard": {
            "standard_proxy_delta": delta(soft, hard, "standard_proxy_score"),
            "wrong_target_delta": delta(soft, hard, "wrong_target_proxy_rate"),
            "action_l1_delta": delta(soft, hard, "action_l1"),
            "action_target_consistency_delta": delta(soft, hard, "action_target_consistency_score"),
        },
        "soft_select_vs_hard": {
            "standard_proxy_delta": delta(soft_select, hard, "standard_proxy_score"),
            "wrong_target_delta": delta(soft_select, hard, "wrong_target_proxy_rate"),
            "action_l1_delta": delta(soft_select, hard, "action_l1"),
            "action_target_consistency_delta": delta(soft_select, hard, "action_target_consistency_score"),
        },
        "instruction_text_prior_vs_hard": {
            "standard_proxy_delta": delta(text_prior, hard, "standard_proxy_score"),
            "wrong_target_delta": delta(text_prior, hard, "wrong_target_proxy_rate"),
            "action_l1_delta": delta(text_prior, hard, "action_l1"),
            "action_target_consistency_delta": delta(text_prior, hard, "action_target_consistency_score"),
        },
        "constant_vs_hard": {
            "standard_proxy_delta": delta(constant, hard, "standard_proxy_score"),
            "wrong_target_delta": delta(constant, hard, "wrong_target_proxy_rate"),
        },
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# TCA Target-Prior Rescue Diagnostic",
        "",
        "This is exploratory tiny offline proxy evidence only. Oracle and metadata-like priors are diagnostics, not paper-grade results.",
        "",
        f"- passed: `{report['tca_target_prior_rescue_passed']}`",
        f"- conclusion: `{report['comparison']['conclusion']}`",
        f"- recommended next milestone: `{report['comparison']['recommended_next_milestone']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- LoRA training happened: `{report['policy']['lora_training_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        "",
        "## Target Head Sanity",
        "",
        f"- target CE loss: `{report['target_head_training_sanity']['initial_target_ce_loss']} -> {report['target_head_training_sanity']['final_target_ce_loss']}`",
        f"- train target top1: `{report['target_head_training_sanity']['train_target_top1_accuracy']}`",
        f"- eval target top1: `{report['target_head_training_sanity']['eval_target_top1_accuracy']}`",
        f"- eval target top-k: `{report['target_head_training_sanity']['eval_target_topk_accuracy']}`",
        "",
        "## Arms",
    ]
    for arm in report["arms"]:
        metrics = arm["evaluation_metrics"]
        lines.extend(
            [
                f"### `{arm['arm']}`",
                f"- oracle: `{arm['oracle']}`",
                f"- standard proxy: `{metrics['standard_proxy_score']}`",
                f"- wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- action L1: `{metrics['action_l1']}`",
                f"- action-target consistency: `{metrics['action_target_consistency_score']}`",
                f"- counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
                f"- target top1: `{metrics['target_top1_accuracy']}`",
                f"- target top-k contains correct: `{metrics.get('target_topk_contains_correct')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            report["target_prior_diagnosis"],
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_tca_target_prior_rescue(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    max_samples: int = 8,
    grid_size: int = 8,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
) -> dict[str, Any]:
    validate_bounds(max_pairs, max_action_steps, max_steps, max_runtime_seconds, learning_rate)
    if max_samples < 2 or max_samples > 32:
        raise TcaTargetPriorRescueError("max_samples must be between 2 and 32")

    started = time.perf_counter()
    records, train_records, eval_records, split, source_metadata = _prepare_records(
        manifest_path=manifest_path,
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_samples=max_samples,
    )
    models = _train_models(train_records, steps=max_steps, learning_rate=learning_rate)
    target_sanity = _target_head_training_sanity(train_records, eval_records, models, steps=max_steps)

    learned_hard = _arm_report(
        "learned_target_hard_tca",
        _learned_hard_pred(eval_records, models),
        eval_records,
        grid_size,
        models,
    )
    oracle = _arm_report(
        "oracle_target_tca_upper_bound",
        _oracle_pred(eval_records, models),
        eval_records,
        grid_size,
        models,
        oracle=True,
    )
    soft = _arm_report(
        "soft_target_marginalized_tca",
        _soft_marginalized_pred(eval_records, models),
        eval_records,
        grid_size,
        models,
    )
    soft_select = _arm_report(
        "soft_target_distributional_select",
        _soft_select_pred(eval_records, models),
        eval_records,
        grid_size,
        models,
    )
    text_prior = _arm_report(
        "instruction_text_prior_tca",
        _instruction_text_prior_pred(eval_records, models),
        eval_records,
        grid_size,
        models,
    )
    constant = _arm_report(
        "constant_majority_target_tca",
        _constant_prior_pred(eval_records, models, constant_target=0),
        eval_records,
        grid_size,
        models,
    )

    arms = [learned_hard, oracle, soft, soft_select, text_prior, constant]
    arm_map = {arm["arm"]: arm for arm in arms}
    comparison = _comparison(arm_map, target_sanity)
    elapsed = time.perf_counter() - started

    if float(oracle["evaluation_metrics"]["standard_proxy_score"]) > 0.0 and float(learned_hard["evaluation_metrics"]["standard_proxy_score"]) == 0.0:
        diagnosis = (
            "Oracle-target TCA remains strong while the learned target prior is weak. The target-conditioned action mechanism "
            "is not killed by this diagnostic; the immediate blocker is target prior/classifier generalization."
        )
    elif float(oracle["evaluation_metrics"]["standard_proxy_score"]) == 0.0:
        diagnosis = "Oracle-target TCA did not recover, so metric or action-conditioning bugs should be investigated before more prior work."
    else:
        diagnosis = "Target-prior variants are inconclusive on this tiny split."

    passed = bool(
        elapsed <= max_runtime_seconds
        and max_steps <= MAX_TRAINING_STEPS
        and target_sanity["loss_decreased"]
        and not any(arm["oracle"] and not arm["not_paper_grade"] for arm in arms)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_samples": max_samples,
        "max_steps": max_steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "grid_size": grid_size,
        "record_count": len(records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "split": split,
        "source_metadata": source_metadata,
        "target_head_training_sanity": target_sanity,
        "arms": arms,
        "comparison": comparison,
        "target_prior_diagnosis": diagnosis,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "tca_target_prior_rescue_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": (
            "Exploratory tiny offline proxy only. Oracle-target and instruction-text-prior arms are diagnostic upper-bound "
            "or engineering checks, not paper-grade evidence. The split, sample order, and metrics are unchanged from the "
            "prior head-only and LoRA diagnostics."
        ),
    }
    _write_reports(report, report_json=report_json, report_md=report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_tca_target_prior_rescue_report.json")
    parser.add_argument("--report-md", default="reports/libero_tca_target_prior_rescue_report.md")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    args = parser.parse_args()
    try:
        report = run_tca_target_prior_rescue(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_samples=args.max_samples,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            max_runtime_seconds=args.max_runtime_seconds,
        )
    except (TcaTargetPriorRescueError, TcaLabelConditioningAuditError) as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
