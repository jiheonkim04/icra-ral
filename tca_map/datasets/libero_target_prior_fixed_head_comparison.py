"""Tiny ActionMap vs TCA-Map comparison with target-prior fixes.

This runner intentionally stays on the same deterministic tiny split used by
the earlier head-only, LoRA, label-audit, and target-prior rescue diagnostics.
It is exploratory offline proxy evidence only.
"""

from __future__ import annotations

import argparse
import json
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
    _combined_loss,
    _instruction_features,
    _pair_features,
    _predict_regressor,
    _predict_targets,
    _round_curve,
    _softmax,
    validate_bounds,
)
from tca_map.datasets.libero_tca_label_conditioning_audit import (
    TcaLabelConditioningAuditError,
    _metrics,
    _policy,
    _predict_actionmap,
    _prepare_records,
    _round_list,
    _train_models,
)
from tca_map.datasets.libero_tca_target_prior_rescue import (
    _instruction_text_prior_pred,
    _learned_hard_pred,
    _oracle_pred,
    _target_head_training_sanity,
    _variant_metrics,
)
from tca_map.inference.tca_select import distributional_tca_select_inference


SCHEMA_VERSION = "2026-07-05.target_prior_fixed_head_comparison.v1"


class TargetPriorFixedHeadComparisonError(RuntimeError):
    """Raised when the bounded fixed-prior comparison cannot run safely."""


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != TARGET_COUNT:
        raise TargetPriorFixedHeadComparisonError("target prior probabilities must be [N, TARGET_COUNT]")
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return probs / row_sums


def _pred_from_probs(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    target_probs: np.ndarray,
    *,
    marginalize: bool = False,
) -> dict[str, Any]:
    probs = _normalize_probs(target_probs)
    selected_targets = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    actions: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, row_probs, selected_target in zip(records, _pair_features(records), probs, selected_targets):
        candidate_actions = _candidate_actions(pair_feature, models["tca_action_weights"])
        if marginalize:
            weighted = np.zeros(len(candidate_actions[0]), dtype=np.float64)
            for target_id, action in enumerate(candidate_actions):
                weighted += float(row_probs[target_id]) * np.asarray(action, dtype=np.float64)
            action = [float(value) for value in weighted.tolist()]
        else:
            action = [float(value) for value in candidate_actions[int(selected_target)]]
        actions.append(action)
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "target_probs": _round_list(row_probs),
                "selected_target": int(selected_target),
                "correct_target_in_top2": bool(
                    int(record["target_id"]) in np.argsort(-row_probs)[: min(2, TARGET_COUNT)].tolist()
                ),
                "marginalized_action": bool(marginalize),
            }
        )
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "targets": selected_targets,
        "target_probs": probs,
        "diagnostics": diagnostics,
    }


def _learned_target_probs(records: list[dict[str, Any]], models: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    _, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    return _softmax(logits), logits


def _learned_text_fusion_pred(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    learned_weight: float = 0.5,
) -> dict[str, Any]:
    learned_probs, logits = _learned_target_probs(records, models)
    text_pred = _instruction_text_prior_pred(records, models)
    text_probs = np.asarray(text_pred["target_probs"], dtype=np.float64)
    fused = _normalize_probs(learned_weight * learned_probs + (1.0 - learned_weight) * text_probs)
    pred = _pred_from_probs(records, models, fused)
    for diagnostic, row_logits, row_learned, row_text in zip(
        pred["diagnostics"], logits, learned_probs, text_probs
    ):
        diagnostic.update(
            {
                "target_prior_variant": "learned_text_fusion",
                "learned_weight": learned_weight,
                "learned_logits": _round_list(row_logits),
                "learned_probs": _round_list(row_learned),
                "text_probs": _round_list(row_text),
            }
        )
    return pred


def _topk_uniform_pred(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    k: int = 2,
) -> dict[str, Any]:
    learned_probs, logits = _learned_target_probs(records, models)
    k = max(1, min(k, TARGET_COUNT))
    uniform = np.zeros_like(learned_probs)
    for row_index, row in enumerate(learned_probs):
        topk = np.argsort(-row)[:k]
        uniform[row_index, topk] = 1.0 / float(k)
    pred = _pred_from_probs(records, models, uniform, marginalize=True)
    for diagnostic, row_logits, row_learned in zip(pred["diagnostics"], logits, learned_probs):
        diagnostic.update(
            {
                "target_prior_variant": "topk_uniform_marginalization",
                "learned_logits": _round_list(row_logits),
                "learned_probs": _round_list(row_learned),
                "topk": k,
            }
        )
    return pred


def _tca_select_with_probs_pred(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    target_probs: np.ndarray,
    *,
    source_variant: str,
) -> dict[str, Any]:
    probs = _normalize_probs(target_probs)
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, row_probs in zip(records, _pair_features(records), probs):
        candidates = []
        values = []
        for target_id, action in enumerate(_candidate_actions(pair_feature, models["tca_action_weights"])):
            action_distance = float(np.mean(np.abs(np.asarray(action) - np.asarray(record["candidate_actions"][target_id]))))
            value = float(row_probs[target_id]) * max(0.0, 1.0 - action_distance)
            candidates.append(
                {
                    "index": int(target_id),
                    "target_index": int(target_id),
                    "voxel": int(target_id),
                    "action": action,
                    "logit": value,
                }
            )
            values.append(value)
        selection = distributional_tca_select_inference(
            action_heatmap={"candidates": candidates, "values": values},
            target_heatmap={"scores": [float(value) for value in row_probs.tolist()], "top_index": int(np.argmax(row_probs))},
            masked_action_heatmap={"values": [0.0 for _ in candidates]},
            negative_action_heatmaps=[],
            K=min(2, len(candidates)),
            temperature=0.5,
            metadata={"source": "target_prior_fixed_head_comparison", "target_prior_variant": source_variant},
            external_verifier=None,
        )
        selected = selection.get("selected") or candidates[0]
        selected_actions.append([float(value) for value in selected.get("action", [])])
        selected_targets.append(int(selected.get("target_index", selected.get("index", 0))))
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "source_target_prior_variant": source_variant,
                "target_probs": _round_list(row_probs),
                "candidate_scores": _round_list(selection.get("scores", [])),
                "selected_target": selected_targets[-1],
                "external_verifier_used": bool(selection.get("external_verifier_used")),
                "privileged_inference_used": bool(selection.get("privileged_inference_used")),
            }
        )
    return {
        "actions": np.asarray(selected_actions, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "target_probs": probs,
        "diagnostics": diagnostics,
    }


def _training_losses(models: dict[str, Any], *, actionmap: bool) -> tuple[list[float], list[float], list[float]]:
    if actionmap:
        action_losses = [float(value) for value in models["actionmap_losses"]]
        return _combined_loss(action_losses), action_losses, []
    action_losses = [float(value) for value in models["tca_action_losses"]]
    target_losses = [float(value) for value in models["target_losses"]]
    return _combined_loss(action_losses, target_losses), action_losses, target_losses


def _arm_report(
    *,
    arm: str,
    model_head: str,
    target_prior_variant: str,
    pred: dict[str, Any],
    train_pred: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    grid_size: int,
    models: dict[str, Any],
    steps: int,
    learning_rate: float,
    actionmap: bool = False,
    oracle: bool = False,
    distributional_tca_select: bool = False,
) -> dict[str, Any]:
    combined_losses, action_losses, target_losses = _training_losses(models, actionmap=actionmap)
    train_metrics = _metrics(train_records, train_pred, grid_size)
    trainable_params = int(models["actionmap_weights"].size) if actionmap else int(
        models["target_weights"].size + models["tca_action_weights"].size
    )
    report = {
        "arm": arm,
        "model_head_trained": model_head,
        "target_prior_variant": target_prior_variant,
        "oracle": bool(oracle),
        "distributional_tca_select_used": bool(distributional_tca_select),
        "training_performed": True,
        "lora_training_performed": False,
        "rollouts_performed": False,
        "head_only": True,
        "target_conditioned": not actionmap,
        "data_source": "local LIBERO HDF5 action snippets from fixed counterfactual split manifest",
        "sample_count": len(train_records) + len(eval_records),
        "train_sample_count": len(train_records),
        "eval_sample_count": len(eval_records),
        "trainable_parameter_count": trainable_params,
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "initial_loss": round(float(combined_losses[0]), 6),
        "final_loss": round(float(combined_losses[-1]), 6),
        "loss_decreased": bool(combined_losses[-1] < combined_losses[0]),
        "loss_curve": _round_curve(combined_losses),
        "action_loss_curve": _round_curve(action_losses),
        "target_loss_curve": _round_curve(target_losses) if target_losses else [],
        "target_probs_available": "target_probs" in pred,
        "train_metrics": train_metrics,
        "evaluation_metrics": _variant_metrics(eval_records, pred, grid_size),
        "diagnostics": pred.get("diagnostics", []),
    }
    return report


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return round(float(left[key]) - float(right[key]), 6)


def _comparison(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actionmap = arms["actionmap_head_only"]["evaluation_metrics"]
    hard = arms["tca_map_hard_learned_target"]["evaluation_metrics"]
    oracle = arms["oracle_target_tca_upper_bound"]["evaluation_metrics"]
    non_oracle_names = [
        "tca_map_hard_learned_target",
        "tca_map_instruction_text_prior",
        "tca_map_learned_text_prior_fusion",
        "tca_map_topk_uniform_marginalization",
    ]
    best_non_oracle_name = max(non_oracle_names, key=lambda name: float(arms[name]["evaluation_metrics"]["standard_proxy_score"]))
    best_non_oracle = arms[best_non_oracle_name]["evaluation_metrics"]
    select_metrics = arms["tca_map_distributional_select_best_prior"]["evaluation_metrics"]
    best_beats_actionmap = (
        float(best_non_oracle["standard_proxy_score"]) > float(actionmap["standard_proxy_score"])
        and float(best_non_oracle["wrong_target_proxy_rate"]) < float(actionmap["wrong_target_proxy_rate"])
    )
    select_helps = (
        _delta(select_metrics, best_non_oracle, "standard_proxy_score") > 0.0
        or _delta(select_metrics, best_non_oracle, "wrong_target_proxy_rate") < 0.0
        or _delta(select_metrics, best_non_oracle, "action_target_consistency_score") > 0.0
    )
    if best_beats_actionmap and select_helps:
        conclusion = "target_prior_fix_recovers_tca_and_tca_select_adds_gain"
        recommendation = "rerun_lora_attribution_with_fixed_target_prior"
    elif best_beats_actionmap:
        conclusion = "target_prior_fix_recovers_tca_but_tca_select_not_supported"
        recommendation = "revise_distributional_tca_select"
    elif float(oracle["standard_proxy_score"]) > float(actionmap["standard_proxy_score"]):
        conclusion = "oracle_only_recovers_tca_non_oracle_priors_fail"
        recommendation = "kill_or_pivot_current_learned_target_head"
    else:
        conclusion = "target_prior_fix_does_not_recover_tca"
        recommendation = "kill_or_pivot_current_tca_target_head"
    return {
        "conclusion": conclusion,
        "recommended_next_milestone": recommendation,
        "best_non_oracle_tca_arm": best_non_oracle_name,
        "best_non_oracle_tca_standard_proxy": best_non_oracle["standard_proxy_score"],
        "best_non_oracle_tca_wrong_target_proxy": best_non_oracle["wrong_target_proxy_rate"],
        "gap_to_oracle_standard_proxy": round(float(oracle["standard_proxy_score"]) - float(best_non_oracle["standard_proxy_score"]), 6),
        "hard_learned_vs_actionmap": {
            "standard_proxy_score_delta": _delta(hard, actionmap, "standard_proxy_score"),
            "wrong_target_proxy_rate_delta": _delta(hard, actionmap, "wrong_target_proxy_rate"),
            "action_target_consistency_score_delta": _delta(hard, actionmap, "action_target_consistency_score"),
            "counterfactual_separation_margin_delta": _delta(hard, actionmap, "counterfactual_separation_margin"),
        },
        "best_non_oracle_vs_actionmap": {
            "standard_proxy_score_delta": _delta(best_non_oracle, actionmap, "standard_proxy_score"),
            "wrong_target_proxy_rate_delta": _delta(best_non_oracle, actionmap, "wrong_target_proxy_rate"),
            "action_target_consistency_score_delta": _delta(best_non_oracle, actionmap, "action_target_consistency_score"),
            "counterfactual_separation_margin_delta": _delta(best_non_oracle, actionmap, "counterfactual_separation_margin"),
        },
        "tca_select_vs_best_non_oracle": {
            "standard_proxy_score_delta": _delta(select_metrics, best_non_oracle, "standard_proxy_score"),
            "wrong_target_proxy_rate_delta": _delta(select_metrics, best_non_oracle, "wrong_target_proxy_rate"),
            "action_target_consistency_score_delta": _delta(select_metrics, best_non_oracle, "action_target_consistency_score"),
            "counterfactual_separation_margin_delta": _delta(select_metrics, best_non_oracle, "counterfactual_separation_margin"),
        },
        "target_prior_fix_recovers_tca_over_actionmap": bool(best_beats_actionmap),
        "distributional_tca_select_helped_after_fix": bool(select_helps),
        "learned_target_head_remains_bottleneck": bool(
            float(arms["tca_map_instruction_text_prior"]["evaluation_metrics"]["standard_proxy_score"])
            > float(hard["standard_proxy_score"])
        ),
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Target-Prior-Fixed Head Comparison",
        "",
        "Exploratory tiny offline proxy only. This is not standard success, not rollout evidence, and not paper-grade.",
        "",
        f"- passed: `{report['target_prior_fixed_head_comparison_passed']}`",
        f"- conclusion: `{report['comparison']['conclusion']}`",
        f"- recommended next milestone: `{report['comparison']['recommended_next_milestone']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- LoRA training happened: `{report['policy']['lora_training_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        f"- best non-oracle TCA arm: `{report['comparison']['best_non_oracle_tca_arm']}`",
        f"- gap to oracle standard proxy: `{report['comparison']['gap_to_oracle_standard_proxy']}`",
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
                f"- target prior variant: `{arm['target_prior_variant']}`",
                f"- oracle: `{arm['oracle']}`",
                f"- trainable parameters: `{arm['trainable_parameter_count']}`",
                f"- loss: `{arm['initial_loss']} -> {arm['final_loss']}`",
                f"- standard proxy: `{metrics['standard_proxy_score']}`",
                f"- wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- action-target consistency: `{metrics['action_target_consistency_score']}`",
                f"- counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
                f"- target top1: `{metrics['target_top1_accuracy']}`",
                f"- target top-k contains correct: `{metrics.get('target_topk_contains_correct')}`",
                "",
            ]
        )
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_target_prior_fixed_head_comparison(
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
    if max_samples != 8:
        raise TargetPriorFixedHeadComparisonError("this diagnostic must use the fixed 8-sample split")
    started = time.perf_counter()
    records, train_records, eval_records, split, source_metadata = _prepare_records(
        manifest_path=manifest_path,
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_samples=max_samples,
    )
    if len(records) != 8:
        raise TargetPriorFixedHeadComparisonError("fixed diagnostic expected exactly 8 records")
    models = _train_models(train_records, steps=max_steps, learning_rate=learning_rate)
    target_sanity = _target_head_training_sanity(train_records, eval_records, models, steps=max_steps)

    actionmap_pred = _predict_actionmap(eval_records, models)
    actionmap_train_pred = _predict_actionmap(train_records, models)
    hard_pred = _learned_hard_pred(eval_records, models)
    hard_train_pred = _learned_hard_pred(train_records, models)
    text_pred = _instruction_text_prior_pred(eval_records, models)
    text_train_pred = _instruction_text_prior_pred(train_records, models)
    fusion_pred = _learned_text_fusion_pred(eval_records, models, learned_weight=0.5)
    fusion_train_pred = _learned_text_fusion_pred(train_records, models, learned_weight=0.5)
    topk_pred = _topk_uniform_pred(eval_records, models, k=2)
    topk_train_pred = _topk_uniform_pred(train_records, models, k=2)
    oracle_pred = _oracle_pred(eval_records, models)
    oracle_train_pred = _oracle_pred(train_records, models)

    arms = [
        _arm_report(
            arm="actionmap_head_only",
            model_head="ActionMap head-only linear action regressor",
            target_prior_variant="none_actionmap_baseline",
            pred=actionmap_pred,
            train_pred=actionmap_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
            actionmap=True,
        ),
        _arm_report(
            arm="tca_map_hard_learned_target",
            model_head="TCA-Map head-only target classifier plus target-conditioned action regressor",
            target_prior_variant="hard_learned_target",
            pred=hard_pred,
            train_pred=hard_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
        ),
        _arm_report(
            arm="tca_map_instruction_text_prior",
            model_head="TCA-Map action regressor with instruction-text target prior at eval",
            target_prior_variant="instruction_text_prior",
            pred=text_pred,
            train_pred=text_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
        ),
        _arm_report(
            arm="tca_map_learned_text_prior_fusion",
            model_head="TCA-Map action regressor with learned/text target-prior fusion at eval",
            target_prior_variant="learned_text_prior_fusion",
            pred=fusion_pred,
            train_pred=fusion_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
        ),
        _arm_report(
            arm="tca_map_topk_uniform_marginalization",
            model_head="TCA-Map action regressor with top-k uniform target marginalization at eval",
            target_prior_variant="topk_uniform_marginalization",
            pred=topk_pred,
            train_pred=topk_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
        ),
        _arm_report(
            arm="oracle_target_tca_upper_bound",
            model_head="TCA-Map action regressor with oracle target labels at eval",
            target_prior_variant="oracle_target_upper_bound",
            pred=oracle_pred,
            train_pred=oracle_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
            oracle=True,
        ),
    ]
    preliminary = {arm["arm"]: arm for arm in arms}
    non_oracle_names = [
        "tca_map_hard_learned_target",
        "tca_map_instruction_text_prior",
        "tca_map_learned_text_prior_fusion",
        "tca_map_topk_uniform_marginalization",
    ]
    best_name = max(non_oracle_names, key=lambda name: float(preliminary[name]["evaluation_metrics"]["standard_proxy_score"]))
    best_probs = np.asarray(
        {
            "tca_map_hard_learned_target": hard_pred,
            "tca_map_instruction_text_prior": text_pred,
            "tca_map_learned_text_prior_fusion": fusion_pred,
            "tca_map_topk_uniform_marginalization": topk_pred,
        }[best_name]["target_probs"],
        dtype=np.float64,
    )
    best_train_probs = np.asarray(
        {
            "tca_map_hard_learned_target": hard_train_pred,
            "tca_map_instruction_text_prior": text_train_pred,
            "tca_map_learned_text_prior_fusion": fusion_train_pred,
            "tca_map_topk_uniform_marginalization": topk_train_pred,
        }[best_name]["target_probs"],
        dtype=np.float64,
    )
    select_pred = _tca_select_with_probs_pred(eval_records, models, best_probs, source_variant=best_name)
    select_train_pred = _tca_select_with_probs_pred(train_records, models, best_train_probs, source_variant=best_name)
    arms.append(
        _arm_report(
            arm="tca_map_distributional_select_best_prior",
            model_head="TCA-Map action regressor plus Distributional TCA-Select using best non-oracle prior",
            target_prior_variant=f"distributional_select_from_{best_name}",
            pred=select_pred,
            train_pred=select_train_pred,
            train_records=train_records,
            eval_records=eval_records,
            grid_size=grid_size,
            models=models,
            steps=max_steps,
            learning_rate=learning_rate,
            distributional_tca_select=True,
        )
    )

    arm_map = {arm["arm"]: arm for arm in arms}
    comparison = _comparison(arm_map)
    oracle_standard = float(arm_map["oracle_target_tca_upper_bound"]["evaluation_metrics"]["standard_proxy_score"])
    for arm in arms:
        arm["evaluation_metrics"]["gap_to_oracle_target_tca_standard_proxy"] = round(
            oracle_standard - float(arm["evaluation_metrics"]["standard_proxy_score"]),
            6,
        )

    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and max_steps <= MAX_TRAINING_STEPS
        and arm_map["actionmap_head_only"]["loss_decreased"]
        and arm_map["tca_map_hard_learned_target"]["loss_decreased"]
        and comparison["best_non_oracle_tca_arm"] != "oracle_target_tca_upper_bound"
    )
    if comparison["target_prior_fix_recovers_tca_over_actionmap"]:
        interpretation = (
            "A non-oracle target prior recovers TCA-Map over ActionMap on this fixed tiny offline proxy split. "
            "The hard learned target head remains weak, so the evidence supports keeping the target-conditioned "
            "action mechanism while redesigning or replacing the learned target prior. Distributional TCA-Select "
            "should be revised because it did not add measurable gain beyond the best target prior in this run."
        )
    else:
        interpretation = (
            "No non-oracle target-prior variant recovered TCA-Map over ActionMap on this fixed tiny offline proxy split. "
            "The current learned-target formulation should be killed or pivoted unless a concrete target-prior bug is found."
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
        "interpretation": interpretation,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "target_prior_fixed_head_comparison_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/libero_offline_counterfactual_split_report.json"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/libero_target_prior_fixed_head_comparison_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/libero_target_prior_fixed_head_comparison_report.md"))
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    args = parser.parse_args()
    try:
        report = run_target_prior_fixed_head_comparison(
            manifest_path=args.manifest,
            report_json=args.report_json,
            report_md=args.report_md,
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_samples=args.max_samples,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            max_runtime_seconds=args.max_runtime_seconds,
        )
    except (TcaLabelConditioningAuditError, TargetPriorFixedHeadComparisonError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
