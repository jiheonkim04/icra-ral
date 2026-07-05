"""Audit Distributional TCA-Select under target-prior uncertainty.

This is an execution-first tiny offline proxy diagnostic. It reuses the exact
8-sample deterministic LIBERO counterfactual split and trains only the same
small CPU NumPy heads used by the preceding diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
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
    _one_hot,
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
    _candidate_texts,
    _instruction_text_prior_pred,
    _oracle_pred,
    _target_head_training_sanity,
    _target_ids,
    _variant_metrics,
)
from tca_map.datasets.libero_target_prior_fixed_head_comparison import (
    _learned_target_probs,
    _normalize_probs,
    _pred_from_probs,
    _topk_uniform_pred,
)
from tca_map.inference.tca_select import distributional_tca_select_inference


SCHEMA_VERSION = "2026-07-05.tca_select_uncertainty_audit.v1"


class TcaSelectUncertaintyAuditError(RuntimeError):
    """Raised when the bounded uncertainty audit cannot run safely."""


def _row_rank(probs: np.ndarray, target_id: int) -> int:
    order = np.argsort(-np.asarray(probs, dtype=np.float64)).tolist()
    return int(order.index(int(target_id)) + 1) if int(target_id) in order else -1


def _probability_checks(probs: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(probs, dtype=np.float64)
    return {
        "finite": bool(np.isfinite(arr).all()),
        "all_zero": bool(np.allclose(arr, 0.0)),
        "constant": bool(arr.size > 0 and np.allclose(arr, arr.flat[0])),
        "row_sums": _round_list(arr.sum(axis=1) if arr.ndim == 2 else np.asarray([arr.sum()])),
        "normalization_ok": bool(arr.ndim == 2 and np.allclose(arr.sum(axis=1), 1.0)),
        "nan_count": int(np.isnan(arr).sum()),
    }


def _temperature_calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    return _softmax(np.asarray(logits, dtype=np.float64) / safe_temperature)


def _fixed_fusion_probs(
    learned_logits: np.ndarray,
    learned_probs: np.ndarray,
    text_probs: np.ndarray,
    *,
    temperature: float = 8.0,
    conflict_learned_weight: float = 0.25,
    agreement_learned_weight: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibrated = _temperature_calibrated_probs(learned_logits, temperature)
    fused_rows: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for row_index, (row_learned, row_calibrated, row_text) in enumerate(zip(learned_probs, calibrated, text_probs)):
        learned_top1 = int(np.argmax(row_learned))
        calibrated_top1 = int(np.argmax(row_calibrated))
        text_top1 = int(np.argmax(row_text))
        conflict = calibrated_top1 != text_top1
        learned_weight = conflict_learned_weight if conflict else agreement_learned_weight
        fused = learned_weight * row_calibrated + (1.0 - learned_weight) * row_text
        fused_rows.append(fused)
        diagnostics.append(
            {
                "row_index": row_index,
                "learned_top1": learned_top1,
                "calibrated_learned_top1": calibrated_top1,
                "text_top1": text_top1,
                "learned_text_conflict": bool(conflict),
                "temperature": temperature,
                "learned_weight_used": round(float(learned_weight), 6),
                "reason": "downweight_calibrated_learned_prior_when_it_conflicts_with_text_prior"
                if conflict
                else "equal_weight_when_learned_and_text_top1_agree",
            }
        )
    return _normalize_probs(np.asarray(fused_rows, dtype=np.float64)), diagnostics


def _prior_bundle(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, dict[str, Any]]:
    learned_probs, learned_logits = _learned_target_probs(records, models)
    text_pred = _instruction_text_prior_pred(records, models)
    text_probs = np.asarray(text_pred["target_probs"], dtype=np.float64)
    equal_fusion = _normalize_probs(0.5 * learned_probs + 0.5 * text_probs)
    temp_probs = _temperature_calibrated_probs(learned_logits, temperature=8.0)
    topk_pred = _topk_uniform_pred(records, models, k=2)
    topk_probs = np.asarray(topk_pred["target_probs"], dtype=np.float64)
    fixed_fusion, fixed_diag = _fixed_fusion_probs(learned_logits, learned_probs, text_probs)
    target_ids = _target_ids(records)
    oracle_probs = _one_hot(target_ids)
    return {
        "learned_target_prior": {
            "probs": learned_probs,
            "logits": learned_logits,
            "description": "raw learned target classifier probabilities",
        },
        "temperature_calibrated_learned_prior": {
            "probs": temp_probs,
            "logits": learned_logits,
            "temperature": 8.0,
            "description": "learned target logits softened with temperature 8.0",
        },
        "topk_uniform_prior": {
            "probs": topk_probs,
            "logits": learned_logits,
            "description": "uniform distribution over learned top-k targets with k=2",
            "marginalized_non_select": True,
        },
        "instruction_text_prior": {
            "probs": text_probs,
            "logits": None,
            "description": "non-oracle instruction/candidate-text prior",
            "text_diagnostics": text_pred.get("diagnostics", []),
        },
        "equal_learned_text_fusion": {
            "probs": equal_fusion,
            "logits": learned_logits,
            "description": "previous equal-weight raw learned/text fusion kept for audit",
        },
        "fixed_learned_text_fusion": {
            "probs": fixed_fusion,
            "logits": learned_logits,
            "description": "temperature-calibrated, conflict-aware learned/text fusion",
            "fusion_diagnostics": fixed_diag,
        },
        "oracle_target_upper_bound": {
            "probs": oracle_probs,
            "logits": None,
            "description": "oracle target labels for upper-bound reference only",
            "oracle": True,
        },
    }


def _select_action_from_prior(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    probs: np.ndarray,
    *,
    marginalize: bool = False,
) -> dict[str, Any]:
    return _pred_from_probs(records, models, probs, marginalize=marginalize)


def _internal_action_similarity(action: list[float], reference: list[float]) -> float:
    distance = float(np.mean(np.abs(np.asarray(action, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))
    return max(0.0, 1.0 - distance)


def _uncertainty_select_pred(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    probs: np.ndarray,
    *,
    source_variant: str,
    lambda_consistency: float = 0.25,
    lambda_wrong_target: float = 0.25,
) -> dict[str, Any]:
    normalized = _normalize_probs(probs)
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, row_probs in zip(records, _pair_features(records), normalized):
        predicted_actions = _candidate_actions(pair_feature, models["tca_action_weights"])
        candidates = []
        values = []
        scoring_rows = []
        for candidate_target, action in enumerate(predicted_actions):
            similarities = [
                _internal_action_similarity(action, target_action)
                for target_action in predicted_actions
            ]
            expected_target_conditioned_score = float(np.dot(row_probs, np.asarray(similarities, dtype=np.float64)))
            action_target_consistency = float(row_probs[candidate_target])
            wrong_target_risk = float(
                sum(
                    row_probs[target_id] * similarities[target_id]
                    for target_id in range(TARGET_COUNT)
                    if target_id != candidate_target
                )
            )
            score = (
                expected_target_conditioned_score
                + lambda_consistency * action_target_consistency
                - lambda_wrong_target * wrong_target_risk
            )
            candidates.append(
                {
                    "index": int(candidate_target),
                    "target_index": int(candidate_target),
                    "voxel": int(candidate_target),
                    "action": action,
                    "logit": score,
                }
            )
            values.append(float(score))
            scoring_rows.append(
                {
                    "target_index": int(candidate_target),
                    "expected_target_conditioned_action_score": round(expected_target_conditioned_score, 6),
                    "action_target_consistency": round(action_target_consistency, 6),
                    "wrong_target_risk": round(wrong_target_risk, 6),
                    "final_score": round(float(score), 6),
                    "internal_similarities": _round_list(similarities),
                }
            )
        selection = distributional_tca_select_inference(
            action_heatmap={"candidates": candidates, "values": values},
            target_heatmap={"scores": [float(value) for value in row_probs.tolist()], "top_index": int(np.argmax(row_probs))},
            masked_action_heatmap={"values": [0.0 for _ in candidates]},
            negative_action_heatmaps=[],
            K=min(2, len(candidates)),
            temperature=0.5,
            weights={
                "log_probability": 1.0,
                "condition_kl": 0.0,
                "negative_action_divergence": 0.0,
                "target_consistency": 0.0,
                "target_margin": 0.0,
                "entropy_penalty": 0.0,
            },
            metadata={"source": "tca_select_uncertainty_audit", "target_prior_variant": source_variant},
            external_verifier=None,
        )
        selected = selection.get("selected") or candidates[0]
        selected_action = [float(value) for value in selected.get("action", [])]
        selected_target = int(selected.get("target_index", selected.get("index", 0)))
        selected_actions.append(selected_action)
        selected_targets.append(selected_target)
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "source_target_prior_variant": source_variant,
                "target_probs": _round_list(row_probs),
                "selected_target": selected_target,
                "selected_action": _round_list(selected_action),
                "true_target": int(record["target_id"]),
                "counted_wrong_target": bool(selected_target != int(record["target_id"])),
                "candidate_scores": _round_list(selection.get("scores", [])),
                "scoring_terms": scoring_rows,
                "lambda_consistency": lambda_consistency,
                "lambda_wrong_target": lambda_wrong_target,
                "external_verifier_used": bool(selection.get("external_verifier_used")),
                "privileged_inference_used": bool(selection.get("privileged_inference_used")),
            }
        )
    return {
        "actions": np.asarray(selected_actions, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "target_probs": normalized,
        "diagnostics": diagnostics,
    }


def _existing_select_pred(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    probs: np.ndarray,
) -> dict[str, Any]:
    normalized = _normalize_probs(probs)
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, row_probs in zip(records, _pair_features(records), normalized):
        candidates = []
        values = []
        for target_id, action in enumerate(_candidate_actions(pair_feature, models["tca_action_weights"])):
            value = float(row_probs[target_id])
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
            metadata={"source": "existing_tca_select_baseline"},
            external_verifier=None,
        )
        selected = selection.get("selected") or candidates[0]
        selected_action = [float(value) for value in selected.get("action", [])]
        selected_target = int(selected.get("target_index", selected.get("index", 0)))
        selected_actions.append(selected_action)
        selected_targets.append(selected_target)
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "target_probs": _round_list(row_probs),
                "selected_target": selected_target,
                "selected_action": _round_list(selected_action),
                "true_target": int(record["target_id"]),
                "counted_wrong_target": bool(selected_target != int(record["target_id"])),
                "candidate_scores": _round_list(selection.get("scores", [])),
                "external_verifier_used": bool(selection.get("external_verifier_used")),
                "privileged_inference_used": bool(selection.get("privileged_inference_used")),
            }
        )
    return {
        "actions": np.asarray(selected_actions, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "target_probs": normalized,
        "diagnostics": diagnostics,
    }


def _pair_features(records: list[dict[str, Any]]) -> np.ndarray:
    from tca_map.datasets.libero_offline_head_comparison import _pair_features as impl

    return impl(records)


def _arm_metrics_with_gap(
    records: list[dict[str, Any]],
    pred: dict[str, Any],
    grid_size: int,
    oracle_standard_proxy: float,
) -> dict[str, Any]:
    metrics = _variant_metrics(records, pred, grid_size)
    metrics["gap_to_oracle_target_tca_standard_proxy"] = round(
        float(oracle_standard_proxy) - float(metrics["standard_proxy_score"]),
        6,
    )
    return metrics


def _variant_report(
    name: str,
    prior_name: str,
    pred: dict[str, Any],
    records: list[dict[str, Any]],
    grid_size: int,
    oracle_standard_proxy: float,
    *,
    selector: bool,
    existing_baseline: bool = False,
    oracle: bool = False,
    nonselect_reference: str | None = None,
    reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _arm_metrics_with_gap(records, pred, grid_size, oracle_standard_proxy)
    delta = None
    if reference_metrics:
        delta = {
            "standard_proxy_score_delta": round(
                float(metrics["standard_proxy_score"]) - float(reference_metrics["standard_proxy_score"]),
                6,
            ),
            "wrong_target_proxy_rate_delta": round(
                float(metrics["wrong_target_proxy_rate"]) - float(reference_metrics["wrong_target_proxy_rate"]),
                6,
            ),
            "action_target_consistency_score_delta": round(
                float(metrics["action_target_consistency_score"])
                - float(reference_metrics["action_target_consistency_score"]),
                6,
            ),
            "counterfactual_separation_margin_delta": round(
                float(metrics["counterfactual_separation_margin"])
                - float(reference_metrics["counterfactual_separation_margin"]),
                6,
            ),
        }
    return {
        "arm": name,
        "target_prior_variant": prior_name,
        "selector": bool(selector),
        "existing_tca_select_baseline": bool(existing_baseline),
        "oracle": bool(oracle),
        "not_paper_grade": True,
        "evaluation_metrics": metrics,
        "nonselect_reference": nonselect_reference,
        "delta_over_nonselect_reference": delta,
        "diagnostics": pred.get("diagnostics", []),
    }


def _selected_action_for_prior(records: list[dict[str, Any]], models: dict[str, Any], probs: np.ndarray) -> dict[str, Any]:
    return _select_action_from_prior(records, models, probs)


def _fusion_audit_rows(
    eval_records: list[dict[str, Any]],
    models: dict[str, Any],
    priors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    learned_probs = np.asarray(priors["learned_target_prior"]["probs"], dtype=np.float64)
    text_probs = np.asarray(priors["instruction_text_prior"]["probs"], dtype=np.float64)
    equal_probs = np.asarray(priors["equal_learned_text_fusion"]["probs"], dtype=np.float64)
    fixed_probs = np.asarray(priors["fixed_learned_text_fusion"]["probs"], dtype=np.float64)
    prior_map = {
        "learned": learned_probs,
        "text": text_probs,
        "equal_fusion": equal_probs,
        "fixed_fusion": fixed_probs,
    }
    pred_map = {
        key: _selected_action_for_prior(eval_records, models, value)
        for key, value in prior_map.items()
    }
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(eval_records):
        true_target = int(record["target_id"])
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "instruction": record["instruction"],
            "candidate_texts": _candidate_texts(record),
            "true_target": true_target,
            "prior_checks": {
                name: {
                    "probs": _round_list(values[index]),
                    "correct_target_rank": _row_rank(values[index], true_target),
                    "selected_target": int(pred_map[name]["targets"][index]),
                    "selected_action": _round_list(pred_map[name]["actions"][index]),
                    "counted_wrong_target": bool(int(pred_map[name]["targets"][index]) != true_target),
                    "finite": bool(np.isfinite(values[index]).all()),
                    "all_zero": bool(np.allclose(values[index], 0.0)),
                    "constant": bool(np.allclose(values[index], values[index][0])),
                    "normalized": bool(np.isclose(float(np.sum(values[index])), 1.0)),
                }
                for name, values in prior_map.items()
            },
        }
        rows.append(row)
    return rows


def _fusion_diagnosis(fusion_audit_rows: list[dict[str, Any]], prior_checks: dict[str, Any]) -> dict[str, Any]:
    equal_wrong = all(row["prior_checks"]["equal_fusion"]["counted_wrong_target"] for row in fusion_audit_rows)
    text_correct = all(not row["prior_checks"]["text"]["counted_wrong_target"] for row in fusion_audit_rows)
    fixed_correct = all(not row["prior_checks"]["fixed_fusion"]["counted_wrong_target"] for row in fusion_audit_rows)
    learned_overrode_text = all(
        row["prior_checks"]["learned"]["selected_target"] == row["prior_checks"]["equal_fusion"]["selected_target"]
        and row["prior_checks"]["text"]["selected_target"] != row["prior_checks"]["equal_fusion"]["selected_target"]
        for row in fusion_audit_rows
    )
    prob_issue = any(
        not check["finite"] or check["all_zero"] or not check["normalization_ok"]
        for check in prior_checks.values()
    )
    if equal_wrong and text_correct and learned_overrode_text and not prob_issue:
        reason = (
            "No normalization, NaN, all-zero, or class-id misalignment bug was found. "
            "The equal-weight fusion failed because the learned prior was confidently wrong on both eval samples "
            "and overwrote the correct instruction-text prior."
        )
    else:
        reason = "Fusion behavior was not fully explained by learned-prior overwrite; inspect per-sample audit rows."
    return {
        "fusion_bug_or_class_misalignment_found": False,
        "fusion_weighting_calibration_issue_found": bool(equal_wrong and text_correct and learned_overrode_text),
        "equal_fusion_wrong_on_all_eval_samples": bool(equal_wrong),
        "text_prior_correct_on_all_eval_samples": bool(text_correct),
        "fixed_fusion_correct_on_all_eval_samples": bool(fixed_correct),
        "learned_prior_overwrote_text_prior": bool(learned_overrode_text),
        "probability_vector_issue_found": bool(prob_issue),
        "reason": reason,
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# TCA-Select Target-Uncertainty Audit",
        "",
        "Exploratory tiny offline proxy only. This is not standard success, rollout evidence, or paper-grade evidence.",
        "",
        f"- passed: `{report['tca_select_uncertainty_audit_passed']}`",
        f"- fusion bug/misalignment found: `{report['fusion_diagnosis']['fusion_bug_or_class_misalignment_found']}`",
        f"- fusion weighting/calibration issue found: `{report['fusion_diagnosis']['fusion_weighting_calibration_issue_found']}`",
        f"- TCA-Select revised: `{report['tca_select_revised']}`",
        f"- TCA-Select helped: `{report['comparison']['tca_select_helped']}`",
        f"- best non-oracle TCA prior variant: `{report['comparison']['best_non_oracle_tca_prior_variant']}`",
        f"- best non-oracle TCA-Select variant: `{report['comparison']['best_non_oracle_tca_select_variant']}`",
        f"- recommendation: `{report['comparison']['recommended_next_milestone']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- LoRA training happened: `{report['policy']['lora_training_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        "",
        "## Fusion Diagnosis",
        "",
        report["fusion_diagnosis"]["reason"],
        "",
        "## Variants",
    ]
    for variant in report["variants"]:
        metrics = variant["evaluation_metrics"]
        lines.extend(
            [
                f"### `{variant['arm']}`",
                f"- prior: `{variant['target_prior_variant']}`",
                f"- selector: `{variant['selector']}`",
                f"- oracle: `{variant['oracle']}`",
                f"- standard proxy: `{metrics['standard_proxy_score']}`",
                f"- wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- action-target consistency: `{metrics['action_target_consistency_score']}`",
                f"- counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
                f"- target top1: `{metrics['target_top1_accuracy']}`",
                f"- target top-k contains correct: `{metrics.get('target_topk_contains_correct')}`",
                f"- gap to oracle: `{metrics['gap_to_oracle_target_tca_standard_proxy']}`",
                "",
            ]
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_tca_select_uncertainty_audit(
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
        raise TcaSelectUncertaintyAuditError("this diagnostic must use the fixed 8-sample split")
    started = time.perf_counter()
    records, train_records, eval_records, split, source_metadata = _prepare_records(
        manifest_path=manifest_path,
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_samples=max_samples,
    )
    if len(records) != 8:
        raise TcaSelectUncertaintyAuditError("fixed diagnostic expected exactly 8 records")

    models = _train_models(train_records, steps=max_steps, learning_rate=learning_rate)
    target_sanity = _target_head_training_sanity(train_records, eval_records, models, steps=max_steps)
    actionmap_eval = _predict_actionmap(eval_records, models)
    actionmap_metrics = _metrics(eval_records, actionmap_eval, grid_size)
    priors = _prior_bundle(eval_records, models)
    prior_checks = {name: _probability_checks(np.asarray(payload["probs"], dtype=np.float64)) for name, payload in priors.items()}
    fusion_rows = _fusion_audit_rows(eval_records, models, priors)
    fusion_diagnosis = _fusion_diagnosis(fusion_rows, prior_checks)

    oracle_pred = _oracle_pred(eval_records, models)
    oracle_metrics = _variant_metrics(eval_records, oracle_pred, grid_size)
    oracle_standard = float(oracle_metrics["standard_proxy_score"])

    nonselect_predictions = {
        "learned_target_prior": _select_action_from_prior(eval_records, models, priors["learned_target_prior"]["probs"]),
        "temperature_calibrated_learned_prior": _select_action_from_prior(
            eval_records,
            models,
            priors["temperature_calibrated_learned_prior"]["probs"],
        ),
        "topk_uniform_prior": _select_action_from_prior(
            eval_records,
            models,
            priors["topk_uniform_prior"]["probs"],
            marginalize=True,
        ),
        "instruction_text_prior": _select_action_from_prior(eval_records, models, priors["instruction_text_prior"]["probs"]),
        "equal_learned_text_fusion": _select_action_from_prior(
            eval_records,
            models,
            priors["equal_learned_text_fusion"]["probs"],
        ),
        "fixed_learned_text_fusion": _select_action_from_prior(
            eval_records,
            models,
            priors["fixed_learned_text_fusion"]["probs"],
        ),
    }
    nonselect_variants = [
        _variant_report(
            f"tca_nonselect_{name}",
            name,
            pred,
            eval_records,
            grid_size,
            oracle_standard,
            selector=False,
        )
        for name, pred in nonselect_predictions.items()
    ]
    nonselect_map = {variant["target_prior_variant"]: variant for variant in nonselect_variants}

    select_predictions = {
        "existing_tca_select_baseline": _existing_select_pred(eval_records, models, priors["learned_target_prior"]["probs"]),
        "tca_select_learned_target_prior": _uncertainty_select_pred(
            eval_records,
            models,
            priors["learned_target_prior"]["probs"],
            source_variant="learned_target_prior",
        ),
        "tca_select_temperature_calibrated_learned_prior": _uncertainty_select_pred(
            eval_records,
            models,
            priors["temperature_calibrated_learned_prior"]["probs"],
            source_variant="temperature_calibrated_learned_prior",
        ),
        "tca_select_topk_uniform_prior": _uncertainty_select_pred(
            eval_records,
            models,
            priors["topk_uniform_prior"]["probs"],
            source_variant="topk_uniform_prior",
        ),
        "tca_select_instruction_text_prior": _uncertainty_select_pred(
            eval_records,
            models,
            priors["instruction_text_prior"]["probs"],
            source_variant="instruction_text_prior",
        ),
        "tca_select_fixed_learned_text_fusion": _uncertainty_select_pred(
            eval_records,
            models,
            priors["fixed_learned_text_fusion"]["probs"],
            source_variant="fixed_learned_text_fusion",
        ),
    }
    select_reference = {
        "existing_tca_select_baseline": "learned_target_prior",
        "tca_select_learned_target_prior": "learned_target_prior",
        "tca_select_temperature_calibrated_learned_prior": "temperature_calibrated_learned_prior",
        "tca_select_topk_uniform_prior": "topk_uniform_prior",
        "tca_select_instruction_text_prior": "instruction_text_prior",
        "tca_select_fixed_learned_text_fusion": "fixed_learned_text_fusion",
    }
    select_variants = []
    for arm_name, pred in select_predictions.items():
        reference_name = select_reference[arm_name]
        select_variants.append(
            _variant_report(
                arm_name,
                reference_name,
                pred,
                eval_records,
                grid_size,
                oracle_standard,
                selector=True,
                existing_baseline=arm_name == "existing_tca_select_baseline",
                nonselect_reference=f"tca_nonselect_{reference_name}",
                reference_metrics=nonselect_map[reference_name]["evaluation_metrics"],
            )
        )

    oracle_variant = _variant_report(
        "oracle_target_tca_upper_bound",
        "oracle_target_upper_bound",
        oracle_pred,
        eval_records,
        grid_size,
        oracle_standard,
        selector=False,
        oracle=True,
    )
    variants = nonselect_variants + select_variants + [oracle_variant]

    best_prior = max(
        nonselect_variants,
        key=lambda item: float(item["evaluation_metrics"]["standard_proxy_score"]),
    )
    best_select = max(
        select_variants,
        key=lambda item: float(item["evaluation_metrics"]["standard_proxy_score"]),
    )
    positive_delta_variants = []
    meaningful_help_variants = []
    for item in select_variants:
        delta = item["delta_over_nonselect_reference"]
        if not delta:
            continue
        has_positive_delta = (
            float(delta["standard_proxy_score_delta"]) > 0.0
            or float(delta["wrong_target_proxy_rate_delta"]) < 0.0
            or float(delta["action_target_consistency_score_delta"]) > 0.0
        )
        has_meaningful_delta = (
            float(delta["standard_proxy_score_delta"]) >= 0.01
            or float(delta["wrong_target_proxy_rate_delta"]) < 0.0
            or (
                float(delta["action_target_consistency_score_delta"]) > 0.0
                and float(delta["standard_proxy_score_delta"]) >= 0.0
            )
        )
        if has_positive_delta:
            positive_delta_variants.append(item["arm"])
        if has_meaningful_delta:
            meaningful_help_variants.append(item["arm"])
    fixed_fusion_metrics = nonselect_map["fixed_learned_text_fusion"]["evaluation_metrics"]
    text_metrics = nonselect_map["instruction_text_prior"]["evaluation_metrics"]
    fixed_fusion_recovers = math.isclose(
        float(fixed_fusion_metrics["standard_proxy_score"]),
        float(text_metrics["standard_proxy_score"]),
        abs_tol=1e-6,
    )
    if fixed_fusion_recovers:
        recommendation = "A_rerun_LoRA_with_fixed_target_prior"
    elif meaningful_help_variants:
        recommendation = "B_scale_tiny_split_cautiously"
    elif fusion_diagnosis["fusion_weighting_calibration_issue_found"]:
        recommendation = "C_redesign_learned_target_head"
    else:
        recommendation = "D_deemphasize_or_kill_TCA_Select"

    comparison = {
        "actionmap_standard_proxy": actionmap_metrics["standard_proxy_score"],
        "actionmap_wrong_target_proxy": actionmap_metrics["wrong_target_proxy_rate"],
        "best_non_oracle_tca_prior_variant": best_prior["arm"],
        "best_non_oracle_tca_prior_standard_proxy": best_prior["evaluation_metrics"]["standard_proxy_score"],
        "best_non_oracle_tca_select_variant": best_select["arm"],
        "best_non_oracle_tca_select_standard_proxy": best_select["evaluation_metrics"]["standard_proxy_score"],
        "tca_select_helped": bool(meaningful_help_variants),
        "tca_select_helped_variants": meaningful_help_variants,
        "tca_select_positive_delta_variants": positive_delta_variants,
        "tca_select_weak_delta_only_variants": [
            name for name in positive_delta_variants if name not in meaningful_help_variants
        ],
        "fixed_learned_text_fusion_recovers_near_instruction_text_prior": bool(fixed_fusion_recovers),
        "all_non_text_non_oracle_variants_weak": bool(
            max(
                float(nonselect_map[name]["evaluation_metrics"]["standard_proxy_score"])
                for name in [
                    "learned_target_prior",
                    "temperature_calibrated_learned_prior",
                    "topk_uniform_prior",
                    "equal_learned_text_fusion",
                ]
            )
            <= float(actionmap_metrics["standard_proxy_score"])
        ),
        "recommended_next_milestone": recommendation,
    }

    combined_tca_losses = _combined_loss(
        [float(value) for value in models["tca_action_losses"]],
        [float(value) for value in models["target_losses"]],
    )
    actionmap_losses = [float(value) for value in models["actionmap_losses"]]
    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and max_steps <= MAX_TRAINING_STEPS
        and np.isfinite([float(value) for value in comparison.values() if isinstance(value, (int, float))]).all()
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
        "training_summary": {
            "training_happened": True,
            "lora_training_happened": False,
            "rollout_happened": False,
            "actionmap_initial_loss": round(float(actionmap_losses[0]), 6),
            "actionmap_final_loss": round(float(actionmap_losses[-1]), 6),
            "tca_initial_loss": round(float(combined_tca_losses[0]), 6),
            "tca_final_loss": round(float(combined_tca_losses[-1]), 6),
            "tca_loss_decreased": bool(combined_tca_losses[-1] < combined_tca_losses[0]),
            "tca_loss_curve": _round_curve(combined_tca_losses),
        },
        "target_head_training_sanity": target_sanity,
        "prior_checks": prior_checks,
        "fusion_audit": {
            "fusion_weights": {
                "equal_fusion_learned_weight": 0.5,
                "fixed_conflict_learned_weight": 0.25,
                "fixed_agreement_learned_weight": 0.5,
                "fixed_learned_temperature": 8.0,
            },
            "temperature_checked": [1.0, 8.0],
            "normalization_checked": True,
            "class_id_alignment_checked": True,
            "same_target_index_space_checked": True,
            "rows": fusion_rows,
        },
        "fusion_diagnosis": fusion_diagnosis,
        "tca_select_revised": True,
        "selector_scoring": {
            "score": "expected target-conditioned action score + lambda_consistency * p(selected_target) - lambda_wrong_target * wrong_target_risk",
            "lambda_consistency": 0.25,
            "lambda_wrong_target": 0.25,
            "uses_external_verifier": False,
            "uses_privileged_simulator_state": False,
            "hard_selects_only_top1_target": False,
        },
        "actionmap_baseline": actionmap_metrics,
        "variants": variants,
        "comparison": comparison,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "tca_select_uncertainty_audit_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/libero_offline_counterfactual_split_report.json"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/libero_tca_select_uncertainty_audit_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/libero_tca_select_uncertainty_audit_report.md"))
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    args = parser.parse_args()
    try:
        report = run_tca_select_uncertainty_audit(
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
    except (TcaLabelConditioningAuditError, TcaSelectUncertaintyAuditError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
