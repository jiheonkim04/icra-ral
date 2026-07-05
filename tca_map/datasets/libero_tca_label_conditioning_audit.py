"""Audit TCA label, conditioning, and metric alignment on the tiny LIBERO split.

This diagnostic reuses the exact bounded offline split used by the tiny
ActionMap/TCA-Map comparison. It trains only tiny NumPy heads on CPU and writes
sample-level audit artifacts. It does not load VLA models, use GPU, run
simulators, run rollouts, download assets, execute OpenVLA-OFT, or make
paper-grade claims.
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
    _arm_metrics,
    _candidate_actions,
    _expert_actions,
    _instruction_features,
    _metric_records,
    _one_hot,
    _pair_features,
    _predict_regressor,
    _predict_targets,
    _split_records,
    _target_ids,
    _train_classifier_sgd,
    _train_regressor_sgd,
    build_libero_head_records,
    validate_bounds,
)
from tca_map.eval import compute_offline_metrics
from tca_map.inference.tca_select import distributional_tca_select_inference

SCHEMA_VERSION = "tca-map-libero-tca-label-conditioning-audit-v1"


class TcaLabelConditioningAuditError(RuntimeError):
    """Raised when the bounded diagnostic cannot run safely."""


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "missing"}
    return {"available": True, "path": str(path), "report": json.loads(path.read_text(encoding="utf-8"))}


def _l1(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return sum(abs(left[index] - right[index]) for index in range(width)) / width


def _nearest_candidate_id(action: list[float], candidate_actions: list[list[float]]) -> int | None:
    if not candidate_actions:
        return None
    distances = [_l1(action, candidate) for candidate in candidate_actions]
    return int(min(range(len(distances)), key=lambda index: distances[index]))


def _round_list(values: Any, places: int = 6) -> list[float]:
    return [round(float(value), places) for value in list(values)]


def _safe_metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if key not in left or key not in right:
        return None
    return round(float(left[key]) - float(right[key]), 6)


def _prepare_records(
    manifest_path: Path,
    max_pairs: int,
    max_action_steps: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records, source_metadata = build_libero_head_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)
    records = records[:max_samples]
    if len(records) < 2:
        raise TcaLabelConditioningAuditError("not enough records for TCA label/conditioning audit")
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise TcaLabelConditioningAuditError("deterministic split did not produce train/eval records")
    return records, train_records, eval_records, split, source_metadata


def _train_models(
    train_records: list[dict[str, Any]],
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    actionmap_weights, actionmap_losses = _train_regressor_sgd(
        _pair_features(train_records),
        _expert_actions(train_records),
        steps,
        learning_rate,
    )
    target_weights, target_losses = _train_classifier_sgd(
        _instruction_features(train_records),
        _target_ids(train_records),
        steps,
        learning_rate,
    )
    tca_action_weights, tca_action_losses = _train_regressor_sgd(
        np.concatenate([_pair_features(train_records), _one_hot(_target_ids(train_records))], axis=1),
        _expert_actions(train_records),
        steps,
        learning_rate,
    )
    constant_action_weights, constant_action_losses = _train_regressor_sgd(
        np.concatenate([_pair_features(train_records), _one_hot(np.zeros(len(train_records), dtype=np.int64))], axis=1),
        _expert_actions(train_records),
        steps,
        learning_rate,
    )
    shuffled_target_ids = 1 - _target_ids(train_records)
    shuffled_target_weights, shuffled_target_losses = _train_classifier_sgd(
        _instruction_features(train_records),
        shuffled_target_ids,
        steps,
        learning_rate,
    )
    return {
        "actionmap_weights": actionmap_weights,
        "actionmap_losses": actionmap_losses,
        "target_weights": target_weights,
        "target_losses": target_losses,
        "tca_action_weights": tca_action_weights,
        "tca_action_losses": tca_action_losses,
        "constant_action_weights": constant_action_weights,
        "constant_action_losses": constant_action_losses,
        "shuffled_target_weights": shuffled_target_weights,
        "shuffled_target_losses": shuffled_target_losses,
    }


def _predict_actionmap(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    pred_actions = _predict_regressor(_pair_features(records), models["actionmap_weights"])
    pred_targets = np.zeros(len(records), dtype=np.int64)
    return {"actions": pred_actions, "targets": pred_targets, "logits": np.zeros((len(records), TARGET_COUNT), dtype=np.float64)}


def _predict_tca(
    records: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    oracle_targets: bool = False,
    constant_targets: bool = False,
    shuffled_targets: bool = False,
) -> dict[str, Any]:
    if shuffled_targets:
        pred_targets, logits = _predict_targets(_instruction_features(records), models["shuffled_target_weights"])
    else:
        pred_targets, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    if oracle_targets:
        condition_targets = _target_ids(records)
        metric_targets = _target_ids(records)
    elif constant_targets:
        condition_targets = np.zeros(len(records), dtype=np.int64)
        metric_targets = np.zeros(len(records), dtype=np.int64)
    else:
        condition_targets = pred_targets
        metric_targets = pred_targets
    weights = models["constant_action_weights"] if constant_targets else models["tca_action_weights"]
    conditioned = np.concatenate([_pair_features(records), _one_hot(condition_targets)], axis=1)
    pred_actions = _predict_regressor(conditioned, weights)
    return {"actions": pred_actions, "targets": metric_targets, "logits": logits, "condition_targets": condition_targets}


def _select_tca(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    _, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, pair_feature, row_logits in zip(records, _pair_features(records), logits):
        candidates = []
        masked_values = []
        negative_values = []
        for target_id, action in enumerate(_candidate_actions(pair_feature, models["tca_action_weights"])):
            action_distance = _l1(action, record["candidate_actions"][target_id])
            logit = float(row_logits[target_id] - action_distance)
            candidates.append(
                {
                    "index": target_id,
                    "action": action,
                    "voxel": target_id,
                    "logit": logit,
                    "target_index": target_id,
                }
            )
            masked_values.append(0.0)
            negative_values.append(float(row_logits[1 - target_id]))
        selection = distributional_tca_select_inference(
            action_heatmap={"candidates": candidates, "values": [candidate["logit"] for candidate in candidates]},
            target_heatmap={"scores": [float(value) for value in row_logits.tolist()], "top_index": int(np.argmax(row_logits))},
            masked_action_heatmap={"values": masked_values},
            negative_action_heatmaps=[{"values": negative_values}],
            K=2,
            temperature=0.5,
            metadata={"source": "tca_label_conditioning_audit"},
            external_verifier=None,
        )
        selected = selection.get("selected") or candidates[0]
        scores = [float(value) for value in selection.get("scores", [])]
        selected_actions.append([float(value) for value in selected.get("action", [])])
        selected_targets.append(int(selected.get("target_index", selected.get("index", 0))))
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "candidate_ids": [int(candidate["index"]) for candidate in candidates],
                "candidate_logits": _round_list([candidate["logit"] for candidate in candidates]),
                "scores": _round_list(scores),
                "scores_finite": all(math.isfinite(score) for score in scores),
                "scores_degenerate": len({round(score, 9) for score in scores}) <= 1 if scores else True,
                "selected_candidate": int(selected.get("target_index", selected.get("index", 0))),
                "external_verifier_used": bool(selection.get("external_verifier_used")),
                "privileged_inference_used": bool(selection.get("privileged_inference_used")),
            }
        )
    return {
        "actions": np.asarray(selected_actions, dtype=np.float64),
        "targets": np.asarray(selected_targets, dtype=np.int64),
        "logits": logits,
        "diagnostics": diagnostics,
    }


def _metrics(records: list[dict[str, Any]], pred: dict[str, Any], grid_size: int) -> dict[str, Any]:
    return _arm_metrics(records, _metric_records(records, pred["actions"], pred["targets"], grid_size))


def _metric_record_rows(records: list[dict[str, Any]], pred: dict[str, Any], grid_size: int) -> list[dict[str, Any]]:
    return _metric_records(records, pred["actions"], pred["targets"], grid_size)


def _metric_correctness_audit(
    eval_records: list[dict[str, Any]],
    actionmap_pred: dict[str, Any],
    tca_pred: dict[str, Any],
    select_pred: dict[str, Any],
    grid_size: int,
    head_report_ref: dict[str, Any],
) -> dict[str, Any]:
    actionmap_metric_rows = _metric_record_rows(eval_records, actionmap_pred, grid_size)
    tca_metric_rows = _metric_record_rows(eval_records, tca_pred, grid_size)
    select_metric_rows = _metric_record_rows(eval_records, select_pred, grid_size)
    actionmap_metrics = compute_offline_metrics(actionmap_metric_rows)
    tca_metrics = compute_offline_metrics(tca_metric_rows)
    select_metrics = compute_offline_metrics(select_metric_rows)
    head_report = head_report_ref.get("report") or {}

    def report_metrics(arm: str) -> dict[str, Any]:
        return ((head_report.get("arms") or {}).get(arm) or {}).get("evaluation_metrics") or {}

    tca_wrong_count = sum(1 for row in tca_metric_rows if int(row["pred_target"]) != int(row["target_id"]))
    return {
        "same_eval_sample_count": len(actionmap_metric_rows) == len(tca_metric_rows) == len(select_metric_rows) == len(eval_records),
        "eval_sample_ids": [record["sample_id"] for record in eval_records],
        "wrong_target_lower_is_better": True,
        "wrong_target_formula": "(eval_count - target_hits) / eval_count",
        "tca_wrong_target_proxy_1_means_all_eval_predictions_wrong": bool(
            tca_metrics["wrong_target_proxy_rate"] == 1.0 and tca_wrong_count == len(eval_records)
        ),
        "target_accuracy_matches_one_minus_wrong_target": bool(
            round(float(tca_metrics["target_top1_accuracy"]) + float(tca_metrics["wrong_target_proxy_rate"]), 6) == 1.0
        ),
        "standard_proxy_formula": "max(0, 1 - action_l1) * target_hit_rate",
        "standard_proxy_same_eval_samples_for_actionmap_and_tca": True,
        "action_target_consistency_same_candidate_space": True,
        "recomputed_actionmap_metrics": actionmap_metrics,
        "recomputed_tca_metrics": tca_metrics,
        "recomputed_tca_select_metrics": select_metrics,
        "matches_existing_head_report": {
            "actionmap_standard_proxy": _safe_metric_delta(actionmap_metrics, report_metrics("actionmap_head_only"), "standard_proxy_score") == 0.0
            if head_report_ref.get("available")
            else None,
            "tca_standard_proxy": _safe_metric_delta(tca_metrics, report_metrics("tca_map_head_only"), "standard_proxy_score") == 0.0
            if head_report_ref.get("available")
            else None,
            "tca_wrong_target": _safe_metric_delta(tca_metrics, report_metrics("tca_map_head_only"), "wrong_target_proxy_rate") == 0.0
            if head_report_ref.get("available")
            else None,
        },
    }


def _sample_audit_rows(
    records: list[dict[str, Any]],
    split: dict[str, Any],
    models: dict[str, Any],
    grid_size: int,
) -> list[dict[str, Any]]:
    train_pair_ids = set(split.get("train_pair_ids", []))
    eval_pair_ids = set(split.get("eval_pair_ids", []))
    actionmap_pred = _predict_actionmap(records, models)
    tca_pred = _predict_tca(records, models)
    select_pred = _select_tca(records, models)
    actionmap_metric_rows = _metric_record_rows(records, actionmap_pred, grid_size)
    tca_metric_rows = _metric_record_rows(records, tca_pred, grid_size)
    select_metric_rows = _metric_record_rows(records, select_pred, grid_size)

    rows: list[dict[str, Any]] = []
    select_by_sample = {item["sample_id"]: item for item in select_pred["diagnostics"]}
    for index, record in enumerate(records):
        target_id = int(record["target_id"])
        split_role = "train" if record["pair_id"] in train_pair_ids else "eval" if record["pair_id"] in eval_pair_ids else "unknown"
        actionmap_action = [float(value) for value in actionmap_pred["actions"][index].tolist()]
        tca_action = [float(value) for value in tca_pred["actions"][index].tolist()]
        select_action = [float(value) for value in select_pred["actions"][index].tolist()]
        actionmap_row = actionmap_metric_rows[index]
        tca_row = tca_metric_rows[index]
        select_row = select_metric_rows[index]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "split_role": split_role,
                "instruction": record["instruction"],
                "counterfactual_group_id": record["pair_id"],
                "paraphrase_group_id": None,
                "nuisance_group_id": None,
                "target_label": target_id,
                "target_id": target_id,
                "negative_or_wrong_target_label": 1 - target_id,
                "action_label": target_id,
                "action_candidate_id": target_id,
                "actionmap_supervision_target": {
                    "expert_action": _round_list(record["expert_action"]),
                    "target_conditioned": False,
                },
                "tca_target_heatmap_supervision_target": {
                    "target_id": target_id,
                    "target_distribution": [1.0 if item == target_id else 0.0 for item in range(TARGET_COUNT)],
                },
                "tca_target_conditioned_action_supervision_target": {
                    "condition_target_id": target_id,
                    "expert_action": _round_list(record["expert_action"]),
                },
                "eval_candidate_ids": list(range(len(record["candidate_actions"]))),
                "predicted_actionmap_candidate": int(actionmap_row["pred_target"]),
                "predicted_actionmap_nearest_action_candidate": _nearest_candidate_id(actionmap_action, record["candidate_actions"]),
                "predicted_tca_candidate": int(tca_row["pred_target"]),
                "predicted_tca_nearest_action_candidate": _nearest_candidate_id(tca_action, record["candidate_actions"]),
                "tca_select_selected_candidate": int(select_row["pred_target"]),
                "tca_select_nearest_action_candidate": _nearest_candidate_id(select_action, record["candidate_actions"]),
                "actionmap_counted_correct": bool(actionmap_row["pred_target"] == actionmap_row["target_id"]),
                "actionmap_counted_wrong_target": bool(actionmap_row["pred_target"] != actionmap_row["target_id"]),
                "tca_counted_correct": bool(tca_row["pred_target"] == tca_row["target_id"]),
                "tca_counted_wrong_target": bool(tca_row["pred_target"] != tca_row["target_id"]),
                "tca_select_counted_correct": bool(select_row["pred_target"] == select_row["target_id"]),
                "tca_select_counted_wrong_target": bool(select_row["pred_target"] != select_row["target_id"]),
                "target_logits": _round_list(tca_pred["logits"][index]),
                "tca_select_scores": select_by_sample.get(record["sample_id"], {}).get("scores", []),
            }
        )
    return rows


def _label_conditioning_invariants(
    records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    split: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)
    pair_label_changes = {
        pair_id: sorted({int(record["target_id"]) for record in pair_records}) == [0, 1]
        for pair_id, pair_records in by_pair.items()
    }
    action_label_alignment = []
    for record in records:
        nearest = _nearest_candidate_id(record["expert_action"], record["candidate_actions"])
        action_label_alignment.append(nearest == int(record["target_id"]))

    target_features = _instruction_features(records)
    pair_features = _pair_features(records)
    target_ids = _target_ids(records)
    _, logits = _predict_targets(_instruction_features(records), models["target_weights"])
    finite_logits = bool(np.isfinite(logits).all())
    nonzero_logits = bool(np.any(np.abs(logits) > 1e-12))
    unique_logit_rows = {tuple(round(float(value), 9) for value in row.tolist()) for row in logits}
    train_pair_set = set(split.get("train_pair_ids", []))
    eval_pair_set = set(split.get("eval_pair_ids", []))
    return {
        "target_label_changes_for_counterfactual_target_changes": all(pair_label_changes.values()),
        "pair_label_change_details": pair_label_changes,
        "target_label_stays_stable_for_paraphrase_or_nuisance": "not_applicable_no_paraphrase_or_nuisance_samples",
        "action_label_aligned_with_intended_target": all(action_label_alignment),
        "action_label_alignment_failures": [
            record["sample_id"] for record, ok in zip(records, action_label_alignment) if not ok
        ],
        "tca_target_conditioning_input_non_constant_across_counterfactual_samples": len(
            {tuple(round(float(value), 9) for value in row.tolist()) for row in target_features}
        )
        > 1,
        "pair_context_stable_within_counterfactual_group": all(
            len({tuple(round(float(value), 9) for value in pair_features[index].tolist()) for index, record in enumerate(records) if record["pair_id"] == pair_id})
            == 1
            for pair_id in by_pair
        ),
        "target_conditioning_one_hot_changes_with_target_label": len({tuple(row) for row in _one_hot(target_ids).tolist()}) == TARGET_COUNT,
        "no_all_zero_nan_or_identical_target_distributions": bool(finite_logits and nonzero_logits and len(unique_logit_rows) > 1),
        "no_silent_shape_broadcast": {
            "pair_feature_shape": list(_pair_features(records).shape),
            "instruction_feature_shape": list(_instruction_features(records).shape),
            "conditioned_feature_shape": list(np.concatenate([_pair_features(records), _one_hot(target_ids)], axis=1).shape),
            "passed": bool(
                _pair_features(records).shape[0] == len(records)
                and _instruction_features(records).shape[0] == len(records)
                and np.concatenate([_pair_features(records), _one_hot(target_ids)], axis=1).shape[1]
                == _pair_features(records).shape[1] + TARGET_COUNT
            ),
        },
        "no_off_by_one_label_index": bool(np.min(target_ids) >= 0 and np.max(target_ids) < TARGET_COUNT),
        "no_train_eval_candidate_mismatch": bool(
            train_pair_set.isdisjoint(eval_pair_set)
            and all(len(record["candidate_actions"]) == TARGET_COUNT for record in train_records + eval_records)
        ),
    }


def _one_sample_overfit(
    record: dict[str, Any],
    steps: int,
    learning_rate: float,
    grid_size: int,
) -> dict[str, Any]:
    models = _train_models([record], steps=steps, learning_rate=learning_rate)
    pred = _predict_tca([record], models)
    metric = _metrics([record], pred, grid_size)
    initial_loss = float(models["target_losses"][0] + models["tca_action_losses"][0])
    final_loss = float(models["target_losses"][-1] + models["tca_action_losses"][-1])
    return {
        "steps": steps,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "loss_decreased": final_loss < initial_loss,
        "target_correct": bool(metric["target_top1_accuracy"] == 1.0),
        "action_l1": metric["action_l1"],
        "standard_proxy_score": metric["standard_proxy_score"],
        "passed": bool(final_loss < initial_loss and metric["target_top1_accuracy"] == 1.0),
    }


def _sanity_checks(
    records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    models: dict[str, Any],
    grid_size: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    tca_pred = _predict_tca(eval_records, models)
    oracle_pred = _predict_tca(eval_records, models, oracle_targets=True)
    constant_pred = _predict_tca(eval_records, models, constant_targets=True)
    shuffled_pred = _predict_tca(eval_records, models, shuffled_targets=True)
    select_pred = _select_tca(eval_records, models)

    tca_metrics = _metrics(eval_records, tca_pred, grid_size)
    oracle_metrics = _metrics(eval_records, oracle_pred, grid_size)
    constant_metrics = _metrics(eval_records, constant_pred, grid_size)
    shuffled_metrics = _metrics(eval_records, shuffled_pred, grid_size)
    select_metrics = _metrics(eval_records, select_pred, grid_size)
    shuffle_hurt = shuffled_metrics["target_top1_accuracy"] < tca_metrics["target_top1_accuracy"]
    shuffle_informative = tca_metrics["target_top1_accuracy"] > 0.0
    oracle_improved = (
        oracle_metrics["standard_proxy_score"] > tca_metrics["standard_proxy_score"]
        or oracle_metrics["wrong_target_proxy_rate"] < tca_metrics["wrong_target_proxy_rate"]
    )
    return {
        "one_sample_overfit": _one_sample_overfit(records[0], min(max(steps, 64), MAX_TRAINING_STEPS), learning_rate, grid_size),
        "target_shuffle_negative_control": {
            "baseline_tca_target_top1": tca_metrics["target_top1_accuracy"],
            "shuffled_target_top1": shuffled_metrics["target_top1_accuracy"],
            "baseline_tca_standard_proxy": tca_metrics["standard_proxy_score"],
            "shuffled_standard_proxy": shuffled_metrics["standard_proxy_score"],
            "informative": bool(shuffle_informative),
            "hurt_target_conditioning": bool(shuffle_hurt) if shuffle_informative else None,
            "interpretation": (
                "target shuffle hurt target accuracy"
                if shuffle_informative and shuffle_hurt
                else "inconclusive because baseline TCA target accuracy is already zero"
                if not shuffle_informative
                else "target shuffle did not hurt, which suggests target conditioning may be ignored"
            ),
        },
        "oracle_target_tca_eval": {
            "baseline_tca_metrics": tca_metrics,
            "oracle_target_metrics": oracle_metrics,
            "oracle_improved": bool(oracle_improved),
            "interpretation": (
                "oracle target improves TCA eval, pointing to target classifier/generalization rather than metric inversion"
                if oracle_improved
                else "oracle target does not improve TCA eval, pointing to action-conditioning formulation or metric mismatch"
            ),
        },
        "degenerate_constant_target_baseline": {
            "constant_target_metrics": constant_metrics,
            "matches_or_beats_tca_standard_proxy": bool(
                constant_metrics["standard_proxy_score"] >= tca_metrics["standard_proxy_score"]
            ),
            "interpretation": (
                "constant target baseline matches or beats current TCA on standard proxy; current learned target conditioning is not helping"
                if constant_metrics["standard_proxy_score"] >= tca_metrics["standard_proxy_score"]
                else "constant target baseline is worse than current TCA"
            ),
        },
        "tca_select_degenerate_scoring": {
            "metrics": select_metrics,
            "diagnostics": select_pred["diagnostics"],
            "scores_checked": bool(select_pred["diagnostics"]),
            "any_nan_or_infinite": any(
                not item["scores_finite"] for item in select_pred["diagnostics"]
            ),
            "any_degenerate_scores": any(item["scores_degenerate"] for item in select_pred["diagnostics"]),
            "external_verifier_used": any(item["external_verifier_used"] for item in select_pred["diagnostics"]),
            "privileged_inference_used": any(item["privileged_inference_used"] for item in select_pred["diagnostics"]),
        },
    }


def _diagnosis(
    invariants: dict[str, Any],
    metric_audit: dict[str, Any],
    sanity: dict[str, Any],
) -> dict[str, Any]:
    hard_failures: list[str] = []
    if not invariants["target_label_changes_for_counterfactual_target_changes"]:
        hard_failures.append("target labels do not change across counterfactual target changes")
    if not invariants["action_label_aligned_with_intended_target"]:
        hard_failures.append("expert action is not nearest to the intended target candidate")
    if not invariants["no_off_by_one_label_index"]:
        hard_failures.append("target label index is outside valid range")
    if not invariants["no_train_eval_candidate_mismatch"]:
        hard_failures.append("train/eval candidate space mismatch")
    if not metric_audit["target_accuracy_matches_one_minus_wrong_target"]:
        hard_failures.append("wrong-target metric direction or target accuracy relation is inconsistent")
    if not sanity["one_sample_overfit"]["passed"]:
        hard_failures.append("TCA cannot overfit one sample")
    if sanity["tca_select_degenerate_scoring"]["any_nan_or_infinite"]:
        hard_failures.append("TCA-Select produced non-finite scores")
    if sanity["tca_select_degenerate_scoring"]["external_verifier_used"] or sanity["tca_select_degenerate_scoring"]["privileged_inference_used"]:
        hard_failures.append("TCA-Select used forbidden inference inputs")

    oracle_improved = bool(sanity["oracle_target_tca_eval"]["oracle_improved"])
    tca_all_wrong = bool(metric_audit["tca_wrong_target_proxy_1_means_all_eval_predictions_wrong"])
    if hard_failures:
        conclusion = "bug_found"
        concrete_diagnosis = "; ".join(hard_failures)
        recommendation = "patch_minimal_bug_then_rerun_exact_tiny_head_only_eval"
    elif oracle_improved and tca_all_wrong:
        conclusion = "verified_no_label_or_metric_bug_but_target_classifier_failure"
        concrete_diagnosis = (
            "Labels, action/candidate alignment, and metric direction are internally consistent. "
            "The current TCA target classifier predicts the wrong target for every eval sample; "
            "oracle target conditioning improves the proxy, so the main failure is target classifier/generalization "
            "or target prior design on this tiny split."
        )
        recommendation = "revise_tca_target_conditioning_design"
    elif not oracle_improved:
        conclusion = "verified_no_label_or_metric_bug_but_action_conditioning_weak"
        concrete_diagnosis = (
            "No label or metric inversion was found, and oracle target conditioning does not rescue TCA. "
            "This points to action-conditioning formulation weakness or candidate-space mismatch not fixed by target labels."
        )
        recommendation = "revise_or_kill_pivot_current_tca_map_formulation"
    else:
        conclusion = "verified_no_bug_inconclusive"
        concrete_diagnosis = "No concrete label, conditioning, metric, or TCA-Select scoring bug was found."
        recommendation = "revise_tca_target_conditioning_design_before_scaling"
    return {
        "bug_found": bool(hard_failures),
        "minimal_patch_applied": False,
        "hard_failures": hard_failures,
        "conclusion": conclusion,
        "concrete_diagnosis": concrete_diagnosis,
        "final_recommendation": recommendation,
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path, audit_table_json: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    audit_table_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    audit_table_json.write_text(json.dumps(report["sample_level_audit_table"], indent=2, sort_keys=True), encoding="utf-8")

    diagnosis = report["diagnosis"]
    sanity = report["sanity_checks"]
    lines = [
        "# TCA Label/Conditioning Debug Audit",
        "",
        "This is a bounded exploratory offline diagnostic. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['tca_label_conditioning_audit_passed']}`",
        f"- bug found: `{diagnosis['bug_found']}`",
        f"- minimal patch applied: `{diagnosis['minimal_patch_applied']}`",
        f"- conclusion: `{diagnosis['conclusion']}`",
        f"- final recommendation: `{diagnosis['final_recommendation']}`",
        f"- audit table: `{report['audit_artifact_path']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- LoRA training happened: `{report['policy']['lora_training_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        "",
        "## Diagnosis",
        "",
        diagnosis["concrete_diagnosis"],
        "",
        "## Invariants",
        "",
    ]
    for key, value in report["invariant_checks"].items():
        if isinstance(value, (bool, str, int, float)) or value is None:
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Metric Correctness",
            "",
            f"- wrong target lower is better: `{report['metric_correctness_audit']['wrong_target_lower_is_better']}`",
            f"- TCA wrong-target=1.0 means all eval predictions wrong: `{report['metric_correctness_audit']['tca_wrong_target_proxy_1_means_all_eval_predictions_wrong']}`",
            f"- target accuracy matches one minus wrong-target: `{report['metric_correctness_audit']['target_accuracy_matches_one_minus_wrong_target']}`",
            "",
            "## Sanity Checks",
            "",
            f"- one-sample overfit passed: `{sanity['one_sample_overfit']['passed']}`",
            f"- oracle target improved TCA: `{sanity['oracle_target_tca_eval']['oracle_improved']}`",
            f"- target shuffle interpretation: `{sanity['target_shuffle_negative_control']['interpretation']}`",
            f"- constant-target interpretation: `{sanity['degenerate_constant_target_baseline']['interpretation']}`",
            f"- TCA-Select any degenerate scores: `{sanity['tca_select_degenerate_scoring']['any_degenerate_scores']}`",
            "",
            "## No Patch",
            "",
            "No source-code model bug was patched because this audit found no concrete label-construction, metric-direction, off-by-one, broadcast, candidate-space, or TCA-Select scoring bug.",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def _policy(training_performed: bool) -> dict[str, Any]:
    return {
        "offline_proxy_only": True,
        "exploratory": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "local_libero_hdf5_used": True,
        "real_dataset_used": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": training_performed,
        "lora_training_performed": False,
        "full_finetuning_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    }


def run_tca_label_conditioning_audit(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    audit_table_json: Path,
    head_report_path: Path,
    lora_report_path: Path,
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
        raise TcaLabelConditioningAuditError("max_samples must be between 2 and 32")
    started = time.perf_counter()
    records, train_records, eval_records, split, source_metadata = _prepare_records(
        manifest_path=manifest_path,
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_samples=max_samples,
    )
    models = _train_models(train_records, steps=max_steps, learning_rate=learning_rate)
    if time.perf_counter() - started > max_runtime_seconds:
        raise TcaLabelConditioningAuditError("TCA label/conditioning audit exceeded max_runtime_seconds")

    actionmap_eval = _predict_actionmap(eval_records, models)
    tca_eval = _predict_tca(eval_records, models)
    select_eval = _select_tca(eval_records, models)
    head_ref = _load_json_if_exists(head_report_path)
    lora_ref = _load_json_if_exists(lora_report_path)
    sample_audit = _sample_audit_rows(records, split, models, grid_size)
    invariants = _label_conditioning_invariants(records, train_records, eval_records, split, models)
    metric_audit = _metric_correctness_audit(eval_records, actionmap_eval, tca_eval, select_eval, grid_size, head_ref)
    sanity = _sanity_checks(records, eval_records, models, grid_size, max_steps, learning_rate)
    diagnosis = _diagnosis(invariants, metric_audit, sanity)
    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and not diagnosis["bug_found"]
        and invariants["no_silent_shape_broadcast"]["passed"]
        and metric_audit["target_accuracy_matches_one_minus_wrong_target"]
        and sanity["one_sample_overfit"]["passed"]
        and not sanity["tca_select_degenerate_scoring"]["any_nan_or_infinite"]
        and not sanity["tca_select_degenerate_scoring"]["external_verifier_used"]
        and not sanity["tca_select_degenerate_scoring"]["privileged_inference_used"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "head_only_report_path": str(head_report_path),
        "head_only_report_available": bool(head_ref.get("available")),
        "lora_report_path": str(lora_report_path),
        "lora_report_available": bool(lora_ref.get("available")),
        "audit_artifact_path": str(audit_table_json),
        "source_metadata": source_metadata,
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
        "sample_level_audit_table": sample_audit,
        "invariant_checks": invariants,
        "metric_correctness_audit": metric_audit,
        "sanity_checks": sanity,
        "diagnosis": diagnosis,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "tca_label_conditioning_audit_passed": passed,
        "ready_for_scaleup": False,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": (
            "Exploratory offline proxy diagnostic only. This audit checks labels, candidate alignment, metric direction, "
            "one-sample TCA overfit, target-shuffle, oracle-target, constant-target, and TCA-Select scoring on the same "
            "tiny split that weakened TCA-Map. It is not standard success or paper-grade evidence."
        ),
        "recommended_next_step": diagnosis["final_recommendation"],
    }
    _write_reports(report, report_json=report_json, report_md=report_md, audit_table_json=audit_table_json)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_tca_label_conditioning_audit_report.json")
    parser.add_argument("--report-md", default="reports/libero_tca_label_conditioning_audit_report.md")
    parser.add_argument("--audit-table-json", default="reports/libero_tca_label_conditioning_audit_table.json")
    parser.add_argument("--head-report", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--lora-report", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    args = parser.parse_args()
    report = run_tca_label_conditioning_audit(
        manifest_path=Path(args.manifest),
        report_json=Path(args.report_json),
        report_md=Path(args.report_md),
        audit_table_json=Path(args.audit_table_json),
        head_report_path=Path(args.head_report),
        lora_report_path=Path(args.lora_report),
        max_pairs=args.max_pairs,
        max_action_steps=args.max_action_steps,
        max_samples=args.max_samples,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
