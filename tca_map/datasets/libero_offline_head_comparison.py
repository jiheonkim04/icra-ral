"""Tiny offline ActionMap vs TCA-Map training/eval over LIBERO HDF5 actions.

This is exploratory offline proxy evidence. It trains tiny NumPy head-only
models over a deterministic local LIBERO counterfactual split. It does not load
VLA models, use GPU, run simulators, run rollouts, download assets, or make
paper-grade claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.eval import compute_offline_metrics
from tca_map.heads import ActionMapHead
from tca_map.inference.tca_select import distributional_tca_select_inference

SCHEMA_VERSION = "tca-map-libero-offline-head-training-comparison-v1"
DEFAULT_MAX_STEPS = 64
DEFAULT_MAX_RUNTIME_SECONDS = 15 * 60
DEFAULT_LEARNING_RATE = 0.05
MAX_TRAINING_STEPS = 300
BATCH_SIZE = 1
TARGET_COUNT = 2
FEATURE_WIDTH = 16


class OfflineHeadComparisonError(RuntimeError):
    """Raised when the bounded offline comparison cannot run safely."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing input manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_action_block(path: Path, max_steps: int) -> list[list[float]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        for demo_name in sorted(data_group.keys()):
            demo = data_group[demo_name]
            if "actions" not in demo:
                continue
            return [[float(value) for value in row.tolist()] for row in demo["actions"][:max_steps]]
    raise ValueError(f"{path} has no demo actions dataset")


def _mean_action(actions: list[list[float]]) -> list[float]:
    width = len(actions[0]) if actions else 0
    return [sum(row[index] for row in actions) / len(actions) for index in range(width)]


def _l1(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return sum(abs(left[index] - right[index]) for index in range(width)) / width


def _hash_features(text: str, width: int = FEATURE_WIDTH) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    hashed = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(width - 4)]
    normalized = text.lower().replace("_", " ")
    words = [word for word in normalized.split() if word]
    scalars = [
        min(len(text), 240) / 240.0,
        min(len(words), 40) / 40.0,
        sum(char in "aeiou" for char in normalized) / max(1, len(normalized)),
        sum(char.isdigit() for char in text) / max(1, len(text)),
    ]
    return scalars + hashed


def _record(
    pair: dict[str, Any],
    pair_index: int,
    target_id: int,
    instruction: str,
    action: list[float],
    candidate_actions: list[list[float]],
) -> dict[str, Any]:
    pair_context = "||".join(
        [
            str(pair.get("suite", "unknown_suite")),
            str(pair.get("positive_task_id", "positive")),
            str(pair.get("counterfactual_task_id", "counterfactual")),
            str(pair.get("swap_type", "swap")),
        ]
    )
    suffix = "positive" if target_id == 0 else "counterfactual"
    return {
        "sample_id": f"{pair['pair_id']}::{suffix}",
        "pair_id": pair["pair_id"],
        "pair_index": pair_index,
        "target_id": target_id,
        "instruction": instruction,
        "expert_action": [float(value) for value in action],
        "candidate_actions": [[float(value) for value in candidate] for candidate in candidate_actions],
        "pair_features": _hash_features(pair_context),
        "instruction_features": _hash_features(instruction),
    }


def build_libero_head_records(
    manifest_path: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    pairs = manifest.get("counterfactual_pairs", [])[:max_pairs]
    for pair_index, pair in enumerate(pairs):
        try:
            positive_action = _mean_action(_read_first_action_block(Path(pair["positive_demo_file"]), max_action_steps))
            counter_action = _mean_action(_read_first_action_block(Path(pair["counterfactual_demo_file"]), max_action_steps))
            candidates = [positive_action, counter_action]
            records.append(
                _record(
                    pair,
                    pair_index,
                    0,
                    pair.get("positive_instruction") or "positive target",
                    positive_action,
                    candidates,
                )
            )
            records.append(
                _record(
                    pair,
                    pair_index,
                    1,
                    pair.get("counterfactual_instruction") or "counterfactual target",
                    counter_action,
                    candidates,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in real-data reports
            exclusions.append({"pair_id": pair.get("pair_id"), "reason": str(exc)})

    return records, {
        "manifest_pair_count": len(manifest.get("counterfactual_pairs", [])),
        "requested_max_pairs": max_pairs,
        "included_pair_count": len({record["pair_id"] for record in records}),
        "excluded_samples": exclusions,
    }


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_ids: list[str] = []
    for record in records:
        if record["pair_id"] not in pair_ids:
            pair_ids.append(record["pair_id"])
    if not pair_ids:
        return [], [], {"split_ready": False, "reason": "no pairs"}
    if len(pair_ids) == 1:
        return records, records, {
            "split_ready": True,
            "split_type": "exploratory_train_eval_same_due_to_single_pair",
            "train_pair_ids": pair_ids,
            "eval_pair_ids": pair_ids,
            "sample_ordering_rule": "manifest order, positive then counterfactual",
            "exploratory": True,
            "confirmatory": False,
        }
    train_pair_count = max(1, math.ceil(len(pair_ids) * 0.75))
    train_pair_count = min(train_pair_count, len(pair_ids) - 1)
    train_pair_ids = set(pair_ids[:train_pair_count])
    eval_pair_ids = set(pair_ids[train_pair_count:])
    train = [record for record in records if record["pair_id"] in train_pair_ids]
    eval_records = [record for record in records if record["pair_id"] in eval_pair_ids]
    return train, eval_records, {
        "split_ready": True,
        "split_type": "deterministic_manifest_order_pair_holdout",
        "train_pair_ids": pair_ids[:train_pair_count],
        "eval_pair_ids": pair_ids[train_pair_count:],
        "sample_ordering_rule": "manifest order, positive then counterfactual",
        "random_seeds_used": [],
        "exploratory": True,
        "confirmatory": False,
        "reason_exploratory": "tiny local offline proxy over a bounded HDF5 subset",
    }


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def _one_hot(indices: np.ndarray, width: int = TARGET_COUNT) -> np.ndarray:
    result = np.zeros((indices.shape[0], width), dtype=np.float64)
    result[np.arange(indices.shape[0]), np.clip(indices, 0, width - 1)] = 1.0
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _pair_features(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["pair_features"] for record in records], dtype=np.float64)


def _instruction_features(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["instruction_features"] for record in records], dtype=np.float64)


def _target_ids(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(record["target_id"]) for record in records], dtype=np.int64)


def _expert_actions(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["expert_action"] for record in records], dtype=np.float64)


def _regression_loss(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    pred = _with_bias(features) @ weights
    return float(np.mean((pred - targets) ** 2))


def _classifier_loss(features: np.ndarray, target_ids: np.ndarray, weights: np.ndarray) -> float:
    logits = _with_bias(features) @ weights
    probs = _softmax(logits)
    labels = _one_hot(target_ids, width=weights.shape[1])
    return float(-np.sum(labels * np.log(probs + 1e-12)) / labels.shape[0])


def _train_regressor_sgd(features: np.ndarray, targets: np.ndarray, steps: int, lr: float) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    weights = np.zeros((x.shape[1], targets.shape[1]), dtype=np.float64)
    losses = [_regression_loss(features, targets, weights)]
    for step in range(steps):
        index = step % targets.shape[0]
        xi = x[index : index + 1]
        yi = targets[index : index + 1]
        diff = xi @ weights - yi
        grad = (2.0 / max(1, targets.shape[1])) * (xi.T @ diff)
        weights -= lr * grad
        losses.append(_regression_loss(features, targets, weights))
    return weights, losses


def _train_classifier_sgd(features: np.ndarray, target_ids: np.ndarray, steps: int, lr: float) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    labels = _one_hot(target_ids)
    weights = np.zeros((x.shape[1], TARGET_COUNT), dtype=np.float64)
    losses = [_classifier_loss(features, target_ids, weights)]
    for step in range(steps):
        index = step % target_ids.shape[0]
        xi = x[index : index + 1]
        yi = labels[index : index + 1]
        probs = _softmax(xi @ weights)
        weights -= lr * (xi.T @ (probs - yi))
        losses.append(_classifier_loss(features, target_ids, weights))
    return weights, losses


def _predict_regressor(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(_with_bias(features) @ weights, -1.0, 1.0)


def _predict_targets(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = _with_bias(features) @ weights
    return np.argmax(logits, axis=1), logits


def _round_curve(losses: list[float], max_points: int = 12) -> list[dict[str, float | int]]:
    if len(losses) <= max_points:
        indices = list(range(len(losses)))
    else:
        indices = sorted({int(round(value)) for value in np.linspace(0, len(losses) - 1, num=max_points)})
    return [{"step": int(index), "loss": round(float(losses[index]), 6)} for index in indices]


def _combined_loss(action_losses: list[float], target_losses: list[float] | None = None) -> list[float]:
    if not target_losses:
        return [float(value) for value in action_losses]
    return [float(action_losses[index] + target_losses[index]) for index in range(min(len(action_losses), len(target_losses)))]


def _target_margin(logits: np.ndarray, target_ids: np.ndarray) -> float:
    margins: list[float] = []
    for row, target_id in zip(logits, target_ids):
        correct = float(row[target_id])
        others = np.delete(row, target_id)
        margins.append(correct - float(np.max(others)))
    return float(np.mean(margins)) if margins else 0.0


def _separation_margin(pred_action: list[float], target_id: int, candidate_actions: list[list[float]]) -> float:
    if len(candidate_actions) < 2:
        return 0.0
    correct = candidate_actions[target_id]
    wrong = candidate_actions[1 - target_id]
    return _l1(pred_action, wrong) - _l1(pred_action, correct)


def _metric_records(
    records: list[dict[str, Any]],
    pred_actions: np.ndarray,
    pred_targets: np.ndarray,
    grid_size: int,
) -> list[dict[str, Any]]:
    head = ActionMapHead(grid_size=grid_size)
    output: list[dict[str, Any]] = []
    for record, pred_action, pred_target in zip(records, pred_actions, pred_targets):
        pred_list = [float(value) for value in pred_action.tolist()]
        expert = record["expert_action"]
        output.append(
            {
                "sample_id": record["sample_id"],
                "pred_action": pred_list,
                "expert_action": expert,
                "pred_voxel": head.action_to_voxel(pred_list),
                "expert_voxel": head.action_to_voxel(expert),
                "pred_target": int(pred_target),
                "target_id": int(record["target_id"]),
                "latency_ms": 0.0,
            }
        )
    return output


def _arm_metrics(records: list[dict[str, Any]], metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_offline_metrics(metric_records)
    margins = [
        _separation_margin(metric["pred_action"], int(record["target_id"]), record["candidate_actions"])
        for record, metric in zip(records, metric_records)
    ]
    consistency = []
    for record, metric in zip(records, metric_records):
        target_ok = 1.0 if int(metric["pred_target"]) == int(record["target_id"]) else 0.0
        consistency.append(target_ok * max(0.0, 1.0 - _l1(metric["pred_action"], metric["expert_action"])))
    metrics["counterfactual_separation_margin"] = round(float(np.mean(margins)) if margins else 0.0, 6)
    metrics["action_target_consistency_score"] = round(float(np.mean(consistency)) if consistency else 0.0, 6)
    metrics["paraphrase_nuisance_stability"] = "not_available_no_paraphrase_variants"
    metrics["max_gpu_memory_mb"] = 0.0
    return metrics


def _actionmap_arm(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    grid_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_features = _pair_features(train_records)
    weights, action_losses = _train_regressor_sgd(train_features, _expert_actions(train_records), steps, lr)
    train_pred = _predict_regressor(train_features, weights)
    eval_pred = _predict_regressor(_pair_features(eval_records), weights)
    train_targets = np.zeros(len(train_records), dtype=np.int64)
    eval_targets = np.zeros(len(eval_records), dtype=np.int64)
    train_metrics = _arm_metrics(train_records, _metric_records(train_records, train_pred, train_targets, grid_size))
    eval_metrics = _arm_metrics(eval_records, _metric_records(eval_records, eval_pred, eval_targets, grid_size))
    combined = _combined_loss(action_losses)
    return _arm_report(
        arm="actionmap_head_only",
        model_head="ActionMap head-only linear action regressor",
        target_conditioned=False,
        trainable_params=int(weights.size),
        steps=steps,
        lr=lr,
        combined_losses=combined,
        action_losses=action_losses,
        target_losses=[],
        train_records=train_records,
        eval_records=eval_records,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        elapsed=time.perf_counter() - started,
    )


def _tca_arm(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    grid_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_pair = _pair_features(train_records)
    train_instr = _instruction_features(train_records)
    train_targets = _target_ids(train_records)
    target_weights, target_losses = _train_classifier_sgd(train_instr, train_targets, steps, lr)
    conditioned_train = np.concatenate([train_pair, _one_hot(train_targets)], axis=1)
    action_weights, action_losses = _train_regressor_sgd(conditioned_train, _expert_actions(train_records), steps, lr)

    train_pred_targets, train_logits = _predict_targets(train_instr, target_weights)
    train_conditioned = np.concatenate([train_pair, _one_hot(train_pred_targets)], axis=1)
    train_pred = _predict_regressor(train_conditioned, action_weights)
    eval_pair = _pair_features(eval_records)
    eval_pred_targets, eval_logits = _predict_targets(_instruction_features(eval_records), target_weights)
    eval_conditioned = np.concatenate([eval_pair, _one_hot(eval_pred_targets)], axis=1)
    eval_pred = _predict_regressor(eval_conditioned, action_weights)
    train_metrics = _arm_metrics(train_records, _metric_records(train_records, train_pred, train_pred_targets, grid_size))
    eval_metrics = _arm_metrics(eval_records, _metric_records(eval_records, eval_pred, eval_pred_targets, grid_size))
    combined = _combined_loss(action_losses, target_losses)
    report = _arm_report(
        arm="tca_map_head_only",
        model_head="TCA-Map head-only target classifier plus target-conditioned action regressor",
        target_conditioned=True,
        trainable_params=int(action_weights.size + target_weights.size),
        steps=steps,
        lr=lr,
        combined_losses=combined,
        action_losses=action_losses,
        target_losses=target_losses,
        train_records=train_records,
        eval_records=eval_records,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        elapsed=time.perf_counter() - started,
    )
    report["target_margin_train"] = round(_target_margin(train_logits, train_targets), 6)
    report["target_margin_eval"] = round(_target_margin(eval_logits, _target_ids(eval_records)), 6)
    report["_weights"] = {"target": target_weights, "action": action_weights}
    return report


def _candidate_actions(pair_feature: np.ndarray, action_weights: np.ndarray) -> list[list[float]]:
    rows: list[list[float]] = []
    for target_id in range(TARGET_COUNT):
        conditioned = np.concatenate([pair_feature, _one_hot(np.asarray([target_id]))[0]], axis=0).reshape(1, -1)
        rows.append([float(value) for value in _predict_regressor(conditioned, action_weights)[0].tolist()])
    return rows


def _tca_select_arm(
    tca_report: dict[str, Any],
    eval_records: list[dict[str, Any]],
    steps: int,
    lr: float,
    grid_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    target_weights: np.ndarray = tca_report["_weights"]["target"]
    action_weights: np.ndarray = tca_report["_weights"]["action"]
    _, logits = _predict_targets(_instruction_features(eval_records), target_weights)
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    selections: list[dict[str, Any]] = []
    for record, pair_feature, row_logits in zip(eval_records, _pair_features(eval_records), logits):
        candidates = []
        masked_values = []
        negative_values = []
        for target_id, action in enumerate(_candidate_actions(pair_feature, action_weights)):
            action_distance = _l1(action, record["candidate_actions"][target_id])
            logit = float(row_logits[target_id] - action_distance)
            candidates.append({"index": target_id, "action": action, "voxel": target_id, "logit": logit, "target_index": target_id})
            masked_values.append(0.0)
            negative_values.append(float(row_logits[1 - target_id]))
        selection = distributional_tca_select_inference(
            action_heatmap={"candidates": candidates, "values": [candidate["logit"] for candidate in candidates]},
            target_heatmap={"scores": [float(value) for value in row_logits.tolist()], "top_index": int(np.argmax(row_logits))},
            masked_action_heatmap={"values": masked_values},
            negative_action_heatmaps=[{"values": negative_values}],
            K=2,
            temperature=0.5,
            metadata={"source": "offline_head_training_eval"},
            external_verifier=None,
        )
        selected = selection.get("selected") or candidates[0]
        selected_actions.append([float(value) for value in selected.get("action", [])])
        selected_targets.append(int(selected.get("target_index", selected.get("index", 0))))
        selections.append(
            {
                "sample_id": record["sample_id"],
                "selected_target": selected_targets[-1],
                "scores": [round(float(value), 6) for value in selection.get("scores", [])],
                "external_verifier_used": bool(selection.get("external_verifier_used")),
                "privileged_inference_used": bool(selection.get("privileged_inference_used")),
            }
        )
    metric_records = _metric_records(eval_records, np.asarray(selected_actions), np.asarray(selected_targets), grid_size)
    metrics = _arm_metrics(eval_records, metric_records)
    arm = dict(tca_report)
    arm.pop("_weights", None)
    arm.update(
        {
            "arm": "tca_map_distributional_select",
            "model_head_trained": "reuses trained TCA-Map head; Distributional TCA-Select adds no trainable parameters",
            "selection_training_performed": False,
            "distributional_tca_select_used": True,
            "additional_trainable_parameter_count": 0,
            "evaluation_metrics": metrics,
            "selections": selections,
            "latency_ms": round((time.perf_counter() - started) * 1000.0 / max(1, len(eval_records)), 6),
        }
    )
    return arm


def _arm_report(
    arm: str,
    model_head: str,
    target_conditioned: bool,
    trainable_params: int,
    steps: int,
    lr: float,
    combined_losses: list[float],
    action_losses: list[float],
    target_losses: list[float],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "model_head_trained": model_head,
        "training_performed": True,
        "head_only": True,
        "target_conditioned": target_conditioned,
        "data_source": "local LIBERO HDF5 action snippets from counterfactual split manifest",
        "sample_count": len(train_records) + len(eval_records),
        "train_sample_count": len(train_records),
        "eval_sample_count": len(eval_records),
        "trainable_parameter_count": trainable_params,
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": lr,
        "initial_loss": round(float(combined_losses[0]), 6),
        "final_loss": round(float(combined_losses[-1]), 6),
        "loss_decreased": bool(combined_losses[-1] < combined_losses[0]),
        "loss_curve": _round_curve(combined_losses),
        "action_loss_curve": _round_curve(action_losses),
        "target_loss_curve": _round_curve(target_losses) if target_losses else [],
        "train_metrics": train_metrics,
        "evaluation_metrics": eval_metrics,
        "latency_ms": round(elapsed * 1000.0 / max(1, len(eval_records)), 6),
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return round(float(left[key]) - float(right[key]), 6)


def _comparison(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actionmap = arms["actionmap_head_only"]["evaluation_metrics"]
    tca = arms["tca_map_head_only"]["evaluation_metrics"]
    select = arms["tca_map_distributional_select"]["evaluation_metrics"]
    tca_vs_actionmap = {
        "standard_proxy_score_delta": _delta(tca, actionmap, "standard_proxy_score"),
        "action_l1_delta": _delta(tca, actionmap, "action_l1"),
        "action_mse_delta": _delta(tca, actionmap, "action_mse"),
        "target_top1_accuracy_delta": _delta(tca, actionmap, "target_top1_accuracy"),
        "wrong_target_proxy_rate_delta": _delta(tca, actionmap, "wrong_target_proxy_rate"),
        "counterfactual_separation_margin_delta": _delta(tca, actionmap, "counterfactual_separation_margin"),
        "action_target_consistency_score_delta": _delta(tca, actionmap, "action_target_consistency_score"),
    }
    select_vs_tca = {
        "standard_proxy_score_delta": _delta(select, tca, "standard_proxy_score"),
        "action_l1_delta": _delta(select, tca, "action_l1"),
        "wrong_target_proxy_rate_delta": _delta(select, tca, "wrong_target_proxy_rate"),
        "counterfactual_separation_margin_delta": _delta(select, tca, "counterfactual_separation_margin"),
        "action_target_consistency_score_delta": _delta(select, tca, "action_target_consistency_score"),
    }
    supports_tca = (
        tca_vs_actionmap["standard_proxy_score_delta"] >= 0.0
        and tca_vs_actionmap["wrong_target_proxy_rate_delta"] <= 0.0
        and tca_vs_actionmap["counterfactual_separation_margin_delta"] >= 0.0
    )
    supports_select = (
        select_vs_tca["wrong_target_proxy_rate_delta"] < 0.0
        or select_vs_tca["action_l1_delta"] < 0.0
        or select_vs_tca["counterfactual_separation_margin_delta"] > 0.0
    )
    if supports_tca and supports_select:
        conclusion = "supports_tca_map_and_tca_select"
    elif supports_tca:
        conclusion = "supports_tca_map_but_tca_select_not_improved_in_this_tiny_proxy"
    else:
        conclusion = "weakens_tca_map"
    return {
        "tca_map_vs_actionmap": tca_vs_actionmap,
        "tca_select_vs_tca_map": select_vs_tca,
        "supports_tca_map_head_only": bool(supports_tca),
        "supports_distributional_tca_select": bool(supports_select),
        "conclusion": conclusion,
    }


def _policy(training_performed: bool) -> dict[str, Any]:
    return {
        "offline_proxy_only": True,
        "exploratory": True,
        "confirmatory": False,
        "not_standard_success": True,
        "not_paper_grade": True,
        "local_libero_hdf5_used": True,
        "real_dataset_used": True,
        "backbone_frozen": True,
        "head_only_training": training_performed,
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


def validate_bounds(max_pairs: int, max_action_steps: int, max_steps: int, max_runtime_seconds: int, lr: float) -> None:
    if max_pairs < 1 or max_pairs > 16:
        raise OfflineHeadComparisonError("max_pairs must be between 1 and 16")
    if max_action_steps < 1 or max_action_steps > 64:
        raise OfflineHeadComparisonError("max_action_steps must be between 1 and 64")
    if max_steps < 1 or max_steps > MAX_TRAINING_STEPS:
        raise OfflineHeadComparisonError(f"max_steps must be between 1 and {MAX_TRAINING_STEPS}")
    if max_runtime_seconds < 1 or max_runtime_seconds > DEFAULT_MAX_RUNTIME_SECONDS:
        raise OfflineHeadComparisonError(f"max_runtime_seconds must be between 1 and {DEFAULT_MAX_RUNTIME_SECONDS}")
    if lr <= 0.0 or lr > 1.0:
        raise OfflineHeadComparisonError("learning rate must be in (0, 1]")


def build_offline_head_comparison(
    manifest_path: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    grid_size: int = 8,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
) -> dict[str, Any]:
    validate_bounds(max_pairs, max_action_steps, max_steps, max_runtime_seconds, learning_rate)
    started = time.perf_counter()
    records, source_metadata = build_libero_head_records(manifest_path, max_pairs, max_action_steps)
    if len(records) < 2:
        raise OfflineHeadComparisonError("not enough LIBERO HDF5 records for ActionMap vs TCA-Map comparison")
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise OfflineHeadComparisonError("deterministic split did not produce train/eval records")

    actionmap = _actionmap_arm(train_records, eval_records, max_steps, learning_rate, grid_size)
    tca = _tca_arm(train_records, eval_records, max_steps, learning_rate, grid_size)
    tca_select = _tca_select_arm(tca, eval_records, max_steps, learning_rate, grid_size)
    tca.pop("_weights", None)
    arms = {
        "actionmap_head_only": actionmap,
        "tca_map_head_only": tca,
        "tca_map_distributional_select": tca_select,
    }
    elapsed = time.perf_counter() - started
    comparison = _comparison(arms)
    finite_losses = all(math.isfinite(float(arm["initial_loss"])) and math.isfinite(float(arm["final_loss"])) for arm in arms.values())
    passed = bool(finite_losses and elapsed <= max_runtime_seconds and max_steps <= MAX_TRAINING_STEPS)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "source_metadata": source_metadata,
        "split": split,
        "data_source": "local LIBERO HDF5 action snippets from reports/libero_offline_counterfactual_split_report.json",
        "fixed_sample_policy_used": True,
        "sample_policy": {
            "dataset_root": "LIBERO_DATA_ROOT from local config/env, redacted in report",
            "subset_name": "manifest-provided LIBERO subset",
            "sample_ordering_rule": "manifest order; positive then counterfactual sample per pair",
            "max_pairs": max_pairs,
            "max_action_steps_per_demo": max_action_steps,
            "batch_size": BATCH_SIZE,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "random_seeds_used": [],
            "excluded_samples": source_metadata["excluded_samples"],
            "exploratory": True,
        },
        "grid_size": grid_size,
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_steps": max_steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "record_count": len(records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "arms": arms,
        "comparison": comparison,
        "libero_offline_head_comparison_passed": passed,
        "libero_offline_head_training_comparison_passed": passed,
        "ready_for_required_tiny_lora_comparison": bool(passed and comparison["supports_tca_map_head_only"]),
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": (
            "Exploratory offline proxy only. ActionMap and TCA-Map train tiny NumPy head-only linear models on local "
            "LIBERO HDF5 action snippets. Distributional TCA-Select reuses the trained TCA-Map head and adds no "
            "trainable parameters. This is not standard success, rollout success, or paper-grade evidence."
        ),
        "recommended_next_step": (
            "Continue to required tiny LoRA comparison only if this head-only result is accepted as a valid exploratory loss/metric milestone."
            if passed and comparison["supports_tca_map_head_only"]
            else "Treat this as weak or blocked evidence for TCA-Map; inspect the fixed split and offline features before running LoRA."
        ),
    }


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    comparison = report["comparison"]
    lines = [
        "# LIBERO Offline ActionMap vs TCA-Map Training/Eval",
        "",
        "This is exploratory offline proxy evidence only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['libero_offline_head_training_comparison_passed']}`",
        f"- conclusion: `{comparison['conclusion']}`",
        f"- record count: `{report['record_count']}`",
        f"- train/eval records: `{report['train_record_count']}` / `{report['eval_record_count']}`",
        f"- steps: `{report['max_steps']}`",
        f"- batch size: `{report['batch_size']}`",
        f"- rollouts performed: `{report['policy']['rollouts_performed']}`",
        f"- LoRA training performed: `{report['policy']['lora_training_performed']}`",
        "",
        "## Arms",
    ]
    for name, arm in report["arms"].items():
        metrics = arm["evaluation_metrics"]
        lines.extend(
            [
                f"### `{name}`",
                f"- model/head trained: `{arm['model_head_trained']}`",
                f"- trainable parameters: `{arm['trainable_parameter_count']}`",
                f"- initial loss: `{arm['initial_loss']}`",
                f"- final loss: `{arm['final_loss']}`",
                f"- loss decreased: `{arm['loss_decreased']}`",
                f"- standard proxy score: `{metrics['standard_proxy_score']}`",
                f"- action L1: `{metrics['action_l1']}`",
                f"- target top1 accuracy: `{metrics['target_top1_accuracy']}`",
                f"- wrong-target proxy rate: `{metrics['wrong_target_proxy_rate']}`",
                f"- counterfactual separation margin: `{metrics['counterfactual_separation_margin']}`",
                f"- action-target consistency score: `{metrics['action_target_consistency_score']}`",
                "",
            ]
        )
    tca_delta = comparison["tca_map_vs_actionmap"]
    select_delta = comparison["tca_select_vs_tca_map"]
    lines.extend(
        [
            "## Key Deltas",
            "",
            f"- TCA-Map vs ActionMap standard proxy delta: `{tca_delta['standard_proxy_score_delta']}`",
            f"- TCA-Map vs ActionMap wrong-target delta: `{tca_delta['wrong_target_proxy_rate_delta']}`",
            f"- TCA-Map vs ActionMap counterfactual margin delta: `{tca_delta['counterfactual_separation_margin_delta']}`",
            f"- TCA-Select vs TCA-Map standard proxy delta: `{select_delta['standard_proxy_score_delta']}`",
            f"- TCA-Select vs TCA-Map wrong-target delta: `{select_delta['wrong_target_proxy_rate_delta']}`",
            "",
            "## Next Step",
            "",
            str(report["recommended_next_step"]),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--report-json", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_actionmap_tca_comparison_report.md")
    args = parser.parse_args()
    report = build_offline_head_comparison(
        manifest_path=Path(args.manifest),
        max_pairs=args.max_pairs,
        max_action_steps=args.max_action_steps,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    write_reports(report, Path(args.report_json), Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
