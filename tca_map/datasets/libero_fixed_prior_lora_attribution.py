"""Tiny fixed-prior LoRA attribution over the fixed LIBERO proxy split.

This runner reuses the existing CPU-only NumPy LoRA machinery. It does not
load SmolVLA/OpenVLA, import heavy VLA models, use GPU, run rollouts, or make
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
    _target_margin,
    _train_lora_classifier,
    _train_lora_regressor,
    ensure_safe_environment,
    validate_smoke_bounds,
)
from tca_map.datasets.libero_offline_lora_comparison import (
    ACTION_PREFIX_DIM,
    _augment_metrics,
    _combined_losses,
    _loss_curve,
    _split_records,
    build_libero_lora_records,
)
from tca_map.inference.tca_select import distributional_tca_select_inference


SCHEMA_VERSION = "2026-07-05.fixed_prior_lora_attribution.v1"
DEFAULT_FIXED_HEAD_REPORT = "reports/libero_target_prior_fixed_head_comparison_report.json"
DEFAULT_PREVIOUS_LORA_REPORT = "reports/libero_offline_lora_comparison_report.json"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "report missing"}
    return {"available": True, "path": str(path), "report": json.loads(path.read_text(encoding="utf-8"))}


def _tokens(text: str) -> set[str]:
    stop = {"a", "an", "and", "both", "in", "it", "of", "on", "put", "the", "to"}
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token and token not in stop}


def _candidate_texts(record: dict[str, Any]) -> list[str]:
    candidates = record.get("candidate_objects") or []
    if len(candidates) >= 2:
        return [str(candidates[0]), str(candidates[1])]
    return ["target 0", "target 1"]


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return probs / row_sums


def _instruction_text_probs(records: list[dict[str, Any]], num_targets: int) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        instruction_tokens = _tokens(str(record.get("target", {}).get("instruction", "")))
        raw_scores = []
        for candidate_text in _candidate_texts(record)[:num_targets]:
            candidate_tokens = _tokens(candidate_text)
            overlap = len(instruction_tokens & candidate_tokens)
            union = max(1, len(instruction_tokens | candidate_tokens))
            raw_scores.append(overlap / union)
        while len(raw_scores) < num_targets:
            raw_scores.append(0.0)
        if len(set(round(score, 12) for score in raw_scores)) == 1:
            raw_scores = [score + 1e-3 * (len(raw_scores) - idx) for idx, score in enumerate(raw_scores)]
        rows.append(raw_scores)
    return _softmax(np.asarray(rows, dtype=np.float64))


def _temperature_calibrated_probs(logits: np.ndarray, temperature: float = 8.0) -> np.ndarray:
    return _softmax(np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6))


def _fixed_fusion_probs(
    learned_logits: np.ndarray,
    learned_probs: np.ndarray,
    text_probs: np.ndarray,
    *,
    temperature: float = 8.0,
    conflict_learned_weight: float = 0.25,
    agreement_learned_weight: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibrated = _temperature_calibrated_probs(learned_logits, temperature=temperature)
    rows: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for index, (row_learned, row_calibrated, row_text) in enumerate(zip(learned_probs, calibrated, text_probs)):
        calibrated_top1 = int(np.argmax(row_calibrated))
        text_top1 = int(np.argmax(row_text))
        conflict = calibrated_top1 != text_top1
        learned_weight = conflict_learned_weight if conflict else agreement_learned_weight
        rows.append(learned_weight * row_calibrated + (1.0 - learned_weight) * row_text)
        diagnostics.append(
            {
                "row_index": index,
                "learned_top1": int(np.argmax(row_learned)),
                "calibrated_learned_top1": calibrated_top1,
                "text_top1": text_top1,
                "learned_text_conflict": bool(conflict),
                "temperature": temperature,
                "learned_weight_used": round(float(learned_weight), 6),
            }
        )
    return _normalize_probs(np.asarray(rows, dtype=np.float64)), diagnostics


def _conditioned_features(features: np.ndarray, target_ids: np.ndarray, num_targets: int) -> np.ndarray:
    return np.concatenate([features, _one_hot(target_ids, num_targets)], axis=1)


def _candidate_actions(
    features: np.ndarray,
    action_base: np.ndarray,
    action_a: np.ndarray,
    action_b: np.ndarray,
    num_targets: int,
) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    for feature in features:
        per_target = []
        for target_id in range(num_targets):
            conditioned = _conditioned_features(feature.reshape(1, -1), np.asarray([target_id]), num_targets)
            action = np.clip(_predict(conditioned, action_base, action_a, action_b)[0], -1.0, 1.0)
            per_target.append([float(value) for value in action.tolist()])
        rows.append(per_target)
    return rows


def _select_from_target_probs(
    eval_features: np.ndarray,
    action_base: np.ndarray,
    action_a: np.ndarray,
    action_b: np.ndarray,
    target_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probs = _normalize_probs(target_probs)
    selected_targets = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    conditioned = _conditioned_features(eval_features, selected_targets, probs.shape[1])
    actions = np.clip(_predict(conditioned, action_base, action_a, action_b), -1.0, 1.0)
    diagnostics = [
        {
            "target_probs": [round(float(value), 6) for value in row.tolist()],
            "selected_target": int(target_id),
        }
        for row, target_id in zip(probs, selected_targets)
    ]
    return actions, selected_targets, diagnostics


def _select_ablation(
    eval_records: list[dict[str, Any]],
    eval_features: np.ndarray,
    action_base: np.ndarray,
    action_a: np.ndarray,
    action_b: np.ndarray,
    target_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probs = _normalize_probs(target_probs)
    all_candidates = _candidate_actions(eval_features, action_base, action_a, action_b, probs.shape[1])
    selected_actions: list[list[float]] = []
    selected_targets: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for record, row_probs, candidates_for_record in zip(eval_records, probs, all_candidates):
        candidates = []
        values = []
        for target_id, action in enumerate(candidates_for_record):
            score = float(row_probs[target_id])
            candidates.append(
                {
                    "index": int(target_id),
                    "target_index": int(target_id),
                    "voxel": int(target_id),
                    "action": action,
                    "logit": score,
                }
            )
            values.append(score)
        result = distributional_tca_select_inference(
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
            metadata={"source": "fixed_prior_lora_attribution", "target_prior_variant": "fixed_learned_text_fusion"},
            external_verifier=None,
        )
        selected = result.get("selected") or candidates[0]
        selected_action = [float(value) for value in selected.get("action", [])]
        selected_target = int(selected.get("target_index", selected.get("index", 0)))
        selected_actions.append(selected_action)
        selected_targets.append(selected_target)
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "target_probs": [round(float(value), 6) for value in row_probs.tolist()],
                "selected_target": selected_target,
                "true_target": int(record["target"]["object_id"]),
                "candidate_scores": [round(float(value), 6) for value in result.get("scores", [])],
                "external_verifier_used": bool(result.get("external_verifier_used")),
                "privileged_inference_used": bool(result.get("privileged_inference_used")),
            }
        )
    return np.asarray(selected_actions, dtype=np.float64), np.asarray(selected_targets, dtype=np.int64), diagnostics


def _train_actionmap_lora(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    max_steps: int,
    lr: float,
    rank: int,
) -> dict[str, Any]:
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    action_base, action_a, action_b, action_losses = _train_lora_regressor(
        features=train_features,
        targets=_expert_actions(train_records),
        max_steps=max_steps,
        lr=lr,
        rank=rank,
        seed=53,
    )
    return {
        "action_base": action_base,
        "action_a": action_a,
        "action_b": action_b,
        "action_losses": action_losses,
        "train_actions": np.clip(_predict(train_features, action_base, action_a, action_b), -1.0, 1.0),
        "eval_actions": np.clip(_predict(eval_features, action_base, action_a, action_b), -1.0, 1.0),
        "train_targets": np.zeros(len(train_records), dtype=np.int64),
        "eval_targets": np.zeros(len(eval_records), dtype=np.int64),
        "trainable_params": int(_lora_param_count(action_a, action_b)),
        "frozen_params": int(action_base.size),
    }


def _train_tca_lora(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    max_steps: int,
    lr: float,
    rank: int,
) -> dict[str, Any]:
    all_records = train_records + eval_records
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    train_targets = _target_ids(train_records)
    num_targets = _candidate_count(all_records)
    target_base, target_a, target_b, target_losses = _train_lora_classifier(
        features=train_features,
        target_ids=train_targets,
        num_targets=num_targets,
        max_steps=max_steps,
        lr=lr,
        rank=rank,
        seed=37,
    )
    train_logits = _predict(train_features, target_base, target_a, target_b)
    eval_logits = _predict(eval_features, target_base, target_a, target_b)
    conditioned_train = _conditioned_features(train_features, train_targets, num_targets)
    action_base, action_a, action_b, action_losses = _train_lora_regressor(
        features=conditioned_train,
        targets=_expert_actions(train_records),
        max_steps=max_steps,
        lr=lr,
        rank=rank,
        seed=53,
    )
    target_param_count = _lora_param_count(target_a, target_b)
    action_param_count = _lora_param_count(action_a, action_b)
    return {
        "num_targets": num_targets,
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
        "trainable_params": int(target_param_count + action_param_count),
        "frozen_params": int(target_base.size + action_base.size),
    }


def _metrics_for(
    records: list[dict[str, Any]],
    actions: np.ndarray,
    targets: np.ndarray,
    grid_size: int,
) -> dict[str, Any]:
    return _augment_metrics(records, _metric_records(records, actions, targets, grid_size))


def _target_topk_contains(target_probs: np.ndarray, records: list[dict[str, Any]], k: int = 2) -> float:
    probs = np.asarray(target_probs, dtype=np.float64)
    target_ids = _target_ids(records)
    if len(target_ids) == 0:
        return 0.0
    topk = np.argsort(-probs, axis=1)[:, : min(k, probs.shape[1])]
    return round(float(np.mean([int(target) in row.tolist() for target, row in zip(target_ids, topk)])), 6)


def _make_arm(
    *,
    arm: str,
    target_prior_variant: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_actions: np.ndarray,
    train_targets: np.ndarray,
    eval_actions: np.ndarray,
    eval_targets: np.ndarray,
    combined_losses: list[float],
    action_losses: list[float],
    target_losses: list[float],
    trainable_params: int,
    frozen_params: int,
    rank: int,
    max_steps: int,
    lr: float,
    grid_size: int,
    lora_target_modules: list[str],
    target_probs: np.ndarray | None = None,
    oracle: bool = False,
    select_ablation: bool = False,
    selection_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eval_metrics = _metrics_for(eval_records, eval_actions, eval_targets, grid_size)
    if target_probs is not None:
        eval_metrics["target_topk_contains_correct"] = _target_topk_contains(target_probs, eval_records, k=2)
    return {
        "arm": arm,
        "target_prior_variant": target_prior_variant,
        "oracle": bool(oracle),
        "tca_select_ablation": bool(select_ablation),
        "lora_target_modules": lora_target_modules,
        "max_steps": max_steps,
        "batch_size": 1,
        "learning_rate": lr,
        "lora_rank": rank,
        "initial_loss": round(float(combined_losses[0]), 6),
        "final_loss": round(float(combined_losses[-1]), 6),
        "loss_decreased": bool(combined_losses[-1] < combined_losses[0]),
        "loss_curve": _loss_curve(combined_losses),
        "action_loss_curve": _loss_curve(action_losses),
        "target_loss_curve": _loss_curve(target_losses) if target_losses else [],
        "trainable_lora_parameter_count": int(trainable_params),
        "frozen_base_parameter_count": int(frozen_params),
        "finite_losses": all(math.isfinite(loss) for loss in action_losses + target_losses),
        "train_metrics": _metrics_for(train_records, train_actions, train_targets, grid_size),
        "evaluation_metrics": eval_metrics,
        "selection_diagnostics": selection_diagnostics or [],
    }


def _reference_metric(reference: dict[str, Any], arm: str, metric: str) -> float | None:
    if not reference.get("available"):
        return None
    report = reference.get("report") or {}
    for item in report.get("arms", []):
        if item.get("arm") == arm:
            value = (item.get("evaluation_metrics") or {}).get(metric)
            return None if value is None else float(value)
    return None


def _add_oracle_gaps(arms: list[dict[str, Any]]) -> None:
    oracle = next((arm for arm in arms if arm["arm"] == "oracle_target_tca_lora_upper_bound"), None)
    if not oracle:
        return
    oracle_score = float(oracle["evaluation_metrics"]["standard_proxy_score"])
    for arm in arms:
        arm["evaluation_metrics"]["gap_to_oracle_target_tca_lora_standard_proxy"] = round(
            oracle_score - float(arm["evaluation_metrics"]["standard_proxy_score"]),
            6,
        )


def _comparison(
    arms: dict[str, dict[str, Any]],
    fixed_head_reference: dict[str, Any],
    previous_lora_reference: dict[str, Any],
) -> dict[str, Any]:
    def metric(arm_name: str, metric_name: str) -> float:
        return float(arms[arm_name]["evaluation_metrics"][metric_name])

    def delta(left: str, right: str, metric_name: str) -> float:
        return round(metric(left, metric_name) - metric(right, metric_name), 6)

    fixed_vs_actionmap = {
        "standard_proxy_score_delta": delta("tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "standard_proxy_score"),
        "wrong_target_proxy_rate_delta": delta("tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "wrong_target_proxy_rate"),
        "action_target_consistency_score_delta": delta(
            "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "action_target_consistency_score"
        ),
        "counterfactual_margin_delta": delta(
            "tca_map_lora_fixed_learned_text_fusion", "actionmap_lora", "counterfactual_separation_margin"
        ),
    }
    select_vs_fixed = {
        "standard_proxy_score_delta": delta(
            "tca_map_lora_fixed_fusion_tca_select_ablation",
            "tca_map_lora_fixed_learned_text_fusion",
            "standard_proxy_score",
        ),
        "wrong_target_proxy_rate_delta": delta(
            "tca_map_lora_fixed_fusion_tca_select_ablation",
            "tca_map_lora_fixed_learned_text_fusion",
            "wrong_target_proxy_rate",
        ),
        "action_target_consistency_score_delta": delta(
            "tca_map_lora_fixed_fusion_tca_select_ablation",
            "tca_map_lora_fixed_learned_text_fusion",
            "action_target_consistency_score",
        ),
    }
    previous_hard_lora = _reference_metric(previous_lora_reference, "tca_map_lora", "standard_proxy_score")
    previous_hard_delta = None
    if previous_hard_lora is not None:
        previous_hard_delta = round(
            metric("tca_map_lora_fixed_learned_text_fusion", "standard_proxy_score") - previous_hard_lora,
            6,
        )
    fixed_beats_actionmap = (
        fixed_vs_actionmap["standard_proxy_score_delta"] > 0.0
        and fixed_vs_actionmap["wrong_target_proxy_rate_delta"] <= 0.0
    )
    select_meaningful = (
        select_vs_fixed["standard_proxy_score_delta"] >= 0.01
        or select_vs_fixed["wrong_target_proxy_rate_delta"] < 0.0
        or (
            select_vs_fixed["action_target_consistency_score_delta"] > 0.0
            and select_vs_fixed["standard_proxy_score_delta"] >= 0.0
        )
    )
    if fixed_beats_actionmap:
        conclusion = "fixed_prior_lora_supports_tca_map"
        recommendation = "A_cautiously_scale_offline_split"
    else:
        conclusion = "fixed_prior_lora_does_not_support_tca_map"
        recommendation = "B_redesign_learned_target_head"
    if not select_meaningful:
        selector_recommendation = "C_deemphasize_or_kill_TCA_Select"
    else:
        selector_recommendation = "keep_TCA_Select_as_secondary_ablation"
    return {
        "fixed_prior_tca_lora_vs_actionmap_lora": fixed_vs_actionmap,
        "tca_select_ablation_vs_fixed_prior_tca_lora": select_vs_fixed,
        "fixed_prior_tca_beats_actionmap_lora": bool(fixed_beats_actionmap),
        "tca_select_meaningful_gain": bool(select_meaningful),
        "selector_recommendation": selector_recommendation,
        "previous_hard_learned_lora_standard_proxy": previous_hard_lora,
        "fixed_prior_vs_previous_hard_lora_standard_proxy_delta": previous_hard_delta,
        "fixed_head_reference_available": bool(fixed_head_reference.get("available")),
        "previous_lora_reference_available": bool(previous_lora_reference.get("available")),
        "conclusion": conclusion,
        "recommended_next_milestone": recommendation,
    }


def _policy(training_performed: bool) -> dict[str, Any]:
    return {
        "bounded_tiny_fixed_prior_lora_attribution": True,
        "risk_assessed_autonomy_for_tiny_training_smoke": True,
        "local_libero_hdf5_used": True,
        "real_dataset_used": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "backbone_frozen": True,
        "trainable_lora_adapter_weights_only": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "adapter_construction_performed": training_performed,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": training_performed,
        "lora_training_performed": training_performed,
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
        "# Fixed-Prior LIBERO Offline LoRA Attribution",
        "",
        "Exploratory offline proxy only. This is not standard success, rollout evidence, a full SmolVLA LoRA result, or paper-grade evidence.",
        "",
        f"- passed: `{report['fixed_prior_lora_attribution_passed']}`",
        f"- conclusion: `{report['comparison']['conclusion']}`",
        f"- recommended next milestone: `{report['comparison']['recommended_next_milestone']}`",
        f"- selector recommendation: `{report['comparison']['selector_recommendation']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- LoRA training happened: `{report['policy']['lora_training_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        "",
        "## Arms",
    ]
    for arm in report["arms"]:
        metrics = arm["evaluation_metrics"]
        lines.extend(
            [
                f"### `{arm['arm']}`",
                f"- target prior variant: `{arm['target_prior_variant']}`",
                f"- LoRA target modules: `{', '.join(arm['lora_target_modules'])}`",
                f"- trainable LoRA params: `{arm['trainable_lora_parameter_count']}`",
                f"- loss: `{arm['initial_loss']} -> {arm['final_loss']}`",
                f"- standard proxy: `{metrics['standard_proxy_score']}`",
                f"- wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- action-target consistency: `{metrics['action_target_consistency_score']}`",
                f"- counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
                f"- gap to oracle: `{metrics.get('gap_to_oracle_target_tca_lora_standard_proxy')}`",
                "",
            ]
        )
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_fixed_prior_lora_attribution(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    max_samples: int = 8,
    rank: int = DEFAULT_LORA_RANK,
    fixed_head_report_path: Path = Path(DEFAULT_FIXED_HEAD_REPORT),
    previous_lora_report_path: Path = Path(DEFAULT_PREVIOUS_LORA_REPORT),
    require_training_gate: bool = True,
) -> dict[str, Any]:
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds, max_samples=max_samples, rank=rank)
    if max_samples != 8:
        raise TinyLoraSmokeError("fixed-prior LoRA attribution must use the fixed 8-sample split first")

    started = time.perf_counter()
    records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)[:max_samples]
    if len(records) != 8:
        raise TinyLoraSmokeError("fixed-prior LoRA attribution expected exactly 8 records")
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise TinyLoraSmokeError("deterministic LoRA split did not produce train/eval records")

    lr = 0.05
    grid_size = 8
    actionmap = _train_actionmap_lora(train_records, eval_records, max_steps=max_steps, lr=lr, rank=rank)
    tca = _train_tca_lora(train_records, eval_records, max_steps=max_steps, lr=lr, rank=rank)
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    num_targets = int(tca["num_targets"])

    learned_train_probs = _softmax(tca["train_logits"])
    learned_eval_probs = _softmax(tca["eval_logits"])
    text_train_probs = _instruction_text_probs(train_records, num_targets)
    text_eval_probs = _instruction_text_probs(eval_records, num_targets)
    fixed_train_probs, fixed_train_diag = _fixed_fusion_probs(tca["train_logits"], learned_train_probs, text_train_probs)
    fixed_eval_probs, fixed_eval_diag = _fixed_fusion_probs(tca["eval_logits"], learned_eval_probs, text_eval_probs)

    tca_combined_losses = _combined_losses(tca["action_losses"], tca["target_losses"])
    actionmap_losses = _combined_losses(actionmap["action_losses"], [])
    tca_modules = ["target_fusion_layers", "target_classifier", "action_head_projection"]

    def tca_variant(
        arm: str,
        target_prior_variant: str,
        train_probs: np.ndarray,
        eval_probs: np.ndarray,
        *,
        oracle: bool = False,
        select_ablation: bool = False,
    ) -> dict[str, Any]:
        if select_ablation:
            eval_actions, eval_targets, select_diag = _select_ablation(
                eval_records,
                eval_features,
                tca["action_base"],
                tca["action_a"],
                tca["action_b"],
                eval_probs,
            )
            train_actions, train_targets, _ = _select_from_target_probs(
                train_features, tca["action_base"], tca["action_a"], tca["action_b"], train_probs
            )
        else:
            train_actions, train_targets, _ = _select_from_target_probs(
                train_features, tca["action_base"], tca["action_a"], tca["action_b"], train_probs
            )
            eval_actions, eval_targets, select_diag = _select_from_target_probs(
                eval_features, tca["action_base"], tca["action_a"], tca["action_b"], eval_probs
            )
        return _make_arm(
            arm=arm,
            target_prior_variant=target_prior_variant,
            train_records=train_records,
            eval_records=eval_records,
            train_actions=train_actions,
            train_targets=train_targets,
            eval_actions=eval_actions,
            eval_targets=eval_targets,
            combined_losses=tca_combined_losses,
            action_losses=tca["action_losses"],
            target_losses=tca["target_losses"],
            trainable_params=tca["trainable_params"],
            frozen_params=tca["frozen_params"],
            rank=rank,
            max_steps=max_steps,
            lr=lr,
            grid_size=grid_size,
            lora_target_modules=tca_modules,
            target_probs=eval_probs,
            oracle=oracle,
            select_ablation=select_ablation,
            selection_diagnostics=select_diag if select_ablation else [],
        )

    oracle_train_probs = _one_hot(_target_ids(train_records), num_targets)
    oracle_eval_probs = _one_hot(_target_ids(eval_records), num_targets)
    arms = [
        _make_arm(
            arm="actionmap_lora",
            target_prior_variant="none_actionmap_baseline",
            train_records=train_records,
            eval_records=eval_records,
            train_actions=actionmap["train_actions"],
            train_targets=actionmap["train_targets"],
            eval_actions=actionmap["eval_actions"],
            eval_targets=actionmap["eval_targets"],
            combined_losses=actionmap_losses,
            action_losses=actionmap["action_losses"],
            target_losses=[],
            trainable_params=actionmap["trainable_params"],
            frozen_params=actionmap["frozen_params"],
            rank=rank,
            max_steps=max_steps,
            lr=lr,
            grid_size=grid_size,
            lora_target_modules=["action_head_projection"],
        ),
        tca_variant("tca_map_lora_hard_learned_target", "hard_learned_target", learned_train_probs, learned_eval_probs),
        tca_variant("tca_map_lora_instruction_text_prior", "instruction_text_prior", text_train_probs, text_eval_probs),
        tca_variant("tca_map_lora_fixed_learned_text_fusion", "fixed_learned_text_fusion", fixed_train_probs, fixed_eval_probs),
        tca_variant("oracle_target_tca_lora_upper_bound", "oracle_target_upper_bound", oracle_train_probs, oracle_eval_probs, oracle=True),
        tca_variant(
            "tca_map_lora_fixed_fusion_tca_select_ablation",
            "fixed_learned_text_fusion_select_ablation",
            fixed_train_probs,
            fixed_eval_probs,
            select_ablation=True,
        ),
    ]
    _add_oracle_gaps(arms)
    arm_map = {arm["arm"]: arm for arm in arms}
    fixed_head_ref = _load_json_if_exists(fixed_head_report_path)
    previous_lora_ref = _load_json_if_exists(previous_lora_report_path)
    comparison = _comparison(arm_map, fixed_head_ref, previous_lora_ref)
    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and max_steps <= 100
        and len(records) <= 200
        and all(arm["finite_losses"] for arm in arms)
        and not any(item.get("external_verifier_used") or item.get("privileged_inference_used") for item in arm_map["tca_map_lora_fixed_fusion_tca_select_ablation"]["selection_diagnostics"])
    )
    interpretation = (
        "Fixed-prior TCA + LoRA beats ActionMap + LoRA on this fixed tiny offline proxy split. "
        "This supports keeping TCA-Map viable under a corrected target prior, but it is not paper-grade and should be scaled cautiously."
        if comparison["fixed_prior_tca_beats_actionmap_lora"]
        else "Fixed-prior TCA + LoRA does not beat ActionMap + LoRA on this fixed tiny split; revise the formulation before scaling."
    )
    if not comparison["tca_select_meaningful_gain"]:
        interpretation += " TCA-Select remains unsupported as a core contribution in this diagnostic."

    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "fixed_head_reference": {
            "available": fixed_head_ref.get("available"),
            "path": fixed_head_ref.get("path"),
        },
        "previous_lora_reference": {
            "available": previous_lora_ref.get("available"),
            "path": previous_lora_ref.get("path"),
        },
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_samples": max_samples,
        "max_steps": max_steps,
        "max_runtime_seconds": max_runtime_seconds,
        "lora_rank": rank,
        "action_prefix_dim": ACTION_PREFIX_DIM,
        "record_count": len(records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "split": split,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "fusion_diagnostics": {
            "train": fixed_train_diag,
            "eval": fixed_eval_diag,
            "fixed_learned_temperature": 8.0,
            "conflict_learned_weight": 0.25,
            "agreement_learned_weight": 0.5,
        },
        "arms": arms,
        "comparison": comparison,
        "fixed_prior_lora_attribution_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": interpretation,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_fixed_prior_lora_attribution_report.json")
    parser.add_argument("--report-md", default="reports/libero_fixed_prior_lora_attribution_report.md")
    parser.add_argument("--fixed-head-report", default=DEFAULT_FIXED_HEAD_REPORT)
    parser.add_argument("--previous-lora-report", default=DEFAULT_PREVIOUS_LORA_REPORT)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    args = parser.parse_args()
    try:
        report = run_fixed_prior_lora_attribution(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            fixed_head_report_path=Path(args.fixed_head_report),
            previous_lora_report_path=Path(args.previous_lora_report),
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
