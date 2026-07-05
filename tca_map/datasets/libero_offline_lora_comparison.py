"""Tiny offline LoRA attribution comparison over local LIBERO HDF5 snippets.

This runner uses the same deterministic tiny split policy as the head-only
ActionMap vs TCA-Map comparison. It trains only small NumPy low-rank adapter
matrices on CPU. It does not load SmolVLA, import heavy VLA models, use GPU,
run simulators, run rollouts, download assets, execute OpenVLA-OFT, or make
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
    _target_ids,
    _target_margin,
    _train_lora_classifier,
    _train_lora_regressor,
    ensure_safe_environment,
    validate_smoke_bounds,
)
from tca_map.eval import compute_offline_metrics
from tca_map.inference.tca_select import distributional_tca_select_inference

SCHEMA_VERSION = "tca-map-libero-offline-lora-attribution-v1"
ACTION_PREFIX_DIM = 4
DEFAULT_HEAD_ONLY_REPORT = "reports/libero_offline_actionmap_tca_comparison_report.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing input manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_action_block(path: Path, max_steps: int = 16) -> list[list[float]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        for demo_name in sorted(data_group.keys()):
            demo = data_group[demo_name]
            if "actions" not in demo:
                continue
            actions = demo["actions"][:max_steps]
            return [[float(value) for value in row.tolist()] for row in actions]
    raise ValueError(f"{path} has no demo actions dataset")


def _mean_action(actions: list[list[float]]) -> list[float]:
    width = len(actions[0]) if actions else 0
    return [sum(row[index] for row in actions) / len(actions) for index in range(width)]


def _l1(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return sum(abs(left[index] - right[index]) for index in range(width)) / width


def _text_features(text: str, width: int = 16) -> list[float]:
    words = [word for word in text.lower().replace("_", " ").split() if word]
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    hashed = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(width - 4)]
    scalar = [
        min(len(text), 240) / 240.0,
        min(len(words), 40) / 40.0,
        sum(char in "aeiou" for char in text.lower()) / max(1, len(text)),
        sum(char.isdigit() for char in text) / max(1, len(text)),
    ]
    return scalar + hashed


def _record(
    pair: dict[str, Any],
    target_id: int,
    instruction: str,
    action: list[float],
    candidate_actions: list[list[float]],
    *,
    action_dim: int = ACTION_PREFIX_DIM,
) -> dict[str, Any]:
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if len(action) < action_dim:
        raise ValueError(f"expert action has {len(action)} values, expected at least {action_dim}")
    for index, candidate in enumerate(candidate_actions):
        if len(candidate) < action_dim:
            raise ValueError(f"candidate action {index} has {len(candidate)} values, expected at least {action_dim}")
    suffix = "positive" if target_id == 0 else "counterfactual"
    return {
        "sample_id": f"{pair['pair_id']}::{suffix}",
        "pair_id": pair["pair_id"],
        "hidden_tokens": _text_features(instruction),
        "expert_action": action[:action_dim],
        "source_action_dim": len(action),
        "record_action_dim": action_dim,
        "target": {"object_id": target_id, "instruction": instruction},
        "candidate_objects": [
            pair.get("positive_instruction") or "positive target",
            pair.get("counterfactual_instruction") or "counterfactual target",
        ],
        "candidate_actions": [candidate[:action_dim] for candidate in candidate_actions],
    }


def build_libero_lora_records(
    manifest_path: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    *,
    action_dim: int = ACTION_PREFIX_DIM,
) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")

    records: list[dict[str, Any]] = []
    for pair in manifest.get("counterfactual_pairs", [])[:max_pairs]:
        positive_action = _mean_action(_read_first_action_block(Path(pair["positive_demo_file"]), max_steps=max_action_steps))
        counter_action = _mean_action(_read_first_action_block(Path(pair["counterfactual_demo_file"]), max_steps=max_action_steps))
        candidates = [positive_action, counter_action]
        records.append(
            _record(
                pair,
                target_id=0,
                instruction=pair.get("positive_instruction") or "positive target",
                action=positive_action,
                candidate_actions=candidates,
                action_dim=action_dim,
            )
        )
        records.append(
            _record(
                pair,
                target_id=1,
                instruction=pair.get("counterfactual_instruction") or "counterfactual target",
                action=counter_action,
                candidate_actions=candidates,
                action_dim=action_dim,
            )
        )
    return records


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_ids: list[str] = []
    for record in records:
        if record["pair_id"] not in pair_ids:
            pair_ids.append(record["pair_id"])
    if len(pair_ids) <= 1:
        return records, records, {
            "split_type": "exploratory_train_eval_same_due_to_single_pair",
            "train_pair_ids": pair_ids,
            "eval_pair_ids": pair_ids,
            "exploratory": True,
            "confirmatory": False,
        }
    train_pair_count = max(1, math.ceil(len(pair_ids) * 0.75))
    train_pair_count = min(train_pair_count, len(pair_ids) - 1)
    train_pair_ids = pair_ids[:train_pair_count]
    eval_pair_ids = pair_ids[train_pair_count:]
    return (
        [record for record in records if record["pair_id"] in set(train_pair_ids)],
        [record for record in records if record["pair_id"] in set(eval_pair_ids)],
        {
            "split_type": "deterministic_manifest_order_pair_holdout",
            "sample_ordering_rule": "manifest order, positive then counterfactual",
            "train_pair_ids": train_pair_ids,
            "eval_pair_ids": eval_pair_ids,
            "random_seeds_used": [],
            "exploratory": True,
            "confirmatory": False,
        },
    )


def _candidate_separation(pred_action: list[float], target_id: int, candidate_actions: list[list[float]]) -> float:
    if len(candidate_actions) < 2:
        return 0.0
    return _l1(pred_action, candidate_actions[1 - target_id]) - _l1(pred_action, candidate_actions[target_id])


def _augment_metrics(records: list[dict[str, Any]], metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_offline_metrics(metric_records)
    margins = []
    consistency = []
    for record, metric in zip(records, metric_records):
        target_id = int(record["target"]["object_id"])
        margins.append(_candidate_separation(metric["pred_action"], target_id, record["candidate_actions"]))
        target_ok = 1.0 if int(metric["pred_target"]) == target_id else 0.0
        consistency.append(target_ok * max(0.0, 1.0 - _l1(metric["pred_action"], metric["expert_action"])))
    metrics["counterfactual_separation_margin"] = round(float(np.mean(margins)) if margins else 0.0, 6)
    metrics["action_target_consistency_score"] = round(float(np.mean(consistency)) if consistency else 0.0, 6)
    metrics["paraphrase_nuisance_stability"] = "not_available_no_paraphrase_variants"
    metrics["max_gpu_memory_mb"] = 0.0
    return metrics


def _loss_curve(losses: list[float], max_points: int = 12) -> list[dict[str, float | int]]:
    if len(losses) <= max_points:
        indices = list(range(len(losses)))
    else:
        indices = sorted({int(round(value)) for value in np.linspace(0, len(losses) - 1, num=max_points)})
    return [{"step": int(index), "loss": round(float(losses[index]), 6)} for index in indices]


def _combined_losses(action_losses: list[float], target_losses: list[float]) -> list[float]:
    if not target_losses:
        return list(action_losses)
    width = min(len(action_losses), len(target_losses))
    return [float(action_losses[index] + target_losses[index]) for index in range(width)]


def _select_actions(
    pred_actions: np.ndarray,
    logits: np.ndarray,
    pred_target_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected_actions = []
    diagnostics = []
    offsets = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [0.03, 0.0, 0.0, 0.0], [-0.03, 0.0, 0.0, 0.0], [0.0, 0.03, 0.0, 0.0]],
        dtype=np.float64,
    )
    for index, action in enumerate(pred_actions):
        target_index = int(pred_target_ids[index])
        candidates = []
        for candidate_index, offset in enumerate(offsets):
            candidate_action = np.clip(action + offset, -1.0, 1.0)
            candidates.append(
                {
                    "index": candidate_index,
                    "action": [float(value) for value in candidate_action.tolist()],
                    "voxel": candidate_index,
                    "logit": 1.0 - 0.1 * candidate_index,
                    "target_index": target_index,
                }
            )
        result = distributional_tca_select_inference(
            action_heatmap={"candidates": candidates},
            target_heatmap={"scores": [float(value) for value in logits[index].tolist()], "top_index": target_index},
            masked_action_heatmap={"candidates": [{**candidate, "logit": float(candidate["logit"]) - 0.05} for candidate in candidates]},
            K=4,
            temperature=0.5,
            metadata=None,
            external_verifier=None,
        )
        selected = result["selected"] or candidates[0]
        selected_actions.append(selected["action"])
        scores = [float(value) for value in result.get("scores", [])]
        diagnostics.append(
            {
                "candidate_count": len(candidates),
                "candidate_logits": [candidate["logit"] for candidate in candidates],
                "score_count": len(scores),
                "scores_degenerate": len({round(score, 9) for score in scores}) <= 1 if scores else True,
                "selected_index": int(selected.get("index", 0)),
                "external_verifier_used": bool(result.get("external_verifier_used")),
                "privileged_inference_used": bool(result.get("privileged_inference_used")),
            }
        )
    return np.asarray(selected_actions, dtype=np.float64), diagnostics


def _arm_report(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    arm_name: str,
    max_steps: int,
    lr: float,
    rank: int,
    grid_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_records = train_records + eval_records
    train_features = _feature_matrix(train_records)
    eval_features = _feature_matrix(eval_records)
    train_targets = _target_ids(train_records)
    eval_targets = _target_ids(eval_records)
    num_targets = _candidate_count(all_records)

    target_param_count = 0
    target_losses: list[float] = []
    train_logits = np.zeros((len(train_records), num_targets), dtype=np.float64)
    eval_logits = np.zeros((len(eval_records), num_targets), dtype=np.float64)

    if arm_name.startswith("tca_map"):
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
        train_pred_targets = np.argmax(train_logits, axis=1)
        eval_pred_targets = np.argmax(eval_logits, axis=1)
        conditioned_train = np.concatenate([train_features, _one_hot(train_targets, num_targets)], axis=1)
        conditioned_eval = np.concatenate([eval_features, _one_hot(eval_pred_targets, num_targets)], axis=1)
        target_param_count = _lora_param_count(target_a, target_b)
    else:
        train_pred_targets = np.zeros(len(train_records), dtype=np.int64)
        eval_pred_targets = np.zeros(len(eval_records), dtype=np.int64)
        conditioned_train = train_features
        conditioned_eval = eval_features

    action_base, action_a, action_b, action_losses = _train_lora_regressor(
        features=conditioned_train,
        targets=_expert_actions(train_records),
        max_steps=max_steps,
        lr=lr,
        rank=rank,
        seed=53,
    )
    train_actions = np.clip(_predict(conditioned_train, action_base, action_a, action_b), -1.0, 1.0)
    eval_actions = np.clip(_predict(conditioned_eval, action_base, action_a, action_b), -1.0, 1.0)
    select_diagnostics: list[dict[str, Any]] = []
    if arm_name == "tca_map_lora_distributional_select":
        eval_actions, select_diagnostics = _select_actions(eval_actions, eval_logits, eval_pred_targets)

    train_metric_records = _metric_records(train_records, train_actions, train_pred_targets, grid_size)
    eval_metric_records = _metric_records(eval_records, eval_actions, eval_pred_targets, grid_size)
    combined = _combined_losses(action_losses, target_losses)
    metrics = _augment_metrics(eval_records, eval_metric_records)
    metrics.update(
        {
            "mode": "tiny_lora_attribution",
            "arm": arm_name,
            "training_loss_start": round(float(action_losses[0]), 6),
            "training_loss_end": round(float(action_losses[-1]), 6),
            "training_loss_delta": round(float(action_losses[0] - action_losses[-1]), 6),
            "target_loss_start": round(float(target_losses[0]), 6) if target_losses else None,
            "target_loss_end": round(float(target_losses[-1]), 6) if target_losses else None,
            "target_margin_eval": round(_target_margin(eval_logits, eval_targets), 6) if arm_name.startswith("tca_map") else None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0 / max(1, len(eval_records)), 6),
        }
    )
    train_metrics = _augment_metrics(train_records, train_metric_records)
    return {
        "arm": arm_name,
        "lora_target_modules": ["action_head_projection"]
        if arm_name == "actionmap_lora"
        else ["target_fusion_layers", "target_classifier", "action_head_projection"],
        "max_steps": max_steps,
        "batch_size": 1,
        "learning_rate": lr,
        "lora_rank": rank,
        "initial_loss": round(float(combined[0]), 6),
        "final_loss": round(float(combined[-1]), 6),
        "loss_decreased": bool(combined[-1] < combined[0]),
        "loss_curve": _loss_curve(combined),
        "action_loss_curve": _loss_curve(action_losses),
        "target_loss_curve": _loss_curve(target_losses) if target_losses else [],
        "trainable_lora_parameter_count": int(_lora_param_count(action_a, action_b) + target_param_count),
        "frozen_base_parameter_count": int(action_base.size + (target_base.size if arm_name.startswith("tca_map") else 0)),
        "finite_losses": all(math.isfinite(loss) for loss in action_losses + target_losses),
        "train_metrics": train_metrics,
        "evaluation_metrics": metrics,
        "selection_diagnostics": select_diagnostics,
    }


def _load_head_only_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "head-only report missing"}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(path),
        "conclusion": report.get("comparison", {}).get("conclusion"),
        "split": report.get("split"),
        "actionmap_eval": report.get("arms", {}).get("actionmap_head_only", {}).get("evaluation_metrics", {}),
        "tca_map_eval": report.get("arms", {}).get("tca_map_head_only", {}).get("evaluation_metrics", {}),
        "tca_select_eval": report.get("arms", {}).get("tca_map_distributional_select", {}).get("evaluation_metrics", {}),
    }


def _same_split_as_head_only(split: dict[str, Any], reference: dict[str, Any]) -> bool | None:
    if not reference.get("available"):
        return None
    ref_split = reference.get("split") or {}
    return bool(
        split.get("train_pair_ids") == ref_split.get("train_pair_ids")
        and split.get("eval_pair_ids") == ref_split.get("eval_pair_ids")
    )


def _comparison(arms: dict[str, dict[str, Any]], head_only_reference: dict[str, Any]) -> dict[str, Any]:
    def metric(arm_name: str, metric_name: str) -> float:
        return float(arms[arm_name]["evaluation_metrics"][metric_name])

    def delta(left: str, right: str, metric_name: str) -> float:
        return round(metric(left, metric_name) - metric(right, metric_name), 6)

    tca_vs_actionmap = {
        "standard_proxy_score_delta": delta("tca_map_lora", "actionmap_lora", "standard_proxy_score"),
        "action_l1_delta": delta("tca_map_lora", "actionmap_lora", "action_l1"),
        "target_top1_accuracy_delta": delta("tca_map_lora", "actionmap_lora", "target_top1_accuracy"),
        "wrong_target_proxy_rate_delta": delta("tca_map_lora", "actionmap_lora", "wrong_target_proxy_rate"),
        "counterfactual_margin_delta": delta("tca_map_lora", "actionmap_lora", "counterfactual_separation_margin"),
        "action_target_consistency_score_delta": delta("tca_map_lora", "actionmap_lora", "action_target_consistency_score"),
        "trainable_lora_parameter_delta": int(
            arms["tca_map_lora"]["trainable_lora_parameter_count"]
            - arms["actionmap_lora"]["trainable_lora_parameter_count"]
        ),
    }
    select_vs_tca = {
        "standard_proxy_score_delta": delta("tca_map_lora_distributional_select", "tca_map_lora", "standard_proxy_score"),
        "action_l1_delta": delta("tca_map_lora_distributional_select", "tca_map_lora", "action_l1"),
        "wrong_target_proxy_rate_delta": delta("tca_map_lora_distributional_select", "tca_map_lora", "wrong_target_proxy_rate"),
        "counterfactual_margin_delta": delta(
            "tca_map_lora_distributional_select", "tca_map_lora", "counterfactual_separation_margin"
        ),
    }
    supports_tca = (
        tca_vs_actionmap["standard_proxy_score_delta"] > 0.0
        and tca_vs_actionmap["wrong_target_proxy_rate_delta"] <= 0.0
    )
    if supports_tca:
        conclusion = "lora_supports_tca_map"
    elif (
        tca_vs_actionmap["standard_proxy_score_delta"] < 0.0
        or tca_vs_actionmap["wrong_target_proxy_rate_delta"] > 0.0
    ):
        conclusion = "lora_weakens_tca_map"
    else:
        conclusion = "lora_inconclusive"
    return {
        "tca_lora_vs_actionmap_lora": tca_vs_actionmap,
        "tca_select_lora_vs_tca_lora": select_vs_tca,
        "head_only_reference_conclusion": head_only_reference.get("conclusion"),
        "conclusion": conclusion,
        "tca_select_helped": bool(
            select_vs_tca["standard_proxy_score_delta"] > 0.0
            or select_vs_tca["wrong_target_proxy_rate_delta"] < 0.0
            or select_vs_tca["action_l1_delta"] < 0.0
        ),
    }


def _sanity_checks(
    records: list[dict[str, Any]],
    split: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    head_only_reference: dict[str, Any],
) -> dict[str, Any]:
    target_ids = [int(record["target"]["object_id"]) for record in records]
    conditioning_vectors = [tuple(round(float(value), 6) for value in record["hidden_tokens"]) for record in records]
    select_diagnostics = arms["tca_map_lora_distributional_select"].get("selection_diagnostics", [])
    return {
        "target_labels_present": sorted(set(target_ids)),
        "target_labels_aligned": sorted(set(target_ids)) == [0, 1],
        "wrong_target_proxy_not_inverted": all(
            0.0 <= float(arm["evaluation_metrics"]["wrong_target_proxy_rate"]) <= 1.0 for arm in arms.values()
        ),
        "target_conditioning_non_constant": len(set(conditioning_vectors)) > 1,
        "same_split_as_head_only": _same_split_as_head_only(split, head_only_reference),
        "tca_select_candidate_scores_degenerate": any(item.get("scores_degenerate") for item in select_diagnostics),
        "tca_select_candidate_scores_checked": bool(select_diagnostics),
        "tca_select_external_verifier_used": any(item.get("external_verifier_used") for item in select_diagnostics),
        "tca_select_privileged_inference_used": any(item.get("privileged_inference_used") for item in select_diagnostics),
    }


def _policy(training_performed: bool) -> dict[str, Any]:
    return {
        "bounded_tiny_lora_attribution": True,
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


def run_libero_offline_lora_comparison(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    max_samples: int = 8,
    rank: int = DEFAULT_LORA_RANK,
    head_only_report_path: Path = Path(DEFAULT_HEAD_ONLY_REPORT),
    require_training_gate: bool = True,
) -> dict[str, Any]:
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds, max_samples=max_samples, rank=rank)

    started = time.perf_counter()
    records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)[:max_samples]
    if not records:
        raise TinyLoraSmokeError("no LIBERO HDF5 records were built for tiny LoRA comparison")
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise TinyLoraSmokeError("deterministic LoRA split did not produce train/eval records")

    arm_reports = []
    for arm_name in ["actionmap_lora", "tca_map_lora", "tca_map_lora_distributional_select"]:
        if time.perf_counter() - started > max_runtime_seconds:
            raise TinyLoraSmokeError("LIBERO offline LoRA comparison exceeded max_runtime_seconds")
        arm_reports.append(
            _arm_report(
                train_records=train_records,
                eval_records=eval_records,
                arm_name=arm_name,
                max_steps=max_steps,
                lr=0.05,
                rank=rank,
                grid_size=8,
            )
        )

    total_elapsed = time.perf_counter() - started
    arms = {arm["arm"]: arm for arm in arm_reports}
    head_only_reference = _load_head_only_reference(head_only_report_path)
    comparison = _comparison(arms, head_only_reference)
    sanity = _sanity_checks(records, split, arms, head_only_reference)
    passed = bool(
        total_elapsed <= max_runtime_seconds
        and max_steps <= 100
        and len(records) <= 200
        and all(arm.get("finite_losses") for arm in arm_reports)
        and sanity["target_labels_aligned"]
        and sanity["wrong_target_proxy_not_inverted"]
        and sanity["target_conditioning_non_constant"]
        and not sanity["tca_select_external_verifier_used"]
        and not sanity["tca_select_privileged_inference_used"]
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "head_only_reference": head_only_reference,
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
        "elapsed_seconds": round(total_elapsed, 6),
        "runtime_within_cap": total_elapsed <= max_runtime_seconds,
        "arms": arm_reports,
        "comparison": comparison,
        "sanity_checks": sanity,
        "libero_offline_lora_comparison_passed": passed,
        "ready_for_bounded_local_pilot_report": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "interpretation": (
            "Exploratory offline proxy diagnostic only. LoRA arms train tiny NumPy low-rank matrices on the same "
            "local LIBERO HDF5 split used by the head-only diagnostic. This is not standard success, not rollout "
            "success, not a full SmolVLA adapter result, and not paper-grade evidence."
        ),
        "recommended_next_step": (
            "Scale the tiny offline split only if treating this as attribution evidence; otherwise debug target conditioning."
            if passed and comparison["conclusion"] == "lora_supports_tca_map"
            else "Debug TCA target/conditioning or pivot the TCA-Map formulation before scaling."
        ),
    }
    write_reports(report, report_json=report_json, report_md=report_md)
    return report


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LIBERO Offline LoRA Attribution Comparison",
        "",
        "This is exploratory offline proxy evidence only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['libero_offline_lora_comparison_passed']}`",
        f"- conclusion: `{report['comparison']['conclusion']}`",
        f"- record count: `{report['record_count']}`",
        f"- train/eval records: `{report['train_record_count']}` / `{report['eval_record_count']}`",
        f"- max steps: `{report['max_steps']}`",
        f"- same split as head-only: `{report['sanity_checks']['same_split_as_head_only']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        "## Arms",
    ]
    for arm in report["arms"]:
        metrics = arm.get("evaluation_metrics", {})
        lines.extend(
            [
                f"### `{arm['arm']}`",
                f"- LoRA target modules: `{', '.join(arm['lora_target_modules'])}`",
                f"- initial loss: `{arm['initial_loss']}`",
                f"- final loss: `{arm['final_loss']}`",
                f"- loss decreased: `{arm['loss_decreased']}`",
                f"- trainable LoRA params: `{arm['trainable_lora_parameter_count']}`",
                f"- standard proxy score: `{metrics.get('standard_proxy_score')}`",
                f"- action L1: `{metrics.get('action_l1')}`",
                f"- target top1 accuracy: `{metrics.get('target_top1_accuracy')}`",
                f"- wrong-target proxy: `{metrics.get('wrong_target_proxy_rate')}`",
                f"- counterfactual margin: `{metrics.get('counterfactual_separation_margin')}`",
                "",
            ]
        )
    delta = report["comparison"]["tca_lora_vs_actionmap_lora"]
    select_delta = report["comparison"]["tca_select_lora_vs_tca_lora"]
    lines.extend(
        [
            "## Key Deltas",
            "",
            f"- TCA-Map + LoRA vs ActionMap + LoRA standard proxy delta: `{delta['standard_proxy_score_delta']}`",
            f"- TCA-Map + LoRA vs ActionMap + LoRA wrong-target delta: `{delta['wrong_target_proxy_rate_delta']}`",
            f"- TCA-Select + LoRA vs TCA-Map + LoRA standard proxy delta: `{select_delta['standard_proxy_score_delta']}`",
            f"- TCA-Select + LoRA vs TCA-Map + LoRA wrong-target delta: `{select_delta['wrong_target_proxy_rate_delta']}`",
            "",
            "## Next Step",
            "",
            report["recommended_next_step"],
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_lora_comparison_report.md")
    parser.add_argument("--head-only-report", default=DEFAULT_HEAD_ONLY_REPORT)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    args = parser.parse_args()

    try:
        report = run_libero_offline_lora_comparison(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            head_only_report_path=Path(args.head_only_report),
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
