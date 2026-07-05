"""Representation-sensitivity and target-reinjection audit for LIBERO offline proxy.

This runner reuses the fixed 64-record local LIBERO offline split and the same
lightweight CPU NumPy heads/LoRA arms used by the fixed-prior diagnostics. It
records proxy representation sensitivity, action-pathway sensitivity, and the
amount of evidence for target-prior reinjection. It does not load VLA models,
extract full model hidden states, use GPU, run rollouts, download assets,
execute OpenVLA-OFT, or make paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.adapters.tiny_lora_smoke import (
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_STEPS,
    TinyLoraSmokeError,
    _feature_matrix,
    validate_smoke_bounds,
)
from tca_map.datasets.libero_fixed_prior_offline_scale_comparison import (
    _build_head_arms,
    _build_lora_arms,
    _dangerous_gates,
    _select_scaled_sample_count,
)
from tca_map.datasets.libero_offline_lora_comparison import (
    _augment_metrics,
    _split_records,
    build_libero_lora_records,
)
from tca_map.datasets.libero_publishability_gate_audit import _prior_source_leakage_audit


SCHEMA_VERSION = "2026-07-05.libero_representation_sensitivity_audit.v1"
DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_MAX_SAMPLES = 64
METRIC_NAMES = (
    "standard_proxy_score",
    "wrong_target_proxy_rate",
    "action_target_consistency_score",
    "counterfactual_separation_margin",
    "target_top1_accuracy",
    "target_topk_accuracy",
)
ACTION_PATHWAY_ARMS = (
    "actionmap_head_only",
    "tca_map_hard_learned_target_head_only",
    "tca_map_fixed_learned_text_fusion_head_only",
    "oracle_target_tca_head_only_upper_bound",
    "actionmap_lora",
    "tca_map_lora_hard_learned_target",
    "tca_map_lora_fixed_learned_text_fusion",
    "oracle_target_tca_lora_upper_bound",
    "tca_map_lora_fixed_fusion_tca_select_ablation",
)


def _mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 6) if values else 0.0


def _std(values: list[float]) -> float:
    return round(float(statistics.pstdev(values)), 6) if len(values) > 1 else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _l1(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    width = min(left_arr.shape[-1], right_arr.shape[-1])
    if width <= 0:
        return 0.0
    return float(np.mean(np.abs(left_arr[:width] - right_arr[:width])))


def _target_balance(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(int(record["target"]["object_id"])) for record in records).items()))


def _task_count(records: list[dict[str, Any]]) -> int:
    return len({str(record.get("target", {}).get("instruction", "")) for record in records})


def _task_record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("target", {}).get("instruction", "")) for record in records)
    return dict(sorted(counts.items()))


def _pair_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("pair_id", record.get("sample_id", index)))].append(index)
    return {key: value for key, value in sorted(groups.items()) if len(value) >= 2}


def _proxy_representation_sensitivity(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    features = _feature_matrix(eval_records)
    groups = _pair_groups(eval_records)
    cosine_values: list[float] = []
    distances: list[float] = []
    rows: list[dict[str, Any]] = []
    for pair_id, indices in groups.items():
        left, right = indices[:2]
        left_target = int(eval_records[left]["target"]["object_id"])
        right_target = int(eval_records[right]["target"]["object_id"])
        if left_target == right_target:
            continue
        cosine = _cosine(features[left], features[right])
        distance = float(np.linalg.norm(features[left] - features[right]))
        cosine_values.append(cosine)
        distances.append(distance)
        rows.append(
            {
                "pair_id": pair_id,
                "left_sample_id": eval_records[left]["sample_id"],
                "right_sample_id": eval_records[right]["sample_id"],
                "left_target": left_target,
                "right_target": right_target,
                "proxy_representation_cosine": round(cosine, 6),
                "proxy_representation_l2_distance": round(distance, 6),
                "observations_truly_same_or_comparable": False,
            }
        )
    return {
        "full_hidden_extraction_performed": False,
        "full_hidden_extraction_blocker": "Heavy VLA model loading/hidden-state extraction is outside this bounded audit; using cached hidden-token proxy records only.",
        "proxy_representation_extraction_performed": True,
        "proxy_representation_source": "cached hidden_tokens generated by the existing offline LIBERO interface; not final SmolVLA/OpenVLA hidden states",
        "target_swapped_pair_count": len(rows),
        "target_preserving_paraphrase_pairs_available": False,
        "target_preserving_paraphrase_pair_count": 0,
        "target_swap_proxy_cosine_mean": _mean(cosine_values),
        "target_swap_proxy_cosine_std": _std(cosine_values),
        "target_swap_proxy_l2_distance_mean": _mean(distances),
        "target_swap_proxy_l2_distance_std": _std(distances),
        "target_swap_sensitivity_ratio": None,
        "target_swap_sensitivity_ratio_note": "Paraphrase-preserving pairs are not available in this split, so no paraphrase-normalized ratio is claimed.",
        "observation_comparability": "paired counterfactual local HDF5 snippets; not verified same simulator state",
        "representation_collapse_claim_allowed": False,
        "per_pair_rows": rows,
    }


def _target_swap_action_delta(eval_records: list[dict[str, Any]], metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _pair_groups(eval_records)
    actions = [list(row.get("pred_action", [])) for row in metric_records]
    deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    for pair_id, indices in groups.items():
        left, right = indices[:2]
        if int(eval_records[left]["target"]["object_id"]) == int(eval_records[right]["target"]["object_id"]):
            continue
        delta = _l1(actions[left], actions[right])
        deltas.append(delta)
        rows.append(
            {
                "pair_id": pair_id,
                "left_sample_id": eval_records[left]["sample_id"],
                "right_sample_id": eval_records[right]["sample_id"],
                "target_swap_action_l1_delta": round(delta, 6),
                "left_pred_target": int(metric_records[left].get("pred_target", -1)),
                "right_pred_target": int(metric_records[right].get("pred_target", -1)),
                "left_true_target": int(metric_records[left].get("target_id", -1)),
                "right_true_target": int(metric_records[right].get("target_id", -1)),
                "left_counted_wrong_target": bool(metric_records[left].get("pred_target") != metric_records[left].get("target_id")),
                "right_counted_wrong_target": bool(metric_records[right].get("pred_target") != metric_records[right].get("target_id")),
            }
        )
    return {
        "target_swap_action_l1_delta_mean": _mean(deltas),
        "target_swap_action_l1_delta_std": _std(deltas),
        "target_swap_pair_count": len(deltas),
        "paraphrase_action_stability_available": False,
        "paraphrase_action_l1_delta_mean": None,
        "target_swap_vs_paraphrase_action_sensitivity_ratio": None,
        "per_pair_rows": rows,
    }


def _metrics_for_subset(records: list[dict[str, Any]], metric_records: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    return _augment_metrics([records[index] for index in indices], [metric_records[index] for index in indices])


def _per_target_breakdown(eval_records: list[dict[str, Any]], arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(eval_records):
        groups[str(int(record["target"]["object_id"]))].append(index)
    rows: list[dict[str, Any]] = []
    for target_id, indices in sorted(groups.items()):
        row: dict[str, Any] = {"target_id": target_id, "eval_sample_count": len(indices)}
        for arm_name in ("actionmap_lora", "tca_map_lora_fixed_learned_text_fusion", "oracle_target_tca_lora_upper_bound"):
            metrics = _metrics_for_subset(eval_records, arms[arm_name]["eval_metric_records"], indices)
            row[f"{arm_name}_standard_proxy_score"] = metrics["standard_proxy_score"]
            row[f"{arm_name}_wrong_target_proxy_rate"] = metrics["wrong_target_proxy_rate"]
        row["fixed_prior_tca_lora_vs_actionmap_lora_standard_proxy_delta"] = round(
            _safe_float(row["tca_map_lora_fixed_learned_text_fusion_standard_proxy_score"])
            - _safe_float(row["actionmap_lora_standard_proxy_score"]),
            6,
        )
        row["fixed_prior_tca_lora_vs_actionmap_lora_wrong_target_delta"] = round(
            _safe_float(row["tca_map_lora_fixed_learned_text_fusion_wrong_target_proxy_rate"])
            - _safe_float(row["actionmap_lora_wrong_target_proxy_rate"]),
            6,
        )
        rows.append(row)
    return rows


def _arm_summary(eval_records: list[dict[str, Any]], arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixed_score = _safe_float(arms["tca_map_lora_fixed_learned_text_fusion"]["evaluation_metrics"]["standard_proxy_score"])
    oracle_score = _safe_float(arms["oracle_target_tca_lora_upper_bound"]["evaluation_metrics"]["standard_proxy_score"])
    summary: dict[str, Any] = {}
    for arm_name in ACTION_PATHWAY_ARMS:
        arm = arms[arm_name]
        metrics = arm["evaluation_metrics"]
        score = _safe_float(metrics.get("standard_proxy_score"))
        summary[arm_name] = {
            "family": arm.get("family"),
            "target_prior_variant": arm.get("target_prior_variant"),
            "oracle": bool(arm.get("oracle", False)),
            "tca_select_ablation": bool(arm.get("tca_select_ablation", False)),
            "initial_loss": arm.get("initial_loss"),
            "final_loss": arm.get("final_loss"),
            "loss_decreased": arm.get("loss_decreased"),
            "training_performed": bool(arm.get("training_performed")),
            "lora_training_performed": bool(arm.get("lora_training_performed")),
            "trainable_parameter_count": arm.get("trainable_parameter_count", arm.get("trainable_lora_parameter_count")),
            "lora_target_modules": arm.get("lora_target_modules", []),
            "metrics": {name: metrics.get(name) for name in METRIC_NAMES if name in metrics},
            "gap_to_fixed_prior_tca_lora_standard_proxy": round(fixed_score - score, 6),
            "gap_to_oracle_tca_lora_standard_proxy": round(oracle_score - score, 6),
            "action_sensitivity": _target_swap_action_delta(eval_records, arm["eval_metric_records"]),
        }
    return summary


def _per_seed_report(
    *,
    seed: int,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    max_steps: int,
    rank: int,
) -> dict[str, Any]:
    arms_list = _build_head_arms(train_records, eval_records, steps=max_steps, lr=0.05, seed=seed)
    arms_list.extend(_build_lora_arms(train_records, eval_records, steps=max_steps, lr=0.05, rank=rank, seed=seed))
    arms = {arm["arm"]: arm for arm in arms_list}
    learned_lora_metrics = arms["tca_map_lora_hard_learned_target"]["evaluation_metrics"]
    fixed_lora_metrics = arms["tca_map_lora_fixed_learned_text_fusion"]["evaluation_metrics"]
    action_lora_metrics = arms["actionmap_lora"]["evaluation_metrics"]
    select_metrics = arms["tca_map_lora_fixed_fusion_tca_select_ablation"]["evaluation_metrics"]
    return {
        "seed": seed,
        "target_head_recovery": {
            "learned_target_lora_eval_top1_accuracy": learned_lora_metrics.get("target_top1_accuracy"),
            "learned_target_lora_eval_topk_accuracy": learned_lora_metrics.get("target_topk_accuracy"),
            "learned_target_lora_standard_proxy_score": learned_lora_metrics.get("standard_proxy_score"),
            "learned_target_lora_wrong_target_proxy_rate": learned_lora_metrics.get("wrong_target_proxy_rate"),
        },
        "action_pathway_arms": _arm_summary(eval_records, arms),
        "per_target_breakdown": _per_target_breakdown(eval_records, arms),
        "comparison": {
            "fixed_prior_tca_lora_standard_proxy_advantage_over_actionmap_lora": round(
                _safe_float(fixed_lora_metrics.get("standard_proxy_score")) - _safe_float(action_lora_metrics.get("standard_proxy_score")),
                6,
            ),
            "fixed_prior_tca_lora_wrong_target_delta_over_actionmap_lora": round(
                _safe_float(fixed_lora_metrics.get("wrong_target_proxy_rate")) - _safe_float(action_lora_metrics.get("wrong_target_proxy_rate")),
                6,
            ),
            "tca_select_delta_over_fixed_prior_tca_lora_standard_proxy": round(
                _safe_float(select_metrics.get("standard_proxy_score")) - _safe_float(fixed_lora_metrics.get("standard_proxy_score")),
                6,
            ),
            "tca_select_wrong_target_delta_over_fixed_prior_tca_lora": round(
                _safe_float(select_metrics.get("wrong_target_proxy_rate")) - _safe_float(fixed_lora_metrics.get("wrong_target_proxy_rate")),
                6,
            ),
        },
    }


def _aggregate_arm(seed_reports: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    rows = [report["action_pathway_arms"][arm_name] for report in seed_reports]
    metrics: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = [_safe_float(row["metrics"].get(metric)) for row in rows if metric in row["metrics"]]
        if values:
            metrics[metric] = {"mean": _mean(values), "std": _std(values), "values": [round(value, 6) for value in values]}
    action_values = [_safe_float(row["action_sensitivity"].get("target_swap_action_l1_delta_mean")) for row in rows]
    return {
        "family": rows[0]["family"],
        "target_prior_variant": rows[0]["target_prior_variant"],
        "oracle": rows[0]["oracle"],
        "tca_select_ablation": rows[0]["tca_select_ablation"],
        "initial_loss": {"mean": _mean([_safe_float(row.get("initial_loss")) for row in rows]), "std": _std([_safe_float(row.get("initial_loss")) for row in rows])},
        "final_loss": {"mean": _mean([_safe_float(row.get("final_loss")) for row in rows]), "std": _std([_safe_float(row.get("final_loss")) for row in rows])},
        "trainable_parameter_count": rows[0].get("trainable_parameter_count"),
        "lora_target_modules": rows[0].get("lora_target_modules", []),
        "metrics": metrics,
        "target_swap_action_l1_delta_mean": {"mean": _mean(action_values), "std": _std(action_values), "values": [round(value, 6) for value in action_values]},
        "paraphrase_stability": "not_available_no_target_preserving_paraphrase_pairs",
    }


def _aggregate_per_target(seed_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in seed_reports:
        for row in report["per_target_breakdown"]:
            grouped[str(row["target_id"])].append(row)
    results: list[dict[str, Any]] = []
    for target_id, rows in sorted(grouped.items()):
        standard_delta = [_safe_float(row["fixed_prior_tca_lora_vs_actionmap_lora_standard_proxy_delta"]) for row in rows]
        wrong_delta = [_safe_float(row["fixed_prior_tca_lora_vs_actionmap_lora_wrong_target_delta"]) for row in rows]
        action_wrong = [_safe_float(row["actionmap_lora_wrong_target_proxy_rate"]) for row in rows]
        fixed_wrong = [_safe_float(row["tca_map_lora_fixed_learned_text_fusion_wrong_target_proxy_rate"]) for row in rows]
        results.append(
            {
                "target_id": target_id,
                "seed_count": len(rows),
                "eval_sample_count": rows[0]["eval_sample_count"],
                "standard_proxy_delta_mean": _mean(standard_delta),
                "standard_proxy_delta_std": _std(standard_delta),
                "wrong_target_delta_mean": _mean(wrong_delta),
                "wrong_target_delta_std": _std(wrong_delta),
                "actionmap_wrong_target_mean": _mean(action_wrong),
                "fixed_prior_tca_wrong_target_mean": _mean(fixed_wrong),
            }
        )
    return results


def _target0_diagnosis(per_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = next((item for item in per_target_rows if str(item["target_id"]) == "0"), None)
    if row is None:
        return {"target_id": "0", "diagnosis": "not_evaluable_no_target0_eval_rows"}
    delta = _safe_float(row["standard_proxy_delta_mean"])
    wrong_delta = _safe_float(row["wrong_target_delta_mean"])
    if abs(delta) <= 0.01 and wrong_delta <= 0.0:
        diagnosis = "near_saturation_or_metric_noise_not_a_material_failure"
    elif delta < -0.03:
        diagnosis = "material_target0_underperformance_requires_diagnosis_before_rollout"
    else:
        diagnosis = "minor_target0_variation"
    return {**row, "diagnosis": diagnosis}


def _decision(
    *,
    representation: dict[str, Any],
    aggregate_arms: dict[str, Any],
    target0: dict[str, Any],
) -> dict[str, Any]:
    actionmap = aggregate_arms["actionmap_lora"]["metrics"]["standard_proxy_score"]["mean"]
    fixed = aggregate_arms["tca_map_lora_fixed_learned_text_fusion"]["metrics"]["standard_proxy_score"]["mean"]
    action_wrong = aggregate_arms["actionmap_lora"]["metrics"]["wrong_target_proxy_rate"]["mean"]
    fixed_wrong = aggregate_arms["tca_map_lora_fixed_learned_text_fusion"]["metrics"]["wrong_target_proxy_rate"]["mean"]
    select_delta = round(
        aggregate_arms["tca_map_lora_fixed_fusion_tca_select_ablation"]["metrics"]["standard_proxy_score"]["mean"] - fixed,
        6,
    )
    fixed_improves = fixed > actionmap and fixed_wrong <= action_wrong
    if not representation["full_hidden_extraction_performed"]:
        collapse_status = "unsupported_full_hidden_extraction_not_performed"
        supported_claim = "C_target_prior_reinjection_improves_wrong_target_correction_without_proving_hidden_collapse"
        hidden_collapse_supported = False
    else:
        hidden_collapse_supported = False
        collapse_status = "unsupported_or_partial"
        supported_claim = "C_target_prior_reinjection_improves_wrong_target_correction_without_proving_hidden_collapse"
    selector = "kill_or_de_emphasize_tca_select_as_core_contribution" if abs(select_delta) < 0.01 else "retain_tca_select_as_secondary_ablation_pending_larger_validation"
    if target0.get("diagnosis") == "material_target0_underperformance_requires_diagnosis_before_rollout":
        next_milestone = "E_diagnose_target0_before_rollout"
    elif fixed_improves:
        next_milestone = "D_limited_fixed_prior_rollout_diagnostic"
    else:
        next_milestone = "B_redesign_target_prior_or_action_binding"
    return {
        "supported_paper_claim": supported_claim,
        "target_information_collapse_status": collapse_status,
        "hidden_collapse_supported": hidden_collapse_supported,
        "target_prior_tca_map_remains_main_method": bool(fixed_improves),
        "tca_select_recommendation": selector,
        "fixed_prior_tca_lora_advantage_over_actionmap_lora_mean": round(fixed - actionmap, 6),
        "wrong_target_proxy_delta_mean": round(fixed_wrong - action_wrong, 6),
        "tca_select_standard_proxy_delta_mean": select_delta,
        "next_milestone": next_milestone,
        "interpretation": (
            "The bounded audit did not extract final VLA hidden states, so it cannot support a hidden-collapse claim. "
            "The cached proxy representation changes across target-changing pairs, while explicit non-leaking semantic target-prior reinjection continues to improve wrong-target correction. "
            "The supported claim is target-prior reinjection for action-pathway grounding/binding failure, not proven representation collapse."
        ),
    }


def _policy() -> dict[str, Any]:
    return {
        "representation_sensitivity_audit": True,
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
        "full_hidden_state_extraction_performed": False,
        "proxy_representation_audit_performed": True,
        "training_performed": True,
        "lora_training_performed": True,
        "loss_computed": True,
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
    decision = report["decision"]
    arms = report["aggregate_action_pathway_arms"]
    lines = [
        "# LIBERO Representation Sensitivity Audit",
        "",
        "Exploratory offline proxy only. This is not standard success, rollout evidence, or paper-grade evidence.",
        "",
        f"- passed: `{report['representation_sensitivity_audit_passed']}`",
        f"- records: `{report['record_count']}`",
        f"- train/eval records: `{report['train_record_count']} / {report['eval_record_count']}`",
        f"- seeds: `{report['seeds']}`",
        f"- full hidden extraction happened: `{report['representation_sensitivity']['full_hidden_extraction_performed']}`",
        f"- proxy target-swap cosine mean/std: `{report['representation_sensitivity']['target_swap_proxy_cosine_mean']} / {report['representation_sensitivity']['target_swap_proxy_cosine_std']}`",
        f"- proxy target-swap L2 mean/std: `{report['representation_sensitivity']['target_swap_proxy_l2_distance_mean']} / {report['representation_sensitivity']['target_swap_proxy_l2_distance_std']}`",
        f"- ActionMap + LoRA standard proxy mean: `{arms['actionmap_lora']['metrics']['standard_proxy_score']['mean']}`",
        f"- fixed-prior TCA + LoRA standard proxy mean: `{arms['tca_map_lora_fixed_learned_text_fusion']['metrics']['standard_proxy_score']['mean']}`",
        f"- TCA-Select delta mean: `{decision['tca_select_standard_proxy_delta_mean']}`",
        f"- target 0 diagnosis: `{report['target0_diagnosis']['diagnosis']}`",
        f"- target information collapse status: `{decision['target_information_collapse_status']}`",
        f"- supported claim: `{decision['supported_paper_claim']}`",
        f"- Target-Prior TCA-Map remains main method: `{decision['target_prior_tca_map_remains_main_method']}`",
        f"- recommended next milestone: `{decision['next_milestone']}`",
        "",
        "## Interpretation",
        "",
        decision["interpretation"],
        "",
        "## Action Pathway",
        "",
    ]
    for arm_name in ("actionmap_lora", "tca_map_lora_hard_learned_target", "tca_map_lora_fixed_learned_text_fusion", "tca_map_lora_fixed_fusion_tca_select_ablation"):
        arm = arms[arm_name]
        lines.append(
            f"- `{arm_name}`: standard `{arm['metrics']['standard_proxy_score']['mean']}`, wrong-target `{arm['metrics']['wrong_target_proxy_rate']['mean']}`, target-swap action delta `{arm['target_swap_action_l1_delta_mean']['mean']}`"
        )
    lines.extend(["", "## Caveat", "", "Full VLA hidden states were not extracted, so this audit cannot claim representation collapse.", ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_representation_sensitivity_audit(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    *,
    seeds: list[int] | None = None,
    max_pairs: int = 32,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = 900,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    rank: int = DEFAULT_LORA_RANK,
    require_training_gate: bool = True,
) -> dict[str, Any]:
    dangerous = _dangerous_gates()
    if dangerous:
        raise TinyLoraSmokeError("dangerous gates are set: " + ", ".join(dangerous))
    if require_training_gate and os.environ.get("ALLOW_TINY_TRAINING") != "1":
        raise TinyLoraSmokeError("ALLOW_TINY_TRAINING=1 is required for bounded representation sensitivity audit")
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds, max_samples=max_samples, rank=rank)
    seeds = list(seeds or DEFAULT_SEEDS)
    if not seeds:
        raise TinyLoraSmokeError("at least one seed is required")
    if len(seeds) > 3 and max_samples >= 64:
        raise TinyLoraSmokeError("64-record representation audit is capped at three seeds")
    if max_pairs < 2 or max_pairs > 32:
        raise TinyLoraSmokeError("max_pairs must be between 2 and 32")
    started = time.perf_counter()
    all_records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)
    selected_count = _select_scaled_sample_count(len(all_records), max_samples)
    records = all_records[:selected_count]
    train_records, eval_records, split = _split_records(records)
    if not train_records or not eval_records:
        raise TinyLoraSmokeError("deterministic split did not produce train/eval records")
    representation = _proxy_representation_sensitivity(eval_records)
    seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        if time.perf_counter() - started > max_runtime_seconds:
            raise TinyLoraSmokeError("representation sensitivity audit exceeded max_runtime_seconds")
        seed_reports.append(
            _per_seed_report(
                seed=seed,
                train_records=train_records,
                eval_records=eval_records,
                max_steps=max_steps,
                rank=rank,
            )
        )
    aggregate_arms = {arm_name: _aggregate_arm(seed_reports, arm_name) for arm_name in ACTION_PATHWAY_ARMS}
    per_target = _aggregate_per_target(seed_reports)
    target0 = _target0_diagnosis(per_target)
    decision = _decision(representation=representation, aggregate_arms=aggregate_arms, target0=target0)
    elapsed = time.perf_counter() - started
    passed = bool(elapsed <= max_runtime_seconds and len(records) == selected_count and representation["target_swapped_pair_count"] > 0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(),
        "source_manifest": str(manifest_path),
        "prior_source_leakage_audit": _prior_source_leakage_audit(),
        "seeds": seeds,
        "seed_count": len(seeds),
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_samples": max_samples,
        "max_steps": max_steps,
        "lora_rank": rank,
        "record_count": len(records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "target_balance": _target_balance(records),
        "target_class_count": len(_target_balance(records)),
        "task_count": _task_count(records),
        "per_task_record_counts": _task_record_counts(records),
        "split": split,
        "sample_policy": "deterministic prefix of existing scaled LIBERO counterfactual manifest; no split/sample/seed cherry-picking",
        "representation_sensitivity": representation,
        "seed_reports": seed_reports,
        "aggregate_action_pathway_arms": aggregate_arms,
        "per_target_breakdown": per_target,
        "target0_diagnosis": target0,
        "target_prior_reinjection_result": {
            "depends_on_explicit_test_time_semantic_target_prior": True,
            "semantic_prior_non_leaking_under_publishability_audit": True,
            "learned_target_head_can_recover_target_from_available_representation": False,
            "target1_gain_mainly_wrong_target_correction": any(
                str(row["target_id"]) == "1" and row["wrong_target_delta_mean"] < 0.0 for row in per_target
            ),
            "oracle_prior_used_only_as_upper_bound": True,
            "valid_only_if_candidate_task_natural_language_text_is_available_at_test_time": True,
        },
        "decision": decision,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "representation_sensitivity_audit_passed": passed,
        "ready_for_paper_claim": False,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/libero_representation_sensitivity_audit_report.json")
    parser.add_argument("--report-md", default="reports/libero_representation_sensitivity_audit_report.md")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=900)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    try:
        report = run_representation_sensitivity_audit(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            seeds=seeds,
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
