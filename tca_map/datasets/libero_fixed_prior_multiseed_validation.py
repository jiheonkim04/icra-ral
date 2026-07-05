"""Multi-seed validation for fixed-prior offline splits.

This runner keeps the data split, metrics, baselines, and hyperparameters fixed
while varying only bounded CPU training seeds. It supports the fixed 16-record
diagnostic split and cautious 32/64-record scaled offline proxy splits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from tca_map.adapters.tiny_lora_smoke import DEFAULT_LORA_RANK, DEFAULT_MAX_STEPS, TinyLoraSmokeError
from tca_map.datasets.libero_fixed_prior_offline_scale_comparison import (
    _dangerous_gates,
    run_fixed_prior_offline_scale_comparison,
)


SCHEMA_VERSION = "2026-07-05.fixed_prior_multiseed_validation.v2"
DEFAULT_SEEDS = (11, 23, 37, 53, 71)
ALLOWED_SAMPLE_COUNTS = (16, 32, 64)
METRICS = (
    "standard_proxy_score",
    "wrong_target_proxy_rate",
    "action_target_consistency_score",
    "counterfactual_separation_margin",
    "target_top1_accuracy",
    "target_topk_accuracy",
)


def _mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 6) if values else 0.0


def _std(values: list[float]) -> float:
    return round(float(statistics.pstdev(values)), 6) if len(values) > 1 else 0.0


def _arm_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm["arm"]: arm for arm in report.get("arms", [])}


def _metric(arms: dict[str, dict[str, Any]], arm: str, metric: str) -> float:
    return float(arms[arm]["evaluation_metrics"][metric])


def _loss(arms: dict[str, dict[str, Any]], arm: str, key: str) -> float:
    return float(arms[arm][key])


def _gap_metric(arm: dict[str, Any]) -> float:
    metrics = arm["evaluation_metrics"]
    for key, value in metrics.items():
        if key.startswith("gap_to_oracle_target_tca_") and key.endswith("_standard_proxy"):
            return float(value)
    return 0.0


def _per_seed_row(seed: int, report: dict[str, Any]) -> dict[str, Any]:
    arms = _arm_map(report)
    fixed_head = arms["tca_map_fixed_learned_text_fusion_head_only"]
    action_head = arms["actionmap_head_only"]
    fixed_lora = arms["tca_map_lora_fixed_learned_text_fusion"]
    action_lora = arms["actionmap_lora"]
    selector = arms["tca_map_lora_fixed_fusion_tca_select_ablation"]
    hard_lora = arms["tca_map_lora_hard_learned_target"]
    return {
        "seed": seed,
        "record_count": report["record_count"],
        "train_record_count": report["train_record_count"],
        "eval_record_count": report["eval_record_count"],
        "split": report["split"],
        "fixed_prior_tca_head_standard_proxy": _metric(arms, fixed_head["arm"], "standard_proxy_score"),
        "actionmap_head_standard_proxy": _metric(arms, action_head["arm"], "standard_proxy_score"),
        "fixed_prior_tca_head_wrong_target": _metric(arms, fixed_head["arm"], "wrong_target_proxy_rate"),
        "actionmap_head_wrong_target": _metric(arms, action_head["arm"], "wrong_target_proxy_rate"),
        "fixed_prior_tca_lora_standard_proxy": _metric(arms, fixed_lora["arm"], "standard_proxy_score"),
        "actionmap_lora_standard_proxy": _metric(arms, action_lora["arm"], "standard_proxy_score"),
        "fixed_prior_tca_lora_wrong_target": _metric(arms, fixed_lora["arm"], "wrong_target_proxy_rate"),
        "actionmap_lora_wrong_target": _metric(arms, action_lora["arm"], "wrong_target_proxy_rate"),
        "hard_learned_tca_lora_standard_proxy": _metric(arms, hard_lora["arm"], "standard_proxy_score"),
        "hard_learned_tca_lora_wrong_target": _metric(arms, hard_lora["arm"], "wrong_target_proxy_rate"),
        "tca_select_lora_standard_proxy": _metric(arms, selector["arm"], "standard_proxy_score"),
        "tca_select_lora_wrong_target": _metric(arms, selector["arm"], "wrong_target_proxy_rate"),
        "fixed_prior_tca_head_advantage": round(
            _metric(arms, fixed_head["arm"], "standard_proxy_score") - _metric(arms, action_head["arm"], "standard_proxy_score"),
            6,
        ),
        "fixed_prior_tca_lora_advantage": round(
            _metric(arms, fixed_lora["arm"], "standard_proxy_score") - _metric(arms, action_lora["arm"], "standard_proxy_score"),
            6,
        ),
        "tca_select_lora_delta": round(
            _metric(arms, selector["arm"], "standard_proxy_score") - _metric(arms, fixed_lora["arm"], "standard_proxy_score"),
            6,
        ),
        "fixed_prior_tca_lora_beats_actionmap_lora": bool(
            _metric(arms, fixed_lora["arm"], "standard_proxy_score") > _metric(arms, action_lora["arm"], "standard_proxy_score")
            and _metric(arms, fixed_lora["arm"], "wrong_target_proxy_rate") <= _metric(arms, action_lora["arm"], "wrong_target_proxy_rate")
        ),
        "fixed_prior_tca_lora_wrong_target_improves": bool(
            _metric(arms, fixed_lora["arm"], "wrong_target_proxy_rate") < _metric(arms, action_lora["arm"], "wrong_target_proxy_rate")
        ),
        "tca_select_nontrivial_gain": bool(
            _metric(arms, selector["arm"], "standard_proxy_score") - _metric(arms, fixed_lora["arm"], "standard_proxy_score") >= 0.01
            or _metric(arms, selector["arm"], "wrong_target_proxy_rate") < _metric(arms, fixed_lora["arm"], "wrong_target_proxy_rate")
        ),
        "fixed_prior_lora_vs_head_standard_proxy_delta": round(
            _metric(arms, fixed_lora["arm"], "standard_proxy_score") - _metric(arms, fixed_head["arm"], "standard_proxy_score"),
            6,
        ),
    }


def _aggregate_arms(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    arm_names = [arm["arm"] for arm in seed_reports[0]["arms"]]
    aggregate: dict[str, Any] = {}
    for arm_name in arm_names:
        arms = [_arm_map(report)[arm_name] for report in seed_reports]
        metric_summary = {}
        for metric_name in METRICS:
            values = [float(arm["evaluation_metrics"][metric_name]) for arm in arms]
            metric_summary[metric_name] = {"mean": _mean(values), "std": _std(values), "values": [round(value, 6) for value in values]}
        gaps = [_gap_metric(arm) for arm in arms]
        aggregate[arm_name] = {
            "family": arms[0]["family"],
            "target_prior_variant": arms[0]["target_prior_variant"],
            "initial_loss": {
                "mean": _mean([_loss(_arm_map(report), arm_name, "initial_loss") for report in seed_reports]),
                "std": _std([_loss(_arm_map(report), arm_name, "initial_loss") for report in seed_reports]),
            },
            "final_loss": {
                "mean": _mean([_loss(_arm_map(report), arm_name, "final_loss") for report in seed_reports]),
                "std": _std([_loss(_arm_map(report), arm_name, "final_loss") for report in seed_reports]),
            },
            "gap_to_oracle_standard_proxy": {"mean": _mean(gaps), "std": _std(gaps), "values": [round(value, 6) for value in gaps]},
            "metrics": metric_summary,
        }
        if arms[0]["family"] == "lora":
            aggregate[arm_name]["lora_target_modules"] = arms[0].get("lora_target_modules", [])
            aggregate[arm_name]["trainable_lora_parameter_count"] = arms[0].get("trainable_lora_parameter_count")
        else:
            aggregate[arm_name]["trainable_parameter_count"] = arms[0].get("trainable_parameter_count")
    return aggregate


def _split_signature(report: dict[str, Any]) -> dict[str, Any]:
    split = report.get("split") or {}
    return {
        "record_count": report.get("record_count"),
        "train_record_count": report.get("train_record_count"),
        "eval_record_count": report.get("eval_record_count"),
        "train_pair_ids": split.get("train_pair_ids"),
        "eval_pair_ids": split.get("eval_pair_ids"),
    }


def _manifest_capacity(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"manifest_readable": False, "manifest_read_error": str(exc), "manifest_pair_count": 0, "manifest_record_capacity": 0}
    pair_count = len(manifest.get("counterfactual_pairs", []))
    return {
        "manifest_readable": True,
        "manifest_read_error": None,
        "manifest_pair_count": pair_count,
        "manifest_record_capacity": pair_count * 2,
    }


def _aggregate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_count = len(rows)
    fixed_lora_beats = sum(1 for row in rows if row["fixed_prior_tca_lora_beats_actionmap_lora"])
    wrong_target_improves = sum(1 for row in rows if row["fixed_prior_tca_lora_wrong_target_improves"])
    select_gains = sum(1 for row in rows if row["tca_select_nontrivial_gain"])
    lora_deltas = [float(row["fixed_prior_lora_vs_head_standard_proxy_delta"]) for row in rows]
    lora_hurts_count = sum(1 for value in lora_deltas if value < 0.0)
    hard_lora_scores = [float(row["hard_learned_tca_lora_standard_proxy"]) for row in rows]
    fixed_lora_scores = [float(row["fixed_prior_tca_lora_standard_proxy"]) for row in rows]
    return {
        "seed_count": seed_count,
        "fixed_prior_tca_lora_beats_actionmap_lora_count": fixed_lora_beats,
        "fixed_prior_tca_lora_wrong_target_improves_count": wrong_target_improves,
        "tca_select_nontrivial_gain_count": select_gains,
        "fixed_prior_tca_lora_advantage": {
            "mean": _mean([float(row["fixed_prior_tca_lora_advantage"]) for row in rows]),
            "std": _std([float(row["fixed_prior_tca_lora_advantage"]) for row in rows]),
            "values": [row["fixed_prior_tca_lora_advantage"] for row in rows],
        },
        "fixed_prior_tca_head_advantage": {
            "mean": _mean([float(row["fixed_prior_tca_head_advantage"]) for row in rows]),
            "std": _std([float(row["fixed_prior_tca_head_advantage"]) for row in rows]),
            "values": [row["fixed_prior_tca_head_advantage"] for row in rows],
        },
        "fixed_prior_lora_vs_head_standard_proxy_delta": {
            "mean": _mean(lora_deltas),
            "std": _std(lora_deltas),
            "values": [round(value, 6) for value in lora_deltas],
            "lora_hurts_count": lora_hurts_count,
        },
        "fixed_prior_tca_advantage_stable": bool(fixed_lora_beats >= max(1, seed_count - 1)),
        "tca_select_meaningful_gain": bool(select_gains > 0),
        "lora_consistently_helps_fixed_prior_tca": bool(all(value > 0.0 for value in lora_deltas)),
        "lora_consistently_hurts_fixed_prior_tca": bool(all(value < 0.0 for value in lora_deltas)),
        "hard_learned_target_remains_unstable_or_weaker": bool(
            _mean(hard_lora_scores) < _mean(fixed_lora_scores) or _std(hard_lora_scores) > 0.05
        ),
    }


def _policy() -> dict[str, Any]:
    return {
        "multi_seed_fixed_prior_offline_proxy": True,
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
    comparison = report["aggregate_comparison"]
    lines = [
        "# Multi-Seed Fixed-Prior Offline Validation",
        "",
        "Exploratory offline proxy only. This is not standard success, rollout evidence, or paper-grade evidence.",
        "",
        f"- passed: `{report['fixed_prior_multiseed_validation_passed']}`",
        f"- seed count: `{report['seed_count']}`",
        f"- seeds: `{report['seeds']}`",
        f"- records: `{report['record_count']}`",
        f"- train/eval records: `{report['train_record_count']} / {report['eval_record_count']}`",
        f"- task count: `{report['task_count']}`",
        f"- target balance: `{report['target_balance']}`",
        f"- available records from manifest: `{report.get('available_record_count')}`",
        f"- full manifest record capacity: `{report.get('manifest_record_capacity')}`",
        f"- split consistent: `{report['split_consistent']}`",
        f"- fixed-prior TCA + LoRA beats ActionMap + LoRA: `{comparison['fixed_prior_tca_lora_beats_actionmap_lora_count']} / {report['seed_count']}`",
        f"- wrong-target proxy improves: `{comparison['fixed_prior_tca_lora_wrong_target_improves_count']} / {report['seed_count']}`",
        f"- TCA-Select nontrivial gain count: `{comparison['tca_select_nontrivial_gain_count']}`",
        f"- fixed-prior TCA + LoRA advantage mean/std: `{comparison['fixed_prior_tca_lora_advantage']['mean']} / {comparison['fixed_prior_tca_lora_advantage']['std']}`",
        f"- LoRA vs head-only fixed-prior delta mean/std: `{comparison['fixed_prior_lora_vs_head_standard_proxy_delta']['mean']} / {comparison['fixed_prior_lora_vs_head_standard_proxy_delta']['std']}`",
        "",
        "## Per-Seed Rows",
        "",
    ]
    for row in report["per_seed_table"]:
        lines.append(
            f"- seed `{row['seed']}`: fixed-prior TCA + LoRA `{row['fixed_prior_tca_lora_standard_proxy']}` vs ActionMap + LoRA `{row['actionmap_lora_standard_proxy']}`, advantage `{row['fixed_prior_tca_lora_advantage']}`, selector delta `{row['tca_select_lora_delta']}`"
        )
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_fixed_prior_multiseed_validation(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    *,
    seeds: list[int] | None = None,
    max_pairs: int = 8,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = 900,
    max_samples: int = 16,
    rank: int = DEFAULT_LORA_RANK,
    require_training_gate: bool = True,
) -> dict[str, Any]:
    dangerous = _dangerous_gates()
    if dangerous:
        raise TinyLoraSmokeError("dangerous gates are set: " + ", ".join(dangerous))
    seeds = list(seeds or DEFAULT_SEEDS)
    if max_samples not in ALLOWED_SAMPLE_COUNTS:
        raise TinyLoraSmokeError("multi-seed validation max_samples must be one of 16, 32, or 64")
    if max_samples in {16, 32} and (len(seeds) < 3 or len(seeds) > 5):
        raise TinyLoraSmokeError("16/32-record multi-seed validation requires between 3 and 5 seeds")
    if max_samples == 64 and (len(seeds) < 1 or len(seeds) > 3):
        raise TinyLoraSmokeError("64-record multi-seed validation requires between 1 and 3 seeds")
    if max_steps > 300:
        raise TinyLoraSmokeError("max_steps must not exceed 300")
    started = time.perf_counter()
    seed_reports: list[dict[str, Any]] = []
    temp_root = Path("runs") / "fixed_prior_multiseed"
    for seed in seeds:
        if time.perf_counter() - started > max_runtime_seconds:
            raise TinyLoraSmokeError("multi-seed validation exceeded max_runtime_seconds")
        seed_reports.append(
            run_fixed_prior_offline_scale_comparison(
                manifest_path=manifest_path,
                report_json=temp_root / f"seed_{seed}.json",
                report_md=temp_root / f"seed_{seed}.md",
                max_pairs=max_pairs,
                max_action_steps=max_action_steps,
                max_steps=max_steps,
                max_runtime_seconds=max_runtime_seconds,
                max_samples=max_samples,
                rank=rank,
                seed=seed,
                require_training_gate=require_training_gate,
            )
        )
    rows = [_per_seed_row(seed, report) for seed, report in zip(seeds, seed_reports)]
    first_signature = _split_signature(seed_reports[0])
    manifest_capacity = _manifest_capacity(manifest_path)
    split_consistent = all(_split_signature(report) == first_signature for report in seed_reports)
    aggregate_comparison = _aggregate_comparison(rows)
    elapsed = time.perf_counter() - started
    passed = bool(
        elapsed <= max_runtime_seconds
        and split_consistent
        and len(seed_reports) == len(seeds)
        and all(report.get("record_count") == max_samples for report in seed_reports)
    )
    if aggregate_comparison["fixed_prior_tca_advantage_stable"]:
        recommendation = "A_64_record_split" if max_samples == 32 else "B_multi_seed_on_larger_split"
        interpretation = (
            f"Fixed-prior TCA + LoRA beats ActionMap + LoRA in most or all seeds on the fixed {max_samples}-record split. "
            "This supports cautious larger-split validation, but remains exploratory offline proxy evidence."
        )
    else:
        recommendation = "B_diagnose_seed_sensitivity"
        interpretation = (
            "Fixed-prior TCA + LoRA advantage is seed-sensitive on the fixed 16-sample split. Diagnose variance before scaling."
        )
    if aggregate_comparison["lora_consistently_hurts_fixed_prior_tca"]:
        interpretation += " LoRA consistently hurts fixed-prior TCA relative to head-only, so treat LoRA as robustness/attribution rather than a performance claim."
    if not aggregate_comparison["tca_select_meaningful_gain"]:
        interpretation += " TCA-Select shows no nontrivial gain in any seed and should be de-emphasized or killed as a core contribution."
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(),
        "source_manifest": str(manifest_path),
        **manifest_capacity,
        "seeds": seeds,
        "seed_count": len(seeds),
        "seed_policy": f"same {max_samples}-record split; seeds vary only CPU head-only SGD order and LoRA low-rank initialization",
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_samples": max_samples,
        "max_steps": max_steps,
        "lora_rank": rank,
        "split_signature": first_signature,
        "split_consistent": split_consistent,
        "sample_selection": seed_reports[0].get("sample_selection", {}),
        "available_record_count": seed_reports[0].get("sample_selection", {}).get("available_record_count"),
        "available_pair_count": int(seed_reports[0].get("sample_selection", {}).get("available_record_count", 0) // 2),
        "record_count": seed_reports[0]["record_count"],
        "train_record_count": seed_reports[0]["train_record_count"],
        "eval_record_count": seed_reports[0]["eval_record_count"],
        "target_balance": seed_reports[0]["target_balance"],
        "task_count": seed_reports[0]["task_count"],
        "per_task_record_counts": seed_reports[0].get("per_task_record_counts", {}),
        "train_per_task_record_counts": seed_reports[0].get("train_per_task_record_counts", {}),
        "eval_per_task_record_counts": seed_reports[0].get("eval_per_task_record_counts", {}),
        "per_seed_table": rows,
        "aggregate_arms": _aggregate_arms(seed_reports),
        "aggregate_comparison": aggregate_comparison,
        "elapsed_seconds": round(elapsed, 6),
        "runtime_within_cap": elapsed <= max_runtime_seconds,
        "fixed_prior_multiseed_validation_passed": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "recommended_next_milestone": recommendation,
        "interpretation": interpretation,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_fixed_prior_multiseed_validation_report.json")
    parser.add_argument("--report-md", default="reports/libero_fixed_prior_multiseed_validation_report.md")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=900)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    args = parser.parse_args()
    seed_values = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    try:
        report = run_fixed_prior_multiseed_validation(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            seeds=seed_values,
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
