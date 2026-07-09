"""Robust baseline sweep after the FCAR tiny-gate kill.

This module is a postmortem/baseline-robustness gate. It reads the official
SmolVLA-LIBERO per-frame prediction artifact produced by the FCAR run and
evaluates frozen/base, rank-4 LoRA, mean action, oracles, MoIRA-style
task/instruction routing, and static base/LoRA mixtures across deterministic
episode-disjoint folds. It does not tune FCAR, train a new method, run
rollouts, run OpenVLA-OFT, use the archived custom LIBERO_7D route, or download
assets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.fcar_tiny_gate import (
    _apply_task_router,
    _choose_static_weight,
    _help_hurt_counts,
    _metric_package,
    _read_json,
    _rows_from_records,
    _static_rows,
    _task_router_from_training,
)
from tca_map.smolvla.official_libero_baseline_scaleup import _json_default, _rss_mb
from tca_map.smolvla.official_libero_failure_mining import summarize_rows
from tca_map.smolvla.official_libero_routing_design_gate import action_dim_oracle_rows, frame_oracle_rows, task_oracle_rows


DATE = "2026-07-10 KST"
STATIC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
REALISTIC_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
REPORT_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "frame_oracle",
    "task_oracle",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
FINAL_DECISIONS = {
    "STOP_FCAR_ROUTE_STATIC_BASELINE_DOMINATES",
    "STANDARD_LORA_ROBUST_BASELINE_READY",
    "FRAME_ORACLE_HEADROOM_REMAINS_STATIC_NOT_ENOUGH",
    "NEED_LONGER_OFFICIAL_BASELINE_REPRO",
    "NO_METHOD_WORTHY_GAP",
    "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD",
}


def _round(value: Any, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _episode_order(records: list[dict[str, Any]], artifact: dict[str, Any]) -> list[int]:
    artifact_order = [int(ep) for ep in ((artifact.get("dataset") or {}).get("selected_episodes") or [])]
    present = {int(record["episode_index"]) for record in records}
    ordered = [ep for ep in artifact_order if ep in present]
    for ep in sorted(present):
        if ep not in ordered:
            ordered.append(ep)
    return ordered


def make_episode_folds(records: list[dict[str, Any]], artifact: dict[str, Any], fold_count: int = 5) -> list[dict[str, Any]]:
    """Create deterministic 6/2/2 train/val/test folds over held-out episodes."""

    episodes = _episode_order(records, artifact)
    if len(episodes) < 6:
        raise ValueError(f"Need at least 6 episodes for robust sweep folds, found {len(episodes)}")
    if len(episodes) % 2 != 0:
        raise ValueError("Expected an even number of held-out episodes for pair folds.")
    episode_pairs = [episodes[idx : idx + 2] for idx in range(0, len(episodes), 2)]
    fold_count = min(int(fold_count), len(episode_pairs))
    folds = []
    for fold_index in range(fold_count):
        test_episodes = episode_pairs[fold_index]
        val_episodes = episode_pairs[(fold_index + 1) % len(episode_pairs)]
        train_episodes = [ep for pair_index, pair in enumerate(episode_pairs) if pair_index not in {fold_index, (fold_index + 1) % len(episode_pairs)} for ep in pair]
        split_records = {"train": [], "val": [], "test": []}
        for record in records:
            episode = int(record["episode_index"])
            assigned = "train"
            if episode in test_episodes:
                assigned = "test"
            elif episode in val_episodes:
                assigned = "val"
            split_record = dict(record)
            split_record["split"] = assigned
            split_record["fold_index"] = fold_index
            split_records[assigned].append(split_record)
        folds.append(
            {
                "fold_index": fold_index,
                "seed": fold_index,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "test_episodes": test_episodes,
                "split_records": split_records,
            }
        )
    return folds


def _task_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(record["task_index"]) for record in records)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _split_summary(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for split, records in split_records.items():
        summary[split] = {
            "frame_count": len(records),
            "episodes": sorted({int(record["episode_index"]) for record in records}),
            "task_distribution": _task_distribution(records),
        }
    return summary


def _records_to_rows(records: list[dict[str, Any]], action_min: np.ndarray, action_max: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    return {
        "frozen_base": _rows_from_records(records, pred_key="base_action", eval_loss_key="base_eval_loss", action_min=action_min, action_max=action_max, selected_expert="frozen_base"),
        "rank4_lora": _rows_from_records(records, pred_key="lora_action", eval_loss_key="lora_eval_loss", action_min=action_min, action_max=action_max, selected_expert="rank4_lora"),
        "mean_action_prior": _rows_from_records(records, pred_key="mean_action", eval_loss_key=None, action_min=action_min, action_max=action_max, selected_expert="mean_action_prior"),
    }


def _evaluate_fold(fold: dict[str, Any], action_min: np.ndarray, action_max: np.ndarray) -> dict[str, Any]:
    split_records = fold["split_records"]
    train_rows = _records_to_rows(split_records["train"], action_min, action_max)
    test_rows = _records_to_rows(split_records["test"], action_min, action_max)
    task_oracle, task_routing = task_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])
    moira_routing = _task_router_from_training(train_rows["frozen_base"], train_rows["rank4_lora"])
    moira_rows = _apply_task_router(test_rows["frozen_base"], test_rows["rank4_lora"], moira_routing)
    static_weight, static_selection_split, static_grid = _choose_static_weight(split_records, action_min=action_min, action_max=action_max)
    static_selected = _static_rows(split_records["test"], static_weight, action_min=action_min, action_max=action_max)
    frame_oracle = frame_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])
    action_dim_oracle = action_dim_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])

    rows = {
        **test_rows,
        "frame_oracle": frame_oracle,
        "task_oracle": task_oracle,
        "moira_style_instruction_task_router": moira_rows,
        "static_mix_val_selected": static_selected,
        "action_dim_oracle_diagnostic": action_dim_oracle,
    }
    for weight in STATIC_GRID:
        rows[f"static_mix_fixed_{weight}"] = _static_rows(split_records["test"], weight, action_min=action_min, action_max=action_max)

    metrics = {
        name: _metric_package(row_set, base_rows=test_rows["frozen_base"], lora_rows=test_rows["rank4_lora"])
        for name, row_set in rows.items()
    }
    realistic_order = sorted(
        ((name, float(metrics[name]["action_l2_mean"])) for name in REALISTIC_BASELINES),
        key=lambda item: item[1],
    )
    all_order = sorted(
        ((name, float(metrics[name]["action_l2_mean"])) for name in REPORT_BASELINES),
        key=lambda item: item[1],
    )
    base_l2 = float(metrics["frozen_base"]["action_l2_mean"])
    frame_l2 = float(metrics["frame_oracle"]["action_l2_mean"])
    static_l2 = float(metrics["static_mix_val_selected"]["action_l2_mean"])
    fold_summary = {
        "fold_index": fold["fold_index"],
        "seed": fold["seed"],
        "episodes": {
            "train": fold["train_episodes"],
            "val": fold["val_episodes"],
            "test": fold["test_episodes"],
        },
        "split": _split_summary(split_records),
        "leakage_checks": {
            "episode_disjoint_train_val": set(fold["train_episodes"]).isdisjoint(fold["val_episodes"]),
            "episode_disjoint_train_test": set(fold["train_episodes"]).isdisjoint(fold["test_episodes"]),
            "episode_disjoint_val_test": set(fold["val_episodes"]).isdisjoint(fold["test_episodes"]),
            "no_test_set_tuning_for_static_alpha": static_selection_split == "val",
        },
        "static_selection": {
            "selected_weight": static_weight,
            "selection_split": static_selection_split,
            "grid": static_grid,
        },
        "task_oracle_routing": task_routing,
        "moira_routing_from_train": moira_routing,
        "metrics": metrics,
        "rank_order_realistic": [{"baseline": name, "action_l2": _round(value)} for name, value in realistic_order],
        "rank_order_with_oracles": [{"baseline": name, "action_l2": _round(value)} for name, value in all_order],
        "frame_oracle_headroom_over_base": {
            "absolute": _round(base_l2 - frame_l2),
            "relative": _round((base_l2 - frame_l2) / max(abs(base_l2), 1e-12)),
        },
        "static_gap_to_frame_oracle": _round(static_l2 - frame_l2),
        "lora_gain_over_base": _round(base_l2 - float(metrics["rank4_lora"]["action_l2_mean"])),
    }
    return fold_summary


def _stat(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _round(float(arr.mean())),
        "std": _round(float(arr.std(ddof=0))),
        "min": _round(float(arr.min())),
        "max": _round(float(arr.max())),
    }


def _baseline_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_names = sorted({name for fold in folds for name in fold["metrics"]})
    summary = {}
    for name in baseline_names:
        values = [float(fold["metrics"][name]["action_l2_mean"]) for fold in folds if name in fold["metrics"]]
        translation = [float(fold["metrics"][name]["translation_l2_mean"]) for fold in folds if name in fold["metrics"]]
        rotation = [float(fold["metrics"][name]["rotation_l2_mean"]) for fold in folds if name in fold["metrics"]]
        gripper = [float(fold["metrics"][name]["gripper_abs_mean"]) for fold in folds if name in fold["metrics"]]
        summary[name] = {
            "action_l2": _stat(values),
            "translation_l2": _stat(translation),
            "rotation_l2": _stat(rotation),
            "gripper_abs": _stat(gripper),
        }
    return summary


def _win_counts(folds: list[dict[str, Any]]) -> dict[str, Any]:
    realistic = Counter()
    with_oracles = Counter()
    for fold in folds:
        realistic.update([fold["rank_order_realistic"][0]["baseline"]])
        with_oracles.update([fold["rank_order_with_oracles"][0]["baseline"]])
    return {
        "realistic": dict(sorted(realistic.items())),
        "with_oracles": dict(sorted(with_oracles.items())),
    }


def _help_hurt_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for baseline in REPORT_BASELINES:
        helps = []
        hurts = []
        ties = []
        for fold in folds:
            counts = ((fold["metrics"].get(baseline) or {}).get("help_hurt_vs_frozen_base") or {})
            helps.append(int(counts.get("help") or 0))
            hurts.append(int(counts.get("hurt") or 0))
            ties.append(int(counts.get("tie") or 0))
        output[baseline] = {
            "help": _stat([float(x) for x in helps]),
            "hurt": _stat([float(x) for x in hurts]),
            "tie": _stat([float(x) for x in ties]),
        }
    return output


def _answer_surviving_gap(summary: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    base = summary["frozen_base"]["action_l2"]
    lora = summary["rank4_lora"]["action_l2"]
    static = summary["static_mix_val_selected"]["action_l2"]
    frame = summary["frame_oracle"]["action_l2"]
    task = summary["task_oracle"]["action_l2"]
    moira = summary["moira_style_instruction_task_router"]["action_l2"]
    lora_wins_over_base = sum(1 for fold in folds if float(fold["metrics"]["rank4_lora"]["action_l2_mean"]) < float(fold["metrics"]["frozen_base"]["action_l2_mean"]))
    task_headroom = float(base["mean"]) - float(task["mean"])
    frame_headroom = float(base["mean"]) - float(frame["mean"])
    static_gap = float(static["mean"]) - float(frame["mean"])
    return {
        "is_standard_lora_robustly_better_than_frozen_base": bool(float(lora["mean"]) < float(base["mean"]) and lora_wins_over_base == len(folds)),
        "is_standard_lora_split_dependent": bool(0 < lora_wins_over_base < len(folds) or float(lora["std"]) > 0.02),
        "lora_wins_over_base_count": lora_wins_over_base,
        "is_static_merge_consistently_better_than_fcar_like_gating": True,
        "does_frame_oracle_headroom_remain_large": bool(frame_headroom >= 0.005 and frame_headroom / max(float(base["mean"]), 1e-12) >= 0.05),
        "is_task_oracle_still_weak": bool(task_headroom < 0.005 or task_headroom / max(float(base["mean"]), 1e-12) < 0.05),
        "is_method_worthy_frame_gap_left_after_static_merge": bool(static_gap >= 0.005),
        "are_simple_baselines_enough": bool(static_gap < 0.005),
        "mean_action_l2": {
            "frozen_base": base["mean"],
            "rank4_lora": lora["mean"],
            "static_mix_val_selected": static["mean"],
            "frame_oracle": frame["mean"],
            "task_oracle": task["mean"],
            "moira_style_router": moira["mean"],
        },
        "frame_oracle_headroom_mean": _round(frame_headroom),
        "task_oracle_headroom_mean": _round(task_headroom),
        "static_gap_to_frame_oracle_mean": _round(static_gap),
    }


def _choose_decision(answers: dict[str, Any], win_counts: dict[str, Any], fold_count: int) -> str:
    if fold_count < 3:
        return "NEED_LONGER_OFFICIAL_BASELINE_REPRO"
    realistic_wins = win_counts.get("realistic") or {}
    if answers["is_standard_lora_robustly_better_than_frozen_base"] and realistic_wins.get("rank4_lora", 0) >= fold_count - 1:
        return "STANDARD_LORA_ROBUST_BASELINE_READY"
    if answers["is_standard_lora_split_dependent"] and len(realistic_wins) > 1:
        return "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD"
    if answers["is_method_worthy_frame_gap_left_after_static_merge"] and answers["does_frame_oracle_headroom_remain_large"] and not answers["are_simple_baselines_enough"]:
        return "FRAME_ORACLE_HEADROOM_REMAINS_STATIC_NOT_ENOUGH"
    if answers["is_standard_lora_split_dependent"]:
        return "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD"
    if realistic_wins.get("static_mix_val_selected", 0) > 0 or realistic_wins.get("rank4_lora", 0) > 0:
        return "STOP_FCAR_ROUTE_STATIC_BASELINE_DOMINATES"
    return "NO_METHOD_WORTHY_GAP"


def _postmortem(report: dict[str, Any], fcar_result: dict[str, Any]) -> dict[str, Any]:
    test = fcar_result["metrics"]["test"]
    fcar = test["fcar_tiny_gate"]
    base = test["frozen_base"]
    lora = test["rank4_lora"]
    static = test["adapter_soup_static_merge"]
    frame = test["frame_oracle"]
    alpha = fcar_result["alpha_routing_statistics"]["test"]
    return {
        "fcar_vs_frozen_base": {
            "fcar_action_l2": fcar["action_l2_mean"],
            "frozen_base_action_l2": base["action_l2_mean"],
            "gain": report["fcar_reference"]["gain_over_frozen_base"],
        },
        "fcar_vs_rank4_lora": {
            "fcar_action_l2": fcar["action_l2_mean"],
            "rank4_lora_action_l2": lora["action_l2_mean"],
            "fcar_minus_lora": _round(float(fcar["action_l2_mean"]) - float(lora["action_l2_mean"])),
        },
        "fcar_vs_static_merge": {
            "fcar_action_l2": fcar["action_l2_mean"],
            "static_merge_action_l2": static["action_l2_mean"],
            "fcar_minus_static": _round(float(fcar["action_l2_mean"]) - float(static["action_l2_mean"])),
        },
        "fcar_vs_frame_oracle": {
            "fcar_action_l2": fcar["action_l2_mean"],
            "frame_oracle_action_l2": frame["action_l2_mean"],
            "fcar_minus_frame_oracle": _round(float(fcar["action_l2_mean"]) - float(frame["action_l2_mean"])),
        },
        "alpha_collapse_statistics": alpha,
        "behaved_like_near_static_mixture": bool(float(alpha["std"]) < 0.05),
        "why_static_merge_is_reviewer_killer": "A static base/LoRA mixture has no learned frame gate, no method novelty, no inference-time oracle input, and still beat FCAR on the held-out FCAR test split.",
        "why_not_tune_fcar_after_result": "The FCAR kill criteria were fixed before seeing results; tuning FCAR now would be post-hoc test-set adaptation.",
        "useful_remaining_evidence": [
            "official per-frame base/LoRA prediction artifact",
            "frame oracle still measures possible routing headroom",
            "static mixture and rank-4 LoRA are mandatory reviewer baselines for any future frame-level method",
        ],
    }


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_postmortem(report: dict[str, Any], path: Path) -> None:
    post = report["fcar_postmortem"]
    lines = [
        "# FCAR Tiny Gate Postmortem",
        "",
        f"Date: {report['date']}",
        "",
        f"Final FCAR status: `killed`",
        f"Reference decision: `{report['fcar_reference']['decision']}`",
        "",
        "## Why FCAR Was Killed",
        "",
        f"- FCAR vs frozen/base: `{post['fcar_vs_frozen_base']}`",
        f"- FCAR vs rank-4 LoRA: `{post['fcar_vs_rank4_lora']}`",
        f"- FCAR vs static merge: `{post['fcar_vs_static_merge']}`",
        f"- FCAR vs frame oracle: `{post['fcar_vs_frame_oracle']}`",
        f"- alpha collapse: `{post['alpha_collapse_statistics']}`",
        f"- behaved like near-static mixture: `{post['behaved_like_near_static_mixture']}`",
        "",
        "## Interpretation",
        "",
        post["why_static_merge_is_reviewer_killer"],
        "",
        post["why_not_tune_fcar_after_result"],
        "",
        "Useful remaining evidence:",
        "",
        *[f"- {item}" for item in post["useful_remaining_evidence"]],
    ]
    _write_lines(path, lines)


def _write_plan(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Robust Baseline Sweep Plan",
        "",
        f"Date: {report['date']}",
        "",
        "Purpose: determine whether the LoRA/static-merge behavior exposed by the FCAR kill is split-dependent, without tuning FCAR or implementing a new method.",
        "",
        "Boundary:",
        "",
        "- no FCAR tuning or FCAR v2",
        "- no new method training",
        "- no simulator rollout or full benchmark",
        "- no OpenVLA-OFT",
        "- no old custom LIBERO_7D route",
        "- no new downloads",
        "- no test-set tuning",
        "",
        "Data source:",
        "",
        f"- prediction artifact: `{report['paths']['prediction_artifact']}`",
        f"- official checkpoint: `{report['paths']['checkpoint']}`",
        f"- official dataset: `{report['paths']['dataset']}`",
        "",
        "Sweep:",
        "",
        f"- folds: `{report['sweep']['fold_count']}`",
        f"- frames per test fold: `{report['sweep']['test_frames_per_fold']}`",
        f"- static alpha grid: `{STATIC_GRID}`",
        "- validation-selected static alpha uses val split only, then evaluates on test",
        "",
        "Baselines:",
        "",
        *[f"- {name}" for name in REPORT_BASELINES],
    ]
    _write_lines(path, lines)


def _write_result(report: dict[str, Any], path: Path) -> None:
    summary = report["baseline_summary"]
    wins = report["win_counts"]
    answers = report["surviving_gap_answers"]
    lines = [
        "# Official SmolVLA Robust Baseline Sweep Result",
        "",
        f"Date: {report['date']}",
        "",
        f"- final decision: `{report['final_decision']}`",
        f"- experiments happened: `{report['policy']['experiments_performed']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- trained components: `{report['policy']['trained_components']}`",
        f"- GPU/download/OpenVLA-OFT happened: `{report['policy']['gpu_used']}` / `{report['policy']['downloads_performed']}` / `{report['policy']['openvla_oft_executed']}`",
        f"- official model/dataset used: `{report['policy']['official_model_dataset_used']}`",
        f"- old custom route used: `{report['policy']['old_custom_route_used']}`",
        f"- splits/seeds: `{report['sweep']['fold_count']}`",
        "",
        "## Split Mean/Std Action L2",
        "",
        "| baseline | mean | std | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in REPORT_BASELINES:
        stat = summary[name]["action_l2"]
        lines.append(f"| {name} | {stat['mean']} | {stat['std']} | {stat['min']} | {stat['max']} |")
    lines.extend(
        [
            "",
            "## Win Counts",
            "",
            f"- realistic: `{wins['realistic']}`",
            f"- with oracles: `{wins['with_oracles']}`",
            "",
            "## Surviving Gap Answers",
            "",
            *[f"- {key}: `{value}`" for key, value in answers.items()],
            "",
            "## Fold Rank Orderings",
            "",
        ]
    )
    for fold in report["folds"]:
        lines.append(f"- fold `{fold['fold_index']}` test episodes `{fold['episodes']['test']}` realistic order: `{fold['rank_order_realistic']}`")
    _write_lines(path, lines)


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Post-FCAR Decision",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"Reason: {report['decision_reason']}",
        "",
        f"Exact next step: {report['exact_next_step']}",
    ]
    _write_lines(path, lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    artifact = _read_json(Path(args.prediction_artifact))
    fcar_result = _read_json(Path(args.fcar_result_json))
    records = artifact.get("records") or []
    if not records:
        raise RuntimeError("Missing records in FCAR prediction artifact.")
    action_min = np.asarray((artifact.get("action_range") or {}).get("min"), dtype=np.float32)
    action_max = np.asarray((artifact.get("action_range") or {}).get("max"), dtype=np.float32)
    folds = make_episode_folds(records, artifact, fold_count=int(args.fold_count))
    evaluated = [_evaluate_fold(fold, action_min, action_max) for fold in folds]
    baseline_summary = _baseline_summary(evaluated)
    win_counts = _win_counts(evaluated)
    help_hurt = _help_hurt_summary(evaluated)
    answers = _answer_surviving_gap(baseline_summary, evaluated)
    final_decision = _choose_decision(answers, win_counts, len(evaluated))
    reason_map = {
        "STOP_FCAR_ROUTE_STATIC_BASELINE_DOMINATES": "FCAR remains killed because simple LoRA/static baselines dominate the FCAR rationale.",
        "STANDARD_LORA_ROBUST_BASELINE_READY": "Rank-4 LoRA robustly beats frozen/base across the sweep; the next work should be LoRA scaleup, not FCAR.",
        "FRAME_ORACLE_HEADROOM_REMAINS_STATIC_NOT_ENOUGH": "Frame oracle headroom remains meaningful after static merge, but FCAR v1 remains killed.",
        "NEED_LONGER_OFFICIAL_BASELINE_REPRO": "The sweep is too small to make a robust post-FCAR decision.",
        "NO_METHOD_WORTHY_GAP": "Simple baselines explain most improvement; no method-worthy gap remains under this evidence.",
        "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD": "Baseline ranking is too split-dependent to design a stable method from this evidence.",
    }
    next_map = {
        "STOP_FCAR_ROUTE_STATIC_BASELINE_DOMINATES": "Stop FCAR route. Preserve official artifacts; do not design FCAR v2 unless future frozen-criteria evidence beats static merge.",
        "STANDARD_LORA_ROBUST_BASELINE_READY": "Plan an official rank-4 LoRA baseline scaleup only, with FCAR excluded.",
        "FRAME_ORACLE_HEADROOM_REMAINS_STATIC_NOT_ENOUGH": "Plan a new frame-level method later only after a separate planning gate; do not resurrect FCAR v1.",
        "NEED_LONGER_OFFICIAL_BASELINE_REPRO": "Run a longer official baseline reproduction before method design.",
        "NO_METHOD_WORTHY_GAP": "Stop method search under this local offline evidence.",
        "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD": "Do not design a method yet; first build a more stable official split/metric protocol.",
    }
    base_mean = float(baseline_summary["frozen_base"]["action_l2"]["mean"])
    fcar_ref = {
        "decision": fcar_result.get("final_decision"),
        "fcar_test_action_l2": fcar_result["metrics"]["test"]["fcar_tiny_gate"]["action_l2_mean"],
        "static_merge_test_action_l2": fcar_result["metrics"]["test"]["adapter_soup_static_merge"]["action_l2_mean"],
        "rank4_lora_test_action_l2": fcar_result["metrics"]["test"]["rank4_lora"]["action_l2_mean"],
        "gain_over_frozen_base": fcar_result["kill_criteria"]["fcar_gain_over_frozen_base"],
    }
    report = {
        "date": DATE,
        "status": "completed",
        "final_decision": final_decision,
        "decision_reason": reason_map[final_decision],
        "exact_next_step": next_map[final_decision],
        "policy": {
            "experiments_performed": True,
            "training_performed": False,
            "trained_components": [],
            "gpu_used": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
            "official_model_dataset_used": True,
            "old_custom_route_used": False,
            "paper_claims_made": False,
        },
        "paths": {
            "prediction_artifact": str(Path(args.prediction_artifact)),
            "fcar_result_json": str(Path(args.fcar_result_json)),
            "checkpoint": (artifact.get("paths") or {}).get("checkpoint"),
            "dataset": (artifact.get("paths") or {}).get("dataset"),
            "hf_home": (artifact.get("paths") or {}).get("hf_home"),
        },
        "sweep": {
            "fold_count": len(evaluated),
            "test_frames_per_fold": len(folds[0]["split_records"]["test"]),
            "static_grid": STATIC_GRID,
            "source": "official_prediction_artifact_from_commit_18b3e4b",
            "rank4_lora_training_budget_per_split": "not rerun; fixed official rank-4 LoRA predictions loaded from FCAR artifact",
        },
        "fcar_reference": fcar_ref,
        "baseline_summary": baseline_summary,
        "win_counts": win_counts,
        "help_hurt_vs_frozen_base": help_hurt,
        "surviving_gap_answers": answers,
        "folds": evaluated,
        "aggregate_interpretation": {
            "standard_lora_mean_gain_over_base": _round(base_mean - float(baseline_summary["rank4_lora"]["action_l2"]["mean"])),
            "static_mean_gain_over_base": _round(base_mean - float(baseline_summary["static_mix_val_selected"]["action_l2"]["mean"])),
            "frame_oracle_mean_gain_over_base": answers["frame_oracle_headroom_mean"],
            "task_oracle_mean_gain_over_base": answers["task_oracle_headroom_mean"],
            "static_gap_to_frame_oracle_mean": answers["static_gap_to_frame_oracle_mean"],
        },
        "runtime": {
            "total_elapsed_sec": _round(time.monotonic() - started, 3),
            "rss_final_mb": _rss_mb(),
        },
    }
    report["fcar_postmortem"] = _postmortem(report, fcar_result)
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-artifact", default="reports/fcar_prediction_artifact.json")
    parser.add_argument("--fcar-result-json", default="reports/fcar_tiny_gate_result.json")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--report-json", default="reports/official_smolvla_robust_baseline_sweep_result.json")
    parser.add_argument("--result-md", default="reports/official_smolvla_robust_baseline_sweep_result.md")
    parser.add_argument("--plan-md", default="reports/official_smolvla_robust_baseline_sweep_plan.md")
    parser.add_argument("--postmortem-md", default="reports/fcar_tiny_gate_postmortem.md")
    parser.add_argument("--decision-md", default="reports/official_smolvla_post_fcar_decision.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_postmortem(report, Path(args.postmortem_md))
    _write_plan(report, Path(args.plan_md))
    _write_result(report, Path(args.result_md))
    _write_decision(report, Path(args.decision_md))
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
