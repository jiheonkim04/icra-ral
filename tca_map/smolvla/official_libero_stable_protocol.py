"""Stable official SmolVLA-LIBERO split and metric protocol builder.

This is a protocol-building utility. It reads official LeRobot LIBERO metadata,
creates a deterministic task-stratified episode-disjoint split manifest, and
writes protocol reports. It does not load SmolVLA, train LoRA, implement a new
method, tune FCAR, run rollouts, run OpenVLA-OFT, or download assets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_libero_baseline_scaleup import _json_default, _rss_mb


DATE = "2026-07-10 KST"
FINAL_DECISIONS = {
    "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT",
    "NEEDS_LARGER_PREDICTION_ARTIFACT",
    "NEEDS_TASK_BALANCED_SPLIT",
    "NEEDS_LONGER_LORA_BASELINE_REPRO",
    "SIMPLE_BASELINES_EXPLAIN_GAP",
    "METHOD_DESIGN_STILL_BLOCKED",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _round(value: Any, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def sample_frame_offsets(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return sorted(set(int(x) for x in np.linspace(0, length - 1, num=count, dtype=int)))


def _task_text_map(tasks_df: Any) -> dict[int, str]:
    return {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}


def load_episode_task_map(dataset_root: Path) -> dict[int, int]:
    import pandas as pd

    mapping: dict[int, int] = {}
    files = sorted((dataset_root / "data" / "chunk-000").glob("*.parquet"))
    for path in files:
        frame = pd.read_parquet(path, columns=["episode_index", "task_index"])
        for row in frame.drop_duplicates("episode_index").itertuples(index=False):
            mapping[int(row.episode_index)] = int(row.task_index)
    return mapping


def build_split_manifest(
    *,
    dataset_root: Path,
    seed: int = 0,
    train_episodes_per_task: int = 2,
    val_episodes_per_task: int = 1,
    test_episodes_per_task: int = 2,
    train_frames_per_episode: int = 15,
    val_frames_per_episode: int = 10,
    test_frames_per_episode: int = 15,
) -> dict[str, Any]:
    import pandas as pd

    info = _read_json(dataset_root / "meta" / "info.json")
    episodes_df = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_text = _task_text_map(tasks_df)
    episode_to_task = load_episode_task_map(dataset_root)
    task_to_episodes: dict[int, list[int]] = defaultdict(list)
    for episode, task in sorted(episode_to_task.items()):
        task_to_episodes[int(task)].append(int(episode))

    length_by_episode = {int(row.episode_index): int(row.length) for row in episodes_df.itertuples(index=False)}
    start_by_episode = {int(row.episode_index): int(row.dataset_from_index) for row in episodes_df.itertuples(index=False)}
    required = train_episodes_per_task + val_episodes_per_task + test_episodes_per_task
    eligible_tasks = [task for task in sorted(task_to_episodes) if len(task_to_episodes[task]) >= required]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    task_episode_plan = []
    for task in eligible_tasks:
        episodes = sorted(task_to_episodes[task])
        train_eps = episodes[:train_episodes_per_task]
        val_eps = episodes[train_episodes_per_task : train_episodes_per_task + val_episodes_per_task]
        test_eps = episodes[train_episodes_per_task + val_episodes_per_task : required]
        task_episode_plan.append(
            {
                "task_index": int(task),
                "task": task_text.get(int(task), f"task_{task}"),
                "available_episode_count": len(episodes),
                "train_episodes": train_eps,
                "val_episodes": val_eps,
                "test_episodes": test_eps,
            }
        )
        for split_name, split_eps, frames_per_episode in [
            ("train", train_eps, train_frames_per_episode),
            ("val", val_eps, val_frames_per_episode),
            ("test", test_eps, test_frames_per_episode),
        ]:
            for episode in split_eps:
                length = length_by_episode[episode]
                offsets = sample_frame_offsets(length, frames_per_episode)
                for offset in offsets:
                    splits[split_name].append(
                        {
                            "sample_id": f"{split_name}_task{task}_episode{episode}_frame{offset}",
                            "split": split_name,
                            "task_index": int(task),
                            "task": task_text.get(int(task), f"task_{task}"),
                            "episode_index": int(episode),
                            "frame_index": int(offset),
                            "episode_length": int(length),
                            "dataset_global_index": int(start_by_episode[episode] + offset),
                            "normalized_phase": _round(offset / max(1, length - 1)),
                        }
                    )

    split_episode_sets = {
        split: {int(record["episode_index"]) for record in records}
        for split, records in splits.items()
    }
    split_task_counts = {
        split: dict(sorted(Counter(str(record["task_index"]) for record in records).items(), key=lambda item: int(item[0])))
        for split, records in splits.items()
    }
    return {
        "date": DATE,
        "manifest_version": 1,
        "source": "official_lerobot_libero_metadata",
        "policy": {
            "official_dataset_used": True,
            "downloads_performed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
            "old_custom_libero_7d_route_used": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
        },
        "paths": {
            "dataset_root": str(dataset_root),
        },
        "dataset": {
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "total_tasks": int(info.get("total_tasks", 0)),
            "fps": float(info.get("fps", 10.0)),
            "official_splits": info.get("splits"),
            "eligible_task_count": len(eligible_tasks),
            "required_episodes_per_task": required,
        },
        "sampling_rule": {
            "seed": int(seed),
            "task_stratified": True,
            "episode_disjoint": True,
            "task_order": "ascending official task_index",
            "episode_order": "ascending official episode_index within each task",
            "train_episodes_per_task": train_episodes_per_task,
            "val_episodes_per_task": val_episodes_per_task,
            "test_episodes_per_task": test_episodes_per_task,
            "train_frames_per_episode": train_frames_per_episode,
            "val_frames_per_episode": val_frames_per_episode,
            "test_frames_per_episode": test_frames_per_episode,
            "frame_sampling": "linearly spaced integer offsets from first to last frame of each selected episode",
            "max_frames_per_episode": max(train_frames_per_episode, val_frames_per_episode, test_frames_per_episode),
        },
        "task_episode_plan": task_episode_plan,
        "splits": splits,
        "summary": {
            "frame_counts": {split: len(records) for split, records in splits.items()},
            "episode_counts": {split: len(split_episode_sets[split]) for split in splits},
            "task_counts": {split: len(split_task_counts[split]) for split in splits},
            "task_frame_counts": split_task_counts,
            "leakage_checks": {
                "episode_disjoint_train_val": split_episode_sets["train"].isdisjoint(split_episode_sets["val"]),
                "episode_disjoint_train_test": split_episode_sets["train"].isdisjoint(split_episode_sets["test"]),
                "episode_disjoint_val_test": split_episode_sets["val"].isdisjoint(split_episode_sets["test"]),
            },
        },
    }


def _load_previous_reports(robust_result_path: Path, fcar_result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json(robust_result_path), _read_json(fcar_result_path)


def _instability_diagnosis(robust: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    answers = robust.get("surviving_gap_answers") or {}
    sweep = robust.get("sweep") or {}
    return {
        "likely_sources": {
            "too_few_heldout_frames": {
                "likely": True,
                "evidence": f"Previous sweep used {sweep.get('fold_count')} folds with {sweep.get('test_frames_per_fold')} test frames per fold from a 200-frame artifact.",
            },
            "task_imbalance": {
                "likely": True,
                "evidence": "Previous fold tests each covered one task pair; stable manifest now covers all eligible official tasks in each split.",
            },
            "episode_leakage": {
                "likely": False,
                "evidence": "Previous and new manifests use episode-disjoint splits; the problem is coverage, not known leakage.",
            },
            "insufficient_episode_disjoint_coverage": {
                "likely": True,
                "evidence": "Previous artifact used 10 held-out episodes across 5 tasks; stable manifest selects 200 episodes across 40 tasks.",
            },
            "lora_regeneration_mismatch": {
                "likely": "unresolved",
                "evidence": "Robust sweep reused one rank-4 LoRA artifact and did not retrain LoRA per fold or seed.",
            },
            "small_prediction_artifact": {
                "likely": True,
                "evidence": "Current artifact has 200 frames; stable manifest requires 2800 prediction records.",
            },
            "metric_definition_variance": {
                "likely": True,
                "evidence": "Action L2 rank order changed by fold; task-balanced and bootstrap intervals were not fixed before FCAR.",
            },
            "action_component_imbalance": {
                "likely": True,
                "evidence": "Raw 7D L2 mixes translation, rotation, and gripper units; component metrics must be reported separately.",
            },
            "gripper_or_rotation_weighting": {
                "likely": True,
                "evidence": "Previous large errors often involved gripper; raw aggregate can hide component-specific behavior.",
            },
            "task_level_distribution_shift": {
                "likely": True,
                "evidence": "Rank-4 LoRA beat frozen/base in only 2/5 folds and won no realistic fold.",
            },
            "static_validation_instability": {
                "likely": True,
                "evidence": "Validation-selected static alpha won 3/5 folds but was selected from very small validation slices.",
            },
        },
        "must_fix_before_method_design": [
            "use the fixed task-stratified episode-disjoint manifest",
            "generate larger official base/LoRA prediction artifacts under the manifest",
            "report both frame-weighted and task-balanced metrics",
            "select static alpha on validation only and freeze before test",
            "add episode/task bootstrap intervals",
            "run independent rank-4 LoRA seeds only after the manifest and metrics are frozen",
            "keep FCAR killed unless a future frozen-criteria report beats static merge",
        ],
        "previous_decision": robust.get("final_decision"),
        "new_manifest_frame_counts": manifest["summary"]["frame_counts"],
    }


def _metric_protocol() -> dict[str, Any]:
    return {
        "primary_metric": "aggregate raw 7D action L2 after official SmolVLA postprocessing",
        "dimensions": {
            "translation": [0, 1, 2],
            "rotation": [3, 4, 5],
            "gripper": [6],
        },
        "secondary_metrics": [
            "normalized SmolVLA eval loss",
            "translation L2 over action dims 0:3",
            "rotation L2 over action dims 3:6",
            "gripper absolute error on dim 6",
            "gripper sign accuracy on dim 6",
            "per-action-dimension absolute error and L2",
            "per-task breakdown",
            "per-episode breakdown",
            "action validity/range violation rate using official action stats",
            "help/hurt counts vs frozen/base",
            "win counts across fixed test tasks/subsets",
            "episode bootstrap confidence interval when cheap",
            "task bootstrap confidence interval when cheap",
        ],
        "averaging": {
            "primary": "frame-weighted mean over fixed test frames",
            "required_secondary": "task-balanced mean of per-task frame means",
            "episode_report": "per-episode means and episode-bootstrap intervals",
            "task_report": "per-task means and task-bootstrap intervals",
        },
        "static_alpha_protocol": {
            "grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "selection": "choose alpha on validation split only by primary action L2, then freeze for test",
            "test_tuning_allowed": False,
        },
        "oracle_reporting": {
            "frame_oracle": "upper bound only; may use labels only for reporting",
            "task_oracle": "upper bound only; may use labels only for reporting",
            "action_dim_oracle": "diagnostic upper bound only, not a realistic baseline",
        },
    }


def _artifact_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    total_frames = sum(int(value) for value in manifest["summary"]["frame_counts"].values())
    return {
        "status": "planned_not_generated",
        "reason_not_generated": "Generating official SmolVLA predictions for the 2800-frame manifest would require a larger bounded GPU run and rank-4 LoRA artifact regeneration; this protocol run freezes the split/metrics first.",
        "required_contents": [
            "frozen/base predictions",
            "rank-4 LoRA predictions",
            "raw 7D labels",
            "official state/action metadata",
            "task and instruction identifiers from official metadata",
            "split membership from reports/official_smolvla_split_manifest.json",
            "per-frame action errors",
            "normalized eval loss when available",
            "CUDA/device/VRAM/runtime audit if LoRA is regenerated",
        ],
        "target_prediction_records": total_frames,
        "recommended_output": "reports/official_smolvla_stable_prediction_artifact.json",
        "exact_next_command": "powershell -ExecutionPolicy Bypass -File scripts\\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\\official_smolvla_split_manifest.json -Output reports\\official_smolvla_stable_prediction_artifact.json",
        "allowed_training": "standard rank-4 LoRA baseline only, fixed small budget, no new method",
        "forbidden": [
            "FCAR tuning",
            "new routing method",
            "simulator rollout",
            "full benchmark",
            "OpenVLA-OFT",
            "old custom LIBERO_7D route",
            "new downloads",
        ],
    }


def _choose_decision(manifest: dict[str, Any], artifact_plan: dict[str, Any]) -> str:
    counts = manifest["summary"]["frame_counts"]
    leakage = manifest["summary"]["leakage_checks"]
    if not all(bool(value) for value in leakage.values()):
        return "NEEDS_TASK_BALANCED_SPLIT"
    if min(int(counts["train"]), int(counts["test"])) < 500 or int(counts["val"]) < 200:
        return "NEEDS_TASK_BALANCED_SPLIT"
    if artifact_plan["status"] != "generated":
        return "NEEDS_LARGER_PREDICTION_ARTIFACT"
    return "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_split_manifest_md(report: dict[str, Any], path: Path) -> None:
    manifest = report["split_manifest"]
    summary = manifest["summary"]
    lines = [
        "# Official SmolVLA Split Manifest",
        "",
        f"Date: {report['date']}",
        "",
        f"- JSON manifest: `reports/official_smolvla_split_manifest.json`",
        f"- source: `{manifest['source']}`",
        f"- eligible tasks: `{manifest['dataset']['eligible_task_count']}`",
        f"- frame counts: `{summary['frame_counts']}`",
        f"- episode counts: `{summary['episode_counts']}`",
        f"- task counts: `{summary['task_counts']}`",
        f"- leakage checks: `{summary['leakage_checks']}`",
        "",
        "## Sampling Rule",
        "",
        *[f"- {key}: `{value}`" for key, value in manifest["sampling_rule"].items()],
        "",
        "The manifest is task-stratified, episode-disjoint, and deterministic. It is a protocol artifact only; no model inference or training happened while creating it.",
    ]
    _write_lines(path, lines)


def _write_plan(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Stable Protocol Plan",
        "",
        f"Date: {report['date']}",
        "",
        "Purpose: freeze the split and metric protocol before any new official SmolVLA-LIBERO baseline or method work.",
        "",
        "Hard boundary:",
        "",
        "- no new method design",
        "- no FCAR revival or tuning",
        "- no simulator rollout or full benchmark",
        "- no OpenVLA-OFT",
        "- no old custom LIBERO_7D route",
        "- no new large downloads",
        "",
        "Stable protocol target:",
        "",
        f"- train frames: `{report['split_manifest']['summary']['frame_counts']['train']}`",
        f"- val frames: `{report['split_manifest']['summary']['frame_counts']['val']}`",
        f"- test frames: `{report['split_manifest']['summary']['frame_counts']['test']}`",
        f"- tasks per split: `{report['split_manifest']['summary']['task_counts']}`",
        "",
        "Decision rule: do not design methods until a larger official prediction artifact is generated and evaluated under this manifest and metric protocol.",
    ]
    _write_lines(path, lines)


def _write_metric_protocol(report: dict[str, Any], path: Path) -> None:
    metric = report["metric_protocol"]
    lines = [
        "# Official SmolVLA Metric Protocol",
        "",
        f"Date: {report['date']}",
        "",
        f"Primary metric: {metric['primary_metric']}",
        "",
        "Dimensions:",
        "",
        *[f"- {key}: `{value}`" for key, value in metric["dimensions"].items()],
        "",
        "Secondary metrics:",
        "",
        *[f"- {item}" for item in metric["secondary_metrics"]],
        "",
        "Averaging:",
        "",
        *[f"- {key}: {value}" for key, value in metric["averaging"].items()],
        "",
        "Static alpha:",
        "",
        *[f"- {key}: `{value}`" for key, value in metric["static_alpha_protocol"].items()],
        "",
        "Oracle baselines are upper bounds only and must not be presented as realistic methods.",
    ]
    _write_lines(path, lines)


def _write_artifact_plan(report: dict[str, Any], path: Path) -> None:
    plan = report["prediction_artifact_plan"]
    lines = [
        "# Official SmolVLA Prediction Artifact Plan",
        "",
        f"Date: {report['date']}",
        "",
        f"- status: `{plan['status']}`",
        f"- target prediction records: `{plan['target_prediction_records']}`",
        f"- recommended output: `{plan['recommended_output']}`",
        "",
        "Reason not generated:",
        "",
        plan["reason_not_generated"],
        "",
        "Required contents:",
        "",
        *[f"- {item}" for item in plan["required_contents"]],
        "",
        "Exact next command:",
        "",
        "```powershell",
        plan["exact_next_command"],
        "```",
    ]
    _write_lines(path, lines)


def _write_result(report: dict[str, Any], path: Path) -> None:
    diag = report["instability_diagnosis"]
    lines = [
        "# Official SmolVLA Stable Protocol Result",
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
        "",
        "## Instability Diagnosis",
        "",
        *[f"- {key}: `{value}`" for key, value in diag["likely_sources"].items()],
        "",
        "Must fix before method design:",
        "",
        *[f"- {item}" for item in diag["must_fix_before_method_design"]],
        "",
        "## Split Manifest",
        "",
        f"- status: `{report['split_manifest_status']}`",
        f"- frame counts: `{report['split_manifest']['summary']['frame_counts']}`",
        f"- leakage checks: `{report['split_manifest']['summary']['leakage_checks']}`",
        "",
        "## Metric Protocol",
        "",
        f"- status: `{report['metric_protocol_status']}`",
        f"- primary metric: `{report['metric_protocol']['primary_metric']}`",
        "",
        "## Artifact",
        "",
        f"- status: `{report['prediction_artifact_plan']['status']}`",
        f"- exact next command: `{report['prediction_artifact_plan']['exact_next_command']}`",
    ]
    _write_lines(path, lines)


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Stable Protocol Decision",
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
    dataset_root = Path(args.dataset_root)
    manifest = build_split_manifest(
        dataset_root=dataset_root,
        seed=int(args.seed),
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        test_episodes_per_task=int(args.test_episodes_per_task),
        train_frames_per_episode=int(args.train_frames_per_episode),
        val_frames_per_episode=int(args.val_frames_per_episode),
        test_frames_per_episode=int(args.test_frames_per_episode),
    )
    robust, fcar = _load_previous_reports(Path(args.robust_result_json), Path(args.fcar_result_json))
    metric = _metric_protocol()
    artifact_plan = _artifact_plan(manifest)
    decision = _choose_decision(manifest, artifact_plan)
    reason_map = {
        "NEEDS_LARGER_PREDICTION_ARTIFACT": "Fixed manifest and metric protocol are ready, but no larger official prediction artifact has been generated under them.",
        "NEEDS_TASK_BALANCED_SPLIT": "The proposed manifest is not sufficiently task-balanced or episode-disjoint.",
        "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT": "Stable manifest, metrics, and larger artifact are all available.",
        "NEEDS_LONGER_LORA_BASELINE_REPRO": "LoRA requires independent baseline seeds under the stable protocol.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Simple baselines explain the observed gain.",
        "METHOD_DESIGN_STILL_BLOCKED": "Metric or split instability remains too high.",
    }
    next_map = {
        "NEEDS_LARGER_PREDICTION_ARTIFACT": artifact_plan["exact_next_command"],
        "NEEDS_TASK_BALANCED_SPLIT": "Revise the split manifest before any model inference.",
        "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT": "Run official baseline reproduction under the fixed protocol.",
        "NEEDS_LONGER_LORA_BASELINE_REPRO": "Run independent standard rank-4 LoRA baseline seeds under the fixed protocol.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Stop method search unless a new official benchmark residual appears.",
        "METHOD_DESIGN_STILL_BLOCKED": "Stabilize metric/split protocol before method design.",
    }
    report = {
        "date": DATE,
        "status": "completed",
        "final_decision": decision,
        "decision_reason": reason_map[decision],
        "exact_next_step": next_map[decision],
        "policy": {
            "experiments_performed": False,
            "training_performed": False,
            "trained_components": [],
            "gpu_used": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "official_model_dataset_used": True,
            "old_custom_route_used": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
            "paper_claims_made": False,
        },
        "paths": {
            "dataset_root": str(dataset_root),
            "robust_result_json": str(Path(args.robust_result_json)),
            "fcar_result_json": str(Path(args.fcar_result_json)),
            "split_manifest_json": str(Path(args.split_manifest_json)),
        },
        "instability_diagnosis": _instability_diagnosis(robust, manifest),
        "split_manifest_status": "created",
        "split_manifest": manifest,
        "metric_protocol_status": "created",
        "metric_protocol": metric,
        "prediction_artifact_plan": artifact_plan,
        "baseline_smoke": {
            "run": False,
            "reason": "No larger prediction artifact was generated in this protocol-building run.",
        },
        "fcar_status": {
            "remains_killed": True,
            "previous_decision": fcar.get("final_decision"),
            "kill_result_changed": False,
        },
        "runtime": {
            "total_elapsed_sec": _round(time.monotonic() - started, 3),
            "rss_final_mb": _rss_mb(),
        },
    }
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"C:\assets\datasets\lerobot_libero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes-per-task", type=int, default=2)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--test-episodes-per-task", type=int, default=2)
    parser.add_argument("--train-frames-per-episode", type=int, default=15)
    parser.add_argument("--val-frames-per-episode", type=int, default=10)
    parser.add_argument("--test-frames-per-episode", type=int, default=15)
    parser.add_argument("--robust-result-json", default="reports/official_smolvla_robust_baseline_sweep_result.json")
    parser.add_argument("--fcar-result-json", default="reports/fcar_tiny_gate_result.json")
    parser.add_argument("--result-json", default="reports/official_smolvla_stable_protocol_result.json")
    parser.add_argument("--result-md", default="reports/official_smolvla_stable_protocol_result.md")
    parser.add_argument("--plan-md", default="reports/official_smolvla_stable_protocol_plan.md")
    parser.add_argument("--split-manifest-md", default="reports/official_smolvla_split_manifest.md")
    parser.add_argument("--split-manifest-json", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--metric-md", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--artifact-plan-md", default="reports/official_smolvla_prediction_artifact_plan.md")
    parser.add_argument("--decision-md", default="reports/official_smolvla_stable_protocol_decision.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    result_json = Path(args.result_json)
    split_json = Path(args.split_manifest_json)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    split_json.write_text(json.dumps(report["split_manifest"], indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_plan(report, Path(args.plan_md))
    _write_split_manifest_md(report, Path(args.split_manifest_md))
    _write_metric_protocol(report, Path(args.metric_md))
    _write_artifact_plan(report, Path(args.artifact_plan_md))
    _write_result(report, Path(args.result_md))
    _write_decision(report, Path(args.decision_md))
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
