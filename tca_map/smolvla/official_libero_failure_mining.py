"""Official SmolVLA-LIBERO failure mining and method-readiness gate.

This module evaluates the already-downloaded official SmolVLA-LIBERO assets on
a bounded, deterministic held-out subset. It does not implement a method, run a
simulator, run a full benchmark, execute OpenVLA-OFT, or download assets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _finite_round_list,
    _gradient_summary,
    _json_default,
    _loss_from_output,
    _parameter_summary,
    _postprocess_action,
    _raw_current_action,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)


FINAL_DECISIONS = {
    "GO_METHOD_DESIGN_GRIPPER_PHASE",
    "GO_METHOD_DESIGN_CONTROL_STABILITY",
    "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING",
    "NEED_LONGER_OFFICIAL_BASELINE_REPRO",
    "NO_METHOD_WORTHY_GAP",
    "METRIC_CONFLICT_BLOCKS_METHOD",
}

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_CLOUD_HANDOFF",
]

MAX_RUNTIME_SECONDS = 2 * 60 * 60
MAX_TRAINING_STEPS = 100
DEFAULT_TRAIN_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 9)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.max(values)), 9)


def _phase(frame_index: int, episode_length: int) -> str:
    if episode_length <= 1:
        return "unknown"
    ratio = frame_index / max(1, episode_length - 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "mid"
    return "late"


def _task_text_map(tasks_df: Any) -> dict[int, str]:
    return {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}


def _episode_task_map(dataset_root: Path, max_files: int | None = None) -> dict[int, int]:
    import pandas as pd

    mapping: dict[int, int] = {}
    files = sorted((dataset_root / "data" / "chunk-000").glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    for path in files:
        frame = pd.read_parquet(path, columns=["episode_index", "task_index"])
        for row in frame.drop_duplicates("episode_index").itertuples(index=False):
            mapping[int(row.episode_index)] = int(row.task_index)
    return mapping


def select_eval_plan(
    *,
    dataset_root: Path,
    train_episode: int,
    max_eval_samples: int,
    min_task_groups: int,
    episodes_per_task: int,
) -> dict[str, Any]:
    import pandas as pd

    episodes_df = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_text = _task_text_map(tasks_df)
    ep_to_task = _episode_task_map(dataset_root)
    task_to_eps: dict[int, list[int]] = defaultdict(list)
    for ep in sorted(ep_to_task):
        if ep == train_episode:
            continue
        task = ep_to_task[ep]
        if task == ep_to_task.get(train_episode):
            continue
        if len(task_to_eps[task]) < episodes_per_task:
            task_to_eps[task].append(ep)
        if len([key for key, value in task_to_eps.items() if len(value) >= episodes_per_task]) >= min_task_groups:
            break

    chosen_tasks = [task for task in sorted(task_to_eps) if len(task_to_eps[task]) >= episodes_per_task][:min_task_groups]
    selected_episodes = [ep for task in chosen_tasks for ep in task_to_eps[task]]
    if not selected_episodes:
        raise RuntimeError("No held-out official episodes selected for failure mining.")

    lengths = {int(row.episode_index): int(row.length) for row in episodes_df.itertuples(index=False)}
    samples_per_episode = max(1, int(math.ceil(max_eval_samples / len(selected_episodes))))
    local_offsets: dict[int, int] = {}
    offset = 0
    for ep in selected_episodes:
        local_offsets[ep] = offset
        offset += lengths[ep]

    samples = []
    for task in chosen_tasks:
        for ep in task_to_eps[task]:
            length = lengths[ep]
            frames = np.linspace(0, max(0, length - 1), num=samples_per_episode, dtype=int)
            for frame_index in sorted(set(int(x) for x in frames)):
                samples.append(
                    {
                        "dataset_local_index": int(local_offsets[ep] + frame_index),
                        "episode_index": int(ep),
                        "frame_index": int(frame_index),
                        "episode_length": int(length),
                        "task_index": int(task),
                        "task": task_text.get(int(task), f"task_{task}"),
                        "phase": _phase(int(frame_index), int(length)),
                    }
                )
                if len(samples) >= max_eval_samples:
                    break
            if len(samples) >= max_eval_samples:
                break
        if len(samples) >= max_eval_samples:
            break

    return {
        "selected_tasks": [
            {"task_index": int(task), "task": task_text.get(int(task), f"task_{task}"), "episodes": task_to_eps[task]}
            for task in chosen_tasks
        ],
        "selected_episodes": selected_episodes,
        "sample_count": len(samples),
        "samples": samples,
        "episode_task_map_size": len(ep_to_task),
    }


def _range_violations(pred: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> int:
    dim = min(pred.shape[0], action_min.shape[0], action_max.shape[0])
    clipped = pred[:dim]
    return int(np.sum((clipped < action_min[:dim]) | (clipped > action_max[:dim])))


def _metric_row(
    *,
    sample_meta: dict[str, Any],
    pred: np.ndarray,
    target: np.ndarray,
    eval_loss: float | None,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict[str, Any]:
    dim = min(pred.shape[0], target.shape[0])
    pred = pred[:dim]
    target = target[:dim]
    diff = pred - target
    return {
        **sample_meta,
        "eval_loss": round(float(eval_loss), 9) if eval_loss is not None else None,
        "action_l2": round(float(np.linalg.norm(diff)), 9),
        "translation_l2": round(float(np.linalg.norm(diff[:3])), 9) if dim >= 3 else None,
        "rotation_l2": round(float(np.linalg.norm(diff[3:6])), 9) if dim >= 6 else None,
        "gripper_abs": round(float(abs(diff[6])), 9) if dim >= 7 else None,
        "gripper_sign_match": bool(np.sign(pred[6]) == np.sign(target[6])) if dim >= 7 else None,
        "finite": bool(np.isfinite(pred).all() and np.isfinite(target).all()),
        "range_violation_count": _range_violations(pred, action_min, action_max),
        "per_dim_abs": _finite_round_list(np.abs(diff), 7),
        "pred_preview": _finite_round_list(pred, 7),
        "target_preview": _finite_round_list(target, 7),
    }


def evaluate_model_rows(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    samples: list[dict[str, Any]],
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    import torch

    rows = []
    policy.eval()
    with torch.no_grad():
        for sample_meta in samples:
            raw_sample = dataset[int(sample_meta["dataset_local_index"])]
            batch = _add_training_batch_dims(preprocessor(raw_sample))
            output = policy.forward(batch)
            eval_loss = _to_float(_loss_from_output(output))
            if hasattr(policy, "reset"):
                policy.reset()
            selected = policy.select_action(batch)
            pred = _postprocess_action(selected, postprocessor)
            target = _raw_current_action(raw_sample)
            rows.append(
                _metric_row(
                    sample_meta=sample_meta,
                    pred=pred,
                    target=target,
                    eval_loss=eval_loss,
                    action_min=action_min,
                    action_max=action_max,
                )
            )
    return rows


def evaluate_mean_prior_rows(
    *,
    dataset: Any,
    samples: list[dict[str, Any]],
    mean_action: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for sample_meta in samples:
        raw_sample = dataset[int(sample_meta["dataset_local_index"])]
        target = _raw_current_action(raw_sample)
        rows.append(
            _metric_row(
                sample_meta=sample_meta,
                pred=mean_action,
                target=target,
                eval_loss=None,
                action_min=action_min,
                action_max=action_max,
            )
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    per_dim = []
    for dim in range(7):
        dim_values = [float(row["per_dim_abs"][dim]) for row in rows if len(row.get("per_dim_abs") or []) > dim]
        per_dim.append(round(float(np.mean(dim_values)), 9) if dim_values else None)
    return {
        "sample_count": len(rows),
        "eval_loss_mean": _mean(values("eval_loss")),
        "eval_loss_max": _max(values("eval_loss")),
        "action_l2_mean": _mean(values("action_l2")),
        "action_l2_max": _max(values("action_l2")),
        "translation_l2_mean": _mean(values("translation_l2")),
        "rotation_l2_mean": _mean(values("rotation_l2")),
        "gripper_abs_mean": _mean(values("gripper_abs")),
        "gripper_sign_accuracy": _mean([1.0 if row.get("gripper_sign_match") else 0.0 for row in rows if row.get("gripper_sign_match") is not None]),
        "finite_all": all(bool(row.get("finite")) for row in rows),
        "range_violation_rate": _mean([1.0 if int(row.get("range_violation_count") or 0) > 0 else 0.0 for row in rows]),
        "per_dim_abs_mean": per_dim,
    }


def _group_summary(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {key: summarize_rows(value) for key, value in sorted(groups.items())}


def compare_variants(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]], mean_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["episode_index"], row["frame_index"], row["task_index"]): row
        for row in base_rows
    }
    lora_by_key = {
        (row["episode_index"], row["frame_index"], row["task_index"]): row
        for row in lora_rows
    }
    mean_by_key = {
        (row["episode_index"], row["frame_index"], row["task_index"]): row
        for row in mean_rows
    }
    merged = []
    for key, base in by_key.items():
        lora = lora_by_key[key]
        mean = mean_by_key[key]
        merged.append(
            {
                "episode_index": base["episode_index"],
                "frame_index": base["frame_index"],
                "task_index": base["task_index"],
                "task": base["task"],
                "phase": base["phase"],
                "base_action_l2": base["action_l2"],
                "lora_action_l2": lora["action_l2"],
                "mean_action_l2": mean["action_l2"],
                "action_l2_gain_base_minus_lora": round(float(base["action_l2"] - lora["action_l2"]), 9),
                "lora_eval_loss_delta": round(float(lora["eval_loss"] - base["eval_loss"]), 9),
                "base_eval_loss": base["eval_loss"],
                "lora_eval_loss": lora["eval_loss"],
                "base_gripper_abs": base["gripper_abs"],
                "lora_gripper_abs": lora["gripper_abs"],
                "base_range_violation_count": base["range_violation_count"],
                "lora_range_violation_count": lora["range_violation_count"],
            }
        )

    helps = sorted(merged, key=lambda row: row["action_l2_gain_base_minus_lora"], reverse=True)
    hurts = sorted(merged, key=lambda row: row["action_l2_gain_base_minus_lora"])
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phase_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        task_rows[str(row["task_index"])].append(row)
        phase_rows[str(row["phase"])].append(row)

    def compact_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        gains = [float(row["action_l2_gain_base_minus_lora"]) for row in rows]
        loss_deltas = [float(row["lora_eval_loss_delta"]) for row in rows]
        mean_l2s = [float(row["mean_action_l2"]) for row in rows]
        lora_l2s = [float(row["lora_action_l2"]) for row in rows]
        return {
            "sample_count": len(rows),
            "lora_action_l2_gain_mean": _mean(gains),
            "lora_action_l2_gain_max": _max(gains),
            "lora_action_l2_gain_min": round(float(np.min(gains)), 9) if gains else None,
            "lora_eval_loss_delta_mean": _mean(loss_deltas),
            "lora_help_fraction": _mean([1.0 if gain > 0 else 0.0 for gain in gains]),
            "mean_prior_better_than_lora_fraction": _mean([1.0 if mean_l2s[i] < lora_l2s[i] else 0.0 for i in range(len(rows))]),
        }

    task_compact = {key: compact_group(value) for key, value in sorted(task_rows.items(), key=lambda item: int(item[0]))}
    task_help_mean_count = sum(1 for value in task_compact.values() if (value.get("lora_action_l2_gain_mean") or 0) > 0)
    task_hurt_mean_count = sum(1 for value in task_compact.values() if (value.get("lora_action_l2_gain_mean") or 0) < 0)
    return {
        "lora_help_count": sum(1 for row in merged if row["action_l2_gain_base_minus_lora"] > 0),
        "lora_hurt_count": sum(1 for row in merged if row["action_l2_gain_base_minus_lora"] < 0),
        "lora_eval_loss_worse_count": sum(1 for row in merged if row["lora_eval_loss_delta"] > 0),
        "mean_prior_better_than_lora_count": sum(1 for row in merged if row["mean_action_l2"] < row["lora_action_l2"]),
        "task_mean_help_count": task_help_mean_count,
        "task_mean_hurt_count": task_hurt_mean_count,
        "examples_lora_helps": helps[:5],
        "examples_lora_hurts": hurts[:5],
        "per_task_delta": task_compact,
        "per_phase_delta": {key: compact_group(value) for key, value in sorted(phase_rows.items())},
    }


def choose_method_readiness_decision(result: dict[str, Any]) -> str:
    comparison = result.get("comparison") or {}
    metrics = result.get("aggregate_metrics") or {}
    base = metrics.get("frozen_base") or {}
    lora = metrics.get("rank4_lora") or {}
    mean_prior = metrics.get("mean_action_prior") or {}
    sample_count = int(base.get("sample_count") or 0)
    if sample_count < 100:
        return "NEED_LONGER_OFFICIAL_BASELINE_REPRO"

    base_l2 = base.get("action_l2_mean")
    lora_l2 = lora.get("action_l2_mean")
    mean_l2 = mean_prior.get("action_l2_mean")
    base_loss = base.get("eval_loss_mean")
    lora_loss = lora.get("eval_loss_mean")
    if None in [base_l2, lora_l2, mean_l2, base_loss, lora_loss]:
        return "NEED_LONGER_OFFICIAL_BASELINE_REPRO"

    action_gain = float(base_l2) - float(lora_l2)
    loss_delta = float(lora_loss) - float(base_loss)
    mean_explains = float(mean_l2) <= min(float(base_l2), float(lora_l2))
    help_count = int(comparison.get("lora_help_count") or 0)
    hurt_count = int(comparison.get("lora_hurt_count") or 0)
    loss_worse_count = int(comparison.get("lora_eval_loss_worse_count") or 0)
    task_help_count = int(comparison.get("task_mean_help_count") or 0)
    task_hurt_count = int(comparison.get("task_mean_hurt_count") or 0)
    gripper_acc = lora.get("gripper_sign_accuracy")
    gripper_abs = lora.get("gripper_abs_mean")
    if mean_explains:
        return "NO_METHOD_WORTHY_GAP"
    if gripper_acc is not None and float(gripper_acc) < 0.90 and gripper_abs is not None and float(gripper_abs) > 0.10:
        return "GO_METHOD_DESIGN_GRIPPER_PHASE"
    if action_gain > 0 and loss_delta > 0 and loss_worse_count >= sample_count * 0.60:
        return "GO_METHOD_DESIGN_CONTROL_STABILITY"
    if (
        help_count >= sample_count * 0.30
        and hurt_count >= sample_count * 0.30
        and task_help_count >= 1
        and task_hurt_count >= 1
    ):
        return "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING"
    if action_gain > 0 and loss_delta > 0:
        return "METRIC_CONFLICT_BLOCKS_METHOD"
    if abs(action_gain) < 0.005:
        return "NO_METHOD_WORTHY_GAP"
    return "NEED_LONGER_OFFICIAL_BASELINE_REPRO"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    rec = report.get("metric_reconciliation") or {}
    metrics = report.get("aggregate_metrics") or {}
    comp = report.get("comparison") or {}
    gaps = report.get("gap_analysis") or {}
    lines = [
        "# Official SmolVLA-LIBERO Failure Mining Result",
        "",
        f"- final decision: `{report.get('final_decision')}`",
        f"- status: `{report.get('status')}`",
        f"- experiments happened: `{report.get('policy', {}).get('experiments_performed')}`",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- downloads/OpenVLA-OFT/rollout: `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}` / `{report.get('policy', {}).get('rollouts_performed')}`",
        f"- official model/dataset used: `{report.get('policy', {}).get('official_model_dataset_used')}`",
        "",
        "## Metric Reconciliation",
        "",
        f"- primary metric recommendation: `{rec.get('primary_metric_recommendation')}`",
        f"- secondary metric recommendation: `{rec.get('secondary_metric_recommendation')}`",
        f"- warning: `{rec.get('warning')}`",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for name in ["frozen_base", "rank4_lora", "mean_action_prior"]:
        item = metrics.get(name) or {}
        lines.extend(
            [
                f"### {name}",
                "",
                f"- sample count: `{item.get('sample_count')}`",
                f"- eval loss mean: `{item.get('eval_loss_mean')}`",
                f"- action L2 mean: `{item.get('action_l2_mean')}`",
                f"- translation L2 mean: `{item.get('translation_l2_mean')}`",
                f"- rotation L2 mean: `{item.get('rotation_l2_mean')}`",
                f"- gripper abs mean/sign accuracy: `{item.get('gripper_abs_mean')}` / `{item.get('gripper_sign_accuracy')}`",
                f"- range violation rate: `{item.get('range_violation_rate')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            f"- LoRA help/hurt count: `{comp.get('lora_help_count')}` / `{comp.get('lora_hurt_count')}`",
            f"- LoRA eval loss worse count: `{comp.get('lora_eval_loss_worse_count')}`",
            f"- mean prior better than LoRA count: `{comp.get('mean_prior_better_than_lora_count')}`",
            "",
            "## Gap Analysis",
            "",
            f"- strongest method-worthy gap: `{gaps.get('strongest_method_worthy_gap')}`",
            f"- estimated kill risk: `{gaps.get('estimated_kill_risk')}`",
            f"- recommended method direction: `{gaps.get('recommended_method_direction')}`",
            "",
            f"Exact next prompt: {report.get('exact_next_prompt')}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    report: dict[str, Any] = {
        "status": "started",
        "final_decision": None,
        "policy": {
            "experiments_performed": True,
            "downloads_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_claims_made": False,
            "custom_libero_7d_route_used": False,
            "method_implemented": False,
            "official_model_dataset_used": True,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "hf_home": str(Path(args.hf_home)),
            "vlm_root": str(Path(args.vlm_root)),
        },
        "risk_assessment": {
            "task": "bounded official SmolVLA-LIBERO failure mining",
            "new_download_expected_bytes": 0,
            "max_eval_samples": int(args.max_eval_samples),
            "training_steps": int(args.steps),
            "decision": "proceed inside bounded local GPU budget",
        },
        "errors": [],
    }

    def fail(message: str, code: int) -> tuple[dict[str, Any], int]:
        report["status"] = "failed"
        report["final_decision"] = "NEED_LONGER_OFFICIAL_BASELINE_REPRO"
        report["errors"].append({"message": message})
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, code

    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        return fail("Forbidden gate(s) set: " + ", ".join(forbidden), 2)
    if not _env_flag("ALLOW_HEAVY_IMPORT") or not _env_flag("ALLOW_GPU_TRAINING"):
        return fail("Requires ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1.", 3)
    if int(args.steps) > MAX_TRAINING_STEPS:
        return fail(f"Training steps exceed previous official scaleup cap: {args.steps}", 4)

    try:
        import pandas as pd
        import torch
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not torch.cuda.is_available():
            return fail("CUDA unavailable; stopping instead of CPU fallback.", 5)
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        device = "cuda"

        checkpoint_path = Path(args.checkpoint_path)
        dataset_root = Path(args.dataset_root)
        hf_home = Path(args.hf_home)
        vlm_root = Path(args.vlm_root)
        info = _read_json(dataset_root / "meta" / "info.json")
        stats = _read_json(dataset_root / "meta" / "stats.json")
        tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
        fps = float(info.get("fps", 10.0))
        chunk_size = int(args.chunk_size)
        delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
        action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
        action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)
        mean_action = np.asarray(_stat_vector(stats, "action", "mean"), dtype=np.float32)

        eval_plan = select_eval_plan(
            dataset_root=dataset_root,
            train_episode=0,
            max_eval_samples=int(args.max_eval_samples),
            min_task_groups=int(args.min_task_groups),
            episodes_per_task=int(args.episodes_per_task),
        )
        eval_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=eval_plan["selected_episodes"],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )
        train_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=[0],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )

        cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
        cfg.device = device
        cfg.load_vlm_weights = True
        cfg.compile_model = False
        cfg.push_to_hub = False
        cfg.vlm_model_name = str(vlm_root)
        cfg.chunk_size = chunk_size
        policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path,
            config=cfg,
            local_files_only=True,
            cache_dir=hf_home,
            token=False,
            strict=False,
        )
        policy.to(device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
                "device_processor": {"device": device},
            },
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        probe = _add_training_batch_dims(preprocessor(train_dataset[0]))
        input_devices = _tensor_devices(probe)
        param_summary = _parameter_summary(policy)
        if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
            value.startswith("cuda") for value in input_devices.values()
        ):
            return fail(f"CUDA fallback detected: params={param_summary}, inputs={input_devices}", 6)

        metric_reconciliation = {
            "eval_loss_measures": "SmolVLAPolicy.forward returns a scalar flow/action-chunk loss over normalized action chunks, with padding removed and action dimensions capped to max_action_dim.",
            "eval_loss_includes": ["normalized action chunk flow objective", "future action chunk timesteps", "padding mask when action_is_pad/actions_id_pad exists"],
            "eval_loss_does_not_measure": ["postprocessed one-step raw action L2", "simulator success", "language/token prediction loss"],
            "action_l2_measures": "Official postprocessor output for select_action compared with raw 7D action label at the current frame.",
            "split_alignment": "LoRA trains on episode 0 only; failure mining excludes that episode and uses deterministic held-out official episodes from multiple task indices inside the official train split because no official eval split is present.",
            "primary_metric_recommendation": "postprocessed held-out action L2 with translation/rotation/gripper breakdown and mean-action prior comparison",
            "secondary_metric_recommendation": "normalized chunk flow eval loss as a stability/retention warning metric",
            "both_should_be_reported": True,
            "warning": "If action L2 improves while eval loss worsens, treat it as a control-stability/retention warning, not as a paper-grade success.",
        }

        base_rows = evaluate_model_rows(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            action_min=action_min,
            action_max=action_max,
        )
        mean_rows = evaluate_mean_prior_rows(
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            mean_action=mean_action,
            action_min=action_min,
            action_max=action_max,
        )

        policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
        policy.to(device)
        policy.train()
        lora_param_summary = _parameter_summary(policy)
        optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))
        loss_curve = []
        grad_curve = []
        training_started = time.monotonic()
        report["policy"]["training_performed"] = True
        train_indices = [idx for idx in DEFAULT_TRAIN_INDICES if idx < len(train_dataset)]
        for step in range(int(args.steps)):
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                return fail("Failure mining exceeded hard runtime cap.", 7)
            sample = train_dataset[train_indices[step % len(train_indices)]]
            batch = _add_training_batch_dims(preprocessor(sample))
            if not all(value.startswith("cuda") for value in _tensor_devices(batch).values()):
                return fail("CPU fallback detected in training batch.", 8)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_from_output(policy.forward(batch))
            loss_value = _to_float(loss)
            if not math.isfinite(loss_value):
                return fail(f"Non-finite training loss at step {step}: {loss_value}", 9)
            loss.backward()
            grad_summary = _gradient_summary(policy)
            if grad_summary["nonzero_grad_tensors"] == 0:
                return fail(f"No nonzero gradients at step {step}.", 10)
            optimizer.step()
            loss_curve.append({"step": step, "loss": round(loss_value, 9), **_cuda_memory(torch)})
            grad_curve.append({"step": step, **grad_summary})
        training_elapsed = time.monotonic() - training_started

        lora_rows = evaluate_model_rows(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            action_min=action_min,
            action_max=action_max,
        )
        train_gap_samples = [
            {
                "dataset_local_index": int(frame),
                "episode_index": 0,
                "frame_index": int(frame),
                "episode_length": len(train_dataset),
                "task_index": 0,
                "task": _task_text_map(tasks_df).get(0, "task_0"),
                "phase": _phase(int(frame), len(train_dataset)),
            }
            for frame in np.linspace(0, len(train_dataset) - 1, num=int(args.train_gap_samples), dtype=int)
        ]
        train_lora_rows = evaluate_model_rows(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=train_dataset,
            samples=train_gap_samples,
            action_min=action_min,
            action_max=action_max,
        )

        aggregate = {
            "frozen_base": summarize_rows(base_rows),
            "rank4_lora": summarize_rows(lora_rows),
            "mean_action_prior": summarize_rows(mean_rows),
            "rank4_lora_train_episode": summarize_rows(train_lora_rows),
        }
        comparison = compare_variants(base_rows, lora_rows, mean_rows)
        task_breakdown = {
            "frozen_base": _group_summary(base_rows, "task_index"),
            "rank4_lora": _group_summary(lora_rows, "task_index"),
            "mean_action_prior": _group_summary(mean_rows, "task_index"),
        }
        phase_breakdown = {
            "frozen_base": _group_summary(base_rows, "phase"),
            "rank4_lora": _group_summary(lora_rows, "phase"),
            "mean_action_prior": _group_summary(mean_rows, "phase"),
        }

        interim = {"aggregate_metrics": aggregate, "comparison": comparison}
        final_decision = choose_method_readiness_decision(interim)
        gap_analysis = build_gap_analysis(final_decision, aggregate, comparison)
        report.update(
            {
                "status": "completed",
                "final_decision": final_decision,
                "metric_reconciliation": metric_reconciliation,
                "dataset_audit": {
                    "total_episodes": int(info.get("total_episodes", 0)),
                    "total_frames": int(info.get("total_frames", 0)),
                    "total_tasks": int(info.get("total_tasks", 0)),
                    "official_splits": info.get("splits") or {},
                    "selected_task_count": len(eval_plan["selected_tasks"]),
                    "selected_tasks": eval_plan["selected_tasks"],
                    "selected_episode_count": len(eval_plan["selected_episodes"]),
                    "selected_episodes": eval_plan["selected_episodes"],
                    "heldout_sample_count": eval_plan["sample_count"],
                    "train_episode": 0,
                    "processor_probe_shapes": _tensor_shapes(probe),
                    "processor_probe_devices": input_devices,
                    "model_parameter_device": param_summary["first_parameter_device"],
                    "task_examples": [item["task"] for item in eval_plan["selected_tasks"][:5]],
                },
                "training": {
                    "variant": "standard_rank4_lora",
                    "steps": int(args.steps),
                    "batch_size": 1,
                    "trainable_params": lora_param_summary["trainable_params"],
                    "total_params": lora_param_summary["total_params"],
                    "loss_before": loss_curve[0]["loss"],
                    "loss_after": loss_curve[-1]["loss"],
                    "loss_decrease_fraction": round((loss_curve[0]["loss"] - loss_curve[-1]["loss"]) / max(abs(loss_curve[0]["loss"]), 1e-12), 9),
                    "loss_curve": loss_curve,
                    "last_grad_norm": grad_curve[-1]["grad_norm"],
                    "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"],
                    "training_elapsed_sec": round(training_elapsed, 3),
                    "steps_per_sec": round(len(loss_curve) / training_elapsed, 6),
                    "autocast_initial_final": _safe_autocast_status(torch),
                },
                "aggregate_metrics": aggregate,
                "task_breakdown": task_breakdown,
                "phase_breakdown": phase_breakdown,
                "comparison": comparison,
                "gap_analysis": gap_analysis,
                "method_shortlist": build_method_shortlist(final_decision),
                "runtime": {
                    "total_elapsed_sec": round(time.monotonic() - started, 3),
                    "rss_final_mb": _rss_mb(),
                    "cuda": {"available": True, "device_name": torch.cuda.get_device_name(0), **_cuda_memory(torch)},
                },
            }
        )
        report["exact_next_prompt"] = exact_next_prompt(final_decision)
        return report, 0
    except Exception as exc:
        report["status"] = "failed"
        report["final_decision"] = "NEED_LONGER_OFFICIAL_BASELINE_REPRO"
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 31


def build_gap_analysis(final_decision: str, aggregate: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    base = aggregate.get("frozen_base") or {}
    lora = aggregate.get("rank4_lora") or {}
    mean_prior = aggregate.get("mean_action_prior") or {}
    action_gain = None
    loss_delta = None
    if base.get("action_l2_mean") is not None and lora.get("action_l2_mean") is not None:
        action_gain = round(float(base["action_l2_mean"]) - float(lora["action_l2_mean"]), 9)
    if base.get("eval_loss_mean") is not None and lora.get("eval_loss_mean") is not None:
        loss_delta = round(float(lora["eval_loss_mean"]) - float(base["eval_loss_mean"]), 9)
    action_gain_text = "improves one-step action L2" if (action_gain or 0) > 0 else "does not improve aggregate one-step action L2"
    eval_loss_text = "worsens normalized chunk eval loss" if (loss_delta or 0) > 0 else "does not worsen aggregate normalized chunk eval loss"
    candidates = {
        "gripper_contact_phase": {
            "subgroup_size": lora.get("sample_count"),
            "baseline_failure_severity": lora.get("gripper_abs_mean"),
            "standard_lora_solves": (lora.get("gripper_sign_accuracy") or 0) >= 0.95,
            "mean_prior_explains": False,
            "method_worthy": final_decision == "GO_METHOD_DESIGN_GRIPPER_PHASE",
            "rejection_reason": "gripper sign accuracy is high and gripper absolute error is small" if final_decision != "GO_METHOD_DESIGN_GRIPPER_PHASE" else None,
        },
        "control_stability_retention": {
            "subgroup_size": lora.get("sample_count"),
            "baseline_failure_severity": {"action_l2_gain": action_gain, "eval_loss_delta": loss_delta},
            "standard_lora_helps_or_hurts": f"rank-4 LoRA {action_gain_text} and {eval_loss_text}",
            "mean_prior_explains": (mean_prior.get("action_l2_mean") or 999) <= min(base.get("action_l2_mean") or 999, lora.get("action_l2_mean") or 999),
            "method_worthy": final_decision == "GO_METHOD_DESIGN_CONTROL_STABILITY",
            "recent_paper_risk": "RTC/AAC already cover inference-time chunk consistency, so novelty must focus on low-data LoRA retention/stability in the official SmolVLA flow objective.",
        },
        "task_adapter_interference": {
            "subgroup_size": lora.get("sample_count"),
            "baseline_failure_severity": {
                "lora_help_count": comparison.get("lora_help_count"),
                "lora_hurt_count": comparison.get("lora_hurt_count"),
                "task_mean_help_count": comparison.get("task_mean_help_count"),
                "task_mean_hurt_count": comparison.get("task_mean_hurt_count"),
            },
            "standard_lora_helps_or_hurts": "mixed by task/frame",
            "method_worthy": final_decision == "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING",
            "recent_paper_risk": "MoIRA is a close task/instruction-routing baseline with low-rank adapters, including LIBERO-like evaluations.",
        },
    }
    strongest = {
        "GO_METHOD_DESIGN_GRIPPER_PHASE": "gripper_contact_phase",
        "GO_METHOD_DESIGN_CONTROL_STABILITY": "control_stability_retention",
        "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING": "task_adapter_interference",
    }.get(final_decision)
    return {
        "candidate_gaps": candidates,
        "strongest_method_worthy_gap": strongest,
        "recommended_method_direction": {
            "GO_METHOD_DESIGN_GRIPPER_PHASE": "Phase/Gripper-Aware Action Adapter",
            "GO_METHOD_DESIGN_CONTROL_STABILITY": "Control-Stable LoRA Adapter",
            "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING": "Task-Conditional Adapter Routing",
        }.get(final_decision),
        "estimated_kill_risk": {
            "GO_METHOD_DESIGN_GRIPPER_PHASE": "medium",
            "GO_METHOD_DESIGN_CONTROL_STABILITY": "medium-high",
            "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING": "high",
        }.get(final_decision, "high"),
    }


def build_method_shortlist(final_decision: str) -> list[dict[str, Any]]:
    if final_decision not in {
        "GO_METHOD_DESIGN_GRIPPER_PHASE",
        "GO_METHOD_DESIGN_CONTROL_STABILITY",
        "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING",
    }:
        return []
    return [
        {
            "method_name": "Control-Stable LoRA Adapter",
            "precise_gap": "rank-4 LoRA improves postprocessed action L2 while worsening normalized action-chunk flow eval loss",
            "latest_paper_comparison": "RTC and AAC already address action chunk consistency at inference time; novelty must be a training/adapter retention objective, not generic chunk smoothing.",
            "method_novelty": "penalize LoRA drift in official flow/chunk objective while preserving one-step action gains",
            "required_baselines": ["frozen/base official SmolVLA", "standard rank-4 LoRA", "mean-action prior", "RTC/AAC-style inference-only ablation if rollout exists"],
            "expected_improvement_axis": "retain or improve action L2 without increasing normalized chunk eval loss",
            "expected_kill_risk": "medium-high",
            "first_experiment": "add a planning-only method spec with predeclared retention penalty and rerun the same failure-mining subset",
            "ral_stability_estimate": "medium if official rollout becomes available; weak without rollout",
        },
        {
            "method_name": "Task-Conditional Adapter Routing",
            "precise_gap": "LoRA helps some task groups and hurts others",
            "latest_paper_comparison": "MoIRA is a close architecture-agnostic routing baseline with low-rank adapters on LIBERO-like benchmarks",
            "method_novelty": "only plausible if routing is simpler, lower-compute, and specifically protects official SmolVLA retention",
            "required_baselines": ["standard LoRA", "task-specific LoRA", "MoIRA-style routing baseline"],
            "expected_improvement_axis": "reduce task-specific negative transfer",
            "expected_kill_risk": "high",
            "first_experiment": "planning-only comparison matrix; do not implement until routing novelty is sharpened",
            "ral_stability_estimate": "low-medium",
        },
        {
            "method_name": "Phase/Gripper-Aware Action Adapter",
            "precise_gap": "gripper/contact phase errors if a future official audit finds them",
            "latest_paper_comparison": "current failure mining must prove gripper phase severity first",
            "method_novelty": "not supported unless gripper/contact subgroup is large and unsolved by LoRA",
            "required_baselines": ["frozen/base", "standard LoRA", "mean gripper prior"],
            "expected_improvement_axis": "gripper error and contact-phase action quality",
            "expected_kill_risk": "high under current evidence",
            "first_experiment": "only run if gripper subgroup failure is observed",
            "ral_stability_estimate": "low under current evidence",
        },
    ]


def exact_next_prompt(final_decision: str) -> str | None:
    if final_decision == "GO_METHOD_DESIGN_CONTROL_STABILITY":
        return (
            "Design, but do not yet run, an official SmolVLA-LIBERO Control-Stable LoRA Adapter experiment. "
            "Use frozen/base, standard rank-4 LoRA, and mean-action prior as mandatory baselines; predeclare action L2, normalized chunk eval loss, task/phase breakdowns, tuning budget, and kill criteria."
        )
    if final_decision == "GO_METHOD_DESIGN_GRIPPER_PHASE":
        return "Design a Phase/Gripper-Aware official SmolVLA-LIBERO method plan with predeclared gripper/contact metrics and baseline anchors."
    if final_decision == "GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING":
        return "Design a task-conditional adapter-routing plan only after explicitly comparing against MoIRA-style routing and standard LoRA anchors."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=r"C:\assets\checkpoints\smolvla_libero")
    parser.add_argument("--dataset-root", default=r"C:\assets\datasets\lerobot_libero")
    parser.add_argument("--hf-home", default=r"C:\assets\hf_home")
    parser.add_argument("--vlm-root", default=r"C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--min-task-groups", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--train-gap-samples", type=int, default=20)
    parser.add_argument("--report-json", default="reports/official_smolvla_failure_mining_result.json")
    parser.add_argument("--report-md", default="reports/official_smolvla_failure_mining_result.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
