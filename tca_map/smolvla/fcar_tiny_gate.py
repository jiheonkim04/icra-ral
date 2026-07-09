"""FCAR tiny-gate experiment on official SmolVLA-LIBERO predictions.

This runner implements the first Frame-Conditional Adapter Retention diagnostic
only. It may regenerate the fixed rank-4 LoRA baseline when the per-frame
prediction artifact is missing, then trains a tiny CPU gate over frozen/base and
rank-4 LoRA action predictions. It does not train the SmolVLA backbone, run
rollouts, run OpenVLA-OFT, download assets, or use the archived custom
LIBERO_7D route.
"""

from __future__ import annotations

import argparse
import copy
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
    _to_float,
)
from tca_map.smolvla.official_libero_failure_mining import (
    DEFAULT_TRAIN_INDICES,
    MAX_TRAINING_STEPS,
    _metric_row,
    evaluate_mean_prior_rows,
    evaluate_model_rows,
    select_eval_plan,
    summarize_rows,
)
from tca_map.smolvla.official_libero_routing_design_gate import (
    action_dim_oracle_rows,
    frame_oracle_rows,
    task_oracle_rows,
)


FINAL_DECISIONS = {
    "GO_FCAR_SCALEUP",
    "WEAK_FCAR_SIGNAL_NEEDS_REPEAT",
    "FCAR_OVERFITS",
    "FCAR_KILLED_BY_STATIC_BASELINE",
    "NO_FCAR_GAIN_OVER_BASE",
    "FCAR_IMPLEMENTATION_BLOCKED",
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

DATE = "2026-07-10 KST"
ARTIFACT_VERSION = 1
MAX_RUNTIME_SECONDS = 45 * 60
MAX_GATE_EPOCHS = 100
GATE_PATIENCE = 15
STATIC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
FCAR_ABS_SUCCESS = 0.005
FCAR_REL_SUCCESS = 0.05


class FcarError(RuntimeError):
    """Raised for a bounded FCAR failure with a reportable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _round_float(value: Any, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_vector(values: Any, digits: int = 9) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [round(float(x), digits) for x in array.tolist()]


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["episode_index"]), int(record["frame_index"]), int(record["task_index"]))


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["episode_index"]), int(row["frame_index"]), int(row["task_index"]))


def _improvement(base_value: float, candidate_value: float) -> dict[str, float]:
    absolute = float(base_value) - float(candidate_value)
    relative = absolute / max(abs(float(base_value)), 1e-12)
    return {"absolute": round(absolute, 9), "relative": round(relative, 9)}


def _sample_state(raw_sample: dict[str, Any]) -> list[float]:
    for key in ("observation.state", "observation_state", "state"):
        if key in raw_sample:
            value = raw_sample[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return _round_vector(np.asarray(value, dtype=np.float32).reshape(-1), 9)
    return []


def _task_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        task = str(record["task_index"])
        counts[task] = counts.get(task, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _episode_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        episode = str(record["episode_index"])
        counts[episode] = counts.get(episode, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _split_counts(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "frame_count": len(records),
            "episode_distribution": _episode_distribution(records),
            "task_distribution": _task_distribution(records),
        }
        for name, records in split_records.items()
    }


def split_records_by_episode(
    records: list[dict[str, Any]],
    selected_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic 60/20/20-ish split using whole episodes where possible."""

    episode_order: list[int] = []
    if selected_tasks:
        for task in sorted(selected_tasks, key=lambda item: int(item["task_index"])):
            for episode in sorted(int(ep) for ep in task.get("episodes", [])):
                if episode not in episode_order:
                    episode_order.append(episode)
    if not episode_order:
        episode_order = sorted({int(record["episode_index"]) for record in records})

    task_first_episodes: list[int] = []
    remaining: list[int] = []
    if selected_tasks:
        for task in sorted(selected_tasks, key=lambda item: int(item["task_index"])):
            episodes = sorted(int(ep) for ep in task.get("episodes", []))
            if episodes:
                task_first_episodes.append(episodes[0])
                remaining.extend(episodes[1:])
    else:
        remaining = list(episode_order)

    target_train_episodes = max(1, int(round(0.60 * len(episode_order))))
    target_val_episodes = max(1, int(round(0.20 * len(episode_order))))
    train_episodes: list[int] = []
    for episode in task_first_episodes:
        if episode in episode_order and episode not in train_episodes:
            train_episodes.append(episode)
    for episode in remaining:
        if len(train_episodes) >= target_train_episodes:
            break
        if episode not in train_episodes:
            train_episodes.append(episode)

    leftover = [episode for episode in episode_order if episode not in train_episodes]
    val_episodes = leftover[:target_val_episodes]
    test_episodes = leftover[target_val_episodes:]
    if not test_episodes and val_episodes:
        test_episodes = [val_episodes.pop()]

    split_by_episode = {
        "train": set(train_episodes),
        "val": set(val_episodes),
        "test": set(test_episodes),
    }
    split_records = {"train": [], "val": [], "test": []}
    for record in records:
        episode = int(record["episode_index"])
        assigned = "test"
        if episode in split_by_episode["train"]:
            assigned = "train"
        elif episode in split_by_episode["val"]:
            assigned = "val"
        split_record = dict(record)
        split_record["split"] = assigned
        split_records[assigned].append(split_record)
    return split_records


def feature_schema() -> list[str]:
    names: list[str] = []
    names.extend([f"base_action_{idx}" for idx in range(7)])
    names.extend([f"lora_action_{idx}" for idx in range(7)])
    names.extend([f"action_delta_{idx}" for idx in range(7)])
    names.extend([f"abs_action_delta_{idx}" for idx in range(7)])
    names.extend(["base_action_norm", "lora_action_norm", "action_delta_norm"])
    names.extend(["base_gripper", "lora_gripper", "gripper_delta", "normalized_phase"])
    names.extend([f"state_{idx}" for idx in range(8)])
    return names


def _feature_vector(record: dict[str, Any]) -> list[float]:
    base = np.asarray(record["base_action"], dtype=np.float32).reshape(-1)[:7]
    lora = np.asarray(record["lora_action"], dtype=np.float32).reshape(-1)[:7]
    target_dim = 7
    if base.shape[0] < target_dim:
        base = np.pad(base, (0, target_dim - base.shape[0]))
    if lora.shape[0] < target_dim:
        lora = np.pad(lora, (0, target_dim - lora.shape[0]))
    delta = lora - base
    state = np.asarray(record.get("state") or [], dtype=np.float32).reshape(-1)[:8]
    if state.shape[0] < 8:
        state = np.pad(state, (0, 8 - state.shape[0]))
    extras = np.asarray(
        [
            np.linalg.norm(base),
            np.linalg.norm(lora),
            np.linalg.norm(delta),
            base[6],
            lora[6],
            delta[6],
            float(record.get("normalized_phase") or 0.0),
        ],
        dtype=np.float32,
    )
    return _round_vector(np.concatenate([base, lora, delta, np.abs(delta), extras, state]), 9)


def build_feature_matrix(
    records: list[dict[str, Any]],
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray([_feature_vector(record) for record in records], dtype=np.float32)
    if matrix.size == 0:
        matrix = np.zeros((0, len(feature_schema())), dtype=np.float32)
    if mean is None:
        mean = matrix.mean(axis=0) if len(matrix) else np.zeros((matrix.shape[1],), dtype=np.float32)
    if std is None:
        std = matrix.std(axis=0) if len(matrix) else np.ones((matrix.shape[1],), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((matrix - mean) / std).astype(np.float32), mean.astype(np.float32), std


def _records_to_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "base": np.asarray([record["base_action"] for record in records], dtype=np.float32),
        "lora": np.asarray([record["lora_action"] for record in records], dtype=np.float32),
        "target": np.asarray([record["target_action"] for record in records], dtype=np.float32),
        "oracle_help": np.asarray(
            [1.0 if float(record["lora_action_l2"]) < float(record["base_action_l2"]) else 0.0 for record in records],
            dtype=np.float32,
        ),
    }


def _sample_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_local_index": int(record["dataset_local_index"]),
        "episode_index": int(record["episode_index"]),
        "frame_index": int(record["frame_index"]),
        "episode_length": int(record["episode_length"]),
        "task_index": int(record["task_index"]),
        "task": str(record["task"]),
        "phase": str(record["phase"]),
        "split": str(record.get("split", "unknown")),
    }


def _rows_from_records(
    records: list[dict[str, Any]],
    *,
    pred_key: str,
    eval_loss_key: str | None,
    action_min: np.ndarray,
    action_max: np.ndarray,
    selected_expert: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(record[pred_key], dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=float(record[eval_loss_key]) if eval_loss_key and record.get(eval_loss_key) is not None else None,
            action_min=action_min,
            action_max=action_max,
        )
        if selected_expert is not None:
            row["selected_expert"] = selected_expert
        rows.append(row)
    return rows


def _rows_for_prediction(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    selected_expert: str,
) -> list[dict[str, Any]]:
    rows = []
    for record, pred in zip(records, predictions):
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(pred, dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=None,
            action_min=action_min,
            action_max=action_max,
        )
        row["selected_expert"] = selected_expert
        rows.append(row)
    return rows


def _per_task(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_index"])].append(row)
    return {task: summarize_rows(group_rows) for task, group_rows in sorted(grouped.items(), key=lambda item: int(item[0]))}


def _help_hurt_counts(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, int]:
    reference = {_row_key(row): row for row in reference_rows}
    help_count = 0
    hurt_count = 0
    tie_count = 0
    for row in candidate_rows:
        delta = float(reference[_row_key(row)]["action_l2"]) - float(row["action_l2"])
        if delta > 1e-9:
            help_count += 1
        elif delta < -1e-9:
            hurt_count += 1
        else:
            tie_count += 1
    return {"help": help_count, "hurt": hurt_count, "tie": tie_count}


def _metric_package(
    rows: list[dict[str, Any]],
    *,
    base_rows: list[dict[str, Any]] | None = None,
    lora_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    package = summarize_rows(rows)
    package["per_task"] = _per_task(rows)
    if base_rows is not None:
        package["help_hurt_vs_frozen_base"] = _help_hurt_counts(rows, base_rows)
    if lora_rows is not None:
        package["help_hurt_vs_rank4_lora"] = _help_hurt_counts(rows, lora_rows)
    return package


def _task_router_from_training(
    train_base_rows: list[dict[str, Any]],
    train_lora_rows: list[dict[str, Any]],
) -> dict[str, str]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"base": [], "lora": []})
    lora_by_key = {_row_key(row): row for row in train_lora_rows}
    for base in train_base_rows:
        task = str(base["task_index"])
        values[task]["base"].append(float(base["action_l2"]))
        values[task]["lora"].append(float(lora_by_key[_row_key(base)]["action_l2"]))
    routing = {}
    for task, task_values in values.items():
        route = "rank4_lora" if float(np.mean(task_values["lora"])) < float(np.mean(task_values["base"])) else "frozen_base"
        routing[task] = route
    return dict(sorted(routing.items(), key=lambda item: int(item[0])))


def _apply_task_router(
    base_rows: list[dict[str, Any]],
    lora_rows: list[dict[str, Any]],
    routing: dict[str, str],
) -> list[dict[str, Any]]:
    lora_by_key = {_row_key(row): row for row in lora_rows}
    rows = []
    for base in base_rows:
        task = str(base["task_index"])
        use_lora = routing.get(task, "frozen_base") == "rank4_lora"
        chosen = copy.deepcopy(lora_by_key[_row_key(base)] if use_lora else base)
        chosen["selected_expert"] = "rank4_lora" if use_lora else "frozen_base"
        rows.append(chosen)
    return rows


def _static_rows(
    records: list[dict[str, Any]],
    weight: float,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    base = np.asarray([record["base_action"] for record in records], dtype=np.float32)
    lora = np.asarray([record["lora_action"] for record in records], dtype=np.float32)
    pred = float(weight) * lora + (1.0 - float(weight)) * base
    return _rows_for_prediction(records, pred, action_min=action_min, action_max=action_max, selected_expert=f"static_w_{weight}")


def _choose_static_weight(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[float, str, dict[str, Any]]:
    selection_split = "val" if split_records.get("val") else "train"
    grid: dict[str, Any] = {}
    best_weight = STATIC_GRID[0]
    best_l2 = math.inf
    for weight in STATIC_GRID:
        split_metrics = {}
        for split, records in split_records.items():
            rows = _static_rows(records, weight, action_min=action_min, action_max=action_max)
            split_metrics[split] = summarize_rows(rows)
        grid[str(weight)] = split_metrics
        candidate_l2 = float(split_metrics[selection_split]["action_l2_mean"])
        if candidate_l2 < best_l2:
            best_l2 = candidate_l2
            best_weight = weight
    return best_weight, selection_split, grid


def _calibration(records: list[dict[str, Any]], alphas: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(
        [1 if float(record["lora_action_l2"]) < float(record["base_action_l2"]) else 0 for record in records],
        dtype=np.int64,
    )
    predicted = (alphas >= 0.5).astype(np.int64)
    bins = []
    edges = np.linspace(0.0, 1.0, 6)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (alphas >= left) & (alphas < right if right < 1.0 else alphas <= right)
        count = int(mask.sum())
        bins.append(
            {
                "range": [round(float(left), 3), round(float(right), 3)],
                "count": count,
                "alpha_mean": _round_float(float(alphas[mask].mean())) if count else None,
                "oracle_lora_help_rate": _round_float(float(labels[mask].mean())) if count else None,
            }
        )
    return {
        "bins": bins,
        "confusion_at_alpha_0_5": {
            "true_positive": int(((predicted == 1) & (labels == 1)).sum()),
            "false_positive": int(((predicted == 1) & (labels == 0)).sum()),
            "true_negative": int(((predicted == 0) & (labels == 0)).sum()),
            "false_negative": int(((predicted == 0) & (labels == 1)).sum()),
        },
    }


def _alpha_stats(alphas: np.ndarray) -> dict[str, Any]:
    if len(alphas) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "fraction_routed_to_lora_alpha_ge_0_5": None,
            "fraction_routed_to_base_alpha_lt_0_5": None,
        }
    return {
        "mean": _round_float(float(alphas.mean())),
        "std": _round_float(float(alphas.std())),
        "min": _round_float(float(alphas.min())),
        "max": _round_float(float(alphas.max())),
        "fraction_routed_to_lora_alpha_ge_0_5": _round_float(float((alphas >= 0.5).mean())),
        "fraction_routed_to_base_alpha_lt_0_5": _round_float(float((alphas < 0.5).mean())),
    }


def _failure_cases(
    candidate_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    base_by_key = {_row_key(row): row for row in base_rows}
    merged = []
    for row in candidate_rows:
        base = base_by_key[_row_key(row)]
        gain = float(base["action_l2"]) - float(row["action_l2"])
        merged.append(
            {
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "task_index": int(row["task_index"]),
                "task": row["task"],
                "phase": row["phase"],
                "fcar_action_l2": row["action_l2"],
                "base_action_l2": base["action_l2"],
                "base_minus_fcar_gain": round(gain, 9),
            }
        )
    return {
        "top_helps_vs_base": sorted(merged, key=lambda item: item["base_minus_fcar_gain"], reverse=True)[:limit],
        "top_hurts_vs_base": sorted(merged, key=lambda item: item["base_minus_fcar_gain"])[:limit],
    }


def _make_record(
    *,
    base: dict[str, Any],
    lora: dict[str, Any],
    mean: dict[str, Any],
    raw_sample: dict[str, Any],
) -> dict[str, Any]:
    episode_length = int(base["episode_length"])
    normalized_phase = float(base["frame_index"]) / max(1, episode_length - 1)
    return {
        "sample_key": {
            "episode_index": int(base["episode_index"]),
            "frame_index": int(base["frame_index"]),
            "task_index": int(base["task_index"]),
        },
        "dataset_local_index": int(base["dataset_local_index"]),
        "episode_index": int(base["episode_index"]),
        "frame_index": int(base["frame_index"]),
        "episode_length": episode_length,
        "task_index": int(base["task_index"]),
        "task": str(base["task"]),
        "phase": str(base["phase"]),
        "normalized_phase": round(normalized_phase, 9),
        "state": _sample_state(raw_sample),
        "base_action": _round_vector(base["pred_preview"], 9),
        "lora_action": _round_vector(lora["pred_preview"], 9),
        "mean_action": _round_vector(mean["pred_preview"], 9),
        "target_action": _round_vector(base["target_preview"], 9),
        "base_eval_loss": base.get("eval_loss"),
        "lora_eval_loss": lora.get("eval_loss"),
        "base_action_l2": base.get("action_l2"),
        "lora_action_l2": lora.get("action_l2"),
        "oracle_help_label": int(float(lora["action_l2"]) < float(base["action_l2"])),
        "allowed_gate_features": {
            "base_action": True,
            "lora_action": True,
            "base_vs_lora_action_disagreement": True,
            "action_norm": True,
            "gripper_value": True,
            "current_8d_state": True,
            "normalized_episode_phase": True,
            "instruction_embedding": False,
            "ground_truth_action": False,
            "oracle_help_label_at_inference": False,
        },
    }


def _load_prediction_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = _read_json(path)
    if int(artifact.get("artifact_version") or 0) != ARTIFACT_VERSION:
        raise FcarError("ARTIFACT_SCHEMA_MISMATCH", f"Unsupported prediction artifact version: {artifact.get('artifact_version')}")
    records = artifact.get("records") or []
    if not records:
        raise FcarError("ARTIFACT_EMPTY", f"Prediction artifact contains no records: {path}")
    return artifact


def _generate_prediction_artifact(args: argparse.Namespace, path: Path, started: float) -> dict[str, Any]:
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        raise FcarError("FORBIDDEN_GATE_SET", "Forbidden gate(s) set: " + ", ".join(forbidden))
    if not _env_flag("ALLOW_HEAVY_IMPORT") or not _env_flag("ALLOW_GPU_TRAINING"):
        raise FcarError("HEAVY_GATE_REQUIRED", "Prediction regeneration requires ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1.")
    if int(args.steps) > MAX_TRAINING_STEPS:
        raise FcarError("LORA_STEPS_TOO_HIGH", f"Rank-4 LoRA regeneration steps exceed cap: {args.steps}")

    import pandas as pd
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise FcarError("CUDA_UNAVAILABLE", "CUDA unavailable; refusing long CPU LoRA regeneration.")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()

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
    cfg.device = "cuda"
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
    policy.to("cuda")
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    probe = _add_training_batch_dims(preprocessor(train_dataset[0]))
    input_devices = _tensor_devices(probe)
    param_summary = _parameter_summary(policy)
    if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
        value.startswith("cuda") for value in input_devices.values()
    ):
        raise FcarError("CPU_FALLBACK_BUG", f"CUDA available but params/inputs are not all CUDA: params={param_summary}, inputs={input_devices}")

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
    policy.to("cuda")
    policy.train()
    lora_param_summary = _parameter_summary(policy)
    optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))
    train_indices = [idx for idx in DEFAULT_TRAIN_INDICES if idx < len(train_dataset)]
    loss_curve = []
    grad_curve = []
    training_started = time.monotonic()
    for step in range(int(args.steps)):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise FcarError("RUNTIME_CAP_EXCEEDED", "FCAR prediction regeneration exceeded 45 minute cap.")
        sample = train_dataset[train_indices[step % len(train_indices)]]
        batch = _add_training_batch_dims(preprocessor(sample))
        if not all(value.startswith("cuda") for value in _tensor_devices(batch).values()):
            raise FcarError("CPU_FALLBACK_BUG", "CUDA available but rank-4 LoRA training batch is on CPU.")
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_output(policy.forward(batch))
        loss_value = _to_float(loss)
        if not math.isfinite(loss_value):
            raise FcarError("NONFINITE_LORA_LOSS", f"Non-finite rank-4 LoRA loss at step {step}: {loss_value}")
        loss.backward()
        grad_summary = _gradient_summary(policy)
        if grad_summary["nonzero_grad_tensors"] == 0:
            raise FcarError("NO_LORA_GRADIENT", f"No nonzero LoRA gradients at step {step}.")
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

    base_by_key = {_row_key(row): row for row in base_rows}
    lora_by_key = {_row_key(row): row for row in lora_rows}
    mean_by_key = {_row_key(row): row for row in mean_rows}
    records = []
    for sample in eval_plan["samples"]:
        raw_sample = eval_dataset[int(sample["dataset_local_index"])]
        key = (int(sample["episode_index"]), int(sample["frame_index"]), int(sample["task_index"]))
        records.append(_make_record(base=base_by_key[key], lora=lora_by_key[key], mean=mean_by_key[key], raw_sample=raw_sample))

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "date": DATE,
        "source": "regenerated_official_smolvla_lora_rank4",
        "policy": {
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "old_custom_route_used": False,
            "official_model_dataset_used": True,
            "rank4_lora_regenerated": True,
            "smolvla_backbone_trained": False,
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_root),
            "hf_home": str(hf_home),
            "vlm_root": str(vlm_root),
        },
        "dataset": {
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "total_tasks": int(info.get("total_tasks", 0)),
            "selected_task_count": len(eval_plan["selected_tasks"]),
            "selected_tasks": eval_plan["selected_tasks"],
            "selected_episode_count": len(eval_plan["selected_episodes"]),
            "selected_episodes": eval_plan["selected_episodes"],
            "heldout_sample_count": eval_plan["sample_count"],
            "train_episode_for_lora": 0,
            "task_examples": [str(index) for index, _row in tasks_df.head(5).iterrows()],
        },
        "action_range": {"min": _round_vector(action_min, 9), "max": _round_vector(action_max, 9)},
        "device_audit": {
            "cuda_available": True,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "model_parameter_device": param_summary["first_parameter_device"],
            "model_parameter_dtype": param_summary["first_parameter_dtype"],
            "input_tensor_devices": input_devices,
            "autocast_status_initial_final": _safe_autocast_status(torch),
            "cuda_memory": _cuda_memory(torch),
        },
        "rank4_lora_regeneration": {
            "steps": int(args.steps),
            "batch_size": 1,
            "trainable_params": lora_param_summary["trainable_params"],
            "total_params": lora_param_summary["total_params"],
            "loss_before": loss_curve[0]["loss"] if loss_curve else None,
            "loss_after": loss_curve[-1]["loss"] if loss_curve else None,
            "loss_curve": loss_curve,
            "last_grad_norm": grad_curve[-1]["grad_norm"] if grad_curve else None,
            "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"] if grad_curve else None,
            "training_elapsed_sec": round(training_elapsed, 3),
            "steps_per_sec": round(len(loss_curve) / max(training_elapsed, 1e-12), 6),
            "autocast_status": _safe_autocast_status(torch),
        },
        "source_metric_check": {
            "frozen_base": summarize_rows(base_rows),
            "rank4_lora": summarize_rows(lora_rows),
            "mean_action_prior": summarize_rows(mean_rows),
        },
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return artifact


def _load_or_generate_prediction_artifact(args: argparse.Namespace, started: float) -> tuple[dict[str, Any], str]:
    artifact_path = Path(args.prediction_artifact)
    artifact = _load_prediction_artifact(artifact_path)
    if artifact is not None:
        return artifact, "loaded"
    return _generate_prediction_artifact(args, artifact_path, started), "regenerated"


def _train_tiny_gate(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_records = split_records["train"]
    val_records = split_records["val"]
    train_features, mean, std = build_feature_matrix(train_records)
    val_features, _, _ = build_feature_matrix(val_records, mean=mean, std=std)
    train_arrays = _records_to_arrays(train_records)
    val_arrays = _records_to_arrays(val_records)

    class TinyGate(torch.nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x).squeeze(-1)

    device = torch.device("cpu")
    model = TinyGate(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    cfg = {
        "model": "tiny_mlp",
        "hidden_sizes": [64, 32],
        "activation": "ReLU",
        "seed": int(seed),
        "max_epochs": MAX_GATE_EPOCHS,
        "early_stop_patience": GATE_PATIENCE,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "lambda_oracle_bce": 0.01,
        "lambda_retention": 0.01,
        "lambda_sparse": 0.001,
        "device": str(device),
        "inference_inputs_use_ground_truth": False,
        "inference_inputs_use_oracle_label": False,
    }

    def tensors(features: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "features": torch.as_tensor(features, dtype=torch.float32, device=device),
            "base": torch.as_tensor(arrays["base"], dtype=torch.float32, device=device),
            "lora": torch.as_tensor(arrays["lora"], dtype=torch.float32, device=device),
            "target": torch.as_tensor(arrays["target"], dtype=torch.float32, device=device),
            "oracle_help": torch.as_tensor(arrays["oracle_help"], dtype=torch.float32, device=device),
        }

    train_t = tensors(train_features, train_arrays)
    val_t = tensors(val_features, val_arrays)

    def eval_l2(batch: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            logits = model(batch["features"])
            alpha = torch.sigmoid(logits).unsqueeze(-1)
            mixed = alpha * batch["lora"] + (1.0 - alpha) * batch["base"]
            l2 = torch.linalg.norm(mixed - batch["target"], dim=1).mean()
        return float(l2.item()), alpha.squeeze(-1).cpu().numpy(), mixed.cpu().numpy()

    best_state = copy.deepcopy(model.state_dict())
    best_val_l2 = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []
    training_started = time.monotonic()
    for epoch in range(MAX_GATE_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_t["features"])
        alpha = torch.sigmoid(logits).unsqueeze(-1)
        mixed = alpha * train_t["lora"] + (1.0 - alpha) * train_t["base"]
        action_l2 = torch.linalg.norm(mixed - train_t["target"], dim=1).mean()
        oracle_bce = bce_loss(logits, train_t["oracle_help"])
        harm_label = 1.0 - train_t["oracle_help"]
        retention = (alpha.squeeze(-1) * harm_label).mean()
        sparse = torch.minimum(alpha, 1.0 - alpha).mean()
        loss = action_l2 + cfg["lambda_oracle_bce"] * oracle_bce + cfg["lambda_retention"] * retention + cfg["lambda_sparse"] * sparse
        loss.backward()
        optimizer.step()
        train_l2, train_alpha, _train_pred = eval_l2(train_t)
        val_l2, val_alpha, _val_pred = eval_l2(val_t)
        history.append(
            {
                "epoch": epoch,
                "loss": round(float(loss.item()), 9),
                "train_action_l2": round(train_l2, 9),
                "val_action_l2": round(val_l2, 9),
                "train_alpha_mean": round(float(train_alpha.mean()), 9),
                "val_alpha_mean": round(float(val_alpha.mean()), 9),
            }
        )
        if val_l2 < best_val_l2 - 1e-9:
            best_val_l2 = val_l2
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= GATE_PATIENCE:
                break
    model.load_state_dict(best_state)
    training_elapsed = time.monotonic() - training_started
    param_count = sum(int(param.numel()) for param in model.parameters())

    return {
        "model": model,
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": feature_schema(),
        "config": cfg,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_action_l2": round(best_val_l2, 9),
        "training_elapsed_sec": round(training_elapsed, 3),
        "parameter_count": int(param_count),
    }


def _predict_tiny_gate(gate: dict[str, Any], records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    features, _, _ = build_feature_matrix(records, mean=gate["feature_mean"], std=gate["feature_std"])
    arrays = _records_to_arrays(records)
    model = gate["model"]
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32)
        alpha = torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
    pred = alpha[:, None] * arrays["lora"] + (1.0 - alpha[:, None]) * arrays["base"]
    return alpha.astype(np.float32), pred.astype(np.float32)


def _evaluate_split(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    split_metrics: dict[str, Any] = {}
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    best_static_weight, static_selection_split, static_grid = _choose_static_weight(
        split_records,
        action_min=action_min,
        action_max=action_max,
    )

    train_base_rows = _rows_from_records(
        split_records["train"],
        pred_key="base_action",
        eval_loss_key="base_eval_loss",
        action_min=action_min,
        action_max=action_max,
        selected_expert="frozen_base",
    )
    train_lora_rows = _rows_from_records(
        split_records["train"],
        pred_key="lora_action",
        eval_loss_key="lora_eval_loss",
        action_min=action_min,
        action_max=action_max,
        selected_expert="rank4_lora",
    )
    moira_routing = _task_router_from_training(train_base_rows, train_lora_rows)

    alphas_by_split: dict[str, np.ndarray] = {}
    for split, records in split_records.items():
        base_rows = _rows_from_records(
            records,
            pred_key="base_action",
            eval_loss_key="base_eval_loss",
            action_min=action_min,
            action_max=action_max,
            selected_expert="frozen_base",
        )
        lora_rows = _rows_from_records(
            records,
            pred_key="lora_action",
            eval_loss_key="lora_eval_loss",
            action_min=action_min,
            action_max=action_max,
            selected_expert="rank4_lora",
        )
        mean_rows = _rows_from_records(
            records,
            pred_key="mean_action",
            eval_loss_key=None,
            action_min=action_min,
            action_max=action_max,
            selected_expert="mean_action_prior",
        )
        frame_rows = frame_oracle_rows(base_rows, lora_rows)
        task_rows, task_routing = task_oracle_rows(base_rows, lora_rows)
        moira_rows = _apply_task_router(base_rows, lora_rows, moira_routing)
        static_rows = _static_rows(records, best_static_weight, action_min=action_min, action_max=action_max)
        action_dim_rows = action_dim_oracle_rows(base_rows, lora_rows)
        alphas, fcar_pred = _predict_tiny_gate(gate, records)
        fcar_rows = _rows_for_prediction(records, fcar_pred, action_min=action_min, action_max=action_max, selected_expert="fcar_tiny_gate")
        alphas_by_split[split] = alphas
        split_rows[split] = {
            "frozen_base": base_rows,
            "rank4_lora": lora_rows,
            "mean_action_prior": mean_rows,
            "frame_oracle": frame_rows,
            "task_oracle": task_rows,
            "moira_style_instruction_task_router": moira_rows,
            "adapter_soup_static_merge": static_rows,
            "action_dim_oracle_diagnostic": action_dim_rows,
            "fcar_tiny_gate": fcar_rows,
        }
        split_metrics[split] = {
            name: _metric_package(rows, base_rows=base_rows, lora_rows=lora_rows)
            for name, rows in split_rows[split].items()
        }
        split_metrics[split]["task_oracle"]["task_routing"] = task_routing
        split_metrics[split]["moira_style_instruction_task_router"]["task_routing_from_train"] = moira_routing
        split_metrics[split]["adapter_soup_static_merge"]["selected_weight"] = best_static_weight
        split_metrics[split]["adapter_soup_static_merge"]["selection_split"] = static_selection_split
        split_metrics[split]["fcar_tiny_gate"]["alpha_stats"] = _alpha_stats(alphas)
        split_metrics[split]["fcar_tiny_gate"]["calibration"] = _calibration(records, alphas)

    return {
        "metrics": split_metrics,
        "rows": split_rows,
        "alphas": alphas_by_split,
        "static_grid": static_grid,
        "static_selected_weight": best_static_weight,
        "static_selection_split": static_selection_split,
        "moira_routing": moira_routing,
    }


def choose_final_decision(
    *,
    base_l2: float,
    lora_l2: float,
    mean_l2: float,
    moira_l2: float,
    static_l2: float,
    fcar_l2: float,
    train_l2: float,
    oracle_inputs_used: bool,
) -> str:
    if oracle_inputs_used:
        return "FCAR_IMPLEMENTATION_BLOCKED"
    if fcar_l2 >= base_l2:
        return "NO_FCAR_GAIN_OVER_BASE"
    train_eval_gap = fcar_l2 - train_l2
    if train_l2 < base_l2 and train_eval_gap > max(0.02, 0.50 * max(train_l2, 1e-12)):
        return "FCAR_OVERFITS"
    if static_l2 <= fcar_l2 + 1e-6 or moira_l2 <= fcar_l2 + 1e-6:
        return "FCAR_KILLED_BY_STATIC_BASELINE"
    absolute_gain = base_l2 - fcar_l2
    relative_gain = absolute_gain / max(abs(base_l2), 1e-12)
    clears_hard_gain = absolute_gain >= FCAR_ABS_SUCCESS or relative_gain >= FCAR_REL_SUCCESS
    beats_required = fcar_l2 < min(lora_l2, mean_l2, moira_l2, static_l2)
    if clears_hard_gain and beats_required:
        return "GO_FCAR_SCALEUP"
    return "WEAK_FCAR_SIGNAL_NEEDS_REPEAT"


def _anchor_reconciliation(report_metrics: dict[str, Any], routing_report_path: Path) -> dict[str, Any]:
    if not routing_report_path.exists():
        return {"available": False, "path": str(routing_report_path)}
    routing = _read_json(routing_report_path)
    anchors = routing.get("aggregate_metrics") or {}
    output = {"available": True, "path": str(routing_report_path), "deltas": {}}
    for name in ["frozen_base", "rank4_lora", "mean_action_prior", "frame_oracle", "task_oracle"]:
        current = ((report_metrics.get("all_200") or {}).get(name) or {}).get("action_l2_mean")
        anchor = (anchors.get(name) or {}).get("action_l2_mean")
        output["deltas"][name] = {
            "current": current,
            "anchor": anchor,
            "absolute_delta": _round_float(float(current) - float(anchor)) if current is not None and anchor is not None else None,
        }
    return output


def _all_200_metrics(records: list[dict[str, Any]], action_min: np.ndarray, action_max: np.ndarray) -> dict[str, Any]:
    base_rows = _rows_from_records(records, pred_key="base_action", eval_loss_key="base_eval_loss", action_min=action_min, action_max=action_max, selected_expert="frozen_base")
    lora_rows = _rows_from_records(records, pred_key="lora_action", eval_loss_key="lora_eval_loss", action_min=action_min, action_max=action_max, selected_expert="rank4_lora")
    mean_rows = _rows_from_records(records, pred_key="mean_action", eval_loss_key=None, action_min=action_min, action_max=action_max, selected_expert="mean_action_prior")
    frame_rows = frame_oracle_rows(base_rows, lora_rows)
    task_rows, task_routing = task_oracle_rows(base_rows, lora_rows)
    action_dim_rows = action_dim_oracle_rows(base_rows, lora_rows)
    metrics = {
        "frozen_base": _metric_package(base_rows, base_rows=base_rows, lora_rows=lora_rows),
        "rank4_lora": _metric_package(lora_rows, base_rows=base_rows, lora_rows=lora_rows),
        "mean_action_prior": _metric_package(mean_rows, base_rows=base_rows, lora_rows=lora_rows),
        "frame_oracle": _metric_package(frame_rows, base_rows=base_rows, lora_rows=lora_rows),
        "task_oracle": _metric_package(task_rows, base_rows=base_rows, lora_rows=lora_rows),
        "action_dim_oracle_diagnostic": _metric_package(action_dim_rows, base_rows=base_rows, lora_rows=lora_rows),
    }
    metrics["task_oracle"]["task_routing"] = task_routing
    return metrics


def _strip_gate_model(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": gate["config"],
        "history": gate["history"],
        "best_epoch": gate["best_epoch"],
        "best_val_action_l2": gate["best_val_action_l2"],
        "training_elapsed_sec": gate["training_elapsed_sec"],
        "parameter_count": gate["parameter_count"],
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    test = (report.get("metrics") or {}).get("test") or {}
    fcar = test.get("fcar_tiny_gate") or {}
    base = test.get("frozen_base") or {}
    lora = test.get("rank4_lora") or {}
    frame = test.get("frame_oracle") or {}
    task = test.get("task_oracle") or {}
    moira = test.get("moira_style_instruction_task_router") or {}
    static = test.get("adapter_soup_static_merge") or {}
    mean = test.get("mean_action_prior") or {}
    gain = ((report.get("kill_criteria") or {}).get("fcar_gain_over_frozen_base") or {})
    alpha = fcar.get("alpha_stats") or {}
    lines = [
        "# FCAR Tiny Gate Result",
        "",
        f"Date: {report.get('date')}",
        "",
        f"- final decision: `{report.get('final_decision')}`",
        f"- status: `{report.get('status')}`",
        f"- experiments happened: `{report.get('policy', {}).get('experiments_performed')}`",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- trained components: `{report.get('policy', {}).get('trained_components')}`",
        f"- GPU/download/OpenVLA-OFT happened: `{report.get('policy', {}).get('gpu_used')}` / `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}`",
        f"- official model/dataset used: `{report.get('policy', {}).get('official_model_dataset_used')}`",
        f"- old custom route used: `{report.get('policy', {}).get('old_custom_route_used')}`",
        "",
        "## Split",
        "",
        f"- prediction artifact: `{report.get('artifact', {}).get('source')}`",
        f"- seed: `{report.get('split', {}).get('seed')}`",
        f"- counts: `{report.get('split', {}).get('counts')}`",
        f"- leakage checks: `{report.get('split', {}).get('leakage_checks')}`",
        "",
        "## Test Metrics",
        "",
        "| variant | action L2 | translation L2 | rotation L2 | gripper abs | gripper sign |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in [
        ("frozen_base", base),
        ("rank4_lora", lora),
        ("mean_action_prior", mean),
        ("frame_oracle", frame),
        ("task_oracle", task),
        ("moira_style_instruction_task_router", moira),
        ("adapter_soup_static_merge", static),
        ("fcar_tiny_gate", fcar),
    ]:
        lines.append(
            f"| {name} | {item.get('action_l2_mean')} | {item.get('translation_l2_mean')} | "
            f"{item.get('rotation_l2_mean')} | {item.get('gripper_abs_mean')} | {item.get('gripper_sign_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## FCAR",
            "",
            f"- gain over frozen/base: `{gain}`",
            f"- recovered fraction of frame-oracle headroom: `{(report.get('oracle_recovery') or {}).get('fraction_of_frame_oracle_headroom')}`",
            f"- alpha stats: `{alpha}`",
            f"- train/eval gap: `{(report.get('train_eval_gap') or {})}`",
            f"- static selected weight: `{static.get('selected_weight')}`",
            f"- MoIRA-style routing: `{moira.get('task_routing_from_train')}`",
            "",
            "## Failure Cases",
            "",
            f"- top helps vs base: `{(report.get('failure_cases') or {}).get('top_helps_vs_base')}`",
            f"- top hurts vs base: `{(report.get('failure_cases') or {}).get('top_hurts_vs_base')}`",
            "",
            f"Exact next prompt: {report.get('exact_next_prompt')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FCAR Tiny Gate Decision",
        "",
        f"Date: {report.get('date')}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"Reason: {report.get('decision_reason')}",
        "",
        f"Exact next prompt: {report.get('exact_next_prompt')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "date": DATE,
        "status": "started",
        "policy": {
            "experiments_performed": True,
            "training_performed": False,
            "trained_components": [],
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "official_model_dataset_used": True,
            "old_custom_route_used": False,
            "paper_claims_made": False,
            "smolvla_backbone_trained": False,
            "fcar_method_implemented": True,
            "gpu_used": False,
            "cpu_fallback_bug": False,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "hf_home": str(Path(args.hf_home)),
            "vlm_root": str(Path(args.vlm_root)),
            "prediction_artifact": str(Path(args.prediction_artifact)),
        },
        "errors": [],
    }

    def fail(code: str, message: str, exit_code: int) -> tuple[dict[str, Any], int]:
        report["status"] = "failed"
        report["final_decision"] = "FCAR_IMPLEMENTATION_BLOCKED"
        report["decision_reason"] = message
        report["errors"].append({"code": code, "message": message})
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        if code == "CPU_FALLBACK_BUG":
            report["policy"]["cpu_fallback_bug"] = True
        return report, exit_code

    try:
        if int(args.max_eval_samples) > 1000:
            return fail("TOO_MANY_FRAMES", f"max_eval_samples exceeds FCAR cap: {args.max_eval_samples}", 2)
        artifact, artifact_source = _load_or_generate_prediction_artifact(args, started)
        records = list(artifact.get("records") or [])
        if not records:
            return fail("NO_RECORDS", "Prediction artifact has no per-frame records.", 3)
        if artifact_source == "regenerated":
            report["policy"]["gpu_used"] = True
            report["policy"]["training_performed"] = True
            report["policy"]["trained_components"].append("fixed rank-4 LoRA baseline regenerated for predictions")
        report["policy"]["downloads_performed"] = bool((artifact.get("policy") or {}).get("downloads_performed", False))
        report["policy"]["openvla_oft_executed"] = bool((artifact.get("policy") or {}).get("openvla_oft_executed", False))
        report["policy"]["rollouts_performed"] = bool((artifact.get("policy") or {}).get("rollouts_performed", False))
        report["policy"]["old_custom_route_used"] = bool((artifact.get("policy") or {}).get("old_custom_route_used", False))
        action_min = np.asarray((artifact.get("action_range") or {}).get("min"), dtype=np.float32)
        action_max = np.asarray((artifact.get("action_range") or {}).get("max"), dtype=np.float32)
        if action_min.size == 0 or action_max.size == 0:
            return fail("MISSING_ACTION_RANGE", "Prediction artifact lacks action min/max range.", 4)

        split_records = split_records_by_episode(records, (artifact.get("dataset") or {}).get("selected_tasks"))
        if not split_records["train"] or not split_records["val"] or not split_records["test"]:
            return fail("BAD_SPLIT", "FCAR split produced an empty train/val/test split.", 5)
        split_episode_sets = {name: set(_episode_distribution(items)) for name, items in split_records.items()}
        leakage_checks = {
            "episode_disjoint_train_val": not bool(split_episode_sets["train"] & split_episode_sets["val"]),
            "episode_disjoint_train_test": not bool(split_episode_sets["train"] & split_episode_sets["test"]),
            "episode_disjoint_val_test": not bool(split_episode_sets["val"] & split_episode_sets["test"]),
            "no_ground_truth_or_oracle_in_inference_features": True,
            "old_custom_route_used": False,
        }

        gate = _train_tiny_gate(split_records, seed=int(args.seed))
        report["policy"]["training_performed"] = True
        report["policy"]["trained_components"].append("FCAR tiny CPU gate")
        evaluated = _evaluate_split(split_records, action_min=action_min, action_max=action_max, gate=gate)
        metrics = evaluated["metrics"]
        all_200 = _all_200_metrics(records, action_min, action_max)
        metrics["all_200"] = all_200

        test = metrics["test"]
        base_l2 = float(test["frozen_base"]["action_l2_mean"])
        lora_l2 = float(test["rank4_lora"]["action_l2_mean"])
        mean_l2 = float(test["mean_action_prior"]["action_l2_mean"])
        moira_l2 = float(test["moira_style_instruction_task_router"]["action_l2_mean"])
        static_l2 = float(test["adapter_soup_static_merge"]["action_l2_mean"])
        frame_l2 = float(test["frame_oracle"]["action_l2_mean"])
        task_l2 = float(test["task_oracle"]["action_l2_mean"])
        fcar_l2 = float(test["fcar_tiny_gate"]["action_l2_mean"])
        train_fcar_l2 = float(metrics["train"]["fcar_tiny_gate"]["action_l2_mean"])
        fcar_gain = _improvement(base_l2, fcar_l2)
        frame_gain = max(base_l2 - frame_l2, 0.0)
        recovered = (base_l2 - fcar_l2) / frame_gain if frame_gain > 1e-12 else None
        oracle_inputs_used = bool(gate["config"]["inference_inputs_use_ground_truth"] or gate["config"]["inference_inputs_use_oracle_label"])
        final_decision = choose_final_decision(
            base_l2=base_l2,
            lora_l2=lora_l2,
            mean_l2=mean_l2,
            moira_l2=moira_l2,
            static_l2=static_l2,
            fcar_l2=fcar_l2,
            train_l2=train_fcar_l2,
            oracle_inputs_used=oracle_inputs_used,
        )
        reasons = {
            "GO_FCAR_SCALEUP": "FCAR clears the hard gain threshold over frozen/base and beats LoRA, MoIRA-style task routing, static merge, and mean-action prior.",
            "WEAK_FCAR_SIGNAL_NEEDS_REPEAT": "FCAR beats frozen/base but does not clear all hard success thresholds.",
            "FCAR_OVERFITS": "FCAR improves train split but has an unacceptable train/eval gap.",
            "FCAR_KILLED_BY_STATIC_BASELINE": "Static merge or MoIRA-style task/instruction routing matches or beats FCAR.",
            "NO_FCAR_GAIN_OVER_BASE": "FCAR does not beat frozen/base on the held-out gate-test split.",
            "FCAR_IMPLEMENTATION_BLOCKED": "FCAR result is invalid because the implementation required disallowed inference inputs.",
        }
        exact_next_prompt = None
        if final_decision == "GO_FCAR_SCALEUP":
            exact_next_prompt = (
                "Scale FCAR only within the official SmolVLA-LIBERO route: predeclare the larger split, keep the same baselines and kill criteria, "
                "save per-frame predictions, and do not run rollouts or paper claims until a separate simulator-readiness gate is passed."
            )

        test_alphas = evaluated["alphas"]["test"]
        fcar_rows = evaluated["rows"]["test"]["fcar_tiny_gate"]
        base_rows = evaluated["rows"]["test"]["frozen_base"]
        report.update(
            {
                "status": "completed",
                "final_decision": final_decision,
                "decision_reason": reasons[final_decision],
                "exact_next_prompt": exact_next_prompt,
                "artifact": {
                    "source": artifact_source,
                    "path": str(Path(args.prediction_artifact)),
                    "record_count": len(records),
                    "rank4_lora_regenerated": bool((artifact.get("policy") or {}).get("rank4_lora_regenerated", False)),
                },
                "split": {
                    "policy": "deterministic episode-grouped split, approximately 60/20/20 over the 200-frame failure-mining selection",
                    "seed": int(args.seed),
                    "counts": _split_counts(split_records),
                    "leakage_checks": leakage_checks,
                },
                "feature_schema": {
                    "feature_names": gate["feature_names"],
                    "feature_count": len(gate["feature_names"]),
                    "uses_ground_truth_action_at_inference": False,
                    "uses_oracle_label_at_inference": False,
                    "uses_future_frames": False,
                    "uses_rollout_reward": False,
                    "uses_custom_libero_metadata": False,
                    "instruction_embedding_used": False,
                    "primary_signals": ["base_action", "rank4_lora_action", "action_disagreement", "current_8d_state", "normalized_phase"],
                },
                "baselines": {
                    "fixed": [
                        "frozen_base",
                        "rank4_lora",
                        "mean_action_prior",
                        "frame_oracle",
                        "task_oracle",
                        "moira_style_instruction_task_router",
                        "adapter_soup_static_merge",
                        "action_dim_oracle_diagnostic",
                    ],
                    "static_grid": evaluated["static_grid"],
                    "static_selected_weight": evaluated["static_selected_weight"],
                    "static_selection_split": evaluated["static_selection_split"],
                    "moira_routing_from_train": evaluated["moira_routing"],
                },
                "fcar_config": _strip_gate_model(gate),
                "metrics": metrics,
                "oracle_recovery": {
                    "base_test_action_l2": round(base_l2, 9),
                    "frame_oracle_test_action_l2": round(frame_l2, 9),
                    "fcar_test_action_l2": round(fcar_l2, 9),
                    "frame_oracle_gain": round(frame_gain, 9),
                    "fcar_gain": round(base_l2 - fcar_l2, 9),
                    "fraction_of_frame_oracle_headroom": _round_float(recovered) if recovered is not None else None,
                },
                "calibration": test["fcar_tiny_gate"]["calibration"],
                "kill_criteria": {
                    "hard_abs_threshold": FCAR_ABS_SUCCESS,
                    "hard_rel_threshold": FCAR_REL_SUCCESS,
                    "fcar_gain_over_frozen_base": fcar_gain,
                    "fcar_beats_frozen_base": bool(fcar_l2 < base_l2),
                    "fcar_beats_rank4_lora": bool(fcar_l2 < lora_l2),
                    "fcar_beats_mean_action_prior": bool(fcar_l2 < mean_l2),
                    "fcar_beats_moira_style_router": bool(fcar_l2 < moira_l2),
                    "fcar_beats_static_merge": bool(fcar_l2 < static_l2),
                    "fcar_uses_oracle_inference_inputs": oracle_inputs_used,
                    "task_oracle_test_action_l2": round(task_l2, 9),
                },
                "train_eval_gap": {
                    "train_action_l2": round(train_fcar_l2, 9),
                    "test_action_l2": round(fcar_l2, 9),
                    "absolute_gap_test_minus_train": round(fcar_l2 - train_fcar_l2, 9),
                    "relative_gap": round((fcar_l2 - train_fcar_l2) / max(abs(train_fcar_l2), 1e-12), 9),
                },
                "alpha_routing_statistics": {
                    "train": metrics["train"]["fcar_tiny_gate"]["alpha_stats"],
                    "val": metrics["val"]["fcar_tiny_gate"]["alpha_stats"],
                    "test": metrics["test"]["fcar_tiny_gate"]["alpha_stats"],
                    "test_alpha_preview": _round_vector(test_alphas[:10], 9),
                },
                "failure_cases": _failure_cases(fcar_rows, base_rows),
                "anchor_reconciliation": _anchor_reconciliation(metrics, Path(args.routing_report_json)),
                "runtime": {
                    "total_elapsed_sec": round(time.monotonic() - started, 3),
                    "rss_final_mb": _rss_mb(),
                    "tiny_gate_training_elapsed_sec": gate["training_elapsed_sec"],
                    "prediction_artifact_source": artifact_source,
                    "device_audit": artifact.get("device_audit"),
                },
            }
        )
        if artifact.get("rank4_lora_regeneration"):
            report["rank4_lora_regeneration"] = artifact["rank4_lora_regeneration"]
        return report, 0
    except FcarError as exc:
        return fail(exc.code, str(exc), 20)
    except Exception as exc:
        report["status"] = "failed"
        report["final_decision"] = "FCAR_IMPLEMENTATION_BLOCKED"
        report["decision_reason"] = str(exc)
        report["errors"].append(
            {
                "code": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 31


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
    parser.add_argument("--prediction-artifact", default="reports/fcar_prediction_artifact.json")
    parser.add_argument("--routing-report-json", default="reports/official_smolvla_routing_design_gate.json")
    parser.add_argument("--report-json", default="reports/fcar_tiny_gate_result.json")
    parser.add_argument("--report-md", default="reports/fcar_tiny_gate_result.md")
    parser.add_argument("--decision-md", default="reports/fcar_tiny_gate_decision.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_path = Path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report, Path(args.report_md))
    _write_decision(report, Path(args.decision_md))
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
