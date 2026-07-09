"""Stable official SmolVLA-LIBERO prediction artifact and baseline evaluator.

This runner executes the fixed split/metric protocol created by
``official_libero_stable_protocol``. It may train only the standard rank-4 LoRA
baseline on the fixed train split, then evaluates frozen/base, rank-4 LoRA,
mean action, static mixtures, MoIRA-style task routing, and oracle upper bounds.

It does not implement a new method, revive FCAR, run rollouts, run a full
benchmark, run OpenVLA-OFT, download assets, or use the archived custom
LIBERO_7D route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.fcar_tiny_gate import (
    _apply_task_router,
    _choose_static_weight,
    _metric_package,
    _rows_from_records,
    _static_rows,
    _task_router_from_training,
)
from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
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
from tca_map.smolvla.official_libero_failure_mining import _metric_row, summarize_rows
from tca_map.smolvla.official_libero_routing_design_gate import (
    action_dim_oracle_rows,
    frame_oracle_rows,
    task_oracle_rows,
)


DATE = "2026-07-10 KST"
ARTIFACT_VERSION = 2
MAX_RUNTIME_SECONDS = 2 * 60 * 60
MAX_TRAINING_STEPS = 100
STATIC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
REPORT_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "frame_oracle",
    "task_oracle",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
REALISTIC_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
FINAL_DECISIONS = {
    "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT",
    "NEEDS_LONGER_LORA_BASELINE_REPRO",
    "SIMPLE_BASELINES_EXPLAIN_GAP",
    "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_STATIC",
    "METHOD_DESIGN_STILL_BLOCKED",
    "ARTIFACT_GENERATION_FAILED",
    "TOO_HEAVY_LOCAL",
    "CPU_FALLBACK_BUG",
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


class StableArtifactError(RuntimeError):
    """Reportable bounded failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _round(value: Any, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_vector(values: Any, digits: int = 9) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [round(float(x), digits) for x in array.tolist()]


def _phase(frame_index: int, episode_length: int) -> str:
    if episode_length <= 1:
        return "unknown"
    ratio = frame_index / max(1, episode_length - 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "mid"
    return "late"


def _sample_state(raw_sample: dict[str, Any]) -> list[float]:
    for key in ("observation.state", "observation_state", "state"):
        if key in raw_sample:
            value = raw_sample[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return _round_vector(np.asarray(value, dtype=np.float32).reshape(-1), 9)
    return []


def _record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["episode_index"]), int(record["frame_index"]), int(record["task_index"]))


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["episode_index"]), int(row["frame_index"]), int(row["task_index"]))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest.get("summary") or {}
    frames = summary.get("frame_counts") or {}
    leakage = summary.get("leakage_checks") or {}
    if int(frames.get("train", 0)) < 500 or int(frames.get("val", 0)) < 200 or int(frames.get("test", 0)) < 500:
        raise StableArtifactError("NEEDS_TASK_BALANCED_SPLIT", f"Manifest frame counts are too small: {frames}")
    if not all(bool(value) for value in leakage.values()):
        raise StableArtifactError("NEEDS_TASK_BALANCED_SPLIT", f"Manifest leakage checks failed: {leakage}")


def _manifest_samples(manifest: dict[str, Any]) -> tuple[list[int], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    _validate_manifest(manifest)
    splits = manifest.get("splits") or {}
    selected_episodes = sorted({int(record["episode_index"]) for records in splits.values() for record in records})
    local_offsets: dict[int, int] = {}
    offset = 0
    lengths: dict[int, int] = {}
    for episode in selected_episodes:
        episode_records = [
            record
            for records in splits.values()
            for record in records
            if int(record["episode_index"]) == episode
        ]
        if not episode_records:
            continue
        length = int(episode_records[0]["episode_length"])
        lengths[episode] = length
        local_offsets[episode] = offset
        offset += length

    split_samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    all_samples: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for record in splits.get(split, []):
            episode = int(record["episode_index"])
            frame = int(record["frame_index"])
            sample = {
                "sample_id": str(record.get("sample_id") or f"{split}_ep{episode}_frame{frame}"),
                "dataset_local_index": int(local_offsets[episode] + frame),
                "dataset_global_index": int(record.get("dataset_global_index", -1)),
                "episode_index": episode,
                "frame_index": frame,
                "episode_length": int(record.get("episode_length") or lengths[episode]),
                "task_index": int(record["task_index"]),
                "task": str(record["task"]),
                "phase": _phase(frame, int(record.get("episode_length") or lengths[episode])),
                "split": split,
                "normalized_phase": float(record.get("normalized_phase", frame / max(1, lengths[episode] - 1))),
            }
            split_samples[split].append(sample)
            all_samples.append(sample)
    return selected_episodes, split_samples, all_samples


def _metric_with_balance(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    package = summarize_rows(rows)
    package["per_task"] = _group_summary(rows, "task_index")
    package["per_episode"] = _group_summary(rows, "episode_index")
    package["per_phase"] = _group_summary(rows, "phase")
    package["task_balanced_action_l2_mean"] = _balanced_mean(package["per_task"], "action_l2_mean")
    package["episode_balanced_action_l2_mean"] = _balanced_mean(package["per_episode"], "action_l2_mean")
    package["per_dim_l2_mean"] = _per_dim_l2(rows)
    package["task_bootstrap_ci95_action_l2"] = _bootstrap_ci(rows, group_key="task_index", seed=seed)
    package["episode_bootstrap_ci95_action_l2"] = _bootstrap_ci(rows, group_key="episode_index", seed=seed + 17)
    return package


def _enrich_package(package: dict[str, Any], rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    package["per_episode"] = _group_summary(rows, "episode_index")
    package["per_phase"] = _group_summary(rows, "phase")
    package["task_balanced_action_l2_mean"] = _balanced_mean(package.get("per_task") or {}, "action_l2_mean")
    package["episode_balanced_action_l2_mean"] = _balanced_mean(package["per_episode"], "action_l2_mean")
    package["per_dim_l2_mean"] = _per_dim_l2(rows)
    package["task_bootstrap_ci95_action_l2"] = _bootstrap_ci(rows, group_key="task_index", seed=seed)
    package["episode_bootstrap_ci95_action_l2"] = _bootstrap_ci(rows, group_key="episode_index", seed=seed + 17)
    return package


def _group_summary(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {key: summarize_rows(value) for key, value in sorted(groups.items(), key=lambda item: _sort_key(item[0]))}


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def _balanced_mean(grouped: dict[str, Any], metric_key: str) -> float | None:
    values = [float(value[metric_key]) for value in grouped.values() if value.get(metric_key) is not None]
    if not values:
        return None
    return _round(float(np.mean(values)))


def _per_dim_l2(rows: list[dict[str, Any]]) -> list[float | None]:
    values: list[float | None] = []
    for dim in range(7):
        diffs = []
        for row in rows:
            pred = row.get("pred_preview") or []
            target = row.get("target_preview") or []
            if len(pred) > dim and len(target) > dim:
                diffs.append(float(pred[dim]) - float(target[dim]))
        values.append(_round(float(np.sqrt(np.mean(np.square(diffs))))) if diffs else None)
    return values


def _bootstrap_ci(rows: list[dict[str, Any]], *, group_key: str, seed: int, iterations: int = 500) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(row["action_l2"]))
    group_means = np.asarray([float(np.mean(values)) for values in grouped.values()], dtype=np.float64)
    if group_means.size == 0:
        return {"group_count": 0, "low": None, "high": None, "mean": None}
    if group_means.size == 1:
        value = _round(float(group_means[0]))
        return {"group_count": 1, "low": value, "high": value, "mean": value}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        sample = rng.choice(group_means, size=group_means.size, replace=True)
        draws.append(float(np.mean(sample)))
    return {
        "group_count": int(group_means.size),
        "low": _round(float(np.percentile(draws, 2.5))),
        "high": _round(float(np.percentile(draws, 97.5))),
        "mean": _round(float(np.mean(draws))),
    }


def _evaluate_policy_rows(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    samples: list[dict[str, Any]],
    action_min: np.ndarray,
    action_max: np.ndarray,
    include_eval_loss: bool,
    label: str,
    started: float,
    progress_every: int,
) -> list[dict[str, Any]]:
    import torch

    rows = []
    policy.eval()
    with torch.no_grad():
        for index, sample_meta in enumerate(samples):
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                raise StableArtifactError("TOO_HEAVY_LOCAL", "Stable artifact generation exceeded the two-hour runtime cap.")
            raw_sample = dataset[int(sample_meta["dataset_local_index"])]
            batch = _add_training_batch_dims(preprocessor(raw_sample))
            devices = _tensor_devices(batch)
            if torch.cuda.is_available() and not all(value.startswith("cuda") for value in devices.values()):
                raise StableArtifactError("CPU_FALLBACK_BUG", f"CUDA available but {label} input tensors are on CPU: {devices}")
            eval_loss = None
            if include_eval_loss:
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
            if progress_every > 0 and (index + 1) % progress_every == 0:
                print(f"[{label}] evaluated {index + 1}/{len(samples)} records", flush=True)
    return rows


def _evaluate_mean_rows(
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


def _make_record(
    *,
    base: dict[str, Any],
    lora: dict[str, Any],
    mean: dict[str, Any],
    sample: dict[str, Any],
    raw_sample: dict[str, Any],
) -> dict[str, Any]:
    base_l2 = float(base["action_l2"])
    lora_l2 = float(lora["action_l2"])
    return {
        "sample_id": str(sample["sample_id"]),
        "sample_key": {
            "episode_index": int(base["episode_index"]),
            "frame_index": int(base["frame_index"]),
            "task_index": int(base["task_index"]),
        },
        "split": str(sample["split"]),
        "dataset_local_index": int(sample["dataset_local_index"]),
        "dataset_global_index": int(sample.get("dataset_global_index", -1)),
        "episode_index": int(base["episode_index"]),
        "frame_index": int(base["frame_index"]),
        "episode_length": int(base["episode_length"]),
        "task_index": int(base["task_index"]),
        "task": str(base["task"]),
        "phase": str(base["phase"]),
        "normalized_phase": _round(sample["normalized_phase"]),
        "state": _sample_state(raw_sample),
        "base_action": _round_vector(base["pred_preview"], 9),
        "lora_action": _round_vector(lora["pred_preview"], 9),
        "mean_action": _round_vector(mean["pred_preview"], 9),
        "target_action": _round_vector(base["target_preview"], 9),
        "base_eval_loss": base.get("eval_loss"),
        "lora_eval_loss": lora.get("eval_loss"),
        "base_action_l2": base.get("action_l2"),
        "lora_action_l2": lora.get("action_l2"),
        "mean_action_l2": mean.get("action_l2"),
        "base_translation_l2": base.get("translation_l2"),
        "lora_translation_l2": lora.get("translation_l2"),
        "base_rotation_l2": base.get("rotation_l2"),
        "lora_rotation_l2": lora.get("rotation_l2"),
        "base_gripper_abs": base.get("gripper_abs"),
        "lora_gripper_abs": lora.get("gripper_abs"),
        "oracle_help_label": int(lora_l2 < base_l2),
        "base_minus_lora_action_l2": _round(base_l2 - lora_l2),
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


def _preflight(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint_path)
    dataset = Path(args.dataset_root)
    manifest_path = Path(args.split_manifest)
    metric_path = Path(args.metric_protocol)
    wrapper = Path("scripts/248_official_smolvla_prediction_artifact_from_manifest.ps1")
    summary = manifest.get("summary") or {}
    estimated_records = sum(int(value) for value in (summary.get("frame_counts") or {}).values())
    return {
        "git_branch_expected": "codex/official-smolvla-stable-artifact-eval",
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "dataset_path": str(dataset),
        "dataset_exists": dataset.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "metric_protocol_path": str(metric_path),
        "metric_protocol_exists": metric_path.exists(),
        "wrapper_path": str(wrapper),
        "wrapper_exists_after_this_commit": True,
        "python_runner": "tca_map.smolvla.official_libero_stable_artifact_eval",
        "output_artifact_path": str(Path(args.output_artifact)),
        "device_plan": "CUDA rank-4 LoRA regeneration; stop with CPU_FALLBACK_BUG if params or tensors remain on CPU",
        "estimated_frame_counts": summary.get("frame_counts"),
        "estimated_prediction_records": estimated_records,
        "estimated_runtime": "bounded by two-hour cap; previous 200-frame artifact took about 228 seconds",
        "old_custom_libero_7d_route_used": False,
    }


def _load_existing_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = _read_json(path)
    if int(artifact.get("artifact_version", 0)) != ARTIFACT_VERSION:
        return None
    if artifact.get("records"):
        return artifact
    return None


def generate_artifact(args: argparse.Namespace, manifest: dict[str, Any], preflight: dict[str, Any], started: float) -> dict[str, Any]:
    existing = _load_existing_artifact(Path(args.output_artifact))
    if existing is not None and not bool(args.force):
        existing["artifact_status"] = "loaded_existing"
        return existing

    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        raise StableArtifactError("ARTIFACT_GENERATION_FAILED", "Forbidden gate(s) set: " + ", ".join(forbidden))
    if int(args.steps) < 1 or int(args.steps) > MAX_TRAINING_STEPS:
        raise StableArtifactError("TOO_HEAVY_LOCAL", f"Rank-4 LoRA steps must be in [1, {MAX_TRAINING_STEPS}], got {args.steps}.")

    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise StableArtifactError("CPU_FALLBACK_BUG", "CUDA unavailable; refusing rank-4 LoRA regeneration on CPU.")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    fps = float(info.get("fps", 10.0))
    chunk_size = int(args.chunk_size)
    selected_episodes, split_samples, all_samples = _manifest_samples(manifest)
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
    action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)
    mean_action = np.asarray(_stat_vector(stats, "action", "mean"), dtype=np.float32)

    dataset = LeRobotDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(vlm_root)
    if hasattr(cfg, "chunk_size"):
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

    probe = _add_training_batch_dims(preprocessor(dataset[int(split_samples["train"][0]["dataset_local_index"])]))
    input_devices = _tensor_devices(probe)
    param_summary = _parameter_summary(policy)
    if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
        value.startswith("cuda") for value in input_devices.values()
    ):
        raise StableArtifactError("CPU_FALLBACK_BUG", f"CUDA available but params/inputs are not all CUDA: params={param_summary}, inputs={input_devices}")

    print(f"[stable-artifact] evaluating frozen/base on {len(all_samples)} records", flush=True)
    base_rows = _evaluate_policy_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        samples=all_samples,
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=bool(args.include_eval_loss),
        label="frozen_base",
        started=started,
        progress_every=int(args.progress_every),
    )
    mean_rows = _evaluate_mean_rows(
        dataset=dataset,
        samples=all_samples,
        mean_action=mean_action,
        action_min=action_min,
        action_max=action_max,
    )

    print(f"[stable-artifact] training standard rank-4 LoRA for {args.steps} steps on manifest train split", flush=True)
    policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.train()
    lora_param_summary = _parameter_summary(policy)
    optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))
    rng = np.random.default_rng(int(args.seed))
    train_order = rng.permutation(len(split_samples["train"])).tolist()
    loss_curve = []
    grad_curve = []
    training_started = time.monotonic()
    first_batch_devices: dict[str, str] | None = None
    first_batch_shapes: dict[str, list[int]] | None = None
    for step in range(int(args.steps)):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise StableArtifactError("TOO_HEAVY_LOCAL", "Stable artifact generation exceeded the two-hour runtime cap during LoRA training.")
        train_sample = split_samples["train"][train_order[step % len(train_order)]]
        raw_sample = dataset[int(train_sample["dataset_local_index"])]
        batch = _add_training_batch_dims(preprocessor(raw_sample))
        devices = _tensor_devices(batch)
        if first_batch_devices is None:
            first_batch_devices = devices
            first_batch_shapes = _tensor_shapes(batch)
        if not all(value.startswith("cuda") for value in devices.values()):
            raise StableArtifactError("CPU_FALLBACK_BUG", f"CUDA available but LoRA training tensors are on CPU: {devices}")
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_output(policy.forward(batch))
        loss_value = _to_float(loss)
        if not math.isfinite(loss_value):
            raise StableArtifactError("ARTIFACT_GENERATION_FAILED", f"Non-finite rank-4 LoRA loss at step {step}: {loss_value}")
        loss.backward()
        grad_summary = _gradient_summary(policy)
        if int(grad_summary["nonzero_grad_tensors"]) == 0:
            raise StableArtifactError("ARTIFACT_GENERATION_FAILED", f"No nonzero rank-4 LoRA gradients at step {step}.")
        optimizer.step()
        cuda_now = _cuda_memory(torch)
        loss_curve.append(
            {
                "step": int(step),
                "loss": _round(loss_value),
                "allocated_mb": cuda_now["allocated_mb"],
                "max_allocated_mb": cuda_now["max_allocated_mb"],
            }
        )
        grad_curve.append({"step": int(step), **grad_summary})
    training_elapsed = time.monotonic() - training_started

    print(f"[stable-artifact] evaluating rank-4 LoRA on {len(all_samples)} records", flush=True)
    lora_rows = _evaluate_policy_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        samples=all_samples,
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=bool(args.include_eval_loss),
        label="rank4_lora",
        started=started,
        progress_every=int(args.progress_every),
    )

    base_by_key = {_row_key(row): row for row in base_rows}
    lora_by_key = {_row_key(row): row for row in lora_rows}
    mean_by_key = {_row_key(row): row for row in mean_rows}
    records = []
    for index, sample in enumerate(all_samples):
        key = _record_key(sample)
        raw_sample = dataset[int(sample["dataset_local_index"])]
        records.append(
            _make_record(
                base=base_by_key[key],
                lora=lora_by_key[key],
                mean=mean_by_key[key],
                sample=sample,
                raw_sample=raw_sample,
            )
        )
        if int(args.progress_every) > 0 and (index + 1) % int(args.progress_every) == 0:
            print(f"[stable-artifact] assembled {index + 1}/{len(all_samples)} records", flush=True)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "date": DATE,
        "artifact_status": "generated",
        "source": "official_smolvla_stable_manifest_rank4_lora",
        "policy": {
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "old_custom_route_used": False,
            "official_model_dataset_used": True,
            "rank4_lora_regenerated": True,
            "smolvla_backbone_trained": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_root),
            "hf_home": str(hf_home),
            "vlm_root": str(vlm_root),
            "split_manifest": str(Path(args.split_manifest)),
            "metric_protocol": str(Path(args.metric_protocol)),
        },
        "preflight": preflight,
        "dataset": {
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "total_tasks": int(info.get("total_tasks", 0)),
            "selected_episode_count": len(selected_episodes),
            "selected_episodes": selected_episodes,
            "split_frame_counts": {split: len(samples) for split, samples in split_samples.items()},
            "split_episode_counts": {
                split: len({int(sample["episode_index"]) for sample in samples})
                for split, samples in split_samples.items()
            },
            "split_task_counts": {
                split: len({int(sample["task_index"]) for sample in samples})
                for split, samples in split_samples.items()
            },
            "prediction_record_count": len(records),
        },
        "action_range": {"min": _round_vector(action_min, 9), "max": _round_vector(action_max, 9)},
        "device_audit": {
            "cuda_available": True,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "model_parameter_device": param_summary["first_parameter_device"],
            "model_parameter_dtype": param_summary["first_parameter_dtype"],
            "input_tensor_devices": input_devices,
            "input_tensor_shapes": _tensor_shapes(probe),
            "autocast_status_initial_final": _safe_autocast_status(torch),
            "cuda_memory": _cuda_memory(torch),
        },
        "rank4_lora_regeneration": {
            "steps": int(args.steps),
            "seed": int(args.seed),
            "batch_size": 1,
            "train_split": "train",
            "train_frame_count": len(split_samples["train"]),
            "trainable_params": lora_param_summary["trainable_params"],
            "total_params": lora_param_summary["total_params"],
            "loss_before": loss_curve[0]["loss"] if loss_curve else None,
            "loss_after": loss_curve[-1]["loss"] if loss_curve else None,
            "loss_curve": loss_curve,
            "last_grad_norm": grad_curve[-1]["grad_norm"] if grad_curve else None,
            "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"] if grad_curve else None,
            "training_elapsed_sec": _round(training_elapsed, 3),
            "steps_per_sec": _round(len(loss_curve) / max(training_elapsed, 1e-12), 6),
            "input_tensor_devices_first_batch": first_batch_devices,
            "input_tensor_shapes_first_batch": first_batch_shapes,
            "autocast_status": _safe_autocast_status(torch),
        },
        "source_metric_check": {
            "frozen_base_all": _metric_with_balance(base_rows, seed=int(args.seed)),
            "rank4_lora_all": _metric_with_balance(lora_rows, seed=int(args.seed) + 1),
            "mean_action_prior_all": _metric_with_balance(mean_rows, seed=int(args.seed) + 2),
            "eval_loss_included": bool(args.include_eval_loss),
        },
        "runtime": {
            "total_elapsed_sec": _round(time.monotonic() - started, 3),
            "rss_final_mb": _rss_mb(),
            "cuda": _cuda_memory(torch),
        },
        "records": records,
    }
    path = Path(args.output_artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return artifact


def _records_to_rows(records: list[dict[str, Any]], action_min: np.ndarray, action_max: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    return {
        "frozen_base": _rows_from_records(records, pred_key="base_action", eval_loss_key="base_eval_loss", action_min=action_min, action_max=action_max, selected_expert="frozen_base"),
        "rank4_lora": _rows_from_records(records, pred_key="lora_action", eval_loss_key="lora_eval_loss", action_min=action_min, action_max=action_max, selected_expert="rank4_lora"),
        "mean_action_prior": _rows_from_records(records, pred_key="mean_action", eval_loss_key=None, action_min=action_min, action_max=action_max, selected_expert="mean_action_prior"),
    }


def _evaluate_baselines(artifact: dict[str, Any], *, seed: int) -> dict[str, Any]:
    records = artifact.get("records") or []
    if not records:
        raise StableArtifactError("ARTIFACT_GENERATION_FAILED", "Stable prediction artifact contains no records.")
    action_min = np.asarray((artifact.get("action_range") or {}).get("min"), dtype=np.float32)
    action_max = np.asarray((artifact.get("action_range") or {}).get("max"), dtype=np.float32)
    split_records = {
        split: [record for record in records if str(record.get("split")) == split]
        for split in ["train", "val", "test"]
    }
    train_rows = _records_to_rows(split_records["train"], action_min, action_max)
    test_rows = _records_to_rows(split_records["test"], action_min, action_max)
    task_oracle, task_routing = task_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])
    frame_oracle = frame_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])
    action_dim_oracle = action_dim_oracle_rows(test_rows["frozen_base"], test_rows["rank4_lora"])
    moira_routing = _task_router_from_training(train_rows["frozen_base"], train_rows["rank4_lora"])
    moira_rows = _apply_task_router(test_rows["frozen_base"], test_rows["rank4_lora"], moira_routing)
    static_weight, static_selection_split, static_grid = _choose_static_weight(split_records, action_min=action_min, action_max=action_max)
    static_selected = _static_rows(split_records["test"], static_weight, action_min=action_min, action_max=action_max)

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

    metrics = {}
    for index, (name, row_set) in enumerate(rows.items()):
        base_package = _metric_package(row_set, base_rows=test_rows["frozen_base"], lora_rows=test_rows["rank4_lora"])
        metrics[name] = _enrich_package(base_package, row_set, seed=seed + index)

    realistic_order = sorted(
        ((name, float(metrics[name]["action_l2_mean"])) for name in REALISTIC_BASELINES),
        key=lambda item: item[1],
    )
    with_oracles_order = sorted(
        ((name, float(metrics[name]["action_l2_mean"])) for name in REPORT_BASELINES),
        key=lambda item: item[1],
    )
    return {
        "split_summary": _split_summary(split_records),
        "metrics": metrics,
        "rows": {name: rows[name] for name in rows},
        "static_selection": {
            "selected_weight": static_weight,
            "selection_split": static_selection_split,
            "grid": static_grid,
            "test_tuning_allowed": False,
        },
        "task_oracle_routing": task_routing,
        "moira_routing_from_train": moira_routing,
        "rank_order_realistic": [{"baseline": name, "action_l2": _round(value)} for name, value in realistic_order],
        "rank_order_with_oracles": [{"baseline": name, "action_l2": _round(value)} for name, value in with_oracles_order],
        "win_counts_by_task": _win_counts_by_group(metrics, group_key="per_task", baseline_names=REALISTIC_BASELINES),
        "win_counts_by_phase": _win_counts_by_group(metrics, group_key="per_phase", baseline_names=REALISTIC_BASELINES),
    }


def _split_summary(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output = {}
    for split, records in split_records.items():
        output[split] = {
            "frame_count": len(records),
            "episode_count": len({int(record["episode_index"]) for record in records}),
            "task_count": len({int(record["task_index"]) for record in records}),
            "task_distribution": dict(sorted(Counter(str(record["task_index"]) for record in records).items(), key=lambda item: int(item[0]))),
        }
    return output


def _win_counts_by_group(metrics: dict[str, Any], *, group_key: str, baseline_names: list[str]) -> dict[str, Any]:
    groups = sorted({group for name in baseline_names for group in (metrics.get(name, {}).get(group_key) or {})}, key=_sort_key)
    wins: Counter[str] = Counter()
    details = []
    for group in groups:
        values = []
        for name in baseline_names:
            group_metrics = (metrics.get(name, {}).get(group_key) or {}).get(group)
            if group_metrics and group_metrics.get("action_l2_mean") is not None:
                values.append((name, float(group_metrics["action_l2_mean"])))
        if not values:
            continue
        winner, value = min(values, key=lambda item: item[1])
        wins[winner] += 1
        details.append({"group": group, "winner": winner, "action_l2": _round(value)})
    return {"counts": dict(sorted(wins.items())), "details": details}


def _stability_analysis(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    base = float(metrics["frozen_base"]["action_l2_mean"])
    lora = float(metrics["rank4_lora"]["action_l2_mean"])
    mean = float(metrics["mean_action_prior"]["action_l2_mean"])
    frame = float(metrics["frame_oracle"]["action_l2_mean"])
    task = float(metrics["task_oracle"]["action_l2_mean"])
    moira = float(metrics["moira_style_instruction_task_router"]["action_l2_mean"])
    static = float(metrics["static_mix_val_selected"]["action_l2_mean"])
    best_realistic = min(base, lora, mean, moira, static)
    base_competitive = base - best_realistic <= max(0.005, 0.05 * max(best_realistic, 1e-12))
    lora_gain = base - lora
    static_gain_over_best_single = min(base, lora) - static
    frame_headroom_over_base = base - frame
    frame_headroom_after_static = static - frame
    task_headroom = base - task
    task_ci = metrics["frozen_base"]["task_bootstrap_ci95_action_l2"]
    ci_width = None
    if task_ci.get("low") is not None and task_ci.get("high") is not None:
        ci_width = float(task_ci["high"]) - float(task_ci["low"])
    task_wins = evaluation["win_counts_by_task"]["counts"]
    lora_task_wins = sum(
        1
        for task_id, base_task in (metrics["frozen_base"].get("per_task") or {}).items()
        if task_id in (metrics["rank4_lora"].get("per_task") or {})
        and float(metrics["rank4_lora"]["per_task"][task_id]["action_l2_mean"]) < float(base_task["action_l2_mean"])
    )
    task_count = len(metrics["frozen_base"].get("per_task") or {})
    stable_enough = bool(task_count >= 20 and evaluation["split_summary"]["test"]["frame_count"] >= 500 and (ci_width is None or ci_width < 0.08))
    method_gap = bool(frame_headroom_after_static >= 0.005 and frame_headroom_after_static / max(static, 1e-12) >= 0.05)
    return {
        "is_frozen_base_still_competitive": base_competitive,
        "is_rank4_lora_robustly_better_than_frozen_base": bool(lora_gain >= 0.005 and lora_task_wins >= math.ceil(0.60 * task_count)),
        "is_rank4_lora_robustly_worse_than_frozen_base": bool(-lora_gain >= 0.005 and lora_task_wins <= math.floor(0.40 * task_count)),
        "rank4_lora_task_wins_over_base": lora_task_wins,
        "rank4_lora_task_count": task_count,
        "does_static_merge_beat_both_base_and_lora": bool(static < base and static < lora),
        "does_frame_oracle_headroom_remain_meaningful": bool(frame_headroom_over_base >= 0.005 and frame_headroom_over_base / max(base, 1e-12) >= 0.05),
        "does_frame_oracle_remain_after_static": method_gap,
        "does_task_oracle_remain_weak": bool(task_headroom < 0.005 or task_headroom / max(base, 1e-12) < 0.05),
        "does_moira_style_router_remain_weak": bool(moira >= min(base, lora, static) or abs(moira - base) < 0.005),
        "are_metrics_stable_enough_for_method_design_later": stable_enough,
        "is_method_worthy_gap_left_after_simple_static_baselines": method_gap,
        "does_larger_artifact_resolve_previous_split_instability": stable_enough,
        "best_realistic_action_l2": _round(best_realistic),
        "action_l2": {
            "frozen_base": _round(base),
            "rank4_lora": _round(lora),
            "mean_action_prior": _round(mean),
            "frame_oracle": _round(frame),
            "task_oracle": _round(task),
            "moira_style_instruction_task_router": _round(moira),
            "static_mix_val_selected": _round(static),
        },
        "base_minus_lora": _round(lora_gain),
        "static_gain_over_best_single": _round(static_gain_over_best_single),
        "frame_oracle_headroom_over_base": _round(frame_headroom_over_base),
        "frame_oracle_headroom_after_static": _round(frame_headroom_after_static),
        "task_oracle_headroom_over_base": _round(task_headroom),
        "frozen_base_task_bootstrap_ci_width": _round(ci_width) if ci_width is not None else None,
        "realistic_win_counts_by_task": task_wins,
    }


def _choose_decision(analysis: dict[str, Any]) -> str:
    if not analysis["are_metrics_stable_enough_for_method_design_later"]:
        return "METHOD_DESIGN_STILL_BLOCKED"
    if analysis["is_method_worthy_gap_left_after_simple_static_baselines"] and analysis["does_task_oracle_remain_weak"]:
        return "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_STATIC"
    if not analysis["does_frame_oracle_headroom_remain_meaningful"] or not analysis["does_frame_oracle_remain_after_static"]:
        return "SIMPLE_BASELINES_EXPLAIN_GAP"
    if analysis["is_rank4_lora_robustly_better_than_frozen_base"] or analysis["is_rank4_lora_robustly_worse_than_frozen_base"]:
        return "NEEDS_LONGER_LORA_BASELINE_REPRO"
    return "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT"


def _decision_reason(decision: str) -> str:
    return {
        "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT": "The larger artifact and metric protocol ran; the next step is official baseline reproduction under fixed seeds.",
        "NEEDS_LONGER_LORA_BASELINE_REPRO": "The artifact works, but the single rank-4 LoRA regeneration seed is now the main unresolved robustness issue.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Frozen/base, standard LoRA, or static merge explain most of the observable gap after the fixed protocol run.",
        "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_STATIC": "Frame oracle remains meaningfully better than frozen/base, rank-4 LoRA, MoIRA-style routing, and validation-selected static merge under the stable protocol.",
        "METHOD_DESIGN_STILL_BLOCKED": "Even with the larger artifact, metric variance or task instability is too high for method design.",
        "ARTIFACT_GENERATION_FAILED": "The stable prediction artifact could not be generated.",
        "TOO_HEAVY_LOCAL": "The bounded local RTX/RAM/runtime budget could not support the stable artifact generation.",
        "CPU_FALLBACK_BUG": "CUDA was available but the intended CUDA training/evaluation path fell back to CPU.",
    }[decision]


def _next_step(decision: str) -> str:
    return {
        "STABLE_PROTOCOL_READY_BASELINE_REPRO_NEXT": "Run official baseline reproduction under the fixed stable protocol with explicitly predeclared rank-4 LoRA seeds.",
        "NEEDS_LONGER_LORA_BASELINE_REPRO": "Run independent standard rank-4 LoRA seeds under the fixed manifest; do not design a new method yet.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Stop method design under this evidence and preserve the stable baseline table.",
        "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_STATIC": "Create a later method-design plan only; do not implement it in this run.",
        "METHOD_DESIGN_STILL_BLOCKED": "Diagnose remaining metric/task instability before method design.",
        "ARTIFACT_GENERATION_FAILED": "Fix the exact artifact generation blocker, then rerun the same fixed manifest command.",
        "TOO_HEAVY_LOCAL": "Reduce the manifest through a predeclared smaller protocol or move the same fixed protocol to a larger GPU host.",
        "CPU_FALLBACK_BUG": "Fix CUDA device placement before rerunning LoRA artifact generation.",
    }[decision]


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    manifest = _read_json(Path(args.split_manifest))
    preflight = _preflight(args, manifest)
    report: dict[str, Any] = {
        "date": DATE,
        "status": "started",
        "final_decision": None,
        "preflight": preflight,
        "policy": {
            "experiments_performed": True,
            "training_performed": True,
            "trained_components": ["standard rank-4 LoRA baseline"],
            "gpu_used": True,
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
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "split_manifest": str(Path(args.split_manifest)),
            "metric_protocol": str(Path(args.metric_protocol)),
            "output_artifact": str(Path(args.output_artifact)),
        },
        "errors": [],
    }
    try:
        artifact = generate_artifact(args, manifest, preflight, started)
        evaluation = _evaluate_baselines(artifact, seed=int(args.seed))
        analysis = _stability_analysis(evaluation)
        decision = _choose_decision(analysis)
        artifact_path = Path(args.output_artifact)
        report.update(
            {
                "status": "completed",
                "final_decision": decision,
                "decision_reason": _decision_reason(decision),
                "exact_next_step": _next_step(decision),
                "artifact": {
                    "generated": artifact.get("artifact_status") == "generated",
                    "status": artifact.get("artifact_status"),
                    "path": str(artifact_path),
                    "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
                    "record_count": len(artifact.get("records") or []),
                    "rank4_lora_regenerated": bool((artifact.get("policy") or {}).get("rank4_lora_regenerated")),
                },
                "manifest_summary": (manifest.get("summary") or {}),
                "rank4_lora_regeneration": artifact.get("rank4_lora_regeneration"),
                "device_audit": artifact.get("device_audit"),
                "baseline_evaluation": {
                    key: value
                    for key, value in evaluation.items()
                    if key != "rows"
                },
                "stability_analysis": analysis,
                "runtime": {
                    "total_elapsed_sec": _round(time.monotonic() - started, 3),
                    "rss_final_mb": _rss_mb(),
                },
            }
        )
        if artifact.get("artifact_status") == "loaded_existing":
            report["policy"]["training_performed"] = False
            report["policy"]["trained_components"] = []
    except StableArtifactError as exc:
        decision = exc.code if exc.code in FINAL_DECISIONS else "ARTIFACT_GENERATION_FAILED"
        report["status"] = "blocked"
        report["final_decision"] = decision
        report["decision_reason"] = _decision_reason(decision)
        report["exact_next_step"] = _next_step(decision)
        report["errors"].append({"code": exc.code, "message": str(exc)})
        report["artifact"] = {
            "generated": False,
            "status": "failed",
            "path": str(Path(args.output_artifact)),
            "size_bytes": Path(args.output_artifact).stat().st_size if Path(args.output_artifact).exists() else None,
            "record_count": 0,
        }
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        if decision == "CPU_FALLBACK_BUG":
            report["policy"]["gpu_used"] = False
        return report, 31
    except Exception as exc:  # pragma: no cover - reportable runtime boundary
        report["status"] = "blocked"
        report["final_decision"] = "ARTIFACT_GENERATION_FAILED"
        report["decision_reason"] = _decision_reason("ARTIFACT_GENERATION_FAILED")
        report["exact_next_step"] = _next_step("ARTIFACT_GENERATION_FAILED")
        report["errors"].append({"code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 32
    return report, 0


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status(report: dict[str, Any], path: Path) -> None:
    artifact = report.get("artifact") or {}
    preflight = report.get("preflight") or {}
    lines = [
        "# Official SmolVLA Stable Prediction Artifact Status",
        "",
        f"Date: {report['date']}",
        "",
        f"- status: `{report['status']}`",
        f"- final decision: `{report.get('final_decision')}`",
        f"- model path: `{preflight.get('checkpoint_path')}`",
        f"- dataset path: `{preflight.get('dataset_path')}`",
        f"- manifest path: `{preflight.get('manifest_path')}`",
        f"- metric protocol path: `{preflight.get('metric_protocol_path')}`",
        f"- output artifact path: `{artifact.get('path') or preflight.get('output_artifact_path')}`",
        f"- artifact generated: `{artifact.get('generated')}`",
        f"- artifact size bytes: `{artifact.get('size_bytes')}`",
        f"- artifact record count: `{artifact.get('record_count')}`",
        f"- device plan: `{preflight.get('device_plan')}`",
        f"- estimated frame counts: `{preflight.get('estimated_frame_counts')}`",
        f"- estimated runtime: `{preflight.get('estimated_runtime')}`",
    ]
    if report.get("errors"):
        lines.extend(["", "Errors:", "", *[f"- `{err}`" for err in report["errors"]]])
    _write_lines(path, lines)


def _write_result(report: dict[str, Any], path: Path) -> None:
    metrics = ((report.get("baseline_evaluation") or {}).get("metrics") or {})
    analysis = report.get("stability_analysis") or {}
    artifact = report.get("artifact") or {}
    lines = [
        "# Official SmolVLA Stable Artifact Eval Result",
        "",
        f"Date: {report['date']}",
        "",
        f"- final decision: `{report.get('final_decision')}`",
        f"- experiments happened: `{report['policy']['experiments_performed']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- trained components: `{report['policy']['trained_components']}`",
        f"- GPU/download/OpenVLA-OFT happened: `{report['policy']['gpu_used']}` / `{report['policy']['downloads_performed']}` / `{report['policy']['openvla_oft_executed']}`",
        f"- official model/dataset used: `{report['policy']['official_model_dataset_used']}`",
        f"- old custom route used: `{report['policy']['old_custom_route_used']}`",
        f"- artifact generated: `{artifact.get('generated')}`",
        f"- artifact path: `{artifact.get('path')}`",
        f"- artifact size bytes: `{artifact.get('size_bytes')}`",
        f"- artifact record count: `{artifact.get('record_count')}`",
        "",
        "## Test Metrics",
        "",
        "| baseline | action L2 | task-balanced L2 | translation L2 | rotation L2 | gripper abs | gripper sign acc | range violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in REPORT_BASELINES:
        metric = metrics.get(name) or {}
        lines.append(
            f"| {name} | {metric.get('action_l2_mean')} | {metric.get('task_balanced_action_l2_mean')} | "
            f"{metric.get('translation_l2_mean')} | {metric.get('rotation_l2_mean')} | "
            f"{metric.get('gripper_abs_mean')} | {metric.get('gripper_sign_accuracy')} | {metric.get('range_violation_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Static Selection",
            "",
            f"`{(report.get('baseline_evaluation') or {}).get('static_selection')}`",
            "",
            "## Stability Analysis",
            "",
            *[f"- {key}: `{value}`" for key, value in analysis.items()],
            "",
            "## Exact Next Step",
            "",
            str(report.get("exact_next_step")),
        ]
    )
    _write_lines(path, lines)


def _write_baseline_table(report: dict[str, Any], path: Path) -> None:
    metrics = ((report.get("baseline_evaluation") or {}).get("metrics") or {})
    rank_order = (report.get("baseline_evaluation") or {}).get("rank_order_realistic")
    lines = [
        "# Official SmolVLA Stable Baseline Table",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        "| baseline | action L2 | task-balanced L2 | episode-balanced L2 | task CI95 | episode CI95 | help/hurt/tie vs base |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name in REPORT_BASELINES:
        metric = metrics.get(name) or {}
        task_ci = metric.get("task_bootstrap_ci95_action_l2") or {}
        episode_ci = metric.get("episode_bootstrap_ci95_action_l2") or {}
        hh = metric.get("help_hurt_vs_frozen_base") or {}
        lines.append(
            f"| {name} | {metric.get('action_l2_mean')} | {metric.get('task_balanced_action_l2_mean')} | "
            f"{metric.get('episode_balanced_action_l2_mean')} | "
            f"[{task_ci.get('low')}, {task_ci.get('high')}] | "
            f"[{episode_ci.get('low')}, {episode_ci.get('high')}] | "
            f"{hh.get('help')}/{hh.get('hurt')}/{hh.get('tie')} |"
        )
    lines.extend(["", f"Realistic rank order: `{rank_order}`"])
    _write_lines(path, lines)


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Stable Artifact Decision",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"Reason: {report.get('decision_reason')}",
        "",
        f"Exact next step: {report.get('exact_next_step')}",
    ]
    _write_lines(path, lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--metric-protocol", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--output-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--report-json", default="reports/official_smolvla_stable_artifact_eval_result.json")
    parser.add_argument("--result-md", default="reports/official_smolvla_stable_artifact_eval_result.md")
    parser.add_argument("--status-md", default="reports/official_smolvla_stable_prediction_artifact_status.md")
    parser.add_argument("--baseline-table-md", default="reports/official_smolvla_stable_baseline_table.md")
    parser.add_argument("--decision-md", default="reports/official_smolvla_stable_artifact_decision.md")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--include-eval-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_status(report, Path(args.status_md))
    _write_result(report, Path(args.result_md))
    _write_baseline_table(report, Path(args.baseline_table_md))
    _write_decision(report, Path(args.decision_md))
    summary = {
        "status": report.get("status"),
        "final_decision": report.get("final_decision"),
        "artifact": report.get("artifact"),
        "runtime": report.get("runtime"),
        "errors": report.get("errors"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
