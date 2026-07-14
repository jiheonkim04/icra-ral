"""MTF-VLA adapter-training contract and runner.

The pure helpers in this module validate the frozen MTF selected-training
manifest without importing CUDA, torch, or LeRobot. The training entry point
imports those heavy dependencies only inside the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tca_map.smolvla.mtf_vla import MTFConfig, PROPOSAL_HASH, build_score_records, compute_mtf_scores
from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _gradient_summary,
    _json_default,
    _loss_from_output,
    _parameter_summary,
    _postprocess_action,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)
from tca_map.smolvla.official_libero_failure_mining import summarize_rows
from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _evaluate_policy_rows,
    _group_summary,
    _metric_with_balance,
    _record_key,
    _round,
    _round_vector,
    _row_key,
)


DATE_KST = "2026-07-14 KST"
TRAINING_REPORT_VERSION = 1
MAX_RUNTIME_SECONDS = 4 * 60 * 60
MAX_STEPS = 200
DEFAULT_STEPS = 100
DEFAULT_SEED = 101
VARIANT_ORDER = (
    "mtf_full",
    "mtf_no_retention_ablation",
    "frameskip_proxy_lora",
    "uniform_retained_ratio_lora",
)
REQUIRED_BUNDLE_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_manifest.json",
    "eval_preprocessor_postprocessor_refs.json",
    "source_repro_lock.yaml",
    "sha256_manifest.json",
)
FORBIDDEN_GATES = (
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
)


class MTFTrainingError(RuntimeError):
    """Reportable MTF adapter-training failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MTFTrainArgs:
    """Small typed facade used by tests and the script wrapper."""

    steps: int = DEFAULT_STEPS
    seed: int = DEFAULT_SEED
    lr: float = 2e-4
    variants: tuple[str, ...] = VARIANT_ORDER
    checkpoint_output_root: str = "runs/mtf_vla_checkpoints"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _hash_paths(paths: Sequence[str | Path]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for raw in paths:
        path = Path(raw)
        out[str(path)] = _sha256_file(path) if path.is_file() else None
    return out


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ["torch", "lerobot", "transformers", "peft", "accelerate", "huggingface_hub", "safetensors"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _frame_identity(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(row["task_index"]), int(row["episode_index"]), int(row["frame_index"]))


def _phase_label(frame_index: int, episode_length: int) -> str:
    if episode_length <= 1:
        return "unknown"
    ratio = frame_index / max(1, episode_length - 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "mid"
    return "late"


def _event_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["task_index"]), int(row["episode_index"]), int(row["frame_index"]), int(row["dataset_global_index"]))


def _as_action7(name: str, value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 7:
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"{name} expected 7 values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"{name} contains nonfinite values")
    return [float(x) for x in array.tolist()]


def _official_samples(split_manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    samples: dict[int, dict[str, Any]] = {}
    for split, rows in (split_manifest.get("splits") or {}).items():
        for row in rows:
            dataset_global_index = int(row["dataset_global_index"])
            if dataset_global_index in samples:
                raise MTFTrainingError(
                    "DATA_OR_SUPERVISION_FAILURE",
                    f"duplicate official dataset_global_index in split manifest: {dataset_global_index}",
                )
            samples[dataset_global_index] = {
                "sample_id": str(row.get("sample_id") or f"{split}_task{row['task_index']}_episode{row['episode_index']}_frame{row['frame_index']}"),
                "dataset_global_index": dataset_global_index,
                "episode_index": int(row["episode_index"]),
                "episode_length": int(row["episode_length"]),
                "frame_index": int(row["frame_index"]),
                "task_index": int(row["task_index"]),
                "task": str(row["task"]),
                "phase": _phase_label(int(row["frame_index"]), int(row["episode_length"])),
                "split": str(split),
                "normalized_phase": float(row.get("normalized_phase", 0.0)),
            }
    return samples


def _stable_records(stable_artifact: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in stable_artifact.get("records") or []:
        dataset_global_index = int(record["dataset_global_index"])
        if dataset_global_index in records:
            raise MTFTrainingError(
                "DATA_OR_SUPERVISION_FAILURE",
                f"duplicate stable-artifact dataset_global_index: {dataset_global_index}",
            )
        records[dataset_global_index] = dict(record)
    return records


def _rows_for_variant(selected_manifest: Mapping[str, Any], variant: str) -> list[dict[str, Any]]:
    variants = selected_manifest.get("variants") or {}
    if variant not in variants:
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"unknown MTF variant in manifest: {variant}")
    block = variants[variant] or {}
    if variant == "mtf_full":
        rows = []
        for row in block.get("high_milestone_frames") or []:
            rows.append({**row, "objective": "demo_action_chunk", "loss_weight": 1.0})
        retention_weight = float(block.get("retention_coefficient", selected_manifest.get("retention_coefficient", 0.0)) or 0.0)
        for row in block.get("base_retention_frames") or []:
            rows.append({**row, "objective": "base_current_action_retention", "loss_weight": retention_weight})
        return sorted(rows, key=_event_sort_key)
    if variant == "mtf_no_retention_ablation":
        return sorted(
            ({**row, "objective": "demo_action_chunk", "loss_weight": 1.0} for row in block.get("high_milestone_frames") or []),
            key=_event_sort_key,
        )
    return sorted(
        ({**row, "objective": "demo_action_chunk", "loss_weight": 1.0} for row in block.get("selected_frames") or []),
        key=_event_sort_key,
    )


def _variant_checkpoint_path(output_root: str | Path, config_id: str, variant: str, seed: int) -> Path:
    return Path(output_root) / str(config_id) / str(variant) / f"seed_{int(seed)}"


def build_training_jobs(
    *,
    selected_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    stable_artifact: Mapping[str, Any],
    train_args: MTFTrainArgs | None = None,
) -> dict[str, Any]:
    """Build a checkable, test-free MTF adapter-training plan."""

    args = train_args or MTFTrainArgs()
    invalid_variants = [variant for variant in args.variants if variant not in VARIANT_ORDER]
    if invalid_variants:
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"invalid requested variants: {invalid_variants}")
    if int(args.steps) < 1 or int(args.steps) > MAX_STEPS:
        raise MTFTrainingError("TOO_HEAVY_LOCAL", f"MTF adapter steps must be in [1, {MAX_STEPS}], got {args.steps}")
    if str(selected_manifest.get("proposal_hash")) != PROPOSAL_HASH:
        raise MTFTrainingError("DESIGN_FAILURE", "selected training manifest proposal hash does not match frozen MTF proposal")
    if bool(selected_manifest.get("confirmatory_test_identities_used")):
        raise MTFTrainingError("DESIGN_FAILURE", "selected training manifest reports confirmatory-test identity usage")

    official_by_global = _official_samples(split_manifest)
    stable_by_global = _stable_records(stable_artifact)
    config_id = str(selected_manifest.get("config_id") or "unknown_config")
    jobs = []
    hard_stop_reasons: list[str] = []
    train_frame_identities: set[tuple[int, int, int]] = set()
    validation_frame_identities: set[tuple[int, int, int]] = {
        _frame_identity(sample) for sample in official_by_global.values() if str(sample.get("split")) == "val"
    }
    test_frame_identities: set[tuple[int, int, int]] = {
        _frame_identity(sample) for sample in official_by_global.values() if str(sample.get("split")) == "test"
    }

    for variant in args.variants:
        rows = _rows_for_variant(selected_manifest, variant)
        events = []
        for ordinal, row in enumerate(rows):
            dataset_global_index = int(row["dataset_global_index"])
            official = official_by_global.get(dataset_global_index)
            stable = stable_by_global.get(dataset_global_index)
            if official is None:
                hard_stop_reasons.append(f"{variant}: missing official split row for dataset_global_index {dataset_global_index}")
                continue
            if stable is None:
                hard_stop_reasons.append(f"{variant}: missing stable artifact row for dataset_global_index {dataset_global_index}")
                continue
            if str(official.get("split")) != "train" or str(stable.get("split")) != "train" or str(row.get("split")) != "train":
                hard_stop_reasons.append(f"{variant}: non-train frame selected for adapter training: {dataset_global_index}")
                continue
            objective = str(row["objective"])
            event = {
                "event_id": f"{variant}:{objective}:{dataset_global_index}:{ordinal}",
                "variant": variant,
                "objective": objective,
                "loss_weight": float(row["loss_weight"]),
                "dataset_global_index": dataset_global_index,
                "sample_id": str(official["sample_id"]),
                "split": "train",
                "task": str(official["task"]),
                "task_index": int(official["task_index"]),
                "episode_index": int(official["episode_index"]),
                "episode_length": int(official["episode_length"]),
                "frame_index": int(official["frame_index"]),
                "phase": str(official["phase"]),
                "normalized_phase": float(official["normalized_phase"]),
                "score": float(row.get("score", 0.0)),
                "target_action": _as_action7("target_action", stable["target_action"]),
            }
            if objective == "base_current_action_retention":
                event["base_action"] = _as_action7("base_action", stable["base_action"])
                event["retention_target_scope"] = "current 7D action only; future chunk remains demonstration target because full base action chunks are not persisted"
            events.append(event)
            train_frame_identities.add(_frame_identity(event))
        if not events:
            hard_stop_reasons.append(f"{variant}: no usable train events")
        event_digest = _hash_payload(events)
        checkpoint_path = _variant_checkpoint_path(args.checkpoint_output_root, config_id, variant, args.seed)
        jobs.append(
            {
                "variant": variant,
                "config_id": config_id,
                "seed": int(args.seed),
                "steps": int(args.steps),
                "learning_rate": float(args.lr),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
                "event_count": len(events),
                "demo_action_chunk_event_count": sum(event["objective"] == "demo_action_chunk" for event in events),
                "base_current_action_retention_event_count": sum(event["objective"] == "base_current_action_retention" for event in events),
                "unique_task_count": len({int(event["task_index"]) for event in events}),
                "unique_episode_count": len({int(event["episode_index"]) for event in events}),
                "event_digest": event_digest,
                "events": events,
            }
        )

    split_overlap = {
        "train_validation": len(train_frame_identities & validation_frame_identities),
        "train_test": len(train_frame_identities & test_frame_identities),
        "validation_test": len(validation_frame_identities & test_frame_identities),
    }
    if split_overlap["train_validation"] or split_overlap["train_test"] or split_overlap["validation_test"]:
        hard_stop_reasons.append(f"split frame overlap nonzero: {split_overlap}")
    result = {
        "schema_version": TRAINING_REPORT_VERSION,
        "date": DATE_KST,
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": config_id,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "learning_rate": float(args.lr),
        "variant_order": list(args.variants),
        "jobs": jobs,
        "split_overlap": split_overlap,
        "validation_frame_count_for_post_training_eval": len(validation_frame_identities),
        "confirmatory_test_identities_used": False,
        "closed_loop_experiment_happened": False,
        "stage_a_allowed": False,
        "retention_target_implementation": {
            "uses_frozen_base_action": True,
            "scope": "current 7D action on retention frames",
            "not_used": "KL between deterministic actions",
            "known_limitation": "full frozen-base action chunks are unavailable in the stable prediction artifact, so retention overrides only the current action before the official preprocessor builds the native loss target",
        },
        "hard_stop_reasons": hard_stop_reasons,
    }
    result["final_decision"] = "MTF_ADAPTER_TRAINING_PLAN_READY" if not hard_stop_reasons else "MTF_ADAPTER_TRAINING_PLAN_BLOCKED"
    result["next_step"] = (
        "Run the MTF adapter trainer to produce disk-reloadable checkpoints for all four trainable Stage A policies."
        if not hard_stop_reasons
        else "Do not train; fix the listed manifest/data failures or archive MTF as an implementation/data failure."
    )
    return result


def _assign_dataset_local_indices(samples: Sequence[Mapping[str, Any]]) -> tuple[list[int], dict[int, dict[str, Any]]]:
    lengths: dict[int, int] = {}
    for sample in samples:
        episode = int(sample["episode_index"])
        lengths[episode] = max(lengths.get(episode, 0), int(sample["episode_length"]))
    selected_episodes = sorted(lengths)
    offsets: dict[int, int] = {}
    offset = 0
    for episode in selected_episodes:
        offsets[episode] = offset
        offset += int(lengths[episode])
    out: dict[int, dict[str, Any]] = {}
    for sample in samples:
        cloned = dict(sample)
        cloned["dataset_local_index"] = int(offsets[int(sample["episode_index"])] + int(sample["frame_index"]))
        out[int(sample["dataset_global_index"])] = cloned
    return selected_episodes, out


def _override_current_action(raw_sample: Mapping[str, Any], base_action: Sequence[float]) -> dict[str, Any]:
    sample = dict(raw_sample)
    action = sample.get("action")
    if action is None:
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", "retention sample has no action tensor")
    base = list(float(x) for x in base_action)
    if hasattr(action, "clone"):
        action_copy = action.clone()
        import torch

        replacement = torch.as_tensor(base, dtype=action_copy.dtype, device=action_copy.device)
        if int(action_copy.ndim) == 2:
            width = min(int(action_copy.shape[-1]), replacement.numel())
            action_copy[0, :width] = replacement[:width]
        elif int(action_copy.ndim) == 1:
            width = min(int(action_copy.shape[0]), replacement.numel())
            action_copy[:width] = replacement[:width]
        else:
            raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"unexpected action ndim for retention override: {action_copy.ndim}")
        sample["action"] = action_copy
        return sample
    array = np.asarray(action, dtype=np.float32).copy()
    if array.ndim == 2:
        array[0, : min(array.shape[-1], len(base))] = np.asarray(base[: array.shape[-1]], dtype=array.dtype)
    elif array.ndim == 1:
        array[: min(array.shape[0], len(base))] = np.asarray(base[: array.shape[0]], dtype=array.dtype)
    else:
        raise MTFTrainingError("IMPLEMENTATION_OR_OPTIMIZATION_FAILURE", f"unexpected action ndim for retention override: {array.ndim}")
    sample["action"] = array
    return sample


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _load_validation_scored_records(stable_artifact: Mapping[str, Any], retained_ratio: float) -> dict[tuple[int, int, int], dict[str, Any]]:
    config = MTFConfig(retained_ratio=float(retained_ratio))
    scored = compute_mtf_scores(build_score_records(stable_artifact.get("records") or []), config)
    return {
        (int(record["episode_index"]), int(record["frame_index"]), int(record["task_index"])): record
        for record in scored
        if str(record.get("split")) == "val"
    }


def _delta_summary(adapter_rows: Sequence[Mapping[str, Any]], stable_records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    all_l2: list[float] = []
    retention_l2: list[float] = []
    translation: list[float] = []
    rotation: list[float] = []
    gripper: list[float] = []
    for row in adapter_rows:
        stable = stable_records[int(row["dataset_global_index"])]
        pred = np.asarray(row["pred_preview"], dtype=np.float64)
        base = np.asarray(stable["base_action"], dtype=np.float64)
        diff = pred[:7] - base[:7]
        value = float(np.linalg.norm(diff))
        all_l2.append(value)
        translation.append(float(np.linalg.norm(diff[:3])))
        rotation.append(float(np.linalg.norm(diff[3:6])))
        gripper.append(float(abs(diff[6])))
        if bool(row.get("mtf_validation_retention_frame")):
            retention_l2.append(value)

    def stat(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": _round(float(np.mean(array))),
            "p95": _round(float(np.percentile(array, 95))),
            "max": _round(float(np.max(array))),
        }

    return {
        "adapter_minus_base_action_l2": stat(all_l2),
        "adapter_minus_base_retention_frame_l2": stat(retention_l2),
        "translation_l2": stat(translation),
        "rotation_l2": stat(rotation),
        "gripper_abs": stat(gripper),
    }


def _bundle_file_hashes(path: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.name != "sha256_manifest.json":
            relative = child.relative_to(path).as_posix()
            files[relative] = {"sha256": _sha256_file(child), "size_bytes": child.stat().st_size}
    return files


def _is_complete_bundle(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_BUNDLE_FILES)


def _save_checkpoint_bundle(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: Any,
    args: argparse.Namespace,
    job: Mapping[str, Any],
    checkpoint_path: Path,
    dataset_root: Path,
    lora_param_summary: Mapping[str, Any],
    loss_curve: list[dict[str, Any]],
    grad_curve: list[dict[str, Any]],
    train_order: list[int],
    training_elapsed: float,
    device_audit: Mapping[str, Any],
) -> dict[str, Any]:
    seed_dir = Path(job["checkpoint_path"])
    if seed_dir.exists():
        if _is_complete_bundle(seed_dir) and not bool(args.force):
            raise MTFTrainingError("CHECKPOINT_IDENTITY_UNPROVEN", f"Refusing to overwrite complete MTF checkpoint bundle: {seed_dir}")
        if not bool(args.force):
            raise MTFTrainingError("CHECKPOINT_BUNDLE_INCOMPLETE", f"Target MTF checkpoint directory exists: {seed_dir}")
        shutil.rmtree(seed_dir)
    seed_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = seed_dir.with_name(f"{seed_dir.name}.tmp_{os.getpid()}_{time.time_ns()}")
    tmp_dir.mkdir(parents=True)
    try:
        if hasattr(policy, "peft_config"):
            for peft_config in policy.peft_config.values():
                peft_config.base_model_name_or_path = str(checkpoint_path)
        policy.save_pretrained(tmp_dir)
        if hasattr(policy, "config") and hasattr(policy.config, "save_pretrained"):
            policy.config.save_pretrained(tmp_dir)
        if hasattr(preprocessor, "save_pretrained"):
            preprocessor.save_pretrained(tmp_dir)
        if hasattr(postprocessor, "save_pretrained"):
            postprocessor.save_pretrained(tmp_dir)

        source_lock = Path(args.source_repro_lock)
        if source_lock.exists():
            shutil.copy2(source_lock, tmp_dir / "source_repro_lock.yaml")
        else:
            (tmp_dir / "source_repro_lock.yaml").write_text("missing_source_repro_lock: true\n", encoding="utf-8")

        import torch

        torch.save(
            {
                "python_random_state_repr": repr(random.getstate()),
                "numpy_random_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "data_order_seed": int(job["seed"]),
                "train_order_first_20": train_order[:20],
            },
            tmp_dir / "rng_state.pt",
        )
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer_state.pt")
        _write_json(
            tmp_dir / "training_manifest.json",
            {
                "schema_version": TRAINING_REPORT_VERSION,
                "method": "MTF-VLA",
                "proposal_hash": PROPOSAL_HASH,
                "status": "CHECKPOINT_TRAINED_SAVED_PENDING_RELOAD",
                "variant": str(job["variant"]),
                "config_id": str(job["config_id"]),
                "seed": int(job["seed"]),
                "lora_rank": 4,
                "training_step_count": int(job["steps"]),
                "batch_size": 1,
                "learning_rate": float(job["learning_rate"]),
                "train_event_count": int(job["event_count"]),
                "demo_action_chunk_event_count": int(job["demo_action_chunk_event_count"]),
                "base_current_action_retention_event_count": int(job["base_current_action_retention_event_count"]),
                "event_digest": str(job["event_digest"]),
                "retention_target_scope": "current 7D base action only on retention events",
                "confirmatory_test_identities_used": False,
                "closed_loop_experiment_happened": False,
                "stage_a_allowed": False,
                "trainable_parameter_count": lora_param_summary.get("trainable_params"),
                "total_parameter_count": lora_param_summary.get("total_params"),
                "base_model": {
                    "repo_id": "lerobot/smolvla_libero",
                    "revision": args.expected_model_revision,
                    "local_path": str(checkpoint_path),
                },
                "dataset": {
                    "repo_id": "lerobot/libero",
                    "revision": args.expected_dataset_revision,
                    "local_path": str(dataset_root),
                },
                "locked_artifact_hashes": _hash_paths(
                    [args.source_repro_lock, args.split_manifest, args.selected_training_manifest, args.stable_artifact]
                ),
                "git_commit_used_for_training": _git_head(),
                "exact_training_command": " ".join([sys.executable, "-m", "tca_map.smolvla.mtf_vla_training", *sys.argv[1:]]),
                "package_versions": _package_versions(),
                "loss_before": loss_curve[0]["loss"] if loss_curve else None,
                "loss_after": loss_curve[-1]["loss"] if loss_curve else None,
                "loss_curve": loss_curve,
                "last_gradient_summary": grad_curve[-1] if grad_curve else None,
                "training_elapsed_sec": _round(training_elapsed, 3),
                "device_audit": dict(device_audit),
                "warnings": [
                    "Retention target uses the persisted frozen-base current 7D action, not an unavailable full base action chunk.",
                ],
            },
        )
        _write_json(
            tmp_dir / "eval_preprocessor_postprocessor_refs.json",
            {
                "preprocessor_saved": hasattr(preprocessor, "save_pretrained"),
                "postprocessor_saved": hasattr(postprocessor, "save_pretrained"),
                "preprocessor_config": "policy_preprocessor.json",
                "postprocessor_config": "policy_postprocessor.json",
                "official_device_processor": "cuda",
                "custom_normalizer_involved": False,
                "custom_action_adapter_involved": False,
            },
        )
        _write_json(
            tmp_dir / "sha256_manifest.json",
            {
                "schema_version": TRAINING_REPORT_VERSION,
                "method": "MTF-VLA",
                "variant": str(job["variant"]),
                "seed": int(job["seed"]),
                "bundle_root": str(seed_dir),
                "files": _bundle_file_hashes(tmp_dir),
            },
        )
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (tmp_dir / name).is_file()]
        if missing:
            raise MTFTrainingError("CHECKPOINT_BUNDLE_INCOMPLETE", f"missing MTF checkpoint bundle files: {missing}")
        tmp_dir.rename(seed_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return {
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "status": "CHECKPOINT_SAVED_PENDING_RELOAD",
        "checkpoint_path": str(seed_dir),
        "required_files": list(REQUIRED_BUNDLE_FILES),
        "file_hashes": _bundle_file_hashes(seed_dir),
        "adapter_model_sha256": _sha256_file(seed_dir / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256_file(seed_dir / "adapter_config.json"),
    }


def _train_one_variant(
    *,
    args: argparse.Namespace,
    job: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    stable_artifact: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    if not torch.cuda.is_available():
        raise MTFTrainingError("CPU_FALLBACK_BUG", "CUDA unavailable; refusing MTF adapter training on CPU.")
    torch.manual_seed(int(job["seed"]))
    np.random.seed(int(job["seed"]))
    random.seed(int(job["seed"]))
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    fps = float(info.get("fps", 10.0))
    chunk_size = int(args.chunk_size)
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
    action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)

    official_by_global = _official_samples(split_manifest)
    validation_samples = [sample for sample in official_by_global.values() if str(sample.get("split")) == "val"]
    event_samples = list(job.get("events") or [])
    selected_episodes, local_by_global = _assign_dataset_local_indices([*event_samples, *validation_samples])
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    runtime_events = [{**event, "dataset_local_index": local_by_global[int(event["dataset_global_index"])]["dataset_local_index"]} for event in event_samples]
    runtime_validation = [
        {**sample, "dataset_local_index": local_by_global[int(sample["dataset_global_index"])]["dataset_local_index"]}
        for sample in validation_samples
    ]

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
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    probe = _add_training_batch_dims(preprocessor(dataset[int(runtime_events[0]["dataset_local_index"])]))
    input_devices = _tensor_devices(probe)
    param_summary = _parameter_summary(policy)
    if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
        value.startswith("cuda") for value in input_devices.values()
    ):
        raise MTFTrainingError("CPU_FALLBACK_BUG", f"MTF params/inputs are not CUDA: params={param_summary}, inputs={input_devices}")

    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.train()
    lora_param_summary = _parameter_summary(policy)
    optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(job["learning_rate"]))
    rng = np.random.default_rng(int(job["seed"]))
    train_order = rng.permutation(len(runtime_events)).tolist()
    loss_curve: list[dict[str, Any]] = []
    grad_curve: list[dict[str, Any]] = []
    objective_counts: dict[str, int] = {}
    first_batch_devices: dict[str, str] | None = None
    first_batch_shapes: dict[str, list[int]] | None = None
    training_started = time.monotonic()
    for step in range(int(job["steps"])):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise MTFTrainingError("TOO_HEAVY_LOCAL", "MTF adapter training exceeded runtime cap.")
        event = runtime_events[train_order[step % len(train_order)]]
        raw_sample = dataset[int(event["dataset_local_index"])]
        if event["objective"] == "base_current_action_retention":
            raw_sample = _override_current_action(raw_sample, event["base_action"])
        batch = _add_training_batch_dims(preprocessor(raw_sample))
        devices = _tensor_devices(batch)
        if first_batch_devices is None:
            first_batch_devices = devices
            first_batch_shapes = _tensor_shapes(batch)
        if not all(value.startswith("cuda") for value in devices.values()):
            raise MTFTrainingError("CPU_FALLBACK_BUG", f"MTF training tensors are on CPU: {devices}")
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_output(policy.forward(batch))
        weighted_loss = loss * float(event["loss_weight"])
        loss_value = _to_float(weighted_loss)
        if not math.isfinite(loss_value):
            raise MTFTrainingError("TRAINING_FAILURE", f"non-finite MTF loss for {job['variant']} at step {step}: {loss_value}")
        weighted_loss.backward()
        grad_summary = _gradient_summary(policy)
        if int(grad_summary["nonzero_grad_tensors"]) == 0:
            raise MTFTrainingError("TRAINING_FAILURE", f"no nonzero MTF LoRA gradients for {job['variant']} at step {step}")
        optimizer.step()
        objective_counts[str(event["objective"])] = objective_counts.get(str(event["objective"]), 0) + 1
        cuda_now = _cuda_memory(torch)
        loss_curve.append(
            {
                "step": int(step),
                "objective": str(event["objective"]),
                "loss_weight": float(event["loss_weight"]),
                "loss": _round(loss_value),
                "allocated_mb": cuda_now["allocated_mb"],
                "max_allocated_mb": cuda_now["max_allocated_mb"],
            }
        )
        grad_curve.append({"step": int(step), **grad_summary})
    training_elapsed = time.monotonic() - training_started

    device_audit = {
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "model_parameter_device_before_peft": param_summary["first_parameter_device"],
        "model_parameter_dtype_before_peft": param_summary["first_parameter_dtype"],
        "model_parameter_device_after_peft": lora_param_summary["first_parameter_device"],
        "model_parameter_dtype_after_peft": lora_param_summary["first_parameter_dtype"],
        "input_tensor_devices": input_devices,
        "input_tensor_shapes": _tensor_shapes(probe),
        "first_training_batch_devices": first_batch_devices,
        "first_training_batch_shapes": first_batch_shapes,
        "autocast_status_initial_final": _safe_autocast_status(torch),
        "cuda_memory": _cuda_memory(torch),
    }
    checkpoint_bundle = _save_checkpoint_bundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        args=args,
        job=job,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        lora_param_summary=lora_param_summary,
        loss_curve=loss_curve,
        grad_curve=grad_curve,
        train_order=train_order,
        training_elapsed=training_elapsed,
        device_audit=device_audit,
    )

    del policy
    torch.cuda.empty_cache()

    reload_cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    reload_cfg.device = "cuda"
    reload_cfg.load_vlm_weights = True
    reload_cfg.compile_model = False
    reload_cfg.push_to_hub = False
    reload_cfg.vlm_model_name = str(vlm_root)
    if hasattr(reload_cfg, "chunk_size"):
        reload_cfg.chunk_size = chunk_size
    base_policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=reload_cfg,
        local_files_only=True,
        cache_dir=hf_home,
        token=False,
        strict=False,
    )
    peft_config = PeftConfig.from_pretrained(checkpoint_bundle["checkpoint_path"])
    loaded_policy = PeftModel.from_pretrained(
        base_policy,
        checkpoint_bundle["checkpoint_path"],
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    loaded_policy.to("cuda")
    loaded_policy.eval()
    reload_preprocessor, reload_postprocessor = make_pre_post_processors(
        reload_cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    reload_probe = _add_training_batch_dims(reload_preprocessor(dataset[int(runtime_validation[0]["dataset_local_index"])]))
    reload_devices = _tensor_devices(reload_probe)
    reload_param_summary = _parameter_summary(loaded_policy)
    if not str(reload_param_summary["first_parameter_device"]).startswith("cuda") or not all(
        value.startswith("cuda") for value in reload_devices.values()
    ):
        raise MTFTrainingError("CPU_FALLBACK_BUG", f"reloaded MTF adapter fell back to CPU: params={reload_param_summary}, inputs={reload_devices}")
    checkpoint_bundle["status"] = "CHECKPOINT_COMPLETE_VERIFIED"
    checkpoint_bundle["disk_reload"] = {
        "loaded_from_disk": True,
        "peft_config_base_model_name_or_path": getattr(peft_config, "base_model_name_or_path", None),
        "loaded_policy_type": type(loaded_policy).__name__,
        "model_parameter_device": reload_param_summary["first_parameter_device"],
        "model_parameter_dtype": reload_param_summary["first_parameter_dtype"],
        "input_tensor_devices": reload_devices,
        "input_tensor_shapes": _tensor_shapes(reload_probe),
        "custom_action_adapter_involved": False,
        "custom_normalizer_involved": False,
    }

    adapter_rows = _evaluate_policy_rows(
        policy=loaded_policy,
        preprocessor=reload_preprocessor,
        postprocessor=reload_postprocessor,
        dataset=dataset,
        samples=runtime_validation,
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=bool(args.include_eval_loss),
        label=f"mtf_{job['variant']}_seed_{job['seed']}",
        started=started,
        progress_every=int(args.progress_every),
    )
    stable_by_global = _stable_records(stable_artifact)
    scored_val = _load_validation_scored_records(stable_artifact, retained_ratio=float(args.retained_ratio))
    for row in adapter_rows:
        score_row = scored_val.get(_row_key(row))
        row["mtf_validation_high_milestone"] = bool(score_row and score_row.get("high_milestone"))
        row["mtf_validation_retention_frame"] = bool(score_row and score_row.get("retention_frame"))
        stable = stable_by_global[int(row["dataset_global_index"])]
        row["base_action_preview"] = _round_vector(stable["base_action"], 9)
        row["adapter_minus_base_action_l2"] = _round(
            float(np.linalg.norm(np.asarray(row["pred_preview"], dtype=np.float64) - np.asarray(stable["base_action"], dtype=np.float64)))
        )
    validation_metric = _metric_with_balance(adapter_rows, seed=int(job["seed"]))
    retention_rows = [row for row in adapter_rows if bool(row.get("mtf_validation_retention_frame"))]
    high_rows = [row for row in adapter_rows if bool(row.get("mtf_validation_high_milestone"))]
    result = {
        "schema_version": TRAINING_REPORT_VERSION,
        "date": DATE_KST,
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED",
        "variant": str(job["variant"]),
        "config_id": str(job["config_id"]),
        "seed": int(job["seed"]),
        "closed_loop_experiment_happened": False,
        "confirmatory_test_identities_used": False,
        "stage_a_allowed": False,
        "checkpoint_bundle": checkpoint_bundle,
        "training": {
            "steps": int(job["steps"]),
            "learning_rate": float(job["learning_rate"]),
            "event_count": int(job["event_count"]),
            "objective_counts_seen": objective_counts,
            "loss_before": loss_curve[0]["loss"] if loss_curve else None,
            "loss_after": loss_curve[-1]["loss"] if loss_curve else None,
            "loss_curve": loss_curve,
            "last_gradient_summary": grad_curve[-1] if grad_curve else None,
            "training_elapsed_sec": _round(training_elapsed, 3),
            "steps_per_sec": _round(len(loss_curve) / max(training_elapsed, 1e-12), 6),
        },
        "validation": {
            "split": "val",
            "record_count": len(adapter_rows),
            "metric": validation_metric,
            "retention_frame_metric": summarize_rows(retention_rows) if retention_rows else {"count": 0},
            "high_milestone_metric": summarize_rows(high_rows) if high_rows else {"count": 0},
            "action_delta_from_base": _delta_summary(adapter_rows, stable_by_global),
        },
        "device_audit": {
            "training": device_audit,
            "reload": checkpoint_bundle["disk_reload"],
            "cuda_memory": _cuda_memory(torch),
        },
        "records": adapter_rows,
        "runtime": {
            "elapsed_sec": _round(time.monotonic() - started, 3),
            "rss_final_mb": _rss_mb(),
        },
    }
    _write_json(Path(args.report_dir) / f"{job['variant']}_seed_{job['seed']}_training_result.json", result)
    return result


def _summarize_variant_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        metric = ((result.get("validation") or {}).get("metric") or {})
        delta = (((result.get("validation") or {}).get("action_delta_from_base") or {}).get("adapter_minus_base_action_l2") or {})
        rows.append(
            {
                "variant": result.get("variant"),
                "seed": result.get("seed"),
                "final_decision": result.get("final_decision"),
                "checkpoint_path": (((result.get("checkpoint_bundle") or {}).get("checkpoint_path"))),
                "validation_action_l2_mean": metric.get("action_l2_mean"),
                "validation_task_balanced_action_l2_mean": metric.get("task_balanced_action_l2_mean"),
                "adapter_minus_base_action_l2_p95": delta.get("p95"),
                "disk_reload": (((result.get("checkpoint_bundle") or {}).get("disk_reload") or {}).get("loaded_from_disk")),
            }
        )
    return {"variant_count": len(rows), "rows": rows}


def _all_stage_a_variants_verified(results: Sequence[Mapping[str, Any]]) -> bool:
    verified = {
        str(result.get("variant"))
        for result in results
        if str(result.get("final_decision")) == "MTF_ADAPTER_CHECKPOINT_VERIFIED"
    }
    return verified == set(VARIANT_ORDER)


def build_or_run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    selected_manifest = _read_json(Path(args.selected_training_manifest))
    split_manifest = _read_json(Path(args.split_manifest))
    stable_artifact = _read_json(Path(args.stable_artifact))
    variants = tuple(part.strip() for part in str(args.variants).split(",") if part.strip())
    plan = build_training_jobs(
        selected_manifest=selected_manifest,
        split_manifest=split_manifest,
        stable_artifact=stable_artifact,
        train_args=MTFTrainArgs(
            steps=int(args.steps),
            seed=int(args.seed),
            lr=float(args.lr),
            variants=variants,
            checkpoint_output_root=str(args.checkpoint_output_root),
        ),
    )
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    _write_json(Path(args.plan_output), {key: value for key, value in plan.items() if key != "jobs"} | {"jobs": [_job_without_events(job) for job in plan["jobs"]]})
    if args.dry_run or plan["final_decision"] != "MTF_ADAPTER_TRAINING_PLAN_READY":
        report = {
            **{key: value for key, value in plan.items() if key != "jobs"},
            "jobs": [_job_without_events(job) for job in plan["jobs"]],
            "dry_run": bool(args.dry_run),
            "training_happened": False,
            "final_decision": plan["final_decision"],
            "runtime": {"elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()},
        }
        _write_json(Path(args.report_json), report)
        return report, 0 if plan["final_decision"] == "MTF_ADAPTER_TRAINING_PLAN_READY" else 40

    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        raise MTFTrainingError("TRAINING_FAILURE", "Forbidden gate(s) set: " + ", ".join(forbidden))

    results = []
    try:
        for job in plan["jobs"]:
            print(f"[mtf-training] training {job['variant']} seed {job['seed']} for {job['steps']} steps", flush=True)
            result = _train_one_variant(
                args=args,
                job=job,
                split_manifest=split_manifest,
                stable_artifact=stable_artifact,
                started=started,
            )
            results.append(result)
            progress = {
                "date": DATE_KST,
                "method": "MTF-VLA",
                "status": "in_progress",
                "completed_variants": [str(item.get("variant")) for item in results],
                "remaining_variants": [str(item["variant"]) for item in plan["jobs"][len(results) :]],
            }
            _write_json(Path(args.progress_json), progress)
        all_stage_a_variants_verified = _all_stage_a_variants_verified(results)
        report = {
            **{key: value for key, value in plan.items() if key != "jobs"},
            "jobs": [_job_without_events(job) for job in plan["jobs"]],
            "dry_run": False,
            "training_happened": True,
            "closed_loop_experiment_happened": False,
            "confirmatory_test_identities_used": False,
            "stage_a_allowed": all_stage_a_variants_verified,
            "variant_results": _summarize_variant_results(results),
            "final_decision": (
                "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY"
                if all_stage_a_variants_verified
                else "MTF_PARTIAL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_BLOCKED"
            ),
            "next_step": (
                "Freeze a matched Stage A rollout manifest before any rollout; do not tune these checkpoints on confirmatory outcomes."
                if all_stage_a_variants_verified
                else "Train and disk-reload verify the remaining MTF Stage A adapter policies before any rollout."
            ),
            "runtime": {"elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()},
        }
        _write_json(Path(args.report_json), report)
        _write_json(
            Path(args.progress_json),
            {
                "date": DATE_KST,
                "method": "MTF-VLA",
                "status": "completed",
                "completed_variants": [str(item.get("variant")) for item in results],
                "remaining_variants": [],
                "final_decision": report["final_decision"],
                "stage_a_allowed": report["stage_a_allowed"],
            },
        )
        return report, 0
    except MTFTrainingError as exc:
        report = {
            **{key: value for key, value in plan.items() if key != "jobs"},
            "jobs": [_job_without_events(job) for job in plan["jobs"]],
            "dry_run": False,
            "training_happened": bool(results),
            "closed_loop_experiment_happened": False,
            "confirmatory_test_identities_used": False,
            "stage_a_allowed": False,
            "variant_results": _summarize_variant_results(results),
            "final_decision": exc.code,
            "errors": [{"code": exc.code, "message": str(exc)}],
            "runtime": {"elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()},
        }
        _write_json(Path(args.report_json), report)
        return report, 41
    except Exception as exc:  # pragma: no cover - runtime reporting boundary
        report = {
            **{key: value for key, value in plan.items() if key != "jobs"},
            "jobs": [_job_without_events(job) for job in plan["jobs"]],
            "dry_run": False,
            "training_happened": bool(results),
            "closed_loop_experiment_happened": False,
            "confirmatory_test_identities_used": False,
            "stage_a_allowed": False,
            "variant_results": _summarize_variant_results(results),
            "final_decision": "TRAINING_FAILURE",
            "errors": [{"code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}],
            "runtime": {"elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()},
        }
        _write_json(Path(args.report_json), report)
        return report, 42


def _job_without_events(job: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "events"}


def _write_training_md(report: Mapping[str, Any], path: Path) -> None:
    rows = (report.get("variant_results") or {}).get("rows") or []
    lines = [
        "# MTF-VLA Adapter Training",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- dry run: `{report.get('dry_run')}`",
        f"- training happened: `{report.get('training_happened')}`",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- confirmatory-test identities used: `{report.get('confirmatory_test_identities_used')}`",
        f"- Stage A allowed: `{report.get('stage_a_allowed')}`",
        f"- config: `{report.get('config_id')}`",
        f"- seed: `{report.get('seed')}`",
        f"- steps: `{report.get('steps')}`",
        "",
        "Retention target implementation:",
        "",
        "```json",
        json.dumps(report.get("retention_target_implementation"), indent=2, sort_keys=True),
        "```",
        "",
        "Jobs:",
        "",
        "| variant | events | demo | retention | tasks | episodes | checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for job in report.get("jobs") or []:
        lines.append(
            f"| `{job.get('variant')}` | {job.get('event_count')} | {job.get('demo_action_chunk_event_count')} | "
            f"{job.get('base_current_action_retention_event_count')} | {job.get('unique_task_count')} | "
            f"{job.get('unique_episode_count')} | `{job.get('checkpoint_path')}` |"
        )
    if rows:
        lines.extend(
            [
                "",
                "Verified checkpoints:",
                "",
                "| variant | disk reload | validation action L2 | task-balanced action L2 | adapter-base p95 | checkpoint |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row.get('variant')}` | `{row.get('disk_reload')}` | {row.get('validation_action_l2_mean')} | "
                f"{row.get('validation_task_balanced_action_l2_mean')} | {row.get('adapter_minus_base_action_l2_p95')} | "
                f"`{row.get('checkpoint_path')}` |"
            )
    reasons = list(report.get("hard_stop_reasons") or [])
    lines.extend(["", "Hard stop reasons:"])
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(["", f"Next step: {report.get('next_step')}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-training-manifest", default="reports/mtf_vla/selected_training_manifest.json")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--source-repro-lock", default="configs/official_smolvla_repro_lock.yaml")
    parser.add_argument("--checkpoint-output-root", default="runs/mtf_vla_checkpoints")
    parser.add_argument("--report-dir", default="reports/mtf_vla/training")
    parser.add_argument("--report-json", default="reports/mtf_vla/adapter_training_result.json")
    parser.add_argument("--report-md", default="reports/mtf_vla/adapter_training_result.md")
    parser.add_argument("--plan-output", default="reports/mtf_vla/adapter_training_plan.json")
    parser.add_argument("--progress-json", default="reports/mtf_vla/adapter_training_progress.json")
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--retained-ratio", type=float, default=0.20)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--include-eval-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-model-revision", default="31d453f7edd78c839a8bbc39744a292686daf0de")
    parser.add_argument("--expected-dataset-revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report, exit_code = build_or_run(args)
    _write_training_md(report, Path(args.report_md))
    print(json.dumps({"final_decision": report.get("final_decision"), "training_happened": report.get("training_happened"), "stage_a_allowed": report.get("stage_a_allowed")}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
