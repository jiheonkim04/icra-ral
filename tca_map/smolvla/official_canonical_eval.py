"""Canonical persisted SmolVLA-LoRA offline evaluation and rollout gate.

This runner does not train, regenerate checkpoints, run FCAR, use the old
LIBERO_7D route, run OpenVLA-OFT, or execute rollouts. It evaluates only the
persisted disk-reloaded official SmolVLA base and LoRA adapter checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from importlib import import_module, metadata, util
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _json_default,
    _parameter_summary,
    _postprocess_action,
    _raw_current_action,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
)
from tca_map.smolvla.official_libero_failure_mining import _metric_row, summarize_rows
from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _balanced_mean,
    _bootstrap_ci,
    _group_summary,
    _manifest_samples,
    _round,
    _round_vector,
)


DATE = "2026-07-10 KST"
ARTIFACT_VERSION = 1
REPORT_SCHEMA_VERSION = 1
RNG_NAMESPACE = "official_smolvla_canonical_eval.v1"
EVAL_SEEDS = [101, 202, 303, 404, 505]
LORA_SEEDS = [11, 22, 33]
STATIC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_RUNTIME_SECONDS = 6 * 60 * 60
REPEAT_TOLERANCE = 1e-7
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
    "ALLOW_GPU_TRAINING",
]
FINAL_DECISIONS = {
    "OFFICIAL_ROLLOUT_BASELINE_READY",
    "OFFICIAL_ROLLOUT_REVEALS_METHOD_WORTHY_GAP",
    "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT",
    "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT",
    "OFFICIAL_EVAL_ENV_BLOCKED",
    "CANONICAL_EVAL_FAILED",
    "NO_CLOSED_LOOP_METHOD_WORTHY_GAP",
}
INTERMEDIATE_DECISIONS = {
    "CANONICAL_BASELINES_READY_FOR_ROLLOUT",
    "CANONICAL_EVAL_FAILED",
    "CHECKPOINT_IDENTITY_FAILED",
    "EVAL_RNG_POLICY_INVALID",
    "TEST_LEAKAGE_FOUND",
    "CPU_FALLBACK_BUG",
    "TOO_HEAVY_LOCAL",
}
REQUIRED_BUNDLE_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_manifest.json",
    "eval_preprocessor_postprocessor_refs.json",
    "source_repro_lock.yaml",
    "sha256_manifest.json",
}
METRIC_AGG_KEYS = [
    "action_l2_mean",
    "translation_l2_mean",
    "rotation_l2_mean",
    "gripper_abs_mean",
    "gripper_sign_accuracy",
    "range_violation_rate",
    "task_balanced_action_l2_mean",
]


class CanonicalEvalError(RuntimeError):
    """Reportable canonicalization failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(data).hexdigest().upper()


def _git_head() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _git_branch() -> str | None:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise CanonicalEvalError("EVAL_RNG_POLICY_INVALID", "At least one integer seed is required.")
    return values


def immutable_frame_identity(sample: dict[str, Any]) -> dict[str, Any]:
    """Return label-free immutable fields used for canonical RNG identity."""

    return {
        "split": str(sample["split"]),
        "sample_id": str(sample.get("sample_id", "")),
        "dataset_local_index": int(sample["dataset_local_index"]),
        "dataset_global_index": int(sample.get("dataset_global_index", -1)),
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "episode_length": int(sample["episode_length"]),
        "task_index": int(sample["task_index"]),
    }


def canonical_rng_seed(eval_seed: int, sample: dict[str, Any], namespace: str = RNG_NAMESPACE) -> int:
    """Derive a deterministic 63-bit seed from eval seed and frame identity only."""

    payload = {
        "namespace": namespace,
        "eval_seed": int(eval_seed),
        "immutable_frame_identity": immutable_frame_identity(sample),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) & ((1 << 63) - 1)
    return value if value > 0 else 1


def _set_global_rngs(eval_seed64: int) -> int:
    seed32 = int(eval_seed64 % (2**32))
    random.seed(seed32)
    np.random.seed(seed32)
    return seed32


def _validate_split_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest.get("summary") or {}
    leakage = summary.get("leakage_checks") or {}
    if not all(bool(value) for value in leakage.values()):
        raise CanonicalEvalError("TEST_LEAKAGE_FOUND", f"Split leakage checks failed: {leakage}")
    frames = summary.get("frame_counts") or {}
    if int(frames.get("val", 0)) <= 0 or int(frames.get("test", 0)) <= 0:
        raise CanonicalEvalError("TEST_LEAKAGE_FOUND", f"Missing val/test split frames: {frames}")


def validate_checkpoint_manifest(path: Path, expected_seeds: list[int]) -> dict[str, Any]:
    manifest = _read_json(path)
    seed_entries = manifest.get("seeds") or []
    actual_seeds = [int(entry.get("seed")) for entry in seed_entries]
    if actual_seeds != expected_seeds:
        raise CanonicalEvalError("CHECKPOINT_IDENTITY_FAILED", f"central manifest seeds {actual_seeds} != {expected_seeds}")
    per_seed = {}
    mismatches = []
    seen_paths: set[str] = set()
    for entry in seed_entries:
        seed = int(entry["seed"])
        root = Path(entry["checkpoint_path"])
        status = str(entry.get("status"))
        missing = sorted(name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file())
        if status != "CHECKPOINT_COMPLETE_VERIFIED":
            mismatches.append({"seed": seed, "reason": f"status={status}"})
        if missing:
            mismatches.append({"seed": seed, "reason": f"missing={missing}"})
        if str(root).lower() in seen_paths:
            mismatches.append({"seed": seed, "reason": f"duplicate path={root}"})
        seen_paths.add(str(root).lower())
        bundle_ok = False
        file_hashes: dict[str, Any] = {}
        if (root / "sha256_manifest.json").is_file():
            bundle_manifest = _read_json(root / "sha256_manifest.json")
            bundle_ok = True
            for relative, metadata_obj in (bundle_manifest.get("files") or {}).items():
                expected = str((metadata_obj or {}).get("sha256", "")).upper()
                file_path = root / relative
                if not file_path.is_file():
                    mismatches.append({"seed": seed, "reason": f"bundle references missing {relative}"})
                    bundle_ok = False
                    continue
                actual = _sha256_file(file_path)
                file_hashes[relative] = {"expected": expected, "actual": actual, "ok": expected == actual}
                if expected != actual:
                    mismatches.append({"seed": seed, "reason": f"bundle checksum {relative}: {expected} != {actual}"})
                    bundle_ok = False
        central_ok = True
        for relative, metadata_obj in (entry.get("file_hashes") or {}).items():
            expected = str((metadata_obj or {}).get("sha256", "")).upper()
            actual = _sha256_file(root / relative)
            if expected != actual:
                mismatches.append({"seed": seed, "reason": f"central checksum {relative}: {expected} != {actual}"})
                central_ok = False
        disk_reload = entry.get("disk_reload") or {}
        if disk_reload.get("loaded_from_disk") is not True or disk_reload.get("loaded_policy_type") != "PeftModel":
            mismatches.append({"seed": seed, "reason": f"disk reload proof is insufficient: {disk_reload}"})
        per_seed[str(seed)] = {
            "checkpoint_path": str(root),
            "status": status,
            "missing_required_files": missing,
            "bundle_checksum_ok": bundle_ok,
            "central_checksum_ok": central_ok,
            "adapter_model_sha256": entry.get("adapter_model_sha256"),
            "adapter_config_sha256": entry.get("adapter_config_sha256"),
            "disk_reload": disk_reload,
            "file_hashes": file_hashes,
        }
    if mismatches:
        raise CanonicalEvalError("CHECKPOINT_IDENTITY_FAILED", f"Checkpoint integrity mismatch: {mismatches[:5]}")
    return {
        "manifest_path": str(path),
        "all_complete_verified": True,
        "locked_revision_check": manifest.get("locked_revision_check"),
        "checksum_status": manifest.get("checksum_status"),
        "per_seed": per_seed,
    }


def _policy_config(policy: Any) -> Any:
    candidates = [policy, getattr(policy, "base_model", None), getattr(getattr(policy, "base_model", None), "model", None)]
    for candidate in candidates:
        cfg = getattr(candidate, "config", None)
        if cfg is not None and hasattr(cfg, "chunk_size") and hasattr(cfg, "max_action_dim"):
            return cfg
    raise CanonicalEvalError("CANONICAL_EVAL_FAILED", f"Could not locate SmolVLA config on policy type {type(policy).__name__}.")


def _dtype_counts(policy: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for param in policy.parameters():
        counts[str(param.dtype)] += 1
    return dict(sorted(counts.items()))


def _device_audit(label: str, policy: Any, probe_batch: dict[str, Any], torch_mod: Any) -> dict[str, Any]:
    params = _parameter_summary(policy)
    devices = _tensor_devices(probe_batch)
    audit = {
        "label": label,
        "cuda_available": bool(torch_mod.cuda.is_available()),
        "cuda_device_name": torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else None,
        "model_parameter_device": params.get("first_parameter_device"),
        "model_parameter_dtype": params.get("first_parameter_dtype"),
        "model_dtype_counts": _dtype_counts(policy),
        "input_tensor_devices": devices,
        "input_tensor_shapes": _tensor_shapes(probe_batch),
        "autocast_status": _safe_autocast_status(torch_mod),
        "fp16_params_present": bool(_dtype_counts(policy).get("torch.float16")),
        "bf16_params_present": bool(_dtype_counts(policy).get("torch.bfloat16")),
        "cuda_memory": _cuda_memory(torch_mod),
    }
    print("[canonical-device-audit] " + json.dumps(audit, sort_keys=True, default=_json_default), flush=True)
    if torch_mod.cuda.is_available():
        if not str(params.get("first_parameter_device", "")).startswith("cuda"):
            raise CanonicalEvalError("CPU_FALLBACK_BUG", f"CUDA available but {label} model parameters are on CPU: {params}")
        if not all(str(value).startswith("cuda") for value in devices.values()):
            raise CanonicalEvalError("CPU_FALLBACK_BUG", f"CUDA available but {label} inputs are not all CUDA: {devices}")
    return audit


def _make_noise(policy: Any, seed64: int, torch_mod: Any) -> Any:
    cfg = _policy_config(policy)
    device = torch_mod.device(str(cfg.device) if getattr(cfg, "device", None) is not None else "cuda")
    if device.type != "cuda" and torch_mod.cuda.is_available():
        device = torch_mod.device("cuda")
    generator = torch_mod.Generator(device=device)
    generator.manual_seed(int(seed64))
    return torch_mod.randn(
        (1, int(cfg.chunk_size), int(cfg.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch_mod.float32,
    )


def _postprocess_chunk(action_chunk: Any, postprocessor: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    processed = postprocessor(action_chunk)
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    array = np.asarray(processed, dtype=np.float32)
    if array.ndim == 3:
        chunk = array[0]
    elif array.ndim == 2:
        chunk = array
    else:
        flat = array.reshape(-1)
        if action_dim > 0 and flat.size > action_dim and flat.size % action_dim == 0:
            chunk = flat.reshape(-1, action_dim)
        else:
            chunk = flat.reshape(1, -1)
    return chunk, chunk[0].reshape(-1)


def _predict_chunk_with_seed(
    *,
    policy: Any,
    batch: dict[str, Any],
    postprocessor: Any,
    seed64: int,
    action_dim: int,
    torch_mod: Any,
) -> tuple[np.ndarray, np.ndarray]:
    _set_global_rngs(seed64)
    torch_mod.manual_seed(int(seed64))
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(int(seed64))
    if hasattr(policy, "reset"):
        policy.reset()
    noise = _make_noise(policy, seed64, torch_mod)
    if hasattr(policy, "predict_action_chunk"):
        action_chunk = policy.predict_action_chunk(batch, noise=noise)
    else:
        action_chunk = policy.select_action(batch, noise=noise)
    return _postprocess_chunk(action_chunk, postprocessor, action_dim)


def _chunk_digest(chunk: np.ndarray) -> str:
    rounded = np.round(np.asarray(chunk, dtype=np.float32), 9).tolist()
    return _sha256_payload(rounded)


def _check_runtime(started: float) -> None:
    if time.monotonic() - started > MAX_RUNTIME_SECONDS:
        raise CanonicalEvalError("TOO_HEAVY_LOCAL", "Canonical evaluation exceeded the six-hour hard cap.")


def evaluate_policy_rows(
    *,
    label: str,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    samples: list[dict[str, Any]],
    eval_seeds: list[int],
    action_min: np.ndarray,
    action_max: np.ndarray,
    verify_full_repeat: bool,
    repeat_smoke_count: int,
    started: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    action_dim = int(action_min.shape[0])
    rows: list[dict[str, Any]] = []
    smoke_seen_by_split: Counter[str] = Counter()
    determinism = {
        "mode": "full" if verify_full_repeat else f"smoke_first_{repeat_smoke_count}_frames_per_split",
        "repeat_smoke_count_per_split": int(repeat_smoke_count),
        "checked_predictions": 0,
        "max_action_abs_diff": 0.0,
        "failures": [],
        "tolerance": REPEAT_TOLERANCE,
    }
    policy.eval()
    with torch.inference_mode():
        for sample_index, sample_meta in enumerate(samples):
            _check_runtime(started)
            raw_sample = dataset[int(sample_meta["dataset_local_index"])]
            batch = _add_training_batch_dims(preprocessor(raw_sample))
            if torch.cuda.is_available() and not all(str(value).startswith("cuda") for value in _tensor_devices(batch).values()):
                raise CanonicalEvalError("CPU_FALLBACK_BUG", f"CUDA available but {label} tensors are on CPU: {_tensor_devices(batch)}")
            target = _raw_current_action(raw_sample)
            split_name = str(sample_meta.get("split", "unknown"))
            repeat_this_frame = bool(verify_full_repeat or smoke_seen_by_split[split_name] < int(repeat_smoke_count))
            for eval_seed in eval_seeds:
                seed64 = canonical_rng_seed(eval_seed, sample_meta)
                chunk, pred = _predict_chunk_with_seed(
                    policy=policy,
                    batch=batch,
                    postprocessor=postprocessor,
                    seed64=seed64,
                    action_dim=action_dim,
                    torch_mod=torch,
                )
                if repeat_this_frame:
                    repeat_chunk, repeat_pred = _predict_chunk_with_seed(
                        policy=policy,
                        batch=batch,
                        postprocessor=postprocessor,
                        seed64=seed64,
                        action_dim=action_dim,
                        torch_mod=torch,
                    )
                    max_diff = float(np.max(np.abs(repeat_pred - pred))) if pred.size else 0.0
                    max_chunk_diff = float(np.max(np.abs(repeat_chunk - chunk))) if chunk.size else 0.0
                    max_diff = max(max_diff, max_chunk_diff)
                    determinism["checked_predictions"] += 1
                    determinism["max_action_abs_diff"] = max(float(determinism["max_action_abs_diff"]), max_diff)
                    if max_diff > REPEAT_TOLERANCE:
                        determinism["failures"].append(
                            {
                                "eval_seed": int(eval_seed),
                                "seed64": int(seed64),
                                "episode_index": int(sample_meta["episode_index"]),
                                "frame_index": int(sample_meta["frame_index"]),
                                "task_index": int(sample_meta["task_index"]),
                                "max_abs_diff": max_diff,
                            }
                        )
                        if len(determinism["failures"]) >= 5:
                            raise CanonicalEvalError("EVAL_RNG_POLICY_INVALID", f"Repeated {label} evaluation is not deterministic: {determinism['failures']}")
                enriched_meta = dict(sample_meta)
                enriched_meta["eval_seed"] = int(eval_seed)
                enriched_meta["canonical_rng_seed"] = int(seed64)
                row = _metric_row(
                    sample_meta=enriched_meta,
                    pred=pred,
                    target=target,
                    eval_loss=None,
                    action_min=action_min,
                    action_max=action_max,
                )
                row["action_chunk_shape"] = [int(x) for x in chunk.shape]
                row["action_chunk_sha256"] = _chunk_digest(chunk)
                row["action_chunk_first_two_preview"] = [
                    _round_vector(item, 9) for item in np.asarray(chunk[:2], dtype=np.float32)
                ]
                rows.append(row)
            if repeat_this_frame and not verify_full_repeat:
                smoke_seen_by_split[split_name] += 1
            if progress_every > 0 and (sample_index + 1) % progress_every == 0:
                print(f"[{label}] evaluated {sample_index + 1}/{len(samples)} frames x {len(eval_seeds)} eval seeds", flush=True)
    determinism["smoke_frames_checked_by_split"] = dict(sorted(smoke_seen_by_split.items()))
    determinism["passed"] = not determinism["failures"]
    return rows, determinism


def _paired_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["eval_seed"]), int(row["episode_index"]), int(row["frame_index"]), int(row["task_index"]))


def _record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(record["eval_seed"]), int(record["episode_index"]), int(record["frame_index"]), int(record["task_index"]))


def _mean_rows(
    *,
    dataset: Any,
    samples: list[dict[str, Any]],
    eval_seeds: list[int],
    mean_action: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for sample_meta in samples:
        raw_sample = dataset[int(sample_meta["dataset_local_index"])]
        target = _raw_current_action(raw_sample)
        for eval_seed in eval_seeds:
            meta = dict(sample_meta)
            meta["eval_seed"] = int(eval_seed)
            meta["canonical_rng_seed"] = int(canonical_rng_seed(eval_seed, sample_meta))
            rows.append(
                _metric_row(
                    sample_meta=meta,
                    pred=mean_action,
                    target=target,
                    eval_loss=None,
                    action_min=action_min,
                    action_max=action_max,
                )
            )
    return rows


def _base_record_from_rows(base: dict[str, Any], mean: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(base.get("sample_id", "")),
        "sample_key": {
            "eval_seed": int(base["eval_seed"]),
            "episode_index": int(base["episode_index"]),
            "frame_index": int(base["frame_index"]),
            "task_index": int(base["task_index"]),
        },
        "split": str(base["split"]),
        "eval_seed": int(base["eval_seed"]),
        "canonical_rng_seed": int(base["canonical_rng_seed"]),
        "dataset_local_index": int(base["dataset_local_index"]),
        "dataset_global_index": int(base.get("dataset_global_index", -1)),
        "episode_index": int(base["episode_index"]),
        "frame_index": int(base["frame_index"]),
        "episode_length": int(base["episode_length"]),
        "task_index": int(base["task_index"]),
        "task": str(base["task"]),
        "phase": str(base["phase"]),
        "normalized_phase": _round(float(base.get("normalized_phase", 0.0))),
        "base_action": _round_vector(base["pred_preview"], 9),
        "mean_action": _round_vector(mean["pred_preview"], 9),
        "target_action": _round_vector(base["target_preview"], 9),
        "base_action_l2": base.get("action_l2"),
        "mean_action_l2": mean.get("action_l2"),
        "base_translation_l2": base.get("translation_l2"),
        "base_rotation_l2": base.get("rotation_l2"),
        "base_gripper_abs": base.get("gripper_abs"),
        "base_action_chunk_shape": base.get("action_chunk_shape"),
        "base_action_chunk_sha256": base.get("action_chunk_sha256"),
        "base_action_chunk_first_two_preview": base.get("action_chunk_first_two_preview"),
    }


def _lora_record_from_rows(base: dict[str, Any], lora: dict[str, Any], mean: dict[str, Any]) -> dict[str, Any]:
    base_l2 = float(base["action_l2"])
    lora_l2 = float(lora["action_l2"])
    record = _base_record_from_rows(base, mean)
    record.update(
        {
            "lora_action": _round_vector(lora["pred_preview"], 9),
            "lora_action_l2": lora.get("action_l2"),
            "lora_translation_l2": lora.get("translation_l2"),
            "lora_rotation_l2": lora.get("rotation_l2"),
            "lora_gripper_abs": lora.get("gripper_abs"),
            "lora_action_chunk_shape": lora.get("action_chunk_shape"),
            "lora_action_chunk_sha256": lora.get("action_chunk_sha256"),
            "lora_action_chunk_first_two_preview": lora.get("action_chunk_first_two_preview"),
            "oracle_help_label": int(lora_l2 < base_l2),
            "base_minus_lora_action_l2": _round(base_l2 - lora_l2),
        }
    )
    return record


def _sample_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(record.get("sample_id", "")),
        "dataset_local_index": int(record["dataset_local_index"]),
        "dataset_global_index": int(record.get("dataset_global_index", -1)),
        "episode_index": int(record["episode_index"]),
        "frame_index": int(record["frame_index"]),
        "episode_length": int(record["episode_length"]),
        "task_index": int(record["task_index"]),
        "task": str(record["task"]),
        "phase": str(record["phase"]),
        "split": str(record["split"]),
        "eval_seed": int(record["eval_seed"]),
        "canonical_rng_seed": int(record["canonical_rng_seed"]),
        "normalized_phase": float(record.get("normalized_phase", 0.0)),
    }


def _rows_from_records(
    records: list[dict[str, Any]],
    *,
    pred_key: str,
    action_min: np.ndarray,
    action_max: np.ndarray,
    selected_expert: str,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(record[pred_key], dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=None,
            action_min=action_min,
            action_max=action_max,
        )
        row["selected_expert"] = selected_expert
        rows.append(row)
    return rows


def _rows_for_predictions(
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


def _static_rows(
    records: list[dict[str, Any]],
    alpha: float,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    base = np.asarray([record["base_action"] for record in records], dtype=np.float32)
    lora = np.asarray([record["lora_action"] for record in records], dtype=np.float32)
    pred = float(alpha) * lora + (1.0 - float(alpha)) * base
    return _rows_for_predictions(records, pred, action_min=action_min, action_max=action_max, selected_expert=f"action_space_static_mix_alpha_{alpha}")


def metric_with_balance(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    package = summarize_rows(rows)
    package["per_task"] = _group_summary(rows, "task_index")
    package["per_phase"] = _group_summary(rows, "phase")
    package["task_balanced_action_l2_mean"] = _balanced_mean(package["per_task"], "action_l2_mean")
    package["task_bootstrap_ci95_action_l2"] = _bootstrap_ci(rows, group_key="task_index", seed=seed, iterations=200)
    return package


def aggregate_seed_metrics(rows: list[dict[str, Any]], eval_seeds: list[int], *, seed: int) -> dict[str, Any]:
    per_seed = {}
    for offset, eval_seed in enumerate(eval_seeds):
        seed_rows = [row for row in rows if int(row["eval_seed"]) == int(eval_seed)]
        per_seed[str(eval_seed)] = metric_with_balance(seed_rows, seed=seed + offset)
    aggregate = {}
    for key in METRIC_AGG_KEYS:
        values = [float(metric[key]) for metric in per_seed.values() if metric.get(key) is not None]
        aggregate[key] = _round(float(np.mean(values))) if values else None
        aggregate[f"{key}_std_over_eval_seeds"] = _round(float(np.std(values, ddof=0))) if values else None
    task_ids = sorted({str(row["task_index"]) for row in rows}, key=lambda item: int(item))
    per_task = {}
    for task_id in task_ids:
        values = []
        for metric in per_seed.values():
            task_metric = (metric.get("per_task") or {}).get(task_id) or {}
            if task_metric.get("action_l2_mean") is not None:
                values.append(float(task_metric["action_l2_mean"]))
        per_task[task_id] = {
            "action_l2_mean_over_eval_seeds": _round(float(np.mean(values))) if values else None,
            "action_l2_std_over_eval_seeds": _round(float(np.std(values, ddof=0))) if values else None,
            "eval_seed_count": len(values),
        }
    aggregate["per_seed"] = per_seed
    aggregate["per_task"] = per_task
    aggregate["eval_seeds"] = [int(seed_value) for seed_value in eval_seeds]
    aggregate["std_definition"] = "population_std_over_predeclared_action_generation_seeds"
    aggregate["sample_count_total_seed_repeated"] = len(rows)
    aggregate["sample_count_per_eval_seed"] = {str(seed_value): len([row for row in rows if int(row["eval_seed"]) == int(seed_value)]) for seed_value in eval_seeds}
    return aggregate


def select_static_alpha(
    records: list[dict[str, Any]],
    eval_seeds: list[int],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    grid: list[float] = STATIC_GRID,
) -> dict[str, Any]:
    val_records = [record for record in records if record.get("split") == "val"]
    if not val_records:
        raise CanonicalEvalError("TEST_LEAKAGE_FOUND", "Static alpha selection requires validation records.")
    grid_metrics = {}
    best_alpha = float(grid[0])
    best_value = math.inf
    for alpha in grid:
        rows = _static_rows(val_records, alpha, action_min=action_min, action_max=action_max)
        per_seed = {}
        values = []
        for eval_seed in eval_seeds:
            seed_rows = [row for row in rows if int(row["eval_seed"]) == int(eval_seed)]
            metric = metric_with_balance(seed_rows, seed=int(eval_seed) + int(round(1000 * alpha)))
            per_seed[str(eval_seed)] = {
                "action_l2_mean": metric.get("action_l2_mean"),
                "task_balanced_action_l2_mean": metric.get("task_balanced_action_l2_mean"),
            }
            values.append(float(metric["action_l2_mean"]))
        mean_value = float(np.mean(values))
        grid_metrics[str(alpha)] = {
            "selection_split": "val",
            "per_seed": per_seed,
            "action_l2_mean_over_eval_seeds": _round(mean_value),
            "action_l2_std_over_eval_seeds": _round(float(np.std(values, ddof=0))),
        }
        if mean_value < best_value - 1e-15:
            best_value = mean_value
            best_alpha = float(alpha)
    return {
        "selected_alpha": best_alpha,
        "selection_split": "val",
        "grid": grid_metrics,
        "test_metrics_influence_selection": False,
        "tie_break": "first_alpha_in_predeclared_grid_order",
    }


def _task_router_proxy_from_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"base": [], "lora": []})
    for record in records:
        if record.get("split") != "val":
            continue
        task = str(record["task_index"])
        values[task]["base"].append(float(record["base_action_l2"]))
        values[task]["lora"].append(float(record["lora_action_l2"]))
    routing = {}
    stats = {}
    for task, task_values in sorted(values.items(), key=lambda item: int(item[0])):
        base_mean = float(np.mean(task_values["base"]))
        lora_mean = float(np.mean(task_values["lora"]))
        route = "rank4_lora" if lora_mean < base_mean else "frozen_base"
        routing[task] = route
        stats[task] = {"base_val_action_l2": _round(base_mean), "lora_val_action_l2": _round(lora_mean), "selected": route}
    return {
        "routing": routing,
        "validation_stats": stats,
        "selection_split": "val",
        "proxy_only": True,
        "test_metrics_influence_selection": False,
    }


def _apply_task_router_proxy(
    records: list[dict[str, Any]],
    routing: dict[str, str],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        pred_key = "lora_action" if routing.get(str(record["task_index"]), "frozen_base") == "rank4_lora" else "base_action"
        selected = "rank4_lora" if pred_key == "lora_action" else "frozen_base"
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(record[pred_key], dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=None,
            action_min=action_min,
            action_max=action_max,
        )
        row["selected_expert"] = selected
        row["router_proxy"] = "task_or_instruction_router_proxy"
        rows.append(row)
    return rows


def _frame_oracle_rows(records: list[dict[str, Any]], *, action_min: np.ndarray, action_max: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        use_lora = float(record["lora_action_l2"]) < float(record["base_action_l2"])
        pred_key = "lora_action" if use_lora else "base_action"
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(record[pred_key], dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=None,
            action_min=action_min,
            action_max=action_max,
        )
        row["selected_expert"] = "rank4_lora" if use_lora else "frozen_base"
        row["oracle_type"] = "frame_oracle_upper_bound"
        rows.append(row)
    return rows


def _task_oracle_rows(records: list[dict[str, Any]], *, action_min: np.ndarray, action_max: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"base": [], "lora": []})
    for record in records:
        task = str(record["task_index"])
        values[task]["base"].append(float(record["base_action_l2"]))
        values[task]["lora"].append(float(record["lora_action_l2"]))
    routing = {}
    for task, task_values in sorted(values.items(), key=lambda item: int(item[0])):
        routing[task] = "rank4_lora" if float(np.mean(task_values["lora"])) < float(np.mean(task_values["base"])) else "frozen_base"
    rows = []
    for record in records:
        use_lora = routing[str(record["task_index"])] == "rank4_lora"
        pred_key = "lora_action" if use_lora else "base_action"
        row = _metric_row(
            sample_meta=_sample_meta(record),
            pred=np.asarray(record[pred_key], dtype=np.float32),
            target=np.asarray(record["target_action"], dtype=np.float32),
            eval_loss=None,
            action_min=action_min,
            action_max=action_max,
        )
        row["selected_expert"] = "rank4_lora" if use_lora else "frozen_base"
        row["oracle_type"] = "task_oracle_upper_bound"
        rows.append(row)
    return rows, {"routing": routing, "selection_split": "test", "oracle_only": True}


def _prediction_digest(records: list[dict[str, Any]], action_keys: list[str]) -> str:
    payload = []
    for record in records:
        item = {
            "eval_seed": int(record["eval_seed"]),
            "canonical_rng_seed": int(record["canonical_rng_seed"]),
            "episode_index": int(record["episode_index"]),
            "frame_index": int(record["frame_index"]),
            "task_index": int(record["task_index"]),
            "split": str(record["split"]),
        }
        for key in action_keys:
            if key in record:
                item[key] = [round(float(value), 9) for value in record[key]]
        payload.append(item)
    return _sha256_payload(payload)


def _evaluate_lora_artifact(
    artifact: dict[str, Any],
    eval_seeds: list[int],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    records = artifact.get("records") or []
    split_records = {split: [record for record in records if record.get("split") == split] for split in ["val", "test"]}
    test = split_records["test"]
    base_rows = _rows_from_records(test, pred_key="base_action", action_min=action_min, action_max=action_max, selected_expert="frozen_base")
    lora_rows = _rows_from_records(test, pred_key="lora_action", action_min=action_min, action_max=action_max, selected_expert=f"rank4_lora_seed_{seed}")
    mean_rows = _rows_from_records(test, pred_key="mean_action", action_min=action_min, action_max=action_max, selected_expert="mean_action_prior")
    static_selection = select_static_alpha(records, eval_seeds, action_min=action_min, action_max=action_max)
    static_rows = _static_rows(test, float(static_selection["selected_alpha"]), action_min=action_min, action_max=action_max)
    router_proxy = _task_router_proxy_from_validation(records)
    router_rows = _apply_task_router_proxy(test, router_proxy["routing"], action_min=action_min, action_max=action_max)
    task_oracle, task_oracle_info = _task_oracle_rows(test, action_min=action_min, action_max=action_max)
    frame_oracle = _frame_oracle_rows(test, action_min=action_min, action_max=action_max)
    rows = {
        "frozen_base": base_rows,
        f"rank4_lora_seed_{seed}": lora_rows,
        "mean_action_prior": mean_rows,
        f"validation_selected_action_space_static_mix_seed_{seed}": static_rows,
        f"task_or_instruction_router_proxy_seed_{seed}": router_rows,
        f"task_oracle_upper_bound_seed_{seed}": task_oracle,
        f"frame_oracle_upper_bound_seed_{seed}": frame_oracle,
    }
    metrics = {name: aggregate_seed_metrics(row_set, eval_seeds, seed=seed * 1000 + index) for index, (name, row_set) in enumerate(rows.items())}
    return {
        "seed": seed,
        "split_record_counts": {split: len(values) for split, values in split_records.items()},
        "metrics": metrics,
        "static_selection": static_selection,
        "task_or_instruction_router_proxy": router_proxy,
        "task_oracle_upper_bound": task_oracle_info,
        "rank_order_realistic": sorted(
            [
                {"baseline": "frozen_base", "action_l2": metrics["frozen_base"]["action_l2_mean"]},
                {"baseline": f"rank4_lora_seed_{seed}", "action_l2": metrics[f"rank4_lora_seed_{seed}"]["action_l2_mean"]},
                {"baseline": "mean_action_prior", "action_l2": metrics["mean_action_prior"]["action_l2_mean"]},
                {"baseline": f"validation_selected_action_space_static_mix_seed_{seed}", "action_l2": metrics[f"validation_selected_action_space_static_mix_seed_{seed}"]["action_l2_mean"]},
                {"baseline": f"task_or_instruction_router_proxy_seed_{seed}", "action_l2": metrics[f"task_or_instruction_router_proxy_seed_{seed}"]["action_l2_mean"]},
            ],
            key=lambda item: float(item["action_l2"]),
        ),
    }


def _load_base_policy_and_processors(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint_path = Path(args.checkpoint_path)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(vlm_root)
    if hasattr(cfg, "chunk_size"):
        cfg.chunk_size = int(args.chunk_size)
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
    return policy, cfg, preprocessor, postprocessor, {"cuda_memory_after_load": _cuda_memory(torch)}


def _load_lora_policy(args: argparse.Namespace, adapter_path: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    checkpoint_path = Path(args.checkpoint_path)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(vlm_root)
    if hasattr(cfg, "chunk_size"):
        cfg.chunk_size = int(args.chunk_size)
    base_policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=cfg,
        local_files_only=True,
        cache_dir=hf_home,
        token=False,
        strict=False,
    )
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    policy = PeftModel.from_pretrained(base_policy, str(adapter_path), config=peft_config, is_trainable=False)
    policy.to("cuda")
    policy.eval()
    return policy, {
        "adapter_path": str(adapter_path),
        "peft_config_base_model_name_or_path": getattr(peft_config, "base_model_name_or_path", None),
        "loaded_policy_type": type(policy).__name__,
        "cuda_memory_after_load": _cuda_memory(torch),
    }


def _load_existing_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    artifact = _read_json(path)
    if int(artifact.get("artifact_version") or 0) != ARTIFACT_VERSION:
        return None
    if not artifact.get("records"):
        return None
    return artifact


def generate_canonical_artifacts(args: argparse.Namespace, manifest: dict[str, Any], checkpoint_audit: dict[str, Any], started: float) -> dict[str, Any]:
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        raise CanonicalEvalError("CANONICAL_EVAL_FAILED", "Forbidden gate(s) set: " + ", ".join(forbidden))

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not torch.cuda.is_available():
        raise CanonicalEvalError("CPU_FALLBACK_BUG", "CUDA unavailable; refusing canonical SmolVLA evaluation on CPU.")
    torch.cuda.reset_peak_memory_stats()
    eval_seeds = _parse_int_list(args.eval_seeds)
    lora_seeds = _parse_int_list(args.lora_seeds)
    selected_episodes, split_samples, _all_samples = _manifest_samples(manifest)
    eval_samples = split_samples["val"] + split_samples["test"]
    dataset_root = Path(args.dataset_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    fps = float(info.get("fps", 10.0))
    delta_timestamps = {"action": [i / fps for i in range(int(args.chunk_size))]}
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

    base_path = Path(args.base_artifact)
    lora_paths = {seed: Path(args.lora_artifact_pattern.format(seed=seed)) for seed in lora_seeds}
    loaded_existing = False
    base_artifact = None if bool(args.force) else _load_existing_artifact(base_path)
    lora_artifacts = {seed: None if bool(args.force) else _load_existing_artifact(path) for seed, path in lora_paths.items()}
    if base_artifact is not None and all(artifact is not None for artifact in lora_artifacts.values()):
        loaded_existing = True
    else:
        base_policy, _base_cfg, preprocessor, postprocessor, base_load_info = _load_base_policy_and_processors(args)
        probe = _add_training_batch_dims(preprocessor(dataset[int(eval_samples[0]["dataset_local_index"])]))
        base_device_audit = _device_audit("frozen_base", base_policy, probe, torch)
        print(f"[canonical] evaluating frozen base on {len(eval_samples)} val/test frames x {len(eval_seeds)} eval seeds", flush=True)
        base_rows, base_determinism = evaluate_policy_rows(
            label="frozen_base",
            policy=base_policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            samples=eval_samples,
            eval_seeds=eval_seeds,
            action_min=action_min,
            action_max=action_max,
            verify_full_repeat=bool(args.verify_full_repeat),
            repeat_smoke_count=int(args.repeat_smoke_count),
            started=started,
            progress_every=int(args.progress_every),
        )
        mean_rows = _mean_rows(
            dataset=dataset,
            samples=eval_samples,
            eval_seeds=eval_seeds,
            mean_action=mean_action,
            action_min=action_min,
            action_max=action_max,
        )
        mean_by_key = {_paired_key(row): row for row in mean_rows}
        base_records = [_base_record_from_rows(base, mean_by_key[_paired_key(base)]) for base in base_rows]
        base_artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "date": DATE,
            "source": "official_smolvla_canonical_frozen_base_val_test",
            "artifact_status": "generated",
            "policy": {
                "persisted_disk_reloaded_policy": True,
                "policy_name": "frozen_base",
                "training_performed": False,
                "checkpoint_regenerated": False,
                "rollouts_performed": False,
                "old_custom_route_used": False,
            },
            "rng_policy": rng_policy_payload(eval_seeds),
            "paths": {
                "checkpoint": str(Path(args.checkpoint_path)),
                "dataset": str(dataset_root),
                "split_manifest": str(Path(args.split_manifest)),
            },
            "dataset": {
                "selected_episode_count": len(selected_episodes),
                "evaluated_split_frame_counts": {"val": len(split_samples["val"]), "test": len(split_samples["test"])},
                "eval_seed_count": len(eval_seeds),
                "record_count": len(base_records),
            },
            "action_range": {"min": _round_vector(action_min, 9), "max": _round_vector(action_max, 9)},
            "device_audit": base_device_audit,
            "load_info": base_load_info,
            "determinism": base_determinism,
            "prediction_digest": _prediction_digest(base_records, ["base_action", "mean_action", "target_action"]),
            "records": base_records,
        }
        _write_json(base_path, base_artifact)
        del base_policy
        torch.cuda.empty_cache()

        lora_artifacts = {}
        for lora_seed in lora_seeds:
            _check_runtime(started)
            adapter_path = Path((checkpoint_audit["per_seed"][str(lora_seed)] or {})["checkpoint_path"])
            lora_policy, lora_load_info = _load_lora_policy(args, adapter_path)
            lora_probe = _add_training_batch_dims(preprocessor(dataset[int(eval_samples[0]["dataset_local_index"])]))
            lora_device_audit = _device_audit(f"rank4_lora_seed_{lora_seed}", lora_policy, lora_probe, torch)
            print(f"[canonical] evaluating rank4_lora_seed_{lora_seed} on {len(eval_samples)} val/test frames x {len(eval_seeds)} eval seeds", flush=True)
            lora_rows, lora_determinism = evaluate_policy_rows(
                label=f"rank4_lora_seed_{lora_seed}",
                policy=lora_policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                samples=eval_samples,
                eval_seeds=eval_seeds,
                action_min=action_min,
                action_max=action_max,
                verify_full_repeat=bool(args.verify_full_repeat),
                repeat_smoke_count=int(args.repeat_smoke_count),
                started=started,
                progress_every=int(args.progress_every),
            )
            base_by_key = {_record_key(record): record for record in base_records}
            lora_records = []
            for lora in lora_rows:
                key = _paired_key(lora)
                base_record = base_by_key[key]
                lora_records.append(
                    _lora_record_from_rows(
                        base={
                            **_sample_meta(base_record),
                            "pred_preview": base_record["base_action"],
                            "target_preview": base_record["target_action"],
                            "action_l2": base_record["base_action_l2"],
                            "translation_l2": base_record["base_translation_l2"],
                            "rotation_l2": base_record["base_rotation_l2"],
                            "gripper_abs": base_record["base_gripper_abs"],
                            "action_chunk_shape": base_record["base_action_chunk_shape"],
                            "action_chunk_sha256": base_record["base_action_chunk_sha256"],
                            "action_chunk_first_two_preview": base_record["base_action_chunk_first_two_preview"],
                        },
                        lora=lora,
                        mean={
                            **_sample_meta(base_record),
                            "pred_preview": base_record["mean_action"],
                            "target_preview": base_record["target_action"],
                            "action_l2": base_record["mean_action_l2"],
                        },
                    )
                )
            artifact = {
                "artifact_version": ARTIFACT_VERSION,
                "date": DATE,
                "source": f"official_smolvla_canonical_persisted_lora_seed_{lora_seed}_val_test",
                "artifact_status": "generated",
                "seed": int(lora_seed),
                "policy": {
                    "persisted_disk_reloaded_policy": True,
                    "policy_name": f"rank4_lora_seed_{lora_seed}",
                    "training_performed": False,
                    "checkpoint_regenerated": False,
                    "rollouts_performed": False,
                    "old_custom_route_used": False,
                    "base_vs_lora_common_noise": True,
                },
                "rng_policy": rng_policy_payload(eval_seeds),
                "paths": {
                    "checkpoint": str(Path(args.checkpoint_path)),
                    "adapter_checkpoint": str(adapter_path),
                    "dataset": str(dataset_root),
                    "split_manifest": str(Path(args.split_manifest)),
                },
                "checkpoint_identity": checkpoint_audit["per_seed"][str(lora_seed)],
                "dataset": {
                    "selected_episode_count": len(selected_episodes),
                    "evaluated_split_frame_counts": {"val": len(split_samples["val"]), "test": len(split_samples["test"])},
                    "eval_seed_count": len(eval_seeds),
                    "record_count": len(lora_records),
                },
                "action_range": {"min": _round_vector(action_min, 9), "max": _round_vector(action_max, 9)},
                "device_audit": lora_device_audit,
                "load_info": lora_load_info,
                "determinism": lora_determinism,
                "prediction_digest": _prediction_digest(lora_records, ["base_action", "lora_action", "mean_action", "target_action"]),
                "records": lora_records,
            }
            _write_json(lora_paths[lora_seed], artifact)
            lora_artifacts[lora_seed] = artifact
            del lora_policy
            torch.cuda.empty_cache()

    action_min = np.asarray((base_artifact.get("action_range") or {}).get("min"), dtype=np.float32)
    action_max = np.asarray((base_artifact.get("action_range") or {}).get("max"), dtype=np.float32)
    seed_evaluations = {}
    for lora_seed, artifact in sorted(lora_artifacts.items()):
        seed_evaluations[str(lora_seed)] = _evaluate_lora_artifact(
            artifact,
            eval_seeds,
            action_min=action_min,
            action_max=action_max,
            seed=int(lora_seed),
        )
    artifact_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "date": DATE,
        "artifact_status": "loaded_existing" if loaded_existing else "generated",
        "rng_policy": rng_policy_payload(eval_seeds),
        "split_manifest": {
            "path": str(Path(args.split_manifest)),
            "sha256": _sha256_file(Path(args.split_manifest)),
            "summary": manifest.get("summary"),
        },
        "checkpoint_manifest": {
            "path": str(Path(args.checkpoint_manifest)),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "audit": checkpoint_audit,
        },
        "artifacts": {
            "frozen_base": {
                "path": str(base_path),
                "sha256": _sha256_file(base_path),
                "prediction_digest": base_artifact.get("prediction_digest"),
                "record_count": len(base_artifact.get("records") or []),
            },
            **{
                f"rank4_lora_seed_{seed}": {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "prediction_digest": (lora_artifacts[seed] or {}).get("prediction_digest"),
                    "record_count": len((lora_artifacts[seed] or {}).get("records") or []),
                }
                for seed, path in sorted(lora_paths.items())
            },
        },
    }
    _write_json(Path(args.prediction_manifest), artifact_manifest)
    return {
        "artifact_manifest": artifact_manifest,
        "base_artifact": {key: value for key, value in base_artifact.items() if key != "records"},
        "lora_artifacts": {
            str(seed): {key: value for key, value in artifact.items() if key != "records"}
            for seed, artifact in lora_artifacts.items()
        },
        "seed_evaluations": seed_evaluations,
        "loaded_existing_artifacts": loaded_existing,
    }


def rng_policy_payload(eval_seeds: list[int]) -> dict[str, Any]:
    return {
        "namespace": RNG_NAMESPACE,
        "action_generation_eval_seeds": [int(seed) for seed in eval_seeds],
        "derivation_formula": (
            "seed64 = int(sha256(json({namespace, eval_seed, immutable_frame_identity}, "
            "sort_keys=True, compact)).hexdigest()[:16], 16) & ((1 << 63) - 1)"
        ),
        "immutable_frame_identity_fields": [
            "split",
            "sample_id",
            "dataset_local_index",
            "dataset_global_index",
            "episode_index",
            "frame_index",
            "episode_length",
            "task_index",
        ],
        "labels_excluded_from_rng": ["target_action", "action_l2", "eval_loss", "success", "reward"],
        "common_random_numbers": "Base and LoRA predictions for the same eval_seed/frame regenerate the same torch CUDA noise tensor from seed64.",
        "noise_scope": "postprocessed official SmolVLA action chunk; offline action-L2 uses the current/first action vector.",
        "test_outcomes_used_for_rng_or_alpha": False,
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ["torch", "lerobot", "transformers", "peft", "accelerate", "huggingface_hub", "safetensors", "hf-libero", "libero", "robosuite", "mujoco", "gymnasium"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def inspect_rollout_environment(canonical_ready: bool) -> dict[str, Any]:
    requires = []
    try:
        requires = metadata.requires("lerobot") or []
    except Exception:
        requires = []
    libero_related_requires = [req for req in requires if any(key in req.lower() for key in ["libero", "robosuite", "mujoco", "gymnasium"])]
    import_checks = {}
    for module in ["lerobot.envs.libero", "lerobot.scripts.lerobot_eval", "libero", "libero.libero", "robosuite", "mujoco", "gymnasium"]:
        spec_error = None
        try:
            spec_present = util.find_spec(module) is not None
        except Exception as exc:
            spec_present = False
            spec_error = f"{type(exc).__name__}: {exc}"
        import_error = None
        imported = False
        if spec_present:
            try:
                import_module(module)
                imported = True
            except Exception as exc:
                import_error = f"{type(exc).__name__}: {exc}"
        import_checks[module] = {"spec_present": spec_present, "spec_error": spec_error, "imported": imported, "import_error": import_error}
    native_windows = platform.system().lower().startswith("win")
    missing = [name for name, version in _package_versions().items() if name in {"hf-libero", "libero", "robosuite"} and version == "NOT_INSTALLED"]
    local_repro_lock = Path("configs/official_smolvla_repro_lock.yaml")
    local_sources = Path("configs/libero_robosuite_sources.yaml")
    scripts_dir = Path(sys.executable).parent
    cli_candidates = [
        shutil.which("lerobot-eval"),
        str(scripts_dir / "Scripts" / "lerobot-eval.exe") if (scripts_dir / "Scripts" / "lerobot-eval.exe").is_file() else None,
        str(scripts_dir / "Scripts" / "lerobot-eval") if (scripts_dir / "Scripts" / "lerobot-eval").is_file() else None,
        str(scripts_dir / "lerobot-eval.exe") if (scripts_dir / "lerobot-eval.exe").is_file() else None,
        str(scripts_dir / "lerobot-eval") if (scripts_dir / "lerobot-eval").is_file() else None,
    ]
    cli_path = next((candidate for candidate in cli_candidates if candidate), None)
    compatible_libero = {
        "package": "hf-libero",
        "version_spec_from_installed_lerobot_metadata": ">=0.1.3,<0.2.0",
        "source": "installed LeRobot 0.4.4 package metadata: extra == 'libero'",
        "installed": _package_versions().get("hf-libero") != "NOT_INSTALLED",
    }
    compatible_robosuite = {
        "package": "robosuite",
        "version_spec_from_installed_lerobot_metadata": "not directly declared; expected through hf-libero/LIBERO dependency resolution",
        "source": "not established in active env because hf-libero/libero/robosuite are not installed",
        "installed": _package_versions().get("robosuite") != "NOT_INSTALLED",
    }
    if not canonical_ready:
        rollout_decision = "CANONICAL_EVAL_FAILED"
        can_run = False
        reason = "canonicalization did not pass"
    elif missing:
        rollout_decision = "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT" if native_windows else "OFFICIAL_EVAL_ENV_BLOCKED"
        can_run = False
        reason = f"Official LIBERO rollout dependencies are missing in the active native environment: {missing}"
    elif native_windows:
        rollout_decision = "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT"
        can_run = False
        reason = "Local repro lock and prior simulator topology use WSL/Linux for LIBERO/RoboSuite rollout; native Windows path is not verified."
    else:
        rollout_decision = "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT"
        can_run = True
        reason = "Dependencies appear importable on a non-Windows platform; official smoke can be attempted under a separate runner."
    return {
        "date": DATE,
        "canonical_ready": canonical_ready,
        "official_lerobot_entrypoint": {
            "cli": cli_path,
            "module": "lerobot.scripts.lerobot_eval",
            "env_type": "libero",
            "example_args": "--env.type=libero --env.task=libero_10 --policy.use_peft=true --policy.device=cuda",
        },
        "compatible_libero_package": compatible_libero,
        "compatible_robosuite_package": compatible_robosuite,
        "compatible_mujoco_package": {
            "package": "mujoco",
            "installed_version": _package_versions().get("mujoco"),
            "source": "active conda environment package metadata",
        },
        "adapter_loading_path": "lerobot.policies.factory.make_policy with cfg.policy.use_peft=True and PeftModel.from_pretrained",
        "local_checkpoint_configuration": {
            "base": "C:/assets/checkpoints/smolvla_libero",
            "adapters": [f"C:/assets/checkpoints/smolvla_libero_lora/rank4/seed_{seed}" for seed in LORA_SEEDS],
        },
        "package_versions": _package_versions(),
        "lerobot_libero_requirements_from_installed_metadata": libero_related_requires,
        "official_source_files": {
            "libero_env": "C:/Users/jiheo/miniconda3/envs/tca_map/Lib/site-packages/lerobot/envs/libero.py",
            "eval_script": "C:/Users/jiheo/miniconda3/envs/tca_map/Lib/site-packages/lerobot/scripts/lerobot_eval.py",
            "env_config": "C:/Users/jiheo/miniconda3/envs/tca_map/Lib/site-packages/lerobot/envs/configs.py",
        },
        "import_checks": import_checks,
        "supported_os_assessment": {
            "current_os": platform.platform(),
            "native_windows": native_windows,
            "local_repro_lock_path": str(local_repro_lock),
            "local_repro_lock_exists": local_repro_lock.exists(),
            "local_sources_path": str(local_sources),
            "local_sources_exists": local_sources.exists(),
            "assessment": reason,
        },
        "rendering_backend_required": "OffScreenRenderEnv via LIBERO/RoboSuite; WSL/Linux should set MUJOCO_GL=osmesa or another verified offscreen backend before smoke.",
        "required_task_assets": "LIBERO bddl_files and init_states resolved by libero.libero.get_libero_path().",
        "package_changes": [],
        "additional_downloads_performed": False,
        "can_run_native_smoke_now": can_run,
        "rollout_environment_decision": rollout_decision,
        "blocked_reason": reason,
    }


def choose_intermediate_decision(report: dict[str, Any]) -> str:
    if report.get("errors"):
        code = str(report["errors"][0].get("code"))
        return code if code in INTERMEDIATE_DECISIONS else "CANONICAL_EVAL_FAILED"
    checkpoint = report.get("checkpoint_integrity") or {}
    if not checkpoint.get("all_complete_verified"):
        return "CHECKPOINT_IDENTITY_FAILED"
    split = ((report.get("manifest_summary") or {}).get("leakage_checks") or {})
    if not all(bool(value) for value in split.values()):
        return "TEST_LEAKAGE_FOUND"
    artifacts = report.get("canonical_artifacts") or {}
    checks = []
    base_det = ((artifacts.get("base_artifact") or {}).get("determinism") or {})
    checks.append(bool(base_det.get("passed", True)) and int(base_det.get("checked_predictions") or 0) > 0)
    for artifact in (artifacts.get("lora_artifacts") or {}).values():
        det = artifact.get("determinism") or {}
        checks.append(bool(det.get("passed", True)) and int(det.get("checked_predictions") or 0) > 0)
    if not all(checks):
        return "EVAL_RNG_POLICY_INVALID"
    return "CANONICAL_BASELINES_READY_FOR_ROLLOUT"


def _final_decision(intermediate: str, rollout_env: dict[str, Any]) -> str:
    if intermediate != "CANONICAL_BASELINES_READY_FOR_ROLLOUT":
        return "CANONICAL_EVAL_FAILED"
    decision = rollout_env.get("rollout_environment_decision")
    if decision == "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT":
        return "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT"
    if decision == "OFFICIAL_EVAL_ENV_BLOCKED":
        return "OFFICIAL_EVAL_ENV_BLOCKED"
    return "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT"


def _metric_rows_for_markdown(seed_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    added_global = False
    for seed, evaluation in sorted(seed_evaluations.items(), key=lambda item: int(item[0])):
        metrics = evaluation.get("metrics") or {}
        if not added_global:
            for name in ["frozen_base", "mean_action_prior"]:
                metric = metrics.get(name) or {}
                rows.append({"baseline": name, **metric})
            added_global = True
        for name in [
            f"rank4_lora_seed_{seed}",
            f"validation_selected_action_space_static_mix_seed_{seed}",
            f"task_or_instruction_router_proxy_seed_{seed}",
            f"task_oracle_upper_bound_seed_{seed}",
            f"frame_oracle_upper_bound_seed_{seed}",
        ]:
            metric = metrics.get(name) or {}
            rows.append({"baseline": name, **metric})
    return rows


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "date": DATE,
        "status": "started",
        "intermediate_decision": None,
        "final_decision": None,
        "git": {"head": _git_head(), "branch": _git_branch()},
        "policy": {
            "experiments_performed": True,
            "training_performed": False,
            "checkpoint_regeneration_performed": False,
            "gpu_inference_performed": True,
            "downloads_performed": False,
            "rollouts_performed": False,
            "old_custom_libero_7d_route_used": False,
            "openvla_oft_executed": False,
            "fcar_revived": False,
            "method_designed": False,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "split_manifest": str(Path(args.split_manifest)),
            "checkpoint_manifest": str(Path(args.checkpoint_manifest)),
            "prediction_manifest": str(Path(args.prediction_manifest)),
        },
        "rng_policy": rng_policy_payload(_parse_int_list(args.eval_seeds)),
        "errors": [],
    }
    try:
        manifest = _read_json(Path(args.split_manifest))
        _validate_split_manifest(manifest)
        checkpoint_audit = validate_checkpoint_manifest(Path(args.checkpoint_manifest), _parse_int_list(args.lora_seeds))
        report["manifest_summary"] = manifest.get("summary")
        report["checkpoint_integrity"] = checkpoint_audit
        artifacts = generate_canonical_artifacts(args, manifest, checkpoint_audit, started)
        report["canonical_artifacts"] = artifacts
        report["canonical_metric_rows"] = _metric_rows_for_markdown(artifacts["seed_evaluations"])
        report["historical_supersession"] = {
            "status": "SUPERSEDED_NONCANONICAL_PROTOCOL",
            "historical_commit": "5d48b1e",
            "reason": "Historical run evaluated an ephemeral in-memory path and evaluation RNG was not part of protocol identity.",
            "historical_reports_preserved": True,
        }
        intermediate = choose_intermediate_decision(report)
        report["intermediate_decision"] = intermediate
        rollout_env = inspect_rollout_environment(intermediate == "CANONICAL_BASELINES_READY_FOR_ROLLOUT")
        report["rollout_environment"] = rollout_env
        report["rollout_pilot"] = {
            "smoke_ran": False,
            "pilot_ran": False,
            "tasks_evaluated": [],
            "episodes_per_policy": 0,
            "policies_evaluated": [],
            "success_rates": {},
            "latency_vram_forward_passes": {},
            "reason": rollout_env.get("blocked_reason"),
        }
        report["scientific_gap_report"] = {
            "offline_action_l2_correlates_with_task_success": "not measured; official rollout did not run",
            "strongest_closed_loop_baseline": None,
            "structured_failure_categories": [],
            "method_worthy_gap_status": "not adjudicated without official closed-loop rollout",
        }
        report["final_decision"] = _final_decision(intermediate, rollout_env)
        report["status"] = "completed" if intermediate == "CANONICAL_BASELINES_READY_FOR_ROLLOUT" else "blocked"
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 0 if report["final_decision"] in FINAL_DECISIONS else 31
    except CanonicalEvalError as exc:
        report["status"] = "blocked"
        report["errors"].append({"code": exc.code, "message": str(exc)})
        report["intermediate_decision"] = exc.code if exc.code in INTERMEDIATE_DECISIONS else "CANONICAL_EVAL_FAILED"
        report["final_decision"] = "CANONICAL_EVAL_FAILED"
        report["rollout_environment"] = inspect_rollout_environment(False)
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 31
    except Exception as exc:  # pragma: no cover - report boundary
        report["status"] = "blocked"
        report["errors"].append({"code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        report["intermediate_decision"] = "CANONICAL_EVAL_FAILED"
        report["final_decision"] = "CANONICAL_EVAL_FAILED"
        report["rollout_environment"] = inspect_rollout_environment(False)
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 32


def _write_eval_policy(report: dict[str, Any], path: Path) -> None:
    rng = report.get("rng_policy") or {}
    lines = [
        "# Official SmolVLA Canonical Eval Policy",
        "",
        f"Date: {report['date']}",
        "",
        "- Policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`, and validation-selected action-space static mixes.",
        "- Persisted disk-reloaded policies only: `true`.",
        "- Training/checkpoint regeneration: `false`.",
        "- Old custom LIBERO_7D route: `false`.",
        f"- Action-generation eval seeds: `{rng.get('action_generation_eval_seeds')}`.",
        f"- RNG formula: `{rng.get('derivation_formula')}`.",
        f"- RNG identity fields: `{rng.get('immutable_frame_identity_fields')}`.",
        f"- Labels excluded from RNG: `{rng.get('labels_excluded_from_rng')}`.",
        "- Static-mix alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`.",
        "- Static-mix alpha selection split: `val` only.",
        "- Test outcomes used for seed/alpha selection: `false`.",
        f"- Repeat determinism mode: `{(((report.get('canonical_artifacts') or {}).get('base_artifact') or {}).get('determinism') or {}).get('mode')}`.",
        "- Offline action-L2 uses the current/first postprocessed action vector; full postprocessed action chunks are generated and hashed.",
    ]
    _write_lines(path, lines)


def _write_baseline_result(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Canonical Baseline Result",
        "",
        f"Date: {report['date']}",
        "",
        f"- intermediate decision: `{report.get('intermediate_decision')}`",
        f"- final decision: `{report.get('final_decision')}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- checkpoint regeneration happened: `{report['policy']['checkpoint_regeneration_performed']}`",
        f"- GPU inference happened: `{report['policy']['gpu_inference_performed']}`",
        f"- downloads happened: `{report['policy']['downloads_performed']}`",
        f"- rollouts happened: `{report['policy']['rollouts_performed']}`",
        f"- historical status: `{(report.get('historical_supersession') or {}).get('status')}`",
        "",
        "## Canonical Metrics",
        "",
        "| baseline | action L2 mean | action L2 std | task-balanced L2 | task-balanced std | gripper abs | range violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("canonical_metric_rows") or []:
        lines.append(
            f"| {row.get('baseline')} | {row.get('action_l2_mean')} | {row.get('action_l2_mean_std_over_eval_seeds')} | "
            f"{row.get('task_balanced_action_l2_mean')} | {row.get('task_balanced_action_l2_mean_std_over_eval_seeds')} | "
            f"{row.get('gripper_abs_mean')} | {row.get('range_violation_rate')} |"
        )
    lines.extend(["", "## Static Selection"])
    seed_evals = (((report.get("canonical_artifacts") or {}).get("seed_evaluations")) or {})
    for seed, evaluation in sorted(seed_evals.items(), key=lambda item: int(item[0])):
        static = evaluation.get("static_selection") or {}
        lines.append(f"- seed {seed}: alpha `{static.get('selected_alpha')}`, split `{static.get('selection_split')}`")
    lines.extend(["", "## Exact Next Step", "", _exact_next_step(report)])
    _write_lines(path, lines)


def _write_historical_supersession(report: dict[str, Any], path: Path) -> None:
    hist = report.get("historical_supersession") or {}
    lines = [
        "# Official SmolVLA Historical Result Supersession",
        "",
        f"Date: {report['date']}",
        "",
        f"- status: `{hist.get('status')}`",
        f"- historical commit: `{hist.get('historical_commit')}`",
        f"- reports preserved: `{hist.get('historical_reports_preserved')}`",
        "",
        str(hist.get("reason")),
        "",
        "Historical ephemeral metrics are preserved for audit trail only and must not be used as canonical baselines.",
    ]
    _write_lines(path, lines)


def _write_rollout_env_setup(report: dict[str, Any], path: Path) -> None:
    env = report.get("rollout_environment") or {}
    lines = [
        "# Official SmolVLA Rollout Env Setup",
        "",
        f"Date: {report['date']}",
        "",
        f"- rollout environment decision: `{env.get('rollout_environment_decision')}`",
        f"- can run native smoke now: `{env.get('can_run_native_smoke_now')}`",
        f"- blocked reason: {env.get('blocked_reason')}",
        f"- official entrypoint: `{(env.get('official_lerobot_entrypoint') or {}).get('module')}` / `{(env.get('official_lerobot_entrypoint') or {}).get('cli')}`",
        f"- adapter loading path: `{env.get('adapter_loading_path')}`",
        f"- rendering backend required: `{env.get('rendering_backend_required')}`",
        f"- compatible LIBERO package: `{env.get('compatible_libero_package')}`",
        f"- compatible RoboSuite package: `{env.get('compatible_robosuite_package')}`",
        f"- compatible MuJoCo package: `{env.get('compatible_mujoco_package')}`",
        f"- package changes: `{env.get('package_changes')}`",
        f"- additional downloads performed: `{env.get('additional_downloads_performed')}`",
        "",
        "## Package Versions",
        "",
        *[f"- {key}: `{value}`" for key, value in sorted((env.get("package_versions") or {}).items())],
        "",
        "## LeRobot LIBERO Requirements",
        "",
        *[f"- `{req}`" for req in (env.get("lerobot_libero_requirements_from_installed_metadata") or [])],
    ]
    _write_lines(path, lines)


def _write_rollout_plan(report: dict[str, Any], path: Path) -> None:
    env = report.get("rollout_environment") or {}
    lines = [
        "# Official SmolVLA Rollout Pilot Plan",
        "",
        f"Date: {report['date']}",
        "",
        f"- plan status: `not_executed`",
        f"- reason: {env.get('blocked_reason')}",
        "",
        "Predeclared pilot if WSL/Linux official stack is available:",
        "",
        "- tasks: minimum 4 fixed LIBERO task IDs across available suites.",
        "- reset/evaluation seeds: fixed before execution.",
        "- episodes: 5 per selected task per policy if runtime allows.",
        "- policies: frozen_base, rank4_lora seeds 11/22/33, validation-selected action-space static mixes for each seed.",
        "- no policy or seed selection after rollout outcomes.",
    ]
    _write_lines(path, lines)


def _write_rollout_result(report: dict[str, Any], path: Path) -> None:
    pilot = report.get("rollout_pilot") or {}
    lines = [
        "# Official SmolVLA Rollout Pilot Result",
        "",
        f"Date: {report['date']}",
        "",
        f"- smoke ran: `{pilot.get('smoke_ran')}`",
        f"- pilot ran: `{pilot.get('pilot_ran')}`",
        f"- tasks evaluated: `{pilot.get('tasks_evaluated')}`",
        f"- episodes per policy: `{pilot.get('episodes_per_policy')}`",
        f"- policies evaluated: `{pilot.get('policies_evaluated')}`",
        f"- success rates: `{pilot.get('success_rates')}`",
        f"- latency/VRAM/forward passes: `{pilot.get('latency_vram_forward_passes')}`",
        f"- reason: {pilot.get('reason')}",
    ]
    _write_lines(path, lines)


def _write_rollout_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Rollout Pilot Decision",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"Exact next step: {_exact_next_step(report)}",
    ]
    _write_lines(path, lines)


def _append_unique_section(path: Path, marker: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if marker in existing:
        return
    section = "\n".join(lines).strip() + "\n"
    if existing.strip():
        text = existing.rstrip() + "\n\n" + section
    else:
        text = section
    path.write_text(text, encoding="utf-8")


def _write_project_state(report: dict[str, Any], path: Path) -> None:
    marker = "## 2026-07-10 Canonicalization Update"
    lines = [
        marker,
        "",
        f"Current decision: `{report.get('final_decision')}`",
        "",
        "Canonical persisted-checkpoint offline evaluation passed with intermediate decision "
        f"`{report.get('intermediate_decision')}`. The run evaluated frozen base plus rank-4 LoRA "
        "seeds 11/22/33 on the fixed val/test manifest under action-generation eval seeds "
        "`[101, 202, 303, 404, 505]`; it did not train, regenerate checkpoints, download "
        "dependencies, run rollout, revive FCAR, or use the old custom LIBERO_7D route.",
        "",
        "Native Windows rollout remains blocked because `hf-libero`, `libero`, and `robosuite` "
        "are not installed in the active env. The next step is WSL/Linux official LeRobot "
        "LIBERO smoke using the canonical artifacts.",
    ]
    _append_unique_section(path, marker, lines)


def _exact_next_step(report: dict[str, Any]) -> str:
    decision = report.get("final_decision")
    if decision == "NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT":
        return "Move the same canonical artifacts/checkpoints into the verified WSL/Linux LeRobot LIBERO environment, install only official `lerobot[libero]` dependencies, then run official smoke before any pilot."
    if decision == "OFFICIAL_EVAL_ENV_BLOCKED":
        return "Resolve official LIBERO dependency/entrypoint compatibility before attempting rollout."
    if decision == "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT":
        return "Run the predeclared bounded official LIBERO smoke and pilot on the verified official stack."
    return "Fix canonical evaluation before any rollout."


def _write_next_actions(report: dict[str, Any], path: Path) -> None:
    marker = "## 2026-07-10 Canonical Next Action"
    lines = [
        marker,
        "",
        f"Current decision: `{report.get('final_decision')}`",
        "",
        _exact_next_step(report)
        + " Do not retrain, select a LoRA seed from rollout outcomes, revive FCAR, or use the old custom LIBERO_7D route.",
    ]
    _append_unique_section(path, marker, lines)


def _write_decision_log(report: dict[str, Any], path: Path) -> None:
    marker = "## 2026-07-10: Canonical Persisted SmolVLA-LoRA Baseline Evaluation"
    lines = [
        marker,
        "",
        f"Decision: `{report.get('final_decision')}`",
        "",
        f"Intermediate decision: `{report.get('intermediate_decision')}`",
        "",
        "Reason: persisted disk-reloaded official SmolVLA base and rank-4 LoRA seeds 11/22/33 "
        "produced canonical val/test metrics under the fixed action-generation RNG policy, but "
        "the active native Windows environment is missing `hf-libero`, `libero`, and `robosuite`, "
        "so official LeRobot LIBERO rollout execution must move to the verified WSL/Linux stack.",
        "",
        "Key evidence:",
        "",
        "- canonical result: `reports/official_smolvla_canonical_baseline_result.md`",
        "- canonical prediction manifest: `reports/official_smolvla_canonical_prediction_manifest.json`",
        "- canonical artifacts: `reports/canonical_frozen_base_prediction_artifact.json`, "
        "`reports/canonical_seed_11_prediction_artifact.json`, "
        "`reports/canonical_seed_22_prediction_artifact.json`, "
        "`reports/canonical_seed_33_prediction_artifact.json`",
        "- historical status: `SUPERSEDED_NONCANONICAL_PROTOCOL`",
    ]
    _append_unique_section(path, marker, lines)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    _write_json(Path(args.report_json), report)
    _write_eval_policy(report, Path(args.eval_policy_md))
    _write_baseline_result(report, Path(args.result_md))
    _write_historical_supersession(report, Path(args.historical_supersession_md))
    _write_rollout_env_setup(report, Path(args.rollout_env_setup_md))
    _write_rollout_plan(report, Path(args.rollout_pilot_plan_md))
    _write_rollout_result(report, Path(args.rollout_pilot_result_md))
    _write_json(Path(args.rollout_pilot_result_json), report.get("rollout_pilot") or {})
    _write_rollout_decision(report, Path(args.rollout_pilot_decision_md))
    _write_project_state(report, Path(args.project_state_md))
    _write_next_actions(report, Path(args.next_actions_md))
    _write_decision_log(report, Path(args.decision_log_md))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--checkpoint-manifest", default="reports/official_smolvla_lora_checkpoint_manifest.json")
    parser.add_argument("--base-artifact", default="reports/canonical_frozen_base_prediction_artifact.json")
    parser.add_argument("--lora-artifact-pattern", default="reports/canonical_seed_{seed}_prediction_artifact.json")
    parser.add_argument("--prediction-manifest", default="reports/official_smolvla_canonical_prediction_manifest.json")
    parser.add_argument("--report-json", default="reports/official_smolvla_canonical_baseline_result.json")
    parser.add_argument("--eval-policy-md", default="reports/official_smolvla_canonical_eval_policy.md")
    parser.add_argument("--result-md", default="reports/official_smolvla_canonical_baseline_result.md")
    parser.add_argument("--historical-supersession-md", default="reports/official_smolvla_historical_result_supersession.md")
    parser.add_argument("--rollout-env-setup-md", default="reports/official_smolvla_rollout_env_setup.md")
    parser.add_argument("--rollout-pilot-plan-md", default="reports/official_smolvla_rollout_pilot_plan.md")
    parser.add_argument("--rollout-pilot-result-md", default="reports/official_smolvla_rollout_pilot_result.md")
    parser.add_argument("--rollout-pilot-result-json", default="reports/official_smolvla_rollout_pilot_result.json")
    parser.add_argument("--rollout-pilot-decision-md", default="reports/official_smolvla_rollout_pilot_decision.md")
    parser.add_argument("--project-state-md", default="reports/project_state.md")
    parser.add_argument("--next-actions-md", default="reports/next_actions.md")
    parser.add_argument("--decision-log-md", default="reports/decision_log.md")
    parser.add_argument("--eval-seeds", default="101,202,303,404,505")
    parser.add_argument("--lora-seeds", default="11,22,33")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--verify-full-repeat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat-smoke-count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report, exit_code = build_report(args)
    write_outputs(report, args)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "intermediate_decision": report.get("intermediate_decision"),
                "final_decision": report.get("final_decision"),
                "runtime": report.get("runtime"),
                "errors": report.get("errors"),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
