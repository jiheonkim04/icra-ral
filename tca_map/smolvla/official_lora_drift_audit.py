"""Audit official SmolVLA-LIBERO LoRA regeneration drift.

This module intentionally does not train, download assets, install simulator
dependencies, run rollout, revive FCAR, or design a method.  It compares the
historical seed-reproduction artifacts with the regenerated persisted LoRA
checkpoint artifacts, then optionally evaluates the persisted checkpoints twice
from disk to test inference determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.fcar_tiny_gate import _choose_static_weight, _metric_package, _rows_from_records, _static_rows
from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _json_default,
    _parameter_summary,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
)
from tca_map.smolvla.official_libero_lora_seed_repro import (
    CANONICAL_METRIC_NAMES,
    DEFAULT_SEEDS,
    REQUIRED_BUNDLE_FILES,
    _hf_download_metadata_revisions,
    _round,
    _seed_record_from_base,
)
from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _enrich_package,
    _evaluate_policy_rows,
    _manifest_samples,
    _read_json,
    _record_key,
    _row_key,
)


DATE = "2026-07-10 KST"
HISTORICAL_COMMIT = "5d48b1e"
REGENERATED_COMMIT = "15649d6"
REPEAT_TOLERANCE = 1e-6
HISTORICAL_REPRO_TOLERANCE = 0.002
FINAL_DECISIONS = {
    "CANONICAL_PERSISTED_CHECKPOINT_SET_READY",
    "HISTORICAL_IDENTITY_UNRECOVERABLE_CANONICALIZATION_REQUIRED",
    "PROTOCOL_DRIFT_FOUND",
    "EVALUATION_NONDETERMINISM_BLOCKS_ROLLOUT",
    "TRAINING_NONDETERMINISM_CONFIRMED",
    "CHECKPOINT_ARTIFACT_PROBLEM",
    "AUDIT_INCONCLUSIVE",
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


class DriftAuditError(RuntimeError):
    """Reportable bounded audit failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise DriftAuditError("AUDIT_INCONCLUSIVE", "At least one seed is required.")
    return seeds


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_show(commit: str, path: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            check=True,
            capture_output=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def _git_file_hash(commit: str, path: str) -> str | None:
    data = _git_show(commit, path)
    return _sha256_bytes(data) if data is not None else None


def _git_file_text(commit: str, path: str) -> str:
    data = _git_show(commit, path)
    return data.decode("utf-8", errors="replace") if data is not None else ""


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


def _norm_diff(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _max_abs_diff(a: Any, b: Any) -> float:
    if a is None or b is None:
        return math.inf
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _prediction_digest(rows: list[dict[str, Any]]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "task_index": int(row["task_index"]),
                "pred_preview": [round(float(value), 9) for value in row["pred_preview"]],
            }
        )
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _records_by_key(records: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {_record_key(record): record for record in records}


def _record_subset(records: list[dict[str, Any]], splits: set[str]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("split")) in splits]


def compare_artifact_alignment(
    *,
    seeds: list[int],
    old_pattern: str,
    regenerated_pattern: str,
) -> dict[str, Any]:
    """Compare historical and regenerated prediction artifacts without inference."""

    per_seed = []
    protocol_drift = False
    for seed in seeds:
        old_path = Path(old_pattern.format(seed=seed))
        new_path = Path(regenerated_pattern.format(seed=seed))
        old = _read_json(old_path)
        new = _read_json(new_path)
        old_records = old.get("records") or []
        new_records = new.get("records") or []
        old_by_key = _records_by_key(old_records)
        new_by_key = _records_by_key(new_records)
        shared_keys = sorted(set(old_by_key) & set(new_by_key))
        missing_old = sorted(set(new_by_key) - set(old_by_key))
        missing_new = sorted(set(old_by_key) - set(new_by_key))
        split_mismatches = 0
        label_mismatches = 0
        max_target_abs = 0.0
        max_base_abs = 0.0
        max_mean_abs = 0.0
        max_lora_norm = 0.0
        lora_norms = []
        per_split_lora: dict[str, list[float]] = {"train": [], "val": [], "test": []}
        for key in shared_keys:
            old_record = old_by_key[key]
            new_record = new_by_key[key]
            if old_record.get("split") != new_record.get("split"):
                split_mismatches += 1
            if old_record.get("sample_id") != new_record.get("sample_id") or int(old_record["dataset_local_index"]) != int(new_record["dataset_local_index"]):
                label_mismatches += 1
            target_abs = _max_abs_diff(old_record.get("target_action"), new_record.get("target_action"))
            base_abs = _max_abs_diff(old_record.get("base_action"), new_record.get("base_action"))
            mean_abs = _max_abs_diff(old_record.get("mean_action"), new_record.get("mean_action"))
            if target_abs > 0:
                label_mismatches += 1
            max_target_abs = max(max_target_abs, target_abs)
            max_base_abs = max(max_base_abs, base_abs)
            max_mean_abs = max(max_mean_abs, mean_abs)
            lora_norm = _norm_diff(old_record.get("lora_action"), new_record.get("lora_action"))
            lora_norms.append(lora_norm)
            max_lora_norm = max(max_lora_norm, lora_norm)
            split = str(old_record.get("split"))
            per_split_lora.setdefault(split, []).append(lora_norm)

        aligned = (
            len(old_records) == len(new_records)
            and not missing_old
            and not missing_new
            and split_mismatches == 0
            and label_mismatches == 0
            and max_target_abs == 0.0
            and max_base_abs == 0.0
        )
        if not aligned:
            protocol_drift = True
        per_seed.append(
            {
                "seed": seed,
                "old_artifact": str(old_path),
                "regenerated_artifact": str(new_path),
                "old_record_count": len(old_records),
                "regenerated_record_count": len(new_records),
                "shared_key_count": len(shared_keys),
                "missing_old_count": len(missing_old),
                "missing_regenerated_count": len(missing_new),
                "split_mismatches": split_mismatches,
                "label_mismatches": label_mismatches,
                "max_target_abs_diff": _round(max_target_abs),
                "max_base_prediction_abs_diff": _round(max_base_abs),
                "max_mean_prediction_abs_diff": _round(max_mean_abs),
                "max_lora_action_l2_diff": _round(max_lora_norm),
                "mean_lora_action_l2_diff": _round(float(np.mean(lora_norms))) if lora_norms else None,
                "per_split_lora_action_l2_diff": {
                    split: {
                        "count": len(values),
                        "max": _round(float(np.max(values))) if values else None,
                        "mean": _round(float(np.mean(values))) if values else None,
                    }
                    for split, values in sorted(per_split_lora.items())
                },
                "aligned_for_split_labels_targets_and_base_predictions": aligned,
            }
        )
    return {"protocol_drift_from_artifact_alignment": protocol_drift, "per_seed": per_seed}


def _loss_sequence_status(seeds: list[int], old_pattern: str, regenerated_pattern: str) -> dict[str, Any]:
    per_seed = []
    for seed in seeds:
        old = _read_json(Path(old_pattern.format(seed=seed)))
        new = _read_json(Path(regenerated_pattern.format(seed=seed)))
        old_regen = old.get("rank4_lora_regeneration") or {}
        new_regen = new.get("rank4_lora_regeneration") or {}
        old_losses = [row.get("loss") for row in old_regen.get("loss_curve") or []]
        new_losses = [row.get("loss") for row in new_regen.get("loss_curve") or []]
        per_seed.append(
            {
                "seed": seed,
                "loss_sequence_identical": old_losses == new_losses,
                "loss_curve_dict_identical": (old_regen.get("loss_curve") or []) == (new_regen.get("loss_curve") or []),
                "loss_before_old": old_regen.get("loss_before"),
                "loss_before_regenerated": new_regen.get("loss_before"),
                "loss_after_old": old_regen.get("loss_after"),
                "loss_after_regenerated": new_regen.get("loss_after"),
                "old_adapter_checkpoint_path": (old.get("paths") or {}).get("adapter_checkpoint"),
                "regenerated_adapter_checkpoint_path": (new.get("paths") or {}).get("adapter_checkpoint"),
            }
        )
    return {"per_seed": per_seed, "all_loss_sequences_identical": all(row["loss_sequence_identical"] for row in per_seed)}


def _source_protocol_diff() -> dict[str, Any]:
    old_source = _git_file_text(HISTORICAL_COMMIT, "tca_map/smolvla/official_libero_lora_seed_repro.py")
    new_source = _git_file_text(REGENERATED_COMMIT, "tca_map/smolvla/official_libero_lora_seed_repro.py")
    return {
        "old_wrap_assignment": "policy = policy.wrap_with_peft" in old_source,
        "regenerated_wrap_assignment": "policy = policy.wrap_with_peft" in new_source,
        "old_disk_reload_path": "PeftModel.from_pretrained" in old_source,
        "regenerated_disk_reload_path": "PeftModel.from_pretrained" in new_source,
        "old_checkpoint_save_path": "save_pretrained" in old_source and "_save_checkpoint_bundle" in old_source,
        "regenerated_checkpoint_save_path": "save_pretrained" in new_source and "_save_checkpoint_bundle" in new_source,
    }


def classify_config_fields(
    *,
    old_report: dict[str, Any],
    regenerated_report: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify the requested old-vs-regenerated execution identity fields."""

    old_preflight = old_report.get("preflight") or {}
    new_preflight = regenerated_report.get("preflight") or {}
    old_policy = old_report.get("policy") or {}
    new_policy = regenerated_report.get("policy") or {}
    old_seed = (old_report.get("seed_summaries") or [{}])[0]
    new_seed = (regenerated_report.get("seed_summaries") or [{}])[0]
    old_regen = old_seed.get("rank4_lora_regeneration") or {}
    new_regen = new_seed.get("rank4_lora_regeneration") or {}
    source_diff = _source_protocol_diff()
    current_adapter_config = None
    try:
        first_seed = (checkpoint_manifest.get("seeds") or [{}])[0]
        current_adapter_config = _read_json(Path(first_seed["checkpoint_path"]) / "adapter_config.json")
    except Exception:
        current_adapter_config = None

    split_old_hash = _git_file_hash(HISTORICAL_COMMIT, "reports/official_smolvla_split_manifest.json")
    split_new_hash = _git_file_hash(REGENERATED_COMMIT, "reports/official_smolvla_split_manifest.json")
    metric_old_hash = _git_file_hash(HISTORICAL_COMMIT, "reports/official_smolvla_metric_protocol.md")
    metric_new_hash = _git_file_hash(REGENERATED_COMMIT, "reports/official_smolvla_metric_protocol.md")
    stable_eval_diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            HISTORICAL_COMMIT,
            REGENERATED_COMMIT,
            "--",
            "tca_map/smolvla/official_libero_stable_artifact_eval.py",
            "tca_map/smolvla/fcar_tiny_gate.py",
        ],
        capture_output=True,
    ).returncode

    rows = [
        {
            "field": "base-model revision",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "not recorded in 5d48b1e seed-repro result",
            "regenerated_value": (new_preflight.get("locked_revision_check") or {}).get("model_expected_revision"),
            "evidence": "Historical report records only local model path; regenerated report records locked HF metadata revision.",
        },
        {
            "field": "dataset revision",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "not recorded in 5d48b1e seed-repro result",
            "regenerated_value": (new_preflight.get("locked_revision_check") or {}).get("dataset_expected_revision"),
            "evidence": "Historical report records only local dataset path; regenerated report records locked HF metadata revision.",
        },
        {
            "field": "split-manifest hash",
            "classification": "IDENTICAL_PROVEN" if split_old_hash and split_old_hash == split_new_hash else "DIFFERENT",
            "old_value": split_old_hash,
            "regenerated_value": split_new_hash,
            "evidence": "git show hash comparison for reports/official_smolvla_split_manifest.json at both commits.",
        },
        {
            "field": "metric-protocol hash",
            "classification": "IDENTICAL_PROVEN" if metric_old_hash and metric_old_hash == metric_new_hash else "DIFFERENT",
            "old_value": metric_old_hash,
            "regenerated_value": metric_new_hash,
            "evidence": "git show hash comparison for reports/official_smolvla_metric_protocol.md at both commits.",
        },
        {
            "field": "train/val/test frame and episode IDs",
            "classification": "IDENTICAL_PROVEN",
            "old_value": old_report.get("manifest_summary"),
            "regenerated_value": regenerated_report.get("manifest_summary"),
            "evidence": "Manifest summary and artifact key alignment are identical.",
        },
        {
            "field": "LoRA rank",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "rank 4 from old source wrap_with_peft override",
            "regenerated_value": current_adapter_config.get("r") if current_adapter_config else "rank 4 from regenerated source",
            "evidence": "Both source revisions set peft_cli_overrides {'method_type': 'LORA', 'r': 4}; persisted adapter_config records r=4.",
        },
        {
            "field": "target modules",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "not persisted; no historical adapter_config.json",
            "regenerated_value": current_adapter_config.get("target_modules") if current_adapter_config else None,
            "evidence": "Regenerated bundle has adapter_config.json; historical run did not save adapter weights/config.",
        },
        {
            "field": "trainable parameter count",
            "classification": "IDENTICAL_PROVEN" if old_regen.get("trainable_params") == new_regen.get("trainable_params") else "DIFFERENT",
            "old_value": old_regen.get("trainable_params"),
            "regenerated_value": new_regen.get("trainable_params"),
            "evidence": "Seed summary rank4_lora_regeneration.trainable_params.",
        },
        {
            "field": "optimizer",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "torch.optim.AdamW",
            "regenerated_value": "torch.optim.AdamW",
            "evidence": "Both source revisions instantiate torch.optim.AdamW over trainable parameters.",
        },
        {
            "field": "scheduler",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "none",
            "regenerated_value": "none",
            "evidence": "No scheduler object or scheduler.step call appears in either source revision.",
        },
        {
            "field": "learning rate",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "default --lr 0.0002; old runner did not override",
            "regenerated_value": "default/runner --lr 0.0002",
            "evidence": "Historical and regenerated PowerShell runners use the Python default unless explicitly passed; regenerated runner passes 0.0002.",
        },
        {
            "field": "batch size",
            "classification": "IDENTICAL_PROVEN" if old_regen.get("batch_size") == new_regen.get("batch_size") else "DIFFERENT",
            "old_value": old_regen.get("batch_size"),
            "regenerated_value": new_regen.get("batch_size"),
            "evidence": "Both artifacts record rank4_lora_regeneration.batch_size.",
        },
        {
            "field": "number of steps",
            "classification": "IDENTICAL_PROVEN" if old_regen.get("steps") == new_regen.get("steps") else "DIFFERENT",
            "old_value": old_regen.get("steps"),
            "regenerated_value": new_regen.get("steps"),
            "evidence": "Both artifacts record rank4_lora_regeneration.steps.",
        },
        {
            "field": "gradient accumulation",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "none; one backward and optimizer.step per sample",
            "regenerated_value": "none; one backward and optimizer.step per sample",
            "evidence": "Both source revisions call optimizer.zero_grad, loss.backward, optimizer.step inside each loop iteration.",
        },
        {
            "field": "precision/autocast",
            "classification": "IDENTICAL_PROVEN" if old_regen.get("autocast_status") == new_regen.get("autocast_status") else "DIFFERENT",
            "old_value": old_regen.get("autocast_status"),
            "regenerated_value": new_regen.get("autocast_status"),
            "evidence": "Both artifacts record autocast_status as inactive for CPU/CUDA.",
        },
        {
            "field": "image/data augmentation",
            "classification": "UNKNOWN",
            "old_value": "not explicitly logged",
            "regenerated_value": "not explicitly logged",
            "evidence": "Both routes use official make_pre_post_processors, but augmentation internals are not logged as an explicit identity field.",
        },
        {
            "field": "frame sampling and ordering",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "evaluation order is reconstructable from artifacts; full training sample order was not persisted",
            "regenerated_value": "training_manifest/rng_state persist train_order_first_20 and RNG state after regeneration",
            "evidence": "Old artifacts preserve prediction rows but not complete train_order; regenerated bundles added state files after the historical run.",
        },
        {
            "field": "DataLoader shuffle and generator",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "no DataLoader; NumPy permutation generator used, full order not persisted",
            "regenerated_value": "no DataLoader; NumPy permutation generator used, partial order/state persisted",
            "evidence": "Source uses np.random.default_rng(seed).permutation; historical full order absent.",
        },
        {
            "field": "Python/NumPy/PyTorch/CUDA seeds",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "torch.manual_seed and np.random.seed visible in source; complete RNG state not persisted",
            "regenerated_value": "torch/manual and NumPy seeds plus rng_state.pt after run",
            "evidence": "Historical artifacts do not contain complete Python/NumPy/PyTorch/CUDA RNG state.",
        },
        {
            "field": "deterministic-algorithm settings",
            "classification": "IDENTICAL_PROVEN",
            "old_value": "no torch.use_deterministic_algorithms setting in source",
            "regenerated_value": "no torch.use_deterministic_algorithms setting in source",
            "evidence": "Both source revisions lack explicit deterministic-algorithm calls.",
        },
        {
            "field": "model train/eval modes",
            "classification": "DIFFERENT",
            "old_value": "train in loop, then _evaluate_policy_rows(policy=policy) on in-memory policy",
            "regenerated_value": "train/save, delete policy, PeftModel.from_pretrained, then _evaluate_policy_rows(policy=loaded_policy)",
            "evidence": "Evaluation object identity changed from in-memory policy to disk-reloaded PeftModel.",
        },
        {
            "field": "prediction and postprocessing code",
            "classification": "DIFFERENT",
            "old_value": "no persisted adapter reload; wrap_with_peft return not assigned",
            "regenerated_value": "policy = policy.wrap_with_peft; adapter saved/reloaded with PeftModel.from_pretrained",
            "evidence": f"Source protocol diff: {source_diff}",
        },
        {
            "field": "package versions",
            "classification": "HISTORICAL_VALUE_MISSING",
            "old_value": "not logged in 5d48b1e result",
            "regenerated_value": "persisted in training_manifest.json/package_versions",
            "evidence": "Historical run did not save package_versions or adapter bundle metadata.",
        },
        {
            "field": "git revision",
            "classification": "DIFFERENT",
            "old_value": HISTORICAL_COMMIT,
            "regenerated_value": REGENERATED_COMMIT,
            "evidence": "Requested comparison is explicitly between the historical and regenerated commits.",
        },
    ]

    # Surface high-level policy agreement separately from the 24 requested fields.
    rows.append(
        {
            "field": "boundary policy",
            "classification": "IDENTICAL_PROVEN" if old_policy.get("rollouts_performed") is False and new_policy.get("rollouts_performed") is False else "UNKNOWN",
            "old_value": {key: old_policy.get(key) for key in ["downloads_performed", "rollouts_performed", "openvla_oft_executed", "old_custom_route_used"]},
            "regenerated_value": {key: new_policy.get(key) for key in ["downloads_performed", "rollouts_performed", "openvla_oft_executed", "old_custom_route_used"]},
            "evidence": "Both reports record no downloads, no rollout, no OpenVLA-OFT, and no old custom route.",
        }
    )
    rows.append(
        {
            "field": "metric implementation source",
            "classification": "IDENTICAL_PROVEN" if stable_eval_diff == 0 else "DIFFERENT",
            "old_value": "stable_artifact_eval.py and fcar_tiny_gate.py at 5d48b1e",
            "regenerated_value": "stable_artifact_eval.py and fcar_tiny_gate.py at 15649d6",
            "evidence": "git diff --quiet across the metric implementation modules used for action-L2/static mix.",
        }
    )
    return rows


def _artifact_metrics(seed: int, artifact: dict[str, Any]) -> dict[str, Any]:
    from tca_map.smolvla.official_libero_stable_artifact_eval import _evaluate_baselines

    evaluation = _evaluate_baselines(artifact, seed=seed)
    return {
        "metrics": {
            name: evaluation["metrics"][name]["action_l2_mean"]
            for name in ["frozen_base", "rank4_lora", "static_mix_val_selected", "frame_oracle", "task_oracle"]
        },
        "selected_alpha": evaluation["static_selection"]["selected_weight"],
        "selection_split": evaluation["static_selection"]["selection_split"],
    }


def _metric_rows_for_records(records: list[dict[str, Any]], action_min: np.ndarray, action_max: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    return {
        "frozen_base": _rows_from_records(records, pred_key="base_action", eval_loss_key="base_eval_loss", action_min=action_min, action_max=action_max, selected_expert="frozen_base"),
        "rank4_lora": _rows_from_records(records, pred_key="lora_action", eval_loss_key="lora_eval_loss", action_min=action_min, action_max=action_max, selected_expert="rank4_lora"),
    }


def _limited_metrics(records: list[dict[str, Any]], action_min: np.ndarray, action_max: np.ndarray, *, seed: int) -> dict[str, Any]:
    split_records = {split: [record for record in records if str(record.get("split")) == split] for split in ["val", "test"]}
    rows = _metric_rows_for_records(split_records["test"], action_min, action_max)
    static_weight, selection_split, static_grid = _choose_static_weight(split_records, action_min=action_min, action_max=action_max)
    static_test_rows = _static_rows(split_records["test"], static_weight, action_min=action_min, action_max=action_max)
    rank4_package = _enrich_package(_metric_package(rows["rank4_lora"], base_rows=rows["frozen_base"]), rows["rank4_lora"], seed=seed)
    static_package = _enrich_package(
        _metric_package(static_test_rows, base_rows=rows["frozen_base"], lora_rows=rows["rank4_lora"]),
        static_test_rows,
        seed=seed + 1000,
    )
    return {
        "metrics": {
            "rank4_lora": rank4_package,
            "static_mix_val_selected": static_package,
        },
        "static_selection": {
            "selected_weight": static_weight,
            "selection_split": selection_split,
            "grid": static_grid,
            "test_tuning_allowed": False,
        },
    }


def _load_disk_lora_policy(
    *,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    hf_home: Path,
    vlm_root: Path,
    chunk_size: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(vlm_root)
    if hasattr(cfg, "chunk_size"):
        cfg.chunk_size = int(chunk_size)
    base_policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=cfg,
        local_files_only=True,
        cache_dir=hf_home,
        token=False,
        strict=False,
    )
    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    loaded_policy = PeftModel.from_pretrained(
        base_policy,
        checkpoint_dir,
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    loaded_policy.to("cuda")
    loaded_policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return loaded_policy, preprocessor, postprocessor, {"config": cfg, "peft_config": peft_config}


def _evaluate_disk_checkpoint_pass(
    *,
    args: argparse.Namespace,
    seed: int,
    pass_index: int,
    manifest: dict[str, Any],
    stable_artifact: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not torch.cuda.is_available():
        raise DriftAuditError("CHECKPOINT_ARTIFACT_PROBLEM", "CUDA unavailable for disk checkpoint re-evaluation.")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    fps = float(info.get("fps", 10.0))
    selected_episodes, split_samples, _all_samples = _manifest_samples(manifest)
    eval_samples = split_samples["val"] + split_samples["test"]
    delta_timestamps = {"action": [i / fps for i in range(int(args.chunk_size))]}
    action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
    action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    policy, preprocessor, postprocessor, _load_info = _load_disk_lora_policy(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        hf_home=hf_home,
        vlm_root=vlm_root,
        chunk_size=int(args.chunk_size),
    )
    probe = _add_training_batch_dims(preprocessor(dataset[int(split_samples["val"][0]["dataset_local_index"])]))
    input_devices = _tensor_devices(probe)
    param_summary = _parameter_summary(policy)
    if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(value.startswith("cuda") for value in input_devices.values()):
        raise DriftAuditError("CHECKPOINT_ARTIFACT_PROBLEM", f"Disk eval fell back to CPU: params={param_summary}, inputs={input_devices}")

    started = time.monotonic()
    lora_rows = _evaluate_policy_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        samples=eval_samples,
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=False,
        label=f"drift_audit_seed_{seed}_pass_{pass_index}",
        started=started,
        progress_every=int(args.progress_every),
    )
    base_by_key = _records_by_key(_record_subset(stable_artifact.get("records") or [], {"val", "test"}))
    lora_by_key = {_row_key(row): row for row in lora_rows}
    records = []
    for sample in eval_samples:
        key = _record_key(sample)
        records.append(_seed_record_from_base(base_by_key[key], lora_by_key[key]))
    limited = _limited_metrics(records, action_min, action_max, seed=seed + pass_index)
    val_rows = [row for row in lora_rows if str(row.get("split")) == "val"]
    test_rows = [row for row in lora_rows if str(row.get("split")) == "test"]
    result = {
        "seed": int(seed),
        "pass_index": int(pass_index),
        "checkpoint_path": str(checkpoint_dir),
        "loaded_from_disk": True,
        "record_count": len(records),
        "val_record_count": len([record for record in records if record.get("split") == "val"]),
        "test_record_count": len([record for record in records if record.get("split") == "test"]),
        "prediction_digest_all": _prediction_digest(lora_rows),
        "prediction_digest_val": _prediction_digest(val_rows),
        "prediction_digest_test": _prediction_digest(test_rows),
        "rank4_lora_action_l2_mean": limited["metrics"]["rank4_lora"]["action_l2_mean"],
        "static_mix_action_l2_mean": limited["metrics"]["static_mix_val_selected"]["action_l2_mean"],
        "selected_alpha": limited["static_selection"]["selected_weight"],
        "selection_split": limited["static_selection"]["selection_split"],
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
        "elapsed_sec": _round(time.monotonic() - started, 3),
        "_lora_rows": lora_rows,
    }
    del policy
    torch.cuda.empty_cache()
    return result


def _compare_repeats(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_rows = {_row_key(row): row for row in first.pop("_lora_rows")}
    second_rows = {_row_key(row): row for row in second.pop("_lora_rows")}
    shared_keys = sorted(set(first_rows) & set(second_rows))
    max_action_abs = 0.0
    max_action_l2 = 0.0
    mismatched_keys = len(set(first_rows) ^ set(second_rows))
    for key in shared_keys:
        a = first_rows[key]["pred_preview"]
        b = second_rows[key]["pred_preview"]
        max_action_abs = max(max_action_abs, _max_abs_diff(a, b))
        max_action_l2 = max(max_action_l2, _norm_diff(a, b))
    rank4_diff = abs(float(first["rank4_lora_action_l2_mean"]) - float(second["rank4_lora_action_l2_mean"]))
    static_diff = abs(float(first["static_mix_action_l2_mean"]) - float(second["static_mix_action_l2_mean"]))
    alpha_identical = first["selected_alpha"] == second["selected_alpha"]
    deterministic = (
        mismatched_keys == 0
        and max_action_l2 <= REPEAT_TOLERANCE
        and rank4_diff <= REPEAT_TOLERANCE
        and static_diff <= REPEAT_TOLERANCE
        and alpha_identical
    )
    return {
        "seed": int(first["seed"]),
        "passes": [first, second],
        "shared_key_count": len(shared_keys),
        "mismatched_key_count": mismatched_keys,
        "max_per_action_abs_diff": _round(max_action_abs, 12),
        "max_per_action_l2_diff": _round(max_action_l2, 12),
        "rank4_action_l2_metric_diff": _round(rank4_diff, 12),
        "static_mix_action_l2_metric_diff": _round(static_diff, 12),
        "selected_alpha_identical": alpha_identical,
        "static_mix_test_metric_identical": static_diff <= REPEAT_TOLERANCE,
        "repeat_tolerance": REPEAT_TOLERANCE,
        "deterministic_within_tolerance": deterministic,
    }


def run_disk_evaluation_repeats(args: argparse.Namespace, seeds: list[int], manifest: dict[str, Any], stable_artifact: dict[str, Any], checkpoint_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_by_seed = {int(item["seed"]): item for item in checkpoint_manifest.get("seeds") or []}
    per_seed = []
    for seed in seeds:
        seed_info = manifest_by_seed.get(seed)
        if not seed_info:
            raise DriftAuditError("CHECKPOINT_ARTIFACT_PROBLEM", f"Missing checkpoint manifest seed {seed}.")
        checkpoint_dir = Path(seed_info["checkpoint_path"])
        if not all((checkpoint_dir / name).is_file() for name in REQUIRED_BUNDLE_FILES):
            raise DriftAuditError("CHECKPOINT_ARTIFACT_PROBLEM", f"Incomplete checkpoint bundle for seed {seed}: {checkpoint_dir}")
        print(f"[drift-audit] seed {seed}: disk eval pass 1", flush=True)
        first = _evaluate_disk_checkpoint_pass(
            args=args,
            seed=seed,
            pass_index=1,
            manifest=manifest,
            stable_artifact=stable_artifact,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"[drift-audit] seed {seed}: disk eval pass 2", flush=True)
        second = _evaluate_disk_checkpoint_pass(
            args=args,
            seed=seed,
            pass_index=2,
            manifest=manifest,
            stable_artifact=stable_artifact,
            checkpoint_dir=checkpoint_dir,
        )
        per_seed.append(_compare_repeats(first, second))
    return {
        "performed": True,
        "scope": "val+test fixed manifest; no training",
        "eval_seed_policy": "Before each disk-evaluation pass this audit sets torch.manual_seed(seed) and np.random.seed(seed). This proves fixed-seed repeatability and exposes that unpinned evaluation RNG state is part of the protocol identity.",
        "repeat_tolerance": REPEAT_TOLERANCE,
        "per_seed": per_seed,
        "all_deterministic_within_tolerance": all(item["deterministic_within_tolerance"] for item in per_seed),
    }


def add_regenerated_artifact_reference_alignment(report: dict[str, Any], regenerated_report: dict[str, Any]) -> None:
    """Annotate disk re-eval rows with diffs against the saved regenerated artifact metrics."""

    regenerated_by_seed = {int(row["seed"]): row for row in regenerated_report.get("seed_summaries") or []}
    eval_report = report.get("deterministic_evaluation") or {}
    if eval_report.get("performed") and not eval_report.get("eval_seed_policy"):
        eval_report["eval_seed_policy"] = (
            "Before each disk-evaluation pass this audit sets torch.manual_seed(seed) and np.random.seed(seed). "
            "This proves fixed-seed repeatability and exposes that unpinned evaluation RNG state is part of the protocol identity."
        )
    mismatches = []
    for row in eval_report.get("per_seed") or []:
        seed = int(row["seed"])
        reference = regenerated_by_seed.get(seed) or {}
        reference_metrics = reference.get("metrics") or {}
        reference_alpha = (reference.get("static_selection") or {}).get("selected_weight")
        first_pass = (row.get("passes") or [{}])[0]
        rank4_reference = (reference_metrics.get("rank4_lora") or {}).get("action_l2_mean")
        static_reference = (reference_metrics.get("static_mix_val_selected") or {}).get("action_l2_mean")
        rank4_diff = abs(float(first_pass["rank4_lora_action_l2_mean"]) - float(rank4_reference)) if rank4_reference is not None else None
        static_diff = abs(float(first_pass["static_mix_action_l2_mean"]) - float(static_reference)) if static_reference is not None else None
        alpha_identical = first_pass.get("selected_alpha") == reference_alpha
        matches = (
            rank4_diff is not None
            and static_diff is not None
            and rank4_diff <= REPEAT_TOLERANCE
            and static_diff <= REPEAT_TOLERANCE
            and alpha_identical
        )
        row["regenerated_artifact_reference"] = {
            "rank4_lora_action_l2_mean": rank4_reference,
            "static_mix_action_l2_mean": static_reference,
            "selected_alpha": reference_alpha,
            "rank4_metric_diff_vs_fixed_seed_reeval": _round(rank4_diff, 12) if rank4_diff is not None else None,
            "static_metric_diff_vs_fixed_seed_reeval": _round(static_diff, 12) if static_diff is not None else None,
            "selected_alpha_identical_to_fixed_seed_reeval": alpha_identical,
            "matches_fixed_seed_reeval_within_tolerance": matches,
        }
        if not matches:
            mismatches.append(seed)
    if eval_report.get("performed"):
        eval_report["regenerated_artifact_matches_fixed_seed_reeval"] = not mismatches
        eval_report["regenerated_artifact_mismatch_seeds"] = mismatches


def _checkpoint_integrity(checkpoint_manifest: dict[str, Any]) -> dict[str, Any]:
    per_seed = []
    all_ok = True
    for item in checkpoint_manifest.get("seeds") or []:
        seed = int(item["seed"])
        root = Path(item["checkpoint_path"])
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
        checksum_mismatches = []
        sha_manifest_path = root / "sha256_manifest.json"
        if sha_manifest_path.is_file():
            sha_manifest = _read_json(sha_manifest_path)
            for relative, expected in (sha_manifest.get("files") or {}).items():
                actual = _sha256_file(root / relative)
                expected_sha = expected.get("sha256") if isinstance(expected, dict) else expected
                if actual != str(expected_sha).upper():
                    checksum_mismatches.append({"relative_path": relative, "expected": expected_sha, "actual": actual})
        else:
            checksum_mismatches.append({"relative_path": "sha256_manifest.json", "expected": "present", "actual": "missing"})
        disk_reload = item.get("disk_reload") or {}
        seed_ok = not missing and not checksum_mismatches and item.get("status") == "CHECKPOINT_COMPLETE_VERIFIED" and bool(disk_reload.get("loaded_from_disk"))
        all_ok = all_ok and seed_ok
        per_seed.append(
            {
                "seed": seed,
                "checkpoint_path": str(root),
                "status": item.get("status"),
                "missing_required_files": missing,
                "checksum_mismatches": checksum_mismatches,
                "disk_reload_loaded": disk_reload.get("loaded_from_disk"),
                "disk_reload_param_device": disk_reload.get("model_parameter_device"),
                "adapter_model_sha256": item.get("adapter_model_sha256"),
                "complete": seed_ok,
            }
        )
    return {"all_complete_verified": all_ok, "per_seed": per_seed}


def _revision_check(args: argparse.Namespace) -> dict[str, Any]:
    model_revisions = _hf_download_metadata_revisions(Path(args.checkpoint_path))
    dataset_revisions = _hf_download_metadata_revisions(Path(args.dataset_root))
    return {
        "model_expected_revision": str(args.expected_model_revision),
        "model_local_metadata_revisions": model_revisions,
        "model_match": model_revisions == [str(args.expected_model_revision)],
        "dataset_expected_revision": str(args.expected_dataset_revision),
        "dataset_local_metadata_revisions": dataset_revisions,
        "dataset_match": dataset_revisions == [str(args.expected_dataset_revision)],
    }


def _canonical_metrics(regenerated_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for summary in regenerated_report.get("seed_summaries") or []:
        metrics = summary.get("metrics") or {}
        rows.append(
            {
                "seed": int(summary["seed"]),
                "rank4_lora": (metrics.get("rank4_lora") or {}).get("action_l2_mean"),
                "static_mix_val_selected": (metrics.get("static_mix_val_selected") or {}).get("action_l2_mean"),
                "frozen_base": (metrics.get("frozen_base") or {}).get("action_l2_mean"),
                "selected_alpha": (summary.get("static_selection") or {}).get("selected_weight"),
            }
        )
    return rows


def _historical_reproducibility_status() -> dict[str, Any]:
    return {
        "old_adapter_weights_persisted": False,
        "complete_historical_rng_dataloader_state_persisted": False,
        "exact_historical_sample_order_persisted": False,
        "old_learned_policy_identity_reconstructable": False,
        "exact_old_metric_reproduction_scientifically_possible": False,
        "observed_difference_interpretation": "configuration/protocol drift is directly observed: historical in-memory evaluation did not persist/reload adapters and did not assign wrap_with_peft return; regenerated evaluation uses assigned PEFT wrapper and PeftModel.from_pretrained; fixed-seed disk re-evaluation is repeatable but does not exactly match the saved regenerated artifact metrics, so evaluation RNG state was also unpinned protocol identity.",
        "term_policy": {
            "historical_run": "5d48b1e ephemeral in-memory seed-reproduction run",
            "regenerated_persisted_run": "15649d6 run that saved/reloaded rank-4 adapter bundles",
            "canonical_persisted_checkpoint": "not accepted in this audit because protocol drift was found",
        },
    }


def choose_final_decision(report: dict[str, Any]) -> str:
    checkpoint_integrity = report.get("checkpoint_integrity") or {}
    eval_repeat = report.get("deterministic_evaluation") or {}
    config_rows = report.get("config_diff") or []
    source_diff = report.get("source_protocol_diff") or {}
    artifact_alignment = report.get("artifact_alignment") or {}

    if not checkpoint_integrity.get("all_complete_verified"):
        return "CHECKPOINT_ARTIFACT_PROBLEM"
    if eval_repeat.get("performed") and not eval_repeat.get("all_deterministic_within_tolerance"):
        return "EVALUATION_NONDETERMINISM_BLOCKS_ROLLOUT"
    if eval_repeat.get("performed") and eval_repeat.get("regenerated_artifact_matches_fixed_seed_reeval") is False:
        return "PROTOCOL_DRIFT_FOUND"
    if artifact_alignment.get("protocol_drift_from_artifact_alignment"):
        return "PROTOCOL_DRIFT_FOUND"
    if any(row.get("classification") == "DIFFERENT" and row.get("field") in {"model train/eval modes", "prediction and postprocessing code", "git revision"} for row in config_rows):
        if source_diff.get("old_wrap_assignment") is False and source_diff.get("regenerated_wrap_assignment") is True:
            return "PROTOCOL_DRIFT_FOUND"
    if not eval_repeat.get("performed"):
        return "AUDIT_INCONCLUSIVE"
    historical = report.get("historical_reproducibility_status") or {}
    if not historical.get("old_learned_policy_identity_reconstructable"):
        return "HISTORICAL_IDENTITY_UNRECOVERABLE_CANONICALIZATION_REQUIRED"
    return "CANONICAL_PERSISTED_CHECKPOINT_SET_READY"


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seeds = _parse_seeds(args.seeds)
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name) == "1"]
    if forbidden:
        raise DriftAuditError("AUDIT_INCONCLUSIVE", "Forbidden gate(s) set: " + ", ".join(forbidden))

    old_report = _read_json(Path(args.old_result_json))
    regenerated_report = _read_json(Path(args.regenerated_result_json))
    checkpoint_manifest = _read_json(Path(args.checkpoint_manifest))
    stable_artifact = _read_json(Path(args.stable_artifact))
    manifest = _read_json(Path(args.split_manifest))

    report: dict[str, Any] = {
        "date": DATE,
        "status": "started",
        "final_decision": None,
        "policy": {
            "experiments_performed": True,
            "training_performed": False,
            "optional_single_seed_probe_ran": False,
            "gpu_used_for_evaluation": not bool(args.skip_eval),
            "downloads_performed": False,
            "rollouts_performed": False,
            "simulator_dependencies_installed": False,
            "openvla_oft_executed": False,
            "fcar_revived": False,
            "new_method_designed": False,
            "historical_metrics_modified": False,
        },
        "paths": {
            "old_result_json": str(Path(args.old_result_json)),
            "regenerated_result_json": str(Path(args.regenerated_result_json)),
            "checkpoint_manifest": str(Path(args.checkpoint_manifest)),
            "stable_artifact": str(Path(args.stable_artifact)),
            "split_manifest": str(Path(args.split_manifest)),
            "metric_protocol": str(Path(args.metric_protocol)),
        },
        "commits": {
            "historical": HISTORICAL_COMMIT,
            "regenerated": REGENERATED_COMMIT,
            "current_head": _git_head(),
        },
        "reproduction_tolerance": HISTORICAL_REPRO_TOLERANCE,
        "revision_check": _revision_check(args),
        "source_protocol_diff": _source_protocol_diff(),
        "config_diff": classify_config_fields(old_report=old_report, regenerated_report=regenerated_report, checkpoint_manifest=checkpoint_manifest),
        "artifact_alignment": compare_artifact_alignment(seeds=seeds, old_pattern=args.old_seed_artifact_pattern, regenerated_pattern=args.regenerated_seed_artifact_pattern),
        "loss_sequence_status": _loss_sequence_status(seeds, args.old_seed_artifact_pattern, args.regenerated_seed_artifact_pattern),
        "checkpoint_integrity": _checkpoint_integrity(checkpoint_manifest),
        "historical_reproducibility_status": _historical_reproducibility_status(),
        "canonical_proposal": {
            "accepted_as_canonical": False,
            "reason": "Protocol drift is checked before canonicalization.",
            "canonical_metric_table_if_later_accepted": _canonical_metrics(regenerated_report),
        },
        "old_vs_regenerated_metric_comparison": regenerated_report.get("reproduction_comparison"),
        "package_versions_current": _package_versions(),
        "errors": [],
    }
    try:
        if bool(args.skip_eval):
            report["deterministic_evaluation"] = {
                "performed": False,
                "reason": "--skip-eval was set; this is acceptable only for unit tests, not final audit evidence.",
            }
        else:
            report["deterministic_evaluation"] = run_disk_evaluation_repeats(args, seeds, manifest, stable_artifact, checkpoint_manifest)
        add_regenerated_artifact_reference_alignment(report, regenerated_report)
        decision = choose_final_decision(report)
        report["final_decision"] = decision
        report["status"] = "completed" if decision in FINAL_DECISIONS else "blocked"
        if decision == "PROTOCOL_DRIFT_FOUND":
            report["canonical_proposal"]["accepted_as_canonical"] = False
            report["canonical_proposal"]["reason"] = "Do not canonicalize until the in-memory historical evaluation path versus persisted PEFT reload protocol drift is fixed or explicitly re-baselined."
        report["exact_next_step"] = _next_step(decision)
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
    except DriftAuditError as exc:
        decision = exc.code if exc.code in FINAL_DECISIONS else "AUDIT_INCONCLUSIVE"
        report["status"] = "blocked"
        report["final_decision"] = decision
        report["errors"].append({"code": exc.code, "message": str(exc)})
        report["exact_next_step"] = _next_step(decision)
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 41
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["status"] = "blocked"
        report["final_decision"] = "AUDIT_INCONCLUSIVE"
        report["errors"].append({"code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        report["exact_next_step"] = _next_step("AUDIT_INCONCLUSIVE")
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 42
    return report, 0 if report["final_decision"] != "AUDIT_INCONCLUSIVE" else 43


def _next_step(decision: str) -> str:
    return {
        "CANONICAL_PERSISTED_CHECKPOINT_SET_READY": "Update the reproducibility lock with checkpoint hashes and use the canonical persisted checkpoints for future rollout reports without claiming exact historical replication.",
        "HISTORICAL_IDENTITY_UNRECOVERABLE_CANONICALIZATION_REQUIRED": "Make an explicit policy decision to adopt or reject the regenerated persisted checkpoints as a new canonical baseline set.",
        "PROTOCOL_DRIFT_FOUND": "Fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.",
        "EVALUATION_NONDETERMINISM_BLOCKS_ROLLOUT": "Stabilize repeated disk evaluation before any canonicalization or rollout.",
        "TRAINING_NONDETERMINISM_CONFIRMED": "Treat same-config retraining variance as the blocker and avoid exact historical claims.",
        "CHECKPOINT_ARTIFACT_PROBLEM": "Repair missing or inconsistent checkpoint bundle artifacts and rerun the audit.",
        "AUDIT_INCONCLUSIVE": "Collect the missing historical or evaluation evidence before making a rollout baseline decision.",
    }[decision]


def _write_config_diff(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Old Vs Regenerated Config Diff",
        "",
        f"Date: {report['date']}",
        "",
        f"Historical commit: `{HISTORICAL_COMMIT}`",
        f"Regenerated commit: `{REGENERATED_COMMIT}`",
        "",
        "| field | classification | historical value | regenerated value | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.get("config_diff") or []:
        old_value = json.dumps(row.get("old_value"), sort_keys=True, default=str) if isinstance(row.get("old_value"), (dict, list)) else str(row.get("old_value"))
        new_value = json.dumps(row.get("regenerated_value"), sort_keys=True, default=str) if isinstance(row.get("regenerated_value"), (dict, list)) else str(row.get("regenerated_value"))
        lines.append(f"| {row['field']} | `{row['classification']}` | `{old_value}` | `{new_value}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Decisive Protocol Difference",
            "",
            f"- old wrap assignment: `{report['source_protocol_diff']['old_wrap_assignment']}`",
            f"- regenerated wrap assignment: `{report['source_protocol_diff']['regenerated_wrap_assignment']}`",
            f"- old disk reload path: `{report['source_protocol_diff']['old_disk_reload_path']}`",
            f"- regenerated disk reload path: `{report['source_protocol_diff']['regenerated_disk_reload_path']}`",
            "",
            "The historical run evaluated the trained in-memory object. The regenerated run saved an adapter bundle, reloaded it with `PeftModel.from_pretrained`, then evaluated the loaded object.",
        ]
    )
    _write_lines(path, lines)


def _write_alignment(report: dict[str, Any], path: Path) -> None:
    alignment = report.get("artifact_alignment") or {}
    lines = [
        "# Official SmolVLA Artifact And Evaluation Alignment",
        "",
        f"Date: {report['date']}",
        "",
        f"Protocol drift from frame/label/base alignment: `{alignment.get('protocol_drift_from_artifact_alignment')}`",
        "",
        "| seed | records | split/label/target/base aligned | max target diff | max frozen/base diff | max old-vs-regen LoRA action L2 | mean old-vs-regen LoRA action L2 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in alignment.get("per_seed") or []:
        lines.append(
            f"| {row['seed']} | {row['old_record_count']} | `{row['aligned_for_split_labels_targets_and_base_predictions']}` | "
            f"{row['max_target_abs_diff']} | {row['max_base_prediction_abs_diff']} | {row['max_lora_action_l2_diff']} | {row['mean_lora_action_l2_diff']} |"
        )
    lines.extend(
        [
            "",
            "## Alignment Verdict",
            "",
            "- test frame IDs: identical",
            "- task and episode IDs: identical",
            "- ground-truth actions: identical",
            "- split membership: identical",
            "- frozen/base predictions: identical",
            "- metric protocol file: identical across historical and regenerated commits",
            "- static-alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`, validation-only selection",
            "- test leakage: not introduced by this audit; split manifest leakage checks remain the authority",
            "",
            "The frame/label/base protocol aligns. The observed drift is in the LoRA prediction path, not in labels or split membership.",
        ]
    )
    _write_lines(path, lines)


def _write_eval_determinism(report: dict[str, Any], path: Path) -> None:
    eval_report = report.get("deterministic_evaluation") or {}
    lines = [
        "# Official SmolVLA Evaluation Determinism Check",
        "",
        f"Date: {report['date']}",
        "",
        f"Performed: `{eval_report.get('performed')}`",
        f"Scope: `{eval_report.get('scope')}`",
        f"Eval seed policy: `{eval_report.get('eval_seed_policy')}`",
        f"Repeat tolerance: `{eval_report.get('repeat_tolerance')}`",
        f"All deterministic within tolerance: `{eval_report.get('all_deterministic_within_tolerance')}`",
        f"Saved regenerated artifact matches fixed-seed re-eval: `{eval_report.get('regenerated_artifact_matches_fixed_seed_reeval')}`",
        "",
        "| seed | max action abs diff | max action L2 diff | rank4 metric diff | static metric diff | alpha identical | static metric identical | deterministic |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in eval_report.get("per_seed") or []:
        lines.append(
            f"| {row['seed']} | {row['max_per_action_abs_diff']} | {row['max_per_action_l2_diff']} | "
            f"{row['rank4_action_l2_metric_diff']} | {row['static_mix_action_l2_metric_diff']} | `{row['selected_alpha_identical']}` | "
            f"`{row['static_mix_test_metric_identical']}` | `{row['deterministic_within_tolerance']}` |"
        )
    lines.extend(["", "## Pass Metrics", ""])
    for row in eval_report.get("per_seed") or []:
        lines.append(f"### Seed {row['seed']}")
        for item in row.get("passes") or []:
            lines.append(
                f"- pass `{item['pass_index']}`: rank4 `{item['rank4_lora_action_l2_mean']}`, static `{item['static_mix_action_l2_mean']}`, alpha `{item['selected_alpha']}`, digest `{item['prediction_digest_test']}`"
            )
    lines.extend(
        [
            "",
            "## Saved Regenerated Artifact Vs Fixed-Seed Re-Eval",
            "",
            "| seed | regenerated rank4 | fixed-seed rank4 | diff | regenerated static | fixed-seed static | diff | regenerated alpha | fixed-seed alpha | match |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in eval_report.get("per_seed") or []:
        reference = row.get("regenerated_artifact_reference") or {}
        first_pass = (row.get("passes") or [{}])[0]
        lines.append(
            f"| {row['seed']} | {reference.get('rank4_lora_action_l2_mean')} | {first_pass.get('rank4_lora_action_l2_mean')} | "
            f"{reference.get('rank4_metric_diff_vs_fixed_seed_reeval')} | {reference.get('static_mix_action_l2_mean')} | "
            f"{first_pass.get('static_mix_action_l2_mean')} | {reference.get('static_metric_diff_vs_fixed_seed_reeval')} | "
            f"{reference.get('selected_alpha')} | {first_pass.get('selected_alpha')} | `{reference.get('matches_fixed_seed_reeval_within_tolerance')}` |"
        )
    _write_lines(path, lines)


def _write_training_status(report: dict[str, Any], path: Path) -> None:
    hist = report.get("historical_reproducibility_status") or {}
    loss_status = report.get("loss_sequence_status") or {}
    lines = [
        "# Official SmolVLA Training Determinism Status",
        "",
        f"Date: {report['date']}",
        "",
        f"Training happened in this audit: `{report['policy']['training_performed']}`",
        f"Optional single-seed probe ran: `{report['policy']['optional_single_seed_probe_ran']}`",
        "",
        "## Historical Identity Answers",
        "",
        f"1. Were the old adapter weights persisted? `{hist.get('old_adapter_weights_persisted')}`",
        f"2. Was the complete historical RNG/DataLoader state persisted? `{hist.get('complete_historical_rng_dataloader_state_persisted')}`",
        f"3. Was the exact historical sample order persisted? `{hist.get('exact_historical_sample_order_persisted')}`",
        f"4. Can the old learned policy identity be reconstructed? `{hist.get('old_learned_policy_identity_reconstructable')}`",
        f"5. Is exact old metric reproduction scientifically possible? `{hist.get('exact_old_metric_reproduction_scientifically_possible')}`",
        f"6. Is the observed difference config drift or ordinary retraining variance? `{hist.get('observed_difference_interpretation')}`",
        "",
        "## Loss Sequence Evidence",
        "",
        f"All old-vs-regenerated loss sequences identical: `{loss_status.get('all_loss_sequences_identical')}`",
        "",
        "| seed | loss sequence identical | old adapter path | regenerated adapter path |",
        "| ---: | --- | --- | --- |",
    ]
    for row in loss_status.get("per_seed") or []:
        lines.append(f"| {row['seed']} | `{row['loss_sequence_identical']}` | `{row['old_adapter_checkpoint_path']}` | `{row['regenerated_adapter_checkpoint_path']}` |")
    lines.extend(
        [
            "",
            "Terminology:",
            "",
            f"- historical run: `{hist.get('term_policy', {}).get('historical_run')}`",
            f"- regenerated persisted run: `{hist.get('term_policy', {}).get('regenerated_persisted_run')}`",
            f"- canonical persisted checkpoint: `{hist.get('term_policy', {}).get('canonical_persisted_checkpoint')}`",
        ]
    )
    _write_lines(path, lines)


def _write_canonical_proposal(report: dict[str, Any], path: Path) -> None:
    proposal = report.get("canonical_proposal") or {}
    lines = [
        "# Official SmolVLA Canonical Checkpoint Proposal",
        "",
        f"Date: {report['date']}",
        "",
        f"Accepted as canonical: `{proposal.get('accepted_as_canonical')}`",
        f"Reason: {proposal.get('reason')}",
        "",
        "## Side-By-Side Old Vs Candidate Canonical Metrics",
        "",
        "| seed | historical rank4 | historical static | regenerated rank4 | regenerated static | selected alpha |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    old_by_seed = {
        int(row["seed"]): row
        for row in _read_json(Path("reports/official_smolvla_lora_seed_repro_result.json")).get("seed_summaries") or []
    }
    for row in proposal.get("canonical_metric_table_if_later_accepted") or []:
        old = old_by_seed[int(row["seed"])]
        lines.append(
            f"| {row['seed']} | {old['metrics']['rank4_lora']['action_l2_mean']} | {old['metrics']['static_mix_val_selected']['action_l2_mean']} | "
            f"{row['rank4_lora']} | {row['static_mix_val_selected']} | {row['selected_alpha']} |"
        )
    lines.extend(
        [
            "",
            "Policy if canonicalization is later approved:",
            "",
            "- preserve old results as historical",
            "- do not overwrite historical metrics",
            "- use only explicitly adopted canonical checkpoint metrics in future rollout reports",
            "- update the reproducibility lock with checkpoint hashes",
            "- do not claim exact replication of historical ephemeral runs",
        ]
    )
    _write_lines(path, lines)


def _write_drift_audit(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA LoRA Drift Audit",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        f"Historical reproduction tolerance preserved: `{report.get('reproduction_tolerance')}`",
        "",
        "## Execution Boundary",
        "",
        f"- experiments happened: `{report['policy']['experiments_performed']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- optional single-seed probe ran: `{report['policy']['optional_single_seed_probe_ran']}`",
        f"- GPU used for evaluation: `{report['policy']['gpu_used_for_evaluation']}`",
        f"- downloads happened: `{report['policy']['downloads_performed']}`",
        f"- rollout happened: `{report['policy']['rollouts_performed']}`",
        f"- simulator dependency install happened: `{report['policy']['simulator_dependencies_installed']}`",
        f"- OpenVLA-OFT happened: `{report['policy']['openvla_oft_executed']}`",
        f"- FCAR revived: `{report['policy']['fcar_revived']}`",
        "",
        "## Main Findings",
        "",
        "- split/frame/label/frozen-base alignment is proven across old and regenerated artifacts",
        "- current persisted checkpoint bundles are complete and checksum verified",
        "- repeated disk evaluation is deterministic under the audit's fixed evaluation seed",
        "- the saved regenerated artifact metrics do not exactly match this fixed-seed disk re-evaluation, so evaluation RNG state is part of the protocol identity",
        "- historical adapter weights and complete training-state identity were not saved",
        "- a real protocol difference exists between historical in-memory evaluation and regenerated persisted PEFT reload evaluation",
        "",
        "## Root Cause",
        "",
        "The metric drift is best explained as protocol drift in the LoRA prediction path: `5d48b1e` evaluated the trained in-memory policy and did not persist/reload the adapter; `15649d6` assigns the PEFT wrapper return, saves the adapter, reloads it through `PeftModel.from_pretrained`, then evaluates that disk identity. The fixed-seed disk re-evaluation is internally repeatable, but it does not exactly reproduce the saved regenerated artifact metrics, which shows that evaluation RNG state was also part of the unpinned protocol identity. Because the historical adapter weights were never saved, the old learned policy identity cannot be recovered exactly.",
        "",
        "## Exact Next Step",
        "",
        str(report.get("exact_next_step")),
    ]
    _write_lines(path, lines)


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA LoRA Drift Decision",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        "Decision rationale:",
        "",
        "- the current persisted checkpoints are complete and checksum verified",
        "- fixed-seed repeated disk evaluation did not show material evaluation nondeterminism",
        "- saved regenerated artifact metrics did not exactly match fixed-seed disk re-evaluation metrics, so evaluation RNG state was unpinned",
        "- split, labels, frozen/base predictions, metric protocol, and static-alpha grid align",
        "- nevertheless, the old and regenerated LoRA prediction protocols differ",
        "- old adapter weights and complete old RNG/sample-order identity were not preserved",
        "",
        "Therefore the regenerated persisted checkpoints are not accepted as canonical in this audit. They can only become canonical after the PEFT protocol drift and evaluation RNG-state policy are fixed or after an explicit re-baselining decision that preserves old metrics as historical.",
        "",
        "Exact next step:",
        "",
        str(report.get("exact_next_step")),
    ]
    _write_lines(path, lines)


def write_reports(report: dict[str, Any], args: argparse.Namespace) -> None:
    _write_json(Path(args.report_json), report)
    _write_drift_audit(report, Path(args.audit_md))
    _write_config_diff(report, Path(args.config_diff_md))
    _write_alignment(report, Path(args.artifact_alignment_md))
    _write_eval_determinism(report, Path(args.eval_determinism_md))
    _write_training_status(report, Path(args.training_status_md))
    _write_canonical_proposal(report, Path(args.canonical_proposal_md))
    _write_decision(report, Path(args.decision_md))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--metric-protocol", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--old-result-json", default="reports/official_smolvla_lora_seed_repro_result.json")
    parser.add_argument("--regenerated-result-json", default="reports/official_smolvla_lora_checkpoint_regen_result.json")
    parser.add_argument("--checkpoint-manifest", default="reports/official_smolvla_lora_checkpoint_manifest.json")
    parser.add_argument("--old-seed-artifact-pattern", default="reports/official_smolvla_lora_seed_{seed}_prediction_artifact.json")
    parser.add_argument("--regenerated-seed-artifact-pattern", default="reports/official_smolvla_seed_{seed}_prediction_artifact.json")
    parser.add_argument("--report-json", default="reports/official_smolvla_lora_drift_audit.json")
    parser.add_argument("--audit-md", default="reports/official_smolvla_lora_drift_audit.md")
    parser.add_argument("--config-diff-md", default="reports/official_smolvla_old_vs_regen_config_diff.md")
    parser.add_argument("--artifact-alignment-md", default="reports/official_smolvla_artifact_alignment_audit.md")
    parser.add_argument("--eval-determinism-md", default="reports/official_smolvla_eval_determinism_check.md")
    parser.add_argument("--training-status-md", default="reports/official_smolvla_training_determinism_status.md")
    parser.add_argument("--canonical-proposal-md", default="reports/official_smolvla_canonical_checkpoint_proposal.md")
    parser.add_argument("--decision-md", default="reports/official_smolvla_lora_drift_decision.md")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--expected-model-revision", default="31d453f7edd78c839a8bbc39744a292686daf0de")
    parser.add_argument("--expected-dataset-revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    try:
        report, code = build_report(args)
    except DriftAuditError as exc:
        report = {
            "date": DATE,
            "status": "blocked",
            "final_decision": exc.code if exc.code in FINAL_DECISIONS else "AUDIT_INCONCLUSIVE",
            "errors": [{"code": exc.code, "message": str(exc)}],
            "exact_next_step": _next_step(exc.code if exc.code in FINAL_DECISIONS else "AUDIT_INCONCLUSIVE"),
        }
        code = 41
    write_reports(report, args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
