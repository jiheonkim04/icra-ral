"""Tiny head-only training smoke over cached feature records.

This module intentionally stays dependency-light and does not import SmolVLA,
OpenVLA-OFT, torch, transformers, simulators, or dataset loaders. It trains
small NumPy linear heads on cached hidden-token records only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from tca_map.eval import compute_offline_metrics
from tca_map.features.cache import validate_feature_cache, write_dummy_feature_cache
from tca_map.heads import ActionMapHead


MAX_TINY_SMOKE_STEPS = 100
DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_RUNTIME_SECONDS = 15 * 60
DANGEROUS_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
]


class TinyHeadOnlySmokeError(RuntimeError):
    """Raised when the tiny smoke would cross a safety or contract boundary."""


def _load_records(cache_dir: Path) -> list[dict]:
    features_path = cache_dir / "features.jsonl"
    return [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _policy(training_performed: bool) -> dict:
    return {
        "bounded_tiny_head_only_smoke": True,
        "standing_approval_for_tiny_training_smoke": True,
        "cached_features_used": True,
        "backbone_frozen": True,
        "offline_proxy_only": True,
        "real_dataset_used": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "head_forward_passes_performed": training_performed,
        "training_performed": training_performed,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "paper_grade_claims_made": False,
    }


def ensure_safe_environment(require_training_gate: bool = True) -> None:
    set_dangerous = [name for name in DANGEROUS_GATES if os.environ.get(name)]
    if set_dangerous:
        raise TinyHeadOnlySmokeError("dangerous gates are set: " + ", ".join(set_dangerous))
    if require_training_gate and os.environ.get("ALLOW_TINY_TRAINING") != "1":
        raise TinyHeadOnlySmokeError("ALLOW_TINY_TRAINING=1 is required for bounded tiny head-only smoke")


def validate_smoke_bounds(max_steps: int, max_runtime_seconds: int) -> None:
    if max_steps < 1:
        raise TinyHeadOnlySmokeError("max_steps must be >= 1")
    if max_steps > MAX_TINY_SMOKE_STEPS:
        raise TinyHeadOnlySmokeError(f"max_steps must be <= {MAX_TINY_SMOKE_STEPS}")
    if max_runtime_seconds < 1:
        raise TinyHeadOnlySmokeError("max_runtime_seconds must be >= 1")
    if max_runtime_seconds > DEFAULT_MAX_RUNTIME_SECONDS:
        raise TinyHeadOnlySmokeError(f"max_runtime_seconds must be <= {DEFAULT_MAX_RUNTIME_SECONDS}")


def load_smoke_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _feature_matrix(records: list[dict]) -> np.ndarray:
    return np.asarray([record["hidden_tokens"] for record in records], dtype=np.float64)


def _target_ids(records: list[dict]) -> np.ndarray:
    return np.asarray([int(record.get("target", {}).get("object_id", 0)) for record in records], dtype=np.int64)


def _expert_actions(records: list[dict]) -> np.ndarray:
    return np.asarray([record["expert_action"] for record in records], dtype=np.float64)


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def _one_hot(indices: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros((indices.shape[0], width), dtype=np.float64)
    result[np.arange(indices.shape[0]), np.clip(indices, 0, width - 1)] = 1.0
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _train_action_regressor(features: np.ndarray, targets: np.ndarray, max_steps: int, lr: float) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    weights = np.zeros((x.shape[1], targets.shape[1]), dtype=np.float64)
    losses: list[float] = []
    for _ in range(max_steps):
        pred = x @ weights
        diff = pred - targets
        loss = float(np.mean(diff**2))
        losses.append(loss)
        grad = (2.0 / targets.size) * (x.T @ diff)
        weights -= lr * grad
    return weights, losses


def _train_target_classifier(features: np.ndarray, target_ids: np.ndarray, num_targets: int, max_steps: int, lr: float) -> tuple[np.ndarray, list[float]]:
    x = _with_bias(features)
    labels = _one_hot(target_ids, num_targets)
    weights = np.zeros((x.shape[1], num_targets), dtype=np.float64)
    losses: list[float] = []
    for _ in range(max_steps):
        logits = x @ weights
        probs = _softmax(logits)
        loss = float(-np.sum(labels * np.log(probs + 1e-12)) / labels.shape[0])
        losses.append(loss)
        grad = x.T @ (probs - labels) / labels.shape[0]
        weights -= lr * grad
    return weights, losses


def _predict_action(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(_with_bias(features) @ weights, -1.0, 1.0)


def _target_margin(logits: np.ndarray, target_ids: np.ndarray) -> float:
    margins = []
    for row, target_id in zip(logits, target_ids):
        correct = row[target_id]
        others = np.delete(row, target_id)
        margins.append(float(correct - np.max(others)))
    return float(np.mean(margins)) if margins else 0.0


def _head_config_summary(config_path: Path, effective_max_steps: int) -> dict:
    config = load_smoke_config(config_path)
    training = config.get("training", {})
    head = config.get("head", {})
    return {
        "config": str(config_path),
        "run_name": config.get("run", {}).get("name"),
        "head": head.get("name"),
        "target_conditioned": head.get("target_conditioned"),
        "config_max_steps": training.get("max_steps"),
        "effective_max_steps": effective_max_steps,
        "config_overridden_for_smoke": int(training.get("max_steps", 0)) != effective_max_steps,
        "train_backbone": training.get("train_backbone"),
        "train_heads": training.get("train_heads"),
        "backbone_freeze": config.get("backbone", {}).get("freeze"),
        "use_cached_features": config.get("backbone", {}).get("use_cached_features"),
        "openvla_oft_enabled": config.get("openvla_oft", {}).get("enabled"),
        "rollouts_allowed": config.get("run", {}).get("rollouts_allowed"),
        "gpu_training_allowed": config.get("run", {}).get("gpu_training_allowed"),
    }


def _candidate_count(records: list[dict]) -> int:
    max_from_objects = max((len(record.get("candidate_objects") or []) for record in records), default=0)
    max_from_targets = max((int(record.get("target", {}).get("object_id", 0)) + 1 for record in records), default=1)
    return max(1, max_from_objects, max_from_targets)


def train_smoke_head(
    records: list[dict],
    head_name: str,
    max_steps: int,
    lr: float,
    grid_size: int,
) -> dict:
    start = time.perf_counter()
    features = _feature_matrix(records)
    expert_actions = _expert_actions(records)
    target_ids = _target_ids(records)
    num_targets = _candidate_count(records)

    if head_name == "tca_map":
        target_weights, target_losses = _train_target_classifier(features, target_ids, num_targets, max_steps, lr)
        logits = _with_bias(features) @ target_weights
        pred_target_ids = np.argmax(logits, axis=1)
        conditioned_train = np.concatenate([features, _one_hot(target_ids, num_targets)], axis=1)
        conditioned_eval = np.concatenate([features, _one_hot(pred_target_ids, num_targets)], axis=1)
    elif head_name == "actionmap":
        target_weights = None
        target_losses = []
        logits = np.zeros((features.shape[0], num_targets), dtype=np.float64)
        pred_target_ids = np.full(features.shape[0], -1, dtype=np.int64)
        conditioned_train = features
        conditioned_eval = features
    else:
        raise TinyHeadOnlySmokeError(f"unsupported head_name: {head_name}")

    action_weights, action_losses = _train_action_regressor(conditioned_train, expert_actions, max_steps, lr)
    pred_actions = _predict_action(conditioned_eval, action_weights)
    action_head = ActionMapHead(grid_size=grid_size)

    metric_records = []
    for record, pred_action, pred_target in zip(records, pred_actions, pred_target_ids):
        expert_action = record["expert_action"]
        metric_records.append(
            {
                "sample_id": record["sample_id"],
                "pred_action": [float(value) for value in pred_action.tolist()],
                "expert_action": expert_action,
                "pred_voxel": action_head.action_to_voxel(pred_action.tolist()),
                "expert_voxel": action_head.action_to_voxel(expert_action),
                "pred_target": int(pred_target),
                "target_id": int(record.get("target", {}).get("object_id", 0)),
                "latency_ms": 0.0,
            }
        )

    metrics = compute_offline_metrics(metric_records)
    metrics.update(
        {
            "mode": "tiny_head_only_smoke",
            "head": head_name,
            "cache_record_count": len(records),
            "training_loss_start": round(float(action_losses[0]), 6),
            "training_loss_end": round(float(action_losses[-1]), 6),
            "training_loss_delta": round(float(action_losses[0] - action_losses[-1]), 6),
            "target_loss_start": round(float(target_losses[0]), 6) if target_losses else None,
            "target_loss_end": round(float(target_losses[-1]), 6) if target_losses else None,
            "counterfactual_separation_margin": round(_target_margin(logits, target_ids), 6)
            if head_name == "tca_map"
            else 0.0,
            "latency_ms": round((time.perf_counter() - start) * 1000.0 / max(1, len(records)), 6),
            "max_gpu_memory_mb": 0.0,
        }
    )

    finite_losses = all(math.isfinite(loss) for loss in action_losses + target_losses)
    return {
        "head": head_name,
        "max_steps": max_steps,
        "learning_rate": lr,
        "trainable_parameter_count": int(action_weights.size + (target_weights.size if target_weights is not None else 0)),
        "finite_losses": finite_losses,
        "metrics": metrics,
    }


def run_tiny_head_only_smoke(
    cache_dir: Path,
    report_path: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    prepare_dummy_cache: bool = False,
    require_training_gate: bool = True,
) -> dict:
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds)
    started = time.perf_counter()

    if prepare_dummy_cache and not (cache_dir / "manifest.json").exists():
        write_dummy_feature_cache(cache_dir, max_samples=4, overwrite=True)

    validation = validate_feature_cache(cache_dir)
    if not validation["valid"]:
        report = {
            "policy": _policy(training_performed=False),
            "cache_dir": str(cache_dir),
            "cache_valid": False,
            "validation_errors": validation["errors"],
            "tiny_head_only_smoke_passed": False,
            "recommended_next_step": "Create or fix the dummy feature cache before tiny head-only smoke.",
        }
        _write_report(report_path, report)
        return report

    records = _load_records(cache_dir)
    configs = [
        _head_config_summary(Path("configs/actionmap_head_only_lowcompute.yaml"), max_steps),
        _head_config_summary(Path("configs/tca_map_head_only_lowcompute.yaml"), max_steps),
    ]
    head_reports = []
    for config in configs:
        elapsed = time.perf_counter() - started
        if elapsed > max_runtime_seconds:
            raise TinyHeadOnlySmokeError("tiny head-only smoke exceeded max_runtime_seconds")
        head_reports.append(
            train_smoke_head(
                records=records,
                head_name=config["head"],
                max_steps=max_steps,
                lr=0.05,
                grid_size=8,
            )
        )

    total_elapsed = time.perf_counter() - started
    passed = bool(
        validation["valid"]
        and total_elapsed <= max_runtime_seconds
        and max_steps <= MAX_TINY_SMOKE_STEPS
        and all(head["finite_losses"] for head in head_reports)
    )
    report = {
        "policy": _policy(training_performed=True),
        "cache_dir": str(cache_dir),
        "cache_valid": True,
        "validation_errors": [],
        "cache_record_count": len(records),
        "max_steps": max_steps,
        "max_steps_cap": MAX_TINY_SMOKE_STEPS,
        "max_runtime_seconds": max_runtime_seconds,
        "elapsed_seconds": round(total_elapsed, 6),
        "runtime_within_cap": total_elapsed <= max_runtime_seconds,
        "configs": configs,
        "heads": head_reports,
        "tiny_head_only_smoke_passed": passed,
        "safe_to_run_real_pilot": False,
        "safe_to_run_rollouts": False,
        "recommended_next_step": (
            "Tiny head-only smoke passed. Treat this as interface validation only; stop before real dataset training, rollouts, simulator execution, OpenVLA-OFT, or paper claims."
            if passed
            else "Fix the tiny head-only smoke before any further pilot work."
        ),
    }
    _write_report(report_path, report)
    return report


def _write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="runs/feature_cache/dummy_contract")
    parser.add_argument("--report-path", default="reports/tiny_head_only_smoke_report.json")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--prepare-dummy-cache", action="store_true")
    args = parser.parse_args()

    try:
        report = run_tiny_head_only_smoke(
            cache_dir=Path(args.cache_dir),
            report_path=Path(args.report_path),
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            prepare_dummy_cache=args.prepare_dummy_cache,
            require_training_gate=True,
        )
    except TinyHeadOnlySmokeError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
