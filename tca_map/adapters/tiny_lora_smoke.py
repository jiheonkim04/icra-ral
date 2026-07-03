"""Bounded tiny LoRA smoke over cached feature records.

This module intentionally stays dependency-light. It does not import SmolVLA,
OpenVLA-OFT, torch, transformers, PEFT, simulators, or dataset loaders. The
smoke trains small NumPy low-rank adapter matrices over cached/dummy features
only, with the backbone represented as frozen feature records.
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

from tca_map.adapters.lora_policy import validate_lora_policy_config
from tca_map.eval import compute_offline_metrics
from tca_map.features.cache import validate_feature_cache, write_dummy_feature_cache
from tca_map.heads import ActionMapHead
from tca_map.inference.tca_select import distributional_tca_select_inference


MAX_TINY_LORA_STEPS = 100
MAX_TINY_LORA_SAMPLES = 200
DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_RUNTIME_SECONDS = 15 * 60
DEFAULT_LORA_RANK = 4
DANGEROUS_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF",
]


class TinyLoraSmokeError(RuntimeError):
    """Raised when the tiny LoRA smoke would cross a safety boundary."""


def ensure_safe_environment(require_training_gate: bool = True) -> None:
    set_dangerous = [name for name in DANGEROUS_GATES if os.environ.get(name)]
    if set_dangerous:
        raise TinyLoraSmokeError("dangerous gates are set: " + ", ".join(set_dangerous))
    if require_training_gate and os.environ.get("ALLOW_TINY_TRAINING") != "1":
        raise TinyLoraSmokeError("ALLOW_TINY_TRAINING=1 is required for bounded tiny LoRA smoke")


def validate_smoke_bounds(max_steps: int, max_runtime_seconds: int, max_samples: int, rank: int) -> None:
    if max_steps < 1:
        raise TinyLoraSmokeError("max_steps must be >= 1")
    if max_steps > MAX_TINY_LORA_STEPS:
        raise TinyLoraSmokeError(f"max_steps must be <= {MAX_TINY_LORA_STEPS}")
    if max_runtime_seconds < 1:
        raise TinyLoraSmokeError("max_runtime_seconds must be >= 1")
    if max_runtime_seconds > DEFAULT_MAX_RUNTIME_SECONDS:
        raise TinyLoraSmokeError(f"max_runtime_seconds must be <= {DEFAULT_MAX_RUNTIME_SECONDS}")
    if max_samples < 1:
        raise TinyLoraSmokeError("max_samples must be >= 1")
    if max_samples > MAX_TINY_LORA_SAMPLES:
        raise TinyLoraSmokeError(f"max_samples must be <= {MAX_TINY_LORA_SAMPLES}")
    if rank < 1:
        raise TinyLoraSmokeError("rank must be >= 1")
    if rank > 16:
        raise TinyLoraSmokeError("rank must be <= 16 for tiny LoRA smoke")


def _policy(training_performed: bool) -> dict:
    return {
        "bounded_tiny_lora_smoke": True,
        "risk_assessed_autonomy_for_tiny_training_smoke": True,
        "cached_features_used": True,
        "backbone_frozen": True,
        "trainable_lora_adapter_weights_only": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "real_dataset_used": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "adapter_construction_performed": training_performed,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": training_performed,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    }


def _load_records(cache_dir: Path, max_samples: int) -> list[dict]:
    features_path = cache_dir / "features.jsonl"
    records = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:max_samples]


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


def _candidate_count(records: list[dict]) -> int:
    max_from_objects = max((len(record.get("candidate_objects") or []) for record in records), default=0)
    max_from_targets = max((int(record.get("target", {}).get("object_id", 0)) + 1 for record in records), default=1)
    return max(1, max_from_objects, max_from_targets)


def _target_margin(logits: np.ndarray, target_ids: np.ndarray) -> float:
    margins = []
    for row, target_id in zip(logits, target_ids):
        correct = row[target_id]
        others = np.delete(row, target_id)
        margins.append(float(correct - np.max(others))) if len(others) else margins.append(float(correct))
    return float(np.mean(margins)) if margins else 0.0


def _frozen_base(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.01, size=(input_dim, output_dim))


def _init_lora(input_dim: int, output_dim: int, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.01, size=(input_dim, rank))
    b = np.zeros((rank, output_dim), dtype=np.float64)
    return a, b


def _train_lora_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    max_steps: int,
    lr: float,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    x = _with_bias(features)
    base = _frozen_base(x.shape[1], targets.shape[1], seed)
    a, b = _init_lora(x.shape[1], targets.shape[1], rank, seed + 1)
    losses: list[float] = []
    for _ in range(max_steps):
        delta = a @ b
        pred = x @ (base + delta)
        diff = pred - targets
        loss = float(np.mean(diff**2))
        losses.append(loss)
        grad_delta = (2.0 / targets.size) * (x.T @ diff)
        grad_a = grad_delta @ b.T
        grad_b = a.T @ grad_delta
        a -= lr * grad_a
        b -= lr * grad_b
    return base, a, b, losses


def _train_lora_classifier(
    features: np.ndarray,
    target_ids: np.ndarray,
    num_targets: int,
    max_steps: int,
    lr: float,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    x = _with_bias(features)
    labels = _one_hot(target_ids, num_targets)
    base = _frozen_base(x.shape[1], num_targets, seed)
    a, b = _init_lora(x.shape[1], num_targets, rank, seed + 1)
    losses: list[float] = []
    for _ in range(max_steps):
        logits = x @ (base + a @ b)
        probs = _softmax(logits)
        loss = float(-np.sum(labels * np.log(probs + 1e-12)) / labels.shape[0])
        losses.append(loss)
        grad_delta = x.T @ (probs - labels) / labels.shape[0]
        grad_a = grad_delta @ b.T
        grad_b = a.T @ grad_delta
        a -= lr * grad_a
        b -= lr * grad_b
    return base, a, b, losses


def _predict(features: np.ndarray, base: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _with_bias(features) @ (base + a @ b)


def _lora_param_count(a: np.ndarray, b: np.ndarray) -> int:
    return int(a.size + b.size)


def _metric_records(
    records: list[dict],
    pred_actions: np.ndarray,
    pred_target_ids: np.ndarray,
    grid_size: int,
) -> list[dict]:
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
    return metric_records


def _apply_distributional_select(
    pred_actions: np.ndarray,
    logits: np.ndarray,
    pred_target_ids: np.ndarray,
    temperature: float = 0.5,
) -> np.ndarray:
    selected_actions = []
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0, 0.0],
            [-0.03, 0.0, 0.0, 0.0],
            [0.0, 0.03, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    for index, action in enumerate(pred_actions):
        target_index = int(pred_target_ids[index])
        candidates = []
        for candidate_index, offset in enumerate(offsets):
            candidate_action = np.clip(action + offset, -1.0, 1.0)
            candidates.append(
                {
                    "index": candidate_index,
                    "action": [float(value) for value in candidate_action.tolist()],
                    "voxel": candidate_index,
                    "logit": 1.0 - 0.1 * candidate_index,
                    "target_index": target_index,
                }
            )
        action_heatmap = {"candidates": candidates}
        masked_action_heatmap = {
            "candidates": [
                {**candidate, "logit": float(candidate["logit"]) - 0.05} for candidate in candidates
            ]
        }
        target_heatmap = {
            "scores": [float(value) for value in logits[index].tolist()],
            "top_index": target_index,
        }
        result = distributional_tca_select_inference(
            action_heatmap=action_heatmap,
            target_heatmap=target_heatmap,
            masked_action_heatmap=masked_action_heatmap,
            K=4,
            temperature=temperature,
            metadata=None,
            external_verifier=None,
        )
        selected = result["selected"] or candidates[0]
        selected_actions.append(selected["action"])
    return np.asarray(selected_actions, dtype=np.float64)


def _arm_report(
    records: list[dict],
    arm_name: str,
    max_steps: int,
    lr: float,
    rank: int,
    grid_size: int,
) -> dict:
    start = time.perf_counter()
    features = _feature_matrix(records)
    expert_actions = _expert_actions(records)
    target_ids = _target_ids(records)
    num_targets = _candidate_count(records)

    target_param_count = 0
    target_losses: list[float] = []
    logits = np.zeros((features.shape[0], num_targets), dtype=np.float64)
    pred_target_ids = np.full(features.shape[0], -1, dtype=np.int64)

    if arm_name.startswith("tca_map"):
        target_base, target_a, target_b, target_losses = _train_lora_classifier(
            features=features,
            target_ids=target_ids,
            num_targets=num_targets,
            max_steps=max_steps,
            lr=lr,
            rank=rank,
            seed=37,
        )
        logits = _predict(features, target_base, target_a, target_b)
        pred_target_ids = np.argmax(logits, axis=1)
        conditioned_train = np.concatenate([features, _one_hot(target_ids, num_targets)], axis=1)
        conditioned_eval = np.concatenate([features, _one_hot(pred_target_ids, num_targets)], axis=1)
        target_param_count = _lora_param_count(target_a, target_b)
    else:
        conditioned_train = features
        conditioned_eval = features

    action_base, action_a, action_b, action_losses = _train_lora_regressor(
        features=conditioned_train,
        targets=expert_actions,
        max_steps=max_steps,
        lr=lr,
        rank=rank,
        seed=53,
    )
    pred_actions = np.clip(_predict(conditioned_eval, action_base, action_a, action_b), -1.0, 1.0)

    if arm_name == "tca_map_lora_distributional_select":
        pred_actions = _apply_distributional_select(pred_actions, logits, pred_target_ids)

    metrics = compute_offline_metrics(_metric_records(records, pred_actions, pred_target_ids, grid_size))
    metrics.update(
        {
            "mode": "tiny_lora_smoke",
            "arm": arm_name,
            "cache_record_count": len(records),
            "training_loss_start": round(float(action_losses[0]), 6),
            "training_loss_end": round(float(action_losses[-1]), 6),
            "training_loss_delta": round(float(action_losses[0] - action_losses[-1]), 6),
            "target_loss_start": round(float(target_losses[0]), 6) if target_losses else None,
            "target_loss_end": round(float(target_losses[-1]), 6) if target_losses else None,
            "counterfactual_separation_margin": round(_target_margin(logits, target_ids), 6)
            if arm_name.startswith("tca_map")
            else 0.0,
            "latency_ms": round((time.perf_counter() - start) * 1000.0 / max(1, len(records)), 6),
            "max_gpu_memory_mb": 0.0,
        }
    )

    finite_losses = all(math.isfinite(loss) for loss in action_losses + target_losses)
    return {
        "arm": arm_name,
        "max_steps": max_steps,
        "learning_rate": lr,
        "lora_rank": rank,
        "trainable_lora_parameter_count": int(_lora_param_count(action_a, action_b) + target_param_count),
        "frozen_base_parameter_count": int(action_base.size + (target_base.size if arm_name.startswith("tca_map") else 0)),
        "finite_losses": finite_losses,
        "metrics": metrics,
    }


def _load_lora_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def run_tiny_lora_smoke(
    cache_dir: Path,
    report_path: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    max_samples: int = 4,
    rank: int = DEFAULT_LORA_RANK,
    prepare_dummy_cache: bool = False,
    require_training_gate: bool = True,
) -> dict:
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        max_samples=max_samples,
        rank=rank,
    )
    started = time.perf_counter()

    if prepare_dummy_cache and not (cache_dir / "manifest.json").exists():
        write_dummy_feature_cache(cache_dir, max_samples=max_samples, overwrite=True)

    validation = validate_feature_cache(cache_dir)
    if not validation["valid"]:
        report = {
            "policy": _policy(training_performed=False),
            "cache_dir": str(cache_dir),
            "cache_valid": False,
            "validation_errors": validation["errors"],
            "tiny_lora_smoke_passed": False,
            "recommended_next_step": "Create or fix the dummy feature cache before tiny LoRA smoke.",
        }
        _write_report(report_path, report)
        return report

    records = _load_records(cache_dir, max_samples=max_samples)
    if len(records) > MAX_TINY_LORA_SAMPLES:
        raise TinyLoraSmokeError(f"record count exceeds {MAX_TINY_LORA_SAMPLES}")

    config_path = Path("configs/lora_adapter_lowcompute.yaml")
    lora_config = _load_lora_config(config_path)
    config_validation = validate_lora_policy_config(lora_config)
    if not config_validation["passed"]:
        raise TinyLoraSmokeError("invalid LoRA config: " + "; ".join(config_validation["errors"]))

    arms = []
    for arm_name in ["actionmap_lora", "tca_map_lora", "tca_map_lora_distributional_select"]:
        elapsed = time.perf_counter() - started
        if elapsed > max_runtime_seconds:
            raise TinyLoraSmokeError("tiny LoRA smoke exceeded max_runtime_seconds")
        arms.append(
            _arm_report(
                records=records,
                arm_name=arm_name,
                max_steps=max_steps,
                lr=0.05,
                rank=rank,
                grid_size=8,
            )
        )

    total_elapsed = time.perf_counter() - started
    passed = bool(
        validation["valid"]
        and config_validation["passed"]
        and total_elapsed <= max_runtime_seconds
        and max_steps <= MAX_TINY_LORA_STEPS
        and len(records) <= MAX_TINY_LORA_SAMPLES
        and all(arm["finite_losses"] for arm in arms)
    )
    report = {
        "policy": _policy(training_performed=True),
        "cache_dir": str(cache_dir),
        "cache_valid": True,
        "validation_errors": [],
        "cache_record_count": len(records),
        "max_samples": max_samples,
        "max_samples_cap": MAX_TINY_LORA_SAMPLES,
        "max_steps": max_steps,
        "max_steps_cap": MAX_TINY_LORA_STEPS,
        "max_runtime_seconds": max_runtime_seconds,
        "elapsed_seconds": round(total_elapsed, 6),
        "runtime_within_cap": total_elapsed <= max_runtime_seconds,
        "lora_config": {
            "path": str(config_path),
            "rank_requested": rank,
            "config_rank": lora_config.get("lora", {}).get("rank"),
            "config_max_steps": lora_config.get("training", {}).get("max_steps"),
            "validation": config_validation,
        },
        "arms": arms,
        "tiny_lora_smoke_passed": passed,
        "safe_to_run_real_pilot": False,
        "safe_to_run_rollouts": False,
        "recommended_next_step": (
            "Tiny LoRA smoke passed. Treat this as offline proxy interface validation only; next safe step is a tiny LoRA comparison report, still no rollout, simulator, OpenVLA-OFT, or paper claim."
            if passed
            else "Fix the tiny LoRA smoke before any further LoRA pilot work."
        ),
    }
    _write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="runs/feature_cache/dummy_contract")
    parser.add_argument("--report-path", default="reports/tiny_lora_smoke_report.json")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--prepare-dummy-cache", action="store_true")
    args = parser.parse_args()

    try:
        report = run_tiny_lora_smoke(
            cache_dir=Path(args.cache_dir),
            report_path=Path(args.report_path),
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            prepare_dummy_cache=args.prepare_dummy_cache,
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
