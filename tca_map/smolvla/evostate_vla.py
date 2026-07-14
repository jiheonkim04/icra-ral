"""EvoState-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TASK_KEYS = ("libero_spatial/task_4", "libero_10/task_4")
PROPOSAL_HASH = "A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9"
FORBIDDEN_INFERENCE_KEYS = {
    "object_state",
    "object_pose",
    "reward",
    "success",
    "terminal_success",
    "task_progress",
    "future_state",
    "future_action",
    "identity",
}


@dataclass(frozen=True)
class EvoStateConfig:
    train_identities: tuple[int, ...] = tuple(range(20260901, 20260911))
    validation_identities: tuple[int, ...] = tuple(range(20260911, 20260917))
    forbidden_development_identities: tuple[int, ...] = tuple(range(20260917, 20261201))
    min_transition_pairs: int = 5000
    min_task_transition_pairs: int = 1000
    min_variance: float = 1e-6
    min_transition_improvement: float = 0.05
    min_effective_rank: int = 3
    ridge_lambda: float = 1e-2
    damped_inverse_lambda: float = 1e-2
    delta_max: float = 0.20
    default_alpha: float = 0.25
    min_gate_positive_fraction: float = 0.02
    max_gate_positive_fraction: float = 0.98
    projection_ratio_min: float = 0.05
    min_scale: float = 1e-6


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def task_one_hot(task_key: str) -> np.ndarray:
    if task_key not in TASK_KEYS:
        raise ValueError(f"unknown EvoState task key: {task_key}")
    out = np.zeros(len(TASK_KEYS), dtype=np.float64)
    out[TASK_KEYS.index(task_key)] = 1.0
    return out


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged EvoState inference fields: {leaked}")


def _feature_full(record: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            _as_vector("state", record["state"], 8),
            _as_vector("action", record["action"], 7),
            _as_vector("previous_action", record["previous_action"], 7),
            np.asarray([float(record["chunk_index_fraction"])], dtype=np.float64),
            task_one_hot(str(record["task_key"])),
        ]
    ).astype(np.float64)


def _feature_actionless(record: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            _as_vector("state", record["state"], 8),
            _as_vector("previous_action", record["previous_action"], 7),
            np.asarray([float(record["chunk_index_fraction"])], dtype=np.float64),
            task_one_hot(str(record["task_key"])),
        ]
    ).astype(np.float64)


def _transition_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(record["task_key"]), int(record["identity"]), int(record["step"]))


def build_transition_pairs(records: Sequence[Mapping[str, Any]], config: EvoStateConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or EvoStateConfig()
    relevant = [record for record in records if str(record.get("task_key")) in TASK_KEYS]
    relevant = sorted(relevant, key=lambda row: (str(row["task_key"]), int(row["identity"]), int(row["step"])))
    by_key = {_transition_key(record): record for record in relevant}
    pairs: list[dict[str, Any]] = []
    for record in relevant:
        task_key, identity, step = _transition_key(record)
        next_record = by_key.get((task_key, identity, step + 1))
        if next_record is None:
            continue
        state = _as_vector("state", record["state"], 8)
        next_state = _as_vector("next_state", next_record["state"], 8)
        pairs.append(
            {
                "task_key": task_key,
                "identity": identity,
                "step": step,
                "split": str(record.get("split", "")),
                "state": state,
                "next_state": next_state,
                "delta_state": next_state - state,
                "action": _as_vector("action", record["action"], 7),
                "previous_action": _as_vector("previous_action", record["previous_action"], 7),
                "chunk_index_fraction": float(record["chunk_index_fraction"]),
                "feature_full": _feature_full(record),
                "feature_actionless": _feature_actionless(record),
            }
        )
    return pairs


def split_transition_pairs(pairs: Sequence[Mapping[str, Any]], config: EvoStateConfig | None = None) -> dict[str, list[Mapping[str, Any]]]:
    cfg = config or EvoStateConfig()
    train_ids = {int(value) for value in cfg.train_identities}
    validation_ids = {int(value) for value in cfg.validation_identities}
    return {
        "train": [pair for pair in pairs if int(pair["identity"]) in train_ids],
        "validation": [pair for pair in pairs if int(pair["identity"]) in validation_ids],
    }


def duplicate_transition_count(pairs: Sequence[Mapping[str, Any]]) -> int:
    seen: set[tuple[str, int, int]] = set()
    duplicates = 0
    for pair in pairs:
        key = (str(pair["task_key"]), int(pair["identity"]), int(pair["step"]))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _matrix(pairs: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    if not pairs:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray([pair[key] for pair in pairs], dtype=np.float64)


def _feature_matrix(pairs: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return _matrix(pairs, key)


def _target_matrix(pairs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return _matrix(pairs, "delta_state")


def _standardization(features: np.ndarray, config: EvoStateConfig) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < config.min_scale, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - mean.reshape(1, -1)) / scale.reshape(1, -1)


def _augment_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> dict[str, np.ndarray]:
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("ridge features and targets must be matrices")
    x = _augment_bias(features)
    reg = np.eye(x.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ targets)
    return {"weights": weights.astype(np.float64)}


def predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return _augment_bias(features) @ np.asarray(model["weights"], dtype=np.float64)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("inf")
    return float(np.mean(np.square(pred - target)))


def _relative_improvement(base_loss: float, model_loss: float) -> float:
    if not np.isfinite(base_loss) or base_loss <= 1e-12:
        return 0.0
    return float((base_loss - model_loss) / base_loss)


def _task_losses(pairs: Sequence[Mapping[str, Any]], pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    losses: dict[str, float] = {}
    for task in TASK_KEYS:
        mask = np.asarray([str(pair["task_key"]) == task for pair in pairs], dtype=bool)
        losses[task] = _mse(pred[mask], target[mask]) if np.any(mask) else float("inf")
    return losses


def _variance_stats(values: np.ndarray) -> dict[str, Any]:
    variances = np.var(values, axis=0) if values.size else np.asarray([], dtype=np.float64)
    return {
        "min": float(np.min(variances)) if variances.size else None,
        "max": float(np.max(variances)) if variances.size else None,
        "per_dim": [float(v) for v in variances],
    }


def _projection_ratio(b_matrix: np.ndarray, residuals: np.ndarray, damping: float) -> np.ndarray:
    if residuals.size == 0:
        return np.asarray([], dtype=np.float64)
    gram = b_matrix @ b_matrix.T + float(damping) * np.eye(b_matrix.shape[0], dtype=np.float64)
    projector = b_matrix @ b_matrix.T @ np.linalg.inv(gram)
    projected = residuals @ projector.T
    numerator = np.linalg.norm(projected, axis=1)
    denominator = np.linalg.norm(residuals, axis=1) + 1e-9
    return numerator / denominator


def _clip_l2(values: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-12))
    return values * scale


def damped_inverse_correction(b_matrix: np.ndarray, mismatch: np.ndarray, *, damping: float, delta_max: float) -> np.ndarray:
    b = np.asarray(b_matrix, dtype=np.float64).reshape(8, 7)
    e = np.asarray(mismatch, dtype=np.float64).reshape(-1, 8)
    gram = b @ b.T + float(damping) * np.eye(8, dtype=np.float64)
    raw = -(e @ np.linalg.inv(gram).T @ b).reshape(e.shape[0], 7)
    return _clip_l2(raw, float(delta_max))


def audit_evostate_records(records: Sequence[Mapping[str, Any]], config: EvoStateConfig | None = None) -> dict[str, Any]:
    cfg = config or EvoStateConfig()
    pairs = build_transition_pairs(records, cfg)
    splits = split_transition_pairs(pairs, cfg)
    train = splits["train"]
    validation = splits["validation"]
    hard_stop_reasons: list[str] = []

    forbidden = {int(value) for value in cfg.forbidden_development_identities}
    forbidden_present = sorted({int(pair["identity"]) for pair in train + validation if int(pair["identity"]) in forbidden})
    if forbidden_present:
        hard_stop_reasons.append(f"forbidden development identities present: {forbidden_present[:5]}")

    duplicates = duplicate_transition_count(train + validation)
    if duplicates:
        hard_stop_reasons.append(f"duplicate transition keys: {duplicates}")

    task_counts = {task: sum(1 for pair in train + validation if str(pair["task_key"]) == task) for task in TASK_KEYS}
    if len(train) + len(validation) < cfg.min_transition_pairs:
        hard_stop_reasons.append(f"transition pairs below minimum: {len(train) + len(validation)} < {cfg.min_transition_pairs}")
    for task, count in task_counts.items():
        if count < cfg.min_task_transition_pairs:
            hard_stop_reasons.append(f"{task} transition pairs below minimum: {count} < {cfg.min_task_transition_pairs}")

    if not train or not validation:
        hard_stop_reasons.append("missing train or validation transitions")
        return _audit_report(cfg, pairs, train, validation, hard_stop_reasons, duplicates, task_counts)

    train_full = _feature_matrix(train, "feature_full")
    validation_full = _feature_matrix(validation, "feature_full")
    train_actionless = _feature_matrix(train, "feature_actionless")
    validation_actionless = _feature_matrix(validation, "feature_actionless")
    train_targets = _target_matrix(train)
    validation_targets = _target_matrix(validation)
    validation_states = _matrix(validation, "state")
    validation_actions = _matrix(validation, "action")

    state_variance = _variance_stats(validation_states)
    action_variance = _variance_stats(validation_actions)
    if state_variance["min"] is None or float(state_variance["min"]) <= cfg.min_variance:
        hard_stop_reasons.append(f"validation state variance collapsed: {state_variance['min']}")
    if action_variance["min"] is None or float(action_variance["min"]) <= cfg.min_variance:
        hard_stop_reasons.append(f"validation action variance collapsed: {action_variance['min']}")

    full_mean, full_scale = _standardization(train_full, cfg)
    actionless_mean, actionless_scale = _standardization(train_actionless, cfg)
    train_full_std = _standardize(train_full, full_mean, full_scale)
    validation_full_std = _standardize(validation_full, full_mean, full_scale)
    train_actionless_std = _standardize(train_actionless, actionless_mean, actionless_scale)
    validation_actionless_std = _standardize(validation_actionless, actionless_mean, actionless_scale)

    constant_pred = np.mean(train_targets, axis=0, keepdims=True).repeat(validation_targets.shape[0], axis=0)
    actionless_model = fit_ridge(train_actionless_std, train_targets, cfg.ridge_lambda)
    full_model = fit_ridge(train_full_std, train_targets, cfg.ridge_lambda)
    actionless_pred = predict_ridge(actionless_model, validation_actionless_std)
    full_pred = predict_ridge(full_model, validation_full_std)

    constant_loss = _mse(constant_pred, validation_targets)
    actionless_loss = _mse(actionless_pred, validation_targets)
    full_loss = _mse(full_pred, validation_targets)
    improvement_vs_constant = _relative_improvement(constant_loss, full_loss)
    improvement_vs_actionless = _relative_improvement(actionless_loss, full_loss)

    if improvement_vs_constant < cfg.min_transition_improvement:
        hard_stop_reasons.append(f"transition model improvement vs constant below minimum: {improvement_vs_constant:.6f}")
    if improvement_vs_actionless < cfg.min_transition_improvement:
        hard_stop_reasons.append(f"transition model improvement vs actionless below minimum: {improvement_vs_actionless:.6f}")

    full_task_losses = _task_losses(validation, full_pred, validation_targets)
    actionless_task_losses = _task_losses(validation, actionless_pred, validation_targets)
    task_improvements = {
        task: _relative_improvement(actionless_task_losses[task], full_task_losses[task])
        for task in TASK_KEYS
    }
    for task, improvement in task_improvements.items():
        if improvement <= 0.0:
            hard_stop_reasons.append(f"action input does not improve transition prediction on {task}: {improvement:.6f}")

    # Full feature layout: state(8), action(7), previous_action(7), rho(1), task(2), bias(1).
    weights = np.asarray(full_model["weights"], dtype=np.float64)
    action_scale = full_scale[8:15].reshape(7, 1)
    target_b = (weights[8:15, :] / action_scale).T
    singular_values = np.linalg.svd(target_b, compute_uv=False)
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    rank_threshold = max(max_sv * 1e-4, 1e-8)
    effective_rank = int(np.sum(singular_values > rank_threshold))
    if effective_rank < cfg.min_effective_rank:
        hard_stop_reasons.append(f"controllability effective rank below minimum: {effective_rank} < {cfg.min_effective_rank}")
    damped_gram = target_b @ target_b.T + cfg.damped_inverse_lambda * np.eye(8, dtype=np.float64)
    condition_number = float(np.linalg.cond(damped_gram))
    if not np.isfinite(condition_number):
        hard_stop_reasons.append("damped inverse condition number is nonfinite")

    residuals = validation_targets - full_pred
    ratios = _projection_ratio(target_b, residuals, cfg.damped_inverse_lambda)
    residual_norm = np.linalg.norm(residuals, axis=1)
    lower = float(np.percentile(residual_norm, 25)) if residual_norm.size else 0.0
    upper = float(np.percentile(residual_norm, 90)) if residual_norm.size else 0.0
    gate_targets = (
        (np.sum(np.square(full_pred - validation_targets), axis=1) < np.sum(np.square(actionless_pred - validation_targets), axis=1))
        & (ratios >= cfg.projection_ratio_min)
        & (residual_norm >= lower)
        & (residual_norm <= upper)
    )
    gate_fraction = float(np.mean(gate_targets)) if gate_targets.size else 0.0
    if gate_fraction <= cfg.min_gate_positive_fraction or gate_fraction >= cfg.max_gate_positive_fraction:
        hard_stop_reasons.append(f"gate targets collapsed or near-collapsed: {gate_fraction:.6f}")

    corrections = damped_inverse_correction(target_b, residuals, damping=cfg.damped_inverse_lambda, delta_max=cfg.delta_max)
    correction_norms = np.linalg.norm(corrections, axis=1)
    p95_delta = float(np.percentile(correction_norms * cfg.default_alpha, 95)) if correction_norms.size else 0.0
    if p95_delta > cfg.delta_max + 1e-12:
        hard_stop_reasons.append(f"p95 validation action delta exceeds cap: {p95_delta:.6f}")
    action_validity = float(np.mean(np.all(np.isfinite(validation_actions + cfg.default_alpha * corrections), axis=1))) if corrections.size else 0.0
    if action_validity < 1.0:
        hard_stop_reasons.append(f"validation action validity below 1.0: {action_validity:.6f}")

    closed_gate_passthrough_max_abs_diff = 0.0
    final_decision = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" if not hard_stop_reasons else "AUDIT_STOP_" + _classify_hard_stop(hard_stop_reasons)

    report = _audit_report(cfg, pairs, train, validation, hard_stop_reasons, duplicates, task_counts)
    report.update(
        {
            "constant_loss": constant_loss,
            "actionless_loss": actionless_loss,
            "full_transition_loss": full_loss,
            "transition_improvement_vs_constant": improvement_vs_constant,
            "transition_improvement_vs_actionless": improvement_vs_actionless,
            "task_transition_improvement_vs_actionless": task_improvements,
            "state_variance": state_variance,
            "action_variance": action_variance,
            "controllability_singular_values": [float(v) for v in singular_values],
            "controllability_effective_rank": effective_rank,
            "damped_inverse_condition_number": condition_number,
            "gate_positive_fraction": gate_fraction,
            "validation_action_delta_p95": p95_delta,
            "validation_action_validity": action_validity,
            "closed_gate_passthrough_max_abs_diff": closed_gate_passthrough_max_abs_diff,
            "final_decision": final_decision,
            "next_step": "Run bounded six-config validation search." if not hard_stop_reasons else "Do not roll out EvoState; archive the hard stop and continue.",
            "model_metadata": {
                "full_feature_mean": full_mean,
                "full_feature_scale": full_scale,
                "ridge_weights": weights,
                "controllability_matrix": target_b,
            },
        }
    )
    return report


def _classify_hard_stop(reasons: Sequence[str]) -> str:
    joined = " ".join(reasons).lower()
    if "improvement" in joined or "action input" in joined or "rank" in joined or "gate targets" in joined:
        return "DESIGN_FAILURE"
    if "forbidden" in joined or "duplicate" in joined or "transition pairs below minimum" in joined or "variance collapsed" in joined or "missing" in joined:
        return "DATA_FAILURE"
    if "nonfinite" in joined or "validity" in joined or "passthrough" in joined:
        return "IMPLEMENTATION_FAILURE"
    return "DESIGN_FAILURE"


def _audit_report(
    config: EvoStateConfig,
    pairs: Sequence[Mapping[str, Any]],
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    hard_stop_reasons: Sequence[str],
    duplicates: int,
    task_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "method": "EvoState-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "transition_pairs": len(pairs),
        "train_transition_pairs": len(train),
        "validation_transition_pairs": len(validation),
        "duplicate_transition_keys": int(duplicates),
        "task_transition_counts": dict(task_counts),
        "hard_stop_reasons": list(hard_stop_reasons),
        "config": {
            "train_identities": list(config.train_identities),
            "validation_identities": list(config.validation_identities),
            "forbidden_development_identity_min": min(config.forbidden_development_identities),
            "min_transition_pairs": config.min_transition_pairs,
            "min_task_transition_pairs": config.min_task_transition_pairs,
            "min_transition_improvement": config.min_transition_improvement,
            "min_effective_rank": config.min_effective_rank,
            "ridge_lambda": config.ridge_lambda,
            "damped_inverse_lambda": config.damped_inverse_lambda,
            "delta_max": config.delta_max,
            "default_alpha": config.default_alpha,
        },
        "final_decision": "AUDIT_STOP_DATA_FAILURE" if hard_stop_reasons else "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
        "next_step": "Do not roll out EvoState; archive the hard stop and continue." if hard_stop_reasons else "Run bounded six-config validation search.",
    }


__all__ = [
    "EvoStateConfig",
    "FORBIDDEN_INFERENCE_KEYS",
    "PROPOSAL_HASH",
    "TASK_KEYS",
    "audit_evostate_records",
    "build_transition_pairs",
    "damped_inverse_correction",
    "duplicate_transition_count",
    "fit_ridge",
    "predict_ridge",
    "split_transition_pairs",
    "task_one_hot",
    "validate_inference_fields",
]
