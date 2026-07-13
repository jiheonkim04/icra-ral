"""RCV-VLA lightweight verifier helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_KEYS = ("libero_spatial/task_4", "libero_10/task_4")
VARIANTS = (
    "queued_frozen_smolvla",
    "sv_deviation_proxy",
    "rcv_full",
    "rcv_no_context_ablation",
    "stateless_first_action",
)
PRIVILEGED_INFERENCE_FIELDS = {
    "success",
    "reward",
    "task_outcome",
    "future_observation",
    "future_action",
    "sim_state",
    "mujoco_state",
    "object_pose",
    "bddl_predicate",
    "current_image",
    "observation.images.camera1",
    "observation.images.camera2",
}
FULL_FEATURE_NAMES = (
    tuple(f"state_{index}" for index in range(8))
    + tuple(f"queued_action_{index}" for index in range(7))
    + tuple(f"previous_action_{index}" for index in range(7))
    + ("chunk_index_fraction",)
    + tuple(f"task_one_hot_{index}" for index in range(len(TASK_KEYS)))
)
NO_CONTEXT_FEATURE_NAMES = (
    tuple(f"queued_action_{index}" for index in range(7))
    + ("chunk_index_fraction",)
    + tuple(f"task_one_hot_{index}" for index in range(len(TASK_KEYS)))
)


@dataclass(frozen=True)
class RCVConfig:
    disagreement_quantile: float = 0.75
    learning_rate: float = 0.05
    l2: float = 1e-4
    max_epochs: int = 500
    seed: int = 260713
    eps: float = 1e-8

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged RCV inference fields: {forbidden}")


def _as_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(length):
        raise ValueError(f"{name} must have length {length}, got {array.size}")
    return array


def task_one_hot(task_key: str) -> np.ndarray:
    if str(task_key) not in TASK_KEYS:
        raise ValueError(f"unknown RCV task key: {task_key}")
    out = np.zeros(len(TASK_KEYS), dtype=np.float64)
    out[TASK_KEYS.index(str(task_key))] = 1.0
    return out


def action_disagreement(queued_action: Any, fresh_action: Any) -> float:
    queued = _as_vector(queued_action, name="queued_action", length=7)
    fresh = _as_vector(fresh_action, name="fresh_action", length=7)
    return float(np.mean(np.abs(queued - fresh)))


def build_rcv_features(
    *,
    state: Any,
    queued_action: Any,
    previous_action: Any,
    chunk_index_fraction: float,
    task_key: str,
    include_context: bool,
) -> np.ndarray:
    assert_no_privileged_inference_fields(
        [
            "observation.state",
            "queued_action",
            "previous_action",
            "chunk_index_fraction",
            "task_key",
        ]
    )
    queued = _as_vector(queued_action, name="queued_action", length=7)
    rho = np.asarray([float(chunk_index_fraction)], dtype=np.float64)
    one_hot = task_one_hot(str(task_key))
    if include_context:
        state_vec = _as_vector(state, name="state", length=8)
        previous = _as_vector(previous_action, name="previous_action", length=7)
        return np.concatenate([state_vec, queued, previous, rho, one_hot]).astype(np.float64)
    return np.concatenate([queued, rho, one_hot]).astype(np.float64)


def feature_names(*, include_context: bool) -> tuple[str, ...]:
    return FULL_FEATURE_NAMES if bool(include_context) else NO_CONTEXT_FEATURE_NAMES


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def _labels_from_disagreement(records: Sequence[Mapping[str, Any]], tau_train: float) -> np.ndarray:
    return np.asarray([float(record["disagreement"]) > float(tau_train) for record in records], dtype=np.float64)


def _features_from_records(records: Sequence[Mapping[str, Any]], *, include_context: bool) -> np.ndarray:
    if not records:
        return np.zeros((0, len(feature_names(include_context=include_context))), dtype=np.float64)
    rows = [
        build_rcv_features(
            state=record["state"],
            queued_action=record["queued_action"],
            previous_action=record["previous_action"],
            chunk_index_fraction=float(record["chunk_index_fraction"]),
            task_key=str(record["task_key"]),
            include_context=include_context,
        )
        for record in records
    ]
    return np.stack(rows, axis=0).astype(np.float64)


def binary_metrics(y_true: Any, probabilities: Any, *, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"metric shape mismatch: {y.size} labels vs {p.size} probabilities")
    if y.size == 0:
        return {
            "count": 0,
            "positive_rate": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "threshold": float(threshold),
        }
    pred = (p >= float(threshold)).astype(np.float64)
    tp = float(np.sum((pred == 1.0) & (y == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y == 0.0)))
    fp = float(np.sum((pred == 1.0) & (y == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y == 1.0)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "count": int(y.size),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def majority_baseline_metrics(y_true: Any) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return binary_metrics(y, y, threshold=0.5)
    majority = 1.0 if float(np.mean(y)) >= 0.5 else 0.0
    probabilities = np.full_like(y, majority, dtype=np.float64)
    return binary_metrics(y, probabilities, threshold=0.5)


def select_threshold(probabilities: Any, labels: Any) -> tuple[float, dict[str, Any]]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size != y.size:
        raise ValueError(f"threshold shape mismatch: {p.size} probabilities vs {y.size} labels")
    if p.size == 0:
        return 0.5, binary_metrics(y, p, threshold=0.5)
    unique = sorted({float(value) for value in p})
    candidates = {0.0, 0.5, 1.0}
    if len(unique) == 1:
        candidates.add(unique[0])
    else:
        candidates.update((left + right) / 2.0 for left, right in zip(unique[:-1], unique[1:]))
        candidates.add(max(0.0, unique[0] - 1e-9))
        candidates.add(min(1.0, unique[-1] + 1e-9))
    candidates = sorted(candidates)
    best_threshold = 0.5
    best_metrics = binary_metrics(y, p, threshold=best_threshold)
    for threshold in candidates:
        metrics = binary_metrics(y, p, threshold=threshold)
        if metrics["f1"] > best_metrics["f1"] or (
            metrics["f1"] == best_metrics["f1"] and float(threshold) > float(best_threshold)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray, config: RCVConfig) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")
    if x_train.shape[0] != y_train.size:
        raise ValueError("training feature/label count mismatch")
    if x_train.shape[0] == 0:
        raise ValueError("cannot train RCV verifier with zero records")
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale < float(config.eps), 1.0, scale)
    x = (x_train - mean) / scale
    rng = np.random.default_rng(int(config.seed))
    weights = rng.normal(loc=0.0, scale=0.01, size=x.shape[1])
    bias = 0.0
    y = y_train.astype(np.float64)
    lr = float(config.learning_rate)
    l2 = float(config.l2)
    for _ in range(int(config.max_epochs)):
        probabilities = _sigmoid(x @ weights + bias)
        residual = probabilities - y
        weights -= lr * ((x.T @ residual) / max(1, y.size) + l2 * weights)
        bias -= lr * float(np.mean(residual))
    return weights.astype(np.float64), float(bias), mean.astype(np.float64), scale.astype(np.float64)


def _predict_matrix(x: np.ndarray, *, weights: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("prediction features must be 2D")
    return _sigmoid(((x - mean) / scale) @ weights + float(bias))


def train_verifier(
    records: Sequence[Mapping[str, Any]],
    *,
    include_context: bool,
    config: RCVConfig | None = None,
    tau_train: float | None = None,
) -> dict[str, Any]:
    cfg = config or RCVConfig()
    train_records = [record for record in records if str(record.get("split", "train")) == "train"]
    calibration_records = [record for record in records if str(record.get("split", "train")) == "calibration"]
    if not train_records:
        raise ValueError("RCV verifier requires train records")
    if tau_train is None:
        tau = float(np.quantile([float(record["disagreement"]) for record in train_records], float(cfg.disagreement_quantile)))
    else:
        tau = float(tau_train)
    x_train = _features_from_records(train_records, include_context=include_context)
    y_train = _labels_from_disagreement(train_records, tau)
    weights, bias, mean, scale = _fit_logistic(x_train, y_train, cfg)
    train_probabilities = _predict_matrix(x_train, weights=weights, bias=bias, mean=mean, scale=scale)
    if calibration_records:
        x_calibration = _features_from_records(calibration_records, include_context=include_context)
        y_calibration = _labels_from_disagreement(calibration_records, tau)
        calibration_probabilities = _predict_matrix(x_calibration, weights=weights, bias=bias, mean=mean, scale=scale)
    else:
        y_calibration = y_train
        calibration_probabilities = train_probabilities
    theta, calibration_metrics = select_threshold(calibration_probabilities, y_calibration)
    return {
        "schema_version": "rcv_logistic_verifier_v1",
        "include_context": bool(include_context),
        "config": cfg.to_json(),
        "feature_names": list(feature_names(include_context=include_context)),
        "tau_train": float(tau),
        "theta_train": float(theta),
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "train_metrics": binary_metrics(y_train, train_probabilities, threshold=float(theta)),
        "calibration_metrics": calibration_metrics,
        "majority_baseline_metrics": majority_baseline_metrics(y_calibration),
        "train_record_count": len(train_records),
        "calibration_record_count": len(calibration_records),
    }


def predict_replan_probability(
    verifier: Mapping[str, Any],
    *,
    state: Any,
    queued_action: Any,
    previous_action: Any,
    chunk_index_fraction: float,
    task_key: str,
) -> float:
    include_context = bool(verifier["include_context"])
    features = build_rcv_features(
        state=state,
        queued_action=queued_action,
        previous_action=previous_action,
        chunk_index_fraction=float(chunk_index_fraction),
        task_key=str(task_key),
        include_context=include_context,
    ).reshape(1, -1)
    weights = np.asarray(verifier["weights"], dtype=np.float64).reshape(-1)
    mean = np.asarray(verifier["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(verifier["scale"], dtype=np.float64).reshape(-1)
    if features.shape[1] != weights.size:
        raise ValueError(f"verifier feature mismatch: {features.shape[1]} vs {weights.size}")
    return float(_predict_matrix(features, weights=weights, bias=float(verifier["bias"]), mean=mean, scale=scale)[0])


def verifier_replans(verifier: Mapping[str, Any], probability: float) -> bool:
    return float(probability) > float(verifier["theta_train"])


def save_verifier(path: str | Path, verifier: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(verifier), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_verifier(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
