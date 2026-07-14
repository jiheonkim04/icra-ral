"""FANG-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


TASK_KEYS = ("libero_spatial/task_4", "libero_10/task_4")
VARIANTS = (
    "base_smolvla",
    "afil_local_proxy",
    "fang_full",
    "fang_no_failure_ablation",
    "nearest_success_replay",
)
FORBIDDEN_INFERENCE_KEYS = {
    "object_state",
    "object_pose",
    "reward",
    "success",
    "terminal_success",
    "task_progress",
    "bddl",
    "future_action",
    "identity",
}


@dataclass(frozen=True)
class FANGAuditConfig:
    train_identities: tuple[int, ...] = tuple(range(20260901, 20260911))
    validation_identities: tuple[int, ...] = tuple(range(20260911, 20260917))
    forbidden_confirmatory_identities: tuple[int, ...] = tuple(range(20260917, 20261091))
    k_neighbors: int = 8
    min_class_rows: int = 250
    min_class_identities_per_task: int = 2
    min_validation_gateable_fraction: float = 0.10
    min_action_field_separation: float = 0.05
    min_target_variance: float = 1e-6
    min_scale: float = 1e-6
    max_abs_action: float = 5.0


class FANGHead(torch.nn.Module):
    def __init__(self, input_dim: int = 25, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.m_plus = torch.nn.Linear(hidden_dim, 7)
        self.m_minus = torch.nn.Linear(hidden_dim, 7)
        self.gate = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.m_plus(h), self.m_minus(h), self.gate(h).reshape(-1)


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    return array


def task_one_hot(task_key: str) -> np.ndarray:
    if task_key not in TASK_KEYS:
        raise ValueError(f"unknown FANG task key: {task_key}")
    out = np.zeros(len(TASK_KEYS), dtype=np.float64)
    out[TASK_KEYS.index(task_key)] = 1.0
    return out


def build_fang_feature(
    *,
    state: Any,
    action: Any,
    previous_action: Any,
    chunk_index_fraction: float,
    task_key: str,
) -> np.ndarray:
    state_vec = _as_vector("state", state, 8)
    action_vec = _as_vector("action", action, 7)
    previous_vec = _as_vector("previous_action", previous_action, 7)
    rho = np.asarray([float(chunk_index_fraction)], dtype=np.float64)
    return np.concatenate([state_vec, action_vec, previous_vec, rho, task_one_hot(task_key)]).astype(np.float64)


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged FANG inference fields: {leaked}")


def _record_feature(record: Mapping[str, Any]) -> np.ndarray:
    return build_fang_feature(
        state=record["state"],
        action=record["action"],
        previous_action=record["previous_action"],
        chunk_index_fraction=float(record["chunk_index_fraction"]),
        task_key=str(record["task_key"]),
    )


def _record_action(record: Mapping[str, Any]) -> np.ndarray:
    return _as_vector("action", record["action"], 7)


def _identity_set(values: Sequence[int]) -> set[int]:
    return {int(value) for value in values}


def _split_name(identity: int, config: FANGAuditConfig) -> str | None:
    if int(identity) in _identity_set(config.train_identities):
        return "DISCOVERY_TRAIN"
    if int(identity) in _identity_set(config.validation_identities):
        return "VALIDATION"
    return None


def _finite_stats(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {"finite": False, "max_abs": None, "mean_l2": None}
    finite = bool(np.all(np.isfinite(actions)))
    max_abs = float(np.max(np.abs(actions))) if finite else float("inf")
    mean_l2 = float(np.mean(np.linalg.norm(actions, axis=1))) if finite else float("inf")
    return {"finite": finite, "max_abs": max_abs, "mean_l2": mean_l2}


def _standardization(features: np.ndarray, config: FANGAuditConfig) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < config.min_scale, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - mean.reshape(1, -1)) / scale.reshape(1, -1)


def _weighted_neighbor_mean(features: np.ndarray, actions: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray | None, float]:
    if features.size == 0 or actions.size == 0:
        return None, float("inf")
    distances = np.linalg.norm(features - query.reshape(1, -1), axis=1)
    count = min(int(k), int(distances.size))
    if count <= 0:
        return None, float("inf")
    idx = np.argpartition(distances, count - 1)[:count]
    local_distances = distances[idx]
    local_actions = actions[idx]
    sigma = float(max(np.median(local_distances), 1e-6))
    weights = np.exp(-local_distances / sigma)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None, float(np.min(local_distances))
    return np.sum(local_actions * (weights / total).reshape(-1, 1), axis=0), float(np.min(local_distances))


def split_development_records(records: Sequence[Mapping[str, Any]], config: FANGAuditConfig | None = None) -> dict[str, list[Mapping[str, Any]]]:
    cfg = config or FANGAuditConfig()
    train_ids = _identity_set(cfg.train_identities)
    val_ids = _identity_set(cfg.validation_identities)
    relevant = [record for record in records if str(record.get("task_key")) in TASK_KEYS]
    return {
        "train": [record for record in relevant if int(record["identity"]) in train_ids],
        "validation": [record for record in relevant if int(record["identity"]) in val_ids],
    }


def records_to_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = np.asarray([_record_feature(record) for record in records], dtype=np.float64).reshape(len(records), 25) if records else np.empty((0, 25), dtype=np.float64)
    actions = np.asarray([_record_action(record) for record in records], dtype=np.float64).reshape(len(records), 7) if records else np.empty((0, 7), dtype=np.float64)
    labels = np.asarray([1 if bool(record["success"]) else 0 for record in records], dtype=np.int64)
    tasks = [str(record["task_key"]) for record in records]
    identities = np.asarray([int(record["identity"]) for record in records], dtype=np.int64)
    return {
        "features": features,
        "actions": actions,
        "labels": labels,
        "tasks": tasks,
        "identities": identities,
    }


def standardize_train_validation(train_features: np.ndarray, validation_features: np.ndarray, config: FANGAuditConfig | None = None) -> dict[str, np.ndarray]:
    cfg = config or FANGAuditConfig()
    mean, scale = _standardization(train_features, cfg)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "train_features": _standardize(train_features, mean, scale),
        "validation_features": _standardize(validation_features, mean, scale),
    }


def compute_gate_targets(
    *,
    train_features_std: np.ndarray,
    train_actions: np.ndarray,
    train_tasks: Sequence[str],
    train_labels: np.ndarray,
    query_features_std: np.ndarray,
    query_tasks: Sequence[str],
    config: FANGAuditConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FANGAuditConfig()
    separations: list[float] = []
    densities: list[float] = []
    for query, task in zip(query_features_std, query_tasks):
        success_mask = np.asarray([task_value == task and bool(label) for task_value, label in zip(train_tasks, train_labels)], dtype=bool)
        failure_mask = np.asarray([task_value == task and not bool(label) for task_value, label in zip(train_tasks, train_labels)], dtype=bool)
        mu_success, min_success = _weighted_neighbor_mean(train_features_std[success_mask], train_actions[success_mask], query, cfg.k_neighbors)
        mu_failure, min_failure = _weighted_neighbor_mean(train_features_std[failure_mask], train_actions[failure_mask], query, cfg.k_neighbors)
        if mu_success is None or mu_failure is None:
            separations.append(0.0)
            densities.append(0.0)
            continue
        separations.append(float(np.linalg.norm(mu_success - mu_failure)))
        local_min = min(float(min_success), float(min_failure))
        densities.append(float(np.exp(-local_min)))
    sep = np.asarray(separations, dtype=np.float64)
    density = np.asarray(densities, dtype=np.float64)
    positive = sep[sep > 0.0]
    if positive.size:
        q25 = float(np.percentile(positive, 25))
        q75 = float(np.percentile(positive, 75))
        gamma = max(q75 - q25, cfg.min_action_field_separation)
    else:
        q25 = 0.0
        q75 = 0.0
        gamma = cfg.min_action_field_separation
    target = density * np.clip((sep - cfg.min_action_field_separation) / max(gamma, 1e-6), 0.0, 1.0)
    return {
        "targets": target.astype(np.float64),
        "separations": sep.astype(np.float64),
        "densities": density.astype(np.float64),
        "eta": cfg.min_action_field_separation,
        "gamma": gamma,
        "q25": q25,
        "q75": q75,
    }


def _clip_l2_np(value: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm <= 1e-12:
        return vector
    return vector * (float(max_norm) / norm)


def load_fang_runtime(
    *,
    checkpoint_path: str,
    records: Sequence[Mapping[str, Any]],
    selected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FANGHead()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = dict((selected_config or {}).get("config") or checkpoint.get("config") or {})
    gate_tau = float((selected_config or {}).get("gate_tau", checkpoint.get("gate_tau", 0.0)))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64).reshape(25)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float64).reshape(25)
    splits = split_development_records(records)
    train_arrays = records_to_arrays(splits["train"])
    train_features_std = _standardize(train_arrays["features"], feature_mean, feature_scale)
    success_memory: dict[str, dict[str, np.ndarray]] = {}
    for task in TASK_KEYS:
        mask = np.asarray([task_value == task and bool(label) for task_value, label in zip(train_arrays["tasks"], train_arrays["labels"])], dtype=bool)
        success_memory[task] = {
            "features": train_features_std[mask],
            "actions": train_arrays["actions"][mask],
        }
    return {
        "model": model,
        "config": config,
        "gate_tau": gate_tau,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "success_memory": success_memory,
        "checkpoint_path": checkpoint_path,
    }


def _nearest_success_action(runtime: Mapping[str, Any], task_key: str, key_std: np.ndarray) -> tuple[np.ndarray | None, float]:
    memory = (runtime.get("success_memory") or {}).get(task_key) or {}
    raw_features = memory.get("features")
    raw_actions = memory.get("actions")
    features = np.asarray(raw_features if raw_features is not None else [], dtype=np.float64).reshape(-1, 25)
    actions = np.asarray(raw_actions if raw_actions is not None else [], dtype=np.float64).reshape(-1, 7)
    if features.size == 0 or actions.size == 0:
        return None, float("inf")
    distances = np.linalg.norm(features - key_std.reshape(1, -1), axis=1)
    index = int(np.argmin(distances))
    return actions[index].astype(np.float64), float(distances[index])


def apply_fang_action(
    runtime: Mapping[str, Any],
    *,
    variant: str,
    state: Any,
    action: Any,
    previous_action: Any,
    chunk_index_fraction: float,
    task_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown FANG variant: {variant}")
    base = _as_vector("action", action, 7)
    if variant == "base_smolvla":
        return base, {"gate": 0.0, "action_delta_l2": 0.0, "head_separation": 0.0, "memory_available": False}
    validate_inference_fields(
        {
            "state_vector": state,
            "action_vector": action,
            "previous_action_vector": previous_action,
            "chunk_index_fraction": chunk_index_fraction,
            "task_key": task_key,
        }
    )
    key = build_fang_feature(
        state=state,
        action=base,
        previous_action=previous_action,
        chunk_index_fraction=chunk_index_fraction,
        task_key=task_key,
    )
    mean = np.asarray(runtime["feature_mean"], dtype=np.float64).reshape(25)
    scale = np.asarray(runtime["feature_scale"], dtype=np.float64).reshape(25)
    key_std = ((key - mean) / scale).astype(np.float64)
    alpha = float((runtime.get("config") or {}).get("alpha", 0.10))
    beta = float((runtime.get("config") or {}).get("beta", 0.50))
    delta_max = 0.35
    if variant == "nearest_success_replay":
        nearest, distance = _nearest_success_action(runtime, task_key, key_std)
        if nearest is None:
            return base, {"gate": 0.0, "action_delta_l2": 0.0, "head_separation": 0.0, "memory_available": False}
        delta = alpha * _clip_l2_np(np.asarray(nearest, dtype=np.float64).reshape(-1) - base, delta_max)
        adjusted = np.clip(base + delta, -1.0, 1.0)
        return adjusted.astype(np.float64), {
            "gate": 1.0,
            "action_delta_l2": float(np.linalg.norm(adjusted - base)),
            "head_separation": 0.0,
            "memory_available": True,
            "nearest_success_distance": float(distance),
        }
    model = runtime["model"]
    with torch.no_grad():
        x = torch.as_tensor(key_std.reshape(1, -1), dtype=torch.float32)
        m_plus, m_minus, gate_logits = model(x)
        plus = m_plus.detach().cpu().numpy().reshape(-1).astype(np.float64)
        minus = m_minus.detach().cpu().numpy().reshape(-1).astype(np.float64)
        raw_gate = float(torch.sigmoid(gate_logits).detach().cpu().numpy().reshape(-1)[0])
        calibrated_gate = float(torch.sigmoid(gate_logits - float(runtime.get("gate_tau", 0.0))).detach().cpu().numpy().reshape(-1)[0])
    if variant == "afil_local_proxy":
        gate = raw_gate
        guidance = (plus - base) + beta * (plus - minus)
    elif variant == "fang_full":
        gate = calibrated_gate
        guidance = (plus - base) + beta * (plus - minus)
    elif variant == "fang_no_failure_ablation":
        gate = calibrated_gate
        guidance = plus - base
    else:  # pragma: no cover
        raise ValueError(variant)
    delta = alpha * gate * _clip_l2_np(guidance, delta_max)
    adjusted = np.clip(base + delta, -1.0, 1.0)
    return adjusted.astype(np.float64), {
        "gate": float(gate),
        "raw_gate": float(raw_gate),
        "calibrated_gate": float(calibrated_gate),
        "action_delta_l2": float(np.linalg.norm(adjusted - base)),
        "head_separation": float(np.linalg.norm(plus - minus)),
        "memory_available": True,
    }


def _class_identity_counts(records: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out = {task: {"success": 0, "failure": 0} for task in TASK_KEYS}
    episode_labels: dict[tuple[str, int], bool] = {}
    for record in records:
        task = str(record["task_key"])
        if task in out:
            episode_labels[(task, int(record["identity"]))] = bool(record["success"])
    for (task, _identity), success in episode_labels.items():
        out[task]["success" if success else "failure"] += 1
    return out


def _class_row_counts(records: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out = {task: {"success": 0, "failure": 0} for task in TASK_KEYS}
    for record in records:
        task = str(record["task_key"])
        if task in out:
            out[task]["success" if bool(record["success"]) else "failure"] += 1
    return out


def _duplicate_count(records: list[Mapping[str, Any]]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    duplicates = 0
    for record in records:
        key = (
            str(record.get("split", "")),
            str(record["task_key"]),
            int(record["identity"]),
            int(record["step"]),
        )
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _variance_by_task_class(records: list[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {task: {"success": None, "failure": None} for task in TASK_KEYS}
    for task in TASK_KEYS:
        for label, success in (("success", True), ("failure", False)):
            actions = np.asarray([_record_action(record) for record in records if str(record["task_key"]) == task and bool(record["success"]) is success], dtype=np.float64)
            if actions.size == 0:
                continue
            out[task][label] = float(np.mean(np.var(actions.reshape(-1, 7), axis=0)))
    return out


def _class_mean_separation(records: list[Mapping[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for task in TASK_KEYS:
        success_actions = np.asarray([_record_action(record) for record in records if str(record["task_key"]) == task and bool(record["success"])], dtype=np.float64).reshape(-1, 7)
        failure_actions = np.asarray([_record_action(record) for record in records if str(record["task_key"]) == task and not bool(record["success"])], dtype=np.float64).reshape(-1, 7)
        if success_actions.size == 0 or failure_actions.size == 0:
            out[task] = None
        else:
            out[task] = float(np.linalg.norm(np.mean(success_actions, axis=0) - np.mean(failure_actions, axis=0)))
    return out


def audit_fang_records(records: Sequence[Mapping[str, Any]], config: FANGAuditConfig | None = None) -> dict[str, Any]:
    cfg = config or FANGAuditConfig()
    relevant = [record for record in records if str(record.get("task_key")) in TASK_KEYS]
    train_ids = _identity_set(cfg.train_identities)
    val_ids = _identity_set(cfg.validation_identities)
    forbidden_ids = _identity_set(cfg.forbidden_confirmatory_identities)
    development_ids = train_ids | val_ids
    development = [record for record in relevant if int(record["identity"]) in development_ids]
    train = [record for record in development if int(record["identity"]) in train_ids]
    validation = [record for record in development if int(record["identity"]) in val_ids]
    hard_stop_reasons: list[str] = []
    classifications: set[str] = set()

    if not train or not validation:
        hard_stop_reasons.append("missing train or validation rows")
        classifications.add("DATA_FAILURE")

    forbidden_present = sorted({int(record["identity"]) for record in relevant if int(record["identity"]) in forbidden_ids})
    if forbidden_present:
        hard_stop_reasons.append(f"confirmatory identities present in development records: {forbidden_present[:5]}")
        classifications.add("DATA_FAILURE")

    duplicates = _duplicate_count(development)
    if duplicates:
        hard_stop_reasons.append(f"duplicate development keys: {duplicates}")
        classifications.add("DATA_FAILURE")

    combined_identity_counts = _class_identity_counts(development)
    train_identity_counts = _class_identity_counts(train)
    row_counts = _class_row_counts(development)
    train_row_counts = _class_row_counts(train)
    validation_row_counts = _class_row_counts(validation)

    for task in TASK_KEYS:
        for label in ("success", "failure"):
            if combined_identity_counts[task][label] < cfg.min_class_identities_per_task:
                hard_stop_reasons.append(f"{task} has too few {label} identities: {combined_identity_counts[task][label]}")
                classifications.add("DATA_FAILURE")
            if train_row_counts[task][label] < cfg.min_class_rows:
                hard_stop_reasons.append(f"{task} has too few train {label} rows: {train_row_counts[task][label]}")
                classifications.add("DATA_FAILURE")

    train_features = np.asarray([_record_feature(record) for record in train], dtype=np.float64).reshape(len(train), 25) if train else np.empty((0, 25), dtype=np.float64)
    train_actions = np.asarray([_record_action(record) for record in train], dtype=np.float64).reshape(len(train), 7) if train else np.empty((0, 7), dtype=np.float64)
    validation_features = np.asarray([_record_feature(record) for record in validation], dtype=np.float64).reshape(len(validation), 25) if validation else np.empty((0, 25), dtype=np.float64)
    action_stats = _finite_stats(np.concatenate([train_actions, np.asarray([_record_action(record) for record in validation], dtype=np.float64).reshape(len(validation), 7)], axis=0) if validation else train_actions)
    if not action_stats["finite"]:
        hard_stop_reasons.append("nonfinite action value detected")
        classifications.add("DATA_FAILURE")
    elif float(action_stats["max_abs"] or 0.0) > cfg.max_abs_action:
        hard_stop_reasons.append(f"action magnitude exceeds audit bound: {action_stats['max_abs']}")
        classifications.add("DESIGN_FAILURE")

    variance = _variance_by_task_class(train)
    for task, by_label in variance.items():
        for label, value in by_label.items():
            if value is None or float(value) < cfg.min_target_variance:
                hard_stop_reasons.append(f"{task} train {label} action-field target variance collapsed: {value}")
                classifications.add("DATA_FAILURE")

    class_mean_sep = _class_mean_separation(train)
    validation_separations: list[float] = []
    validation_gateable = 0
    if train_features.size and validation_features.size:
        mean, scale = _standardization(train_features, cfg)
        train_std = _standardize(train_features, mean, scale)
        val_std = _standardize(validation_features, mean, scale)
        train_tasks = [str(record["task_key"]) for record in train]
        train_labels = [bool(record["success"]) for record in train]
        train_actions_arr = train_actions.reshape(-1, 7)
        for query, record in zip(val_std, validation):
            task = str(record["task_key"])
            success_mask = np.asarray([task_value == task and label for task_value, label in zip(train_tasks, train_labels)], dtype=bool)
            failure_mask = np.asarray([task_value == task and not label for task_value, label in zip(train_tasks, train_labels)], dtype=bool)
            mu_success, _ = _weighted_neighbor_mean(train_std[success_mask], train_actions_arr[success_mask], query, cfg.k_neighbors)
            mu_failure, _ = _weighted_neighbor_mean(train_std[failure_mask], train_actions_arr[failure_mask], query, cfg.k_neighbors)
            if mu_success is None or mu_failure is None:
                continue
            separation = float(np.linalg.norm(mu_success - mu_failure))
            validation_separations.append(separation)
            validation_gateable += 1

    total_validation = len(validation)
    gateable_fraction = float(validation_gateable / total_validation) if total_validation else 0.0
    separation_array = np.asarray(validation_separations, dtype=np.float64)
    separation_summary = {
        "count": int(separation_array.size),
        "median": float(np.median(separation_array)) if separation_array.size else None,
        "q25": float(np.percentile(separation_array, 25)) if separation_array.size else None,
        "q75": float(np.percentile(separation_array, 75)) if separation_array.size else None,
        "mean": float(np.mean(separation_array)) if separation_array.size else None,
    }
    if gateable_fraction < cfg.min_validation_gateable_fraction:
        hard_stop_reasons.append(f"validation gateable fraction too low: {gateable_fraction:.6f}")
        classifications.add("NO_HEADROOM")
    if separation_summary["median"] is None or float(separation_summary["median"]) < cfg.min_action_field_separation:
        hard_stop_reasons.append(f"median validation action-field separation below {cfg.min_action_field_separation}: {separation_summary['median']}")
        classifications.add("NO_HEADROOM")

    final_decision = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" if not hard_stop_reasons else "AUDIT_STOP_" + "_".join(sorted(classifications))
    return {
        "schema_version": 1,
        "method": "FANG-VLA",
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "config": {
            "train_identities": list(cfg.train_identities),
            "validation_identities": list(cfg.validation_identities),
            "forbidden_confirmatory_identity_min": min(cfg.forbidden_confirmatory_identities),
            "forbidden_confirmatory_identity_max": max(cfg.forbidden_confirmatory_identities),
            "k_neighbors": cfg.k_neighbors,
            "min_class_rows": cfg.min_class_rows,
            "min_class_identities_per_task": cfg.min_class_identities_per_task,
            "min_validation_gateable_fraction": cfg.min_validation_gateable_fraction,
            "min_action_field_separation": cfg.min_action_field_separation,
        },
        "final_decision": final_decision,
        "hard_stop_reasons": hard_stop_reasons,
        "failure_classifications": sorted(classifications),
        "total_input_records": len(records),
        "development_records": len(development),
        "train_records": len(train),
        "validation_records": len(validation),
        "duplicate_development_keys": duplicates,
        "forbidden_confirmatory_identities_present": forbidden_present,
        "combined_identity_counts": combined_identity_counts,
        "train_identity_counts": train_identity_counts,
        "row_counts": row_counts,
        "train_row_counts": train_row_counts,
        "validation_row_counts": validation_row_counts,
        "action_stats": action_stats,
        "train_action_variance_by_task_class": variance,
        "train_class_mean_separation_by_task": class_mean_sep,
        "validation_gateable_records": validation_gateable,
        "validation_gateable_fraction": gateable_fraction,
        "validation_action_field_separation": separation_summary,
        "next_step": "Run bounded train-validate search." if not hard_stop_reasons else "Do not train or rollout this formulation; classify the audit stop before pivot or redesign.",
    }


__all__ = [
    "FANGAuditConfig",
    "FANGHead",
    "TASK_KEYS",
    "VARIANTS",
    "apply_fang_action",
    "audit_fang_records",
    "build_fang_feature",
    "compute_gate_targets",
    "load_fang_runtime",
    "records_to_arrays",
    "split_development_records",
    "standardize_train_validation",
    "validate_inference_fields",
]
