"""CAVM-VLA lightweight outcome-contrastive memory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TASK_KEYS = ("libero_spatial/task_4", "libero_10/task_4")
VARIANTS = (
    "frozen_smolvla",
    "success_only_memory_proxy",
    "nearest_success_replay",
    "cavm_no_contrast_ablation",
    "cavm_full",
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
class CAVMConfig:
    k_success: int = 8
    k_failure: int = 8
    alpha: float = 0.35
    beta: float = 0.50
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    min_task_success_episodes: int = 2
    min_task_failure_episodes: int = 2
    min_gateable_fraction: float = 0.10
    min_action_separation: float = 0.05
    min_scale: float = 1e-6


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    return array


def task_one_hot(task_key: str) -> np.ndarray:
    if task_key not in TASK_KEYS:
        raise ValueError(f"unknown CAVM task key: {task_key}")
    out = np.zeros(len(TASK_KEYS), dtype=np.float64)
    out[TASK_KEYS.index(task_key)] = 1.0
    return out


def build_cavm_key(
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
        raise ValueError(f"privileged CAVM inference fields: {leaked}")


def _record_key(record: Mapping[str, Any]) -> np.ndarray:
    return build_cavm_key(
        state=record["state"],
        action=record["action"],
        previous_action=record["previous_action"],
        chunk_index_fraction=float(record["chunk_index_fraction"]),
        task_key=str(record["task_key"]),
    )


def _episode_counts(records: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    by_episode: dict[tuple[str, int], bool] = {}
    for record in records:
        by_episode[(str(record["task_key"]), int(record["identity"]))] = bool(record["success"])
    counts = {task: {"success": 0, "failure": 0} for task in TASK_KEYS}
    for (task, _identity), success in by_episode.items():
        if task not in counts:
            continue
        counts[task]["success" if success else "failure"] += 1
    return counts


def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean.reshape(1, -1)) / scale.reshape(1, -1)


def _median_same_task_nearest(features: np.ndarray, tasks: list[str], config: CAVMConfig, *, chunk_size: int = 256) -> float:
    nearest: list[float] = []
    for task in TASK_KEYS:
        indices = [index for index, value in enumerate(tasks) if value == task]
        if len(indices) < 2:
            continue
        task_features = features[np.asarray(indices, dtype=np.int64)]
        squared_norms = np.sum(task_features * task_features, axis=1)
        for start in range(0, int(task_features.shape[0]), int(chunk_size)):
            stop = min(start + int(chunk_size), int(task_features.shape[0]))
            block = task_features[start:stop]
            distances_sq = np.sum(block * block, axis=1).reshape(-1, 1) + squared_norms.reshape(1, -1) - 2.0 * (block @ task_features.T)
            distances_sq = np.maximum(distances_sq, 0.0)
            for local, global_index in enumerate(range(start, stop)):
                distances_sq[local, global_index] = np.inf
            finite = np.sqrt(np.min(distances_sq, axis=1))
            nearest.extend(float(value) for value in finite[np.isfinite(finite)])
    if not nearest:
        return 1.0
    return float(np.clip(np.median(np.asarray(nearest, dtype=np.float64)), config.sigma_min, config.sigma_max))


def _nearest_weighted_mean(
    features: np.ndarray,
    actions: np.ndarray,
    key: np.ndarray,
    *,
    k: int,
    sigma: float,
) -> tuple[np.ndarray | None, float, float]:
    if features.size == 0 or actions.size == 0:
        return None, float("inf"), 0.0
    distances = np.linalg.norm(features - key.reshape(1, -1), axis=1)
    count = min(int(k), int(distances.size))
    if count <= 0:
        return None, float("inf"), 0.0
    idx = np.argpartition(distances, count - 1)[:count]
    local_distances = distances[idx]
    local_actions = actions[idx]
    weights = np.exp(-local_distances / max(float(sigma), 1e-6))
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return None, float(np.min(local_distances)), 0.0
    mean = np.sum(local_actions * (weights / total).reshape(-1, 1), axis=0)
    return mean.astype(np.float64), float(np.min(local_distances)), float(np.mean(local_distances))


def _nearest_action(features: np.ndarray, actions: np.ndarray, key: np.ndarray) -> tuple[np.ndarray | None, float]:
    if features.size == 0 or actions.size == 0:
        return None, float("inf")
    distances = np.linalg.norm(features - key.reshape(1, -1), axis=1)
    index = int(np.argmin(distances))
    return actions[index].astype(np.float64), float(distances[index])


def _memory_arrays(records: list[Mapping[str, Any]], mean: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASK_KEYS:
        task_records = [record for record in records if str(record["task_key"]) == task]
        success_records = [record for record in task_records if bool(record["success"])]
        failure_records = [record for record in task_records if not bool(record["success"])]
        for label, subset in [("success", success_records), ("failure", failure_records)]:
            keys = np.asarray([_record_key(record) for record in subset], dtype=np.float64).reshape(len(subset), 25)
            actions = np.asarray([_as_vector("action", record["action"], 7) for record in subset], dtype=np.float64).reshape(len(subset), 7)
            out.setdefault(task, {})[f"{label}_features"] = _standardize(keys, mean, scale).tolist() if len(subset) else []
            out.setdefault(task, {})[f"{label}_actions"] = actions.tolist() if len(subset) else []
    return out


def _runtime_task(memory: Mapping[str, Any], task_key: str) -> dict[str, np.ndarray]:
    task = (memory.get("by_task") or {}).get(task_key) or {}
    return {
        "success_features": np.asarray(task.get("success_features") or [], dtype=np.float64).reshape(-1, 25),
        "success_actions": np.asarray(task.get("success_actions") or [], dtype=np.float64).reshape(-1, 7),
        "failure_features": np.asarray(task.get("failure_features") or [], dtype=np.float64).reshape(-1, 25),
        "failure_actions": np.asarray(task.get("failure_actions") or [], dtype=np.float64).reshape(-1, 7),
    }


def _query_stats(memory: Mapping[str, Any], key: np.ndarray, task_key: str) -> dict[str, Any]:
    cfg = memory["config"]
    task = _runtime_task(memory, task_key)
    mu_success, min_success, mean_success_distance = _nearest_weighted_mean(
        task["success_features"],
        task["success_actions"],
        key,
        k=int(cfg["k_success"]),
        sigma=float(memory["sigma"]),
    )
    mu_failure, min_failure, mean_failure_distance = _nearest_weighted_mean(
        task["failure_features"],
        task["failure_actions"],
        key,
        k=int(cfg["k_failure"]),
        sigma=float(memory["sigma"]),
    )
    nearest_success, nearest_success_distance = _nearest_action(task["success_features"], task["success_actions"], key)
    if mu_success is None:
        return {"available": False}
    separation = float(np.linalg.norm(mu_success - mu_failure)) if mu_failure is not None else 0.0
    success_density = float(np.exp(-min_success / max(float(memory["sigma"]), 1e-6))) if np.isfinite(min_success) else 0.0
    failure_density = float(np.exp(-min_failure / max(float(memory["sigma"]), 1e-6))) if np.isfinite(min_failure) else 0.0
    density = min(success_density, failure_density) if mu_failure is not None else success_density
    margin = (separation - float(memory["eta"])) / max(float(memory["gamma"]), 1e-6)
    contrast_gate = float(np.clip(margin, 0.0, 1.0)) * density if mu_failure is not None else 0.0
    return {
        "available": True,
        "mu_success": mu_success,
        "mu_failure": mu_failure,
        "nearest_success": nearest_success,
        "min_success_distance": min_success,
        "min_failure_distance": min_failure,
        "mean_success_distance": mean_success_distance,
        "mean_failure_distance": mean_failure_distance,
        "nearest_success_distance": nearest_success_distance,
        "success_density": success_density,
        "failure_density": failure_density,
        "density": density,
        "separation": separation,
        "contrast_gate": contrast_gate,
    }


def apply_cavm_action(
    memory: Mapping[str, Any],
    *,
    variant: str,
    state: Any,
    action: Any,
    previous_action: Any,
    chunk_index_fraction: float,
    task_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown CAVM variant: {variant}")
    base = _as_vector("action", action, 7)
    if variant == "frozen_smolvla":
        return base, {"gate": 0.0, "action_delta_l2": 0.0, "memory_available": False}
    validate_inference_fields(
        {
            "state_vector": state,
            "action_vector": action,
            "previous_action_vector": previous_action,
            "chunk_index_fraction": chunk_index_fraction,
            "task_key": task_key,
        }
    )
    mean = np.asarray(memory["feature_mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(memory["feature_scale"], dtype=np.float64).reshape(-1)
    key_raw = build_cavm_key(
        state=state,
        action=base,
        previous_action=previous_action,
        chunk_index_fraction=chunk_index_fraction,
        task_key=task_key,
    )
    key = ((key_raw - mean) / scale).astype(np.float64)
    stats = _query_stats(memory, key, task_key)
    if not stats.get("available"):
        return base, {"gate": 0.0, "action_delta_l2": 0.0, "memory_available": False}
    alpha = float(memory["config"]["alpha"])
    beta = float(memory["config"]["beta"])
    if variant == "nearest_success_replay":
        target = stats.get("nearest_success")
        gate = float(stats.get("success_density", 0.0))
    elif variant == "success_only_memory_proxy":
        target = stats["mu_success"]
        gate = float(stats.get("success_density", 0.0))
    elif variant == "cavm_no_contrast_ablation":
        target = stats["mu_success"]
        gate = float(stats.get("contrast_gate", 0.0))
    elif variant == "cavm_full":
        if stats.get("mu_failure") is None:
            target = stats["mu_success"]
            gate = 0.0
        else:
            target = stats["mu_success"] + beta * (stats["mu_success"] - stats["mu_failure"])
            gate = float(stats.get("contrast_gate", 0.0))
    else:  # pragma: no cover
        raise ValueError(variant)
    if target is None:
        return base, {"gate": 0.0, "action_delta_l2": 0.0, "memory_available": False}
    gate = float(np.clip(gate, 0.0, 1.0))
    adjusted = (1.0 - alpha * gate) * base + alpha * gate * np.asarray(target, dtype=np.float64).reshape(-1)
    adjusted = np.clip(adjusted, -1.0, 1.0)
    diagnostics = {
        "gate": gate,
        "memory_available": True,
        "action_delta_l2": float(np.linalg.norm(adjusted - base)),
        "success_failure_separation": float(stats.get("separation", 0.0)),
        "success_density": float(stats.get("success_density", 0.0)),
        "failure_density": float(stats.get("failure_density", 0.0)),
        "min_success_distance": float(stats.get("min_success_distance", 0.0)),
        "min_failure_distance": float(stats.get("min_failure_distance", 0.0)) if np.isfinite(float(stats.get("min_failure_distance", 0.0))) else 0.0,
    }
    return adjusted.astype(np.float64), diagnostics


def calibration_separations(memory: Mapping[str, Any], calibration_records: list[Mapping[str, Any]]) -> list[float]:
    out: list[float] = []
    mean = np.asarray(memory["feature_mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(memory["feature_scale"], dtype=np.float64).reshape(-1)
    for record in calibration_records:
        key = (_record_key(record) - mean) / scale
        stats = _query_stats(memory, key, str(record["task_key"]))
        if stats.get("available") and stats.get("mu_failure") is not None:
            out.append(float(stats.get("separation", 0.0)))
    return out


def fit_cavm_memory(
    acquisition_records: list[Mapping[str, Any]],
    calibration_records: list[Mapping[str, Any]],
    config: CAVMConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CAVMConfig()
    if not acquisition_records:
        raise ValueError("CAVM requires acquisition records")
    features = np.asarray([_record_key(record) for record in acquisition_records], dtype=np.float64)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < cfg.min_scale, 1.0, scale)
    standardized = _standardize(features, mean, scale)
    sigma = _median_same_task_nearest(standardized, [str(record["task_key"]) for record in acquisition_records], cfg)
    memory: dict[str, Any] = {
        "schema_version": "cavm_memory_v1",
        "task_keys": list(TASK_KEYS),
        "feature_dim": 25,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "sigma": float(sigma),
        "eta": float(cfg.min_action_separation),
        "gamma": float(cfg.min_action_separation),
        "config": {
            "k_success": int(cfg.k_success),
            "k_failure": int(cfg.k_failure),
            "alpha": float(cfg.alpha),
            "beta": float(cfg.beta),
            "min_gateable_fraction": float(cfg.min_gateable_fraction),
            "min_action_separation": float(cfg.min_action_separation),
        },
        "by_task": _memory_arrays(acquisition_records, mean, scale),
        "episode_counts": _episode_counts(acquisition_records),
    }
    separations = calibration_separations(memory, calibration_records)
    if separations:
        arr = np.asarray(separations, dtype=np.float64)
        q25 = float(np.quantile(arr, 0.25))
        q75 = float(np.quantile(arr, 0.75))
        memory["eta"] = float(max(cfg.min_action_separation, q25))
        memory["gamma"] = float(max(q75 - q25, cfg.min_action_separation))
    gateable_fraction = float(len(separations) / max(1, len(calibration_records)))
    median_separation = float(np.median(np.asarray(separations, dtype=np.float64))) if separations else 0.0
    hard_kill_reasons: list[str] = []
    counts = memory["episode_counts"]
    for task in TASK_KEYS:
        if int(counts[task]["success"]) < cfg.min_task_success_episodes:
            hard_kill_reasons.append(f"{task} has fewer than {cfg.min_task_success_episodes} successful acquisition episodes")
        if int(counts[task]["failure"]) < cfg.min_task_failure_episodes:
            hard_kill_reasons.append(f"{task} has fewer than {cfg.min_task_failure_episodes} failed acquisition episodes")
    if gateable_fraction < cfg.min_gateable_fraction:
        hard_kill_reasons.append(f"gateable calibration fraction {gateable_fraction:.6f} below {cfg.min_gateable_fraction:.6f}")
    if median_separation < cfg.min_action_separation:
        hard_kill_reasons.append(f"median separation {median_separation:.6f} below {cfg.min_action_separation:.6f}")
    memory["calibration_metrics"] = {
        "gateable_record_count": int(len(separations)),
        "calibration_record_count": int(len(calibration_records)),
        "gateable_fraction": gateable_fraction,
        "median_success_failure_separation": median_separation,
        "mean_success_failure_separation": float(np.mean(np.asarray(separations, dtype=np.float64))) if separations else 0.0,
        "hard_kill_reasons": hard_kill_reasons,
    }
    memory["final_decision"] = "STAGE_1_PROCEED_TO_STAGE_2A" if not hard_kill_reasons else "STAGE_0_PERMANENT_KILL_NO_CONTRASTIVE_MEMORY"
    return memory
