"""Outcome-conditioned flow-noise prior utilities for SmolVLA prototypes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PRIVILEGED_INFERENCE_FIELDS = {
    "sim_state",
    "mujoco_state",
    "success",
    "reward",
    "future_observation",
    "future_action_target",
    "object_pose",
    "bddl_predicate",
    "reset_identity",
    "task_outcome",
    "episode_success",
    "train_success",
}


@dataclass(frozen=True)
class OCFNConfig:
    noise_count: int = 4
    chunk_size: int = 50
    max_action_dim: int = 32
    seed_base: int = 2026071203
    task_shuffle_seed: int = 2026071204

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoiseSelection:
    variant: str
    task_key: str
    noise_id: int | None
    source: str
    stats: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "task_key": self.task_key,
            "noise_id": self.noise_id,
            "source": self.source,
            "stats": self.stats,
        }


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged OCFN inference fields: {forbidden}")


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}/task_{int(task_id)}"


def make_noise_bank(config: OCFNConfig, *, include_batch_dim: bool = True) -> dict[int, np.ndarray]:
    if int(config.noise_count) < 1:
        raise ValueError("noise_count must be positive")
    if int(config.chunk_size) < 1 or int(config.max_action_dim) < 1:
        raise ValueError("chunk_size and max_action_dim must be positive")
    bank: dict[int, np.ndarray] = {}
    for noise_id in range(int(config.noise_count)):
        rng = np.random.default_rng(int(config.seed_base) + int(noise_id))
        noise = rng.standard_normal((int(config.chunk_size), int(config.max_action_dim))).astype(np.float32)
        if include_batch_dim:
            noise = noise.reshape(1, int(config.chunk_size), int(config.max_action_dim))
        bank[int(noise_id)] = noise
    return bank


def zero_noise(config: OCFNConfig, *, include_batch_dim: bool = True) -> np.ndarray:
    shape = (int(config.chunk_size), int(config.max_action_dim))
    if include_batch_dim:
        shape = (1, *shape)
    return np.zeros(shape, dtype=np.float32)


def noise_sha256(noise: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(noise, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _row_success(row: Mapping[str, Any]) -> bool:
    return bool(row.get("success"))


def _row_steps(row: Mapping[str, Any]) -> float:
    value = row.get("episode_steps")
    if value is None:
        value = row.get("steps")
    return float(value if value is not None else 1e9)


def _row_reward(row: Mapping[str, Any]) -> float:
    value = row.get("reward_sum")
    return float(value if value is not None else 0.0)


def aggregate_noise_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for row in rows:
        task = str(row["task_key"])
        noise_id = int(row["noise_id"])
        grouped.setdefault(task, {}).setdefault(noise_id, []).append(row)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for task, by_noise in grouped.items():
        out[task] = {}
        for noise_id, items in by_noise.items():
            successes = int(sum(1 for item in items if _row_success(item)))
            total = int(len(items))
            out[task][noise_id] = {
                "successes": successes,
                "total": total,
                "success_rate": float(successes / max(1, total)),
                "mean_episode_steps": _mean([_row_steps(item) for item in items]),
                "mean_reward_sum": _mean([_row_reward(item) for item in items]),
            }
    return out


def _best_noise_from_stats(stats: Mapping[int, Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    if not stats:
        raise ValueError("cannot choose noise from empty stats")

    def key(item: tuple[int, Mapping[str, Any]]) -> tuple[float, float, float, int]:
        noise_id, values = item
        return (
            float(values.get("successes", 0)),
            -float(values.get("mean_episode_steps", 1e9)),
            float(values.get("mean_reward_sum", 0.0)),
            -int(noise_id),
        )

    best_noise, best_stats = max(stats.items(), key=key)
    return int(best_noise), dict(best_stats)


def choose_task_noise_prior(rows: Sequence[Mapping[str, Any]], task_keys: Sequence[str]) -> dict[str, NoiseSelection]:
    stats = aggregate_noise_stats(rows)
    selections: dict[str, NoiseSelection] = {}
    for key in task_keys:
        if str(key) not in stats:
            raise ValueError(f"missing train rows for task {key}")
        noise_id, noise_stats = _best_noise_from_stats(stats[str(key)])
        selections[str(key)] = NoiseSelection(
            variant="ocfn_full",
            task_key=str(key),
            noise_id=int(noise_id),
            source="task_success_noise_prior",
            stats=noise_stats,
        )
    return selections


def choose_global_noise_prior(rows: Sequence[Mapping[str, Any]], task_keys: Sequence[str]) -> dict[str, NoiseSelection]:
    pooled: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        pooled.setdefault(int(row["noise_id"]), []).append(row)
    pooled_stats = {
        noise_id: {
            "successes": int(sum(1 for item in items if _row_success(item))),
            "total": int(len(items)),
            "success_rate": float(sum(1 for item in items if _row_success(item)) / max(1, len(items))),
            "mean_episode_steps": _mean([_row_steps(item) for item in items]),
            "mean_reward_sum": _mean([_row_reward(item) for item in items]),
        }
        for noise_id, items in pooled.items()
    }
    noise_id, noise_stats = _best_noise_from_stats(pooled_stats)
    return {
        str(key): NoiseSelection(
            variant="global_success_noise_prior",
            task_key=str(key),
            noise_id=int(noise_id),
            source="global_success_noise_prior",
            stats=dict(noise_stats),
        )
        for key in task_keys
    }


def shuffled_task_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    task_keys = sorted({str(row["task_key"]) for row in rows})
    if len(task_keys) < 2:
        return [dict(row) for row in rows]
    rng = np.random.default_rng(int(seed))
    permuted = task_keys[:]
    while permuted == task_keys:
        permuted = [task_keys[int(index)] for index in rng.permutation(len(task_keys))]
    mapping = dict(zip(task_keys, permuted))
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["original_task_key"] = str(row["task_key"])
        copied["task_key"] = mapping[str(row["task_key"])]
        out.append(copied)
    return out


def build_all_selections(rows: Sequence[Mapping[str, Any]], task_keys: Sequence[str], config: OCFNConfig) -> dict[str, dict[str, NoiseSelection]]:
    full = choose_task_noise_prior(rows, task_keys)
    global_prior = choose_global_noise_prior(rows, task_keys)
    shuffled = choose_task_noise_prior(shuffled_task_rows(rows, seed=int(config.task_shuffle_seed)), task_keys)
    shuffled = {
        key: NoiseSelection(
            variant="task_shuffled_noise_prior",
            task_key=key,
            noise_id=value.noise_id,
            source="task_shuffled_noise_prior",
            stats=value.stats,
        )
        for key, value in shuffled.items()
    }
    zero = {
        str(key): NoiseSelection("zero_noise_smolvla", str(key), None, "zero_noise", {"all_zero_noise": True})
        for key in task_keys
    }
    frozen = {
        str(key): NoiseSelection("frozen_smolvla", str(key), None, "default_policy_noise", {"custom_noise": False})
        for key in task_keys
    }
    return {
        "frozen_smolvla": frozen,
        "zero_noise_smolvla": zero,
        "global_success_noise_prior": global_prior,
        "task_shuffled_noise_prior": shuffled,
        "ocfn_full": full,
    }


def selections_to_json(selections: Mapping[str, Mapping[str, NoiseSelection]]) -> dict[str, dict[str, Any]]:
    return {
        str(variant): {str(task): selection.to_json() for task, selection in by_task.items()}
        for variant, by_task in selections.items()
    }


def full_equals_baseline(
    selections: Mapping[str, Mapping[str, NoiseSelection]],
    baseline: str,
    task_keys: Sequence[str],
) -> bool:
    full = selections.get("ocfn_full") or {}
    other = selections.get(str(baseline)) or {}
    for key in task_keys:
        if full.get(str(key)) is None or other.get(str(key)) is None:
            return False
        if full[str(key)].noise_id != other[str(key)].noise_id:
            return False
    return True


def stage_a_decision(
    summary_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    full_global_equivalent: bool,
    full_shuffled_equivalent: bool,
    full_action_delta_vs_global: float,
    full_action_delta_vs_shuffled: float,
) -> str:
    baseline_names = [
        "frozen_smolvla",
        "zero_noise_smolvla",
        "global_success_noise_prior",
        "task_shuffled_noise_prior",
    ]
    if full_global_equivalent and full_shuffled_equivalent and abs(float(full_action_delta_vs_global)) < 1e-6 and abs(float(full_action_delta_vs_shuffled)) < 1e-6:
        return "STAGE_A_PERMANENT_KILL_TRIVIAL_EQUIVALENCE"
    full = float((summary_by_variant.get("ocfn_full") or {}).get("task_balanced_success_rate", 0.0))
    strongest = max(float((summary_by_variant.get(name) or {}).get("task_balanced_success_rate", 0.0)) for name in baseline_names)
    if strongest - full >= 0.30:
        return "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    full_successes = int((summary_by_variant.get("ocfn_full") or {}).get("successes", 0))
    if full_successes == 0 and any(int((summary_by_variant.get(name) or {}).get("successes", 0)) >= 4 for name in baseline_names):
        return "STAGE_A_PERMANENT_KILL_ZERO_SUCCESS_WITH_BASELINE_HEADROOM"
    frozen = float((summary_by_variant.get("frozen_smolvla") or {}).get("task_balanced_success_rate", 0.0))
    shuffled = float((summary_by_variant.get("task_shuffled_noise_prior") or {}).get("task_balanced_success_rate", 0.0))
    if full > frozen and full > shuffled:
        return "STAGE_A_POSITIVE_TO_STAGE_B"
    return "STAGE_A_NON_GO_TO_STAGE_B_REQUIRED"


def stage_b_decision(
    summary_by_variant: Mapping[str, Mapping[str, Any]],
    paired_vs_full: Mapping[str, Mapping[str, Any]],
    *,
    mechanism_active: bool,
    complete: bool,
    exception_count: int,
    pairs_per_policy: int,
) -> str:
    baseline_names = [
        "frozen_smolvla",
        "zero_noise_smolvla",
        "global_success_noise_prior",
        "task_shuffled_noise_prior",
    ]
    if not bool(complete) or int(exception_count) > 0:
        return "STAGE_B_MEASUREMENT_INVALID"
    if not bool(mechanism_active):
        return "STAGE_B_PERMANENT_KILL_TRIVIAL_EQUIVALENCE"

    full = float((summary_by_variant.get("ocfn_full") or {}).get("task_balanced_success_rate", 0.0))
    strongest_name = max(
        baseline_names,
        key=lambda name: float((summary_by_variant.get(name) or {}).get("task_balanced_success_rate", 0.0)),
    )
    strongest = float((summary_by_variant.get(strongest_name) or {}).get("task_balanced_success_rate", 0.0))
    shuffled = float((summary_by_variant.get("task_shuffled_noise_prior") or {}).get("task_balanced_success_rate", 0.0))
    strongest_pair = paired_vs_full.get(strongest_name) or {}
    strongest_ci = strongest_pair.get("paired_bootstrap_ci", [0.0, 0.0])
    strongest_ci_lower = float(strongest_ci[0]) if len(strongest_ci) > 0 else 0.0
    strongest_ci_upper = float(strongest_ci[1]) if len(strongest_ci) > 1 else 0.0
    failure_reduction = float(strongest_pair.get("failure_rate_reduction", 0.0) or 0.0)

    full_beats_all = all(
        full > float((summary_by_variant.get(name) or {}).get("task_balanced_success_rate", 0.0))
        for name in baseline_names
    )
    if full_beats_all and (full - strongest >= 0.10 or (strongest_ci_lower > 0.0 and failure_reduction > 0.10)):
        return "STAGE_B_PROTOTYPE_GO"

    shuffled_pair = paired_vs_full.get("task_shuffled_noise_prior") or {}
    shuffled_ci = shuffled_pair.get("paired_bootstrap_ci", [0.0, 0.0])
    shuffled_ci_upper = float(shuffled_ci[1]) if len(shuffled_ci) > 1 else 0.0
    if full <= shuffled and shuffled_ci_upper < 0.10:
        return "STAGE_B_PERMANENT_KILL_ABLATION_EXPLAINS_METHOD"
    if full <= strongest and strongest_ci_upper < 0.10:
        return "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    if int(pairs_per_policy) >= 80:
        return "STAGE_B_UNRESOLVED_80_NON_GO_ARCHIVE"
    return "STAGE_B_UNRESOLVED_EXPAND_TO_80_REQUIRED"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
