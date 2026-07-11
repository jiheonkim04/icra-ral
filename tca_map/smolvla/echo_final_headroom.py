"""Pure utilities for the final ECHO-VLA candidate-headroom gate.

The simulator runner owns heavy SmolVLA/LIBERO imports.  This module keeps the
decision-critical math testable: candidate diversity, downstream oracle
headroom, and final decision precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FINAL_DECISIONS = {
    "ECHO_POLICY_CANDIDATE_HEADROOM_CONFIRMED",
    "ECHO_POLICY_CANDIDATES_IMPOVERISHED",
    "ECHO_ONLY_STRUCTURED_CANDIDATES_HAVE_HEADROOM",
    "NO_ECHO_HEADROOM_CONFIRMED",
    "ECHO_GATE_MEASUREMENT_INVALID",
    "ECHO_RESULT_INCONCLUSIVE",
}

NEAR_IDENTICAL_FULL_L2 = 1e-3
NEAR_IDENTICAL_COMPONENT_L2 = 1e-4
MEAN_PAIRWISE_MEANINGFUL_L2 = 1e-2
IMPOVERISHED_STATE_FRACTION = 2.0 / 3.0
MIN_ORACLE_IMPROVEMENT_PP = 10.0
MIN_RECOVERABLE_DEFAULT_FAILURE_RATE = 0.15
MIN_TASKS_WITH_RECOVERY = 2


@dataclass(frozen=True)
class DiversityThresholds:
    near_identical_full_l2: float = NEAR_IDENTICAL_FULL_L2
    near_identical_component_l2: float = NEAR_IDENTICAL_COMPONENT_L2
    mean_pairwise_meaningful_l2: float = MEAN_PAIRWISE_MEANINGFUL_L2
    impoverished_state_fraction: float = IMPOVERISHED_STATE_FRACTION


def _round(value: float | int | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _as_chunk(candidate: Mapping[str, Any], key: str = "postprocessed_action_chunk") -> np.ndarray:
    chunk = np.asarray(candidate.get(key), dtype=np.float64)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    if chunk.ndim != 2:
        raise ValueError(f"{key} must be a 2D action chunk, got shape {chunk.shape}")
    return chunk


def pairwise_action_differences(candidates: Sequence[Mapping[str, Any]], *, key: str = "postprocessed_action_chunk") -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_pos, left in enumerate(candidates):
        left_chunk = _as_chunk(left, key=key)
        for right_pos in range(left_pos + 1, len(candidates)):
            right = candidates[right_pos]
            right_chunk = _as_chunk(right, key=key)
            horizon = min(left_chunk.shape[0], right_chunk.shape[0])
            width = min(left_chunk.shape[1], right_chunk.shape[1])
            delta = left_chunk[:horizon, :width] - right_chunk[:horizon, :width]
            trans = delta[:, : min(3, width)]
            rot = delta[:, 3: min(6, width)] if width > 3 else np.zeros((horizon, 0), dtype=np.float64)
            grip = delta[:, -1:] if width >= 1 else np.zeros((horizon, 0), dtype=np.float64)
            pairs.append(
                {
                    "left_index": int(left.get("candidate_index", left_pos)),
                    "right_index": int(right.get("candidate_index", right_pos)),
                    "full_l2": float(np.linalg.norm(delta)),
                    "translation_l2": float(np.linalg.norm(trans)),
                    "rotation_l2": float(np.linalg.norm(rot)),
                    "gripper_l2": float(np.linalg.norm(grip)),
                }
            )
    return pairs


def _effective_distinct_count(candidates: Sequence[Mapping[str, Any]], thresholds: DiversityThresholds) -> int:
    representatives: list[np.ndarray] = []
    for candidate in candidates:
        chunk = _as_chunk(candidate)
        is_new = True
        for existing in representatives:
            horizon = min(existing.shape[0], chunk.shape[0])
            width = min(existing.shape[1], chunk.shape[1])
            if float(np.linalg.norm(existing[:horizon, :width] - chunk[:horizon, :width])) <= thresholds.near_identical_full_l2:
                is_new = False
                break
        if is_new:
            representatives.append(chunk)
    return len(representatives)


def summarize_candidate_diversity(
    candidates: Sequence[Mapping[str, Any]],
    *,
    thresholds: DiversityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DiversityThresholds()
    pairs = pairwise_action_differences(candidates)
    if not pairs:
        return {
            "candidate_count": len(candidates),
            "pair_count": 0,
            "mean_pairwise_action_l2": 0.0,
            "max_pairwise_action_l2": 0.0,
            "mean_translation_l2": 0.0,
            "mean_rotation_l2": 0.0,
            "mean_gripper_l2": 0.0,
            "effective_distinct_candidates": len(candidates),
            "exactly_identical_pair_fraction": 0.0,
            "nearly_identical_pair_fraction": 0.0,
            "state_candidates_impoverished": len(candidates) < 2,
            "pairwise_action_distances": pairs,
        }
    full = np.asarray([pair["full_l2"] for pair in pairs], dtype=np.float64)
    translation = np.asarray([pair["translation_l2"] for pair in pairs], dtype=np.float64)
    rotation = np.asarray([pair["rotation_l2"] for pair in pairs], dtype=np.float64)
    gripper = np.asarray([pair["gripper_l2"] for pair in pairs], dtype=np.float64)
    exact = full <= 1e-9
    near = (full <= thresholds.near_identical_full_l2) | (
        (translation <= thresholds.near_identical_component_l2)
        & (rotation <= thresholds.near_identical_component_l2)
        & (gripper <= thresholds.near_identical_component_l2)
    )
    effective = _effective_distinct_count(candidates, thresholds)
    impoverished = bool(
        effective < 2
        or float(np.mean(full)) < thresholds.mean_pairwise_meaningful_l2
        or float(np.mean(near)) >= 0.75
    )
    return {
        "candidate_count": len(candidates),
        "pair_count": len(pairs),
        "mean_pairwise_action_l2": _round(float(np.mean(full)), 9),
        "max_pairwise_action_l2": _round(float(np.max(full)), 9),
        "mean_translation_l2": _round(float(np.mean(translation)), 9),
        "mean_rotation_l2": _round(float(np.mean(rotation)), 9),
        "mean_gripper_l2": _round(float(np.mean(gripper)), 9),
        "effective_distinct_candidates": int(effective),
        "exactly_identical_pair_fraction": _round(float(np.mean(exact)), 6),
        "nearly_identical_pair_fraction": _round(float(np.mean(near)), 6),
        "state_candidates_impoverished": impoverished,
        "pairwise_action_distances": pairs,
    }


def summarize_diversity_across_states(state_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(state_summaries)
    impoverished = [item for item in state_summaries if item.get("state_candidates_impoverished")]
    fraction = 0.0 if count == 0 else len(impoverished) / count
    return {
        "state_count": int(count),
        "impoverished_state_count": int(len(impoverished)),
        "impoverished_state_fraction": _round(fraction, 6),
        "policy_candidates_impoverished": bool(count > 0 and fraction >= IMPOVERISHED_STATE_FRACTION),
        "predeclared_rule": (
            "official policy candidates are impoverished when at least two-thirds of states have "
            "effective distinct candidates <2, mean pairwise action L2 <0.01, or >=75% nearly-identical pairs"
        ),
    }


def downstream_headroom_metrics(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defaults = []
    oracles = []
    randoms = []
    local_effect_oracles = []
    default_failures = 0
    recoverable = 0
    tasks_with_recovery: set[str] = set()
    phases_with_recovery: set[str] = set()
    recovered_group_ids: list[str] = []
    for group in groups:
        candidates = sorted(group.get("candidates", []), key=lambda item: int(item.get("candidate_index", 0)))
        if not candidates:
            continue
        default = candidates[0]
        oracle = max(candidates, key=lambda item: (bool(item.get("downstream_success")), float(item.get("max_reward", 0.0)), -int(item.get("candidate_index", 0))))
        local = max(candidates, key=lambda item: (float(item.get("effect_compatibility", 0.0)), bool(item.get("downstream_success")), -int(item.get("candidate_index", 0))))
        random_index = int(group.get("random_candidate_index", 0))
        random_candidate = next((item for item in candidates if int(item.get("candidate_index", -1)) == random_index), candidates[0])
        defaults.append(default)
        oracles.append(oracle)
        randoms.append(random_candidate)
        local_effect_oracles.append(local)
        if not bool(default.get("downstream_success")):
            default_failures += 1
            if any(bool(item.get("downstream_success")) for item in candidates[1:]):
                recoverable += 1
                task_key = str(group.get("task_key"))
                phase = str(group.get("phase"))
                tasks_with_recovery.add(task_key)
                phases_with_recovery.add(phase)
                recovered_group_ids.append(str(group.get("group_id")))

    group_count = len(defaults)
    default_success = 0.0 if group_count == 0 else float(np.mean([bool(item.get("downstream_success")) for item in defaults]))
    oracle_success = 0.0 if group_count == 0 else float(np.mean([bool(item.get("downstream_success")) for item in oracles]))
    random_success = 0.0 if group_count == 0 else float(np.mean([bool(item.get("downstream_success")) for item in randoms]))
    local_success = 0.0 if group_count == 0 else float(np.mean([bool(item.get("downstream_success")) for item in local_effect_oracles]))
    improvement_pp = 100.0 * (oracle_success - default_success)
    recoverable_rate = 0.0 if default_failures == 0 else recoverable / default_failures
    passes_thresholds = improvement_pp >= MIN_ORACLE_IMPROVEMENT_PP and recoverable_rate >= MIN_RECOVERABLE_DEFAULT_FAILURE_RATE
    spans_tasks = len(tasks_with_recovery) >= MIN_TASKS_WITH_RECOVERY
    not_one_state = recoverable > 1 and len(phases_with_recovery) > 1
    return {
        "group_count": int(group_count),
        "candidate_count_total": int(sum(len(group.get("candidates", [])) for group in groups)),
        "default_success_rate": _round(default_success, 6),
        "random_success_rate": _round(random_success, 6),
        "local_effect_oracle_success_rate": _round(local_success, 6),
        "final_task_success_oracle_rate": _round(oracle_success, 6),
        "oracle_improvement_pp": _round(improvement_pp, 3),
        "default_failure_group_count": int(default_failures),
        "recoverable_default_failure_count": int(recoverable),
        "recoverable_default_failure_rate": _round(recoverable_rate, 6),
        "tasks_with_recovery": sorted(tasks_with_recovery),
        "tasks_with_recovery_count": int(len(tasks_with_recovery)),
        "phases_with_recovery": sorted(phases_with_recovery),
        "recovered_group_ids": recovered_group_ids,
        "passes_original_thresholds": bool(passes_thresholds),
        "headroom_spans_multiple_tasks": bool(spans_tasks),
        "not_solely_one_phase_or_state": bool(not_one_state),
        "passes_final_gate": bool(passes_thresholds and spans_tasks and not_one_state),
        "threshold_rule": (
            "non-relaxed original hard gate: oracle improvement >=10pp and recoverable default-failure rate >=15%, "
            "plus recovery across at least two tasks and not solely one phase/state"
        ),
    }


def choose_final_decision(
    *,
    measurement_valid: bool,
    official_diversity: Mapping[str, Any],
    official_metrics: Mapping[str, Any],
    structured_metrics: Mapping[str, Any],
    outcome_variance_high: bool = False,
) -> str:
    if not measurement_valid:
        return "ECHO_GATE_MEASUREMENT_INVALID"
    if outcome_variance_high:
        return "ECHO_RESULT_INCONCLUSIVE"
    if bool(official_diversity.get("policy_candidates_impoverished")):
        return "ECHO_POLICY_CANDIDATES_IMPOVERISHED"
    if bool(official_metrics.get("passes_final_gate")):
        return "ECHO_POLICY_CANDIDATE_HEADROOM_CONFIRMED"
    if bool(structured_metrics.get("passes_final_gate")):
        return "ECHO_ONLY_STRUCTURED_CANDIDATES_HAVE_HEADROOM"
    return "NO_ECHO_HEADROOM_CONFIRMED"

