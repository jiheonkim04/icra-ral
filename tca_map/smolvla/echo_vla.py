"""ECHO-VLA lightweight protocol and scoring utilities.

This module deliberately separates privileged training labels from deployment
inputs.  The pure utilities are used by tests and by the bounded first
prototype runner; heavy SmolVLA/LIBERO imports stay in the runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "echo_vla_effect_schema_v1"

PHASES = (
    "approach",
    "grasp_contact",
    "lift",
    "transport",
    "placement",
    "release",
)

EFFECT_COMPONENTS = (
    "eef_delta_norm",
    "target_distance_delta",
    "contact_transition",
    "gripper_transition",
    "object_retained",
    "object_lift_delta",
    "object_goal_delta",
    "placement_alignment",
    "release_stability",
)

PRIVILEGED_EFFECT_COMPONENTS = {
    "target_distance_delta",
    "contact_transition",
    "object_retained",
    "object_lift_delta",
    "object_goal_delta",
    "placement_alignment",
    "release_stability",
}

DEPLOYMENT_FORBIDDEN_KEYS = {
    "sim_state",
    "simulator_state",
    "object_pose",
    "object_poses",
    "ground_truth_pose",
    "bddl_success",
    "success",
    "reward",
    "future_effect",
    "realized_effect",
    "oracle_effect",
    "privileged_effect",
}

PHASE_REQUIRED_EFFECTS: dict[str, dict[str, float]] = {
    "approach": {
        "eef_delta_norm": 0.20,
        "target_distance_delta": 1.00,
        "contact_transition": -0.20,
        "object_retained": 0.00,
        "object_lift_delta": 0.00,
        "object_goal_delta": 0.20,
        "placement_alignment": 0.00,
        "release_stability": 0.00,
    },
    "grasp_contact": {
        "target_distance_delta": 0.40,
        "contact_transition": 1.00,
        "gripper_transition": 0.60,
        "object_retained": 0.80,
    },
    "lift": {
        "object_retained": 1.00,
        "object_lift_delta": 1.00,
        "release_stability": -0.30,
    },
    "transport": {
        "object_retained": 0.80,
        "object_goal_delta": 1.00,
        "placement_alignment": 0.30,
    },
    "placement": {
        "object_goal_delta": 0.70,
        "placement_alignment": 1.00,
        "object_retained": 0.20,
    },
    "release": {
        "gripper_transition": -0.50,
        "release_stability": 1.00,
        "placement_alignment": 0.70,
        "object_retained": -0.20,
    },
}


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_hash(value: Any) -> str:
    """Return a stable hash for arrays, mappings, and scalar metadata."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        return digest.hexdigest()
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def round_float(value: float | int | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def effect_template(fill: float = 0.0) -> dict[str, float]:
    return {name: float(fill) for name in EFFECT_COMPONENTS}


def normalize_effect(effect: Mapping[str, Any]) -> dict[str, float]:
    normalized = effect_template()
    for key, value in effect.items():
        if key in normalized:
            normalized[key] = float(value)
    return normalized


def required_effect_for_phase(phase: str) -> dict[str, float]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    required = effect_template()
    required.update(PHASE_REQUIRED_EFFECTS[phase])
    return required


def compatibility_score(effect: Mapping[str, Any], phase: str, risk_weight: float = 0.15) -> float:
    actual = normalize_effect(effect)
    required = required_effect_for_phase(phase)
    score = 0.0
    risk = 0.0
    for key in EFFECT_COMPONENTS:
        score += required[key] * actual[key]
    if actual["object_retained"] < -0.1:
        risk += abs(actual["object_retained"])
    if actual["release_stability"] < -0.1 and phase != "release":
        risk += abs(actual["release_stability"])
    return float(score - risk_weight * risk)


def infer_phase_from_effect_history(history: Sequence[Mapping[str, Any]]) -> str:
    """A deterministic non-privileged-style fallback phase estimator.

    This heuristic is intentionally simple and serves as a reviewer-killer
    baseline as well as a safe default for tests.  Real prototype training can
    replace it with a learned head while preserving the same phase names.
    """
    if not history:
        return "approach"
    latest = normalize_effect(history[-1])
    if latest["release_stability"] > 0.5:
        return "release"
    if latest["placement_alignment"] > 0.5:
        return "release"
    if latest["object_goal_delta"] > 0.4:
        return "placement"
    if latest["object_lift_delta"] > 0.2 or latest["object_retained"] > 0.5:
        return "transport"
    if latest["contact_transition"] > 0.4:
        return "lift"
    if latest["target_distance_delta"] > 0.2:
        return "grasp_contact"
    return "approach"


@dataclass(frozen=True)
class CandidateRecord:
    group_id: str
    candidate_index: int
    start_state_hash: str
    start_observation_hash: str
    action_chunk: list[list[float]]
    horizon: int
    phase: str
    realized_effect: dict[str, float]
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    source: str = "unknown"

    def compatibility(self) -> float:
        return compatibility_score(self.realized_effect, self.phase)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def validate_counterfactual_group(candidates: Sequence[CandidateRecord]) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("counterfactual intervention group needs at least two candidates")
    group_ids = {item.group_id for item in candidates}
    state_hashes = {item.start_state_hash for item in candidates}
    obs_hashes = {item.start_observation_hash for item in candidates}
    horizons = {int(item.horizon) for item in candidates}
    phases = {item.phase for item in candidates}
    valid = len(group_ids) == 1 and len(state_hashes) == 1 and len(obs_hashes) == 1 and len(horizons) == 1 and len(phases) == 1
    return {
        "valid": bool(valid),
        "candidate_count": len(candidates),
        "group_ids": sorted(group_ids),
        "start_state_hashes": sorted(state_hashes),
        "start_observation_hashes": sorted(obs_hashes),
        "horizons": sorted(horizons),
        "phases": sorted(phases),
    }


def assert_no_privileged_deployment_inputs(payload: Mapping[str, Any]) -> None:
    present = sorted(key for key in payload if key in DEPLOYMENT_FORBIDDEN_KEYS)
    if present:
        raise ValueError(f"privileged deployment inputs are forbidden: {present}")


def select_oracle_candidate(candidates: Sequence[CandidateRecord]) -> CandidateRecord:
    if not candidates:
        raise ValueError("no candidates")
    return max(candidates, key=lambda item: (item.compatibility(), item.success, -item.candidate_index))


def select_random_candidate(candidates: Sequence[CandidateRecord], seed: int) -> CandidateRecord:
    if not candidates:
        raise ValueError("no candidates")
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def select_action_norm_candidate(candidates: Sequence[CandidateRecord]) -> CandidateRecord:
    if not candidates:
        raise ValueError("no candidates")

    def norm(item: CandidateRecord) -> float:
        arr = np.asarray(item.action_chunk, dtype=np.float64)
        return float(np.linalg.norm(arr))

    return min(candidates, key=lambda item: (norm(item), item.candidate_index))


def select_phase_predicate_heuristic(candidates: Sequence[CandidateRecord]) -> CandidateRecord:
    if not candidates:
        raise ValueError("no candidates")
    # This uses realized effects only for the oracle headroom diagnostic.  The
    # learned deployment selector must use predicted effects instead.
    return max(candidates, key=lambda item: (compatibility_score(item.realized_effect, item.phase), -item.candidate_index))


def candidate_headroom_metrics(groups: Sequence[Sequence[CandidateRecord]]) -> dict[str, Any]:
    if not groups:
        return {
            "group_count": 0,
            "default_success_rate": 0.0,
            "oracle_success_rate": 0.0,
            "oracle_improvement_pp": 0.0,
            "default_failure_group_count": 0,
            "default_failure_recoverable_rate": 0.0,
            "passes_headroom_gate": False,
        }
    validated = [validate_counterfactual_group(group) for group in groups]
    invalid = [item for item in validated if not item["valid"]]
    if invalid:
        raise ValueError(f"invalid counterfactual groups: {invalid[:3]}")

    defaults: list[CandidateRecord] = []
    oracles: list[CandidateRecord] = []
    recoverable_default_failures = 0
    default_failures = 0
    material_better = 0
    for group in groups:
        ordered = sorted(group, key=lambda item: item.candidate_index)
        default = ordered[0]
        oracle = select_oracle_candidate(ordered)
        defaults.append(default)
        oracles.append(oracle)
        if not default.success:
            default_failures += 1
            if oracle.success or oracle.compatibility() > default.compatibility() + 0.25:
                recoverable_default_failures += 1
        if oracle.compatibility() > default.compatibility() + 0.25:
            material_better += 1

    default_success = float(np.mean([item.success for item in defaults]))
    oracle_success = float(np.mean([item.success for item in oracles]))
    improvement_pp = 100.0 * (oracle_success - default_success)
    recoverable_rate = 0.0 if default_failures == 0 else recoverable_default_failures / default_failures
    passes = improvement_pp >= 10.0 and recoverable_rate >= 0.15
    return {
        "group_count": len(groups),
        "candidate_count_total": int(sum(len(group) for group in groups)),
        "default_success_rate": round_float(default_success, 6),
        "oracle_success_rate": round_float(oracle_success, 6),
        "oracle_improvement_pp": round_float(improvement_pp, 3),
        "default_failure_group_count": int(default_failures),
        "default_failure_recoverable_count": int(recoverable_default_failures),
        "default_failure_recoverable_rate": round_float(recoverable_rate, 6),
        "materially_better_group_count": int(material_better),
        "passes_headroom_gate": bool(passes),
        "hard_kill_reason": None
        if passes
        else "oracle improvement <10pp or fewer than 15% of default-failure states contain a successful/materially better candidate",
    }


def pairwise_ranking_pairs(candidates: Sequence[CandidateRecord], margin: float = 0.05) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    scores = [(item.candidate_index, item.compatibility()) for item in candidates]
    for left_index, left_score in scores:
        for right_index, right_score in scores:
            if left_score > right_score + margin:
                pairs.append((left_index, right_index, float(left_score - right_score)))
    return pairs


def build_candidate_record(
    *,
    group_id: str,
    candidate_index: int,
    start_state: Any,
    start_observation: Any,
    action_chunk: Sequence[Sequence[float]] | np.ndarray,
    horizon: int,
    phase: str,
    realized_effect: Mapping[str, Any],
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
    source: str = "unknown",
) -> CandidateRecord:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    chunk = np.asarray(action_chunk, dtype=np.float64)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    if chunk.ndim != 2:
        raise ValueError(f"action_chunk must be 2D after flattening horizon, got shape {chunk.shape}")
    if int(horizon) != int(chunk.shape[0]):
        raise ValueError(f"horizon {horizon} does not match action chunk length {chunk.shape[0]}")
    start_state_hash = str(start_state) if isinstance(start_state, str) and len(start_state) == 64 else stable_hash(start_state)
    start_observation_hash = (
        str(start_observation)
        if isinstance(start_observation, str) and len(start_observation) == 64
        else stable_hash(start_observation)
    )
    return CandidateRecord(
        group_id=str(group_id),
        candidate_index=int(candidate_index),
        start_state_hash=start_state_hash,
        start_observation_hash=start_observation_hash,
        action_chunk=[[float(value) for value in row] for row in chunk.tolist()],
        horizon=int(horizon),
        phase=phase,
        realized_effect=normalize_effect(realized_effect),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        source=str(source),
    )


def serialize_groups(groups: Sequence[Sequence[CandidateRecord]]) -> list[list[dict[str, Any]]]:
    return [[candidate.to_json() for candidate in group] for group in groups]


def deserialize_groups(payload: Iterable[Iterable[Mapping[str, Any]]]) -> list[list[CandidateRecord]]:
    groups: list[list[CandidateRecord]] = []
    for group in payload:
        records = []
        for item in group:
            records.append(
                CandidateRecord(
                    group_id=str(item["group_id"]),
                    candidate_index=int(item["candidate_index"]),
                    start_state_hash=str(item["start_state_hash"]),
                    start_observation_hash=str(item["start_observation_hash"]),
                    action_chunk=[[float(value) for value in row] for row in item["action_chunk"]],
                    horizon=int(item["horizon"]),
                    phase=str(item["phase"]),
                    realized_effect=normalize_effect(item["realized_effect"]),
                    success=bool(item.get("success", False)),
                    terminated=bool(item.get("terminated", False)),
                    truncated=bool(item.get("truncated", False)),
                    source=str(item.get("source", "unknown")),
                )
            )
        groups.append(records)
    return groups
