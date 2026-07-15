"""Pure FAMR-VLA contracts, objective, and Stage 0 audit helpers."""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69"
TARGET_TASK_IDENTITIES = (
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
)
TRAIN_EPISODES = tuple(range(35))
VALIDATION_EPISODES = tuple(range(35, 45))
TEST_EPISODES = tuple(range(45, 50))
COARSE_GROUPS = ("vlm_expert", "action_flow", "state_projection")
FINE_GROUPS = (
    "vlm_layers_0_7",
    "vlm_layers_8_15",
    "action_input_output",
    "action_time_mlp",
    "state_projection",
)
ACTION_SCALE_FLOOR = 0.05
COEFFICIENT_STEPS = 500
COEFFICIENT_LEARNING_RATE = 0.05


def canonical_task_identity(value: str) -> str:
    """Normalize a LIBERO identifier or language string for exact identity checks."""

    text = value.strip().lower().replace("-", "_")
    text = re.sub(r"^(kitchen|living_room|study)_scene\d+_", "", text)
    text = re.sub(r"_demo(?:\.hdf5)?$", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def task_identity_audit(pretraining_tasks: Sequence[str], target_tasks: Sequence[str]) -> dict[str, Any]:
    pretraining = [canonical_task_identity(task) for task in pretraining_tasks]
    targets = [canonical_task_identity(task) for task in target_tasks]
    duplicates = [name for name, count in Counter(pretraining).items() if count > 1]
    intersection = sorted(set(pretraining) & set(targets))
    return {
        "pretraining_task_count": len(pretraining),
        "pretraining_unique_count": len(set(pretraining)),
        "target_task_count": len(targets),
        "target_unique_count": len(set(targets)),
        "pretraining_duplicates": duplicates,
        "normalized_intersection": intersection,
        "intersection_count": len(intersection),
    }


def episode_partitions() -> dict[str, tuple[int, ...]]:
    return {
        "train": TRAIN_EPISODES,
        "validation": VALIDATION_EPISODES,
        "test": TEST_EPISODES,
    }


def validate_episode_partitions(partitions: Mapping[str, Sequence[int]], available: Sequence[int]) -> dict[str, Any]:
    expected_names = {"train", "validation", "test"}
    if set(partitions) != expected_names:
        raise ValueError(f"episode partitions must be exactly {sorted(expected_names)}")
    available_set = {int(value) for value in available}
    normalized = {name: [int(value) for value in rows] for name, rows in partitions.items()}
    for name, rows in normalized.items():
        if len(rows) != len(set(rows)):
            raise ValueError(f"duplicate episode in {name}")
        missing = set(rows) - available_set
        if missing:
            raise ValueError(f"unavailable episodes in {name}: {sorted(missing)}")
    pairwise = {}
    names = sorted(normalized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(normalized[left]) & set(normalized[right])
            pairwise[f"{left}__{right}"] = sorted(overlap)
            if overlap:
                raise ValueError(f"episode overlap between {left} and {right}: {sorted(overlap)}")
    return {
        "counts": {name: len(rows) for name, rows in normalized.items()},
        "pairwise_overlap": pairwise,
        "all_available": True,
    }


def _parameter_family(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if "state_proj" in lowered:
        return "state_projection", "state_projection"
    if "action_time_mlp" in lowered:
        return "action_flow", "action_time_mlp"
    if "action_in_proj" in lowered or "action_out_proj" in lowered:
        return "action_flow", "action_input_output"
    if "lm_expert" in lowered and ("q_proj" in lowered or "v_proj" in lowered):
        match = re.search(r"(?:layers|layer)\.(\d+)\.", lowered)
        if match is None:
            raise ValueError(f"cannot identify VLM layer for {name}")
        layer = int(match.group(1))
        if 0 <= layer <= 7:
            fine = "vlm_layers_0_7"
        elif 8 <= layer <= 15:
            fine = "vlm_layers_8_15"
        else:
            raise ValueError(f"VLM layer outside frozen 0-15 range for {name}")
        return "vlm_expert", fine
    raise ValueError(f"trainable parameter is outside frozen FAMR targets: {name}")


def assign_parameter_groups(parameter_names: Sequence[str]) -> dict[str, dict[str, str]]:
    if not parameter_names:
        raise ValueError("FAMR requires at least one trainable parameter")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("duplicate trainable parameter name")
    coarse: dict[str, str] = {}
    fine: dict[str, str] = {}
    for name in sorted(parameter_names):
        coarse[name], fine[name] = _parameter_family(name)
    missing_coarse = set(COARSE_GROUPS) - set(coarse.values())
    missing_fine = set(FINE_GROUPS) - set(fine.values())
    if missing_coarse or missing_fine:
        raise ValueError(
            f"frozen parameter grouping is incomplete: coarse={sorted(missing_coarse)}, fine={sorted(missing_fine)}"
        )
    return {"coarse": coarse, "fine": fine}


def action_scales(actions: Any, floor: float = ACTION_SCALE_FLOOR) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"actions must have shape [N,7], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions contain nonfinite values")
    if floor <= 0.0:
        raise ValueError("action scale floor must be positive")
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    return np.maximum(iqr, float(floor))


def build_response_matrix(base_actions: Any, group_actions: Any) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float64)
    groups = np.asarray(group_actions, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 7:
        raise ValueError(f"base actions must have shape [N,7], got {base.shape}")
    if groups.ndim != 3 or groups.shape[0] != base.shape[0] or groups.shape[2] != 7:
        raise ValueError(f"group actions must have shape [N,M,7], got {groups.shape}")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(groups)):
        raise ValueError("response inputs contain nonfinite values")
    return np.transpose(groups - base[:, None, :], (0, 2, 1))


def predict_linear_actions(base_actions: Any, responses: Any, coefficients: Any) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    coefficient = np.asarray(coefficients, dtype=np.float64)
    if response.ndim != 3 or response.shape[:2] != base.shape or response.shape[2] != coefficient.size:
        raise ValueError("linear response shapes are inconsistent")
    return base + np.einsum("nam,m->na", response, coefficient)


def _huber(value: np.ndarray) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= 1.0, 0.5 * value * value, absolute - 0.5)


def objective_and_gradients(
    coefficients: Any,
    base_actions: Any,
    responses: Any,
    target_actions: Any,
    retention_responses: Any,
    scales: Any,
    retention_weight: float,
) -> dict[str, Any]:
    c = np.asarray(coefficients, dtype=np.float64)
    base = np.asarray(base_actions, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    target = np.asarray(target_actions, dtype=np.float64)
    retention = np.asarray(retention_responses, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if base.shape != target.shape or base.ndim != 2 or base.shape[1] != 7:
        raise ValueError("target and Base actions must share shape [N,7]")
    if response.shape != (base.shape[0], 7, c.size):
        raise ValueError("target response shape mismatch")
    if retention.ndim != 3 or retention.shape[1:] != (7, c.size):
        raise ValueError("retention response shape mismatch")
    if scale.shape != (7,) or np.any(scale <= 0.0):
        raise ValueError("scales must be positive shape [7]")
    if retention_weight < 0.0:
        raise ValueError("retention weight must be nonnegative")

    normalized_target = (predict_linear_actions(base, response, c) - target) / scale
    target_loss = float(np.mean(_huber(normalized_target)))
    target_derivative = np.clip(normalized_target, -1.0, 1.0) / normalized_target.size
    target_gradient = np.einsum("na,nam->m", target_derivative / scale, response)

    normalized_retention = np.einsum("ram,m->ra", retention, c) / scale
    retention_loss = float(np.mean(normalized_retention * normalized_retention))
    retention_gradient = np.einsum(
        "ra,ram->m", 2.0 * normalized_retention / normalized_retention.size / scale, retention
    )
    weighted_retention_gradient = float(retention_weight) * retention_gradient
    return {
        "total_loss": target_loss + float(retention_weight) * retention_loss,
        "target_loss": target_loss,
        "retention_loss": retention_loss,
        "weighted_retention_loss": float(retention_weight) * retention_loss,
        "gradient": target_gradient + weighted_retention_gradient,
        "target_gradient": target_gradient,
        "weighted_retention_gradient": weighted_retention_gradient,
    }


def solve_coefficients(
    base_actions: Any,
    responses: Any,
    target_actions: Any,
    retention_responses: Any,
    scales: Any,
    retention_weight: float,
    *,
    steps: int = COEFFICIENT_STEPS,
    learning_rate: float = COEFFICIENT_LEARNING_RATE,
) -> dict[str, Any]:
    response = np.asarray(responses, dtype=np.float64)
    if response.ndim != 3:
        raise ValueError("responses must have shape [N,7,M]")
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("projected Adam steps and learning rate must be positive")
    coefficients = np.full(response.shape[2], 0.5, dtype=np.float64)
    first = np.zeros_like(coefficients)
    second = np.zeros_like(coefficients)
    curve = []
    for step in range(1, steps + 1):
        audit = objective_and_gradients(
            coefficients,
            base_actions,
            response,
            target_actions,
            retention_responses,
            scales,
            retention_weight,
        )
        gradient = np.asarray(audit["gradient"], dtype=np.float64)
        if not np.all(np.isfinite(gradient)):
            raise ValueError("coefficient gradient contains nonfinite values")
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        coefficients = np.clip(
            coefficients - learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8), 0.0, 1.0
        )
        if step in {1, steps} or step % 50 == 0:
            curve.append({"step": step, "total_loss": float(audit["total_loss"])})
    final = objective_and_gradients(
        coefficients,
        base_actions,
        response,
        target_actions,
        retention_responses,
        scales,
        retention_weight,
    )
    return {
        "coefficients": coefficients,
        "initial_audit": objective_and_gradients(
            np.full(response.shape[2], 0.5),
            base_actions,
            response,
            target_actions,
            retention_responses,
            scales,
            retention_weight,
        ),
        "final": final,
        "curve": curve,
        "steps": steps,
        "learning_rate": learning_rate,
    }


def scale_lora_b(weight: Any, coefficient: float) -> Any:
    if not math.isfinite(float(coefficient)) or not 0.0 <= float(coefficient) <= 1.0:
        raise ValueError("LoRA coefficient must be finite and in [0,1]")
    if hasattr(weight, "mul"):
        return weight.mul(float(coefficient))
    return np.asarray(weight) * float(coefficient)


def response_fidelity(
    base_actions: Any, linear_actions: Any, direct_actions: Any, scales: Any
) -> dict[str, Any]:
    base = np.asarray(base_actions, dtype=np.float64)
    linear = np.asarray(linear_actions, dtype=np.float64)
    direct = np.asarray(direct_actions, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if base.shape != linear.shape or base.shape != direct.shape or base.ndim != 2 or base.shape[1] != 7:
        raise ValueError("fidelity actions must share shape [N,7]")
    error = linear - direct
    normalized_rmse = float(np.sqrt(np.mean((error / scale) ** 2)))
    direct_norm = np.linalg.norm(direct - base, axis=1)
    error_norm = np.linalg.norm(error, axis=1)
    relative = error_norm / np.maximum(direct_norm, 1e-12)
    linear_norm = np.linalg.norm(linear - base, axis=1)
    if np.std(linear_norm) > 1e-8 and np.std(direct_norm) > 1e-8:
        norm_correlation = float(np.corrcoef(linear_norm, direct_norm)[0, 1])
    else:
        norm_correlation = None
    return {
        "normalized_rmse": normalized_rmse,
        "median_relative_error": float(np.median(relative)),
        "relative_errors": relative,
        "per_dimension_rmse": np.sqrt(np.mean(error * error, axis=0)),
        "norm_correlation": norm_correlation,
    }


def ordering_agreement(predicted_scores: Sequence[float], direct_scores: Sequence[float]) -> float:
    predicted = np.asarray(predicted_scores, dtype=np.float64)
    direct = np.asarray(direct_scores, dtype=np.float64)
    if predicted.shape != direct.shape or predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("ordering scores must be equal one-dimensional arrays with at least two entries")
    agreements = []
    for left in range(predicted.size):
        for right in range(left + 1, predicted.size):
            predicted_sign = np.sign(predicted[left] - predicted[right])
            direct_sign = np.sign(direct[left] - direct[right])
            agreements.append(predicted_sign == direct_sign)
    return float(np.mean(agreements))


def practical_action_threshold(repeated_base_l2: Sequence[float]) -> float:
    values = np.asarray(repeated_base_l2, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("repeated Base distances must be finite, nonnegative, and nonempty")
    return max(1e-4, float(np.percentile(values, 95)))


def action_validity(ours: Any, base: Any, *, simulator_accepted: bool = True) -> dict[str, Any]:
    candidate = np.asarray(ours, dtype=np.float64)
    reference = np.asarray(base, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim < 2 or candidate.shape[-1] != 7:
        raise ValueError("candidate and Base actions must share shape [...,7]")
    finite_fraction = float(np.mean(np.isfinite(candidate)))
    candidate_abs_max = float(np.nanmax(np.abs(candidate)))
    base_abs_max = float(np.nanmax(np.abs(reference)))
    candidate_exceedance = np.maximum(np.abs(candidate) - 1.0, 0.0)
    base_exceedance = np.maximum(np.abs(reference) - 1.0, 0.0)
    outside_fraction = float(np.mean(candidate_exceedance > 0.0))
    base_outside_fraction = float(np.mean(base_exceedance > 0.0))
    p99_exceedance = float(np.nanpercentile(candidate_exceedance, 99))
    base_p99_exceedance = float(np.nanpercentile(base_exceedance, 99))
    absolute_limit = max(1.25, base_abs_max + 0.05)
    passed = bool(
        finite_fraction == 1.0
        and candidate_abs_max <= absolute_limit
        and outside_fraction <= base_outside_fraction + 0.01
        and p99_exceedance <= base_p99_exceedance + 0.02
        and simulator_accepted
    )
    return {
        "passed": passed,
        "finite_fraction": finite_fraction,
        "candidate_abs_max": candidate_abs_max,
        "base_abs_max": base_abs_max,
        "absolute_limit": absolute_limit,
        "outside_fraction": outside_fraction,
        "base_outside_fraction": base_outside_fraction,
        "p99_exceedance": p99_exceedance,
        "base_p99_exceedance": base_p99_exceedance,
        "simulator_accepted": bool(simulator_accepted),
    }


def result_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        return (
            str(row["policy"]),
            str(row["suite"]),
            str(row["task"]),
            int(row["reset_identity"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed result key: {row}") from exc


def validate_result_manifest(
    manifest_rows: Sequence[Mapping[str, Any]], result_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = [result_key(row) for row in manifest_rows]
    observed = [result_key(row) for row in result_rows]
    duplicate_manifest = len(expected) - len(set(expected))
    duplicate_results = len(observed) - len(set(observed))
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    exceptions = sum(bool(row.get("exception")) for row in result_rows)
    return {
        "passed": duplicate_manifest == 0 and duplicate_results == 0 and not missing and not extra and exceptions == 0,
        "planned_count": len(expected),
        "completed_count": len(observed),
        "duplicate_manifest_key_count": duplicate_manifest,
        "duplicate_result_key_count": duplicate_results,
        "missing_keys": missing,
        "extra_keys": extra,
        "exception_count": exceptions,
    }


def missing_manifest_rows(
    manifest_rows: Sequence[Mapping[str, Any]], completed_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    completed_keys = [result_key(row) for row in completed_rows]
    if len(completed_keys) != len(set(completed_keys)):
        raise ValueError("completed rows contain duplicate keys")
    completed = set(completed_keys)
    return [dict(row) for row in manifest_rows if result_key(row) not in completed]


def validate_partial_payload(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("partial result is not valid JSON") from exc
    else:
        parsed = dict(payload)
    if not isinstance(parsed, dict):
        raise ValueError("partial result must be a JSON object")
    completed = parsed.get("completed_count", parsed.get("completed_pair_count"))
    planned = parsed.get("planned_count", parsed.get("planned_pair_count"))
    exceptions = parsed.get("exception_count")
    if any(not isinstance(value, int) for value in (completed, planned, exceptions)):
        raise ValueError("partial result requires integer completed, planned, and exception counts")
    if completed < 0 or planned < 0 or completed > planned or exceptions < 0:
        raise ValueError("partial result counts are inconsistent")
    return {
        "parsed": True,
        "completed_count": completed,
        "planned_count": planned,
        "exception_count": exceptions,
    }


def resource_overlap(
    started_unix: float, finished_unix: float, intervals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if finished_unix < started_unix:
        raise ValueError("resource interval ends before it starts")
    overlaps = []
    for interval in intervals:
        left = interval.get("started_unix", interval.get("start_unix"))
        right = interval.get("finished_unix", interval.get("end_unix"))
        if left is None or right is None:
            continue
        overlap_start = max(float(started_unix), float(left))
        overlap_end = min(float(finished_unix), float(right))
        if overlap_end >= overlap_start:
            overlaps.append({**dict(interval), "overlap_seconds": max(0.0, overlap_end - overlap_start)})
    return {
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "timing_resource_evidence_eligible": len(overlaps) == 0,
        "closed_loop_success_requires_row_level_checks": len(overlaps) > 0,
    }


def classify_stage0(summary: Mapping[str, Any]) -> str:
    if bool(summary.get("essential_source_unavailable")) or int(summary.get("target_overlap_count", 0)) > 0:
        return "FAMR_FATAL_PREIMPLEMENTATION"
    implementation_checks = (
        "preflight_passed",
        "data_semantics_passed",
        "split_integrity_passed",
        "identity_passed",
        "target_modules_passed",
        "gradient_health_passed",
        "checkpoint_reload_passed",
        "group_assignment_passed",
        "scaling_identity_passed",
        "base_unchanged",
        "memory_passed",
        "confirmatory_sealed",
    )
    if not all(bool(summary.get(name)) for name in implementation_checks):
        return "FAMR_IMPLEMENTATION_OR_DATA_FAILURE"
    if not bool(summary.get("subset_fit_passed")):
        if not bool(summary.get("capacity_check_used")):
            return "FAMR_UNDERPOWERED_ONE_CHECK_ALLOWED"
        return "FAMR_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT"
    return "FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED"
