"""Frozen helpers for the Epoch 7 persistent-completion problem gate."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("schema_version") != "epoch7.persistent_completion.problem_protocol.v1":
        errors.append("unexpected schema_version")
    if protocol.get("status") != "FROZEN_BEFORE_TASK_PERSISTENCE_OUTCOMES":
        errors.append("protocol is not frozen")
    if protocol.get("policy_loaded_or_queried") is not False:
        errors.append("policy_loaded_or_queried must be false")
    if protocol.get("ours_authorized") is not False:
        errors.append("ours_authorized must be false")

    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10:
        errors.append("exactly ten tasks are required")
    else:
        ids = [task.get("task_id") for task in tasks]
        if ids != list(range(10)):
            errors.append("task ids must be ordered 0..9")
        for task in tasks:
            for key in ("task_name", "mechanism", "hdf5", "hdf5_sha256"):
                if not task.get(key):
                    errors.append(f"task {task.get('task_id')} missing {key}")

    hold = protocol.get("hold_contract", {})
    if hold.get("steps") != 30:
        errors.append("primary hold length must remain 30")
    if hold.get("pose_delta") != [0.0] * 6:
        errors.append("primary pose delta must remain zero")

    gates = protocol.get("gates", {})
    if gates.get("execution", {}).get("completed_tasks_min") != 10:
        errors.append("execution gate must require all ten tasks")
    if gates.get("problem", {}).get("immediate_persistence_failure_tasks_min") != 3:
        errors.append("problem gate task minimum changed")
    if gates.get("headroom", {}).get("suffix_recovered_tasks_min") != 2:
        errors.append("headroom gate task minimum changed")
    return errors


def numeric_demo_key(value: str) -> tuple[int, str]:
    try:
        return int(str(value).rsplit("_", 1)[-1]), str(value)
    except ValueError:
        return 10**9, str(value)


def neutral_action(last_action: Sequence[float]) -> np.ndarray:
    action = np.asarray(last_action, dtype=np.float64).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("last_action must be a finite seven-dimensional vector")
    result = np.zeros(7, dtype=np.float64)
    result[6] = 1.0 if float(action[6]) >= 0.0 else -1.0
    return result


def persistence_summary(trace: Sequence[bool], expected_steps: int) -> dict[str, Any]:
    values = [bool(value) for value in trace]
    first_failure = next((index for index, value in enumerate(values) if not value), None)
    return {
        "hold_steps_completed": len(values),
        "hold_success_count": int(sum(values)),
        "hold_success_fraction": float(sum(values) / expected_steps) if expected_steps else 0.0,
        "persistent_success": len(values) == expected_steps and all(values),
        "first_hold_failure_index": first_failure,
        "final_hold_success": values[-1] if values else None,
    }


def collapsed_mechanism(mechanism: str) -> str:
    value = str(mechanism)
    if "articulation" in value:
        return "articulation"
    if value == "planar_push":
        return "planar_push"
    if "containment" in value or "insertion" in value:
        return "containment_or_insertion"
    if "placement" in value:
        return "placement"
    return value


def predicate_family(task_name: str) -> str:
    name = str(task_name)
    if name.startswith("open_") and "put_" in name:
        return "composite_open_and_in"
    if name.startswith("open_"):
        return "open"
    if name.startswith("turn_on_"):
        return "turn_on"
    if name.startswith("push_"):
        return "spatial_push"
    if "_in_the_" in name or "_inside" in name:
        return "in"
    if "_on_the_" in name or "_on_top_of_" in name:
        return "on"
    return "other"


def branch_signature(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "native_success": bool(branch.get("native_success", False)),
        "first_success_step": branch.get("first_success_step"),
        "hold_success_trace": [bool(value) for value in branch.get("hold_success_trace", [])],
        "persistent_success": bool(branch.get("persistent_success", False)),
        "final_hold_success": branch.get("final_hold_success"),
        "error": branch.get("error"),
    }


def adjudicate_result(protocol: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    execution_gate = protocol["gates"]["execution"]
    coverage_gate = protocol["gates"]["coverage"]
    problem_gate = protocol["gates"]["problem"]
    headroom_gate = protocol["gates"]["headroom"]

    completed = [row for row in tasks if row.get("completed")]
    exceptions = sum(
        1
        for row in tasks
        if row.get("error")
        or any(branch.get("error") for branch in row.get("branches", {}).values())
    )
    finite = all(bool(row.get("finite_actions", False)) for row in tasks)
    cold_rows = [row for row in tasks if row.get("cold_repeat_required")]
    cold_repeat_pass = len(cold_rows) == len(execution_gate["cold_repeat_task_ids"]) and all(
        row.get("cold_repeat_match") is True for row in cold_rows
    )
    execution_pass = (
        len(completed) >= int(execution_gate["completed_tasks_min"])
        and exceptions <= int(execution_gate["exception_count_max"])
        and finite
        and cold_repeat_pass
    )

    native_rows = [
        row
        for row in tasks
        if row.get("branches", {}).get("immediate_neutral_hold", {}).get("native_success")
    ]
    native_mechanisms = {collapsed_mechanism(row["mechanism"]) for row in native_rows}
    required_mechanisms = set(coverage_gate["required_mechanisms"])
    coverage_pass = (
        len(native_rows) >= int(coverage_gate["native_success_tasks_min"])
        and required_mechanisms.issubset(native_mechanisms)
    )

    disagreements = [
        row
        for row in native_rows
        if not row.get("branches", {}).get("immediate_neutral_hold", {}).get("persistent_success", False)
    ]
    disagreement_mechanisms = {collapsed_mechanism(row["mechanism"]) for row in disagreements}
    disagreement_fraction = len(disagreements) / len(native_rows) if native_rows else 0.0
    family_counts: dict[str, int] = {}
    for row in disagreements:
        family = predicate_family(row["task_name"])
        family_counts[family] = family_counts.get(family, 0) + 1
    maximum_family_fraction = max(family_counts.values(), default=0) / len(disagreements) if disagreements else 0.0
    problem_pass = (
        len(disagreements) >= int(problem_gate["immediate_persistence_failure_tasks_min"])
        and len(disagreement_mechanisms) >= int(problem_gate["failure_mechanisms_min"])
        and disagreement_fraction >= float(problem_gate["failure_fraction_among_native_success_min"])
        and maximum_family_fraction <= float(problem_gate["single_predicate_explanation_fraction_max"])
    )

    recovered = [
        row
        for row in disagreements
        if row.get("branches", {}).get("expert_suffix_then_hold", {}).get("persistent_success", False)
    ]
    recovered_mechanisms = {collapsed_mechanism(row["mechanism"]) for row in recovered}
    headroom_pass = (
        len(recovered) >= int(headroom_gate["suffix_recovered_tasks_min"])
        and len(recovered_mechanisms) >= int(headroom_gate["suffix_recovered_mechanisms_min"])
    )

    if not execution_pass:
        decision = "EVALUATION_INVALID"
    elif not coverage_pass:
        decision = "BASE_NOT_COMPETENT"
    elif not problem_pass:
        decision = "NO_REPEATABLE_GAP"
    elif not headroom_pass:
        decision = "NO_LEGAL_HEADROOM"
    else:
        decision = "PROBLEM_VERIFIED_STRONG_COMPARATOR_RESIDUAL"

    return {
        "decision": decision,
        "execution_gate_pass": execution_pass,
        "coverage_gate_pass": coverage_pass,
        "problem_gate_pass": problem_pass,
        "headroom_gate_pass": headroom_pass,
        "completed_tasks": len(completed),
        "exception_count": exceptions,
        "finite_actions": finite,
        "cold_repeat_pass": cold_repeat_pass,
        "native_success_tasks": len(native_rows),
        "native_success_task_ids": [row["task_id"] for row in native_rows],
        "native_success_mechanisms": sorted(native_mechanisms),
        "immediate_persistence_failure_tasks": len(disagreements),
        "immediate_persistence_failure_task_ids": [row["task_id"] for row in disagreements],
        "immediate_persistence_failure_mechanisms": sorted(disagreement_mechanisms),
        "immediate_persistence_failure_fraction": disagreement_fraction,
        "failure_predicate_family_counts": family_counts,
        "maximum_failure_predicate_family_fraction": maximum_family_fraction,
        "suffix_recovered_tasks": len(recovered),
        "suffix_recovered_task_ids": [row["task_id"] for row in recovered],
        "suffix_recovered_mechanisms": sorted(recovered_mechanisms),
    }
