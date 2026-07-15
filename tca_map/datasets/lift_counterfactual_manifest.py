"""LIFT-specific same-scene counterfactual manifest construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "lift-vla-counterfactual-manifest-v1"
PARTITIONS = {
    "discovery": (0, 1, 2, 3),
    "validation": (4, 5, 6),
    "confirmatory": (7, 8, 9),
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _payload_hash(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_hash(state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(state))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _typed_names(groups: Mapping[str, Sequence[str]]) -> list[str]:
    return sorted(str(name) for names in groups.values() for name in names)


def _goal_entities(goal_state: Sequence[Sequence[str]]) -> set[str]:
    return {str(item) for predicate in goal_state for item in predicate[1:]}


def _scene_payload(parsed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "problem_name": parsed.get("problem_name"),
        "fixtures": _canonical(parsed.get("fixtures", {})),
        "objects": _canonical(parsed.get("objects", {})),
        "regions": _canonical(parsed.get("regions", {})),
        "scene_properties": _canonical(parsed.get("scene_properties", {})),
        "initial_state": _canonical(parsed.get("initial_state", [])),
    }


def parse_goal_tasks(
    bddl_root: Path,
    parser: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if parser is None:
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        parser = robosuite_parse_problem
    paths = sorted(bddl_root.glob("*.bddl"), key=lambda path: path.name)
    tasks: list[dict[str, Any]] = []
    for sorted_index, path in enumerate(paths):
        parsed = _canonical(parser(str(path)))
        scene = _scene_payload(parsed)
        language_value = parsed.get("language_instruction", "")
        language = (
            " ".join(str(token) for token in language_value)
            if isinstance(language_value, list)
            else str(language_value)
        )
        tasks.append(
            {
                "sorted_index": int(sorted_index),
                "task_id": path.stem,
                "bddl_path": str(path),
                "bddl_sha256": _file_hash(path),
                "language": language,
                "problem_name": str(parsed.get("problem_name", "")),
                "objects": _typed_names(parsed.get("objects", {})),
                "fixtures": _typed_names(parsed.get("fixtures", {})),
                "regions": sorted(str(key) for key in parsed.get("regions", {})),
                "obj_of_interest": sorted(str(value) for value in parsed.get("obj_of_interest", [])),
                "goal_state": _canonical(parsed.get("goal_state", [])),
                "initial_predicates": _canonical(parsed.get("initial_state", [])),
                "scene_sha256": _payload_hash(scene),
                "parsed": parsed,
            }
        )
    return tasks


def _partition_for_index(index: int) -> str:
    for name, indices in PARTITIONS.items():
        if index in indices:
            return name
    raise ValueError(f"target index {index} is outside the frozen partition")


def _source_index(target_index: int) -> int:
    partition = _partition_for_index(target_index)
    indices = PARTITIONS[partition]
    position = indices.index(target_index)
    return int(indices[(position + 1) % len(indices)])


def _static_gate(source: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if source["scene_sha256"] != target["scene_sha256"]:
        errors.append("scene_signature_mismatch")
    if source["goal_state"] == target["goal_state"]:
        errors.append("source_and_target_goal_are_equal")
    available = set(source["objects"]) | set(source["fixtures"]) | set(source["regions"])
    missing = sorted(_goal_entities(target["goal_state"]) - available)
    if missing:
        errors.append("missing_target_entities:" + ",".join(missing))
    if not target["goal_state"]:
        errors.append("empty_target_goal")
    return not errors, errors


def build_counterfactual_manifest(
    tasks: Sequence[Mapping[str, Any]],
    initial_state_loader: Callable[[Mapping[str, Any], int], Any],
    dynamic_validator: Callable[[dict[str, Any], Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build rows without selecting any task or reset from observed outcomes."""

    if len(tasks) != 10:
        raise ValueError(f"frozen libero_goal audit requires 10 sorted tasks, found {len(tasks)}")
    by_index = {int(task["sorted_index"]): dict(task) for task in tasks}
    if sorted(by_index) != list(range(10)):
        raise ValueError("sorted task indices must be exactly 0..9")

    rows: list[dict[str, Any]] = []
    for target_index in range(10):
        target = by_index[target_index]
        source = by_index[_source_index(target_index)]
        partition = _partition_for_index(target_index)
        static_valid, static_errors = _static_gate(source, target)
        for repeat in range(2):
            initial_state_index = 2 * target_index + repeat
            reset_seed = 15000 + initial_state_index
            initial_state = initial_state_loader(source, initial_state_index)
            initial_array = np.asarray(initial_state)
            manifest_key = (
                f"libero_goal/{source['task_id']}/{target['task_id']}/{reset_seed}"
            )
            row: dict[str, Any] = {
                "manifest_key": manifest_key,
                "suite": "libero_goal",
                "evidence_partition": partition,
                "source_sorted_index": int(source["sorted_index"]),
                "source_task": source["task_id"],
                "source_language": source["language"],
                "source_bddl_path": source["bddl_path"],
                "source_bddl_sha256": source["bddl_sha256"],
                "source_goal_state": source["goal_state"],
                "target_sorted_index": int(target["sorted_index"]),
                "target_task": target["task_id"],
                "target_language": target["language"],
                "target_bddl_path": target["bddl_path"],
                "target_bddl_sha256": target["bddl_sha256"],
                "target_goal_state": target["goal_state"],
                "target_objects": target["objects"],
                "target_fixtures": target["fixtures"],
                "target_regions": target["regions"],
                "target_obj_of_interest": target["obj_of_interest"],
                "scene_sha256": target["scene_sha256"],
                "reset_seed": int(reset_seed),
                "source_initial_state_index": int(initial_state_index),
                "initial_state_shape": [int(dim) for dim in initial_array.shape],
                "initial_state_dtype": str(initial_array.dtype),
                "initial_state_sha256": _state_hash(initial_state),
                "feasibility_audit": {
                    "pairing_rule": "next target task within the frozen evidence partition",
                    "reset_rule": "source init-state indices 2*target_sorted_index and +1",
                    "same_scene": source["scene_sha256"] == target["scene_sha256"],
                    "target_goal_differs": source["goal_state"] != target["goal_state"],
                    "static_valid": bool(static_valid),
                    "static_errors": static_errors,
                },
                "grounding_scorer": {
                    "environment": "target BDDL environment",
                    "definition": "conjunction of target goal predicates that mention target obj_of_interest",
                    "goal_state": target["goal_state"],
                    "obj_of_interest": target["obj_of_interest"],
                },
                "task_success_scorer": {
                    "environment": "target BDDL environment",
                    "definition": "OffScreenRenderEnv.check_success -> target environment _check_success",
                    "goal_state": target["goal_state"],
                },
                "old_offline_proxy_used": False,
                "valid": bool(static_valid),
                "errors": list(static_errors),
            }
            if dynamic_validator is not None:
                dynamic = dict(dynamic_validator(row, initial_state))
                row["dynamic_scorer_audit"] = dynamic
                if not bool(dynamic.get("valid", False)):
                    row["valid"] = False
                    row["errors"].extend(str(item) for item in dynamic.get("errors", []))
            rows.append(row)

    key_sets = {
        name: {row["manifest_key"] for row in rows if row["evidence_partition"] == name}
        for name in PARTITIONS
    }
    state_sets = {
        name: {row["initial_state_sha256"] for row in rows if row["evidence_partition"] == name}
        for name in PARTITIONS
    }
    overlaps = []
    names = list(PARTITIONS)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if key_sets[left] & key_sets[right]:
                overlaps.append(f"manifest_key:{left}:{right}")
            if state_sets[left] & state_sets[right]:
                overlaps.append(f"initial_state:{left}:{right}")
    target_counts = {
        name: len({row["target_sorted_index"] for row in rows if row["evidence_partition"] == name})
        for name in PARTITIONS
    }
    valid_rows = [row for row in rows if row["valid"]]
    minimums = {"discovery": 4, "validation": 2, "confirmatory": 2}
    minimums_passed = all(target_counts[name] >= minimums[name] for name in minimums)
    development_scoreable = sum(
        1 for row in valid_rows if row["evidence_partition"] in {"discovery", "validation"}
    )
    ready = (
        len(valid_rows) == len(rows)
        and minimums_passed
        and development_scoreable >= 10
        and not overlaps
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "partition_rule": {name: list(indices) for name, indices in PARTITIONS.items()},
        "pairing_rule": "cyclic next task within each frozen target-task partition",
        "reset_rule": "two source states per target at indices 2*target_sorted_index and +1",
        "task_count": len(tasks),
        "row_count": len(rows),
        "valid_row_count": len(valid_rows),
        "target_task_counts": target_counts,
        "minimum_target_task_counts": minimums,
        "development_scoreable_episode_count": int(development_scoreable),
        "partition_overlaps": overlaps,
        "confirmatory_policy_observations_decoded": 0,
        "confirmatory_policy_actions_computed": 0,
        "ready_for_stage_0_model_load": bool(ready),
        "rows": rows,
    }
    payload["canonical_payload_sha256"] = _payload_hash(payload)
    return payload
