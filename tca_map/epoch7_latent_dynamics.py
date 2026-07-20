"""Outcome-independent utilities for the Epoch 7 latent-dynamics gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("status") != "FROZEN_BEFORE_CLAIM_SPECIFIC_CLOSED_LOOP_OUTCOMES":
        errors.append("protocol status is not frozen")
    if protocol.get("ours_authorized") is not False:
        errors.append("Ours must remain unauthorized")
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 4:
        errors.append("exactly four frozen discovery tasks are required")
        return errors
    eval_ids = [int(task.get("eval_id", -1)) for task in tasks]
    if eval_ids != [0, 2, 3, 9]:
        errors.append("task identities must remain [0, 2, 3, 9]")
    if len({str(task.get("target_body")) for task in tasks}) != len(tasks):
        errors.append("target bodies must be unique")
    axes = {str(task.get("intervention", {}).get("axis")) for task in tasks}
    if axes != {"joint_damping", "target_contact_friction", "target_mass"}:
        errors.append("unexpected or missing intervention axes")
    identities = protocol.get("identities", {})
    discovery = list(identities.get("discovery_state_indices", []))
    confirmation = list(identities.get("sealed_confirmation_state_indices", []))
    expansion = list(identities.get("bounded_expansion_state_indices", []))
    if discovery != [0, 1, 2]:
        errors.append("discovery identities must remain [0, 1, 2]")
    if set(discovery) & set(confirmation):
        errors.append("discovery and confirmation identities overlap")
    if set(discovery) & set(expansion) or set(confirmation) & set(expansion):
        errors.append("bounded-expansion identities overlap another partition")
    if protocol.get("gates", {}).get("execution_validity", {}).get("planned_base_episodes") != 24:
        errors.append("planned Base episode count must remain 24")
    return errors


def iter_episode_specs(protocol: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    indices = [int(value) for value in protocol["identities"]["discovery_state_indices"]]
    for task in protocol["tasks"]:
        eval_id = int(task["eval_id"])
        for state_index in indices:
            seed = 100_000 + 1_000 * eval_id + state_index
            for condition in ("standard", "latent_dynamics_intervention"):
                yield {
                    "episode_id": f"eval{eval_id}_state{state_index}_{condition}",
                    "eval_id": eval_id,
                    "family": str(task["family"]),
                    "goal_bddl": str(task["goal_bddl"]),
                    "instruction": str(task["instruction"]),
                    "target_body": str(task["target_body"]),
                    "intervention": dict(task["intervention"]),
                    "state_index": state_index,
                    "seed": seed,
                    "model_seed": seed,
                    "condition": condition,
                }


def _name(model: Any, kind: str, index: int) -> str:
    function = getattr(model, f"{kind}_id2name", None)
    if callable(function):
        value = function(int(index))
        if value is not None:
            return value.decode() if isinstance(value, bytes) else str(value)
    names = getattr(model, f"{kind}_names", ())
    if int(index) < len(names):
        value = names[int(index)]
        return value.decode() if isinstance(value, bytes) else str(value)
    return ""


def _id(model: Any, kind: str, name: str) -> int:
    function = getattr(model, f"{kind}_name2id", None)
    if callable(function):
        return int(function(str(name)))
    count = int(getattr(model, f"n{kind[:3]}", 0))
    for index in range(count):
        if _name(model, kind, index) == name:
            return index
    raise KeyError(f"unknown {kind} name: {name}")


def body_descendant_ids(model: Any, target_body: str) -> set[int]:
    target_id = _id(model, "body", target_body)
    descendants: set[int] = set()
    for body_id in range(int(model.nbody)):
        cursor = body_id
        visited: set[int] = set()
        while cursor not in visited and cursor >= 0:
            if cursor == target_id:
                descendants.add(body_id)
                break
            visited.add(cursor)
            if cursor == 0:
                break
            cursor = int(model.body_parentid[cursor])
    return descendants


def is_gripper_body_name(name: str) -> bool:
    normalized = str(name).lower()
    return any(token in normalized for token in ("gripper", "finger", "hand"))


def target_contact_state(sim: Any, target_body: str) -> dict[str, Any]:
    model = sim.model
    data = sim.data
    target_ids = body_descendant_ids(model, target_body)
    target_contacts: list[dict[str, Any]] = []
    other_contacts: list[dict[str, Any]] = []
    for contact_index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        name1 = _name(model, "body", body1)
        name2 = _name(model, "body", body2)
        record = {
            "contact_index": contact_index,
            "geom1": _name(model, "geom", geom1),
            "geom2": _name(model, "geom", geom2),
            "body1": name1,
            "body2": name2,
        }
        if is_gripper_body_name(name1):
            (target_contacts if body2 in target_ids else other_contacts).append(record)
        elif is_gripper_body_name(name2):
            (target_contacts if body1 in target_ids else other_contacts).append(record)
    return {
        "target_body": target_body,
        "target_body_ids": sorted(target_ids),
        "target_contact": bool(target_contacts),
        "target_contacts": target_contacts,
        "other_gripper_contacts": other_contacts,
    }


def _array_hash(array: Any) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def apply_intervention(model: Any, intervention: Mapping[str, Any], *, factor_override: float | None = None) -> dict[str, Any]:
    axis = str(intervention["axis"])
    factor = float(intervention["factor"] if factor_override is None else factor_override)
    records: list[dict[str, Any]] = []

    if axis == "joint_damping":
        joint_name = str(intervention["joint_name"])
        joint_id = _id(model, "joint", joint_name)
        start = int(model.jnt_dofadr[joint_id])
        stop = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < int(model.njnt) else int(model.nv)
        indices = list(range(start, stop))
        before = np.asarray(model.dof_damping[indices], dtype=np.float64).copy()
        model.dof_damping[indices] = before * factor
        after = np.asarray(model.dof_damping[indices], dtype=np.float64).copy()
        records.append({"array": "dof_damping", "indices": indices, "before": before.tolist(), "after": after.tolist()})
    elif axis == "target_contact_friction":
        body_name = str(intervention["body_name"])
        body_ids = body_descendant_ids(model, body_name)
        geom_indices = [
            index
            for index in range(int(model.ngeom))
            if int(model.geom_bodyid[index]) in body_ids
            and (not intervention.get("collision_geoms_only") or int(model.geom_contype[index]) != 0)
        ]
        if not geom_indices:
            raise ValueError(f"no eligible collision geoms for {body_name}")
        components = [int(value) for value in intervention["components"]]
        before = np.asarray(model.geom_friction[np.ix_(geom_indices, components)], dtype=np.float64).copy()
        model.geom_friction[np.ix_(geom_indices, components)] = before * factor
        after = np.asarray(model.geom_friction[np.ix_(geom_indices, components)], dtype=np.float64).copy()
        records.append(
            {
                "array": "geom_friction",
                "geom_indices": geom_indices,
                "geom_names": [_name(model, "geom", index) for index in geom_indices],
                "components": components,
                "before": before.tolist(),
                "after": after.tolist(),
            }
        )
    elif axis == "target_mass":
        body_name = str(intervention["body_name"])
        body_id = _id(model, "body", body_name)
        mass_before = float(model.body_mass[body_id])
        inertia_before = np.asarray(model.body_inertia[body_id], dtype=np.float64).copy()
        model.body_mass[body_id] = mass_before * factor
        model.body_inertia[body_id] = inertia_before * factor
        records.extend(
            [
                {
                    "array": "body_mass",
                    "body_id": body_id,
                    "body_name": body_name,
                    "before": [mass_before],
                    "after": [float(model.body_mass[body_id])],
                },
                {
                    "array": "body_inertia",
                    "body_id": body_id,
                    "body_name": body_name,
                    "before": inertia_before.tolist(),
                    "after": np.asarray(model.body_inertia[body_id], dtype=np.float64).tolist(),
                },
            ]
        )
    else:
        raise ValueError(f"unsupported intervention axis: {axis}")

    changed_values = sum(
        int(np.count_nonzero(np.asarray(record["before"]) != np.asarray(record["after"]))) for record in records
    )
    return {
        "axis": axis,
        "factor": factor,
        "records": records,
        "changed_values": changed_values,
        "records_sha256": _array_hash(
            np.concatenate(
                [
                    np.asarray(record["before"], dtype=np.float64).reshape(-1)
                    for record in records
                ]
                + [
                    np.asarray(record["after"], dtype=np.float64).reshape(-1)
                    for record in records
                ]
            )
        ),
    }


def compare_observations(
    standard: Mapping[str, Any],
    intervened: Mapping[str, Any],
    *,
    image_keys: Iterable[str] = ("agentview_image", "robot0_eye_in_hand_image"),
    proprio_keys: Iterable[str] = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
    ),
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key in tuple(image_keys) + tuple(proprio_keys):
        if key not in standard and key not in intervened:
            continue
        if key not in standard or key not in intervened:
            rows[key] = {"present_in_both": False, "shape_equal": False, "max_abs": None}
            continue
        left = np.asarray(standard[key])
        right = np.asarray(intervened[key])
        shape_equal = left.shape == right.shape
        max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if shape_equal and left.size else 0.0
        rows[key] = {"present_in_both": True, "shape_equal": shape_equal, "max_abs": max_abs}
    image_ok = all(rows.get(key, {}).get("max_abs") == 0.0 for key in image_keys if key in rows)
    proprio_ok = all(
        rows.get(key, {}).get("max_abs") is not None and rows[key]["max_abs"] <= 1e-9
        for key in proprio_keys
        if key in rows
    )
    required_images_present = all(key in rows and rows[key]["present_in_both"] for key in image_keys)
    return {
        "keys": rows,
        "required_images_present": required_images_present,
        "image_exact": required_images_present and image_ok,
        "proprio_within_tolerance": proprio_ok,
        "eligible": required_images_present and image_ok and proprio_ok,
    }


def adjudicate_discovery(protocol: Mapping[str, Any], episodes: Iterable[Mapping[str, Any]], expert_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [dict(row) for row in episodes if row.get("completed") is True]
    experts = [dict(row) for row in expert_rows]
    eligible_tasks = {
        int(row["eval_id"])
        for row in experts
        if row.get("standard_success") is True and row.get("intervention_success") is True
    }
    eligible = [row for row in completed if int(row["eval_id"]) in eligible_tasks]
    pairs: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for row in eligible:
        key = (int(row["eval_id"]), int(row["state_index"]))
        pairs.setdefault(key, {})[str(row["condition"])] = row
    paired = [value for value in pairs.values() if set(value) == {"standard", "latent_dynamics_intervention"}]
    standard_wins = [
        value
        for value in paired
        if value["standard"].get("success") is True and value["latent_dynamics_intervention"].get("success") is False
    ]
    intervention_wins = [
        value
        for value in paired
        if value["standard"].get("success") is False and value["latent_dynamics_intervention"].get("success") is True
    ]
    contact_failures = [
        value
        for value in standard_wins
        if value["latent_dynamics_intervention"].get("target_contact_any") is True
    ]
    standard_successes = sum(value["standard"].get("success") is True for value in paired)
    intervention_successes = sum(value["latent_dynamics_intervention"].get("success") is True for value in paired)
    eligible_pairs = len(paired)
    required_wins = max(3, math.ceil(0.25 * eligible_pairs))
    drop_pp = 100.0 * (standard_successes - intervention_successes) / eligible_pairs if eligible_pairs else None
    checks = {
        "eligible_task_count_min": len(eligible_tasks) >= 3,
        "eligible_family_count_min": len({row["family"] for row in eligible if int(row["eval_id"]) in eligible_tasks}) >= 3,
        "all_pairs_complete": eligible_pairs == 3 * len(eligible_tasks),
        "standard_success_fraction_min": bool(eligible_pairs) and standard_successes / eligible_pairs >= 0.75,
        "standard_success_task_count_min": len(
            {
                int(value["standard"]["eval_id"])
                for value in paired
                if value["standard"].get("success") is True
            }
        )
        >= 3,
        "standard_success_family_count_min": len(
            {
                str(value["standard"]["family"])
                for value in paired
                if value["standard"].get("success") is True
            }
        )
        >= 3,
        "paired_drop_min": drop_pp is not None and drop_pp >= 20.0,
        "standard_win_count_min": len(standard_wins) >= required_wins,
        "adverse_task_count_min": len({int(value["standard"]["eval_id"]) for value in standard_wins}) >= 2,
        "adverse_family_count_min": len({str(value["standard"]["family"]) for value in standard_wins}) >= 2,
        "contact_failure_count_min": len(contact_failures) >= 2,
        "contact_failure_family_count_min": len(
            {str(value["standard"]["family"]) for value in contact_failures}
        )
        >= 2,
        "intervention_wins_do_not_dominate": len(intervention_wins) <= len(standard_wins),
    }
    return {
        "eligible_task_ids": sorted(eligible_tasks),
        "eligible_pairs": eligible_pairs,
        "standard_successes": standard_successes,
        "intervention_successes": intervention_successes,
        "paired_drop_percentage_points": drop_pp,
        "required_standard_wins": required_wins,
        "standard_wins": len(standard_wins),
        "intervention_wins": len(intervention_wins),
        "contact_preserving_failures": len(contact_failures),
        "checks": checks,
        "pass": all(checks.values()),
    }

