"""Pure PCAV-VLA Stage 0A manifest, action, and decision helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA"
TARGET_TASK_IDENTITIES = (
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
)
INITIAL_PHASE_QUOTAS = {"early": 3, "middle": 2, "late": 3}
EXPANDED_PHASE_QUOTAS = {"early": 11, "middle": 10, "late": 11}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def row_key(row: Mapping[str, Any]) -> str:
    return f"{row['task_identity']}|episode={int(row['episode'])}|frame={int(row['frame'])}"


def row_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(row_key(row).encode("utf-8")).hexdigest().upper()


def phase_for_frame(frame: int, episode_length: int, *, future_offset: int = 10) -> str:
    valid_count = int(episode_length) - int(future_offset)
    if valid_count <= 0 or frame < 0 or frame >= valid_count:
        raise ValueError("frame must have a valid future target")
    fraction = frame / valid_count
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def select_stage0_rows(
    population: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    *,
    proposal_hash: str = PROPOSAL_HASH,
) -> list[dict[str, Any]]:
    expected = set(TARGET_TASK_IDENTITIES)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for raw in population:
        row = dict(raw)
        task = str(row["task_identity"])
        if task not in expected:
            continue
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate row identity: {key}")
        seen.add(key)
        phase = str(row.get("phase") or phase_for_frame(row["frame"], row["episode_length"]))
        row["phase"] = phase
        grouped.setdefault((task, phase), []).append(row)

    selected: list[dict[str, Any]] = []
    for task in TARGET_TASK_IDENTITIES:
        for phase in ("early", "middle", "late"):
            count = int(quotas[phase])
            rows = grouped.get((task, phase), [])
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"{proposal_hash}|{task}|{int(row['episode'])}|{int(row['frame'])}".encode("utf-8")
                ).hexdigest()
            )
            if len(rows) < count:
                raise ValueError(f"insufficient {task} {phase} rows: {len(rows)} < {count}")
            selected.extend(rows[:count])

    selected.sort(key=lambda row: (TARGET_TASK_IDENTITIES.index(str(row["task_identity"])), row["phase"], row_key(row)))
    if len({row_key(row) for row in selected}) != len(selected):
        raise ValueError("selected rows contain duplicate identities")
    return selected


def partition_identity_audit(partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    keys = {name: [row_key(row) for row in rows] for name, rows in partitions.items()}
    duplicates = {name: len(values) - len(set(values)) for name, values in keys.items()}
    names = sorted(keys)
    overlap = {
        f"{left}__{right}": len(set(keys[left]) & set(keys[right]))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return {
        "counts": {name: len(values) for name, values in keys.items()},
        "duplicate_counts": duplicates,
        "pairwise_overlap_counts": overlap,
        "passed": not any(duplicates.values()) and not any(overlap.values()),
    }


def chunk_sha256(action_chunk: Any, *, decimals: int = 7) -> str:
    array = np.asarray(action_chunk, dtype=np.float32)
    if array.shape != (50, 7):
        raise ValueError(f"expected action chunk [50,7], received {array.shape}")
    rounded = np.round(array.astype(np.float64), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest().upper()


def candidate_diversity(chunks: Sequence[Any]) -> dict[str, Any]:
    arrays = [np.asarray(chunk, dtype=np.float64) for chunk in chunks]
    if len(arrays) < 2 or any(array.shape != (50, 7) for array in arrays):
        raise ValueError("candidate diversity requires at least two [50,7] chunks")
    hashes = [chunk_sha256(array) for array in arrays]
    distances = [
        float(np.linalg.norm(arrays[left] - arrays[right]) / np.sqrt(arrays[left].size))
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    ]
    nonzero = [value for value in distances if value > 0.0]
    return {
        "candidate_count": len(arrays),
        "unique_chunk_count": len(set(hashes)),
        "chunk_hashes": hashes,
        "pairwise_rms_l2": distances,
        "median_nonzero_pairwise_rms_l2": float(np.median(nonzero)) if nonzero else 0.0,
    }


def discovery_action_scales(action_chunks: Sequence[Any]) -> dict[str, float]:
    array = np.concatenate([np.asarray(chunk, dtype=np.float64)[:10] for chunk in action_chunks], axis=0)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError("action chunks must end in seven action dimensions")
    return {
        "translation": max(float(np.sqrt(np.mean(np.square(array[:, :3])))), 1e-6),
        "rotation": max(float(np.sqrt(np.mean(np.square(array[:, 3:6])))), 1e-6),
        "gripper": max(float(np.sqrt(np.mean(np.square(array[:, 6])))), 1e-6),
    }


def grouped_action_error(candidate: Any, expert: Any, scales: Mapping[str, float]) -> dict[str, float]:
    candidate_array = np.asarray(candidate, dtype=np.float64)[:10]
    expert_array = np.asarray(expert, dtype=np.float64)[:10]
    if candidate_array.shape != (10, 7) or expert_array.shape != (10, 7):
        raise ValueError("grouped action error requires at least ten [7] actions")
    delta = candidate_array - expert_array
    translation = float(np.mean(np.linalg.norm(delta[:, :3], axis=1)))
    rotation = float(np.mean(np.linalg.norm(delta[:, 3:6], axis=1)))
    gripper = float(np.mean(np.abs(delta[:, 6])))
    aggregate = (
        translation / max(float(scales["translation"]), 1e-6)
        + rotation / max(float(scales["rotation"]), 1e-6)
        + gripper / max(float(scales["gripper"]), 1e-6)
    ) / 3.0
    return {
        "translation": translation,
        "rotation": rotation,
        "gripper": gripper,
        "aggregate": float(aggregate),
    }


def oracle_headroom(row_errors: Sequence[Sequence[float]]) -> dict[str, Any]:
    reductions: list[float] = []
    strictly_better = 0
    materially_better = 0
    for values in row_errors:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
            raise ValueError("each row must contain finite Base and alternative errors")
        base = float(array[0])
        best = float(np.min(array))
        reduction = (base - best) / max(base, 1e-6)
        reductions.append(float(reduction))
        strictly_better += int(best < base)
        materially_better += int(reduction >= 0.05)
    improvable = [value for value in reductions if value > 0.0]
    count = len(reductions)
    return {
        "row_count": count,
        "strictly_better_row_count": strictly_better,
        "strictly_better_fraction": strictly_better / count if count else 0.0,
        "materially_better_row_count": materially_better,
        "materially_better_fraction": materially_better / count if count else 0.0,
        "median_oracle_relative_reduction_all": float(np.median(reductions)) if reductions else 0.0,
        "median_oracle_relative_reduction_improvable": float(np.median(improvable)) if improvable else 0.0,
        "pass_threshold": bool(
            count
            and materially_better / count >= 0.25
            and improvable
            and float(np.median(improvable)) >= 0.05
        ),
    }


def action_validity(candidate: Any, base: Any) -> dict[str, Any]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    base_array = np.asarray(base, dtype=np.float64)
    if candidate_array.shape != (50, 7) or base_array.shape != (50, 7):
        raise ValueError("action validity requires [50,7] candidate and Base chunks")

    def metrics(array: np.ndarray) -> dict[str, float]:
        absolute = np.abs(array)
        exceedance = np.maximum(absolute - 1.0, 0.0)
        return {
            "finite_fraction": float(np.mean(np.isfinite(array))),
            "absolute_max": float(np.nanmax(absolute)),
            "outside_fraction": float(np.mean(absolute > 1.0)),
            "p99_exceedance": float(np.nanpercentile(exceedance, 99)),
        }

    observed = metrics(candidate_array)
    base_metrics = metrics(base_array)
    limits = {
        "absolute_max": 1.25,
        "outside_fraction": base_metrics["outside_fraction"] + 0.01,
        "p99_exceedance": base_metrics["p99_exceedance"] + 0.02,
    }
    passed = bool(
        observed["finite_fraction"] == 1.0
        and observed["absolute_max"] <= limits["absolute_max"]
        and observed["outside_fraction"] <= limits["outside_fraction"]
        and observed["p99_exceedance"] <= limits["p99_exceedance"]
    )
    return {"passed": passed, "observed": observed, "base": base_metrics, "limits": limits}


def aggregate_candidate_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique_counts = [int(row["diversity"]["unique_chunk_count"]) for row in rows]
    medians = [float(row["diversity"]["median_nonzero_pairwise_rms_l2"]) for row in rows]
    candidate_validity = [bool(candidate["validity"]["passed"]) for row in rows for candidate in row["candidates"]]
    base_validity = [bool(row["candidates"][0]["validity"]["passed"]) for row in rows]
    rows_with_valid_alternative = [
        any(bool(candidate["validity"]["passed"]) for candidate in row["candidates"][1:]) for row in rows
    ]
    phase_counts = Counter(str(row["phase"]) for row in rows)
    task_counts = Counter(str(row["task_identity"]) for row in rows)
    count = len(rows)
    return {
        "row_count": count,
        "rows_with_two_unique_chunks": sum(value >= 2 for value in unique_counts),
        "fraction_rows_with_two_unique_chunks": sum(value >= 2 for value in unique_counts) / count if count else 0.0,
        "median_nonzero_pairwise_rms_l2": float(np.median([value for value in medians if value > 0.0]))
        if any(value > 0.0 for value in medians)
        else 0.0,
        "all_base_candidates_valid": bool(base_validity and all(base_validity)),
        "rows_with_valid_alternative": sum(rows_with_valid_alternative),
        "fraction_rows_with_valid_alternative": sum(rows_with_valid_alternative) / count if count else 0.0,
        "invalid_candidate_count": sum(not value for value in candidate_validity),
        "phase_counts": dict(sorted(phase_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }


def validate_partial_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("proposal_hash") != PROPOSAL_HASH:
        raise ValueError("partial proposal hash mismatch")
    planned = int(payload.get("planned_row_count", -1))
    completed = int(payload.get("completed_row_count", -1))
    rows = list(payload.get("rows") or [])
    keys = list(payload.get("completed_row_keys") or [])
    if planned not in {24, 96}:
        raise ValueError("partial planned row count must be 24 or 96")
    if completed != len(rows) or completed != len(keys):
        raise ValueError("partial completed counts disagree")
    if completed > planned:
        raise ValueError("partial completed count exceeds planned count")
    if len(keys) != len(set(keys)):
        raise ValueError("partial contains duplicate completed row keys")
    for row, key in zip(rows, keys, strict=True):
        if row_key(row) != key:
            raise ValueError("partial row order/key mismatch")
        candidates = list(row.get("candidates") or [])
        if len(candidates) != 4:
            raise ValueError("each completed row must contain four candidates")
        indices = [int(candidate["candidate_index"]) for candidate in candidates]
        if indices != [0, 1, 2, 3]:
            raise ValueError("candidate indices must be exactly 0,1,2,3")
    return {
        "planned_row_count": planned,
        "completed_row_count": completed,
        "missing_row_count": planned - completed,
        "exception_count": int(payload.get("exception_count", 0)),
        "duplicate_row_key_count": len(keys) - len(set(keys)),
    }


def classify_stage0a(audit: Mapping[str, Any]) -> str:
    implementation_flags = (
        bool(audit.get("exception_count")),
        bool(audit.get("duplicate_key_count")),
        not bool(audit.get("mapping_passed")),
        not bool(audit.get("partition_passed")),
        not bool(audit.get("reload_passed")),
        not bool(audit.get("source_health_passed")),
        not bool(audit.get("manifest_passed")),
        float(audit.get("base_identity_max_abs_error", float("inf"))) != 0.0,
        not bool(audit.get("all_base_candidates_valid")),
        int(audit.get("confirmatory_observations_decoded", -1)) != 0,
        int(audit.get("confirmatory_actions_computed", -1)) != 0,
    )
    if any(implementation_flags):
        return "PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE"

    if (
        float(audit.get("fraction_rows_with_two_unique_chunks", 0.0)) <= 0.5
        or float(audit.get("fraction_rows_with_valid_alternative", 0.0)) <= 0.5
        or float(audit.get("median_nonzero_pairwise_rms_l2", 0.0)) <= 1e-4
    ):
        return "PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED"

    headroom = audit.get("headroom") or {}
    if bool(headroom.get("pass_threshold")):
        return "PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    if int(headroom.get("strictly_better_row_count", 0)) == 0:
        return "PCAV_STAGE_0A_NO_USABLE_HEADROOM"
    if int(audit.get("completed_row_count", 0)) < 96:
        return "PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED"
    return "PCAV_STAGE_0A_NO_USABLE_HEADROOM"


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()
