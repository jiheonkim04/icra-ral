"""Small ContactSet-VLA offline action-head diagnostic.

This runner trains tiny CPU NumPy action heads over local LIBERO HDF5 action
chunks. It compares geometry injection variants without importing VLA models,
running simulators, using GPU, downloading assets, or making paper-grade claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_metadata_subset import read_asset_paths

SCHEMA_VERSION = "contactset-vla-diagnostic-v1"
DEFAULT_MAX_DEMOS = 6
DEFAULT_MAX_ACTION_STEPS = 140
DEFAULT_FEATURE_WIDTH = 48
DEFAULT_RIDGE = 1e-3
VARIANTS = (
    "no_geometry_injection",
    "single_3d_point_injection",
    "source_object_point_only",
    "destination_placement_point_only",
    "source_destination_two_point_injection",
    "full_contact_set_injection",
)
ROLE_TO_ID = {
    "source": 0,
    "destination": 1,
    "support": 2,
    "safety": 3,
    "normal": 4,
}
ROLE_WIDTH = len(ROLE_TO_ID)


class ContactSetDiagnosticError(RuntimeError):
    """Raised when the ContactSet-VLA diagnostic cannot run safely."""


@dataclass(frozen=True)
class DemoCase:
    file: str
    demo_name: str
    instruction: str
    actions: np.ndarray
    eef_pos: np.ndarray
    ee_ori: np.ndarray
    gripper_aperture: np.ndarray
    source: np.ndarray | None
    destination: np.ndarray | None
    support: np.ndarray | None
    safety: np.ndarray | None
    normal: np.ndarray | None
    source_name: str | None
    destination_name: str | None
    safety_name: str | None
    geometry_sources: dict[str, str]
    leakage_audit: dict[str, Any]


def _round(value: Any, digits: int = 9) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _normalize_name(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(main|joint|joint0|default|site|body|geom|pos|position)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "it",
        "in",
        "on",
        "to",
        "of",
        "at",
        "for",
        "with",
        "put",
        "pick",
        "place",
        "turn",
        "close",
        "open",
        "while",
        "avoid",
        "avoiding",
    }
    return {token for token in re.findall(r"[a-z0-9]+", _normalize_name(text)) if token not in stop}


def _split_instruction_for_geometry(instruction: str) -> tuple[str, str]:
    """Return rough source and destination phrases without using task labels."""

    normalized = " " + _normalize_name(instruction) + " "
    separators = (
        " into ",
        " in ",
        " inside ",
        " on top of ",
        " on ",
        " onto ",
        " to ",
    )
    matches: list[tuple[int, str]] = []
    for sep in separators:
        index = normalized.rfind(sep)
        if index > 0:
            matches.append((index, sep))
    if not matches:
        return instruction, instruction
    index, sep = max(matches, key=lambda item: item[0])
    source = normalized[:index].strip()
    destination = normalized[index + len(sep) :].strip()
    return source or instruction, destination or instruction


def _score_name(name: str, instruction: str) -> float:
    name_tokens = _tokens(name)
    instruction_tokens = _tokens(instruction)
    if not name_tokens or not instruction_tokens:
        return 0.0
    overlap = name_tokens & instruction_tokens
    return float(len(overlap) / max(1, len(name_tokens)))


def _hash_features(text: str, width: int) -> np.ndarray:
    vector = np.zeros(width, dtype=np.float64)
    cleaned = _normalize_name(text)
    words = re.findall(r"[a-z0-9]+", cleaned)
    scalars = [
        min(len(cleaned), 240) / 240.0,
        min(len(words), 48) / 48.0,
        sum(char in "aeiou" for char in cleaned) / max(1, len(cleaned)),
        1.0 if "stove" in words else 0.0,
        1.0 if "plate" in words or "basket" in words else 0.0,
        1.0 if "drawer" in words or "cabinet" in words else 0.0,
    ]
    vector[: min(width, len(scalars))] = scalars[:width]
    usable = max(1, width - len(scalars))
    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = len(scalars) + (value % usable)
        if index < width:
            vector[index] += 1.0 if value & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _parse_vec(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    values = [float(item) for item in text.split()[:3]]
    while len(values) < 3:
        values.append(0.0)
    return np.asarray(values[:3], dtype=np.float64)


def _joint_qpos_starts(root: ET.Element) -> dict[str, tuple[int, int, str]]:
    starts: dict[str, tuple[int, int, str]] = {}
    cursor = 0
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "hinge")
        width = 7 if joint_type == "free" else (4 if joint_type == "ball" else 1)
        if name:
            starts[name] = (cursor, width, joint_type)
        cursor += width
    return starts


def _is_robotish(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("robot", "gripper", "mount", "eef", "finger", "joint"))


def _parse_model_points(xml_text: str) -> tuple[dict[str, tuple[str, int]], dict[str, np.ndarray], int]:
    if not xml_text:
        return {}, {}, 0
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}, {}, 0
    starts = _joint_qpos_starts(root)
    free_joints: dict[str, tuple[str, int]] = {}
    static_points: dict[str, np.ndarray] = {}

    def walk(element: ET.Element, parent_pos: np.ndarray) -> None:
        current_pos = parent_pos
        is_free_body = False
        body_name = element.attrib.get("name", "")
        if element.tag == "body":
            current_pos = parent_pos + _parse_vec(element.attrib.get("pos"))
            for child in element:
                if child.tag == "joint" and child.attrib.get("type") == "free":
                    joint_name = child.attrib.get("name", "")
                    if joint_name in starts:
                        start, _width, _joint_type = starts[joint_name]
                        clean = re.sub(r"_(main|body)$", "", body_name)
                        free_joints[clean or joint_name] = (joint_name, start)
                        is_free_body = True
            if body_name and not is_free_body and not _is_robotish(body_name):
                static_points[body_name] = current_pos.copy()
        elif element.tag == "site":
            site_name = element.attrib.get("name", "")
            if site_name and not _is_robotish(site_name):
                static_points[site_name] = parent_pos + _parse_vec(element.attrib.get("pos"))
        for child in element:
            if child.tag in {"body", "site"}:
                walk(child, current_pos)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in worldbody:
            if child.tag in {"body", "site"}:
                walk(child, np.zeros(3, dtype=np.float64))
    qpos_width = 0
    for start, width, _joint_type in starts.values():
        qpos_width = max(qpos_width, start + width)
    return free_joints, static_points, qpos_width


def _detect_qpos_offset(states: np.ndarray, joint_states: np.ndarray | None, qpos_width: int) -> int:
    if states.ndim != 2 or states.shape[1] < qpos_width:
        return 0
    if joint_states is not None and joint_states.ndim == 2 and joint_states.shape[1] >= 7:
        candidates: list[tuple[float, int]] = []
        for offset in (0, 1):
            if states.shape[1] >= offset + 7:
                err = float(np.mean(np.abs(states[: min(5, len(states)), offset : offset + 7] - joint_states[: min(5, len(joint_states)), :7])))
                candidates.append((err, offset))
        if candidates:
            return min(candidates)[1]
    return 1 if states.shape[1] >= qpos_width + 1 and abs(float(states[0, 0])) > 1e-9 else 0


def _extract_obs_position_traces(obs: Any, limit: int) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    if obs is None:
        return traces
    for key in sorted(str(item) for item in obs.keys()):
        lowered = key.lower()
        if not (lowered.endswith("_pos") or lowered.endswith("_position")):
            continue
        if any(skip in lowered for skip in ("robot", "eef", "ee_", "joint", "gripper")):
            continue
        arr = np.asarray(obs[key][:limit], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            traces[key] = arr[:, :3]
    return traces


def _extract_state_free_traces(
    states: np.ndarray | None,
    joint_states: np.ndarray | None,
    free_joints: dict[str, tuple[str, int]],
    qpos_width: int,
    limit: int,
) -> tuple[dict[str, np.ndarray], int | None]:
    if states is None or states.ndim != 2 or qpos_width <= 0:
        return {}, None
    offset = _detect_qpos_offset(states, joint_states, qpos_width)
    traces: dict[str, np.ndarray] = {}
    for name, (_joint_name, start) in free_joints.items():
        begin = offset + start
        end = begin + 3
        if states.shape[1] >= end:
            traces[name] = np.asarray(states[:limit, begin:end], dtype=np.float64)
    return traces, offset


def _select_named_trace(
    traces: dict[str, np.ndarray],
    instruction: str,
    *,
    exclude: set[str] | None = None,
    require_positive_score: bool = False,
) -> tuple[str | None, np.ndarray | None, float, str]:
    exclude = exclude or set()
    candidates = [(name, trace, _score_name(name, instruction)) for name, trace in traces.items() if name not in exclude]
    if not candidates:
        return None, None, 0.0, "unavailable"
    candidates.sort(key=lambda item: (item[2], -len(item[0])), reverse=True)
    name, trace, score = candidates[0]
    if score <= 0.0 and require_positive_score:
        return None, None, score, "no_instruction_overlap"
    source = "instruction_token_overlap" if score > 0.0 else "fallback_first_observable_trace"
    return name, trace, score, source


def _select_static_destination(
    static_points: dict[str, np.ndarray],
    object_traces: dict[str, np.ndarray],
    instruction: str,
    source_name: str | None,
    horizon: int,
) -> tuple[str | None, np.ndarray | None, float, str]:
    candidates: list[tuple[str, np.ndarray, float, str]] = []
    for name, point in static_points.items():
        if source_name and _normalize_name(source_name) in _normalize_name(name):
            continue
        score = _score_name(name, instruction)
        candidates.append((name, np.repeat(point.reshape(1, 3), horizon, axis=0), score, "xml_static_body_or_site"))
    for name, trace in object_traces.items():
        if name == source_name:
            continue
        score = _score_name(name, instruction)
        candidates.append((name, trace, score, "observable_position_trace"))
    if not candidates:
        return None, None, 0.0, "unavailable"
    support_tokens = ("region", "site", "zone", "target", "plate", "stove", "drawer", "cabinet", "basket", "microwave")
    candidates.sort(key=lambda item: (item[2], any(token in item[0].lower() for token in support_tokens)), reverse=True)
    name, trace, score, source = candidates[0]
    if score <= 0.0:
        return None, None, score, "no_instruction_overlap"
    return name, trace, score, source


def _nearest_safety_trace(
    traces: dict[str, np.ndarray],
    source_name: str | None,
    destination_name: str | None,
    source: np.ndarray | None,
) -> tuple[str | None, np.ndarray | None]:
    candidates = {name: trace for name, trace in traces.items() if name not in {source_name, destination_name}}
    if not candidates:
        return None, None
    if source is None:
        name = sorted(candidates)[0]
        return name, candidates[name]
    source0 = np.asarray(source[0], dtype=np.float64)
    name = min(candidates, key=lambda key: float(np.linalg.norm(candidates[key][0] - source0)))
    return name, candidates[name]


def _instruction_from_file(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_demo$", "", stem)
    stem = re.sub(r"^[A-Z_0-9]+_", "", stem)
    return stem.replace("_", " ")


def _read_demo_case(path: Path, max_action_steps: int) -> DemoCase | None:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            return None
        demo_name = sorted(str(name) for name in data.keys())[0]
        demo = data[demo_name]
        if "actions" not in demo or "obs" not in demo:
            return None
        actions = np.asarray(demo["actions"][:max_action_steps], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 7 or actions.shape[0] < 8:
            return None
        actions = actions[:, :7]
        horizon = actions.shape[0]
        obs = demo["obs"]
        eef_key = "ee_pos" if "ee_pos" in obs else ("robot0_eef_pos" if "robot0_eef_pos" in obs else "")
        if not eef_key:
            return None
        eef_pos = np.asarray(obs[eef_key][:horizon], dtype=np.float64)[:, :3]
        if "ee_ori" in obs:
            ee_ori = np.asarray(obs["ee_ori"][:horizon], dtype=np.float64)[:, :3]
        else:
            ee_ori = np.zeros((horizon, 3), dtype=np.float64)
        if "gripper_states" in obs:
            gripper = np.asarray(obs["gripper_states"][:horizon], dtype=np.float64)
            gripper_aperture = np.mean(np.abs(gripper.reshape(horizon, -1)), axis=1)
            phase_source = "obs_gripper_states_aperture_midpoint"
        else:
            gripper_aperture = np.linspace(1.0, 0.0, horizon, dtype=np.float64)
            phase_source = "fallback_time_phase_no_gripper_state"
        instruction = str(demo.attrs.get("language", "") or _instruction_from_file(path))
        xml_text = str(demo.attrs.get("model_file", "") or "")
        free_joints, static_points, qpos_width = _parse_model_points(xml_text)
        obs_traces = _extract_obs_position_traces(obs, horizon)
        states = np.asarray(demo["states"][:horizon], dtype=np.float64) if "states" in demo else None
        joint_states = np.asarray(obs["joint_states"][:horizon], dtype=np.float64) if "joint_states" in obs else None
        state_traces, qpos_offset = _extract_state_free_traces(states, joint_states, free_joints, qpos_width, horizon)
        object_traces = {**state_traces, **obs_traces}
        source_hint, destination_hint = _split_instruction_for_geometry(instruction)
        source_name, source, source_score, source_mode = _select_named_trace(
            object_traces,
            source_hint,
            require_positive_score=True,
        )
        if source is None:
            source_name, source, source_score, source_mode = _select_named_trace(object_traces, instruction)
        destination_name, destination, dest_score, dest_mode = _select_static_destination(
            static_points=static_points,
            object_traces=object_traces,
            instruction=destination_hint,
            source_name=source_name,
            horizon=horizon,
        )
        if destination is None:
            destination_name, destination, dest_score, dest_mode = _select_static_destination(
                static_points=static_points,
                object_traces=object_traces,
                instruction=instruction,
                source_name=source_name,
                horizon=horizon,
            )
        safety_name, safety = _nearest_safety_trace(object_traces, source_name, destination_name, source)
        support = destination.copy() if destination is not None else None
        normal = None
        if destination is not None:
            normal = destination + np.asarray([0.0, 0.0, 0.08], dtype=np.float64).reshape(1, 3)
        geometry_sources = {
            "eef": f"obs/{eef_key}",
            "source": source_mode,
            "destination": dest_mode,
            "support": "destination_static_or_trace_copy" if support is not None else "unavailable",
            "safety": "nearest_non_source_object_trace" if safety is not None else "unavailable",
            "normal": "destination_plus_z_axis_proxy" if normal is not None else "unavailable",
            "qpos_offset": "n/a" if qpos_offset is None else str(qpos_offset),
        }
        leakage_audit = {
            "uses_reward_or_done_labels_for_features": False,
            "uses_eval_success_labels": False,
            "uses_task_id_or_filename_as_target_label": False,
            "uses_future_actions_as_geometry": False,
            "uses_hdf5_position_observations": bool(obs_traces),
            "uses_hdf5_state_freejoint_positions": bool(state_traces),
            "uses_xml_static_body_or_site_positions": bool(static_points),
            "source_selection": source_mode,
            "source_instruction_hint": source_hint,
            "source_instruction_overlap_score": _round(source_score, 6),
            "destination_selection": dest_mode,
            "destination_instruction_hint": destination_hint,
            "destination_instruction_overlap_score": _round(dest_score, 6),
            "phase_source": phase_source,
        }
        return DemoCase(
            file=str(path),
            demo_name=demo_name,
            instruction=instruction,
            actions=actions,
            eef_pos=eef_pos,
            ee_ori=ee_ori,
            gripper_aperture=gripper_aperture,
            source=source,
            destination=destination,
            support=support,
            safety=safety,
            normal=normal,
            source_name=source_name,
            destination_name=destination_name,
            safety_name=safety_name,
            geometry_sources=geometry_sources,
            leakage_audit=leakage_audit,
        )


def _find_hdf5_files(root: Path, max_demos: int) -> list[Path]:
    if not root.exists():
        return []
    files = sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])
    return files[:max_demos]


def _unit_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-9)


def _contact_points_for_variant(case: DemoCase, index: int, variant: str, closed: bool) -> tuple[list[np.ndarray], list[str]]:
    source = None if case.source is None else case.source[index]
    destination = None if case.destination is None else case.destination[index]
    support = None if case.support is None else case.support[index]
    safety = None if case.safety is None else case.safety[index]
    normal = None if case.normal is None else case.normal[index]

    if variant == "no_geometry_injection":
        return [], []
    if variant == "single_3d_point_injection":
        if closed and destination is not None:
            return [destination], ["destination"]
        if source is not None:
            return [source], ["source"]
        if destination is not None:
            return [destination], ["destination"]
        return [], []
    if variant == "source_object_point_only":
        return ([] if source is None else [source]), ([] if source is None else ["source"])
    if variant == "destination_placement_point_only":
        return ([] if destination is None else [destination]), ([] if destination is None else ["destination"])
    if variant == "source_destination_two_point_injection":
        points: list[np.ndarray] = []
        roles: list[str] = []
        for point, role in ((source, "source"), (destination, "destination")):
            if point is not None:
                points.append(point)
                roles.append(role)
        return points, roles
    if variant == "full_contact_set_injection":
        points = []
        roles = []
        for point, role in (
            (source, "source"),
            (destination, "destination"),
            (support, "support"),
            (safety, "safety"),
            (normal, "normal"),
        ):
            if point is not None:
                points.append(point)
                roles.append(role)
        return points, roles
    raise ValueError(f"unknown variant: {variant}")


def _set_encoder(points: list[np.ndarray], roles: list[str], eef: np.ndarray) -> np.ndarray:
    width = 3 + 3 + 1 + ROLE_WIDTH
    if not points:
        return np.zeros((width * 4) + 1, dtype=np.float64)
    rows = []
    for point, role in zip(points, roles):
        point_arr = np.asarray(point, dtype=np.float64).reshape(3)
        rel = point_arr - eef.reshape(3)
        role_vec = np.zeros(ROLE_WIDTH, dtype=np.float64)
        role_vec[ROLE_TO_ID.get(role, 0)] = 1.0
        rows.append(np.concatenate([rel, point_arr, np.asarray([np.linalg.norm(rel)], dtype=np.float64), role_vec]))
    mat = np.vstack(rows)
    return np.concatenate(
        [
            mat.mean(axis=0),
            mat.std(axis=0),
            mat.min(axis=0),
            mat.max(axis=0),
            np.asarray([len(points) / max(1, len(ROLE_TO_ID))], dtype=np.float64),
        ]
    )


def _base_features(case: DemoCase, index: int, feature_width: int) -> np.ndarray:
    horizon = max(1, case.actions.shape[0] - 1)
    phase = index / horizon
    return np.concatenate(
        [
            np.asarray(
                [
                    1.0,
                    phase,
                    math.sin(math.pi * phase),
                    math.cos(math.pi * phase),
                    case.gripper_aperture[index],
                ],
                dtype=np.float64,
            ),
            case.eef_pos[index],
            case.ee_ori[index],
            _hash_features(case.instruction, feature_width),
        ]
    )


def _closed_mask(case: DemoCase) -> np.ndarray:
    values = np.asarray(case.gripper_aperture, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = (float(values.min()) + float(values.max())) / 2.0
    return values <= threshold


def _build_design(cases: list[DemoCase], variant: str, feature_width: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        closed = _closed_mask(case)
        for step in range(case.actions.shape[0]):
            points, roles = _contact_points_for_variant(case, step, variant, bool(closed[step]))
            x = np.concatenate([_base_features(case, step, feature_width), _set_encoder(points, roles, case.eef_pos[step])])
            features.append(x)
            targets.append(case.actions[step, :7])
            rows.append(
                {
                    "case_index": case_index,
                    "step": step,
                    "closed": bool(closed[step]),
                    "source_vec": None if case.source is None else case.source[step] - case.eef_pos[step],
                    "destination_vec": None if case.destination is None else case.destination[step] - case.eef_pos[step],
                }
            )
    return np.vstack(features), np.vstack(targets), rows


def _split_rows(rows: list[dict[str, Any]], cases: list[DemoCase]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = np.zeros(len(rows), dtype=bool)
    eval_mask = np.zeros(len(rows), dtype=bool)
    cursor = 0
    split_points = []
    for case in cases:
        horizon = case.actions.shape[0]
        split = max(4, min(horizon - 2, int(round(horizon * 0.7))))
        train_mask[cursor : cursor + split] = True
        eval_mask[cursor + split : cursor + horizon] = True
        split_points.append({"file": case.file, "train_steps": split, "eval_steps": horizon - split})
        cursor += horizon
    return train_mask, eval_mask, {
        "split_type": "deterministic_per_demo_time_holdout",
        "confirmatory": False,
        "exploratory": True,
        "split_points": split_points,
    }


def _standardize_train_eval(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (eval_x - mean) / std, {"feature_count": int(train_x.shape[1])}


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, ridge: float) -> np.ndarray:
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + ridge * eye, x.T @ train_y)


def _predict(weights: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    x = np.concatenate([x_values, np.ones((x_values.shape[0], 1), dtype=np.float64)], axis=1)
    return x @ weights


def _l2(pred: np.ndarray, target: np.ndarray, slc: slice) -> float:
    return float(np.sqrt(np.mean((pred[:, slc] - target[:, slc]) ** 2)))


def _direction_score(translations: np.ndarray, vectors: np.ndarray | None) -> float | None:
    if vectors is None or len(vectors) == 0:
        return None
    t = _unit_rows(translations)
    v = _unit_rows(vectors)
    return float(np.mean(np.sum(t * v, axis=1)))


def _variant_metrics(pred: np.ndarray, target: np.ndarray, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_vectors = [row["source_vec"] for row in eval_rows if row["source_vec"] is not None and not row["closed"]]
    dest_vectors = [row["destination_vec"] for row in eval_rows if row["destination_vec"] is not None and row["closed"]]
    source_indices = [index for index, row in enumerate(eval_rows) if row["source_vec"] is not None and not row["closed"]]
    dest_indices = [index for index, row in enumerate(eval_rows) if row["destination_vec"] is not None and row["closed"]]
    source_score = _direction_score(pred[source_indices, :3], np.vstack(source_vectors) if source_vectors else None)
    dest_score = _direction_score(pred[dest_indices, :3], np.vstack(dest_vectors) if dest_vectors else None)
    expert_source_score = _direction_score(target[source_indices, :3], np.vstack(source_vectors) if source_vectors else None)
    expert_dest_score = _direction_score(target[dest_indices, :3], np.vstack(dest_vectors) if dest_vectors else None)
    contact_scores = [score for score in (source_score, dest_score) if score is not None]
    expert_contact_scores = [score for score in (expert_source_score, expert_dest_score) if score is not None]
    return {
        "action_l2": _round(_l2(pred, target, slice(0, 7)), 9),
        "translation_l2": _round(_l2(pred, target, slice(0, 3)), 9),
        "rotation_l2": _round(_l2(pred, target, slice(3, 6)), 9),
        "gripper_error": _round(float(np.mean(np.abs(pred[:, 6] - target[:, 6]))), 9),
        "target_directed_movement_proxy": _round(dest_score if dest_score is not None else source_score, 9),
        "source_consistency": _round(source_score, 9),
        "destination_consistency": _round(dest_score, 9),
        "contact_placement_consistency": _round(None if not contact_scores else float(np.mean(contact_scores)), 9),
        "expert_contact_placement_consistency": _round(None if not expert_contact_scores else float(np.mean(expert_contact_scores)), 9),
    }


def _metrics_better(left: dict[str, Any], right: dict[str, Any], key: str = "action_l2", rel: float = 0.01) -> bool:
    lval = left.get(key)
    rval = right.get(key)
    if lval is None or rval is None:
        return False
    return float(lval) < float(rval) * (1.0 - rel)


def _metrics_match_or_beat(left: dict[str, Any], right: dict[str, Any], key: str = "action_l2", rel: float = 0.01) -> bool:
    lval = left.get(key)
    rval = right.get(key)
    if lval is None or rval is None:
        return False
    return float(lval) <= float(rval) * (1.0 + rel)


def _summarize_decision(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("variants", {})
    full = variants.get("full_contact_set_injection", {}).get("metrics", {})
    single = variants.get("single_3d_point_injection", {}).get("metrics", {})
    source = variants.get("source_object_point_only", {}).get("metrics", {})
    dest = variants.get("destination_placement_point_only", {}).get("metrics", {})
    two = variants.get("source_destination_two_point_injection", {}).get("metrics", {})
    observable = report.get("data", {}).get("full_contact_set_observable", False)
    object_observable = report.get("data", {}).get("source_object_points_observable", False)
    placement_observable = report.get("data", {}).get("destination_points_observable", False)
    contact_beats_single = _metrics_better(full, single)
    simple_matches = any(_metrics_match_or_beat(metrics, full) for metrics in (source, dest, two))
    if not observable or not object_observable or not placement_observable:
        decision = "kill"
        reason = "source object, placement/support, and contact-set points are not all observable without leakage"
    elif not report.get("model", {}).get("loss_computed"):
        decision = "blocked"
        reason = "no bounded action-head loss was computed"
    elif not contact_beats_single:
        decision = "kill"
        reason = "full contact-set injection did not beat the active single-3D-point injection baseline"
    elif simple_matches:
        decision = "kill"
        reason = "a simple source-only, destination-only, or two-point baseline matched the full contact set"
    else:
        decision = "continue"
        reason = "full contact-set injection beat single-point and simple point baselines on the held-out action metric"
    return {
        "decision": decision,
        "reason": reason,
        "contact_set_beats_single_point": contact_beats_single,
        "simple_point_baseline_matches_contact_set": simple_matches,
        "single_point_action_l2": single.get("action_l2"),
        "contact_set_action_l2": full.get("action_l2"),
        "next_state": (
            "STATE 2: risk-assessed exact-init replay/progress diagnostic"
            if decision == "continue"
            else ("resolve_diagnostic_blocker" if decision == "blocked" else "archive_or_reframe_contactset_vla")
        ),
    }


def build_contactset_vla_diagnostic(
    *,
    libero_data_root: Path,
    max_demos: int = DEFAULT_MAX_DEMOS,
    max_action_steps: int = DEFAULT_MAX_ACTION_STEPS,
    feature_width: int = DEFAULT_FEATURE_WIDTH,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, Any]:
    started = time.perf_counter()
    if max_demos < 1 or max_demos > 24:
        raise ContactSetDiagnosticError("max_demos must be between 1 and 24")
    if max_action_steps < 8 or max_action_steps > 320:
        raise ContactSetDiagnosticError("max_action_steps must be between 8 and 320")
    if feature_width < 16 or feature_width > 256:
        raise ContactSetDiagnosticError("feature_width must be between 16 and 256")
    files = _find_hdf5_files(libero_data_root, max_demos=max_demos)
    cases: list[DemoCase] = []
    exclusions: list[dict[str, Any]] = []
    for path in files:
        try:
            case = _read_demo_case(path, max_action_steps=max_action_steps)
            if case is None:
                exclusions.append({"file": str(path), "reason": "missing 7D actions, obs, or EEF position"})
            else:
                cases.append(case)
        except Exception as exc:  # pragma: no cover - surfaced in real-data reports
            exclusions.append({"file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not cases:
        raise ContactSetDiagnosticError(f"no usable local LIBERO HDF5 demos found under {libero_data_root}")

    train_mask: np.ndarray | None = None
    eval_mask: np.ndarray | None = None
    split_audit: dict[str, Any] = {}
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        x, y, rows = _build_design(cases, variant, feature_width)
        if train_mask is None or eval_mask is None:
            train_mask, eval_mask, split_audit = _split_rows(rows, cases)
        train_x, eval_x, standardization = _standardize_train_eval(x[train_mask], x[eval_mask])
        train_y, eval_y = y[train_mask], y[eval_mask]
        weights = _fit_ridge(train_x, train_y, ridge=ridge)
        pred = _predict(weights, eval_x)
        eval_rows = [row for row, keep in zip(rows, eval_mask) if keep]
        train_pred = _predict(weights, train_x)
        variants[variant] = {
            "metrics": _variant_metrics(pred, eval_y, eval_rows),
            "train_action_l2": _round(_l2(train_pred, train_y, slice(0, 7)), 9),
            "feature_count": standardization["feature_count"],
        }

    case_summaries = []
    for case in cases:
        case_summaries.append(
            {
                "file": case.file,
                "demo_name": case.demo_name,
                "instruction": case.instruction,
                "steps": int(case.actions.shape[0]),
                "source_name": case.source_name,
                "destination_name": case.destination_name,
                "safety_name": case.safety_name,
                "source_observable": case.source is not None,
                "destination_observable": case.destination is not None,
                "support_observable": case.support is not None,
                "safety_observable": case.safety is not None,
                "normal_observable": case.normal is not None,
                "geometry_sources": case.geometry_sources,
                "leakage_audit": case.leakage_audit,
            }
        )
    source_observable = all(case.source is not None for case in cases)
    dest_observable = all(case.destination is not None for case in cases)
    support_observable = all(case.support is not None for case in cases)
    normal_observable = all(case.normal is not None for case in cases)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "exploratory_offline_action_head_diagnostic",
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": True,
            "tiny_cpu_numpy_training_only": True,
            "rollouts_performed": False,
            "exact_init_replay_progress_happened": False,
            "simulator_executed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
        },
        "model": {
            "name": "tiny_numpy_ridge_action_head",
            "loss_computed": True,
            "real_vla_model_metric_produced": False,
            "ridge": ridge,
        },
        "data": {
            "libero_data_root": str(libero_data_root),
            "candidate_file_count": len(files),
            "usable_demo_count": len(cases),
            "excluded_files": exclusions,
            "train_record_count": int(np.sum(train_mask)) if train_mask is not None else 0,
            "eval_record_count": int(np.sum(eval_mask)) if eval_mask is not None else 0,
            "split_audit": split_audit,
            "source_object_points_observable": source_observable,
            "destination_points_observable": dest_observable,
            "support_points_observable": support_observable,
            "normal_cues_observable": normal_observable,
            "full_contact_set_observable": bool(source_observable and dest_observable and support_observable and normal_observable),
            "uses_eval_label_leakage": False,
        },
        "cases": case_summaries,
        "variants": variants,
        "replay_progress": {
            "happened": False,
            "reason": "STATE 1 is bounded to offline action-head loss; exact-init replay is the next separate risk-assessed milestone if the offline gate continues",
        },
        "elapsed_seconds": None,
    }
    report["decision"] = _summarize_decision(report)
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _md(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return str(_round(value, 6))
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = report.get("decision", {})
    lines = [
        "# ContactSet-VLA Diagnostic Report",
        "",
        "Bounded offline action-head diagnostic only. This is not standard LIBERO success, rollout evidence, or a paper-grade claim.",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- reason: {decision.get('reason')}",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- loss computed: `{report.get('model', {}).get('loss_computed')}`",
        f"- replay/control metric happened: `{report.get('policy', {}).get('rollouts_performed')}`",
        f"- GPU/download/OpenVLA-OFT: `{report.get('policy', {}).get('gpu_jobs_performed')}` / `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}`",
        f"- usable demos: `{report.get('data', {}).get('usable_demo_count')}`",
        f"- source/destination/support observable: `{report.get('data', {}).get('source_object_points_observable')}` / `{report.get('data', {}).get('destination_points_observable')}` / `{report.get('data', {}).get('support_points_observable')}`",
        f"- single-point action L2: `{decision.get('single_point_action_l2')}`",
        f"- contact-set action L2: `{decision.get('contact_set_action_l2')}`",
        f"- contact-set beats single-point: `{decision.get('contact_set_beats_single_point')}`",
        f"- simple baselines matched contact-set: `{decision.get('simple_point_baseline_matches_contact_set')}`",
        "",
        "## Variants",
        "",
        "| variant | action L2 | translation L2 | rotation L2 | gripper error | target-directed | source consistency | destination consistency | contact/place consistency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in VARIANTS:
        metrics = (report.get("variants", {}).get(name) or {}).get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _md(metrics.get("action_l2")),
                    _md(metrics.get("translation_l2")),
                    _md(metrics.get("rotation_l2")),
                    _md(metrics.get("gripper_error")),
                    _md(metrics.get("target_directed_movement_proxy")),
                    _md(metrics.get("source_consistency")),
                    _md(metrics.get("destination_consistency")),
                    _md(metrics.get("contact_placement_consistency")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases", []):
        lines.extend(
            [
                f"- file: `{case.get('file')}`",
                f"  instruction: {case.get('instruction')}",
                f"  source/destination/safety: `{case.get('source_name')}` / `{case.get('destination_name')}` / `{case.get('safety_name')}`",
                f"  geometry sources: `{case.get('geometry_sources')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--max-demos", type=int, default=DEFAULT_MAX_DEMOS)
    parser.add_argument("--max-action-steps", type=int, default=DEFAULT_MAX_ACTION_STEPS)
    parser.add_argument("--feature-width", type=int, default=DEFAULT_FEATURE_WIDTH)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--report-json", default="reports/contactset_vla_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/contactset_vla_diagnostic_report.md")
    args = parser.parse_args(argv)
    paths = read_asset_paths(Path(args.paths_file))
    data_root = Path(args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero"))
    report = build_contactset_vla_diagnostic(
        libero_data_root=data_root,
        max_demos=args.max_demos,
        max_action_steps=args.max_action_steps,
        feature_width=args.feature_width,
        ridge=args.ridge,
    )
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    console = {
        "decision": report.get("decision"),
        "data": {
            "usable_demo_count": report.get("data", {}).get("usable_demo_count"),
            "train_record_count": report.get("data", {}).get("train_record_count"),
            "eval_record_count": report.get("data", {}).get("eval_record_count"),
            "full_contact_set_observable": report.get("data", {}).get("full_contact_set_observable"),
        },
        "reports": {"json": str(json_path), "markdown": str(md_path)},
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
