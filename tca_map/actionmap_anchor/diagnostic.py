"""Mini ActionMap-style anchor feasibility gate over local LIBERO HDF5 actions.

This diagnostic is a minimal feasibility gate, not official ActionMap
reproduction. It compares a small voxel heatmap action head against mean,
linear, and cheap MLP baselines using local HDF5 action labels. It does not
import VLA checkpoints, run simulators, use GPU, download assets, design an
extension, mine failures, or make paper-grade claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_metadata_subset import read_asset_paths

SCHEMA_VERSION = "actionmap-anchor-diagnostic-v1"
DEFAULT_MAX_DEMOS = 8
DEFAULT_MAX_ACTION_STEPS = 180
DEFAULT_FEATURE_WIDTH = 48
DEFAULT_MAX_STEPS = 120
DEFAULT_TRANS_BINS = 7
DEFAULT_ROT_BINS = 7
MAX_TRAINING_STEPS = 300


class ActionMapAnchorError(RuntimeError):
    """Raised when the bounded ActionMap anchor diagnostic cannot run safely."""


@dataclass(frozen=True)
class DemoCase:
    file: str
    demo_name: str
    task_id: str
    instruction: str
    actions: np.ndarray
    features: np.ndarray
    phases: np.ndarray


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


def _instruction_from_file(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    parts = stem.split("_")
    while parts and (parts[0].isupper() or parts[0].isdigit() or "SCENE" in parts[0]):
        parts = parts[1:]
    return " ".join(parts or stem.split("_"))


def _hash_features(text: str, width: int) -> np.ndarray:
    cleaned = " ".join(text.lower().replace("_", " ").split())
    words = cleaned.split()
    vector = np.zeros(width, dtype=np.float64)
    scalars = [
        min(len(cleaned), 240) / 240.0,
        min(len(words), 48) / 48.0,
        sum(char in "aeiou" for char in cleaned) / max(1, len(cleaned)),
        1.0 if "put" in words or "place" in words else 0.0,
        1.0 if "open" in words or "close" in words else 0.0,
        1.0 if "turn" in words else 0.0,
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
    return vector / norm if norm > 0.0 else vector


def _obs_array(obs: Any, key: str, steps: int, width: int) -> np.ndarray:
    if obs is not None and key in obs:
        arr = np.asarray(obs[key][:steps], dtype=np.float64).reshape(steps, -1)
        if arr.shape[1] >= width:
            return arr[:, :width]
    return np.zeros((steps, width), dtype=np.float64)


def _build_features(obs: Any, instruction: str, steps: int, feature_width: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, steps, dtype=np.float64).reshape(-1, 1)
    phase_features = np.concatenate(
        [phase, np.sin(math.pi * phase), np.cos(math.pi * phase), phase**2],
        axis=1,
    )
    eef = _obs_array(obs, "ee_pos", steps, 3)
    ori = _obs_array(obs, "ee_ori", steps, 3)
    joints = _obs_array(obs, "joint_states", steps, 7)
    gripper = _obs_array(obs, "gripper_states", steps, 2)
    text = np.repeat(_hash_features(instruction, feature_width).reshape(1, -1), steps, axis=0)
    return np.concatenate([phase_features, eef, ori, joints, gripper, text], axis=1)


def _read_demo_case(path: Path, max_action_steps: int, feature_width: int) -> DemoCase | None:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            return None
        demo_name = sorted(str(name) for name in data.keys())[0]
        demo = data[demo_name]
        if "actions" not in demo:
            return None
        actions = np.asarray(demo["actions"][:max_action_steps], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 7 or actions.shape[0] < 12:
            return None
        actions = np.clip(actions[:, :7], -1.0, 1.0)
        instruction = str(demo.attrs.get("language", "") or _instruction_from_file(path))
        obs = demo.get("obs")
        features = _build_features(obs, instruction, actions.shape[0], feature_width)
        phases = np.linspace(0.0, 1.0, actions.shape[0], dtype=np.float64)
        task_id = path.stem[:-5] if path.stem.endswith("_demo") else path.stem
        return DemoCase(
            file=str(path),
            demo_name=demo_name,
            task_id=task_id,
            instruction=instruction,
            actions=actions,
            features=features,
            phases=phases,
        )


def _find_hdf5_files(root: Path, max_demos: int) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])[:max_demos]


def _load_cases(libero_data_root: Path, max_demos: int, max_action_steps: int, feature_width: int) -> tuple[list[DemoCase], list[dict[str, str]]]:
    cases: list[DemoCase] = []
    exclusions: list[dict[str, str]] = []
    for path in _find_hdf5_files(libero_data_root, max_demos):
        try:
            case = _read_demo_case(path, max_action_steps=max_action_steps, feature_width=feature_width)
            if case is None:
                exclusions.append({"file": str(path), "reason": "missing usable 7D action demo"})
            else:
                cases.append(case)
        except Exception as exc:  # pragma: no cover - real data path
            exclusions.append({"file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return cases, exclusions


def _stack_cases(cases: list[DemoCase]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    train_mask_parts: list[np.ndarray] = []
    eval_mask_parts: list[np.ndarray] = []
    split_points = []
    cursor = 0
    for case_index, case in enumerate(cases):
        n = case.actions.shape[0]
        split = max(8, min(n - 4, int(round(n * 0.7))))
        train = np.zeros(n, dtype=bool)
        eval_mask = np.zeros(n, dtype=bool)
        train[:split] = True
        eval_mask[split:] = True
        train_mask_parts.append(train)
        eval_mask_parts.append(eval_mask)
        features.append(case.features)
        actions.append(case.actions)
        for step in range(n):
            phase = "early" if case.phases[step] < 0.33 else ("mid" if case.phases[step] < 0.66 else "late")
            rows.append({"case_index": case_index, "task_id": case.task_id, "step": step, "phase": phase})
        split_points.append({"file": case.file, "task_id": case.task_id, "train_steps": split, "eval_steps": n - split})
        cursor += n
    return (
        np.vstack(features),
        np.vstack(actions),
        rows,
        np.concatenate(train_mask_parts),
        np.concatenate(eval_mask_parts),
        {
            "split_type": "deterministic_per_demo_time_holdout",
            "exploratory": True,
            "confirmatory": False,
            "eval_action_label_leakage_detected": False,
            "split_points": split_points,
            "record_count": cursor,
        },
    )


def _standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (eval_x - mean) / std, {"feature_count": int(train_x.shape[1])}


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    xb = _with_bias(x)
    eye = np.eye(xb.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ y)


def _predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.clip(_with_bias(x) @ weights, -1.0, 1.0)


def _train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    steps: int,
    lr: float,
    hidden: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, list[float], int]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.05, size=(train_x.shape[1], hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(0.0, 0.05, size=(hidden, train_y.shape[1]))
    b2 = np.zeros(train_y.shape[1], dtype=np.float64)
    losses: list[float] = []
    batch = min(96, train_x.shape[0])
    for step in range(steps + 1):
        h = np.tanh(train_x @ w1 + b1)
        pred = np.clip(h @ w2 + b2, -1.0, 1.0)
        losses.append(float(np.mean((pred - train_y) ** 2)))
        if step == steps:
            break
        indices = np.arange((step * batch) % train_x.shape[0], ((step * batch) % train_x.shape[0]) + batch) % train_x.shape[0]
        xb = train_x[indices]
        yb = train_y[indices]
        hb = np.tanh(xb @ w1 + b1)
        raw = hb @ w2 + b2
        diff = np.clip(raw, -1.0, 1.0) - yb
        grad_raw = (2.0 / max(1, diff.size)) * diff
        grad_w2 = hb.T @ grad_raw
        grad_b2 = grad_raw.sum(axis=0)
        grad_h = grad_raw @ w2.T
        grad_z = grad_h * (1.0 - hb**2)
        grad_w1 = xb.T @ grad_z
        grad_b1 = grad_z.sum(axis=0)
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
    eval_pred = np.clip(np.tanh(eval_x @ w1 + b1) @ w2 + b2, -1.0, 1.0)
    param_count = int(w1.size + b1.size + w2.size + b2.size)
    return eval_pred, losses, param_count


def _make_grid(bins: int) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, bins, dtype=np.float64)
    x, y, z = np.meshgrid(values, values, values, indexing="ij")
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _nearest_grid_labels(actions: np.ndarray, centers: np.ndarray, slc: slice) -> np.ndarray:
    part = actions[:, slc]
    distances = np.sum((part[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def _gripper_labels(actions: np.ndarray) -> np.ndarray:
    return (actions[:, 6] >= 0.0).astype(np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(labels.shape[0]), labels] + 1e-12)))


def _train_softmax_classifier(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    eval_x: np.ndarray,
    *,
    classes: int,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    xb = _with_bias(train_x)
    eb = _with_bias(eval_x)
    weights = np.zeros((xb.shape[1], classes), dtype=np.float64)
    losses: list[float] = []
    one_hot = np.zeros((train_labels.shape[0], classes), dtype=np.float64)
    one_hot[np.arange(train_labels.shape[0]), train_labels] = 1.0
    for step in range(steps + 1):
        probs = _softmax(xb @ weights)
        losses.append(_cross_entropy(probs, train_labels))
        if step == steps:
            break
        grad = xb.T @ (probs - one_hot) / train_labels.shape[0]
        weights -= lr * grad
    return weights, _softmax(eb @ weights), losses


def _topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = max(1, min(k, probs.shape[1]))
    top = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == labels.reshape(-1, 1), axis=1)))


def _entropy(probs: np.ndarray) -> float:
    return float(np.mean(-np.sum(probs * np.log(probs + 1e-12), axis=1)))


def _actionmap_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    *,
    trans_bins: int,
    rot_bins: int,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, dict[str, Any], list[float], int]:
    trans_centers = _make_grid(trans_bins)
    rot_centers = _make_grid(rot_bins)
    train_trans = _nearest_grid_labels(train_y, trans_centers, slice(0, 3))
    train_rot = _nearest_grid_labels(train_y, rot_centers, slice(3, 6))
    train_grip = _gripper_labels(train_y)
    eval_trans = _nearest_grid_labels(eval_y, trans_centers, slice(0, 3))
    eval_rot = _nearest_grid_labels(eval_y, rot_centers, slice(3, 6))
    eval_grip = _gripper_labels(eval_y)
    trans_w, trans_probs, trans_losses = _train_softmax_classifier(
        train_x,
        train_trans,
        eval_x,
        classes=trans_centers.shape[0],
        steps=steps,
        lr=lr,
    )
    rot_w, rot_probs, rot_losses = _train_softmax_classifier(
        train_x,
        train_rot,
        eval_x,
        classes=rot_centers.shape[0],
        steps=steps,
        lr=lr,
    )
    grip_w, grip_probs, grip_losses = _train_softmax_classifier(
        train_x,
        train_grip,
        eval_x,
        classes=2,
        steps=steps,
        lr=lr,
    )
    trans_pred = np.argmax(trans_probs, axis=1)
    rot_pred = np.argmax(rot_probs, axis=1)
    grip_pred = np.argmax(grip_probs, axis=1)
    pred = np.concatenate(
        [
            trans_centers[trans_pred],
            rot_centers[rot_pred],
            np.where(grip_pred.reshape(-1, 1) > 0, 1.0, -1.0),
        ],
        axis=1,
    )
    losses = [
        float(trans_losses[index] + rot_losses[index] + grip_losses[index])
        for index in range(min(len(trans_losses), len(rot_losses), len(grip_losses)))
    ]
    diagnostics = {
        "trans_grid": [trans_bins, trans_bins, trans_bins],
        "rot_grid": [rot_bins, rot_bins, rot_bins],
        "candidate_count": int(trans_centers.shape[0] + rot_centers.shape[0] + 2),
        "candidate_top1_accuracy": _round(float(np.mean((trans_pred == eval_trans) & (rot_pred == eval_rot) & (grip_pred == eval_grip))), 9),
        "translation_top1_accuracy": _round(float(np.mean(trans_pred == eval_trans)), 9),
        "translation_top3_accuracy": _round(_topk_accuracy(trans_probs, eval_trans, 3), 9),
        "rotation_top1_accuracy": _round(float(np.mean(rot_pred == eval_rot)), 9),
        "rotation_top3_accuracy": _round(_topk_accuracy(rot_probs, eval_rot, 3), 9),
        "gripper_top1_accuracy": _round(float(np.mean(grip_pred == eval_grip)), 9),
        "heatmap_nll": _round(float(_cross_entropy(trans_probs, eval_trans) + _cross_entropy(rot_probs, eval_rot) + _cross_entropy(grip_probs, eval_grip)), 9),
        "translation_entropy": _round(_entropy(trans_probs), 9),
        "rotation_entropy": _round(_entropy(rot_probs), 9),
        "unique_translation_bins": int(np.unique(trans_pred).size),
        "unique_rotation_bins": int(np.unique(rot_pred).size),
        "unique_gripper_bins": int(np.unique(grip_pred).size),
    }
    param_count = int(trans_w.size + rot_w.size + grip_w.size)
    return pred, diagnostics, losses, param_count


def _oracle_candidate(eval_y: np.ndarray, trans_bins: int, rot_bins: int) -> tuple[np.ndarray, dict[str, Any]]:
    trans_centers = _make_grid(trans_bins)
    rot_centers = _make_grid(rot_bins)
    trans_labels = _nearest_grid_labels(eval_y, trans_centers, slice(0, 3))
    rot_labels = _nearest_grid_labels(eval_y, rot_centers, slice(3, 6))
    grip = _gripper_labels(eval_y)
    pred = np.concatenate(
        [
            trans_centers[trans_labels],
            rot_centers[rot_labels],
            np.where(grip.reshape(-1, 1) > 0, 1.0, -1.0),
        ],
        axis=1,
    )
    return pred, {
        "oracle_type": "nearest_actionmap_grid_candidate_upper_bound",
        "invalid_as_method_evidence": True,
        "candidate_top1_accuracy": 1.0,
    }


def _l2(pred: np.ndarray, target: np.ndarray, slc: slice) -> float:
    return float(np.sqrt(np.mean((pred[:, slc] - target[:, slc]) ** 2)))


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    return {
        "action_l2": _round(_l2(pred, target, slice(0, 7)), 9),
        "translation_l2": _round(_l2(pred, target, slice(0, 3)), 9),
        "rotation_l2": _round(_l2(pred, target, slice(3, 6)), 9),
        "gripper_error": _round(float(np.mean(np.abs(pred[:, 6] - target[:, 6]))), 9),
        "action_l1": _round(float(np.mean(np.abs(pred - target))), 9),
    }


def _loss_curve(losses: list[float], max_points: int = 12) -> list[dict[str, Any]]:
    if not losses:
        return []
    if len(losses) <= max_points:
        indices = list(range(len(losses)))
    else:
        indices = sorted({int(round(value)) for value in np.linspace(0, len(losses) - 1, max_points)})
    return [{"step": int(index), "loss": _round(losses[index], 9)} for index in indices]


def _breakdown(pred: np.ndarray, target: np.ndarray, eval_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(eval_rows):
        groups.setdefault(str(row[key]), []).append(index)
    result = {}
    for name, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        result[name] = {"count": int(idx.size), **_metrics(pred[idx], target[idx])}
    return result


def _variant_report(
    *,
    name: str,
    pred: np.ndarray,
    target: np.ndarray,
    train_losses: list[float] | None,
    trainable_params: int,
    extra: dict[str, Any] | None = None,
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "metrics": _metrics(pred, target),
        "loss_curve": _loss_curve(train_losses or []),
        "initial_loss": None if not train_losses else _round(train_losses[0], 9),
        "final_loss": None if not train_losses else _round(train_losses[-1], 9),
        "trainable_params": trainable_params,
        "per_task": _breakdown(pred, target, eval_rows, "task_id"),
        "per_phase": _breakdown(pred, target, eval_rows, "phase"),
    }
    if extra:
        payload.update(extra)
    return payload


def _beats(left: dict[str, Any], right: dict[str, Any], key: str = "action_l2", rel: float = 0.01) -> bool:
    lval = (left.get("metrics") or {}).get(key)
    rval = (right.get("metrics") or {}).get(key)
    if lval is None or rval is None:
        return False
    return float(lval) < float(rval) * (1.0 - rel)


def _matches_or_beats(left: dict[str, Any], right: dict[str, Any], key: str = "action_l2", rel: float = 0.01) -> bool:
    lval = (left.get("metrics") or {}).get(key)
    rval = (right.get("metrics") or {}).get(key)
    if lval is None or rval is None:
        return False
    return float(lval) <= float(rval) * (1.0 + rel)


def _decision(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("variants", {})
    mean = variants.get("mean_action_baseline", {})
    linear = variants.get("linear_l1_action_head", {})
    mlp = variants.get("simple_mlp_action_head", {})
    actionmap = variants.get("actionmap_heatmap_candidate_head", {})
    oracle = variants.get("oracle_nearest_action_candidate_upper_bound", {})
    actionmap_beats_mean = _beats(actionmap, mean)
    actionmap_beats_linear = _beats(actionmap, linear)
    mlp_matches = _matches_or_beats(mlp, actionmap)
    oracle_headroom = _beats(oracle, actionmap, rel=0.05)
    diversity = actionmap.get("candidate_diagnostics", {})
    collapse = bool(
        int(diversity.get("unique_translation_bins") or 0) <= 1
        or int(diversity.get("unique_rotation_bins") or 0) <= 1
    )
    triggered: list[str] = []
    if not report.get("data", {}).get("real_hdf5_metric_produced"):
        decision = "blocked"
        triggered.append("no real LIBERO/HDF5-backed metric appeared")
    elif report.get("model", {}).get("oracle_gate_passed") is False:
        triggered.append("oracle nearest candidate upper bound does not clearly beat mean-action and linear/L1 baselines")
        decision = "kill"
    else:
        if not oracle_headroom:
            triggered.append("oracle nearest candidate upper bound has too little headroom over the learned ActionMap-style head")
        if collapse:
            triggered.append("ActionMap-style head collapsed to too few action candidates")
        if not actionmap_beats_mean:
            triggered.append("mean-action baseline matches or beats the ActionMap-style heatmap head")
        if not actionmap_beats_linear:
            triggered.append("linear/L1 action head matches or beats the ActionMap-style heatmap head")
        if mlp_matches:
            triggered.append("cheap MLP action head matches or beats the ActionMap-style heatmap head")
        decision = "kill" if triggered else "continue"
    reason = (
        "; ".join(triggered)
        if triggered
        else "ActionMap-style heatmap head beats mean and linear baselines without candidate collapse"
    )
    return {
        "decision": decision,
        "reason": reason,
        "triggered_kill_criteria": triggered,
        "actionmap_beats_mean": actionmap_beats_mean,
        "actionmap_beats_linear_l1": actionmap_beats_linear,
        "simple_mlp_matches_or_beats_actionmap": mlp_matches,
        "oracle_candidate_has_headroom": oracle_headroom,
        "candidate_collapse_detected": collapse,
        "mean_action_l2": (mean.get("metrics") or {}).get("action_l2"),
        "linear_l1_action_l2": (linear.get("metrics") or {}).get("action_l2"),
        "simple_mlp_action_l2": (mlp.get("metrics") or {}).get("action_l2"),
        "actionmap_action_l2": (actionmap.get("metrics") or {}).get("action_l2"),
        "oracle_candidate_action_l2": (oracle.get("metrics") or {}).get("action_l2"),
        "next_state": "STATE 2 failure mining" if decision == "continue" else "kill_or_reframe_anchor_reproduction",
    }


def _final_decision(report: dict[str, Any]) -> str:
    """Map the local diagnostic outcome to the user-facing hard gate labels."""

    if not report.get("data", {}).get("real_hdf5_metric_produced"):
        return "NO_REAL_METRIC"
    policy = report.get("policy", {})
    if policy.get("downloads_performed") or policy.get("gpu_jobs_performed") or policy.get("openvla_oft_executed"):
        return "TOO_HEAVY_LOCAL"
    if policy.get("official_actionmap_reproduction_attempted"):
        return "NEED_OFFICIAL_ACTIONMAP_REPRO"
    decision = report.get("decision", {}).get("decision")
    if decision == "continue":
        return "GO_TARGET_GROUNDED_ACTIONMAP_STATE1"
    return "KILL_ACTIONMAP_ANCHOR"


def _exact_next_step(final_decision: str) -> str:
    if final_decision == "GO_TARGET_GROUNDED_ACTIONMAP_STATE1":
        return "Start Target-Grounded ActionMap STATE 1 feasibility-only planning; do not train a large VLA."
    if final_decision == "NEED_OFFICIAL_ACTIONMAP_REPRO":
        return "Stop local extension work and plan an official ActionMap reproduction/source gate."
    if final_decision == "TOO_HEAVY_LOCAL":
        return "Stop; document the local source/compute blocker before any further ActionMap work."
    if final_decision == "NO_REAL_METRIC":
        return "Stop; resolve local LIBERO/HDF5 access before any method discussion."
    return "Stop; do not proceed to Target-Grounded ActionMap from this mini-anchor result."


def build_actionmap_anchor_diagnostic(
    *,
    libero_data_root: Path,
    max_demos: int = DEFAULT_MAX_DEMOS,
    max_action_steps: int = DEFAULT_MAX_ACTION_STEPS,
    feature_width: int = DEFAULT_FEATURE_WIDTH,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 0.18,
    trans_bins: int = DEFAULT_TRANS_BINS,
    rot_bins: int = DEFAULT_ROT_BINS,
) -> dict[str, Any]:
    started = time.perf_counter()
    if max_demos < 1 or max_demos > 32:
        raise ActionMapAnchorError("max_demos must be between 1 and 32")
    if max_action_steps < 12 or max_action_steps > 320:
        raise ActionMapAnchorError("max_action_steps must be between 12 and 320")
    if feature_width < 16 or feature_width > 256:
        raise ActionMapAnchorError("feature_width must be between 16 and 256")
    if max_steps < 1 or max_steps > MAX_TRAINING_STEPS:
        raise ActionMapAnchorError(f"max_steps must be between 1 and {MAX_TRAINING_STEPS}")
    if trans_bins < 3 or trans_bins > 11 or rot_bins < 3 or rot_bins > 11:
        raise ActionMapAnchorError("trans_bins and rot_bins must be between 3 and 11")
    cases, exclusions = _load_cases(libero_data_root, max_demos, max_action_steps, feature_width)
    if not cases:
        raise ActionMapAnchorError(f"no usable local LIBERO HDF5 demos found under {libero_data_root}")
    x, y, rows, train_mask, eval_mask, split_audit = _stack_cases(cases)
    train_x, eval_x, standardization = _standardize(x[train_mask], x[eval_mask])
    train_y, eval_y = y[train_mask], y[eval_mask]
    eval_rows = [row for row, keep in zip(rows, eval_mask) if keep]

    mean_action = np.mean(train_y, axis=0, keepdims=True)
    mean_pred = np.repeat(mean_action, eval_y.shape[0], axis=0)
    oracle_pred, oracle_extra = _oracle_candidate(eval_y, trans_bins=trans_bins, rot_bins=rot_bins)
    linear_weights = _fit_ridge(train_x, train_y)
    linear_pred = _predict_ridge(linear_weights, eval_x)
    train_linear_pred = _predict_ridge(linear_weights, train_x)
    linear_losses = [float(np.mean((train_linear_pred - train_y) ** 2))]
    mean_variant = _variant_report(
            name="mean_action_baseline",
            pred=mean_pred,
            target=eval_y,
            train_losses=[],
            trainable_params=0,
            eval_rows=eval_rows,
    )
    linear_variant = _variant_report(
            name="linear_l1_action_head",
            pred=linear_pred,
            target=eval_y,
            train_losses=linear_losses,
            trainable_params=int(linear_weights.size),
            eval_rows=eval_rows,
    )
    oracle_variant = _variant_report(
            name="oracle_nearest_action_candidate_upper_bound",
            pred=oracle_pred,
            target=eval_y,
            train_losses=[],
            trainable_params=0,
            extra={"candidate_diagnostics": oracle_extra},
            eval_rows=eval_rows,
    )
    oracle_gate_passed = _beats(oracle_variant, mean_variant, rel=0.05) and _beats(oracle_variant, linear_variant, rel=0.05)

    variants = {
        "mean_action_baseline": mean_variant,
        "linear_l1_action_head": linear_variant,
        "oracle_nearest_action_candidate_upper_bound": oracle_variant,
    }
    actionmap_head_fit_performed = False
    simple_mlp_fit_performed = False
    if oracle_gate_passed:
        actionmap_pred, candidate_diagnostics, actionmap_losses, actionmap_params = _actionmap_head(
            train_x,
            train_y,
            eval_x,
            eval_y,
            trans_bins=trans_bins,
            rot_bins=rot_bins,
            steps=max_steps,
            lr=learning_rate,
        )
        actionmap_head_fit_performed = True
        mlp_pred, mlp_losses, mlp_params = _train_mlp(
            train_x,
            train_y,
            eval_x,
            steps=max_steps,
            lr=learning_rate * 0.35,
        )
        simple_mlp_fit_performed = True
        variants["simple_mlp_action_head"] = _variant_report(
            name="simple_mlp_action_head",
            pred=mlp_pred,
            target=eval_y,
            train_losses=mlp_losses,
            trainable_params=mlp_params,
            eval_rows=eval_rows,
        )
        variants["actionmap_heatmap_candidate_head"] = _variant_report(
            name="actionmap_heatmap_candidate_head",
            pred=actionmap_pred,
            target=eval_y,
            train_losses=actionmap_losses,
            trainable_params=actionmap_params,
            extra={"candidate_diagnostics": candidate_diagnostics},
            eval_rows=eval_rows,
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "mini_actionmap_anchor_feasibility_gate",
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": True,
            "tiny_cpu_numpy_training_only": True,
            "rollouts_performed": False,
            "simulator_executed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "openvla_oft_executed": False,
            "official_actionmap_reproduction_attempted": False,
            "new_extension_implemented": False,
            "failure_mining_performed": False,
            "paper_grade_claims_made": False,
        },
        "anchor": {
            "paper": "ActionMap: Robot Policy Learning via Voxel Action Heatmap",
            "arxiv": "https://arxiv.org/abs/2606.06904",
            "official_code_preview": "https://github.com/showlab/ActionMap",
            "approximation": "local CPU NumPy translation/rotation/gripper heatmap classifiers over HDF5 observations, not a VLA reproduction",
        },
        "data": {
            "libero_data_root": str(libero_data_root),
            "usable_demo_count": len(cases),
            "excluded_files": exclusions,
            "train_record_count": int(np.sum(train_mask)),
            "eval_record_count": int(np.sum(eval_mask)),
            "split_audit": split_audit,
            "real_hdf5_metric_produced": True,
            "eval_label_leakage_detected": False,
        },
        "model": {
            "loss_computed": True,
            "feature_count": standardization["feature_count"],
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "oracle_gate_passed": oracle_gate_passed,
            "oracle_gate_rule": "oracle nearest candidate action L2 must beat mean-action and linear/L1 action L2 by at least 5 percent before fitting the tiny ActionMap-style head",
            "actionmap_head_fit_performed": actionmap_head_fit_performed,
            "simple_mlp_fit_performed": simple_mlp_fit_performed,
            "execution_order": [
                "load local HDF5 action labels",
                "compute mean-action baseline",
                "compute oracle nearest candidate upper bound",
                "fit linear/L1 baseline for oracle-headroom comparison",
                "fit tiny ActionMap-style head only if oracle gate passes",
                "fit cheap MLP baseline only inside the passed oracle gate",
                "stop after STATE 1 decision",
            ],
            "real_vla_model_metric_produced": False,
        },
        "cases": [
            {
                "file": case.file,
                "demo_name": case.demo_name,
                "task_id": case.task_id,
                "instruction": case.instruction,
                "steps": int(case.actions.shape[0]),
            }
            for case in cases
        ],
        "variants": variants,
        "replay_progress": {
            "happened": False,
            "reason": "STATE 1 first gate is local HDF5 action-head quality; exact-init replay is only considered if reproduction passes",
        },
        "elapsed_seconds": None,
    }
    report["decision"] = _decision(report)
    report["final_decision"] = _final_decision(report)
    report["exact_next_step"] = _exact_next_step(str(report["final_decision"]))
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
    final_decision = report.get("final_decision")
    lines = [
        "# ActionMap Mini-Anchor STATE 1 Result",
        "",
        "Bounded local HDF5 action-head diagnostic only. This is not a full VLA reproduction, standard LIBERO success, rollout evidence, or a paper-grade claim.",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- final decision: `{final_decision}`",
        f"- reason: {decision.get('reason')}",
        f"- training happened: `{report.get('policy', {}).get('training_performed')}`",
        f"- loss computed: `{report.get('model', {}).get('loss_computed')}`",
        f"- replay/control happened: `{report.get('policy', {}).get('rollouts_performed')}`",
        f"- GPU/download/OpenVLA-OFT: `{report.get('policy', {}).get('gpu_jobs_performed')}` / `{report.get('policy', {}).get('downloads_performed')}` / `{report.get('policy', {}).get('openvla_oft_executed')}`",
        f"- official ActionMap reproduction / extension / failure mining: `{report.get('policy', {}).get('official_actionmap_reproduction_attempted')}` / `{report.get('policy', {}).get('new_extension_implemented')}` / `{report.get('policy', {}).get('failure_mining_performed')}`",
        f"- dataset/split: `{report.get('data', {}).get('usable_demo_count')}` demos, `{(report.get('data', {}).get('split_audit') or {}).get('split_type')}`",
        f"- oracle gate passed: `{report.get('model', {}).get('oracle_gate_passed')}`",
        f"- mean-action action L2: `{decision.get('mean_action_l2')}`",
        f"- linear/L1 action L2: `{decision.get('linear_l1_action_l2')}`",
        f"- simple MLP action L2: `{decision.get('simple_mlp_action_l2')}`",
        f"- ActionMap-style action L2: `{decision.get('actionmap_action_l2')}`",
        f"- oracle candidate action L2: `{decision.get('oracle_candidate_action_l2')}`",
        f"- ActionMap beats mean/linear: `{decision.get('actionmap_beats_mean')}` / `{decision.get('actionmap_beats_linear_l1')}`",
        f"- simple MLP matches or beats ActionMap: `{decision.get('simple_mlp_matches_or_beats_actionmap')}`",
        f"- next state: `{decision.get('next_state')}`",
        f"- exact next step: {report.get('exact_next_step')}",
        "",
        "Triggered kill criteria:",
        "",
        *[f"- {item}" for item in decision.get("triggered_kill_criteria", [])],
        "",
        "## Variants",
        "",
        "| variant | action L2 | translation L2 | rotation L2 | gripper error | action L1 | top-k / NLL | collapse notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, payload in report.get("variants", {}).items():
        metrics = payload.get("metrics") or {}
        diag = payload.get("candidate_diagnostics") or {}
        topk = "n/a"
        collapse = "n/a"
        if name == "actionmap_heatmap_candidate_head":
            topk = f"top1={_md(diag.get('candidate_top1_accuracy'))}, trans@3={_md(diag.get('translation_top3_accuracy'))}, rot@3={_md(diag.get('rotation_top3_accuracy'))}, nll={_md(diag.get('heatmap_nll'))}"
            collapse = f"uniq trans/rot/grip={diag.get('unique_translation_bins')}/{diag.get('unique_rotation_bins')}/{diag.get('unique_gripper_bins')}"
        elif name == "oracle_nearest_action_candidate_upper_bound":
            topk = "oracle upper bound"
            collapse = "invalid as method evidence"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _md(metrics.get("action_l2")),
                    _md(metrics.get("translation_l2")),
                    _md(metrics.get("rotation_l2")),
                    _md(metrics.get("gripper_error")),
                    _md(metrics.get("action_l1")),
                    topk,
                    collapse,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- The ActionMap-style head is a local CPU approximation of the voxel heatmap decoder idea, not the official VLA training recipe.",
            "- The oracle candidate row is a discretization upper bound and is invalid as method evidence.",
            "- Failed or weak reproduction should be reported before any new extension is proposed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--max-demos", type=int, default=DEFAULT_MAX_DEMOS)
    parser.add_argument("--max-action-steps", type=int, default=DEFAULT_MAX_ACTION_STEPS)
    parser.add_argument("--feature-width", type=int, default=DEFAULT_FEATURE_WIDTH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--trans-bins", type=int, default=DEFAULT_TRANS_BINS)
    parser.add_argument("--rot-bins", type=int, default=DEFAULT_ROT_BINS)
    parser.add_argument("--report-json", default="reports/actionmap_anchor_state1_result.json")
    parser.add_argument("--report-md", default="reports/actionmap_anchor_state1_result.md")
    args = parser.parse_args(argv)
    paths = read_asset_paths(Path(args.paths_file))
    data_root = Path(args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero"))
    report = build_actionmap_anchor_diagnostic(
        libero_data_root=data_root,
        max_demos=args.max_demos,
        max_action_steps=args.max_action_steps,
        feature_width=args.feature_width,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        trans_bins=args.trans_bins,
        rot_bins=args.rot_bins,
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
        },
        "reports": {"json": str(json_path), "markdown": str(md_path)},
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
