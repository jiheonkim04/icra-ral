"""Small PRISM-VLA paraphrase robustness diagnostic.

This module runs a CPU-only exploratory proxy. It uses local LIBERO task/demo
metadata when available and optionally consumes the official LIBERO-Para
metadata CSV. It does not import VLA checkpoints, simulators, or run rollouts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tca_map.datasets.libero_metadata_subset import DEFAULT_SUITES, discover_bddl_tasks, read_asset_paths

SCHEMA_VERSION = "prism-vla-paraphrase-diagnostic-v0"
DEFAULT_METADATA_CSV = "C:/assets/data/libero_para/libero_para_metadata.csv"
DEFAULT_MAX_STEPS = 160
MAX_TRAINING_STEPS = 300
DEFAULT_FEATURE_WIDTH = 96
DEFAULT_MAX_ACTION_STEPS = 8
DEFAULT_MAX_TASKS = 5
DEFAULT_MAX_PARAPHRASES_PER_TASK = 18


class PrismDiagnosticError(RuntimeError):
    """Raised when the bounded PRISM-VLA diagnostic cannot run safely."""


@dataclass(frozen=True)
class TaskRecord:
    label: int
    task_id: str
    suite: str
    instruction: str
    demo_file: str
    action_chunk: np.ndarray
    canonical_objects: tuple[str, ...]


@dataclass(frozen=True)
class TextExample:
    text: str
    label: int
    source: str
    original_instruction: str
    high: str = "clean"
    mid: str = "clean"
    low: str = "clean"
    structural_similarity: float = 1.0
    keyword_similarity: float = 1.0
    difficulty: float = 0.0


@dataclass(frozen=True)
class VariantSpec:
    name: str
    transform: str
    supervised_examples: tuple[TextExample, ...]
    same_pairs: tuple[tuple[TextExample, TextExample], ...] = ()
    counterfactual_pairs: tuple[tuple[TextExample, TextExample], ...] = ()
    lambda_consistency: float = 0.0
    lambda_counterfactual: float = 0.0
    difficulty_weighting: bool = False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()


_CANONICAL_REPLACEMENTS = {
    "hob": "stove",
    "cooktop": "stove",
    "range": "stove",
    "burner": "stove",
    "dish": "bowl",
    "basin": "bowl",
    "container": "bowl",
    "vessel": "bowl",
    "flask": "bottle",
    "bottle of wine": "wine bottle",
    "shelf": "rack",
    "stand": "rack",
    "move": "put",
    "set": "put",
    "place": "put",
    "position": "put",
    "push": "push",
    "activate": "turn on",
    "switch on": "turn on",
    "start": "turn on",
}


def _canonicalize_text(text: str) -> str:
    normalized = _normalize_text(text)
    for source, target in sorted(_CANONICAL_REPLACEMENTS.items(), key=lambda item: -len(item[0])):
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _hash_index(token: str, width: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & 1 else -1.0
    return value % width, sign


def _text_features(text: str, width: int = DEFAULT_FEATURE_WIDTH, transform: str = "raw") -> np.ndarray:
    working = _canonicalize_text(text) if transform == "canonical" else _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", working)
    vector = np.zeros(width, dtype=np.float64)
    scalars = [
        min(len(working), 180) / 180.0,
        min(len(tokens), 40) / 40.0,
        sum(char in "aeiou" for char in working) / max(1, len(working)),
        1.0 if "stove" in working else 0.0,
        1.0 if "bowl" in working else 0.0,
        1.0 if "plate" in working else 0.0,
        1.0 if "rack" in working or "cabinet" in working else 0.0,
        1.0 if "put" in working or "place" in working or "set" in working else 0.0,
    ]
    vector[: len(scalars)] = scalars
    for token in tokens:
        index, sign = _hash_index("tok:" + token, width - len(scalars))
        vector[len(scalars) + index] += sign
    for left, right in zip(tokens, tokens[1:]):
        index, sign = _hash_index("bigram:" + left + ":" + right, width - len(scalars))
        vector[len(scalars) + index] += 0.5 * sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _softmax_backward(probs: np.ndarray, grad_probs: np.ndarray) -> np.ndarray:
    return probs * (grad_probs - float(np.dot(probs, grad_probs)))


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.asarray([1.0], dtype=np.float64)])


def _one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=np.float64)
    out[max(0, min(width - 1, int(index)))] = 1.0
    return out


def _predict(weights: np.ndarray, features: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(_with_bias(features) @ weights)
    return probs, probs @ actions


def _action_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _distribution_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _task_distance_scale(actions: np.ndarray) -> float:
    distances: list[float] = []
    for left in range(actions.shape[0]):
        for right in range(left + 1, actions.shape[0]):
            distances.append(_action_l2(actions[left], actions[right]))
    if not distances:
        return 1.0
    return max(1e-6, float(np.median(distances)))


def _supervised_grad(
    weights: np.ndarray,
    example: TextExample,
    actions: np.ndarray,
    feature_fn: Callable[[str], np.ndarray],
    action_loss_weight: float = 0.35,
) -> tuple[np.ndarray, float]:
    x = feature_fn(example.text)
    probs, pred_action = _predict(weights, x, actions)
    label = example.label
    target = _one_hot(label, actions.shape[0])
    ce = -math.log(float(probs[label]) + 1e-12)
    diff = pred_action - actions[label]
    action_loss = float(np.mean(diff**2))
    grad_logits = probs - target
    grad_probs_action = (2.0 / actions.shape[1]) * (actions @ diff)
    grad_logits += action_loss_weight * _softmax_backward(probs, grad_probs_action)
    grad = np.outer(_with_bias(x), grad_logits)
    return grad, float(ce + action_loss_weight * action_loss)


def _same_pair_grad(
    weights: np.ndarray,
    left: TextExample,
    right: TextExample,
    actions: np.ndarray,
    feature_fn: Callable[[str], np.ndarray],
    action_loss_weight: float,
    difficulty_weight: float,
) -> tuple[np.ndarray, float]:
    x_left = feature_fn(left.text)
    x_right = feature_fn(right.text)
    p_left, a_left = _predict(weights, x_left, actions)
    p_right, a_right = _predict(weights, x_right, actions)
    diff_p = p_left - p_right
    diff_a = a_left - a_right
    loss = float(np.mean(diff_p**2) + action_loss_weight * np.mean(diff_a**2))
    grad_p_left = (2.0 / p_left.shape[0]) * diff_p
    grad_p_right = -grad_p_left
    grad_a = (2.0 / actions.shape[1]) * diff_a
    grad_p_left += action_loss_weight * (actions @ grad_a)
    grad_p_right -= action_loss_weight * (actions @ grad_a)
    grad_left = _softmax_backward(p_left, grad_p_left)
    grad_right = _softmax_backward(p_right, grad_p_right)
    grad = np.outer(_with_bias(x_left), grad_left) + np.outer(_with_bias(x_right), grad_right)
    return difficulty_weight * grad, difficulty_weight * loss


def _counterfactual_grad(
    weights: np.ndarray,
    left: TextExample,
    right: TextExample,
    actions: np.ndarray,
    feature_fn: Callable[[str], np.ndarray],
    margin: float,
    action_loss_weight: float,
) -> tuple[np.ndarray, float]:
    x_left = feature_fn(left.text)
    x_right = feature_fn(right.text)
    p_left, a_left = _predict(weights, x_left, actions)
    p_right, a_right = _predict(weights, x_right, actions)
    diff_p = p_left - p_right
    diff_a = a_left - a_right
    distance = float(np.mean(diff_p**2) + action_loss_weight * np.mean(diff_a**2))
    if distance >= margin:
        return np.zeros_like(weights), 0.0
    grad_p_left = -(2.0 / p_left.shape[0]) * diff_p
    grad_p_right = -grad_p_left
    grad_a = -(2.0 / actions.shape[1]) * diff_a
    grad_p_left += action_loss_weight * (actions @ grad_a)
    grad_p_right -= action_loss_weight * (actions @ grad_a)
    grad_left = _softmax_backward(p_left, grad_p_left)
    grad_right = _softmax_backward(p_right, grad_p_right)
    grad = np.outer(_with_bias(x_left), grad_left) + np.outer(_with_bias(x_right), grad_right)
    return grad, float(margin - distance)


def _variant_loss(
    weights: np.ndarray,
    spec: VariantSpec,
    actions: np.ndarray,
    feature_fn: Callable[[str], np.ndarray],
) -> float:
    losses: list[float] = []
    for example in spec.supervised_examples:
        _, loss = _supervised_grad(weights, example, actions, feature_fn)
        losses.append(loss)
    for left, right in spec.same_pairs:
        weight = 1.0 + (2.5 * right.difficulty if spec.difficulty_weighting else 0.0)
        _, loss = _same_pair_grad(weights, left, right, actions, feature_fn, 0.6, weight)
        losses.append(spec.lambda_consistency * loss)
    for left, right in spec.counterfactual_pairs:
        _, loss = _counterfactual_grad(weights, left, right, actions, feature_fn, 0.12, 0.4)
        losses.append(spec.lambda_counterfactual * loss)
    return float(np.mean(losses)) if losses else 0.0


def _train_variant(
    spec: VariantSpec,
    actions: np.ndarray,
    max_steps: int,
    learning_rate: float,
    feature_width: int,
) -> dict[str, Any]:
    feature_fn = lambda text: _text_features(text, width=feature_width, transform=spec.transform)
    weights = np.zeros((feature_width + 1, actions.shape[0]), dtype=np.float64)
    curve = [{"step": 0, "loss": round(_variant_loss(weights, spec, actions, feature_fn), 6)}]
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        grad = np.zeros_like(weights)
        example = spec.supervised_examples[(step - 1) % len(spec.supervised_examples)]
        supervised_grad, _ = _supervised_grad(weights, example, actions, feature_fn)
        grad += supervised_grad
        if spec.same_pairs:
            left, right = spec.same_pairs[(step - 1) % len(spec.same_pairs)]
            pair_weight = 1.0 + (2.5 * right.difficulty if spec.difficulty_weighting else 0.0)
            same_grad, _ = _same_pair_grad(weights, left, right, actions, feature_fn, 0.6, pair_weight)
            grad += spec.lambda_consistency * same_grad
        if spec.counterfactual_pairs:
            left, right = spec.counterfactual_pairs[(step - 1) % len(spec.counterfactual_pairs)]
            cf_grad, _ = _counterfactual_grad(weights, left, right, actions, feature_fn, 0.12, 0.4)
            grad += spec.lambda_counterfactual * cf_grad
        weights -= learning_rate * np.clip(grad, -5.0, 5.0)
        if step in {1, max_steps // 4, max_steps // 2, (3 * max_steps) // 4, max_steps}:
            curve.append({"step": step, "loss": round(_variant_loss(weights, spec, actions, feature_fn), 6)})
    return {
        "weights": weights,
        "feature_transform": spec.transform,
        "feature_width": feature_width,
        "loss_curve": curve,
        "initial_loss": curve[0]["loss"],
        "final_loss": curve[-1]["loss"],
        "loss_decreased": bool(curve[-1]["loss"] <= curve[0]["loss"]),
        "elapsed_seconds": round(time.perf_counter() - start, 6),
    }


def _read_first_action_chunk(path: Path, max_action_steps: int) -> np.ndarray:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        for demo_name in sorted(data_group.keys()):
            demo = data_group[demo_name]
            if "actions" not in demo:
                continue
            actions = np.asarray(demo["actions"][:max_action_steps], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[0] == 0:
                continue
            if actions.shape[0] < max_action_steps:
                pad = np.repeat(actions[-1:], max_action_steps - actions.shape[0], axis=0)
                actions = np.concatenate([actions, pad], axis=0)
            return actions.reshape(-1)
    raise ValueError(f"{path} has no readable actions dataset")


def _demo_paths_by_task(data_root: Path) -> dict[str, Path]:
    if not data_root.exists():
        return {}
    paths: dict[str, Path] = {}
    for path in sorted(data_root.rglob("*.hdf5")):
        stem = path.stem
        task_id = stem[: -len("_demo")] if stem.endswith("_demo") else stem
        paths.setdefault(task_id.lower(), path)
    return paths


def _parse_metadata_csv(path: Path, alpha: float = 0.5) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            original = _normalize_text(row.get("original_instruction", ""))
            new_instruction = row.get("new_instruction", "").strip()
            if not original or not new_instruction:
                continue
            structural = float(row.get("structural_similarity") or 0.0)
            keyword = float(row.get("keyword_similarity") or 0.0)
            similarity = max(0.0, min(1.0, alpha * keyword + (1.0 - alpha) * structural))
            grouped.setdefault(original, []).append(
                {
                    "new_instruction": new_instruction,
                    "original_instruction": row.get("original_instruction", "").strip(),
                    "high": row.get("high", ""),
                    "mid": row.get("mid", ""),
                    "low": row.get("low", ""),
                    "eval": row.get("eval", ""),
                    "batch_idx": int(row.get("batch_idx") or 0),
                    "structural_similarity": structural,
                    "keyword_similarity": keyword,
                    "difficulty": 1.0 - similarity,
                    "source": "official_libero_para_metadata",
                }
            )
    return grouped


def _local_paraphrases(instruction: str) -> list[dict[str, Any]]:
    normalized = _normalize_text(instruction)
    replacements = [
        ("stove", "cooktop", "obj"),
        ("bowl", "dish", "obj"),
        ("plate", "serving plate", "obj"),
        ("wine bottle", "bottle of wine", "obj"),
        ("rack", "stand", "obj"),
        ("put", "set", "act"),
        ("place", "position", "act"),
        ("push", "move", "act"),
        ("turn on", "activate", "act"),
    ]
    out: list[dict[str, Any]] = []
    for index, (source, target, high) in enumerate(replacements):
        if source not in normalized:
            continue
        new_instruction = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        out.append(
            {
                "new_instruction": new_instruction,
                "original_instruction": instruction,
                "high": high,
                "mid": "local_rule",
                "low": "exploratory_synonym",
                "eval": "local",
                "batch_idx": index,
                "structural_similarity": 0.82,
                "keyword_similarity": 0.70 if high == "obj" else 0.78,
                "difficulty": 0.24 if high == "obj" else 0.20,
                "source": "local_exploratory_paraphrase",
            }
        )
    out.append(
        {
            "new_instruction": f"please {normalized}",
            "original_instruction": instruction,
            "high": "act",
            "mid": "local_rule",
            "low": "politeness_prefix",
            "eval": "local",
            "batch_idx": len(out),
            "structural_similarity": 0.9,
            "keyword_similarity": 1.0,
            "difficulty": 0.05,
            "source": "local_exploratory_paraphrase",
        }
    )
    return out


def _select_balanced(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"obj": [], "act": [], "comp": [], "other": []}
    for row in rows:
        high = str(row.get("high") or "other")
        groups.setdefault(high if high in groups else "other", []).append(row)
    for key in groups:
        groups[key].sort(key=lambda row: (-float(row.get("difficulty", 0.0)), str(row.get("mid", "")), int(row.get("batch_idx", 0))))
    selected: list[dict[str, Any]] = []
    order = ["obj", "act", "comp", "other"]
    while len(selected) < max_rows and any(groups[key] for key in order):
        for key in order:
            if groups[key] and len(selected) < max_rows:
                selected.append(groups[key].pop(0))
    return selected


def _build_dataset(
    libero_root: Path,
    libero_data_root: Path,
    metadata_csv: Path,
    max_tasks: int,
    max_paraphrases_per_task: int,
    max_action_steps: int,
) -> tuple[list[TaskRecord], list[TextExample], dict[str, Any]]:
    try:
        import h5py  # noqa: F401

        hdf5_reader_available = True
        hdf5_reader_error = None
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        hdf5_reader_available = False
        hdf5_reader_error = str(exc)
    if not hdf5_reader_available:
        raise PrismDiagnosticError(f"h5py is required for local action chunks: {hdf5_reader_error}")

    metadata = _parse_metadata_csv(metadata_csv)
    demo_paths = _demo_paths_by_task(libero_data_root)
    tasks = discover_bddl_tasks(
        libero_root=libero_root,
        suites=["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
        max_tasks_per_suite=None,
    )
    candidates: list[tuple[dict[str, Any], Path, list[dict[str, Any]]]] = []
    for task in tasks:
        demo_path = demo_paths.get(str(task["task_id"]).lower())
        if demo_path is None:
            continue
        normalized_instruction = _normalize_text(str(task["language"]))
        rows = metadata.get(normalized_instruction) or []
        if rows or not metadata:
            candidates.append((task, demo_path, rows or _local_paraphrases(str(task["language"]))))
    candidates.sort(key=lambda item: (0 if item[2] and item[2][0].get("source") == "official_libero_para_metadata" else 1, item[0]["suite"], item[0]["task_id"]))
    selected_candidates = candidates[:max_tasks]
    if len(selected_candidates) < 2:
        raise PrismDiagnosticError("need at least two local LIBERO tasks with action chunks and paraphrases")

    task_records: list[TaskRecord] = []
    paraphrases: list[TextExample] = []
    width: int | None = None
    skipped: list[dict[str, str]] = []
    for label, (task, demo_path, rows) in enumerate(selected_candidates):
        try:
            action_chunk = _read_first_action_chunk(demo_path, max_action_steps=max_action_steps)
        except Exception as exc:
            skipped.append({"task_id": str(task.get("task_id")), "reason": str(exc)})
            continue
        if width is None:
            width = int(action_chunk.shape[0])
        if int(action_chunk.shape[0]) != width:
            skipped.append({"task_id": str(task.get("task_id")), "reason": "action chunk width mismatch"})
            continue
        task_records.append(
            TaskRecord(
                label=len(task_records),
                task_id=str(task["task_id"]),
                suite=str(task["suite"]),
                instruction=str(task["language"]),
                demo_file=str(demo_path),
                action_chunk=action_chunk,
                canonical_objects=tuple(str(item) for item in task.get("canonical_objects_of_interest", [])),
            )
        )
        for row in _select_balanced(rows, max_paraphrases_per_task):
            structural = float(row.get("structural_similarity", 1.0))
            keyword = float(row.get("keyword_similarity", 1.0))
            difficulty = float(row.get("difficulty", max(0.0, 1.0 - 0.5 * (structural + keyword))))
            paraphrases.append(
                TextExample(
                    text=str(row["new_instruction"]),
                    label=len(task_records) - 1,
                    source=str(row.get("source", "unknown")),
                    original_instruction=str(task["language"]),
                    high=str(row.get("high", "unknown")),
                    mid=str(row.get("mid", "unknown")),
                    low=str(row.get("low", "unknown")),
                    structural_similarity=structural,
                    keyword_similarity=keyword,
                    difficulty=difficulty,
                )
            )
    if len(task_records) < 2 or not paraphrases:
        raise PrismDiagnosticError("not enough task/paraphrase records after action inspection")

    metadata_sources = sorted({example.source for example in paraphrases})
    return task_records, paraphrases, {
        "libero_root": str(libero_root),
        "libero_data_root": str(libero_data_root),
        "libero_para_metadata_csv": str(metadata_csv),
        "libero_para_metadata_csv_exists": metadata_csv.exists(),
        "official_libero_para_metadata_used": "official_libero_para_metadata" in metadata_sources,
        "local_exploratory_paraphrases_used": "local_exploratory_paraphrase" in metadata_sources,
        "metadata_original_count": len(metadata),
        "candidate_task_count": len(candidates),
        "selected_task_count": len(task_records),
        "selected_paraphrase_count": len(paraphrases),
        "object_paraphrase_count": sum(1 for example in paraphrases if example.high == "obj"),
        "skipped_tasks": skipped,
        "task_ids": [record.task_id for record in task_records],
        "suites": sorted({record.suite for record in task_records}),
        "evidence_label": "exploratory_proxy_local_libero_action_chunks",
    }


def _clean_examples(tasks: list[TaskRecord]) -> tuple[TextExample, ...]:
    return tuple(
        TextExample(
            text=task.instruction,
            label=task.label,
            source="local_libero_clean_instruction",
            original_instruction=task.instruction,
        )
        for task in tasks
    )


def _same_pairs(clean: tuple[TextExample, ...], paraphrases: list[TextExample]) -> tuple[tuple[TextExample, TextExample], ...]:
    by_label = {example.label: example for example in clean}
    return tuple((by_label[item.label], item) for item in paraphrases if item.label in by_label)


def _counterfactual_pairs(clean: tuple[TextExample, ...]) -> tuple[tuple[TextExample, TextExample], ...]:
    pairs: list[tuple[TextExample, TextExample]] = []
    for left in clean:
        for right in clean:
            if left.label < right.label:
                pairs.append((left, right))
    return tuple(pairs)


def _score_examples(
    weights: np.ndarray,
    examples: list[TextExample] | tuple[TextExample, ...],
    actions: np.ndarray,
    feature_transform: str,
    feature_width: int,
    success_threshold: float,
) -> dict[str, Any]:
    if not examples:
        return {
            "count": 0,
            "success_proxy": None,
            "target_accuracy": None,
            "action_trajectory_divergence": None,
            "continuous_proxy_score": None,
            "pride": None,
        }
    feature_fn = lambda text: _text_features(text, width=feature_width, transform=feature_transform)
    successes: list[float] = []
    target_hits: list[float] = []
    divergences: list[float] = []
    continuous: list[float] = []
    pride_num = 0.0
    pride_den = 0.0
    scale = _task_distance_scale(actions)
    for example in examples:
        probs, pred = _predict(weights, feature_fn(example.text), actions)
        divergence = _action_l2(pred, actions[example.label])
        target_hit = float(int(np.argmax(probs) == example.label))
        success = float(target_hit > 0.5 and divergence <= success_threshold)
        proxy = 0.5 * float(probs[example.label]) + 0.5 * math.exp(-divergence / scale)
        target_hits.append(target_hit)
        divergences.append(divergence)
        successes.append(success)
        continuous.append(proxy)
        pride_num += success * example.difficulty
        pride_den += example.difficulty
    return {
        "count": len(examples),
        "success_proxy": round(float(np.mean(successes)), 6),
        "target_accuracy": round(float(np.mean(target_hits)), 6),
        "action_trajectory_divergence": round(float(np.mean(divergences)), 6),
        "continuous_proxy_score": round(float(np.mean(continuous)), 6),
        "pride": round(float(100.0 * pride_num / pride_den), 6) if pride_den > 0.0 else None,
        "difficulty_weighted_robustness": round(float(sum(c * e.difficulty for c, e in zip(continuous, examples)) / pride_den), 6) if pride_den > 0.0 else None,
    }


def _consistency_metrics(
    weights: np.ndarray,
    pairs: tuple[tuple[TextExample, TextExample], ...],
    actions: np.ndarray,
    feature_transform: str,
    feature_width: int,
) -> dict[str, Any]:
    if not pairs:
        return {"pair_count": 0}
    feature_fn = lambda text: _text_features(text, width=feature_width, transform=feature_transform)
    action_divergences: list[float] = []
    distribution_divergences: list[float] = []
    scale = _task_distance_scale(actions)
    for left, right in pairs:
        probs_left, action_left = _predict(weights, feature_fn(left.text), actions)
        probs_right, action_right = _predict(weights, feature_fn(right.text), actions)
        action_divergences.append(_action_l2(action_left, action_right))
        distribution_divergences.append(_distribution_l2(probs_left, probs_right))
    mean_action = float(np.mean(action_divergences))
    mean_distribution = float(np.mean(distribution_divergences))
    return {
        "pair_count": len(pairs),
        "action_consistency_divergence": round(mean_action, 6),
        "distribution_consistency_divergence": round(mean_distribution, 6),
        "consistency_score": round(math.exp(-mean_action / scale) * math.exp(-mean_distribution), 6),
    }


def _counterfactual_sensitivity_metrics(
    weights: np.ndarray,
    pairs: tuple[tuple[TextExample, TextExample], ...],
    actions: np.ndarray,
    feature_transform: str,
    feature_width: int,
) -> dict[str, Any]:
    if not pairs:
        return {"pair_count": 0}
    feature_fn = lambda text: _text_features(text, width=feature_width, transform=feature_transform)
    action_divergences: list[float] = []
    distribution_divergences: list[float] = []
    same_top = 0
    for left, right in pairs:
        probs_left, action_left = _predict(weights, feature_fn(left.text), actions)
        probs_right, action_right = _predict(weights, feature_fn(right.text), actions)
        action_divergences.append(_action_l2(action_left, action_right))
        distribution_divergences.append(_distribution_l2(probs_left, probs_right))
        same_top += int(np.argmax(probs_left) == np.argmax(probs_right))
    return {
        "pair_count": len(pairs),
        "counterfactual_action_divergence": round(float(np.mean(action_divergences)), 6),
        "counterfactual_distribution_divergence": round(float(np.mean(distribution_divergences)), 6),
        "same_top_prediction_rate": round(float(same_top / len(pairs)), 6),
        "counterfactual_sensitivity_score": round(float(np.mean(distribution_divergences)) + 0.25 * float(np.mean(action_divergences)), 6),
    }


def _evaluate_variant(
    trained: dict[str, Any],
    clean_examples: tuple[TextExample, ...],
    paraphrases: list[TextExample],
    same_pairs: tuple[tuple[TextExample, TextExample], ...],
    counterfactual_pairs: tuple[tuple[TextExample, TextExample], ...],
    actions: np.ndarray,
    base_clean_proxy: float | None,
) -> dict[str, Any]:
    weights: np.ndarray = trained["weights"]
    transform = str(trained["feature_transform"])
    feature_width = int(trained["feature_width"])
    success_threshold = max(0.02, 0.55 * _task_distance_scale(actions))
    object_examples = [example for example in paraphrases if example.high == "obj"]
    clean = _score_examples(weights, clean_examples, actions, transform, feature_width, success_threshold)
    para = _score_examples(weights, paraphrases, actions, transform, feature_width, success_threshold)
    obj = _score_examples(weights, object_examples, actions, transform, feature_width, success_threshold)
    consistency = _consistency_metrics(weights, same_pairs, actions, transform, feature_width)
    counterfactual = _counterfactual_sensitivity_metrics(weights, counterfactual_pairs, actions, transform, feature_width)
    clean_proxy = clean.get("continuous_proxy_score")
    clean_retention = None
    if base_clean_proxy and clean_proxy is not None:
        clean_retention = round(float(clean_proxy) / max(1e-9, float(base_clean_proxy)), 6)
    return {
        "initial_loss": trained["initial_loss"],
        "final_loss": trained["final_loss"],
        "loss_decreased": trained["loss_decreased"],
        "loss_curve": trained["loss_curve"],
        "elapsed_seconds": trained["elapsed_seconds"],
        "clean": clean,
        "paraphrase": para,
        "object_lexical_variation": obj,
        "paraphrase_consistency": consistency,
        "counterfactual_sensitivity": counterfactual,
        "clean_retention_vs_base_clean": clean_retention,
    }


def _comparison(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = variants["base_no_paraphrase_training"]
    aug = variants["simple_paraphrase_augmentation"]
    prism = variants["prism_vla_consistency"]
    base_clean = float(base["clean"]["continuous_proxy_score"])
    base_para = float(base["paraphrase"]["continuous_proxy_score"])
    degradation = round(base_clean - base_para, 6)
    prism_vs_aug = {
        "paraphrase_proxy_delta": round(float(prism["paraphrase"]["continuous_proxy_score"]) - float(aug["paraphrase"]["continuous_proxy_score"]), 6),
        "pride_delta": round(float(prism["paraphrase"]["pride"] or 0.0) - float(aug["paraphrase"]["pride"] or 0.0), 6),
        "consistency_score_delta": round(float(prism["paraphrase_consistency"]["consistency_score"]) - float(aug["paraphrase_consistency"]["consistency_score"]), 6),
        "action_trajectory_divergence_delta": round(float(prism["paraphrase"]["action_trajectory_divergence"]) - float(aug["paraphrase"]["action_trajectory_divergence"]), 6),
        "object_lexical_proxy_delta": round(
            float(prism["object_lexical_variation"]["continuous_proxy_score"] or 0.0)
            - float(aug["object_lexical_variation"]["continuous_proxy_score"] or 0.0),
            6,
        ),
    }
    best_robustness_delta = max(
        prism_vs_aug["paraphrase_proxy_delta"],
        prism_vs_aug["pride_delta"] / 100.0,
        prism_vs_aug["consistency_score_delta"],
        prism_vs_aug["object_lexical_proxy_delta"],
        -prism_vs_aug["action_trajectory_divergence_delta"],
    )
    clean_retained = float(prism["clean_retention_vs_base_clean"] or 0.0) >= 0.80
    sensitivity = float(prism["counterfactual_sensitivity"]["counterfactual_sensitivity_score"])
    aug_sensitivity = float(aug["counterfactual_sensitivity"]["counterfactual_sensitivity_score"])
    cf_preserved = sensitivity >= 0.80 * max(1e-9, aug_sensitivity)
    collapse = float(prism["counterfactual_sensitivity"]["same_top_prediction_rate"]) >= 0.80
    reasons: list[str] = []
    if degradation < 0.02:
        reasons.append("paraphrase degradation is not measurable in the base proxy")
    if best_robustness_delta <= 1e-4:
        reasons.append("simple paraphrase augmentation matches PRISM on robustness metrics")
    if not clean_retained:
        reasons.append("clean performance drops substantially")
    if not cf_preserved or collapse:
        reasons.append("PRISM appears to improve consistency by weakening counterfactual sensitivity")
    decision = "continue" if not reasons else "kill_or_block"
    return {
        "base_clean_to_paraphrase_proxy_drop": degradation,
        "prism_vs_simple_augmentation": prism_vs_aug,
        "best_prism_robustness_delta_over_augmentation": round(best_robustness_delta, 6),
        "clean_retained": clean_retained,
        "counterfactual_sensitivity_preserved": bool(cf_preserved and not collapse),
        "continue_criteria_met": decision == "continue",
        "decision": decision,
        "kill_or_block_reasons": reasons,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    variants = report["variants"]
    lines = [
        "# PRISM-VLA Paraphrase Robustness Diagnostic",
        "",
        "This is exploratory offline proxy evidence only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- decision: `{report['decision']['decision']}`",
        f"- continue criteria met: `{report['decision']['continue_criteria_met']}`",
        f"- model: `{report['model']['model_name']}`",
        f"- dataset: `{report['data']['dataset_used']}`",
        f"- training performed: `{report['policy']['training_performed']}`",
        f"- loss computed: `{report['policy']['loss_computed']}`",
        f"- rollouts performed: `{report['policy']['rollouts_performed']}`",
        f"- selected tasks: `{report['data']['selected_task_count']}`",
        f"- selected paraphrases: `{report['data']['selected_paraphrase_count']}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | clean proxy | paraphrase proxy | object proxy | PRIDE | consistency | cf sensitivity | final loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in variants.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(payload["clean"]["continuous_proxy_score"]),
                    str(payload["paraphrase"]["continuous_proxy_score"]),
                    str(payload["object_lexical_variation"]["continuous_proxy_score"]),
                    str(payload["paraphrase"]["pride"]),
                    str(payload["paraphrase_consistency"].get("consistency_score")),
                    str(payload["counterfactual_sensitivity"].get("counterfactual_sensitivity_score")),
                    str(payload["final_loss"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- base clean-to-paraphrase proxy drop: `{report['decision']['base_clean_to_paraphrase_proxy_drop']}`",
            f"- PRISM best robustness delta over simple augmentation: `{report['decision']['best_prism_robustness_delta_over_augmentation']}`",
            f"- clean retained: `{report['decision']['clean_retained']}`",
            f"- counterfactual sensitivity preserved: `{report['decision']['counterfactual_sensitivity_preserved']}`",
            f"- kill/block reasons: `{report['decision']['kill_or_block_reasons']}`",
            "",
            "## Limitations",
            "",
            "- The policy is a tiny NumPy surrogate, not a real VLA checkpoint.",
            "- Metrics are offline action-distribution/action-chunk proxies, not simulator success.",
            "- If official LIBERO-Para metadata is unavailable, local paraphrases are exploratory and non-paper-grade.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_bounds(max_tasks: int, max_paraphrases_per_task: int, max_action_steps: int, max_steps: int, learning_rate: float) -> None:
    if max_tasks < 2 or max_tasks > 12:
        raise PrismDiagnosticError("max_tasks must be between 2 and 12")
    if max_paraphrases_per_task < 1 or max_paraphrases_per_task > 80:
        raise PrismDiagnosticError("max_paraphrases_per_task must be between 1 and 80")
    if max_action_steps < 1 or max_action_steps > 64:
        raise PrismDiagnosticError("max_action_steps must be between 1 and 64")
    if max_steps < 1 or max_steps > MAX_TRAINING_STEPS:
        raise PrismDiagnosticError(f"max_steps must be between 1 and {MAX_TRAINING_STEPS}")
    if learning_rate <= 0.0 or learning_rate > 1.0:
        raise PrismDiagnosticError("learning_rate must be in (0, 1]")


def build_prism_vla_diagnostic(
    libero_root: Path,
    libero_data_root: Path,
    libero_para_metadata_csv: Path = Path(DEFAULT_METADATA_CSV),
    max_tasks: int = DEFAULT_MAX_TASKS,
    max_paraphrases_per_task: int = DEFAULT_MAX_PARAPHRASES_PER_TASK,
    max_action_steps: int = DEFAULT_MAX_ACTION_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 0.12,
    feature_width: int = DEFAULT_FEATURE_WIDTH,
) -> dict[str, Any]:
    validate_bounds(max_tasks, max_paraphrases_per_task, max_action_steps, max_steps, learning_rate)
    started = time.perf_counter()
    tasks, paraphrases, data_metadata = _build_dataset(
        libero_root=libero_root,
        libero_data_root=libero_data_root,
        metadata_csv=libero_para_metadata_csv,
        max_tasks=max_tasks,
        max_paraphrases_per_task=max_paraphrases_per_task,
        max_action_steps=max_action_steps,
    )
    clean = _clean_examples(tasks)
    same = _same_pairs(clean, paraphrases)
    counterfactual = _counterfactual_pairs(clean)
    actions = np.asarray([task.action_chunk for task in tasks], dtype=np.float64)
    specs = [
        VariantSpec("base_no_paraphrase_training", "raw", clean),
        VariantSpec("simple_paraphrase_augmentation", "raw", tuple(list(clean) + paraphrases)),
        VariantSpec("instruction_canonicalization_baseline", "canonical", clean),
        VariantSpec(
            "prism_vla_consistency",
            "raw",
            tuple(list(clean) + paraphrases),
            same_pairs=same,
            counterfactual_pairs=counterfactual,
            lambda_consistency=0.85,
            lambda_counterfactual=0.30,
            difficulty_weighting=True,
        ),
    ]
    trained = {
        spec.name: _train_variant(spec, actions, max_steps=max_steps, learning_rate=learning_rate, feature_width=feature_width)
        for spec in specs
    }
    base_clean = _score_examples(
        trained["base_no_paraphrase_training"]["weights"],
        clean,
        actions,
        str(trained["base_no_paraphrase_training"]["feature_transform"]),
        int(trained["base_no_paraphrase_training"]["feature_width"]),
        max(0.02, 0.55 * _task_distance_scale(actions)),
    )["continuous_proxy_score"]
    variants = {
        spec.name: _evaluate_variant(trained[spec.name], clean, paraphrases, same, counterfactual, actions, float(base_clean))
        for spec in specs
    }
    decision = _comparison(variants)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": int(time.time()),
        "policy": {
            "offline_proxy_only": True,
            "exploratory": True,
            "confirmatory": False,
            "not_standard_success": True,
            "not_paper_grade": True,
            "downloads_performed_by_this_script": False,
            "gpu_jobs_performed": False,
            "training_performed": True,
            "tiny_cpu_training_performed": True,
            "loss_computed": True,
            "rollouts_performed": False,
            "simulator_executed": False,
            "heavy_model_imports_performed": False,
            "vla_checkpoint_loaded": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
        },
        "model": {
            "model_name": "tiny_numpy_semantic_action_distribution_policy",
            "real_vla_model_metric_produced": False,
            "candidate_action_distribution": "softmax over local LIBERO task action chunks",
            "feature_width": feature_width,
            "trainable_parameter_count_per_variant": int((feature_width + 1) * len(tasks)),
        },
        "data": {
            **data_metadata,
            "dataset_used": (
                "official LIBERO-Para metadata + local LIBERO HDF5 action chunks"
                if data_metadata["official_libero_para_metadata_used"]
                else "local LIBERO HDF5 action chunks + generated exploratory paraphrases"
            ),
            "real_dataset_metric_produced": True,
            "real_vla_model_metric_produced": False,
            "action_chunk_width": int(actions.shape[1]),
            "max_action_steps": max_action_steps,
            "evaluation_split": "exploratory_same_selected_paraphrases_for_tiny_training_and_eval",
        },
        "training": {
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "batch_size": 1,
            "variants": [spec.name for spec in specs],
            "prism_objective": {
                "same_task_paraphrase_consistency": True,
                "object_target_counterfactual_pairs_not_forced_close": True,
                "difficulty_weighting": True,
                "clean_retention_tracked": True,
            },
        },
        "metrics": {
            "clean_success_proxy": "continuous proxy and binary action/target proxy on clean local LIBERO instructions",
            "paraphrase_success_proxy": "continuous proxy and binary action/target proxy on paraphrased instructions",
            "paraphrase_consistency": "same-task action/distribution divergence between original and paraphrase",
            "object_lexical_variation_robustness": "same metrics restricted to LIBERO-Para high=obj rows",
            "counterfactual_object_sensitivity": "distribution/action divergence across different local LIBERO tasks",
            "action_trajectory_divergence": "L2 over predicted vs expert local action chunks",
            "clean_retention": "clean proxy divided by base clean proxy",
            "pride": "difficulty-weighted binary success proxy using LIBERO-Para similarity-derived difficulty",
        },
        "variants": variants,
        "decision": decision,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "recommended_next_step": (
            "Run a held-out and then real-VLA adapter paraphrase diagnostic only after this proxy is reviewed."
            if decision["continue_criteria_met"]
            else "Do not scale PRISM-VLA yet; inspect the kill/block reasons and baseline parity first."
        ),
    }
    return report


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--libero-root", default="")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--libero-para-metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--max-paraphrases-per-task", type=int, default=DEFAULT_MAX_PARAPHRASES_PER_TASK)
    parser.add_argument("--max-action-steps", type=int, default=DEFAULT_MAX_ACTION_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--feature-width", type=int, default=DEFAULT_FEATURE_WIDTH)
    parser.add_argument("--report-json", default="reports/prism_vla_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/prism_vla_diagnostic_report.md")
    args = parser.parse_args()

    paths = read_asset_paths(Path(args.paths_file))
    libero_root = Path(args.libero_root or paths.get("libero_root", "C:/assets/repos/LIBERO"))
    libero_data_root = Path(args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero"))
    report = build_prism_vla_diagnostic(
        libero_root=libero_root,
        libero_data_root=libero_data_root,
        libero_para_metadata_csv=Path(args.libero_para_metadata_csv),
        max_tasks=args.max_tasks,
        max_paraphrases_per_task=args.max_paraphrases_per_task,
        max_action_steps=args.max_action_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        feature_width=args.feature_width,
    )
    write_reports(report, Path(args.report_json), Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
