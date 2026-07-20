"""Outcome-independent utilities for the Epoch 7 language-grounding gate."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

FAMILIES = ("act", "obj", "comp")


def cag_guidance(conditional: Any, unconditional: Any, omega: float) -> Any:
    """Apply CAG Eq. 4; omega=1 recovers the conditional policy."""

    return unconditional + float(omega) * (conditional - unconditional)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_bddl_instruction(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("(:language"):
            return line.removeprefix("(:language").rstrip(")").strip()
    raise ValueError(f"no (:language ...) instruction in {path}")


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def character_ngrams(text: str, n: int = 3) -> Counter[str]:
    normalized = normalized_text(text)
    if n <= 0:
        raise ValueError("n must be positive")
    return Counter(normalized[index : index + n] for index in range(max(0, len(normalized) - n + 1)))


def counter_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def canonicalize_instruction(instruction: str, catalog: Mapping[int, str]) -> dict[str, Any]:
    """Return the frozen character-trigram retrieval result.

    The preflight prototype sorted ``(score, eval_id)`` in descending order,
    so a score tie resolves to the larger numeric eval_id. This behavior is
    intentional and frozen rather than silently improved after seeing data.
    """

    query = character_ngrams(instruction)
    ranked = sorted(
        ((counter_cosine(query, character_ngrams(text)), int(eval_id), str(text)) for eval_id, text in catalog.items()),
        reverse=True,
    )
    if not ranked:
        raise ValueError("canonical instruction catalog is empty")
    score, eval_id, text = ranked[0]
    runner_up_score = ranked[1][0] if len(ranked) > 1 else None
    return {
        "selected_eval_id": eval_id,
        "selected_instruction": text,
        "score": score,
        "runner_up_score": runner_up_score,
        "margin": score - runner_up_score if runner_up_score is not None else None,
    }


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("status") != "FROZEN_BEFORE_CLOSED_LOOP_OUTCOMES":
        errors.append("protocol status is not frozen")
    if protocol.get("ours_authorized") is not False:
        errors.append("Ours must remain unauthorized")
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10:
        errors.append("exactly ten task records are required")
        return errors
    seen_eval_ids: set[int] = set()
    seen_files: set[str] = set()
    for task in tasks:
        eval_id = int(task.get("eval_id", -1))
        if eval_id in seen_eval_ids:
            errors.append(f"duplicate eval_id {eval_id}")
        seen_eval_ids.add(eval_id)
        conditions = task.get("conditions", {})
        if set(conditions) != set(FAMILIES):
            errors.append(f"eval{eval_id} does not contain exactly {FAMILIES}")
        for family in FAMILIES:
            filename = conditions.get(family)
            if not isinstance(filename, str) or not filename.endswith(".bddl"):
                errors.append(f"eval{eval_id}/{family} has invalid BDDL identity")
            elif filename in seen_files:
                errors.append(f"duplicate BDDL identity {filename}")
            else:
                seen_files.add(filename)
    if seen_eval_ids != set(range(10)):
        errors.append("eval IDs must equal 0..9")
    if len(seen_files) != 30:
        errors.append("exactly 30 unique paraphrase identities are required")
    pairing = protocol.get("pairing", {})
    for family in FAMILIES:
        if family not in pairing:
            errors.append(f"missing pairing for {family}")
    return errors


def iter_pair_specs(protocol: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    pairing = protocol["pairing"]
    for task in sorted(protocol["tasks"], key=lambda item: int(item["eval_id"])):
        for family in FAMILIES:
            pair = pairing[family]
            yield {
                "pair_id": f"eval{int(task['eval_id'])}_{family}",
                "eval_id": int(task["eval_id"]),
                "family": family,
                "goal_bddl": str(task["goal_bddl"]),
                "canonical_instruction": str(task["canonical_instruction"]),
                "paraphrase_bddl": str(task["conditions"][family]),
                "seed": int(pair["seed"]),
                "initial_state_index": int(pair["initial_state_index"]),
                "model_seed": 7_000_000 + 100 * int(task["eval_id"]) + FAMILIES.index(family),
            }


def select_pair_specs(protocol: Mapping[str, Any], requested: Iterable[str], max_pairs: int | None) -> list[dict[str, Any]]:
    all_specs = list(iter_pair_specs(protocol))
    requested_set = {item.strip() for item in requested if item.strip()}
    if requested_set:
        unknown = requested_set - {item["pair_id"] for item in all_specs}
        if unknown:
            raise ValueError(f"unknown pair IDs: {sorted(unknown)}")
        all_specs = [item for item in all_specs if item["pair_id"] in requested_set]
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        all_specs = all_specs[:max_pairs]
    return all_specs


def summarize_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [episode for episode in episodes if episode.get("completed") is True]
    successful = [episode for episode in completed if episode.get("success") is True]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(episode.get("condition")) for episode in completed}):
        subset = [episode for episode in completed if str(episode.get("condition")) == condition]
        count = len(subset)
        successes = sum(bool(episode.get("success")) for episode in subset)
        by_condition[condition] = {
            "episodes": count,
            "successes": successes,
            "success_rate": successes / count if count else None,
        }
    return {
        "completed_episodes": len(completed),
        "successful_episodes": len(successful),
        "success_rate": len(successful) / len(completed) if completed else None,
        "by_condition": by_condition,
    }
