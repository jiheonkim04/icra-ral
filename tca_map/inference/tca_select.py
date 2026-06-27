"""TCA-Select inference utilities.

TCA-Select is an inference-time candidate selector over target-conditioned
continuous/voxel action heatmaps. It does not call external verifiers and does
not use privileged simulator state.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_K = 4
DEFAULT_TEMPERATURE = 0.5
_PRIVILEGED_METADATA_KEYS = {
    "oracle_target",
    "privileged_target",
    "simulator_state",
    "sim_state",
    "ground_truth_object_pose",
    "gt_object_pose",
    "oracle_action",
}


def _ensure_no_privileged_metadata(metadata: dict | None) -> None:
    if not metadata:
        return
    blocked = sorted(key for key in metadata if key in _PRIVILEGED_METADATA_KEYS)
    if blocked or metadata.get("privileged", False):
        raise ValueError(f"TCA-Select forbids privileged inference metadata: {blocked or ['privileged']}")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _softmax(logits: list[float], temperature: float) -> list[float]:
    safe_temperature = max(float(temperature), 1e-6)
    scaled = [value / safe_temperature for value in logits]
    max_value = max(scaled) if scaled else 0.0
    exp_values = [math.exp(value - max_value) for value in scaled]
    total = sum(exp_values) or 1.0
    return [value / total for value in exp_values]


def _flatten_action_heatmap(action_heatmap: Any) -> list[dict]:
    """Convert supported dummy heatmap formats into candidate entries."""
    entries: list[dict] = []
    if isinstance(action_heatmap, dict):
        if isinstance(action_heatmap.get("candidates"), list):
            for idx, item in enumerate(action_heatmap["candidates"]):
                if isinstance(item, dict):
                    entries.append(
                        {
                            "index": item.get("index", idx),
                            "action": item.get("action", item.get("voxel", item.get("candidate", idx))),
                            "voxel": item.get("voxel", item.get("index", idx)),
                            "logit": _as_float(item.get("logit", item.get("score", item.get("probability", 0.0)))),
                            "target_index": item.get("target_index"),
                        }
                    )
                else:
                    entries.append({"index": idx, "action": item, "voxel": idx, "logit": _as_float(item)})
            return entries
        if isinstance(action_heatmap.get("values"), list):
            actions = action_heatmap.get("actions") or []
            voxels = action_heatmap.get("voxels") or []
            target_indices = action_heatmap.get("target_indices") or []
            for idx, value in enumerate(action_heatmap["values"]):
                entries.append(
                    {
                        "index": idx,
                        "action": actions[idx] if idx < len(actions) else idx,
                        "voxel": voxels[idx] if idx < len(voxels) else idx,
                        "logit": _as_float(value),
                        "target_index": target_indices[idx] if idx < len(target_indices) else None,
                    }
                )
            return entries
        if "top_voxel" in action_heatmap:
            return [
                {
                    "index": 0,
                    "action": action_heatmap.get("expected_action", action_heatmap["top_voxel"]),
                    "voxel": action_heatmap["top_voxel"],
                    "logit": _as_float(action_heatmap.get("score", 1.0), 1.0),
                    "target_index": action_heatmap.get("target_index"),
                }
            ]
    if isinstance(action_heatmap, list):
        for idx, value in enumerate(action_heatmap):
            if isinstance(value, dict):
                entries.append(
                    {
                        "index": value.get("index", idx),
                        "action": value.get("action", value.get("voxel", idx)),
                        "voxel": value.get("voxel", value.get("index", idx)),
                        "logit": _as_float(value.get("logit", value.get("score", value.get("probability", 0.0)))),
                        "target_index": value.get("target_index"),
                    }
                )
            else:
                entries.append({"index": idx, "action": idx, "voxel": idx, "logit": _as_float(value)})
    return entries


def _heatmap_value_at(heatmap: Any, candidate: dict) -> float:
    idx = int(candidate.get("index", 0))
    if isinstance(heatmap, dict):
        if isinstance(heatmap.get("values"), list) and 0 <= idx < len(heatmap["values"]):
            return _as_float(heatmap["values"][idx])
        if isinstance(heatmap.get("candidates"), list) and 0 <= idx < len(heatmap["candidates"]):
            item = heatmap["candidates"][idx]
            if isinstance(item, dict):
                return _as_float(item.get("logit", item.get("score", item.get("probability", 0.0))))
            return _as_float(item)
    if isinstance(heatmap, list) and 0 <= idx < len(heatmap):
        item = heatmap[idx]
        if isinstance(item, dict):
            return _as_float(item.get("logit", item.get("score", item.get("probability", 0.0))))
        return _as_float(item)
    return _as_float(candidate.get("logit", candidate.get("probability", 0.0)))


def _target_score(target_heatmap: Any, target_index: Any) -> float:
    if target_index is None:
        if isinstance(target_heatmap, dict) and "top_index" in target_heatmap:
            target_index = target_heatmap["top_index"]
        else:
            return 0.0
    try:
        idx = int(target_index)
    except (TypeError, ValueError):
        return 0.0
    if isinstance(target_heatmap, dict):
        scores = target_heatmap.get("scores") or target_heatmap.get("values") or []
        if 0 <= idx < len(scores):
            return _as_float(scores[idx])
        if target_heatmap.get("top_index") == idx:
            return _as_float(target_heatmap.get("top_score", 1.0), 1.0)
    if isinstance(target_heatmap, list) and 0 <= idx < len(target_heatmap):
        return _as_float(target_heatmap[idx])
    return 0.0


def sample_heatmap_candidates(action_heatmap: Any, K: int = DEFAULT_K, temperature: float = DEFAULT_TEMPERATURE) -> list[dict]:
    """Return the top-K target-conditioned action candidates from a heatmap.

    The scaffold uses deterministic top-K selection for repeatable tests while
    retaining a temperature-scaled probability for each candidate.
    """
    entries = _flatten_action_heatmap(action_heatmap)
    if not entries:
        return []
    probabilities = _softmax([entry["logit"] for entry in entries], temperature)
    for entry, probability in zip(entries, probabilities):
        entry["probability"] = probability
        entry["temperature"] = temperature
    entries.sort(key=lambda item: (item["probability"], item["logit"]), reverse=True)
    return entries[: max(0, int(K))]


def score_candidate_target_consistency(
    candidate: dict,
    action_heatmap: Any,
    target_heatmap: Any,
    metadata: dict | None = None,
) -> float:
    """Score internal target/action consistency without privileged state."""
    _ensure_no_privileged_metadata(metadata)
    action_score = _as_float(candidate.get("probability", _heatmap_value_at(action_heatmap, candidate)))
    target_score = _target_score(target_heatmap, candidate.get("target_index"))
    geometry_bonus = 0.0
    if metadata:
        # Non-privileged metadata can provide predicted target/action distances.
        geometry_bonus = -abs(_as_float(metadata.get("predicted_target_action_distance", 0.0)))
    return action_score + 0.1 * target_score + 0.01 * geometry_bonus


def score_candidate_condition_sensitivity(candidate: dict, full_heatmap: Any, masked_heatmap: Any) -> float:
    """Prefer candidates that are strong with full language and weaker when masked."""
    full_value = _heatmap_value_at(full_heatmap, candidate)
    masked_value = _heatmap_value_at(masked_heatmap, candidate)
    return full_value - masked_value


def select_tca_candidate(candidates: list[dict], scores: list[float] | dict[Any, float]) -> dict | None:
    if not candidates:
        return None
    if isinstance(scores, dict):
        def score_for(candidate: dict) -> float:
            return _as_float(scores.get(candidate.get("index"), scores.get(str(candidate.get("index")), 0.0)))
    else:
        def score_for(candidate: dict) -> float:
            idx = candidates.index(candidate)
            return _as_float(scores[idx] if idx < len(scores) else 0.0)
    best = max(candidates, key=lambda item: (score_for(item), _as_float(item.get("probability", 0.0))))
    selected = dict(best)
    selected["tca_select_score"] = score_for(best)
    return selected


def tca_select_inference(
    action_heatmap: Any,
    target_heatmap: Any,
    full_heatmap: Any | None = None,
    masked_heatmap: Any | None = None,
    K: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    metadata: dict | None = None,
    external_verifier: Any | None = None,
) -> dict:
    """Run TCA-Select with internal heatmap consistency signals only."""
    if external_verifier is not None:
        raise ValueError("TCA-Select must not use external verifiers.")
    _ensure_no_privileged_metadata(metadata)
    candidates = sample_heatmap_candidates(action_heatmap, K=K, temperature=temperature)
    scores: list[float] = []
    for candidate in candidates:
        consistency = score_candidate_target_consistency(candidate, action_heatmap, target_heatmap, metadata=metadata)
        sensitivity = 0.0
        if full_heatmap is not None and masked_heatmap is not None:
            sensitivity = score_candidate_condition_sensitivity(candidate, full_heatmap, masked_heatmap)
        scores.append(consistency + 0.25 * sensitivity)
    selected = select_tca_candidate(candidates, scores)
    return {
        "selected": selected,
        "candidates": candidates,
        "scores": scores,
        "K": K,
        "temperature": temperature,
        "external_verifier_used": False,
        "privileged_inference_used": False,
    }
