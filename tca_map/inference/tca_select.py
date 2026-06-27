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
_DEFAULT_DISTRIBUTIONAL_WEIGHTS = {
    "log_probability": 1.0,
    "condition_kl": 0.25,
    "negative_action_divergence": 0.25,
    "target_consistency": 0.5,
    "target_margin": 0.25,
    "entropy_penalty": 0.05,
    "use_kl": True,
    "use_js": True,
}
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
        if isinstance(heatmap.get("scores"), list) and 0 <= idx < len(heatmap["scores"]):
            return _as_float(heatmap["scores"][idx])
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


def _heatmap_values(heatmap: Any) -> list[float]:
    if heatmap is None:
        return []
    if isinstance(heatmap, dict):
        if isinstance(heatmap.get("values"), list):
            return [_as_float(value) for value in heatmap["values"]]
        if isinstance(heatmap.get("scores"), list):
            return [_as_float(value) for value in heatmap["scores"]]
        if isinstance(heatmap.get("candidates"), list):
            return [entry["logit"] for entry in _flatten_action_heatmap(heatmap)]
        if "top_score" in heatmap:
            top_index = int(_as_float(heatmap.get("top_index", 0)))
            values = [0.0] * max(top_index + 1, 1)
            values[top_index] = _as_float(heatmap.get("top_score", 1.0), 1.0)
            return values
    if isinstance(heatmap, list):
        if all(isinstance(item, dict) for item in heatmap):
            return [entry["logit"] for entry in _flatten_action_heatmap(heatmap)]
        return [_as_float(value) for value in heatmap]
    return []


def _aligned_distributions(first: Any, second: Any, eps: float) -> tuple[list[float], list[float]]:
    first_dist = normalize_heatmap_distribution(first, eps=eps)
    second_dist = normalize_heatmap_distribution(second, eps=eps)
    length = max(len(first_dist), len(second_dist))
    if length == 0:
        return [], []
    first_dist = (first_dist + [eps] * (length - len(first_dist)))[:length]
    second_dist = (second_dist + [eps] * (length - len(second_dist)))[:length]
    first_total = sum(first_dist) or 1.0
    second_total = sum(second_dist) or 1.0
    return [value / first_total for value in first_dist], [value / second_total for value in second_dist]


def _as_heatmap_sequence(heatmaps: Any) -> list[Any]:
    if heatmaps is None:
        return []
    if isinstance(heatmaps, tuple):
        return list(heatmaps)
    if isinstance(heatmaps, list):
        if not heatmaps:
            return []
        if all(not isinstance(item, (dict, list, tuple)) for item in heatmaps):
            return [heatmaps]
        if all(isinstance(item, dict) and ("logit" in item or "score" in item or "probability" in item) for item in heatmaps):
            return [heatmaps]
        return heatmaps
    return [heatmaps]


def _candidate_probability(heatmap: Any, candidate: dict, eps: float = 1e-8) -> float:
    distribution = normalize_heatmap_distribution(heatmap, eps=eps)
    try:
        idx = int(candidate.get("index", 0))
    except (TypeError, ValueError):
        idx = -1
    if 0 <= idx < len(distribution):
        return max(distribution[idx], eps)
    return max(_as_float(candidate.get("probability", eps), eps), eps)


def normalize_heatmap_distribution(heatmap: Any, eps: float = 1e-8) -> list[float]:
    """Normalize a list/dict heatmap into a finite probability distribution.

    Nonnegative heatmaps are treated as unnormalized masses. Heatmaps containing
    negative values are treated as logits and normalized with a stable softmax.
    """
    values = _heatmap_values(heatmap)
    if not values:
        return []
    if all(value >= 0.0 for value in values) and sum(values) > 0.0:
        masses = [max(value, 0.0) + eps for value in values]
    else:
        masses = [max(value, eps) for value in _softmax(values, temperature=1.0)]
    total = sum(masses) or 1.0
    return [value / total for value in masses]


def heatmap_kl(q_heatmap: Any, p_heatmap: Any, eps: float = 1e-8) -> float:
    """Compute KL(q || p) over normalized heatmap distributions."""
    q_dist, p_dist = _aligned_distributions(q_heatmap, p_heatmap, eps=eps)
    if not q_dist:
        return 0.0
    return sum(q * math.log(max(q, eps) / max(p, eps)) for q, p in zip(q_dist, p_dist))


def heatmap_js(p_heatmap: Any, q_heatmap: Any, eps: float = 1e-8) -> float:
    """Compute finite Jensen-Shannon divergence between two heatmaps."""
    p_dist, q_dist = _aligned_distributions(p_heatmap, q_heatmap, eps=eps)
    if not p_dist:
        return 0.0
    midpoint = [(p + q) * 0.5 for p, q in zip(p_dist, q_dist)]
    return 0.5 * heatmap_kl(p_dist, midpoint, eps=eps) + 0.5 * heatmap_kl(q_dist, midpoint, eps=eps)


def heatmap_entropy(p_heatmap: Any, eps: float = 1e-8) -> float:
    """Compute Shannon entropy of a normalized heatmap distribution."""
    distribution = normalize_heatmap_distribution(p_heatmap, eps=eps)
    return -sum(value * math.log(max(value, eps)) for value in distribution)


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


def score_candidate_distributional(
    candidate: dict,
    full_action_heatmap: Any,
    masked_action_heatmap: Any | None = None,
    negative_action_heatmaps: Any | None = None,
    full_target_heatmap: Any | None = None,
    negative_target_heatmaps: Any | None = None,
    weights: dict | None = None,
) -> float:
    """Score a candidate with verifier-free distributional heatmap signals."""
    effective_weights = dict(_DEFAULT_DISTRIBUTIONAL_WEIGHTS)
    if weights:
        effective_weights.update(weights)

    probability = _candidate_probability(full_action_heatmap, candidate)
    log_probability = math.log(probability)
    score = _as_float(effective_weights.get("log_probability"), 1.0) * log_probability

    if masked_action_heatmap is not None:
        if effective_weights.get("use_kl", True):
            condition_divergence = heatmap_kl(full_action_heatmap, masked_action_heatmap)
        else:
            condition_divergence = heatmap_js(full_action_heatmap, masked_action_heatmap)
        masked_probability = _candidate_probability(masked_action_heatmap, candidate)
        condition_margin = math.log(probability) - math.log(masked_probability)
        score += _as_float(effective_weights.get("condition_kl"), 0.25) * (condition_divergence + condition_margin)

    negative_divergences: list[float] = []
    for negative_heatmap in _as_heatmap_sequence(negative_action_heatmaps):
        if effective_weights.get("use_js", True):
            negative_divergences.append(heatmap_js(full_action_heatmap, negative_heatmap))
        else:
            negative_divergences.append(heatmap_kl(full_action_heatmap, negative_heatmap))
    if negative_divergences:
        score += _as_float(effective_weights.get("negative_action_divergence"), 0.25) * (
            sum(negative_divergences) / len(negative_divergences)
        )

    if full_target_heatmap is not None:
        target_consistency = score_candidate_target_consistency(candidate, full_action_heatmap, full_target_heatmap)
        score += _as_float(effective_weights.get("target_consistency"), 0.5) * target_consistency

        target_index = candidate.get("target_index")
        full_target_score = _target_score(full_target_heatmap, target_index)
        negative_target_scores = [
            _target_score(negative_target_heatmap, target_index)
            for negative_target_heatmap in _as_heatmap_sequence(negative_target_heatmaps)
        ]
        if negative_target_scores:
            target_margin = full_target_score - max(negative_target_scores)
        else:
            target_margin = full_target_score
        score += _as_float(effective_weights.get("target_margin"), 0.25) * target_margin

    entropy = heatmap_entropy(full_action_heatmap)
    score -= _as_float(effective_weights.get("entropy_penalty"), 0.05) * entropy
    return float(score)


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


def distributional_tca_select_inference(
    action_heatmap: Any,
    target_heatmap: Any | None = None,
    masked_action_heatmap: Any | None = None,
    negative_action_heatmaps: Any | None = None,
    negative_target_heatmaps: Any | None = None,
    K: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    weights: dict | None = None,
    use_kl: bool = True,
    use_js_optional: bool = True,
    metadata: dict | None = None,
    external_verifier: Any | None = None,
) -> dict:
    """Run Distributional TCA-Select with heatmap KL/JS and target margins."""
    if external_verifier is not None:
        raise ValueError("Distributional TCA-Select must not use external verifiers.")
    _ensure_no_privileged_metadata(metadata)

    scoring_weights = dict(weights or {})
    scoring_weights.setdefault("use_kl", use_kl)
    scoring_weights.setdefault("use_js", use_js_optional)

    candidates = sample_heatmap_candidates(action_heatmap, K=K, temperature=temperature)
    scores = [
        score_candidate_distributional(
            candidate,
            full_action_heatmap=action_heatmap,
            masked_action_heatmap=masked_action_heatmap,
            negative_action_heatmaps=negative_action_heatmaps,
            full_target_heatmap=target_heatmap,
            negative_target_heatmaps=negative_target_heatmaps,
            weights=scoring_weights,
        )
        for candidate in candidates
    ]
    selected = select_tca_candidate(candidates, scores)
    return {
        "method": "distributional_tca_select",
        "selected": selected,
        "candidates": candidates,
        "scores": scores,
        "K": K,
        "temperature": temperature,
        "use_kl": use_kl,
        "use_js_optional": use_js_optional,
        "external_verifier_used": False,
        "privileged_inference_used": False,
    }
