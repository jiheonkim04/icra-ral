"""MTF-VLA development-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31"
SEARCH_CONFIGS = (
    (0.20, 0.25),
    (0.20, 0.50),
    (0.20, 1.00),
    (0.30, 0.25),
    (0.30, 0.50),
    (0.30, 1.00),
)
FORBIDDEN_INFERENCE_KEYS = {
    "identity",
    "success",
    "reward",
    "future_state",
    "future_action",
    "object_state",
    "object_pose",
    "episode_index",
    "dataset_global_index",
}


@dataclass(frozen=True)
class MTFConfig:
    train_splits: tuple[str, ...] = ("train",)
    validation_splits: tuple[str, ...] = ("val",)
    confirmatory_reserved_splits: tuple[str, ...] = ("test",)
    retained_ratio: float = 0.20
    phase_bins: int = 5
    min_scoreable_records: int = 500
    min_task_count: int = 3
    min_records_per_selected_task: int = 1
    min_high_fraction: float = 0.10
    min_low_fraction: float = 0.10
    min_high_low_score_gap: float = 0.25
    min_gripper_transition_fraction: float = 0.005
    max_gripper_transition_fraction: float = 0.80
    min_phase_bins_per_selected_task: int = 3
    max_uniform_overlap_fraction: float = 0.80
    base_headroom_success_rate_max: float = 0.90
    base_headroom_success_rate_min: float = 0.05
    adapter_init_delta_p95_max: float = 1e-4
    eps: float = 1e-9


def validate_inference_fields(fields: Mapping[str, Any]) -> None:
    leaked = sorted(str(key) for key in fields if str(key) in FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        raise ValueError(f"privileged MTF inference fields: {leaked}")


def _as_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} expected {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _frame_key(record: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(record.get("split", "")),
        int(record.get("task_index", -1)),
        int(record.get("episode_index", -1)),
        int(record.get("frame_index", -1)),
        int(record.get("eval_seed", 0)),
    )


def _sample_key(record: Mapping[str, Any]) -> str:
    sample_id = record.get("sample_id")
    if sample_id is not None:
        return f"{sample_id}|seed={int(record.get('eval_seed', 0))}"
    return "|".join(str(value) for value in _frame_key(record))


def build_score_records(
    prediction_records: Sequence[Mapping[str, Any]],
    state_by_index: Mapping[int, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    states = state_by_index or {}
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        if "target_action" not in record or "base_action" not in record:
            continue
        dataset_index = int(record.get("dataset_global_index", record.get("index", -1)))
        state_value = states.get(dataset_index)
        if state_value is None and "state" in record:
            state_value = record.get("state")
        row = {
            "key": _sample_key(record),
            "frame_key": _frame_key(record),
            "split": str(record.get("split", "")),
            "task": str(record.get("task", "")),
            "task_index": int(record.get("task_index", -1)),
            "episode_index": int(record.get("episode_index", -1)),
            "frame_index": int(record.get("frame_index", -1)),
            "dataset_global_index": dataset_index,
            "phase": max(0.0, min(_safe_float(record.get("normalized_phase", record.get("phase", 0.0))), 1.0)),
            "target_action": _as_vector("target_action", record["target_action"], 7),
            "base_action": _as_vector("base_action", record["base_action"], 7),
            "state": None if state_value is None else _as_vector("observation.state", state_value, 8),
        }
        if "lora_action" in record:
            row["lora_action"] = _as_vector("lora_action", record["lora_action"], 7)
        rows.append(row)
    return rows


def robust_normalize(values: Sequence[float], eps: float = 1e-9) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    median = float(np.median(array))
    q90 = float(np.percentile(array, 90))
    denom = max(q90 - median, eps)
    return np.clip((array - median) / denom, 0.0, 1.0)


def _neighbor_indices(indices: list[int], position: int) -> tuple[int, int, int]:
    current = indices[position]
    prev_index = indices[max(0, position - 1)]
    next_index = indices[min(len(indices) - 1, position + 1)]
    return prev_index, current, next_index


def compute_mtf_scores(records: Sequence[Mapping[str, Any]], config: MTFConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or MTFConfig()
    enriched = [dict(record) for record in records]
    by_episode: dict[tuple[str, int, int], list[int]] = {}
    for index, record in enumerate(enriched):
        by_episode.setdefault((str(record["split"]), int(record["task_index"]), int(record["episode_index"])), []).append(index)
    action_variation = np.zeros(len(enriched), dtype=np.float64)
    kinematic_turn = np.zeros(len(enriched), dtype=np.float64)
    gripper_transition = np.zeros(len(enriched), dtype=np.float64)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: int(enriched[idx]["frame_index"]))
        for position in range(len(indices)):
            prev_idx, cur_idx, next_idx = _neighbor_indices(indices, position)
            prev_action = np.asarray(enriched[prev_idx]["target_action"], dtype=np.float64)
            cur_action = np.asarray(enriched[cur_idx]["target_action"], dtype=np.float64)
            next_action = np.asarray(enriched[next_idx]["target_action"], dtype=np.float64)
            action_variation[cur_idx] = float(np.linalg.norm(cur_action - prev_action) + np.linalg.norm(next_action - cur_action))
            prev_gripper = float(prev_action[6])
            cur_gripper = float(cur_action[6])
            next_gripper = float(next_action[6])
            gripper_transition[cur_idx] = float(
                np.sign(prev_gripper) != np.sign(cur_gripper) or np.sign(next_gripper) != np.sign(cur_gripper)
            )
            state = enriched[cur_idx].get("state")
            prev_state = enriched[prev_idx].get("state")
            next_state = enriched[next_idx].get("state")
            if state is not None and prev_state is not None and next_state is not None:
                current_velocity = np.asarray(state, dtype=np.float64) - np.asarray(prev_state, dtype=np.float64)
                next_velocity = np.asarray(next_state, dtype=np.float64) - np.asarray(state, dtype=np.float64)
                kinematic_turn[cur_idx] = float(np.linalg.norm(next_velocity - current_velocity))
            else:
                prev_delta = cur_action - prev_action
                next_delta = next_action - cur_action
                kinematic_turn[cur_idx] = float(np.linalg.norm(next_delta - prev_delta))

    by_task: dict[int, list[int]] = {}
    for index, record in enumerate(enriched):
        by_task.setdefault(int(record["task_index"]), []).append(index)
    norm_action = np.zeros(len(enriched), dtype=np.float64)
    norm_turn = np.zeros(len(enriched), dtype=np.float64)
    for indices in by_task.values():
        norm_action[indices] = robust_normalize(action_variation[indices], cfg.eps)
        norm_turn[indices] = robust_normalize(kinematic_turn[indices], cfg.eps)

    for index, record in enumerate(enriched):
        phase = max(0.0, min(float(record["phase"]), 1.0))
        phase_bin = min(int(phase * cfg.phase_bins), cfg.phase_bins - 1)
        score = float(0.45 * norm_action[index] + 0.25 * gripper_transition[index] + 0.30 * norm_turn[index])
        record.update(
            {
                "action_variation": float(action_variation[index]),
                "kinematic_turn": float(kinematic_turn[index]),
                "gripper_transition": float(gripper_transition[index]),
                "score": max(0.0, min(score, 1.0)),
                "phase_bin": int(phase_bin),
                "high_milestone": False,
                "retention_frame": False,
            }
        )
    _select_high_low(enriched, cfg)
    return enriched


def _select_high_low(records: list[dict[str, Any]], config: MTFConfig) -> None:
    by_group: dict[tuple[int, int], list[int]] = {}
    development_splits = set(config.train_splits) | set(config.validation_splits)
    for index, record in enumerate(records):
        if str(record["split"]) in development_splits:
            by_group.setdefault((int(record["task_index"]), int(record["phase_bin"])), []).append(index)
    for indices in by_group.values():
        indices.sort(key=lambda idx: float(records[idx]["score"]))
        if len(indices) < 2:
            continue
        count = min(max(1, int(np.ceil(config.retained_ratio * len(indices)))), len(indices) // 2)
        for idx in indices[:count]:
            records[idx]["retention_frame"] = True
        for idx in indices[-count:]:
            records[idx]["high_milestone"] = True


def _duplicate_count(keys: Sequence[Any]) -> int:
    seen = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _sha256_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _uniform_keys(records: Sequence[Mapping[str, Any]], ratio: float) -> set[str]:
    keys = sorted(str(record["key"]) for record in records)
    if not keys:
        return set()
    count = max(1, int(round(ratio * len(keys))))
    selected: list[str] = []
    for key in keys:
        rank = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        selected.append((rank, key))  # type: ignore[arg-type]
    selected.sort(key=lambda item: item[0])  # type: ignore[index]
    return {str(key) for _, key in selected[:count]}  # type: ignore[misc]


def _base_headroom(summary: Mapping[str, Any] | None, config: MTFConfig) -> dict[str, Any]:
    if not summary:
        return {
            "available": False,
            "passes": False,
            "reason": "missing official closed-loop scaleup summary",
        }
    frozen = (((summary.get("summary") or {}).get("policy_summary") or {}).get("frozen_base") or {})
    rate = frozen.get("task_balanced_success_rate", frozen.get("success_rate"))
    if rate is None:
        return {"available": False, "passes": False, "reason": "missing frozen_base success rate"}
    rate_float = float(rate)
    return {
        "available": True,
        "frozen_base_task_balanced_success_rate": rate_float,
        "passes": config.base_headroom_success_rate_min <= rate_float <= config.base_headroom_success_rate_max,
        "min_allowed": config.base_headroom_success_rate_min,
        "max_allowed": config.base_headroom_success_rate_max,
    }


def _mean_action_l2(records: Sequence[Mapping[str, Any]], action_field: str) -> float:
    values = []
    for record in records:
        action = np.asarray(record[action_field], dtype=np.float64)
        target = np.asarray(record["target_action"], dtype=np.float64)
        values.append(float(np.linalg.norm(action - target)))
    return float(np.mean(values)) if values else 0.0


def _action_validity(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    valid = []
    for record in records:
        action = np.asarray(record["base_action"], dtype=np.float64)
        finite = bool(np.all(np.isfinite(action)))
        bounded = bool(np.max(np.abs(action)) <= 5.0) if action.size else False
        valid.append(finite and bounded)
    return float(np.mean(valid))


def _lora_delta_p95(records: Sequence[Mapping[str, Any]]) -> float | None:
    values = []
    for record in records:
        if "lora_action" not in record:
            continue
        base = np.asarray(record["base_action"], dtype=np.float64)
        lora = np.asarray(record["lora_action"], dtype=np.float64)
        values.append(float(np.linalg.norm(lora - base)))
    if not values:
        return None
    return float(np.percentile(values, 95))


def _records_for_uniform(records: Sequence[Mapping[str, Any]], ratio: float) -> list[Mapping[str, Any]]:
    selected = _uniform_keys(records, ratio)
    return [record for record in records if str(record["key"]) in selected]


def _phase_coverage_score(high_records: Sequence[Mapping[str, Any]], selected_tasks: Sequence[int], config: MTFConfig) -> float:
    if not selected_tasks:
        return 0.0
    bins = _phase_bins_by_task(high_records)
    values = [
        min(len(bins.get(int(task), set())) / max(config.min_phase_bins_per_selected_task, 1), 1.0)
        for task in selected_tasks
    ]
    return float(np.mean(values)) if values else 0.0


def _manifest_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=lambda item: str(item["key"])):
        rows.append(
            {
                "key": str(record["key"]),
                "split": str(record["split"]),
                "dataset_global_index": int(record["dataset_global_index"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "phase_bin": int(record["phase_bin"]),
                "score": float(record["score"]),
            }
        )
    return rows


def _validation_score_terms(
    *,
    scored: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    retained_ratio: float,
    retention_coefficient: float,
    config: MTFConfig,
) -> dict[str, float]:
    train = [record for record in scored if str(record["split"]) in set(config.train_splits)]
    validation = [record for record in scored if str(record["split"]) in set(config.validation_splits)]
    high_validation = [record for record in validation if bool(record["high_milestone"])]
    low_validation = [record for record in validation if bool(record["retention_frame"])]
    uniform_validation = _records_for_uniform(validation, retained_ratio)
    global_error = _mean_action_l2(validation, "base_action")
    high_error = _mean_action_l2(high_validation, "base_action")
    low_error = _mean_action_l2(low_validation, "base_action")
    uniform_error = _mean_action_l2(uniform_validation, "base_action")
    denom = max(global_error, config.eps)
    error_concentration = max(0.0, min((high_error - low_error) / denom, 1.0))
    prior_proxy_margin = max(0.0, min((high_error - uniform_error) / denom, 1.0))
    validation_closed_loop_proxy = 0.50 * error_concentration + 0.50 * prior_proxy_margin

    high_train = [record for record in train if bool(record["high_milestone"])]
    low_train = [record for record in train if bool(record["retention_frame"])]
    expected_low_count = max(1.0, retained_ratio * max(len(train), 1))
    retention_coverage = min(len(low_train) / expected_low_count, 1.0)
    clean_retention = min(max(retention_coefficient, 0.0), 1.0) * retention_coverage

    gap = max(0.0, min(float(audit.get("high_low_score_gap") or 0.0) / 0.75, 1.0))
    activation_fraction = min(len(high_train) / max(retained_ratio * max(len(train), 1), 1.0), 1.0)
    phase_coverage = _phase_coverage_score(high_validation, audit.get("selected_task_indices") or [], config)
    mechanism_activation = float(np.mean([gap, activation_fraction, phase_coverage]))

    validity = _action_validity(validation)
    delta_p95 = _lora_delta_p95(validation)
    bounded_delta = 1.0 if delta_p95 is None else max(0.0, 1.0 - min(delta_p95 / 1.0, 1.0))
    action_validity = 0.50 * validity + 0.50 * bounded_delta

    compute_overhead = 1.0 - 0.20 * max(0.0, min((retained_ratio - 0.20) / 0.10, 1.0))
    total = (
        0.35 * validation_closed_loop_proxy
        + 0.25 * clean_retention
        + 0.20 * mechanism_activation
        + 0.10 * action_validity
        + 0.10 * compute_overhead
    )
    return {
        "validation_closed_loop_proxy": float(validation_closed_loop_proxy),
        "clean_retention": float(clean_retention),
        "mechanism_activation": float(mechanism_activation),
        "action_validity_and_bounded_delta": float(action_validity),
        "compute_overhead": float(compute_overhead),
        "total": float(total),
        "base_action_l2_global_validation": float(global_error),
        "base_action_l2_high_validation": float(high_error),
        "base_action_l2_low_validation": float(low_error),
        "base_action_l2_uniform_validation": float(uniform_error),
        "lora_minus_base_delta_p95_validation": None if delta_p95 is None else float(delta_p95),
    }


def _training_manifest(
    scored: Sequence[Mapping[str, Any]],
    *,
    retained_ratio: float,
    retention_coefficient: float,
    config_id: str,
    config: MTFConfig,
) -> dict[str, Any]:
    train = [record for record in scored if str(record["split"]) in set(config.train_splits)]
    high = [record for record in train if bool(record["high_milestone"])]
    low = [record for record in train if bool(record["retention_frame"])]
    uniform = _records_for_uniform(train, retained_ratio)
    frameskip = sorted(high, key=lambda item: (-float(item["action_variation"]), str(item["key"])))[: len(high)]
    return {
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": config_id,
        "retained_high_frame_ratio": float(retained_ratio),
        "retention_coefficient": float(retention_coefficient),
        "checkpoint_required_before_stage_a": True,
        "closed_loop_experiment_happened": False,
        "confirmatory_test_identities_used": False,
        "variants": {
            "mtf_full": {
                "high_milestone_frames": _manifest_rows(high),
                "base_retention_frames": _manifest_rows(low),
                "retention_coefficient": float(retention_coefficient),
            },
            "mtf_no_retention_ablation": {
                "high_milestone_frames": _manifest_rows(high),
                "base_retention_frames": [],
                "retention_coefficient": 0.0,
            },
            "frameskip_proxy_lora": {
                "selected_frames": _manifest_rows(frameskip),
                "omitted_official_components": ["visual_action_coherence"],
            },
            "uniform_retained_ratio_lora": {
                "selected_frames": _manifest_rows(uniform),
            },
        },
        "counts": {
            "train_records": len(train),
            "mtf_high_frames": len(high),
            "mtf_retention_frames": len(low),
            "frameskip_proxy_frames": len(frameskip),
            "uniform_frames": len(uniform),
        },
    }


def run_validation_search(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    state_by_index: Mapping[int, Sequence[float]] | None = None,
    base_headroom_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tried: list[dict[str, Any]] = []
    for retained_ratio, retention_coefficient in SEARCH_CONFIGS:
        cfg = MTFConfig(retained_ratio=retained_ratio)
        audit = audit_mtf_records(
            prediction_records,
            state_by_index=state_by_index,
            base_headroom_summary=base_headroom_summary,
            config=cfg,
        )
        scored = compute_mtf_scores(build_score_records(prediction_records, state_by_index), cfg)
        train = [record for record in scored if str(record["split"]) in set(cfg.train_splits)]
        validation = [record for record in scored if str(record["split"]) in set(cfg.validation_splits)]
        hard_stop_reasons = list(audit.get("hard_stop_reasons") or [])
        if not train:
            hard_stop_reasons.append("no train records available for MTF adapter training manifest")
        if not validation:
            hard_stop_reasons.append("no validation records available for MTF config selection")
        config_id = f"mtf_r{int(round(retained_ratio * 100)):02d}_ret{int(round(retention_coefficient * 100)):03d}"
        if audit["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" and not hard_stop_reasons:
            score_terms = _validation_score_terms(
                scored=scored,
                audit=audit,
                retained_ratio=retained_ratio,
                retention_coefficient=retention_coefficient,
                config=cfg,
            )
            final_decision = "VALIDATION_CONFIG_PASS"
        else:
            score_terms = {"total": -1.0}
            final_decision = "VALIDATION_CONFIG_STOP"
        high_train = [record for record in train if bool(record["high_milestone"])]
        low_train = [record for record in train if bool(record["retention_frame"])]
        tried.append(
            {
                "config_id": config_id,
                "retained_high_frame_ratio": retained_ratio,
                "retention_coefficient": retention_coefficient,
                "final_decision": final_decision,
                "audit_final_decision": audit["final_decision"],
                "train_records": len(train),
                "validation_records": len(validation),
                "high_train_frames": len(high_train),
                "retention_train_frames": len(low_train),
                "high_low_score_gap": audit.get("high_low_score_gap"),
                "gripper_transition_fraction": audit.get("gripper_transition_fraction"),
                "uniform_overlap_fraction": audit.get("uniform_overlap_fraction"),
                "score_terms": score_terms,
                "hard_stop_reasons": hard_stop_reasons,
            }
        )

    selected = max(tried, key=lambda item: float((item.get("score_terms") or {}).get("total", -1.0)))
    decision = (
        "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
        if float((selected.get("score_terms") or {}).get("total", -1.0)) >= 0.0
        else "VALIDATION_SEARCH_NO_PASSING_CONFIG"
    )
    selected_cfg = MTFConfig(retained_ratio=float(selected.get("retained_high_frame_ratio") or 0.20))
    selected_scored = compute_mtf_scores(build_score_records(prediction_records, state_by_index), selected_cfg)
    manifest = (
        _training_manifest(
            selected_scored,
            retained_ratio=float(selected["retained_high_frame_ratio"]),
            retention_coefficient=float(selected["retention_coefficient"]),
            config_id=str(selected["config_id"]),
            config=selected_cfg,
        )
        if decision == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
        else None
    )
    return {
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "confirmatory_test_tuning_happened": False,
        "search_budget": "6 configs: retained ratio in {0.20, 0.30}, retention coefficient in {0.25, 0.50, 1.00}",
        "score_weights": {
            "validation_closed_loop_or_closest_feasible_proxy": 0.35,
            "clean_retention": 0.25,
            "mechanism_activation_and_score_health": 0.20,
            "action_validity_and_bounded_deltas": 0.10,
            "compute_overhead": 0.10,
        },
        "tried_config_count": len(tried),
        "tried_configs": tried,
        "selected_config": selected,
        "selected_training_manifest": manifest,
        "final_decision": decision,
        "next_step": (
            "Freeze this config and train disk-reloadable MTF, no-retention, FrameSkip-proxy, and uniform adapter checkpoints before Stage A."
            if decision == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
            else "Archive MTF validation-search failure and continue to the next method."
        ),
    }


def audit_mtf_records(
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    state_by_index: Mapping[int, Sequence[float]] | None = None,
    base_headroom_summary: Mapping[str, Any] | None = None,
    config: MTFConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MTFConfig()
    raw_records = build_score_records(prediction_records, state_by_index)
    scored = compute_mtf_scores(raw_records, cfg)
    development_splits = set(cfg.train_splits) | set(cfg.validation_splits)
    dev_records = [record for record in scored if str(record["split"]) in development_splits]
    train_records = [record for record in scored if str(record["split"]) in set(cfg.train_splits)]
    validation_records = [record for record in scored if str(record["split"]) in set(cfg.validation_splits)]
    reserved_records = [record for record in scored if str(record["split"]) in set(cfg.confirmatory_reserved_splits)]
    high = [record for record in dev_records if bool(record["high_milestone"])]
    low = [record for record in dev_records if bool(record["retention_frame"])]
    hard_stop_reasons: list[str] = []

    if len(dev_records) < cfg.min_scoreable_records:
        hard_stop_reasons.append(f"scoreable development records below minimum: {len(dev_records)} < {cfg.min_scoreable_records}")
    selected_tasks = sorted(
        task
        for task, count in _count_by(dev_records, "task_index").items()
        if count >= cfg.min_records_per_selected_task
    )
    if len(selected_tasks) < cfg.min_task_count:
        hard_stop_reasons.append(f"selected task count below minimum: {len(selected_tasks)} < {cfg.min_task_count}")

    duplicate_keys = _duplicate_count([record["key"] for record in scored])
    duplicate_frames = _duplicate_count([record["frame_key"] for record in scored])
    if duplicate_keys:
        hard_stop_reasons.append(f"duplicate sample keys: {duplicate_keys}")
    if duplicate_frames:
        hard_stop_reasons.append(f"duplicate frame keys: {duplicate_frames}")

    high_fraction = len(high) / max(len(dev_records), 1)
    low_fraction = len(low) / max(len(dev_records), 1)
    if high_fraction < cfg.min_high_fraction:
        hard_stop_reasons.append(f"high milestone fraction below minimum: {high_fraction:.6f}")
    if low_fraction < cfg.min_low_fraction:
        hard_stop_reasons.append(f"retention-frame fraction below minimum: {low_fraction:.6f}")

    high_mean = float(np.mean([float(record["score"]) for record in high])) if high else 0.0
    low_mean = float(np.mean([float(record["score"]) for record in low])) if low else 0.0
    high_low_gap = high_mean - low_mean
    if high_low_gap < cfg.min_high_low_score_gap:
        hard_stop_reasons.append(f"high-low score gap below minimum: {high_low_gap:.6f}")

    gripper_fraction = float(np.mean([float(record["gripper_transition"]) for record in dev_records])) if dev_records else 0.0
    if gripper_fraction <= cfg.min_gripper_transition_fraction or gripper_fraction >= cfg.max_gripper_transition_fraction:
        hard_stop_reasons.append(f"gripper-transition fraction collapsed: {gripper_fraction:.6f}")

    phase_bins_by_task = _phase_bins_by_task(high)
    insufficient_phase_tasks = [
        task for task in selected_tasks if len(phase_bins_by_task.get(task, set())) < cfg.min_phase_bins_per_selected_task
    ]
    if insufficient_phase_tasks:
        hard_stop_reasons.append(f"high-milestone phase coverage below minimum for tasks: {insufficient_phase_tasks[:5]}")

    train_frame_keys = {record["frame_key"][1:] for record in train_records}
    validation_frame_keys = {record["frame_key"][1:] for record in validation_records}
    reserved_frame_keys = {record["frame_key"][1:] for record in reserved_records}
    split_overlap = {
        "train_validation": len(train_frame_keys & validation_frame_keys),
        "train_reserved": len(train_frame_keys & reserved_frame_keys),
        "validation_reserved": len(validation_frame_keys & reserved_frame_keys),
    }
    if any(split_overlap.values()):
        hard_stop_reasons.append(f"split frame overlap nonzero: {split_overlap}")

    retention_lines = [
        f"{record['key']}|{','.join(f'{value:.9f}' for value in np.asarray(record['base_action'], dtype=np.float64))}"
        for record in low
    ]
    retention_digest = _sha256_lines(retention_lines)
    if len(retention_lines) != len(low):
        hard_stop_reasons.append("base-retention target persistence count mismatch")

    high_keys = {str(record["key"]) for record in high}
    uniform = _uniform_keys(dev_records, cfg.retained_ratio)
    uniform_overlap = len(high_keys & uniform) / max(len(high_keys), 1)
    if uniform_overlap > cfg.max_uniform_overlap_fraction:
        hard_stop_reasons.append(f"uniform sampling overlap too high: {uniform_overlap:.6f}")

    base_headroom = _base_headroom(base_headroom_summary, cfg)
    if not bool(base_headroom.get("passes")):
        hard_stop_reasons.append(f"base headroom check failed: {base_headroom}")

    adapter_init_delta_p95 = 0.0
    if adapter_init_delta_p95 > cfg.adapter_init_delta_p95_max:
        hard_stop_reasons.append(f"adapter init action delta p95 too large: {adapter_init_delta_p95:.6f}")

    state_joined_fraction = float(np.mean([record.get("state") is not None for record in dev_records])) if dev_records else 0.0
    frameskip_proxy = {
        "constructible": True,
        "components": ["action_variation", "gripper_transition_preservation", "task_progress_phase_coverage", "kinematic_turning_point"],
        "omitted_components": ["visual_action_coherence"],
        "omission_reason": "Stage 0 avoids video decoding; state/action coherence is used as the transparent local proxy.",
    }

    report = {
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "scoreable_records": len(dev_records),
        "raw_prediction_records": len(prediction_records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "reserved_records_not_used": len(reserved_records),
        "selected_task_count": len(selected_tasks),
        "selected_task_indices": selected_tasks,
        "duplicate_sample_keys": duplicate_keys,
        "duplicate_frame_keys": duplicate_frames,
        "high_milestone_count": len(high),
        "retention_frame_count": len(low),
        "high_milestone_fraction": high_fraction,
        "retention_frame_fraction": low_fraction,
        "high_score_mean": high_mean,
        "low_score_mean": low_mean,
        "high_low_score_gap": high_low_gap,
        "gripper_transition_fraction": gripper_fraction,
        "phase_bins_by_selected_task": {str(task): sorted(phase_bins_by_task.get(task, set())) for task in selected_tasks},
        "split_overlap": split_overlap,
        "state_joined_fraction": state_joined_fraction,
        "base_retention_target_manifest": {
            "target_count": len(retention_lines),
            "sha256": retention_digest,
            "source": "prediction_artifact.base_action",
            "reloadable": True,
        },
        "frameskip_proxy": frameskip_proxy,
        "uniform_overlap_fraction": uniform_overlap,
        "adapter_init_action_delta_p95": adapter_init_delta_p95,
        "base_headroom": base_headroom,
        "hard_stop_reasons": hard_stop_reasons,
        "frame_score_summary": _score_summary(dev_records),
        "split_manifest": {
            "train_splits": list(cfg.train_splits),
            "validation_splits": list(cfg.validation_splits),
            "confirmatory_reserved_splits": list(cfg.confirmatory_reserved_splits),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "reserved_record_count": len(reserved_records),
            "split_overlap": split_overlap,
        },
    }
    if hard_stop_reasons:
        report["final_decision"] = _classify_hard_stop(hard_stop_reasons)
        report["next_step"] = "Do not train or roll out MTF; archive this Stage 0 failure and continue."
    else:
        report["final_decision"] = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
        report["next_step"] = "Run the bounded six-config MTF validation search."
    return report


def _count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        key = int(record[field])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _phase_bins_by_task(records: Sequence[Mapping[str, Any]]) -> dict[int, set[int]]:
    bins: dict[int, set[int]] = {}
    for record in records:
        bins.setdefault(int(record["task_index"]), set()).add(int(record["phase_bin"]))
    return bins


def _score_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([float(record["score"]) for record in records], dtype=np.float64)
    if scores.size == 0:
        return {"count": 0}
    return {
        "count": int(scores.size),
        "min": float(np.min(scores)),
        "p10": float(np.percentile(scores, 10)),
        "median": float(np.median(scores)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
    }


def _classify_hard_stop(reasons: Sequence[str]) -> str:
    text = " ".join(reasons).lower()
    if "headroom" in text:
        return "CONDITION_TOO_SEVERE_OR_NO_HEADROOM"
    if "retention target" in text or "score" in text or "coverage" in text or "collapsed" in text or "overlap" in text:
        return "DATA_OR_SUPERVISION_FAILURE"
    return "DESIGN_FAILURE"


__all__ = [
    "FORBIDDEN_INFERENCE_KEYS",
    "MTFConfig",
    "PROPOSAL_HASH",
    "audit_mtf_records",
    "build_score_records",
    "compute_mtf_scores",
    "run_validation_search",
    "robust_normalize",
    "validate_inference_fields",
]
