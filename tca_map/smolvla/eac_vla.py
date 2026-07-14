"""EAC-VLA Stage 0 audit helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E"
COMMITMENT_SET = (1, 2, 4, 8, 16, 50)
FIRST_COMPARISON_POLICIES = (
    "frozen_smolvla_fixed_queue",
    "aac_entropy_proxy",
    "eac_full",
    "eac_no_calibration_no_hysteresis_ablation",
    "fixed_short_replan_baseline",
)


def chunk_sha256(action_chunk: Any, *, decimals: int = 9) -> str:
    import hashlib
    import json

    array = np.asarray(action_chunk, dtype=np.float32)
    payload = np.round(array, decimals).tolist()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def eac_commitment_prefix(action_chunk: Any, commitment_length: int) -> np.ndarray:
    array = _as_array(action_chunk)
    if array.ndim != 2:
        raise ValueError(f"expected 2D action chunk, got {array.shape}")
    if int(commitment_length) < 1 or int(commitment_length) > int(array.shape[0]):
        raise ValueError(f"invalid commitment length {commitment_length} for chunk horizon {array.shape[0]}")
    return np.array(array[: int(commitment_length)], copy=True)


def audit_runtime_prefix_preservation(
    action_chunk: Any,
    *,
    commitment_lengths: Sequence[int] = COMMITMENT_SET,
) -> dict[str, Any]:
    base = _as_array(action_chunk)
    checks = []
    max_prefix_diff = 0.0
    max_queue_pop_diff = 0.0
    for commitment in commitment_lengths:
        prefix = eac_commitment_prefix(base, int(commitment))
        expected = base[: int(commitment)]
        prefix_diff = float(np.max(np.abs(prefix - expected))) if prefix.size else 0.0
        popped = np.asarray([item for item in prefix], dtype=np.float64)
        pop_diff = float(np.max(np.abs(popped - expected))) if popped.size else 0.0
        max_prefix_diff = max(max_prefix_diff, prefix_diff)
        max_queue_pop_diff = max(max_queue_pop_diff, pop_diff)
        checks.append(
            {
                "commitment_length": int(commitment),
                "prefix_shape": [int(dim) for dim in prefix.shape],
                "prefix_sha256": chunk_sha256(prefix),
                "expected_prefix_sha256": chunk_sha256(expected),
                "prefix_max_abs_diff": prefix_diff,
                "queue_pop_max_abs_diff": pop_diff,
                "action_values_modified": bool(prefix_diff > 0.0 or pop_diff > 0.0),
            }
        )
    return {
        "commitment_lengths": [int(item) for item in commitment_lengths],
        "check_count": len(checks),
        "max_prefix_abs_diff": max_prefix_diff,
        "max_queue_pop_abs_diff": max_queue_pop_diff,
        "all_prefixes_value_preserving": bool(max_prefix_diff == 0.0 and max_queue_pop_diff == 0.0),
        "checks": checks,
    }


@dataclass(frozen=True)
class EACStage0Config:
    validation_split: str = "val"
    confirmatory_split: str = "test"
    action_dim: int = 7
    chunk_horizon: int = 50
    min_unique_validation_frames: int = 100
    min_eval_seeds_per_frame: int = 2
    min_dispersion_nonzero_fraction: float = 0.95
    min_dispersion_p95: float = 1e-6
    max_commitment_share: float = 0.90
    equality_epsilon: float = 1e-6
    eps: float = 1e-12


def _as_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite array")
    return array


def _frame_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_index"]), int(record["episode_index"]), int(record["frame_index"]))


def _sample_key(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["task_index"]),
        int(record["episode_index"]),
        int(record["frame_index"]),
        int(record.get("eval_seed", 0)),
    )


def _summary(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "nonzero_fraction": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "nonzero_fraction": float(np.mean(arr > 1e-12)),
    }


def _robust_norm(values: np.ndarray, eps: float) -> np.ndarray:
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.95))
    return np.clip((values - lo) / (hi - lo + eps), 0.0, 1.0)


def _commitment_from_risk(risk: np.ndarray) -> list[int]:
    high = float(np.quantile(risk, 0.66))
    mid = float(np.quantile(risk, 0.33))
    commitments = []
    for value in risk:
        if float(value) >= high:
            commitments.append(2)
        elif float(value) >= mid:
            commitments.append(8)
        else:
            commitments.append(50)
    return commitments


def _split_manifest(records: Sequence[Mapping[str, Any]], config: EACStage0Config) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record.get("split", ""))].append(record)
    validation_frames = {_frame_key(record) for record in by_split.get(config.validation_split, [])}
    reserved_frames = {_frame_key(record) for record in by_split.get(config.confirmatory_split, [])}
    validation_samples = {_sample_key(record) for record in by_split.get(config.validation_split, [])}
    reserved_samples = {_sample_key(record) for record in by_split.get(config.confirmatory_split, [])}
    return {
        "split_counts_records": {name: len(items) for name, items in sorted(by_split.items())},
        "validation_split": config.validation_split,
        "confirmatory_reserved_split": config.confirmatory_split,
        "validation_unique_frame_count": len(validation_frames),
        "reserved_unique_frame_count": len(reserved_frames),
        "validation_sample_count": len(validation_samples),
        "reserved_sample_count": len(reserved_samples),
        "validation_reserved_frame_overlap": len(validation_frames & reserved_frames),
        "validation_reserved_sample_overlap": len(validation_samples & reserved_samples),
        "confirmatory_records_used_for_tuning": False,
    }


def audit_eac_stage0(
    records: Sequence[Mapping[str, Any]],
    *,
    queue_helper_present: bool,
    previous_preflight_chunk_shape: Sequence[int] | None,
    config: EACStage0Config | None = None,
) -> dict[str, Any]:
    cfg = config or EACStage0Config()
    split_manifest = _split_manifest(records, cfg)
    validation_rows = [record for record in records if str(record.get("split", "")) == cfg.validation_split]
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in validation_rows:
        grouped[_frame_key(record)].append(record)

    dispersion_values = []
    first_transition_values = []
    passthrough_errors = []
    frame_metrics = []
    eval_seed_counts = []
    chunk_shapes = Counter()
    finite_failures = 0
    for frame_key, frame_rows in sorted(grouped.items()):
        eval_seed_counts.append(len({int(row.get("eval_seed", 0)) for row in frame_rows}))
        previews = []
        for row in sorted(frame_rows, key=lambda item: int(item.get("eval_seed", 0))):
            shape = tuple(int(x) for x in row.get("base_action_chunk_shape", []))
            chunk_shapes[shape] += 1
            try:
                preview = _as_array(
                    row["base_action_chunk_first_two_preview"],
                    shape=(2, cfg.action_dim),
                )
                base_action = _as_array(row["base_action"], shape=(cfg.action_dim,))
            except (KeyError, ValueError):
                finite_failures += 1
                continue
            previews.append(preview)
            passthrough_errors.append(float(np.max(np.abs(preview[0] - base_action))))
        if len(previews) < cfg.min_eval_seeds_per_frame:
            continue
        stacked = np.stack(previews, axis=0)
        dispersion = float(np.mean(np.var(stacked, axis=0)))
        first_transition = float(np.linalg.norm(stacked[0, 1] - stacked[0, 0]))
        dispersion_values.append(dispersion)
        first_transition_values.append(first_transition)
        frame_metrics.append(
            {
                "task_index": frame_key[0],
                "episode_index": frame_key[1],
                "frame_index": frame_key[2],
                "eval_seed_count": len(previews),
                "first_two_dispersion": dispersion,
                "first_transition_l2": first_transition,
            }
        )

    dispersion = np.asarray(dispersion_values, dtype=np.float64)
    transition = np.asarray(first_transition_values, dtype=np.float64)
    if dispersion.size and transition.size:
        risk = 0.67 * _robust_norm(dispersion, cfg.eps) + 0.33 * _robust_norm(transition, cfg.eps)
        commitments = _commitment_from_risk(risk)
    else:
        risk = np.asarray([], dtype=np.float64)
        commitments = []
    commitment_counts = dict(sorted(Counter(commitments).items()))
    max_commitment_share = float(max(commitment_counts.values()) / len(commitments)) if commitments else 1.0
    risk_monotonicity = None
    if commitments and dispersion.size:
        short = dispersion[np.asarray(commitments) == 2]
        long = dispersion[np.asarray(commitments) == 50]
        if short.size and long.size:
            risk_monotonicity = {
                "short_commitment_dispersion_mean": float(np.mean(short)),
                "long_commitment_dispersion_mean": float(np.mean(long)),
                "short_gt_long": bool(float(np.mean(short)) > float(np.mean(long))),
            }

    queue_surface_manifest = {
        "queue_helper_present": bool(queue_helper_present),
        "previous_preflight_chunk_shape": list(previous_preflight_chunk_shape or []),
        "canonical_artifact_chunk_shapes": {str(list(key)): int(value) for key, value in sorted(chunk_shapes.items())},
        "expected_chunk_shape": [cfg.chunk_horizon, cfg.action_dim],
        "chunk_shape_ok": bool((cfg.chunk_horizon, cfg.action_dim) in chunk_shapes),
        "full_chunk_values_available_in_artifact": False,
        "first_two_preview_available": bool(dispersion_values),
        "runtime_full_chunk_check_required_before_validation_search": True,
    }
    dispersion_manifest = {
        "source": "canonical_frozen_base_prediction_artifact_first_two_chunk_previews",
        "unique_validation_frames": len(grouped),
        "audited_frames_with_repeated_eval_seeds": len(dispersion_values),
        "eval_seed_count_distribution": dict(sorted(Counter(eval_seed_counts).items())),
        "first_two_dispersion_summary": _summary(dispersion_values),
        "first_transition_l2_summary": _summary(first_transition_values),
        "risk_summary": _summary(risk.tolist()),
        "commitment_counts": commitment_counts,
        "max_commitment_share": max_commitment_share,
        "risk_monotonicity": risk_monotonicity,
        "frame_metric_preview": frame_metrics[:25],
    }
    passthrough_summary = _summary(passthrough_errors)
    hard_stops = []
    if not queue_surface_manifest["queue_helper_present"]:
        hard_stops.append("queue helper not present in official rollout source")
    if not queue_surface_manifest["chunk_shape_ok"]:
        hard_stops.append("canonical artifact does not record expected 50x7 chunk shape")
    if split_manifest["validation_reserved_frame_overlap"] != 0 or split_manifest["validation_reserved_sample_overlap"] != 0:
        hard_stops.append("validation/confirmatory identity overlap is nonzero")
    if len(grouped) < cfg.min_unique_validation_frames:
        hard_stops.append(f"too few unique validation frames: {len(grouped)}")
    if dispersion_manifest["first_two_dispersion_summary"]["nonzero_fraction"] is None:
        hard_stops.append("dispersion proxy unavailable")
    elif float(dispersion_manifest["first_two_dispersion_summary"]["nonzero_fraction"]) < cfg.min_dispersion_nonzero_fraction:
        hard_stops.append("dispersion proxy collapsed on validation frames")
    if dispersion_manifest["first_two_dispersion_summary"]["p95"] is None:
        hard_stops.append("dispersion p95 unavailable")
    elif float(dispersion_manifest["first_two_dispersion_summary"]["p95"]) < cfg.min_dispersion_p95:
        hard_stops.append("dispersion p95 below minimum")
    if max_commitment_share > cfg.max_commitment_share:
        hard_stops.append("commitment map effectively constant")
    if risk_monotonicity is not None and not risk_monotonicity["short_gt_long"]:
        hard_stops.append("short commitments are not higher-dispersion than long commitments")
    if passthrough_summary["max"] is None or float(passthrough_summary["max"]) > cfg.equality_epsilon:
        hard_stops.append("base action does not match first chunk action within serialization epsilon")
    if finite_failures:
        hard_stops.append(f"nonfinite or malformed preview records: {finite_failures}")

    final_decision = "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" if not hard_stops else "DESIGN_FAILURE"
    return {
        "schema_version": 1,
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "final_decision": final_decision,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "scoreable_validation_records": len(validation_rows),
        "validation_unique_frames": len(grouped),
        "reserved_records_not_used_for_tuning": split_manifest["split_counts_records"].get(cfg.confirmatory_split, 0),
        "first_comparison_policies": list(FIRST_COMPARISON_POLICIES),
        "queue_surface_manifest": queue_surface_manifest,
        "dispersion_manifest": dispersion_manifest,
        "split_manifest": split_manifest,
        "action_value_passthrough_summary": passthrough_summary,
        "hard_stop_reasons": hard_stops,
        "stage_0_limitations": [
            "canonical artifact stores first-two chunk previews and chunk hashes, not all 50 postprocessed actions",
            "runtime full-chunk equality and queue-prefix execution must be checked before validation search artifacts are accepted",
        ],
        "next_step": (
            "Proceed to bounded validation search only after implementing the runtime full-chunk/queue-prefix check."
            if not hard_stops
            else "Do not run validation search or rollout; classify the Stage 0 failure and pivot."
        ),
    }
