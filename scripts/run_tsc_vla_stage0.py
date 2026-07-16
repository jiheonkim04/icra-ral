"""Run the frozen TSC-VLA Stage 0 temporal-spatial completion audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.tsc_vla import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    DIAGNOSTIC_ALPHA,
    HEADROOM_ABSOLUTE_HUBER_GATE,
    HEADROOM_RELATIVE_GATE,
    MASK_QUANTILE,
    MASK_THRESHOLD,
    PHASE_BINS,
    PROPOSAL_HASH,
    RIDGE_COEFFICIENT,
    STD_FLOOR,
    TASK_COUNT,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_chunk,
    action_delta_summary,
    apply_discovery_zscore,
    apply_masked_completion,
    binary_cross_entropy,
    canonical_json_sha256,
    classify_stage0,
    deterministic_random_mask,
    fit_completion_model,
    fit_discovery_zscore,
    fit_magnitude_mask_baseline,
    fit_mask_label_stats,
    fit_structured_mask_probe,
    flattened_chunks,
    hard_mask,
    json_default,
    make_error_mask_labels,
    mean_huber,
    phase_bin,
    predict_completion_residual,
    predict_magnitude_mask_probability,
    predict_structured_mask_scores,
    prediction_metrics,
    raw_tsc_feature,
    trivial_mask_probability,
    tsc_feature_key,
    tsc_row_key,
    unselected_clamp_error,
    validate_manifest,
)
from scripts.run_cfr_vla_stage0 import (  # noqa: E402
    _array_sha256,
    _asset_path,
    _base_chunk_path,
    _edge_hash,
    _evenly_spaced,
    _load_base_chunk,
    _load_feature,
    _load_or_decode_base_chunk,
    _load_or_extract_feature,
    _problem_language,
    _proprio_from_obs,
    _read_json,
    _sha256,
    _utc_now,
    _write_json,
    _write_text,
)
from scripts.run_famr_vla_stage0 import (  # noqa: E402
    VLM_PATH,
    _active_linux_workers,
    _hash_base_parameters,
    _load_policy_and_processors,
    _loss,
    _preprocess,
    _raw_sample,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_vdr_vla_stage0a import (  # noqa: E402
    _core_policy,
    _decoded_chunk,
    _gradient_values,
    _native_velocity,
)


POLICY_PROBE = "tsc_stage0_temporal_spatial_masked_completion"
SEED = 20262800
FLOW_TIME = 0.5
DISCOVERY_ROWS_PER_TASK = 128
VALIDATION_ROWS_PER_TASK = 32
GRADIENT_RATIO_MAX = 100.0
PROPOSAL_FILE = REPO_ROOT / "reports" / "tsc_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "tsc_vla" / "proposal_hash.txt"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"

TASK_SOURCES = (
    (
        "libero_spatial",
        "libero_spatial/task_3",
        "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
    ),
    (
        "libero_object",
        "libero_object/task_3",
        "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5",
    ),
    (
        "libero_goal",
        "libero_goal/task_5",
        "libero_goal/put_the_bowl_on_top_of_the_cabinet_demo.hdf5",
    ),
    (
        "libero_10",
        "libero_10/task_5",
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
    ),
)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "feature_dir": run / "visual_features",
        "base_chunk_dir": run / "base_chunks",
        "adapter_dir": run / "identity_adapter",
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "pid": report / "stage_0_pid.txt",
        "heartbeat": report / "stage_0_heartbeat.json",
        "status": report / "stage_0_status.json",
        "preflight": report / "stage_0_preflight.json",
        "official_prior_asset_check": report / "stage_0_official_prior_asset_check.json",
        "action_semantics": report / "stage_0_action_semantics.json",
        "serializer_preflight": report / "stage_0_serializer_preflight.json",
        "manifest": report / "stage_0_manifest.json",
        "partial": report / "stage_0_partial.json",
        "result_json": report / "stage_0_result.json",
        "result_md": report / "stage_0_result.md",
        "validation": report / "stage_0_validation.json",
        "adjudication": report / "stage_0_adjudication.md",
        "blocker": report / "stage_0_implementation_blocker.json",
        "exit_code": report / "stage_0_exit_code.txt",
    }


def _proposal_hash_text() -> str:
    if not PROPOSAL_HASH_FILE.is_file():
        return ""
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _serializer_preflight(path: Path) -> dict[str, Any]:
    manifest_row = {
        "partition": "validation",
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": 8,
        "frame_index": 3,
        "proxy_variant": "tsc_full",
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = tsc_row_key(manifest_row)
    fixture = {
        "method": "TSC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "mask_quantile": np.float32(MASK_QUANTILE),
        "mask_threshold": np.float32(MASK_THRESHOLD),
        "diagnostic_alpha": np.float32(DIAGNOSTIC_ALPHA),
        "base_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
        "mask": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=bool),
        "decision_inputs": Stage0DecisionInputs(
            proposal_hash_ok=True,
            serializer_preflight_ok=True,
            official_prior_asset_check_persisted=True,
            manifest_integrity_ok=True,
            source_alignment_ok=True,
            feature_action_proprio_finite_aligned=True,
            minimum_discovery_windows=512,
            minimum_validation_windows=128,
            all_tasks_reported=True,
            maximum_validation_task_fraction=0.25,
            labels_noncollapsed_discovery=True,
            labels_noncollapsed_validation=True,
            structured_mask_beats_trivial=True,
            structured_mask_beats_magnitude=True,
            tsc_beats_prior_relative=0.05,
            tsc_beats_prior_absolute_huber=0.0,
            tsc_beats_ablation_relative=0.05,
            tsc_beats_ablation_absolute_huber=0.0,
            unselected_cell_clamp_max_error=0.0,
            changed_cell_fraction=0.10,
            deltas_finite_and_bounded=True,
            tsc_distinct_from_prior_and_ablation=True,
            finite_objectives_and_gradients=True,
            tsc_gradient_nonzero=True,
            gradient_ratio_at_most_100=True,
            frozen_parameter_gradient_count=0,
            identity_max_error=0.0,
            base_hash_unchanged=True,
            checkpoint_reload_ok=True,
            action_validity_ok=True,
            reward_read_count=0,
            success_read_count=0,
            done_read_count=0,
            confirmatory_records_read=0,
            exception_count=0,
        ).__dict__,
    }
    fixture["decision"] = classify_stage0(Stage0DecisionInputs(**fixture["decision_inputs"]))
    fixture_hash = canonical_json_sha256(fixture)
    _write_json(path, {"fixture": fixture, "fixture_hash": fixture_hash, "written_at": _utc_now()})
    parsed = _read_json(path)
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        **parsed,
        "parsed": True,
        "reproduced_hash": reproduced,
        "passed": reproduced == fixture_hash and parsed.get("fixture_hash") == fixture_hash,
    }
    _write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("TSC serializer preflight hash did not reproduce")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = {
        "official_repository": REPO_ROOT / "third_party" / "TS-Mask-VLA",
        "asset_repository": _asset_path("TS-Mask-VLA"),
        "checkpoint_dir": _asset_path("checkpoints", "ts_mask_vla"),
        "inference_code": _asset_path("ts_mask_vla", "inference"),
        "training_code": _asset_path("ts_mask_vla", "training"),
    }
    observed = {name: str(value) for name, value in candidates.items()}
    exists = {name: value.exists() for name, value in candidates.items()}
    official_ready = bool(exists["official_repository"] and exists["checkpoint_dir"] and exists["inference_code"])
    label = "ts_mask_vla_official" if official_ready else "ts_mask_continuous_proxy"
    deviations = [] if official_ready else [
        "official TS-Mask VLA repository/checkpoint/inference assets are not all locally verified",
        "Stage 0 fixes policy 2 as a transparent local continuous TS-Mask-style proxy until official assets are installed",
    ]
    result = {
        "method": "TSC-VLA",
        "closest_prior": "TS-Mask VLA",
        "official_ready": official_ready,
        "policy_2_label": label,
        "observed_paths": observed,
        "path_exists": exists,
        "deviations": deviations,
        "checked_at": _utc_now(),
    }
    _write_json(path, result)
    return result


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import h5py

    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task_index, (suite, task_identity, relative) in enumerate(TASK_SOURCES):
        source = data_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = _edge_hash(source)
        demo_reports = []
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            for demo_id in range(10):
                demo_key = f"demo_{demo_id}"
                if demo_key not in data:
                    raise KeyError(f"missing {demo_key} in {source}")
                demo = data[demo_key]
                actions = np.asarray(demo["actions"], dtype=np.float64)
                observations = demo["obs"]
                required = ("agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states")
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action array {actions.shape} in {source}:{demo_key}")
                if any(name not in observations for name in required):
                    raise KeyError(f"missing required observation in {source}:{demo_key}")
                if any(len(observations[name]) != len(actions) for name in required):
                    raise ValueError(f"observation/action length mismatch in {source}:{demo_key}")
                states = np.asarray(observations["ee_states"], dtype=np.float64)
                gripper = np.asarray(observations["gripper_states"], dtype=np.float64)
                if states.shape[:2] != (len(actions), 6) or not np.isfinite(states).all():
                    raise ValueError(f"invalid ee_states {states.shape} in {source}:{demo_key}")
                if gripper.shape[0] != len(actions) or not np.isfinite(gripper).all():
                    raise ValueError(f"invalid gripper_states {gripper.shape} in {source}:{demo_key}")
                valid_count = len(actions) - CHUNK_SIZE + 1
                partition = "discovery" if demo_id <= 7 else "validation"
                demo_reports.append({"demo_id": demo_id, "partition": partition, "length": len(actions), "valid_count": valid_count})
                for frame in range(max(0, valid_count)):
                    phase = frame / max(valid_count - 1, 1)
                    row: dict[str, Any] = {
                        "partition": partition,
                        "suite": suite,
                        "task_identity": task_identity,
                        "task_index": task_index,
                        "task_language": language,
                        "source_path": str(source),
                        "source_edge_sha256": source_hash,
                        "demo_id": demo_id,
                        "episode": demo_id,
                        "frame_index": frame,
                        "frame": frame,
                        "chunk_size": CHUNK_SIZE,
                        "proxy_variant": "tsc_full",
                        "policy_probe": POLICY_PROBE,
                        "episode_length": int(len(actions)),
                        "phase": float(phase),
                        "phase_bin": phase_bin(phase),
                    }
                    row["row_key"] = tsc_row_key(row)
                    row["feature_key"] = tsc_feature_key(row)
                    candidates.append(row)
        sources.append(
            {
                "suite": suite,
                "task_identity": task_identity,
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "edge_sha256": source_hash,
                "language": language,
                "demonstrations": demo_reports,
            }
        )

    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault((str(row["partition"]), str(row["task_identity"])), []).append(row)
    for group in sorted(groups):
        partition, _ = group
        target_count = DISCOVERY_ROWS_PER_TASK if partition == "discovery" else VALIDATION_ROWS_PER_TASK
        ordered = sorted(groups[group], key=lambda item: (int(item["demo_id"]), int(item["frame_index"])))
        selected = _evenly_spaced(ordered, target_count)
        if len(selected) != target_count:
            raise RuntimeError(f"TSC manifest group {group} has {len(selected)} rows, expected {target_count}")
        rows.extend(selected)
    rows.sort(
        key=lambda row: (
            row["partition"],
            row["suite"],
            row["task_identity"],
            int(row["demo_id"]),
            int(row["frame_index"]),
        )
    )
    return rows, sources


def _partial_row(
    row: Mapping[str, Any],
    feature_path: Path,
    feature: np.ndarray,
    base_chunk_path: Path,
    base_chunk: np.ndarray,
) -> dict[str, Any]:
    import h5py

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        chunk = action_chunk(np.asarray(demo["actions"], dtype=np.float64), int(row["frame_index"]))
        observations = demo["obs"]
        proprio = _proprio_from_obs(observations, int(row["frame_index"]))
    return {
        "row_key": str(row["row_key"]),
        "partition": str(row["partition"]),
        "suite": str(row["suite"]),
        "task_identity": str(row["task_identity"]),
        "source_edge_sha256": str(row["source_edge_sha256"]),
        "demo_id": int(row["demo_id"]),
        "frame_index": int(row["frame_index"]),
        "proxy_variant": str(row["proxy_variant"]),
        "policy_probe": str(row["policy_probe"]),
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "base_chunk_cache_path": str(base_chunk_path),
        "base_chunk_cache_sha256": _sha256(base_chunk_path),
        "base_chunk_sha256": _array_sha256(base_chunk),
        "base_action_shape": list(np.asarray(base_chunk).shape),
        "base_action_min": float(np.min(base_chunk)),
        "base_action_max": float(np.max(base_chunk)),
        "base_action_finite": bool(np.isfinite(base_chunk).all()),
        "visual_feature_dim": int(feature.shape[0]),
        "visual_feature_finite_fraction": float(np.mean(np.isfinite(feature))),
        "proprio_dim": int(proprio.shape[0]),
        "proprio_finite_fraction": float(np.mean(np.isfinite(proprio))),
        "action_chunk_sha256": _array_sha256(chunk),
        "action_min": float(np.min(chunk)),
        "action_max": float(np.max(chunk)),
        "action_finite": bool(np.isfinite(chunk).all()),
    }


def _partial_payload(
    manifest_hash: str | None,
    planned: int | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "TSC-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": planned,
        "completed_model_row_count": len(rows),
        "completed_row_keys": [str(row["row_key"]) for row in rows],
        "rows": list(rows),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "updated_at": _utc_now(),
    }


def _load_resume(
    path: Path, manifest_rows: Sequence[Mapping[str, Any]], manifest_hash: str
) -> tuple[list[dict[str, Any]], int, str | None]:
    if not path.is_file():
        return [], 0, None
    partial = _read_json(path)
    if partial.get("method") != "TSC-VLA" or partial.get("proposal_hash") != PROPOSAL_HASH:
        raise RuntimeError("partial result identity does not match frozen TSC proposal/manifest")
    rows = list(partial.get("rows") or [])
    if partial.get("manifest_hash") is None and not rows:
        return [], 0, None
    if partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial result identity does not match frozen TSC manifest")
    audit = validate_manifest(manifest_rows, rows)
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"partial contains duplicate or off-manifest keys: {audit}")
    for row in rows:
        cache = Path(str(row["feature_cache_path"]))
        if not cache.is_file() or _sha256(cache) != row["feature_cache_sha256"]:
            raise RuntimeError(f"feature cache hash mismatch for {row['row_key']}")
        base_cache = Path(str(row["base_chunk_cache_path"]))
        if not base_cache.is_file() or _sha256(base_cache) != row["base_chunk_cache_sha256"]:
            raise RuntimeError(f"base chunk cache hash mismatch for {row['row_key']}")
    return rows, int(partial.get("exception_count") or 0), partial.get("last_exception")


def _materialize_arrays(manifest: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    import h5py

    partial_by_key = {str(row["row_key"]): row for row in partial_rows}
    visuals = []
    proprios = []
    raw_features = []
    chunks = []
    base_chunks = []
    task_indices = []
    partitions = []
    phases = []
    task_names = []
    keys = []
    action_finite = []
    for row in manifest:
        partial = partial_by_key[str(row["row_key"])]
        visual = _load_feature(Path(str(partial["feature_cache_path"])))
        base_chunk = _load_base_chunk(Path(str(partial["base_chunk_cache_path"])))
        with h5py.File(str(row["source_path"]), "r") as handle:
            demo = handle["data"][f"demo_{int(row['demo_id'])}"]
            observations = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            proprio = _proprio_from_obs(observations, int(row["frame_index"]))
            chunk = action_chunk(actions, int(row["frame_index"]))
        if _array_sha256(chunk) != partial["action_chunk_sha256"]:
            raise RuntimeError(f"action chunk hash mismatch for {row['row_key']}")
        if _array_sha256(base_chunk) != partial["base_chunk_sha256"]:
            raise RuntimeError(f"base chunk hash mismatch for {row['row_key']}")
        raw_feature = raw_tsc_feature(visual, proprio, int(row["task_index"]), float(row["phase"]))
        visuals.append(visual)
        proprios.append(proprio)
        raw_features.append(raw_feature)
        chunks.append(chunk)
        base_chunks.append(base_chunk)
        task_indices.append(int(row["task_index"]))
        partitions.append(str(row["partition"]))
        phases.append(float(row["phase"]))
        task_names.append(str(row["task_identity"]))
        keys.append(str(row["row_key"]))
        action_finite.append(bool(np.isfinite(chunk).all()))
    return {
        "visual": np.asarray(visuals, dtype=np.float64),
        "proprio": np.asarray(proprios, dtype=np.float64),
        "raw_feature": np.asarray(raw_features, dtype=np.float64),
        "chunk": np.asarray(chunks, dtype=np.float64),
        "base_chunk": np.asarray(base_chunks, dtype=np.float64),
        "residual_chunk": np.asarray(chunks, dtype=np.float64) - np.asarray(base_chunks, dtype=np.float64),
        "task_index": np.asarray(task_indices, dtype=np.int64),
        "partition": np.asarray(partitions, dtype=object),
        "phase": np.asarray(phases, dtype=np.float64),
        "task_identity": np.asarray(task_names, dtype=object),
        "row_key": np.asarray(keys, dtype=object),
        "action_finite": np.asarray(action_finite, dtype=bool),
    }


def _fit_tsc_models(
    manifest: Sequence[Mapping[str, Any]],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    discovery = arrays["partition"] == "discovery"
    validation = arrays["partition"] == "validation"
    discovery_indices = np.flatnonzero(discovery)
    validation_indices = np.flatnonzero(validation)

    zscore = fit_discovery_zscore(arrays["raw_feature"][discovery])
    zfeatures = apply_discovery_zscore(zscore, arrays["raw_feature"])
    arrays["zfeature"] = zfeatures

    label_stats = fit_mask_label_stats(arrays["residual_chunk"][discovery])
    labels = make_error_mask_labels(arrays["residual_chunk"], label_stats)
    labels_discovery = labels[discovery]
    labels_validation = labels[validation]

    mask_model = fit_structured_mask_probe(zfeatures[discovery], labels_discovery)
    structured_probability = predict_structured_mask_scores(mask_model, zfeatures[validation])
    trivial_probability = np.broadcast_to(trivial_mask_probability(labels_discovery), labels_validation.shape)
    magnitude_model = fit_magnitude_mask_baseline(arrays["base_chunk"][discovery], labels_discovery)
    magnitude_probability = predict_magnitude_mask_probability(magnitude_model, arrays["base_chunk"][validation])
    structured_bce = binary_cross_entropy(structured_probability, labels_validation)
    trivial_bce = binary_cross_entropy(trivial_probability, labels_validation)
    magnitude_bce = binary_cross_entropy(magnitude_probability, labels_validation)

    completion_model = fit_completion_model(
        zfeatures[discovery],
        arrays["base_chunk"][discovery],
        arrays["chunk"][discovery],
    )
    residual_prediction = predict_completion_residual(
        completion_model,
        zfeatures[validation],
        arrays["base_chunk"][validation],
    )
    tsc_mask = hard_mask(structured_probability, MASK_THRESHOLD)
    prior_mask = deterministic_random_mask(
        [str(key) for key in arrays["row_key"][validation]],
        float(label_stats["positive_rate"]),
        salt="ts_mask_continuous_proxy",
    )
    ablation_mask = hard_mask(magnitude_probability, MASK_THRESHOLD)

    tsc_prediction = apply_masked_completion(arrays["base_chunk"][validation], residual_prediction, tsc_mask)
    prior_prediction = apply_masked_completion(arrays["base_chunk"][validation], residual_prediction, prior_mask)
    ablation_prediction = apply_masked_completion(arrays["base_chunk"][validation], residual_prediction, ablation_mask)

    arrays["label"] = labels
    arrays["tsc_full_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["ts_mask_proxy_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["no_targeted_ablation_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["completion_residual"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["tsc_full_mask"] = np.zeros_like(arrays["chunk"], dtype=bool)
    arrays["ts_mask_proxy_mask"] = np.zeros_like(arrays["chunk"], dtype=bool)
    arrays["no_targeted_ablation_mask"] = np.zeros_like(arrays["chunk"], dtype=bool)
    arrays["tsc_full_chunk"][validation_indices] = tsc_prediction
    arrays["ts_mask_proxy_chunk"][validation_indices] = prior_prediction
    arrays["no_targeted_ablation_chunk"][validation_indices] = ablation_prediction
    arrays["completion_residual"][validation_indices] = residual_prediction
    arrays["tsc_full_mask"][validation_indices] = tsc_mask
    arrays["ts_mask_proxy_mask"][validation_indices] = prior_mask
    arrays["no_targeted_ablation_mask"][validation_indices] = ablation_mask

    prior_headroom = prediction_metrics(
        flattened_chunks(tsc_prediction),
        flattened_chunks(prior_prediction),
        flattened_chunks(arrays["chunk"][validation]),
    )
    ablation_headroom = prediction_metrics(
        flattened_chunks(tsc_prediction),
        flattened_chunks(ablation_prediction),
        flattened_chunks(arrays["chunk"][validation]),
    )
    base_metrics = prediction_metrics(
        flattened_chunks(tsc_prediction),
        flattened_chunks(arrays["base_chunk"][validation]),
        flattened_chunks(arrays["chunk"][validation]),
    )
    full_prior_delta = flattened_chunks(tsc_prediction) - flattened_chunks(prior_prediction)
    full_ablation_delta = flattened_chunks(tsc_prediction) - flattened_chunks(ablation_prediction)
    delta_summary = action_delta_summary(arrays["base_chunk"][validation], tsc_prediction, tsc_mask)

    models = {
        "zscore": zscore,
        "label_stats": label_stats,
        "mask_probe": mask_model,
        "magnitude_mask": magnitude_model,
        "completion": completion_model,
        "diagnostic_alpha": DIAGNOSTIC_ALPHA,
        "discovery_row_keys": arrays["row_key"][discovery].tolist(),
        "validation_row_keys": arrays["row_key"][validation].tolist(),
    }
    audit = {
        "model_hash": canonical_json_sha256(models),
        "discovery_row_count": int(len(discovery_indices)),
        "validation_row_count": int(len(validation_indices)),
        "mask_quantile": MASK_QUANTILE,
        "mask_threshold": MASK_THRESHOLD,
        "diagnostic_alpha": DIAGNOSTIC_ALPHA,
        "label_stats": label_stats,
        "discovery_positive_mask_cell_count": int(labels_discovery.sum()),
        "discovery_negative_mask_cell_count": int(labels_discovery.size - labels_discovery.sum()),
        "validation_positive_mask_cell_count": int(labels_validation.sum()),
        "validation_negative_mask_cell_count": int(labels_validation.size - labels_validation.sum()),
        "structured_mask_bce": structured_bce,
        "trivial_majority_mask_bce": trivial_bce,
        "magnitude_only_mask_bce": magnitude_bce,
        "structured_mask_beats_trivial": bool(structured_bce < trivial_bce),
        "structured_mask_beats_magnitude": bool(structured_bce < magnitude_bce),
        "tsc_full_huber": mean_huber(tsc_prediction, arrays["chunk"][validation]),
        "ts_mask_proxy_huber": mean_huber(prior_prediction, arrays["chunk"][validation]),
        "no_targeted_ablation_huber": mean_huber(ablation_prediction, arrays["chunk"][validation]),
        "base_to_expert_huber": mean_huber(arrays["base_chunk"][validation], arrays["chunk"][validation]),
        "tsc_vs_base": base_metrics,
        "tsc_vs_ts_mask_proxy": prior_headroom,
        "tsc_vs_no_targeted_ablation": ablation_headroom,
        "tsc_prior_delta_norm_mean": float(np.mean(np.linalg.norm(full_prior_delta, axis=1))),
        "tsc_ablation_delta_norm_mean": float(np.mean(np.linalg.norm(full_ablation_delta, axis=1))),
        "tsc_distinct_from_prior_and_ablation": bool(
            float(np.mean(np.linalg.norm(full_prior_delta, axis=1))) > 1e-12
            and float(np.mean(np.linalg.norm(full_ablation_delta, axis=1))) > 1e-12
        ),
        "unselected_cell_clamp_max_error": unselected_clamp_error(arrays["base_chunk"][validation], tsc_prediction, tsc_mask),
        "action_delta_summary": delta_summary,
    }
    return models, audit, arrays


def _data_summary(
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    model_audit: Mapping[str, Any],
    manifest_audit: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {
        partition: sum(row["partition"] == partition for row in manifest)
        for partition in ("discovery", "validation")
    }
    validation_counts: dict[str, int] = {}
    for row in manifest:
        if row["partition"] == "validation":
            validation_counts[str(row["task_identity"])] = validation_counts.get(str(row["task_identity"]), 0) + 1
    validation_total = sum(validation_counts.values())
    validation_fractions = {task: count / max(validation_total, 1) for task, count in validation_counts.items()}
    finite_alignment = bool(
        np.isfinite(arrays["visual"]).all()
        and np.isfinite(arrays["proprio"]).all()
        and np.isfinite(arrays["raw_feature"]).all()
        and np.isfinite(arrays["chunk"]).all()
        and np.isfinite(arrays["base_chunk"]).all()
        and np.isfinite(arrays["residual_chunk"]).all()
        and np.all(arrays["action_finite"])
    )
    discovery = arrays["partition"] == "discovery"
    validation = arrays["partition"] == "validation"
    labels = np.asarray(arrays["label"], dtype=bool)
    discovery_rate = float(labels[discovery].mean())
    validation_rate = float(labels[validation].mean())
    prior = model_audit["tsc_vs_ts_mask_proxy"]
    ablation = model_audit["tsc_vs_no_targeted_ablation"]
    delta_summary = model_audit["action_delta_summary"]
    return {
        "counts": counts,
        "minimum_discovery_windows": counts["discovery"],
        "minimum_validation_windows": counts["validation"],
        "feature_action_proprio_finite_aligned": finite_alignment,
        "validation_task_counts": validation_counts,
        "validation_task_fractions": validation_fractions,
        "maximum_validation_task_fraction": max(validation_fractions.values(), default=1.0),
        "all_tasks_reported": len(validation_counts) == len(TASK_SOURCES),
        "discovery_mask_positive_rate": discovery_rate,
        "validation_mask_positive_rate": validation_rate,
        "labels_noncollapsed_discovery": bool(0.0 < discovery_rate < 1.0),
        "labels_noncollapsed_validation": bool(0.0 < validation_rate < 1.0),
        "structured_mask_bce": model_audit["structured_mask_bce"],
        "trivial_majority_mask_bce": model_audit["trivial_majority_mask_bce"],
        "magnitude_only_mask_bce": model_audit["magnitude_only_mask_bce"],
        "structured_mask_beats_trivial": bool(model_audit["structured_mask_beats_trivial"]),
        "structured_mask_beats_magnitude": bool(model_audit["structured_mask_beats_magnitude"]),
        "tsc_full_huber": model_audit["tsc_full_huber"],
        "ts_mask_proxy_huber": model_audit["ts_mask_proxy_huber"],
        "no_targeted_ablation_huber": model_audit["no_targeted_ablation_huber"],
        "base_to_expert_huber": model_audit["base_to_expert_huber"],
        "tsc_beats_prior_relative": prior["relative_mse_improvement"],
        "tsc_beats_prior_absolute_huber": prior["absolute_huber_improvement"],
        "tsc_beats_ablation_relative": ablation["relative_mse_improvement"],
        "tsc_beats_ablation_absolute_huber": ablation["absolute_huber_improvement"],
        "headroom_relative_gate": HEADROOM_RELATIVE_GATE,
        "headroom_absolute_huber_gate": HEADROOM_ABSOLUTE_HUBER_GATE,
        "unselected_cell_clamp_max_error": model_audit["unselected_cell_clamp_max_error"],
        "changed_cell_fraction": delta_summary["changed_cell_fraction"],
        "action_delta_summary": delta_summary,
        "deltas_finite_and_bounded": bool(delta_summary["delta_finite"] and np.isfinite(delta_summary["delta_abs_max"])),
        "tsc_distinct_from_prior_and_ablation": bool(model_audit["tsc_distinct_from_prior_and_ablation"]),
        "demo_action_validity_ok": bool(
            np.all(arrays["action_finite"])
            and np.isfinite(arrays["base_chunk"]).all()
            and arrays["base_chunk"].shape[1:] == (CHUNK_SIZE, ACTION_DIM)
        ),
        "manifest_audit": manifest_audit,
        "model_audit": model_audit,
    }


def _action_stats(postprocessor: Any) -> dict[str, Any]:
    for step in postprocessor.steps:
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not tensor_stats or "action" not in tensor_stats:
            continue
        stats = tensor_stats["action"]
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError("TSC requires checkpoint MEAN_STD action statistics")
        mean = stats["mean"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        std = stats["std"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        return {"mode": "MEAN_STD", "mean": mean, "std": std, "processor_step": type(step).__name__}
    raise RuntimeError("checkpoint postprocessor has no action unnormalizer statistics")


def _write_action_semantics(path: Path, action_stats: Mapping[str, Any]) -> dict[str, Any]:
    semantics = {
        "method": "TSC-VLA",
        "stage": "0",
        "model_native_action_shape": [CHUNK_SIZE, ACTION_DIM],
        "environment_action_shape": [ACTION_DIM],
        "postprocessor_mode": action_stats.get("mode"),
        "postprocessor_step": action_stats.get("processor_step"),
        "postprocessor_action_mean_shape": list(np.asarray(action_stats.get("mean"), dtype=np.float64).shape),
        "postprocessor_action_std_shape": list(np.asarray(action_stats.get("std"), dtype=np.float64).shape),
        "environment_action_space_low_high_exposed": False,
        "environment_action_space_low": None,
        "environment_action_space_high": None,
        "gripper_convention": "LIBERO/SmolVLA checkpoint action dimension 6 after postprocessor",
        "finite_checks_required": True,
        "action_bound_validity_rule": "not_used_without_official_environment_bounds",
        "final_action_validity_definition": (
            "valid iff postprocessed action chunk has shape [50,7], all entries are finite, "
            "and the same SmolVLA postprocessor statistics are used for every policy"
        ),
        "same_definition_applies_to_policies": [
            "smolvla_base",
            "ts_mask_continuous_proxy",
            "tsc_full",
            "tsc_no_targeted_mask_ablation",
            "standard_lora",
        ],
        "written_at": _utc_now(),
    }
    _write_json(path, semantics)
    return semantics


def _stable_seed(identity: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{purpose}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _noise(identity: str, purpose: str, shape: Sequence[int], device: str) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(identity, purpose))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


def _normalize_raw_action(raw: Any, action_stats: Mapping[str, Any]) -> Any:
    import torch

    mean = torch.as_tensor(action_stats["mean"], dtype=raw.dtype, device=raw.device)
    std = torch.as_tensor(action_stats["std"], dtype=raw.dtype, device=raw.device)
    return (raw - mean) / std


def _tsc_loss(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    time_value: Any,
    action_stats: Mapping[str, Any],
    target_raw_chunk: np.ndarray,
) -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    flow_loss = _loss(policy, batch, noise, time_value)
    _, x_t, velocity = _native_velocity(policy, batch, noise, time_value)
    clean = x_t - time_value[:, None, None] * velocity
    clean_action = clean[:, :CHUNK_SIZE, :ACTION_DIM]
    target_raw = torch.as_tensor(target_raw_chunk, dtype=clean_action.dtype, device=clean_action.device).reshape(1, CHUNK_SIZE, ACTION_DIM)
    target_native = _normalize_raw_action(target_raw, action_stats)
    tsc_target_loss = functional.smooth_l1_loss(clean_action, target_native, beta=1.0)
    clean_loss = functional.smooth_l1_loss(clean_action, batch["action"][:, :CHUNK_SIZE, :ACTION_DIM], beta=1.0)
    total = flow_loss + tsc_target_loss + 0.1 * clean_loss
    return total, flow_loss, tsc_target_loss, clean_loss


def _gradient_audit(
    policy: Any,
    preprocessor: Any,
    action_stats: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    import torch

    validation_indices = np.flatnonzero(arrays["partition"] == "validation")
    row_index = int(validation_indices[0])
    row = manifest[row_index]
    raw = _raw_sample(row)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    noise = _noise(str(row["row_key"]), "gradient", shape, "cuda")
    time_value = torch.full((1,), FLOW_TIME, dtype=torch.float32, device="cuda")
    named = sorted(
        [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    if not named or not all("lora_" in name.lower() for name, _ in named):
        raise RuntimeError("TSC expected only LoRA trainable parameters")

    policy.train()
    policy.zero_grad(set_to_none=True)
    flow_loss = _loss(policy, batch, noise, time_value)
    flow_gradients, flow_norm, flow_finite = _gradient_values(flow_loss, named)
    flow_value = float(flow_loss.detach().item())
    del flow_loss
    gc.collect()
    torch.cuda.empty_cache()

    target = np.asarray(arrays["tsc_full_chunk"][row_index], dtype=np.float64)
    policy.zero_grad(set_to_none=True)
    tsc_loss, flow_component, target_component, clean_component = _tsc_loss(
        policy, batch, noise, time_value, action_stats, target
    )
    tsc_gradients, tsc_norm, tsc_finite = _gradient_values(tsc_loss, named)
    tsc_value = float(tsc_loss.detach().item())
    target_value = float(target_component.detach().item())
    clean_value = float(clean_component.detach().item())
    dot = 0.0
    for flow_gradient, tsc_gradient in zip(flow_gradients, tsc_gradients, strict=True):
        if flow_gradient is not None and tsc_gradient is not None:
            dot += float(torch.sum(flow_gradient * tsc_gradient).item())
    cosine = dot / max(flow_norm * tsc_norm, 1e-12)
    frozen_gradient_names = [
        name for name, parameter in policy.named_parameters() if "lora_" not in name.lower() and parameter.grad is not None
    ]
    policy.zero_grad(set_to_none=True)
    policy.eval()
    return {
        "flow_time": FLOW_TIME,
        "flow_loss": flow_value,
        "tsc_loss": tsc_value,
        "L_flow": float(flow_component.detach().item()),
        "L_tsc_target": target_value,
        "L_clean": clean_value,
        "trainable_parameter_names": [name for name, _ in named],
        "trainable_parameter_count": len(named),
        "trainable_numel": sum(int(parameter.numel()) for _, parameter in named),
        "flow_gradient_norm": flow_norm,
        "tsc_gradient_norm": tsc_norm,
        "tsc_to_flow_gradient_ratio": tsc_norm / max(flow_norm, 1e-12),
        "gradient_cosine": cosine,
        "flow_gradient_finite_fraction": flow_finite,
        "tsc_gradient_finite_fraction": tsc_finite,
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "frozen_parameter_gradient_names": frozen_gradient_names,
        "tsc_gradient_nonzero": tsc_norm > 0.0,
        "finite_objectives_and_gradients": bool(
            np.isfinite([flow_value, tsc_value, target_value, clean_value, flow_norm, tsc_norm, cosine]).all()
            and flow_finite == 1.0
            and tsc_finite == 1.0
        ),
        "gradient_row_key": str(row["row_key"]),
    }


def _identity_and_gradient_audit(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    action_stats: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from peft import PeftConfig, PeftModel

    validation_indices = np.flatnonzero(arrays["partition"] == "validation")
    row_index = int(validation_indices[0])
    row = manifest[row_index]
    raw = _raw_sample(row)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    flow_noise = _noise(str(row["row_key"]), "identity_flow", shape, "cuda")
    solver_noise = _noise(str(row["row_key"]), "identity_solver", shape, "cuda")
    time_value = torch.full((1,), FLOW_TIME, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        _, _, base_flow = _native_velocity(policy, batch, flow_noise, time_value)
    base_native, base_actions = _decoded_chunk(policy, batch, postprocessor, solver_noise)
    base_hash_before = _hash_base_parameters(policy)

    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.eval()
    with torch.no_grad():
        _, _, initialized_flow = _native_velocity(policy, batch, flow_noise, time_value)
    initialized_native, initialized_actions = _decoded_chunk(policy, batch, postprocessor, solver_noise)
    base_hash_after = _hash_base_parameters(policy)
    initialized_errors = {
        "flow": float(torch.max(torch.abs(initialized_flow - base_flow)).item()),
        "native": float(torch.max(torch.abs(initialized_native - base_native)).item()),
        "actions": float(np.max(np.abs(initialized_actions - base_actions))),
    }

    gradient = _gradient_audit(policy, preprocessor, action_stats, manifest, arrays)
    paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "peft_config"):
        for config in policy.peft_config.values():
            config.base_model_name_or_path = str(paths["checkpoint"])
    policy.save_pretrained(paths["adapter_dir"], safe_serialization=True)
    adapter_files = sorted(path.name for path in paths["adapter_dir"].iterdir() if path.is_file())

    del policy
    gc.collect()
    torch.cuda.empty_cache()
    base, _, reloaded_preprocessor, reloaded_postprocessor = _load_policy_and_processors(paths["checkpoint"])
    peft_config = PeftConfig.from_pretrained(paths["adapter_dir"])
    peft_config.base_model_name_or_path = str(paths["checkpoint"])
    reloaded = PeftModel.from_pretrained(
        base,
        paths["adapter_dir"],
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    reloaded.to("cuda")
    reloaded.eval()
    reloaded_batch = _preprocess(reloaded_preprocessor, raw)
    with torch.no_grad():
        _, _, reloaded_flow = _native_velocity(reloaded, reloaded_batch, flow_noise, time_value)
    reloaded_native, reloaded_actions = _decoded_chunk(reloaded, reloaded_batch, reloaded_postprocessor, solver_noise)
    reload_errors = {
        "flow": float(torch.max(torch.abs(reloaded_flow - base_flow)).item()),
        "native": float(torch.max(torch.abs(reloaded_native - base_native)).item()),
        "actions": float(np.max(np.abs(reloaded_actions - base_actions))),
    }
    identity_max = max(*initialized_errors.values(), *reload_errors.values())
    ours_target = np.asarray(arrays["tsc_full_chunk"][row_index], dtype=np.float64)
    prior_target = np.asarray(arrays["ts_mask_proxy_chunk"][row_index], dtype=np.float64)
    ablation_target = np.asarray(arrays["no_targeted_ablation_chunk"][row_index], dtype=np.float64)
    mask = np.asarray(arrays["tsc_full_mask"][row_index], dtype=bool)
    projection_delta = ours_target - np.asarray(arrays["base_chunk"][row_index], dtype=np.float64)
    dimension_delta = np.max(np.abs(projection_delta), axis=0)
    identity = {
        "rank": 4,
        "base_flow_shape": list(base_flow.shape),
        "base_native_shape": list(base_native.shape),
        "base_action_shape": list(base_actions.shape),
        "base_action": base_actions[: min(3, len(base_actions))].tolist(),
        "ours_action_diagnostic": ours_target[: min(3, len(ours_target))].tolist(),
        "ts_mask_proxy_action_diagnostic": prior_target[: min(3, len(prior_target))].tolist(),
        "no_targeted_ablation_action_diagnostic": ablation_target[: min(3, len(ablation_target))].tolist(),
        "initialized_max_abs_errors": initialized_errors,
        "reload_max_abs_errors": reload_errors,
        "identity_max_abs_error": identity_max,
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_hash_unchanged": base_hash_before == base_hash_after,
        "checkpoint_reload_ok": identity_max <= 1e-6 and bool(adapter_files),
        "adapter_checkpoint": str(paths["adapter_dir"]),
        "adapter_files": adapter_files,
        "base_action_finite": bool(np.isfinite(base_actions).all()),
        "base_action_valid_under_official_semantics": bool(
            np.asarray(base_actions).shape == (CHUNK_SIZE, ACTION_DIM) and np.isfinite(base_actions).all()
        ),
        "tsc_residual_norm": float(np.linalg.norm(projection_delta)),
        "tsc_gate_value": 1.0,
        "mask_positive_count": int(mask.sum()),
        "mask_positive_fraction": float(mask.mean()),
        "translation_delta_norm": float(np.linalg.norm(projection_delta[:, :3])),
        "rotation_delta_norm": float(np.linalg.norm(projection_delta[:, 3:6])),
        "gripper_delta_norm": float(np.linalg.norm(projection_delta[:, 6])),
        "changed_dimensions": [int(index) for index, value in enumerate(dimension_delta) if float(value) > 1e-12],
        "activation_context": str(row["row_key"]),
        "training_only_tsc_absent_from_policy_parameters": not any(
            "tsc" in name.lower() for name, _ in reloaded.named_parameters()
        ),
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return identity, gradient


def _decision_inputs(
    proposal_ok: bool,
    serializer_ok: bool,
    prior_check: Mapping[str, Any],
    manifest_ok: bool,
    data: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    gradient: Mapping[str, Any] | None,
    exception_count: int,
) -> Stage0DecisionInputs:
    identity = identity or {}
    gradient = gradient or {}
    return Stage0DecisionInputs(
        proposal_hash_ok=proposal_ok,
        serializer_preflight_ok=serializer_ok,
        official_prior_asset_check_persisted=bool(prior_check),
        manifest_integrity_ok=manifest_ok,
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=bool(data["feature_action_proprio_finite_aligned"]),
        minimum_discovery_windows=int(data["minimum_discovery_windows"]),
        minimum_validation_windows=int(data["minimum_validation_windows"]),
        all_tasks_reported=bool(data["all_tasks_reported"]),
        maximum_validation_task_fraction=float(data["maximum_validation_task_fraction"]),
        labels_noncollapsed_discovery=bool(data["labels_noncollapsed_discovery"]),
        labels_noncollapsed_validation=bool(data["labels_noncollapsed_validation"]),
        structured_mask_beats_trivial=bool(data["structured_mask_beats_trivial"]),
        structured_mask_beats_magnitude=bool(data["structured_mask_beats_magnitude"]),
        tsc_beats_prior_relative=float(data["tsc_beats_prior_relative"]),
        tsc_beats_prior_absolute_huber=float(data["tsc_beats_prior_absolute_huber"]),
        tsc_beats_ablation_relative=float(data["tsc_beats_ablation_relative"]),
        tsc_beats_ablation_absolute_huber=float(data["tsc_beats_ablation_absolute_huber"]),
        unselected_cell_clamp_max_error=float(data["unselected_cell_clamp_max_error"]),
        changed_cell_fraction=float(data["changed_cell_fraction"]),
        deltas_finite_and_bounded=bool(data["deltas_finite_and_bounded"]),
        tsc_distinct_from_prior_and_ablation=bool(data["tsc_distinct_from_prior_and_ablation"]),
        finite_objectives_and_gradients=bool(gradient.get("finite_objectives_and_gradients", False)),
        tsc_gradient_nonzero=bool(gradient.get("tsc_gradient_nonzero", False)),
        gradient_ratio_at_most_100=float(gradient.get("tsc_to_flow_gradient_ratio", float("inf"))) <= GRADIENT_RATIO_MAX,
        frozen_parameter_gradient_count=int(gradient.get("frozen_parameter_gradient_count", 0)),
        identity_max_error=float(identity.get("identity_max_abs_error", 0.0)),
        base_hash_unchanged=bool(identity.get("base_hash_unchanged", True)),
        checkpoint_reload_ok=bool(identity.get("checkpoint_reload_ok", True)),
        action_validity_ok=bool(identity.get("base_action_valid_under_official_semantics", True))
        and bool(data.get("demo_action_validity_ok", True)),
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        exception_count=int(exception_count),
    )


def _result_markdown(result: Mapping[str, Any]) -> str:
    data = result["data_audit"]
    gradient = result.get("gradient") or {}
    identity = result.get("identity") or {}
    return "\n".join(
        [
            "# TSC-VLA Stage 0 Result",
            "",
            f"Final decision: `{result['final_decision']}`.",
            "",
            f"Rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`.",
            f"TS-Mask prior label: `{result['official_prior_asset_check']['policy_2_label']}`.",
            f"Structured / trivial / magnitude mask BCE: `{data['structured_mask_bce']} / {data['trivial_majority_mask_bce']} / {data['magnitude_only_mask_bce']}`.",
            f"TSC / TS-Mask-proxy / ablation Huber: `{data['tsc_full_huber']} / {data['ts_mask_proxy_huber']} / {data['no_targeted_ablation_huber']}`.",
            f"TSC minus prior relative / absolute Huber gain: `{data['tsc_beats_prior_relative']} / {data['tsc_beats_prior_absolute_huber']}`.",
            f"TSC minus ablation relative / absolute Huber gain: `{data['tsc_beats_ablation_relative']} / {data['tsc_beats_ablation_absolute_huber']}`.",
            f"Changed-cell fraction: `{data['changed_cell_fraction']}`.",
            f"Unselected clamp max error: `{data['unselected_cell_clamp_max_error']}`.",
            f"Flow / TSC gradient norm: `{gradient.get('flow_gradient_norm')} / {gradient.get('tsc_gradient_norm')}`.",
            f"Identity maximum error: `{identity.get('identity_max_abs_error')}`.",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            "No simulator rollout, reward/success/done read, confirmatory identity access, validation search, or closed-loop experiment occurred.",
            "",
        ]
    )


def _adjudication_markdown(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TSC-VLA Stage 0 Adjudication",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            "This is a development-only audit, not a closed-loop scientific kill.",
            f"Bounded validation allowed: `{result['bounded_validation_allowed']}`.",
            f"Valid scientific result: `{result['valid_scientific_result']}`.",
            "",
            "The frozen Stage 0 gates were applied without changing task sources, TS-Mask proxy definition, TSC mask construction, action semantics, baselines, or thresholds.",
            "",
        ]
    )


def _preflight(paths: Mapping[str, Path], started_unix: float) -> dict[str, Any]:
    import torch

    required = {
        "checkpoint": paths["checkpoint"],
        "vlm": VLM_PATH,
        "data_root": paths["data_root"],
        "proposal": PROPOSAL_FILE,
        "proposal_hash": PROPOSAL_HASH_FILE,
        "reviewer_attack": REPO_ROOT / "reports" / "tsc_vla" / "reviewer_attack.md",
        "researcher_rebuttal": REPO_ROOT / "reports" / "tsc_vla" / "researcher_rebuttal.md",
        "mathematical_audit": REPO_ROOT / "reports" / "tsc_vla" / "mathematical_mechanism_audit.md",
        "preregistration": REPO_ROOT / "reports" / "tsc_vla" / "preregistration.md",
        "prototype_protocol": REPO_ROOT / "reports" / "tsc_vla" / "prototype_protocol.md",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    partial_parse_error = None
    partial_summary = None
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            partial_summary = {
                "planned_model_row_count": partial.get("planned_model_row_count"),
                "completed_model_row_count": partial.get("completed_model_row_count"),
                "exception_count": partial.get("exception_count"),
            }
        except Exception as exc:
            partial_parse_error = f"{type(exc).__name__}: {exc}"
    registry = _read_json(RESOURCE_REGISTRY) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    parent_pid = os.getppid()
    workers = []
    for worker in _active_linux_workers():
        command = str(worker.get("command", ""))
        if "run_tsc_vla_stage0.py" not in command:
            continue
        if int(worker.get("pid", -1)) == parent_pid:
            continue
        if command.lstrip().startswith(("bash -lc ", "/bin/bash -lc ")):
            continue
        workers.append(worker)
    return {
        "passed": bool(
            not missing
            and torch.cuda.is_available()
            and _proposal_hash_text() == PROPOSAL_HASH
            and _sha256(PROPOSAL_FILE) == PROPOSAL_HASH
            and partial_parse_error is None
            and not paths["result_json"].exists()
            and not workers
        ),
        "missing_paths": missing,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "proposal_hash_file": _proposal_hash_text(),
        "proposal_hash_observed": _sha256(PROPOSAL_FILE) if PROPOSAL_FILE.is_file() else None,
        "partial_summary": partial_summary,
        "partial_parse_error": partial_parse_error,
        "result_absent": not paths["result_json"].exists(),
        "active_tsc_linux_workers": workers,
        "resource_evidence": _resource_evidence(registry, started_unix),
    }


def run(args: argparse.Namespace, paths: Mapping[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    import torch

    _set_offline_environment()
    started_unix = time.time()
    preflight = _preflight(paths, started_unix)
    _write_json(paths["preflight"], preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"TSC Stage 0 preflight failed: {preflight}")
    if not paths["serializer_preflight"].is_file():
        _serializer_preflight(paths["serializer_preflight"])
    serializer = _read_json(paths["serializer_preflight"])
    serializer_ok = bool(
        serializer.get("passed") and canonical_json_sha256(serializer["fixture"]) == serializer.get("fixture_hash")
    )
    if not serializer_ok:
        raise RuntimeError("foreground TSC serializer preflight is absent or invalid")
    prior_check = _official_prior_asset_check(paths["official_prior_asset_check"])
    proposal_observed = _sha256(PROPOSAL_FILE)
    proposal_registry = _proposal_hash_text()
    proposal_ok = proposal_observed == proposal_registry == PROPOSAL_HASH
    if not proposal_ok:
        raise RuntimeError("frozen TSC proposal hash mismatch")

    state.update({"phase": "manifest", "status": "running"})
    rows, sources = _build_manifest(paths["data_root"])
    manifest_payload = {
        "method": "TSC-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "sources": sources,
        "chunk_size": CHUNK_SIZE,
        "phase_bins": PHASE_BINS,
        "mask_quantile": MASK_QUANTILE,
        "mask_threshold": MASK_THRESHOLD,
        "diagnostic_alpha": DIAGNOSTIC_ALPHA,
        "ridge_coefficient": RIDGE_COEFFICIENT,
        "std_floor": STD_FLOOR,
        "discovery_rows_per_task": DISCOVERY_ROWS_PER_TASK,
        "validation_rows_per_task": VALIDATION_ROWS_PER_TASK,
        "policy_probe": POLICY_PROBE,
        "planned_model_row_count": len(rows),
        "rows": rows,
    }
    manifest_hash = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_hash"] = manifest_hash
    _write_json(paths["manifest"], manifest_payload)
    parsed = _read_json(paths["manifest"])
    parsed_without_hash = dict(parsed)
    parsed_without_hash.pop("manifest_hash")
    if canonical_json_sha256(parsed_without_hash) != manifest_hash:
        raise RuntimeError("persisted TSC manifest hash did not reproduce")
    manifest_audit = validate_manifest(rows, [{"row_key": row["row_key"]} for row in rows])

    partial_rows, prior_exception_count, prior_last_exception = _load_resume(paths["partial"], rows, manifest_hash)
    resumed_count = len(partial_rows)
    completed = {str(row["row_key"]) for row in partial_rows}
    state.update(
        {
            "phase": "feature_extraction",
            "planned_model_row_count": len(rows),
            "completed_model_row_count": len(partial_rows),
            "exception_count": prior_exception_count,
        }
    )
    _write_json(
        paths["partial"],
        _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
    )

    policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
    policy.eval()
    for row in rows:
        key = str(row["row_key"])
        if key in completed:
            continue
        feature_path, feature = _load_or_extract_feature(policy, paths, row)
        base_path, base_chunk = _load_or_decode_base_chunk(policy, preprocessor, postprocessor, paths, row)
        partial_rows.append(_partial_row(row, feature_path, feature, base_path, base_chunk))
        completed.add(key)
        state["completed_model_row_count"] = len(partial_rows)
        _write_json(
            paths["partial"],
            _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
        )
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        if len(partial_rows) % 16 == 0 or len(partial_rows) == len(rows):
            print(f"[tsc-stage0] rows {len(partial_rows)}/{len(rows)}", flush=True)

    ordered = {str(row["row_key"]): row for row in partial_rows}
    partial_rows = [ordered[str(row["row_key"])] for row in rows]
    partial_audit = validate_manifest(rows, partial_rows)
    manifest_ok = bool(
        manifest_audit["duplicate_manifest_key_count"] == 0
        and manifest_audit["split_overlap_key_count"] == 0
        and partial_audit["duplicate_manifest_key_count"] == 0
        and partial_audit["duplicate_partial_key_count"] == 0
        and partial_audit["missing_manifest_key_count"] == 0
        and partial_audit["extra_partial_key_count"] == 0
        and partial_audit["split_overlap_key_count"] == 0
        and partial_audit["key_sets_equal"]
    )

    state["phase"] = "fit_tsc_audit_models"
    arrays = _materialize_arrays(rows, partial_rows)
    models, model_audit, arrays = _fit_tsc_models(rows, arrays)
    data = _data_summary(rows, arrays, model_audit, partial_audit)

    state["phase"] = "identity_gradient"
    action_stats = _action_stats(postprocessor)
    action_semantics = _write_action_semantics(paths["action_semantics"], action_stats)
    identity, gradient = _identity_and_gradient_audit(policy, preprocessor, postprocessor, action_stats, rows, arrays, paths)
    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()

    decision_inputs = _decision_inputs(
        proposal_ok, serializer_ok, prior_check, manifest_ok, data, identity, gradient, prior_exception_count
    )
    decision = classify_stage0(decision_inputs)
    result = {
        "method": "TSC-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "model_hash": model_audit["model_hash"],
        "worker_pid": os.getpid(),
        "planned_model_row_count": len(rows),
        "completed_model_row_count": len(partial_rows),
        "resumed_model_row_count": resumed_count,
        "exception_count": prior_exception_count,
        "official_prior_asset_check": prior_check,
        "manifest_audit": manifest_audit,
        "partial_audit": partial_audit,
        "data_audit": data,
        "action_normalization": action_stats,
        "action_semantics": action_semantics,
        "identity": identity,
        "gradient": gradient,
        "decision_inputs": decision_inputs.__dict__,
        "final_decision": decision,
        "bounded_validation_allowed": decision == "TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION",
        "valid_scientific_result": False,
        "scientific_kill": False,
        "adapter_training_happened": False,
        "optimizer_step_count": 0,
        "simulator_load_count": 0,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "closed_loop_experiment_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "resource_evidence": preflight["resource_evidence"],
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "elapsed_seconds_not_paper_evidence": time.time() - started_unix,
        "completed_at": _utc_now(),
    }
    validation = {
        "proposal_hash_ok": proposal_ok,
        "serializer_preflight_ok": serializer_ok,
        "official_prior_asset_check_persisted": bool(prior_check),
        "manifest_json_parsed": True,
        "partial_json_parsed": True,
        "result_decision_recomputed": classify_stage0(decision_inputs),
        "worker_completed": True,
        "exception_count": prior_exception_count,
        "final_decision": decision,
        **partial_audit,
    }
    _write_json(paths["result_json"], result)
    _write_text(paths["result_md"], _result_markdown(result))
    _write_json(paths["validation"], validation)
    _write_text(paths["adjudication"], _adjudication_markdown(result))
    _write_json(
        paths["partial"],
        _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
    )
    state.update({"status": "completed", "phase": "complete", "completed_model_row_count": len(partial_rows)})
    _write_json(paths["status"], {**state, "completed_at": _utc_now(), "final_decision": decision})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now(), "final_decision": decision})
    return result


def _write_blocker(paths: Mapping[str, Path], state: Mapping[str, Any], exc: BaseException) -> None:
    detail = traceback.format_exc()
    manifest_hash = None
    planned = None
    rows: list[dict[str, Any]] = []
    previous_exceptions = 0
    if paths["manifest"].is_file():
        try:
            manifest = _read_json(paths["manifest"])
            manifest_hash = manifest.get("manifest_hash")
            planned = manifest.get("planned_model_row_count")
        except Exception:
            pass
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            rows = list(partial.get("rows") or [])
            previous_exceptions = int(partial.get("exception_count") or 0)
            manifest_hash = partial.get("manifest_hash", manifest_hash)
            planned = partial.get("planned_model_row_count", planned)
        except Exception:
            rows = []
    _write_json(
        paths["partial"],
        _partial_payload(
            manifest_hash,
            planned,
            rows,
            exception_count=previous_exceptions + 1,
            last_exception=f"{type(exc).__name__}: {exc}",
        ),
    )
    blocker = {
        "method": "TSC-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "status": "implementation_blocker",
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": detail,
        "state": dict(state),
        "partial_rows_preserved": len(rows),
        "written_at": _utc_now(),
    }
    _write_json(paths["blocker"], blocker)
    _write_json(
        paths["status"],
        {
            **state,
            "status": "failed",
            "phase": "implementation_blocker",
            "exception_count": previous_exceptions + 1,
            "completed_model_row_count": len(rows),
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        paths["heartbeat"],
        {
            **state,
            "status": "failed",
            "phase": "implementation_blocker",
            "exception_count": previous_exceptions + 1,
            "completed_model_row_count": len(rows),
            "updated_at": _utc_now(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="audit", choices=["audit"])
    parser.add_argument("--serializer-preflight", action="store_true")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "tsc_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "tsc_vla" / "stage0"))
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"TSC serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0

    state: dict[str, Any] = {
        "method": "TSC-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "pid": os.getpid(),
        "status": "starting",
        "phase": "startup",
        "started_at": _utc_now(),
    }
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["status"], state)
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    exit_code = 0
    try:
        result = run(args, paths, state)
        print(json.dumps({"final_decision": result["final_decision"], "result": str(paths["result_json"])}))
    except BaseException as exc:
        exit_code = 1
        _write_blocker(paths, state, exc)
        print(f"TSC Stage 0 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
    _write_text(paths["exit_code"], f"{exit_code}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
