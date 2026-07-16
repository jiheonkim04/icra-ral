"""Run the frozen AMP-VLA Stage 0 action-manifold development audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
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

from tca_map.smolvla.amp_vla import (  # noqa: E402
    ACTION_DIM,
    ABOT_HEADROOM_ABSOLUTE_HUBER_GATE,
    ABOT_HEADROOM_RELATIVE_GATE,
    CHUNK_SIZE,
    COORDINATE_ABSOLUTE_HUBER_GATE,
    COORDINATE_RELATIVE_GATE,
    LATENT_DIMS,
    MANIFOLD_RECON_ABSOLUTE_HUBER_GATE,
    MANIFOLD_RECON_RELATIVE_GATE,
    PHASE_BINS,
    PROPRIO_DIM,
    PROPOSAL_HASH,
    RIDGE_COEFFICIENT,
    STD_FLOOR,
    TASK_COUNT,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_chunk,
    amp_feature_key,
    amp_row_key,
    canonical_json_sha256,
    classify_stage0,
    decode_manifold,
    encode_manifold,
    fit_action_manifold,
    fit_ridge,
    flattened_chunks,
    json_default,
    manifold_consistency,
    mean_huber,
    phase_bin,
    predict_ridge,
    prediction_metrics,
    project_to_manifold,
    task_phase_mean_chunks,
    task_phase_mean_coordinates,
    validate_manifest,
)
from scripts.run_famr_vla_stage0 import (  # noqa: E402
    VLM_PATH,
    _active_linux_workers,
    _clone_batch,
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
    _extract_visual_feature,
    _gradient_values,
    _native_velocity,
)


POLICY_PROBE = "amp_stage0_action_manifold_projection"
SEED = 20262600
FLOW_TIME = 0.5
DISCOVERY_ROWS_PER_TASK = 128
VALIDATION_ROWS_PER_TASK = 32
GRADIENT_RATIO_MAX = 100.0
PROPOSAL_FILE = REPO_ROOT / "reports" / "amp_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "amp_vla" / "proposal_hash.txt"
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


def _asset_path(*parts: str) -> Path:
    root = Path("C:/assets") if os.name == "nt" else Path("/mnt/c/assets")
    return root.joinpath(*parts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.1)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "feature_dir": run / "visual_features",
        "adapter_dir": run / "identity_adapter",
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "pid": report / "stage_0_pid.txt",
        "heartbeat": report / "stage_0_heartbeat.json",
        "status": report / "stage_0_status.json",
        "preflight": report / "stage_0_preflight.json",
        "official_prior_asset_check": report / "stage_0_official_prior_asset_check.json",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _edge_hash(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest().upper()


def _array_sha256(value: Any) -> str:
    array = np.asarray(value, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


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
        "latent_dim": LATENT_DIMS[0],
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = amp_row_key(manifest_row)
    fixture = {
        "method": "AMP-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "action_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
        "manifold": {
            "latent_dim": LATENT_DIMS[0],
            "mean_preview": np.zeros(4, dtype=np.float32),
            "component_preview": np.eye(2, 4, dtype=np.float32),
            "coordinate_std_preview": np.ones(4, dtype=np.float32),
        },
        "projection_diagnostic": {
            "base_manifold_distance": np.float32(1.0),
            "clipped_base_manifold_distance": np.float32(0.9),
            "amp_manifold_distance": np.float32(0.5),
        },
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
            coordinate_variance_all_positive=True,
            manifold_reconstruction_relative_improvement=0.10,
            manifold_reconstruction_absolute_huber_improvement=0.0,
            coordinate_probe_relative_improvement=0.05,
            coordinate_probe_absolute_huber_improvement=0.0,
            abot_proxy_headroom_relative_improvement=0.05,
            abot_proxy_headroom_absolute_huber_improvement=0.0,
            clipping_explains_projection=False,
            projection_path_distinct=True,
            finite_objectives_and_gradients=True,
            amp_gradient_nonzero=True,
            gradient_ratio_at_most_100=True,
            frozen_parameter_gradient_count=0,
            identity_max_error=0.0,
            base_hash_unchanged=True,
            checkpoint_reload_ok=True,
            action_validity_ok=True,
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
        raise RuntimeError("AMP serializer preflight hash did not reproduce")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = {
        "official_repository": REPO_ROOT / "third_party" / "ABot-Manipulation",
        "asset_repository": _asset_path("ABot-Manipulation"),
        "checkpoint_dir": _asset_path("checkpoints", "abot_m0"),
        "inference_code": _asset_path("abot_m0", "inference"),
        "training_code": _asset_path("abot_m0", "training"),
    }
    observed = {name: str(value) for name, value in candidates.items()}
    exists = {name: value.exists() for name, value in candidates.items()}
    official_ready = bool(exists["official_repository"] and exists["checkpoint_dir"] and exists["inference_code"])
    label = "abot_m0_official" if official_ready else "abot_m0_action_manifold_proxy"
    deviations = [] if official_ready else [
        "official ABot-M0 repository/checkpoint/inference assets are not all locally verified",
        "Stage 0 fixes policy 2 as a transparent action-manifold proxy until official assets are installed",
    ]
    result = {
        "method": "AMP-VLA",
        "closest_prior": "ABot-M0",
        "official_ready": official_ready,
        "policy_2_label": label,
        "observed_paths": observed,
        "path_exists": exists,
        "deviations": deviations,
        "checked_at": _utc_now(),
    }
    _write_json(path, result)
    return result


def _problem_language(data: Any) -> str:
    raw = data.attrs.get("problem_info", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(str(raw))
        return str(parsed.get("language_instruction") or parsed.get("language") or "")
    except json.JSONDecodeError:
        return str(raw)


def _evenly_spaced(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    indices = np.floor(np.linspace(0, len(rows) - 1, count) + 0.5).astype(int).tolist()
    if len(indices) != len(set(indices)):
        raise RuntimeError("deterministic AMP sampler produced duplicate indices")
    return [rows[index] for index in indices]


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
                    for latent_dim in LATENT_DIMS:
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
                            "latent_dim": int(latent_dim),
                            "policy_probe": POLICY_PROBE,
                            "episode_length": int(len(actions)),
                            "phase": float(phase),
                            "phase_bin": phase_bin(phase),
                        }
                        row["row_key"] = amp_row_key(row)
                        row["feature_key"] = amp_feature_key(row)
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
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault((str(row["partition"]), str(row["task_identity"]), int(row["latent_dim"])), []).append(row)
    for group in sorted(groups):
        partition, _, _ = group
        target_count = DISCOVERY_ROWS_PER_TASK if partition == "discovery" else VALIDATION_ROWS_PER_TASK
        ordered = sorted(groups[group], key=lambda item: (int(item["demo_id"]), int(item["frame_index"])))
        selected = _evenly_spaced(ordered, target_count)
        if len(selected) != target_count:
            raise RuntimeError(f"AMP manifest group {group} has {len(selected)} rows, expected {target_count}")
        rows.extend(selected)
    rows.sort(
        key=lambda row: (
            row["partition"],
            row["suite"],
            row["task_identity"],
            int(row["latent_dim"]),
            int(row["demo_id"]),
            int(row["frame_index"]),
        )
    )
    return rows, sources


def _visual_feature_path(feature_dir: Path, feature_key: str) -> Path:
    digest = hashlib.sha256(feature_key.encode("utf-8")).hexdigest().upper()
    return feature_dir / f"{digest}.npz"


def _save_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, feature=np.asarray(feature, dtype=np.float16))
    temporary.replace(path)


def _load_feature(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        feature = np.asarray(loaded["feature"], dtype=np.float32)
    if feature.shape != (VISUAL_FEATURE_DIM,) or not np.isfinite(feature).all():
        raise RuntimeError(f"invalid AMP visual feature {path}: {feature.shape}")
    return feature


def _load_or_extract_feature(policy: Any, paths: Mapping[str, Path], row: Mapping[str, Any]) -> tuple[Path, np.ndarray]:
    path = _visual_feature_path(paths["feature_dir"], str(row["feature_key"]))
    if path.is_file():
        return path, _load_feature(path)
    feature = _extract_visual_feature(policy, row, int(row["frame_index"]))
    _save_feature(path, feature)
    return path, _load_feature(path)


def _proprio_from_obs(observations: Any, frame_index: int) -> np.ndarray:
    ee = np.asarray(observations["ee_states"][int(frame_index)], dtype=np.float64).reshape(-1)
    gripper = np.asarray(observations["gripper_states"][int(frame_index)], dtype=np.float64).reshape(-1)
    proprio = np.concatenate([ee, gripper])
    if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
        raise ValueError(f"AMP proprio must have shape [{PROPRIO_DIM}], received {proprio.shape}")
    return proprio


def _one_hot(index: int, width: int = TASK_COUNT) -> np.ndarray:
    value = np.zeros(int(width), dtype=np.float64)
    value[int(index)] = 1.0
    return value


def _raw_deployment_feature(visual: Any, proprio: Any, task_index: int, phase: float) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float64).reshape(-1)
    proprio_value = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if visual_value.shape != (VISUAL_FEATURE_DIM,):
        raise ValueError(f"visual feature must have shape [{VISUAL_FEATURE_DIM}], received {visual_value.shape}")
    if proprio_value.shape != (PROPRIO_DIM,):
        raise ValueError(f"proprio feature must have shape [{PROPRIO_DIM}], received {proprio_value.shape}")
    continuous = np.concatenate([visual_value, proprio_value, np.asarray([float(phase)], dtype=np.float64)])
    if not np.isfinite(continuous).all():
        raise ValueError("deployment feature contains nonfinite values")
    return np.concatenate([continuous, _one_hot(int(task_index))])


def _fit_zscore(features: np.ndarray) -> dict[str, Any]:
    continuous_dim = VISUAL_FEATURE_DIM + PROPRIO_DIM + 1
    continuous = np.asarray(features, dtype=np.float64)[:, :continuous_dim]
    return {
        "continuous_dim": continuous_dim,
        "task_count": TASK_COUNT,
        "mean": continuous.mean(axis=0),
        "std": np.maximum(continuous.std(axis=0, ddof=0), STD_FLOOR),
    }


def _apply_zscore(stats: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    continuous_dim = int(stats["continuous_dim"])
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(1, continuous_dim)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(1, continuous_dim)
    continuous = (value[:, :continuous_dim] - mean) / std
    return np.concatenate([continuous, value[:, continuous_dim:]], axis=1)


def _partial_row(row: Mapping[str, Any], feature_path: Path, feature: np.ndarray) -> dict[str, Any]:
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
        "latent_dim": int(row["latent_dim"]),
        "policy_probe": str(row["policy_probe"]),
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
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
        "method": "AMP-VLA",
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


def _load_resume(path: Path, manifest_rows: Sequence[Mapping[str, Any]], manifest_hash: str) -> tuple[list[dict[str, Any]], int, str | None]:
    if not path.is_file():
        return [], 0, None
    partial = _read_json(path)
    if partial.get("method") != "AMP-VLA" or partial.get("proposal_hash") != PROPOSAL_HASH:
        raise RuntimeError("partial result identity does not match frozen AMP proposal/manifest")
    rows = list(partial.get("rows") or [])
    if partial.get("manifest_hash") is None and not rows:
        return [], 0, None
    if partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial result identity does not match frozen AMP manifest")
    audit = validate_manifest(manifest_rows, rows)
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"partial contains duplicate or off-manifest keys: {audit}")
    for row in rows:
        cache = Path(str(row["feature_cache_path"]))
        if not cache.is_file() or _sha256(cache) != row["feature_cache_sha256"]:
            raise RuntimeError(f"feature cache hash mismatch for {row['row_key']}")
    return rows, int(partial.get("exception_count") or 0), partial.get("last_exception")


def _materialize_arrays(manifest: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    import h5py

    partial_by_key = {str(row["row_key"]): row for row in partial_rows}
    visuals = []
    proprios = []
    raw_features = []
    chunks = []
    latent_dims = []
    task_indices = []
    partitions = []
    phases = []
    task_names = []
    keys = []
    action_mins = []
    action_maxs = []
    action_finite = []
    for row in manifest:
        partial = partial_by_key[str(row["row_key"])]
        visual = _load_feature(Path(str(partial["feature_cache_path"])))
        with h5py.File(str(row["source_path"]), "r") as handle:
            demo = handle["data"][f"demo_{int(row['demo_id'])}"]
            observations = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            proprio = _proprio_from_obs(observations, int(row["frame_index"]))
            chunk = action_chunk(actions, int(row["frame_index"]))
        if _array_sha256(chunk) != partial["action_chunk_sha256"]:
            raise RuntimeError(f"action chunk hash mismatch for {row['row_key']}")
        raw_feature = _raw_deployment_feature(visual, proprio, int(row["task_index"]), float(row["phase"]))
        visuals.append(visual)
        proprios.append(proprio)
        raw_features.append(raw_feature)
        chunks.append(chunk)
        latent_dims.append(int(row["latent_dim"]))
        task_indices.append(int(row["task_index"]))
        partitions.append(str(row["partition"]))
        phases.append(float(row["phase"]))
        task_names.append(str(row["task_identity"]))
        keys.append(str(row["row_key"]))
        action_mins.append(float(np.min(chunk)))
        action_maxs.append(float(np.max(chunk)))
        action_finite.append(bool(np.isfinite(chunk).all()))
    return {
        "visual": np.asarray(visuals, dtype=np.float64),
        "proprio": np.asarray(proprios, dtype=np.float64),
        "raw_feature": np.asarray(raw_features, dtype=np.float64),
        "chunk": np.asarray(chunks, dtype=np.float64),
        "latent_dim": np.asarray(latent_dims, dtype=np.int64),
        "task_index": np.asarray(task_indices, dtype=np.int64),
        "partition": np.asarray(partitions, dtype=object),
        "phase": np.asarray(phases, dtype=np.float64),
        "task_identity": np.asarray(task_names, dtype=object),
        "row_key": np.asarray(keys, dtype=object),
        "action_min": np.asarray(action_mins, dtype=np.float64),
        "action_max": np.asarray(action_maxs, dtype=np.float64),
        "action_finite": np.asarray(action_finite, dtype=bool),
    }


def _fit_amp_models(
    manifest: Sequence[Mapping[str, Any]],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    arrays["zfeature"] = np.zeros_like(arrays["raw_feature"], dtype=np.float64)
    arrays["amp_prediction_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["abot_proxy_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["projected_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["predicted_coordinate"] = np.zeros((len(arrays["chunk"]), max(LATENT_DIMS)), dtype=np.float64)
    arrays["baseline_coordinate"] = np.zeros((len(arrays["chunk"]), max(LATENT_DIMS)), dtype=np.float64)
    arrays["true_coordinate"] = np.zeros((len(arrays["chunk"]), max(LATENT_DIMS)), dtype=np.float64)

    for latent_dim in LATENT_DIMS:
        selected = arrays["latent_dim"] == int(latent_dim)
        discovery = selected & (arrays["partition"] == "discovery")
        validation = selected & (arrays["partition"] == "validation")
        discovery_indices = np.flatnonzero(discovery)
        validation_indices = np.flatnonzero(validation)
        zscore = _fit_zscore(arrays["raw_feature"][discovery])
        zfeatures = _apply_zscore(zscore, arrays["raw_feature"][selected])
        selected_indices = np.flatnonzero(selected)
        arrays["zfeature"][selected_indices] = zfeatures
        selected_to_local = {int(index): local for local, index in enumerate(selected_indices)}
        local_discovery = [selected_to_local[int(index)] for index in discovery_indices]
        local_validation = [selected_to_local[int(index)] for index in validation_indices]

        manifold = fit_action_manifold(arrays["chunk"][discovery], latent_dim=int(latent_dim))
        projected_validation = project_to_manifold(manifold, arrays["chunk"][validation])
        action_baseline = task_phase_mean_chunks(
            arrays["chunk"][discovery],
            arrays["task_identity"][discovery],
            arrays["phase"][discovery],
            arrays["task_identity"][validation],
            arrays["phase"][validation],
            bins=PHASE_BINS,
        )
        manifold_metrics = prediction_metrics(
            flattened_chunks(projected_validation),
            flattened_chunks(action_baseline),
            flattened_chunks(arrays["chunk"][validation]),
        )

        discovery_coordinates = encode_manifold(manifold, arrays["chunk"][discovery], standardized=True)
        validation_coordinates = encode_manifold(manifold, arrays["chunk"][validation], standardized=True)
        raw_discovery_coordinates = encode_manifold(manifold, arrays["chunk"][discovery], standardized=False)
        coordinate_baseline = task_phase_mean_coordinates(
            discovery_coordinates,
            arrays["task_identity"][discovery],
            arrays["phase"][discovery],
            arrays["task_identity"][validation],
            arrays["phase"][validation],
            bins=PHASE_BINS,
        )
        coordinate_model = fit_ridge(zfeatures[local_discovery], discovery_coordinates)
        predicted_coordinates = predict_ridge(coordinate_model, zfeatures[local_validation])
        coordinate_metrics = prediction_metrics(predicted_coordinates, coordinate_baseline, validation_coordinates)

        amp_prediction = decode_manifold(manifold, predicted_coordinates, standardized=True)
        abot_proxy = decode_manifold(manifold, coordinate_baseline, standardized=True)
        headroom_metrics = prediction_metrics(
            flattened_chunks(amp_prediction),
            flattened_chunks(abot_proxy),
            flattened_chunks(arrays["chunk"][validation]),
        )
        clipped_abot = np.clip(abot_proxy, -1.0, 1.0)
        amp_manifold_distance = manifold_consistency(manifold, amp_prediction)
        abot_manifold_distance = manifold_consistency(manifold, abot_proxy)
        clipped_manifold_distance = manifold_consistency(manifold, clipped_abot)
        amp_validation_huber = mean_huber(amp_prediction, arrays["chunk"][validation])
        clipped_validation_huber = mean_huber(clipped_abot, arrays["chunk"][validation])
        projection_delta = flattened_chunks(amp_prediction) - flattened_chunks(abot_proxy)
        coordinate_variance = np.var(raw_discovery_coordinates, axis=0)

        arrays["projected_chunk"][validation_indices] = projected_validation
        arrays["amp_prediction_chunk"][validation_indices] = amp_prediction
        arrays["abot_proxy_chunk"][validation_indices] = abot_proxy
        arrays["predicted_coordinate"][validation_indices, : int(latent_dim)] = predicted_coordinates
        arrays["baseline_coordinate"][validation_indices, : int(latent_dim)] = coordinate_baseline
        arrays["true_coordinate"][validation_indices, : int(latent_dim)] = validation_coordinates
        arrays["true_coordinate"][discovery_indices, : int(latent_dim)] = discovery_coordinates

        model = {
            "latent_dim": int(latent_dim),
            "zscore": zscore,
            "manifold": manifold,
            "coordinate_model": coordinate_model,
            "discovery_row_keys": arrays["row_key"][discovery].tolist(),
            "validation_row_keys": arrays["row_key"][validation].tolist(),
        }
        models[str(latent_dim)] = model
        audits[str(latent_dim)] = {
            "latent_dim": int(latent_dim),
            "model_hash": canonical_json_sha256(model),
            "discovery_row_count": int(np.sum(discovery)),
            "validation_row_count": int(np.sum(validation)),
            "manifold_reconstruction": manifold_metrics,
            "coordinate_probe": coordinate_metrics,
            "abot_proxy_headroom": headroom_metrics,
            "coordinate_variance_by_dim": coordinate_variance,
            "coordinate_variance_all_positive": bool(np.all(coordinate_variance > 0.0)),
            "projection_delta_norm_mean": float(np.mean(np.linalg.norm(projection_delta, axis=1))),
            "projection_path_distinct": bool(float(np.mean(np.linalg.norm(projection_delta, axis=1))) > 1e-12),
            "amp_prediction_huber": amp_validation_huber,
            "abot_proxy_huber": mean_huber(abot_proxy, arrays["chunk"][validation]),
            "clipping_diagnostic": {
                "amp_manifold_distance": amp_manifold_distance,
                "abot_proxy_manifold_distance": abot_manifold_distance,
                "clipped_abot_manifold_distance": clipped_manifold_distance,
                "amp_validation_huber": amp_validation_huber,
                "clipped_abot_validation_huber": clipped_validation_huber,
                "clipping_explains_projection": bool(
                    clipped_manifold_distance <= amp_manifold_distance
                    and clipped_validation_huber <= amp_validation_huber
                ),
            },
        }

    model_hash = canonical_json_sha256(models)
    audit = {
        "model_hash": model_hash,
        "latent_dim_audits": audits,
        "latent_dims": list(LATENT_DIMS),
        "minimum_discovery_row_count": min(audits[str(dim)]["discovery_row_count"] for dim in LATENT_DIMS),
        "minimum_validation_row_count": min(audits[str(dim)]["validation_row_count"] for dim in LATENT_DIMS),
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
    latent_audits = model_audit["latent_dim_audits"]
    manifold = [latent_audits[str(dim)]["manifold_reconstruction"] for dim in LATENT_DIMS]
    coordinate = [latent_audits[str(dim)]["coordinate_probe"] for dim in LATENT_DIMS]
    headroom = [latent_audits[str(dim)]["abot_proxy_headroom"] for dim in LATENT_DIMS]
    finite_alignment = bool(
        np.isfinite(arrays["visual"]).all()
        and np.isfinite(arrays["proprio"]).all()
        and np.isfinite(arrays["raw_feature"]).all()
        and np.isfinite(arrays["chunk"]).all()
        and np.all(arrays["action_finite"])
    )
    return {
        "counts": counts,
        "minimum_discovery_windows": counts["discovery"],
        "minimum_validation_windows": counts["validation"],
        "feature_action_proprio_finite_aligned": finite_alignment,
        "validation_task_counts": validation_counts,
        "validation_task_fractions": validation_fractions,
        "maximum_validation_task_fraction": max(validation_fractions.values(), default=1.0),
        "all_tasks_reported": len(validation_counts) == len(TASK_SOURCES),
        "coordinate_variance_all_positive": all(
            bool(latent_audits[str(dim)]["coordinate_variance_all_positive"]) for dim in LATENT_DIMS
        ),
        "manifold_reconstruction_relative_improvement": min(item["relative_mse_improvement"] for item in manifold),
        "manifold_reconstruction_absolute_huber_improvement": min(item["absolute_huber_improvement"] for item in manifold),
        "manifold_reconstruction_relative_gate": MANIFOLD_RECON_RELATIVE_GATE,
        "manifold_reconstruction_absolute_huber_gate": MANIFOLD_RECON_ABSOLUTE_HUBER_GATE,
        "coordinate_probe_relative_improvement": min(item["relative_mse_improvement"] for item in coordinate),
        "coordinate_probe_absolute_huber_improvement": min(item["absolute_huber_improvement"] for item in coordinate),
        "coordinate_relative_gate": COORDINATE_RELATIVE_GATE,
        "coordinate_absolute_huber_gate": COORDINATE_ABSOLUTE_HUBER_GATE,
        "abot_proxy_headroom_relative_improvement": min(item["relative_mse_improvement"] for item in headroom),
        "abot_proxy_headroom_absolute_huber_improvement": min(item["absolute_huber_improvement"] for item in headroom),
        "abot_headroom_relative_gate": ABOT_HEADROOM_RELATIVE_GATE,
        "abot_headroom_absolute_huber_gate": ABOT_HEADROOM_ABSOLUTE_HUBER_GATE,
        "clipping_explains_projection": any(
            bool(latent_audits[str(dim)]["clipping_diagnostic"]["clipping_explains_projection"]) for dim in LATENT_DIMS
        ),
        "projection_path_distinct": all(
            bool(latent_audits[str(dim)]["projection_path_distinct"]) for dim in LATENT_DIMS
        ),
        "projection_delta_norm_mean_by_latent": {
            str(dim): latent_audits[str(dim)]["projection_delta_norm_mean"] for dim in LATENT_DIMS
        },
        "demo_action_validity_ok": bool(
            np.all(arrays["action_finite"])
            and float(np.min(arrays["action_min"])) >= -1.0
            and float(np.max(arrays["action_max"])) <= 1.0
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
            raise RuntimeError("AMP requires checkpoint MEAN_STD action statistics")
        mean = stats["mean"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        std = stats["std"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        return {"mode": "MEAN_STD", "mean": mean, "std": std, "processor_step": type(step).__name__}
    raise RuntimeError("checkpoint postprocessor has no action unnormalizer statistics")


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


def _unnormalize_native_action(native: Any, action_stats: Mapping[str, Any]) -> Any:
    import torch

    mean = torch.as_tensor(action_stats["mean"], dtype=native.dtype, device=native.device)
    std = torch.as_tensor(action_stats["std"], dtype=native.dtype, device=native.device)
    return native * std + mean


def _torch_standardized_coordinates(clean_native: Any, action_stats: Mapping[str, Any], manifold: Mapping[str, Any]) -> Any:
    import torch

    clean_raw = _unnormalize_native_action(clean_native, action_stats)
    flat = clean_raw.reshape(clean_raw.shape[0], CHUNK_SIZE * ACTION_DIM)
    mean = torch.as_tensor(manifold["mean"], dtype=flat.dtype, device=flat.device).reshape(1, -1)
    components = torch.as_tensor(manifold["components"], dtype=flat.dtype, device=flat.device)
    coord_mean = torch.as_tensor(manifold["coordinate_mean"], dtype=flat.dtype, device=flat.device).reshape(1, -1)
    coord_std = torch.as_tensor(manifold["coordinate_std"], dtype=flat.dtype, device=flat.device).reshape(1, -1)
    return ((flat - mean) @ components.T - coord_mean) / coord_std


def _amp_loss(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    time_value: Any,
    action_stats: Mapping[str, Any],
    manifold: Mapping[str, Any],
    target_raw_chunk: np.ndarray,
    target_coordinates: np.ndarray,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    flow_loss = _loss(policy, batch, noise, time_value)
    _, x_t, velocity = _native_velocity(policy, batch, noise, time_value)
    clean = x_t - time_value[:, None, None] * velocity
    clean_action = clean[:, :CHUNK_SIZE, :ACTION_DIM]
    target_raw = torch.as_tensor(target_raw_chunk, dtype=clean_action.dtype, device=clean_action.device).reshape(1, CHUNK_SIZE, ACTION_DIM)
    target_native = _normalize_raw_action(target_raw, action_stats)
    predicted_coordinates = _torch_standardized_coordinates(clean_action, action_stats, manifold)
    target_coord = torch.as_tensor(target_coordinates, dtype=predicted_coordinates.dtype, device=predicted_coordinates.device).reshape(1, -1)
    coord_loss = functional.smooth_l1_loss(predicted_coordinates, target_coord, beta=1.0)
    projection_loss = functional.smooth_l1_loss(clean_action, target_native, beta=1.0)
    clean_loss = functional.smooth_l1_loss(clean_action, batch["action"][:, :CHUNK_SIZE, :ACTION_DIM], beta=1.0)
    total = flow_loss + coord_loss + projection_loss + 0.1 * clean_loss
    return total, flow_loss, coord_loss, projection_loss, clean_loss


def _gradient_audit(
    policy: Any,
    preprocessor: Any,
    action_stats: Mapping[str, Any],
    models: Mapping[str, Any],
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
        raise RuntimeError("AMP expected only LoRA trainable parameters")

    policy.train()
    policy.zero_grad(set_to_none=True)
    flow_loss = _loss(policy, batch, noise, time_value)
    flow_gradients, flow_norm, flow_finite = _gradient_values(flow_loss, named)
    flow_value = float(flow_loss.detach().item())
    del flow_loss
    gc.collect()
    torch.cuda.empty_cache()

    latent_dim = int(row["latent_dim"])
    model = models[str(latent_dim)]
    target = np.asarray(arrays["amp_prediction_chunk"][row_index], dtype=np.float64)
    target_coordinates = encode_manifold(model["manifold"], target.reshape(1, CHUNK_SIZE, ACTION_DIM), standardized=True)[0]
    policy.zero_grad(set_to_none=True)
    amp_loss, flow_component, coord_component, projection_component, clean_component = _amp_loss(
        policy, batch, noise, time_value, action_stats, model["manifold"], target, target_coordinates
    )
    amp_gradients, amp_norm, amp_finite = _gradient_values(amp_loss, named)
    amp_value = float(amp_loss.detach().item())
    coord_value = float(coord_component.detach().item())
    projection_value = float(projection_component.detach().item())
    clean_value = float(clean_component.detach().item())
    dot = 0.0
    for flow_gradient, amp_gradient in zip(flow_gradients, amp_gradients, strict=True):
        if flow_gradient is not None and amp_gradient is not None:
            dot += float(torch.sum(flow_gradient * amp_gradient).item())
    cosine = dot / max(flow_norm * amp_norm, 1e-12)
    frozen_gradient_names = [
        name for name, parameter in policy.named_parameters() if "lora_" not in name.lower() and parameter.grad is not None
    ]
    policy.zero_grad(set_to_none=True)
    policy.eval()
    return {
        "flow_time": FLOW_TIME,
        "flow_loss": flow_value,
        "amp_loss": amp_value,
        "L_flow": float(flow_component.detach().item()),
        "L_coord": coord_value,
        "L_proj": projection_value,
        "L_clean": clean_value,
        "trainable_parameter_names": [name for name, _ in named],
        "trainable_parameter_count": len(named),
        "trainable_numel": sum(int(parameter.numel()) for _, parameter in named),
        "flow_gradient_norm": flow_norm,
        "amp_gradient_norm": amp_norm,
        "amp_to_flow_gradient_ratio": amp_norm / max(flow_norm, 1e-12),
        "gradient_cosine": cosine,
        "flow_gradient_finite_fraction": flow_finite,
        "amp_gradient_finite_fraction": amp_finite,
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "frozen_parameter_gradient_names": frozen_gradient_names,
        "amp_gradient_nonzero": amp_norm > 0.0,
        "finite_objectives_and_gradients": bool(
            np.isfinite([flow_value, amp_value, coord_value, projection_value, clean_value, flow_norm, amp_norm, cosine]).all()
            and flow_finite == 1.0
            and amp_finite == 1.0
        ),
        "gradient_row_key": str(row["row_key"]),
        "gradient_latent_dim": latent_dim,
    }


def _identity_and_gradient_audit(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    action_stats: Mapping[str, Any],
    models: Mapping[str, Any],
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

    gradient = _gradient_audit(policy, preprocessor, action_stats, models, manifest, arrays)
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
    amp_target = np.asarray(arrays["amp_prediction_chunk"][row_index], dtype=np.float64)
    abot_target = np.asarray(arrays["abot_proxy_chunk"][row_index], dtype=np.float64)
    projection_delta = amp_target - abot_target
    residual_norm = float(np.linalg.norm(projection_delta))
    dimension_delta = np.max(np.abs(projection_delta), axis=0)
    identity = {
        "rank": 4,
        "base_flow_shape": list(base_flow.shape),
        "base_native_shape": list(base_native.shape),
        "base_action_shape": list(base_actions.shape),
        "base_action": base_actions[: min(3, len(base_actions))].tolist(),
        "ours_action_diagnostic": amp_target[: min(3, len(amp_target))].tolist(),
        "abot_proxy_action_diagnostic": abot_target[: min(3, len(abot_target))].tolist(),
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
        "base_action_in_bounds": bool(np.all(np.abs(base_actions) <= 1.0)),
        "amp_residual_norm": residual_norm,
        "amp_gate_value": 0.0,
        "projection_delta_norm": residual_norm,
        "translation_delta_norm": float(np.linalg.norm(projection_delta[:, :3])),
        "rotation_delta_norm": float(np.linalg.norm(projection_delta[:, 3:6])),
        "gripper_delta_norm": float(np.linalg.norm(projection_delta[:, 6])),
        "changed_dimensions": [int(index) for index, value in enumerate(dimension_delta) if float(value) > 1e-12],
        "activation_context": str(row["row_key"]),
        "training_only_amp_absent_from_policy_parameters": not any(
            "amp" in name.lower() for name, _ in reloaded.named_parameters()
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
        coordinate_variance_all_positive=bool(data["coordinate_variance_all_positive"]),
        manifold_reconstruction_relative_improvement=float(data["manifold_reconstruction_relative_improvement"]),
        manifold_reconstruction_absolute_huber_improvement=float(data["manifold_reconstruction_absolute_huber_improvement"]),
        coordinate_probe_relative_improvement=float(data["coordinate_probe_relative_improvement"]),
        coordinate_probe_absolute_huber_improvement=float(data["coordinate_probe_absolute_huber_improvement"]),
        abot_proxy_headroom_relative_improvement=float(data["abot_proxy_headroom_relative_improvement"]),
        abot_proxy_headroom_absolute_huber_improvement=float(data["abot_proxy_headroom_absolute_huber_improvement"]),
        clipping_explains_projection=bool(data["clipping_explains_projection"]),
        projection_path_distinct=bool(data["projection_path_distinct"]),
        finite_objectives_and_gradients=bool(gradient.get("finite_objectives_and_gradients", False)),
        amp_gradient_nonzero=bool(gradient.get("amp_gradient_nonzero", False)),
        gradient_ratio_at_most_100=float(gradient.get("amp_to_flow_gradient_ratio", float("inf"))) <= GRADIENT_RATIO_MAX,
        frozen_parameter_gradient_count=int(gradient.get("frozen_parameter_gradient_count", 0)),
        identity_max_error=float(identity.get("identity_max_abs_error", 0.0)),
        base_hash_unchanged=bool(identity.get("base_hash_unchanged", True)),
        checkpoint_reload_ok=bool(identity.get("checkpoint_reload_ok", True)),
        action_validity_ok=bool(identity.get("base_action_in_bounds", True)) and bool(data.get("demo_action_validity_ok", True)),
        exception_count=int(exception_count),
    )


def _result_markdown(result: Mapping[str, Any]) -> str:
    data = result["data_audit"]
    gradient = result.get("gradient") or {}
    identity = result.get("identity") or {}
    return "\n".join(
        [
            "# AMP-VLA Stage 0 Result",
            "",
            f"Final decision: `{result['final_decision']}`.",
            "",
            f"Rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`.",
            f"ABot-M0 prior label: `{result['official_prior_asset_check']['policy_2_label']}`.",
            f"Manifold reconstruction relative / absolute Huber gain: `{data['manifold_reconstruction_relative_improvement']} / {data['manifold_reconstruction_absolute_huber_improvement']}`.",
            f"Coordinate probe relative / absolute Huber gain: `{data['coordinate_probe_relative_improvement']} / {data['coordinate_probe_absolute_huber_improvement']}`.",
            f"AMP minus ABot-proxy relative / absolute Huber gain: `{data['abot_proxy_headroom_relative_improvement']} / {data['abot_proxy_headroom_absolute_huber_improvement']}`.",
            f"Clipping explains projection: `{data['clipping_explains_projection']}`.",
            f"Flow / AMP gradient norm: `{gradient.get('flow_gradient_norm')} / {gradient.get('amp_gradient_norm')}`.",
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
            "# AMP-VLA Stage 0 Adjudication",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            "This is a development-only audit, not a closed-loop scientific kill.",
            f"Bounded validation allowed: `{result['bounded_validation_allowed']}`.",
            f"Valid scientific result: `{result['valid_scientific_result']}`.",
            "",
            "The frozen Stage 0 gates were applied without changing manifold construction, latent dimensions, task sources, prior proxy, clipping diagnostic, baselines, or thresholds.",
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
        if "run_amp_vla_stage0.py" not in command:
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
        "active_amp_linux_workers": workers,
        "resource_evidence": _resource_evidence(registry, started_unix),
    }


def run(args: argparse.Namespace, paths: Mapping[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    import torch

    _set_offline_environment()
    started_unix = time.time()
    preflight = _preflight(paths, started_unix)
    _write_json(paths["preflight"], preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"AMP Stage 0 preflight failed: {preflight}")
    serializer = _read_json(paths["serializer_preflight"])
    serializer_ok = bool(
        serializer.get("passed") and canonical_json_sha256(serializer["fixture"]) == serializer.get("fixture_hash")
    )
    if not serializer_ok:
        raise RuntimeError("foreground AMP serializer preflight is absent or invalid")
    prior_check = _official_prior_asset_check(paths["official_prior_asset_check"])
    proposal_observed = _sha256(PROPOSAL_FILE)
    proposal_registry = _proposal_hash_text()
    proposal_ok = proposal_observed == proposal_registry == PROPOSAL_HASH
    if not proposal_ok:
        raise RuntimeError("frozen AMP proposal hash mismatch")

    state.update({"phase": "manifest", "status": "running"})
    rows, sources = _build_manifest(paths["data_root"])
    manifest_payload = {
        "method": "AMP-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "sources": sources,
        "latent_dims": list(LATENT_DIMS),
        "chunk_size": CHUNK_SIZE,
        "phase_bins": PHASE_BINS,
        "ridge_coefficient": RIDGE_COEFFICIENT,
        "std_floor": STD_FLOOR,
        "discovery_rows_per_task_latent": DISCOVERY_ROWS_PER_TASK,
        "validation_rows_per_task_latent": VALIDATION_ROWS_PER_TASK,
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
        raise RuntimeError("persisted AMP manifest hash did not reproduce")
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
        partial_rows.append(_partial_row(row, feature_path, feature))
        completed.add(key)
        state["completed_model_row_count"] = len(partial_rows)
        _write_json(
            paths["partial"],
            _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
        )
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        if len(partial_rows) % 16 == 0 or len(partial_rows) == len(rows):
            print(f"[amp-stage0] rows {len(partial_rows)}/{len(rows)}", flush=True)

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

    state["phase"] = "fit_amp_audit_models"
    arrays = _materialize_arrays(rows, partial_rows)
    models, model_audit, arrays = _fit_amp_models(rows, arrays)
    data = _data_summary(rows, arrays, model_audit, partial_audit)

    state["phase"] = "identity_gradient"
    action_stats = _action_stats(postprocessor)
    identity, gradient = _identity_and_gradient_audit(
        policy, preprocessor, postprocessor, action_stats, models, rows, arrays, paths
    )
    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()

    decision_inputs = _decision_inputs(
        proposal_ok, serializer_ok, prior_check, manifest_ok, data, identity, gradient, prior_exception_count
    )
    decision = classify_stage0(decision_inputs)
    result = {
        "method": "AMP-VLA",
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
        "identity": identity,
        "gradient": gradient,
        "decision_inputs": decision_inputs.__dict__,
        "final_decision": decision,
        "bounded_validation_allowed": decision == "AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION",
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
        "method": "AMP-VLA",
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
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "amp_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "amp_vla" / "stage0"))
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"AMP serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0

    state: dict[str, Any] = {
        "method": "AMP-VLA",
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
        print(f"AMP Stage 0 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
    _write_text(paths["exit_code"], f"{exit_code}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
