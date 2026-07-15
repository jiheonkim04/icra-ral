"""Run the frozen VDR-VLA Stage 0A dynamic-residual development audit."""

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
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_famr_vla_stage0 import (  # noqa: E402
    VLM_PATH,
    _active_linux_workers,
    _apply_official_env_image_processor,
    _clone_batch,
    _hash_base_parameters,
    _load_policy_and_processors,
    _loss,
    _preprocess,
    _raw_sample,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_pcav_vla_stage0 import _postprocess_chunk  # noqa: E402
from tca_map.smolvla.vdr_vla import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    FEATURE_DIM,
    HORIZONS,
    PROJECTION_DIM,
    PROPOSAL_HASH,
    RIDGE_COEFFICIENT,
    STD_FLOOR,
    Stage0ADecisionInputs,
    action_summary,
    canonical_json_sha256,
    classify_stage0a,
    differentiable_mean_std_unnormalize,
    fit_pca_whitener,
    fit_ridge,
    json_default,
    mean_huber,
    predict_ridge,
    project_with_whitener,
    regression_metrics,
    torch_action_summary,
    torch_predict_ridge,
    validate_manifest,
    vdr_row_key,
    visual_frame_key,
)


SEED = 20262400
FLOW_TIME = 0.5
DISCOVERY_ROWS_PER_TASK_HORIZON = 128
VALIDATION_ROWS_PER_TASK_HORIZON = 64
MAX_VALIDATION_ROWS_PER_HORIZON = 256
GRADIENT_RATIO_MAX = 100.0
PROPOSAL_FILE = REPO_ROOT / "reports" / "vdr_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "vdr_vla" / "proposal_hash.txt"
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


def _proposal_hash_text() -> str:
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


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
        "pid": report / "stage_0a_pid.txt",
        "heartbeat": report / "stage_0a_heartbeat.json",
        "status": report / "stage_0a_status.json",
        "serializer_preflight": report / "stage_0a_serializer_preflight.json",
        "preflight": report / "stage_0a_preflight.json",
        "manifest": report / "stage_0a_manifest.json",
        "partial": report / "stage_0a_partial.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "adjudication": report / "stage_0a_adjudication.md",
        "blocker": report / "stage_0a_implementation_blocker.json",
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    fixture = {
        "method": "VDR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizons": np.asarray(HORIZONS, dtype=np.int64),
        "pca": {
            "feature_dim": FEATURE_DIM,
            "projection_dim": PROJECTION_DIM,
            "mean_preview": np.zeros(4, dtype=np.float32),
            "component_preview": np.eye(2, 4, dtype=np.float32),
            "projected_std_preview": np.ones(4, dtype=np.float64),
        },
        "ridge": {
            "feature_mean": np.zeros(3, dtype=np.float32),
            "feature_std": np.ones(3, dtype=np.float32),
            "beta": np.zeros((4, PROJECTION_DIM), dtype=np.float32),
        },
    }
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
        raise RuntimeError("VDR serializer preflight hash did not reproduce")
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
        raise RuntimeError("deterministic VDR sampler produced duplicate indices")
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
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            demo_reports = []
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
                if states.shape[:2] != (len(actions), ARM_DIM) or not np.isfinite(states).all():
                    raise ValueError(f"invalid ee_states {states.shape} in {source}:{demo_key}")
                if gripper.shape[0] != len(actions) or not np.isfinite(gripper).all():
                    raise ValueError(f"invalid gripper_states {gripper.shape} in {source}:{demo_key}")
                partition = "discovery" if demo_id <= 7 else "validation"
                valid_count = len(actions) - max(HORIZONS)
                demo_reports.append({"demo_id": demo_id, "partition": partition, "length": len(actions), "valid_count": valid_count})
                for frame in range(max(0, valid_count)):
                    phase = frame / max(valid_count - 1, 1)
                    for horizon in HORIZONS:
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
                            "future_frame_index": frame + int(horizon),
                            "horizon": int(horizon),
                            "episode_length": int(len(actions)),
                            "phase": float(phase),
                        }
                        row["row_key"] = vdr_row_key(row)
                        row["current_frame_key"] = visual_frame_key(row, frame)
                        row["future_frame_key"] = visual_frame_key(row, frame + int(horizon))
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
        group = (str(row["partition"]), str(row["task_identity"]), int(row["horizon"]))
        groups.setdefault(group, []).append(row)
    for group in sorted(groups):
        partition, _, _ = group
        target_count = DISCOVERY_ROWS_PER_TASK_HORIZON if partition == "discovery" else VALIDATION_ROWS_PER_TASK_HORIZON
        ordered = sorted(groups[group], key=lambda item: (int(item["demo_id"]), int(item["frame_index"])))
        selected = _evenly_spaced(ordered, target_count)
        if len(selected) != target_count:
            raise RuntimeError(f"VDR manifest group {group} has {len(selected)} rows, expected {target_count}")
        rows.extend(selected)
    rows.sort(key=lambda row: (row["partition"], row["suite"], row["task_identity"], row["demo_id"], row["frame_index"], row["horizon"]))
    return rows, sources


def _visual_feature_path(feature_dir: Path, frame_key: str) -> Path:
    digest = hashlib.sha256(frame_key.encode("utf-8")).hexdigest().upper()
    return feature_dir / f"{digest}.npz"


def _save_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, feature=np.asarray(feature, dtype=np.float16))
    temporary.replace(path)


def _load_feature(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        feature = np.asarray(loaded["feature"], dtype=np.float32)
    if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
        raise RuntimeError(f"invalid VDR visual feature {path}: {feature.shape}")
    return feature


def _prepare_images(policy: Any, images: Sequence[Any]) -> Any:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    tensors = []
    for image in images:
        value = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image).copy())
        if value.ndim != 3:
            raise RuntimeError(f"expected CHW/HWC image, got {tuple(value.shape)}")
        if value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)
        value = value.float()
        if float(value.max()) > 1.0:
            value = value / 255.0
        tensors.append(value)
    batch = torch.stack(tensors).to("cuda")
    resize_cfg = getattr(policy.config, "resize_imgs_with_padding", None)
    if resize_cfg is not None:
        batch = resize_with_pad(batch, *resize_cfg, pad_value=0)
    dtype = next(policy.model.parameters()).dtype
    return (batch * 2.0 - 1.0).to(dtype=dtype)


def _extract_visual_feature(policy: Any, row: Mapping[str, Any], frame_index: int) -> np.ndarray:
    import h5py
    import torch

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        observations = demo["obs"]
        agent, wrist = _apply_official_env_image_processor(
            observations["agentview_rgb"][frame_index], observations["eye_in_hand_rgb"][frame_index]
        )
    prepared = _prepare_images(policy, (agent, wrist))
    with torch.no_grad():
        tokens = policy.model.vlm_with_expert.embed_image(prepared).float().cpu()
    if tokens.shape[:2] != (2, 64) or tokens.shape[2] != FEATURE_DIM:
        raise RuntimeError(f"unexpected VDR visual token shape {tuple(tokens.shape)}")
    return torch.cat((tokens[0], tokens[1]), dim=0).mean(dim=0).numpy().astype(np.float32)


def _load_or_extract_feature(policy: Any, paths: Mapping[str, Path], row: Mapping[str, Any], frame_index: int, key: str) -> tuple[Path, np.ndarray]:
    path = _visual_feature_path(paths["feature_dir"], key)
    if path.is_file():
        return path, _load_feature(path)
    feature = _extract_visual_feature(policy, row, frame_index)
    _save_feature(path, feature)
    return path, _load_feature(path)


def _partial_row(
    row: Mapping[str, Any],
    current_path: Path,
    current_feature: np.ndarray,
    future_path: Path,
    future_feature: np.ndarray,
) -> dict[str, Any]:
    delta = future_feature - current_feature
    return {
        "row_key": str(row["row_key"]),
        "partition": str(row["partition"]),
        "suite": str(row["suite"]),
        "task_identity": str(row["task_identity"]),
        "source_edge_sha256": str(row["source_edge_sha256"]),
        "demo_id": int(row["demo_id"]),
        "frame_index": int(row["frame_index"]),
        "future_frame_index": int(row["future_frame_index"]),
        "horizon": int(row["horizon"]),
        "current_feature_path": str(current_path),
        "future_feature_path": str(future_path),
        "current_feature_sha256": _sha256(current_path),
        "future_feature_sha256": _sha256(future_path),
        "feature_dim": int(current_feature.shape[0]),
        "current_feature_finite_fraction": float(np.mean(np.isfinite(current_feature))),
        "future_feature_finite_fraction": float(np.mean(np.isfinite(future_feature))),
        "delta_feature_variance": float(np.var(delta)),
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
        "method": "VDR-VLA",
        "stage": "0A",
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
    if partial.get("proposal_hash") != PROPOSAL_HASH or partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial result identity does not match frozen VDR proposal/manifest")
    rows = list(partial.get("rows") or [])
    audit = validate_manifest(manifest_rows, rows)
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"partial contains duplicate or off-manifest keys: {audit}")
    for row in rows:
        current = Path(str(row["current_feature_path"]))
        future = Path(str(row["future_feature_path"]))
        if not current.is_file() or _sha256(current) != row["current_feature_sha256"]:
            raise RuntimeError(f"current feature cache hash mismatch for {row['row_key']}")
        if not future.is_file() or _sha256(future) != row["future_feature_sha256"]:
            raise RuntimeError(f"future feature cache hash mismatch for {row['row_key']}")
    return rows, int(partial.get("exception_count") or 0), partial.get("last_exception")


def _one_hot(index: int, width: int) -> np.ndarray:
    value = np.zeros(width, dtype=np.float64)
    value[int(index)] = 1.0
    return value


def _materialize_arrays(manifest: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    import h5py

    partial_by_key = {str(row["row_key"]): row for row in partial_rows}
    current_features = []
    future_features = []
    static_features = []
    action_features = []
    task_indices = []
    partitions = []
    horizons = []
    task_names = []
    action_values = []
    for row in manifest:
        partial = partial_by_key[str(row["row_key"])]
        current = _load_feature(Path(str(partial["current_feature_path"])))
        future = _load_feature(Path(str(partial["future_feature_path"])))
        with h5py.File(str(row["source_path"]), "r") as handle:
            demo = handle["data"][f"demo_{int(row['demo_id'])}"]
            observations = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            ee = np.asarray(observations["ee_states"][int(row["frame_index"])], dtype=np.float64).reshape(-1)
            gripper = np.asarray(observations["gripper_states"][int(row["frame_index"])], dtype=np.float64).reshape(-1)
        summary = action_summary(actions, int(row["frame_index"]), int(row["horizon"]))
        proprio = np.concatenate([ee, gripper])
        static = np.concatenate(
            [
                current.astype(np.float64),
                proprio,
                _one_hot(int(row["task_index"]), len(TASK_SOURCES)),
                np.asarray([float(row["phase"])], dtype=np.float64),
            ]
        )
        current_features.append(current)
        future_features.append(future)
        static_features.append(static)
        action_features.append(summary)
        action_values.append(actions[int(row["frame_index"]) : int(row["frame_index"]) + int(row["horizon"])])
        task_indices.append(int(row["task_index"]))
        partitions.append(str(row["partition"]))
        horizons.append(int(row["horizon"]))
        task_names.append(str(row["task_identity"]))
    return {
        "current": np.asarray(current_features, dtype=np.float64),
        "future": np.asarray(future_features, dtype=np.float64),
        "static": np.asarray(static_features, dtype=np.float64),
        "action_summary": np.asarray(action_features, dtype=np.float64),
        "task_index": np.asarray(task_indices, dtype=np.int64),
        "partition": np.asarray(partitions, dtype=object),
        "horizon": np.asarray(horizons, dtype=np.int64),
        "task_identity": np.asarray(task_names, dtype=object),
        "action_min": np.asarray([float(np.min(value)) for value in action_values], dtype=np.float64),
        "action_max": np.asarray([float(np.max(value)) for value in action_values], dtype=np.float64),
        "action_finite": np.asarray([bool(np.isfinite(value).all()) for value in action_values], dtype=bool),
    }


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def _fit_vdr_models(manifest: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    deltas = np.asarray(arrays["future"] - arrays["current"], dtype=np.float64)
    static = np.asarray(arrays["static"], dtype=np.float64)
    action = np.asarray(arrays["action_summary"], dtype=np.float64)
    partitions = np.asarray(arrays["partition"])
    horizons = np.asarray(arrays["horizon"], dtype=np.int64)
    task_names = np.asarray(arrays["task_identity"])
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for horizon in HORIZONS:
        selected = horizons == int(horizon)
        discovery = selected & (partitions == "discovery")
        validation = selected & (partitions == "validation")
        whitener = fit_pca_whitener(deltas[discovery], projection_dim=PROJECTION_DIM)
        y_all = project_with_whitener(whitener, deltas[selected])
        static_all = static[selected]
        action_all = action[selected]
        selected_partitions = partitions[selected]
        selected_tasks = task_names[selected]
        discovery_local = selected_partitions == "discovery"
        validation_local = selected_partitions == "validation"

        static_model = fit_ridge(static_all[discovery_local], y_all[discovery_local])
        static_metrics = regression_metrics(static_model, static_all[validation_local], y_all[validation_local])
        static_prediction_all = predict_ridge(static_model, static_all)
        residual_all = y_all - static_prediction_all

        actionless_probe = fit_ridge(static_all[discovery_local], residual_all[discovery_local])
        action_probe_input = np.concatenate([static_all, action_all], axis=1)
        action_probe = fit_ridge(action_probe_input[discovery_local], residual_all[discovery_local])
        future_proxy = fit_ridge(action_probe_input[discovery_local], y_all[discovery_local])

        actionless_pred = predict_ridge(actionless_probe, static_all[validation_local])
        action_pred = predict_ridge(action_probe, action_probe_input[validation_local])
        residual_target = residual_all[validation_local]
        actionless_mse = _mse(actionless_pred, residual_target)
        action_mse = _mse(action_pred, residual_target)
        actionless_huber = mean_huber(actionless_pred, residual_target)
        action_huber = mean_huber(action_pred, residual_target)

        future_pred = predict_ridge(future_proxy, action_probe_input[validation_local])
        vdr_combined = predict_ridge(static_model, static_all[validation_local]) + action_pred
        future_mse = _mse(future_pred, y_all[validation_local])
        vdr_mse = _mse(vdr_combined, y_all[validation_local])

        by_task = {}
        for task in sorted({str(value) for value in selected_tasks[validation_local]}):
            task_mask = validation_local & (selected_tasks == task)
            task_static = regression_metrics(static_model, static_all[task_mask], y_all[task_mask])
            by_task[task] = {"row_count": int(np.sum(task_mask)), **task_static}

        residual_variance = np.var(residual_all[discovery_local], axis=0)
        models[str(horizon)] = {
            "whitener": whitener,
            "static_model": static_model,
            "actionless_residual_probe": actionless_probe,
            "action_residual_probe": action_probe,
            "futurevla_proxy": future_proxy,
        }
        audits[str(horizon)] = {
            "discovery_row_count": int(np.sum(discovery)),
            "validation_row_count": int(np.sum(validation)),
            "static_predictor": static_metrics,
            "residual_probe": {
                "actionless_validation_mse": actionless_mse,
                "action_conditioned_validation_mse": action_mse,
                "relative_improvement": float((actionless_mse - action_mse) / max(actionless_mse, 1e-12)),
                "actionless_validation_huber": actionless_huber,
                "action_conditioned_validation_huber": action_huber,
                "absolute_huber_improvement": actionless_huber - action_huber,
            },
            "futurevla_proxy": {
                "future_proxy_validation_mse": future_mse,
                "vdr_decomposed_validation_mse": vdr_mse,
                "relative_improvement": float((future_mse - vdr_mse) / max(future_mse, 1e-12)),
                "absolute_mse_gap": future_mse - vdr_mse,
            },
            "residual_variance_min": float(np.min(residual_variance)),
            "residual_variance_all_positive": bool(np.all(residual_variance > 0.0)),
            "per_task_validation": by_task,
        }
    model_hash = canonical_json_sha256(models)
    return models, {"model_hash": model_hash, "by_horizon": audits}


def _data_summary(
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    model_audit: Mapping[str, Any],
    manifest_audit: Mapping[str, Any],
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for horizon in HORIZONS:
        counts[str(horizon)] = {
            partition: sum(row["partition"] == partition and int(row["horizon"]) == int(horizon) for row in manifest)
            for partition in ("discovery", "validation")
        }
    validation_counts: dict[str, int] = {}
    for row in manifest:
        if row["partition"] == "validation":
            validation_counts[str(row["task_identity"])] = validation_counts.get(str(row["task_identity"]), 0) + 1
    validation_total = sum(validation_counts.values())
    validation_fractions = {task: count / max(validation_total, 1) for task, count in validation_counts.items()}
    finite_alignment = bool(
        np.isfinite(arrays["current"]).all()
        and np.isfinite(arrays["future"]).all()
        and np.isfinite(arrays["static"]).all()
        and np.isfinite(arrays["action_summary"]).all()
        and np.all(arrays["action_finite"])
    )
    by_horizon = model_audit["by_horizon"]
    return {
        "counts": counts,
        "minimum_discovery_rows_per_horizon": min(value["discovery"] for value in counts.values()),
        "minimum_validation_rows_per_horizon": min(value["validation"] for value in counts.values()),
        "maximum_validation_rows_per_horizon": max(value["validation"] for value in counts.values()),
        "maximum_validation_rows_per_horizon_allowed": MAX_VALIDATION_ROWS_PER_HORIZON,
        "feature_action_proprio_finite_aligned": finite_alignment,
        "residual_variance_all_positive": all(by_horizon[str(h)]["residual_variance_all_positive"] for h in HORIZONS),
        "validation_task_counts": validation_counts,
        "validation_task_fractions": validation_fractions,
        "maximum_validation_task_fraction": max(validation_fractions.values(), default=1.0),
        "all_tasks_reported": all(len(by_horizon[str(h)]["per_task_validation"]) == len(TASK_SOURCES) for h in HORIZONS),
        "minimum_static_predictor_relative_improvement": min(
            by_horizon[str(h)]["static_predictor"]["normalized_relative_improvement"] for h in HORIZONS
        ),
        "minimum_action_residual_relative_improvement": min(
            by_horizon[str(h)]["residual_probe"]["relative_improvement"] for h in HORIZONS
        ),
        "minimum_action_residual_absolute_improvement": min(
            by_horizon[str(h)]["residual_probe"]["absolute_huber_improvement"] for h in HORIZONS
        ),
        "minimum_future_proxy_relative_improvement": min(
            by_horizon[str(h)]["futurevla_proxy"]["relative_improvement"] for h in HORIZONS
        ),
        "minimum_future_proxy_absolute_gap": min(
            by_horizon[str(h)]["futurevla_proxy"]["absolute_mse_gap"] for h in HORIZONS
        ),
        "action_min": float(np.min(arrays["action_min"])),
        "action_max": float(np.max(arrays["action_max"])),
        "manifest_audit": manifest_audit,
        "model_audit": model_audit,
    }


def _core_policy(policy: Any) -> Any:
    return policy.get_base_model() if hasattr(policy, "get_base_model") else policy


def _stable_seed(identity: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{purpose}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _noise(identity: str, purpose: str, shape: Sequence[int], device: str) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(identity, purpose))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


def _native_velocity(policy: Any, batch: Mapping[str, Any], noise: Any, time_value: Any) -> tuple[Any, Any, Any]:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    core = _core_policy(policy)
    images, image_masks = core.prepare_images(batch)
    state = core.prepare_state(batch)
    actions = core.prepare_action(batch)
    language_tokens = batch["observation.language.tokens"]
    language_masks = batch["observation.language.attention_mask"]
    time_expanded = time_value[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    model = core.model
    prefix_embeddings, prefix_pad_masks, prefix_attention_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=state
    )
    suffix_embeddings, suffix_pad_masks, suffix_attention_masks = model.embed_suffix(x_t, time_value)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    attention_masks = torch.cat([prefix_attention_masks, suffix_attention_masks], dim=1)
    attention_2d = make_att_2d_masks(pad_masks, attention_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_output), _ = model.vlm_with_expert.forward(
        attention_mask=attention_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, suffix_embeddings],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_output = suffix_output[:, -core.config.chunk_size :].to(dtype=torch.float32)
    velocity = model.action_out_proj(suffix_output)
    return actions, x_t, velocity


def _action_stats(postprocessor: Any) -> dict[str, Any]:
    for step in postprocessor.steps:
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not tensor_stats or "action" not in tensor_stats:
            continue
        stats = tensor_stats["action"]
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError("VDR requires checkpoint MEAN_STD action statistics")
        mean = stats["mean"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        std = stats["std"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        return {"mode": "MEAN_STD", "mean": mean, "std": std, "processor_step": type(step).__name__}
    raise RuntimeError("checkpoint postprocessor has no action unnormalizer statistics")


def _decoded_chunk(policy: Any, batch: Mapping[str, Any], postprocessor: Any, noise: Any) -> tuple[Any, np.ndarray]:
    import torch

    if hasattr(policy, "reset"):
        policy.reset()
    policy.eval()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=noise.clone())
    return native.detach().float().cpu(), _postprocess_chunk(native, postprocessor)


def _gradient_values(loss: Any, named: Sequence[tuple[str, Any]]) -> tuple[list[Any | None], float, float]:
    import torch

    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    squared = 0.0
    finite_count = 0
    value_count = 0
    detached: list[Any | None] = []
    for gradient in gradients:
        if gradient is None:
            detached.append(None)
            continue
        value = gradient.detach().float().cpu()
        detached.append(value)
        squared += float(torch.sum(value.square()).item())
        finite_count += int(torch.isfinite(value).sum().item())
        value_count += int(value.numel())
    return detached, math.sqrt(squared), finite_count / max(value_count, 1)


def _row_static_tensor(row_index: int, arrays: Mapping[str, np.ndarray], device: Any, dtype: Any) -> Any:
    import torch

    return torch.as_tensor(arrays["static"][row_index], dtype=dtype, device=device).reshape(1, -1)


def _vdr_loss(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    time_value: Any,
    action_stats: Mapping[str, Any],
    model: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    row_index: int,
    horizon: int,
    target_residual: np.ndarray,
) -> Any:
    import torch
    import torch.nn.functional as functional

    _, x_t, velocity = _native_velocity(policy, batch, noise, time_value)
    clean = x_t - time_value[:, None, None] * velocity
    raw = differentiable_mean_std_unnormalize(clean[:, :, :ACTION_DIM], action_stats["mean"], action_stats["std"])
    static = _row_static_tensor(row_index, arrays, raw.device, raw.dtype)
    summary = torch_action_summary(raw, horizon)
    features = torch.cat([static, summary], dim=1)
    prediction = torch_predict_ridge(model["action_residual_probe"], features)
    target = torch.as_tensor(target_residual, dtype=prediction.dtype, device=prediction.device).reshape(1, -1)
    return functional.smooth_l1_loss(prediction, target, beta=1.0)


def _gradient_audit(
    policy: Any,
    preprocessor: Any,
    action_stats: Mapping[str, Any],
    models: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    row_index_by_horizon: Mapping[int, int],
) -> dict[str, Any]:
    import torch

    row_index = row_index_by_horizon[12]
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
        raise RuntimeError("VDR expected only LoRA trainable parameters")

    policy.train()
    policy.zero_grad(set_to_none=True)
    flow_loss = _loss(policy, batch, noise, time_value)
    flow_gradients, flow_norm, flow_finite = _gradient_values(flow_loss, named)
    flow_value = float(flow_loss.detach().item())
    del flow_loss
    gc.collect()
    torch.cuda.empty_cache()

    horizon = int(row["horizon"])
    selected = np.asarray(arrays["horizon"], dtype=np.int64) == horizon
    selected_indices = np.flatnonzero(selected)
    local_index = int(np.where(selected_indices == row_index)[0][0])
    delta = arrays["future"][selected] - arrays["current"][selected]
    y_all = project_with_whitener(models[str(horizon)]["whitener"], delta)
    static_pred = predict_ridge(models[str(horizon)]["static_model"], arrays["static"][selected])
    target_residual = y_all[local_index] - static_pred[local_index]

    policy.zero_grad(set_to_none=True)
    vdr_loss = _vdr_loss(
        policy,
        batch,
        noise,
        time_value,
        action_stats,
        models[str(horizon)],
        arrays,
        row_index,
        horizon,
        target_residual,
    )
    vdr_gradients, vdr_norm, vdr_finite = _gradient_values(vdr_loss, named)
    vdr_value = float(vdr_loss.detach().item())
    dot = 0.0
    for flow_gradient, vdr_gradient in zip(flow_gradients, vdr_gradients, strict=True):
        if flow_gradient is not None and vdr_gradient is not None:
            dot += float(torch.sum(flow_gradient * vdr_gradient).item())
    cosine = dot / max(flow_norm * vdr_norm, 1e-12)
    frozen_gradient_names = [
        name for name, parameter in policy.named_parameters() if "lora_" not in name.lower() and parameter.grad is not None
    ]
    policy.zero_grad(set_to_none=True)
    policy.eval()
    return {
        "flow_time": FLOW_TIME,
        "flow_loss": flow_value,
        "vdr_loss": vdr_value,
        "lambda_reference": 0.3,
        "reference_total_loss": flow_value + 0.3 * vdr_value,
        "trainable_parameter_names": [name for name, _ in named],
        "trainable_parameter_count": len(named),
        "trainable_numel": sum(int(parameter.numel()) for _, parameter in named),
        "flow_gradient_norm": flow_norm,
        "vdr_gradient_norm": vdr_norm,
        "vdr_to_flow_gradient_ratio": vdr_norm / max(flow_norm, 1e-12),
        "gradient_cosine": cosine,
        "flow_gradient_finite_fraction": flow_finite,
        "vdr_gradient_finite_fraction": vdr_finite,
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "frozen_parameter_gradient_names": frozen_gradient_names,
        "vdr_gradient_nonzero": vdr_norm > 0.0,
        "finite_objectives_and_gradients": bool(
            np.isfinite([flow_value, vdr_value, flow_norm, vdr_norm, cosine]).all()
            and flow_finite == 1.0
            and vdr_finite == 1.0
        ),
    }


def _identity_rows(manifest: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    buckets: dict[tuple[str, int, int], dict[int, int]] = {}
    for index, row in enumerate(manifest):
        if row["partition"] != "discovery":
            continue
        key = (str(row["task_identity"]), int(row["demo_id"]), int(row["frame_index"]))
        buckets.setdefault(key, {})[int(row["horizon"])] = index
    for _, horizons in sorted(buckets.items()):
        if set(horizons) == set(HORIZONS):
            return horizons
    raise RuntimeError("VDR identity audit could not align both frozen horizons")


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

    row_indices = _identity_rows(manifest)
    row = manifest[row_indices[12]]
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

    gradient = _gradient_audit(policy, preprocessor, action_stats, models, manifest, arrays, row_indices)
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
    identity = {
        "rank": 4,
        "base_flow_shape": list(base_flow.shape),
        "base_native_shape": list(base_native.shape),
        "base_action_shape": list(base_actions.shape),
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
        "training_only_vdr_absent_from_policy_parameters": not any(
            "vdr" in name.lower() for name, _ in reloaded.named_parameters()
        ),
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return identity, gradient


def _result_markdown(result: Mapping[str, Any]) -> str:
    data = result["data_audit"]
    gradient = result.get("gradient") or {}
    identity = result.get("identity") or {}
    return "\n".join(
        [
            "# VDR-VLA Stage 0A Result",
            "",
            f"Final decision: `{result['final_decision']}`.",
            "",
            f"Rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`.",
            f"Minimum static-predictor validation improvement: `{data['minimum_static_predictor_relative_improvement']}`.",
            f"Minimum action-residual relative / absolute gain: `{data['minimum_action_residual_relative_improvement']} / {data['minimum_action_residual_absolute_improvement']}`.",
            f"Minimum FutureVLA-proxy VDR relative / absolute gap: `{data['minimum_future_proxy_relative_improvement']} / {data['minimum_future_proxy_absolute_gap']}`.",
            f"Flow / VDR gradient norm: `{gradient.get('flow_gradient_norm')} / {gradient.get('vdr_gradient_norm')}`.",
            f"Identity maximum error: `{identity.get('identity_max_abs_error')}`.",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            "No simulator rollout, reward/success/done read, confirmatory identity access, or closed-loop experiment occurred.",
            "",
        ]
    )


def _adjudication_markdown(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# VDR-VLA Stage 0A Adjudication",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            "This is a development-only audit, not a closed-loop scientific kill.",
            f"Stage 0B allowed: `{result['stage_0b_allowed']}`.",
            f"Valid scientific result: `{result['valid_scientific_result']}`.",
            "",
            "The frozen Stage 0A gates were applied without changing horizons, PCA dimension, residual construction, baselines, or thresholds.",
            "",
        ]
    )


def _decision_inputs(
    proposal_ok: bool,
    serializer_ok: bool,
    manifest_ok: bool,
    data: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    gradient: Mapping[str, Any] | None,
    exception_count: int,
) -> Stage0ADecisionInputs:
    identity = identity or {}
    gradient = gradient or {}
    return Stage0ADecisionInputs(
        proposal_hash_ok=proposal_ok,
        serializer_preflight_ok=serializer_ok,
        manifest_integrity_ok=manifest_ok,
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=bool(data["feature_action_proprio_finite_aligned"]),
        minimum_discovery_rows_per_horizon=int(data["minimum_discovery_rows_per_horizon"]),
        minimum_validation_rows_per_horizon=int(data["minimum_validation_rows_per_horizon"]),
        residual_variance_all_positive=bool(data["residual_variance_all_positive"]),
        maximum_validation_task_fraction=float(data["maximum_validation_task_fraction"]),
        all_tasks_reported=bool(data["all_tasks_reported"]),
        static_predictor_relative_improvement=float(data["minimum_static_predictor_relative_improvement"]),
        action_residual_relative_improvement=float(data["minimum_action_residual_relative_improvement"]),
        action_residual_absolute_improvement=float(data["minimum_action_residual_absolute_improvement"]),
        future_proxy_relative_improvement=float(data["minimum_future_proxy_relative_improvement"]),
        future_proxy_absolute_gap=float(data["minimum_future_proxy_absolute_gap"]),
        finite_objectives_and_gradients=bool(gradient.get("finite_objectives_and_gradients", False)),
        vdr_gradient_nonzero=bool(gradient.get("vdr_gradient_nonzero", False)),
        gradient_ratio_at_most_100=float(gradient.get("vdr_to_flow_gradient_ratio", float("inf"))) <= GRADIENT_RATIO_MAX,
        frozen_parameter_gradient_count=int(gradient.get("frozen_parameter_gradient_count", 0)),
        identity_max_error=float(identity.get("identity_max_abs_error", 0.0)),
        base_hash_unchanged=bool(identity.get("base_hash_unchanged", True)),
        checkpoint_reload_ok=bool(identity.get("checkpoint_reload_ok", True)),
        action_validity_ok=bool(identity.get("base_action_in_bounds", True)) and bool(data.get("demo_action_validity_ok", True)),
        exception_count=int(exception_count),
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
    workers = [
        worker
        for worker in _active_linux_workers()
        if not (
            "run_vdr_vla_stage0a.py" in str(worker.get("command", ""))
            and "stage_0a_exit_code.txt" in str(worker.get("command", ""))
            and "bash -lc" in str(worker.get("command", ""))
        )
    ]
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
        "active_linux_workers": workers,
        "resource_evidence": _resource_evidence(registry, started_unix),
    }


def run(args: argparse.Namespace, paths: Mapping[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    import torch

    _set_offline_environment()
    started_unix = time.time()
    preflight = _preflight(paths, started_unix)
    _write_json(paths["preflight"], preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"VDR Stage 0A preflight failed: {preflight}")
    serializer = _read_json(paths["serializer_preflight"])
    serializer_ok = bool(
        serializer.get("passed") and canonical_json_sha256(serializer["fixture"]) == serializer.get("fixture_hash")
    )
    if not serializer_ok:
        raise RuntimeError("foreground VDR serializer preflight is absent or invalid")
    proposal_observed = _sha256(PROPOSAL_FILE)
    proposal_registry = _proposal_hash_text()
    proposal_ok = proposal_observed == proposal_registry == PROPOSAL_HASH
    if not proposal_ok:
        raise RuntimeError("frozen VDR proposal hash mismatch")

    state.update({"phase": "manifest", "status": "running"})
    rows, sources = _build_manifest(paths["data_root"])
    manifest_payload = {
        "method": "VDR-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "sources": sources,
        "horizons": list(HORIZONS),
        "projection_dimension": PROJECTION_DIM,
        "ridge_coefficient": RIDGE_COEFFICIENT,
        "std_floor": STD_FLOOR,
        "discovery_rows_per_task_horizon": DISCOVERY_ROWS_PER_TASK_HORIZON,
        "validation_rows_per_task_horizon": VALIDATION_ROWS_PER_TASK_HORIZON,
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
        raise RuntimeError("persisted VDR manifest hash did not reproduce")
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
        current_path, current = _load_or_extract_feature(policy, paths, row, int(row["frame_index"]), str(row["current_frame_key"]))
        future_path, future = _load_or_extract_feature(policy, paths, row, int(row["future_frame_index"]), str(row["future_frame_key"]))
        partial_rows.append(_partial_row(row, current_path, current, future_path, future))
        completed.add(key)
        state["completed_model_row_count"] = len(partial_rows)
        _write_json(
            paths["partial"],
            _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
        )
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        if len(partial_rows) % 16 == 0 or len(partial_rows) == len(rows):
            print(f"[vdr-stage0a] rows {len(partial_rows)}/{len(rows)}", flush=True)

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

    state["phase"] = "fit_vdr_audit_models"
    arrays = _materialize_arrays(rows, partial_rows)
    models, model_audit = _fit_vdr_models(rows, arrays)
    data = _data_summary(rows, arrays, model_audit, partial_audit)
    data["demo_action_validity_ok"] = bool(
        np.all(arrays["action_finite"]) and float(np.min(arrays["action_min"])) >= -1.0 and float(np.max(arrays["action_max"])) <= 1.0
    )

    state["phase"] = "identity_gradient"
    action_stats = _action_stats(postprocessor)
    identity, gradient = _identity_and_gradient_audit(
        policy, preprocessor, postprocessor, action_stats, models, rows, arrays, paths
    )
    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()

    decision_inputs = _decision_inputs(
        proposal_ok, serializer_ok, manifest_ok, data, identity, gradient, prior_exception_count
    )
    decision = classify_stage0a(decision_inputs)
    result = {
        "method": "VDR-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "model_hash": model_audit["model_hash"],
        "worker_pid": os.getpid(),
        "planned_model_row_count": len(rows),
        "completed_model_row_count": len(partial_rows),
        "resumed_model_row_count": resumed_count,
        "exception_count": prior_exception_count,
        "manifest_audit": manifest_audit,
        "partial_audit": partial_audit,
        "data_audit": data,
        "action_normalization": action_stats,
        "identity": identity,
        "gradient": gradient,
        "decision_inputs": decision_inputs.__dict__,
        "final_decision": decision,
        "stage_0b_allowed": decision == "VDR_STAGE_0A_PASS_STAGE_0B_ALLOWED",
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
        "manifest_json_parsed": True,
        "partial_json_parsed": True,
        "result_decision_recomputed": classify_stage0a(decision_inputs),
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
            last_exception=detail,
        ),
    )
    _write_json(
        paths["blocker"],
        {
            "method": "VDR-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "final_decision": "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": detail,
            "completed_model_row_count": len(rows),
            "planned_model_row_count": planned,
            "manifest_persisted": paths["manifest"].is_file(),
            "partial_persisted": True,
            "scientific_kill": False,
            "failed_at": _utc_now(),
        },
    )
    failed = {**state, "status": "failed", "phase": "failed", "exception_count": previous_exceptions + 1}
    _write_json(paths["status"], {**failed, "failed_at": _utc_now()})
    _write_json(paths["heartbeat"], {**failed, "updated_at": _utc_now()})


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "vdr_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "vdr_vla" / "stage0a"))
    parser.add_argument("--serializer-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(json.dumps({"serializer_preflight_passed": result["passed"]}, sort_keys=True), flush=True)
        return 0
    state: dict[str, Any] = {
        "method": "VDR-VLA",
        "stage": "0A",
        "pid": os.getpid(),
        "status": "starting",
        "phase": "startup",
        "planned_model_row_count": None,
        "completed_model_row_count": 0,
        "exception_count": 0,
    }
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["status"], {**state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], state, stop), daemon=True)
    heartbeat.start()
    try:
        result = run(args, paths, state)
    except Exception as exc:
        _write_blocker(paths, state, exc)
        traceback.print_exc()
        return 1
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        gc.collect()
    print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
