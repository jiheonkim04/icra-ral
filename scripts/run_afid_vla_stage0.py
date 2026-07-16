"""Run the frozen AFID-VLA Stage 0 action-factor development audit."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
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

from tca_map.smolvla.afid_vla import (  # noqa: E402
    ACTION_DIM,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    TAU_CONF,
    Stage0DecisionInputs,
    action_delta_summary,
    afid_row_key,
    apply_afid_gate,
    apply_finevla_proxy,
    binary_prediction_metrics,
    canonical_json_sha256,
    chunk_matrix,
    classify_stage0,
    clean_retention_summary,
    extract_action_factors,
    factor_keys,
    factor_label_health,
    factor_mask,
    fit_group_mean,
    fit_linear_factor_predictor,
    fit_residual_scale,
    gradient_smoke,
    group_clip,
    json_default,
    mask_health,
    mean_huber,
    predict_factor_confidence,
    predict_group_mean,
    relative_improvement,
    validate_manifest,
)


POLICY_PROBE = "afid_stage0_action_factor_densification"
FACTOR_KEY = "all_action_factors"
SEED = 20263300
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "afid_vla" / "proposal_hash.txt"
DEFAULT_CCIF_MANIFEST = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_manifest.json"
DEFAULT_CCIF_PARTIAL = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_partial.json"
DEFAULT_CCIF_ACTION_SEMANTICS = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_action_semantics.json"
ALLOWED_TASKS = {
    "libero_spatial/task_3",
    "libero_object/task_3",
    "libero_goal/task_5",
    "libero_10/task_5",
}


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    if not report.is_absolute():
        report = REPO_ROOT / report
    if not run.is_absolute():
        run = REPO_ROOT / run
    return {
        "report": report,
        "run": run,
        "identity_dir": run / "identity_gate",
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
        "adjudication": report / "stage_0_adjudication.md",
        "blocker": report / "stage_0_implementation_blocker.json",
        "exit_code": report / "stage_0_exit_code.txt",
        "checkpoint": Path(args.checkpoint) if args.checkpoint else Path(""),
        "data_root": Path(args.data_root) if args.data_root else Path(""),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_path(path: str | Path) -> Path:
    value = str(path)
    if os.name == "nt" and value.startswith("/mnt/c/"):
        return Path("C:/" + value[len("/mnt/c/") :])
    return Path(value)


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


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_local_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _local_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _array_sha256(value: Any) -> str:
    array = np.asarray(value, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _read_npz_array(path: str | Path, preferred_key: str) -> np.ndarray:
    with np.load(_local_path(path), allow_pickle=False) as payload:
        if preferred_key in payload.files:
            return np.asarray(payload[preferred_key])
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]])
        raise ValueError(f"{path} does not contain key {preferred_key}")


def _proposal_hash_text() -> str:
    if not PROPOSAL_HASH_FILE.is_file():
        return ""
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _serializer_preflight(path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    base = rng.normal(scale=0.01, size=(8, HORIZON, ACTION_DIM)).astype(np.float32)
    expert = base.copy()
    expert[:, :, 0] += 0.04
    expert[:, :, 6] -= 0.22
    residual = expert - base
    scale = fit_residual_scale(residual)
    mask = factor_mask(residual, scale)
    labels = extract_action_factors(base, expert)
    predictor = fit_linear_factor_predictor(base, mask)
    confidence = predict_factor_confidence(predictor, base)
    changed, gate = apply_afid_gate(base, residual, mask, confidence)
    identity, _ = apply_afid_gate(base, residual, np.zeros_like(mask), np.zeros_like(confidence))
    manifest_row = {
        "split": "validation",
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": 8,
        "window_start": 20,
        "factor_key": FACTOR_KEY,
        "policy": "afid_full",
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = afid_row_key(manifest_row)
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        factor_labels_noncollapsed=True,
        usable_factor_count=2,
        factor_mask_global_positive_fraction=0.25,
        validation_task_mask_fraction_min=0.25,
        validation_task_mask_fraction_max=0.25,
        factor_predictor_beats_majority=True,
        factor_predictor_beats_task_phase=True,
        factor_conditioned_oracle_reduction=0.05,
        finevla_proxy_residual_headroom=0.03,
        afid_differs_from_base=True,
        afid_differs_from_finevla_proxy=True,
        afid_differs_from_no_factor=True,
        afid_differs_from_standard_lora=True,
        identity_max_abs_error=float(np.max(np.abs(identity - base))),
        inactive_gate_max_abs_error=0.0,
        finite_objectives_and_gradients=True,
        expected_parameter_gradient_nonzero=True,
        frozen_base_gradient_count=0,
        weighted_gradient_norm_ratio_max=1.0,
        gate_activation_fraction=0.25,
        action_deltas_bounded=True,
        action_validity_ok=True,
        clean_retention_ok=True,
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=0,
    )
    fixture: dict[str, Any] = {
        "method": "AFID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "residual_scale": scale,
        "factor_labels": labels,
        "factor_label_health": factor_label_health(labels),
        "factor_mask": mask,
        "confidence": confidence,
        "gate": gate,
        "changed_chunk": changed,
        "action_delta_summary": action_delta_summary(base, changed),
        "gradient": gradient_smoke(base, residual, gate, expert),
        "path_value": path,
        "decision_inputs": healthy,
        "decision": classify_stage0(healthy),
    }
    fixture_hash = canonical_json_sha256(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixture": fixture, "fixture_hash": fixture_hash}, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        "method": "AFID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(reproduced == fixture_hash and fixture["decision"] == "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"),
        "fixture_hash": fixture_hash,
        "reproduced_hash": reproduced,
        "fixture": parsed["fixture"],
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("AFID serializer preflight hash did not reproduce")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = {
        "official_repository": REPO_ROOT / "third_party" / "FineVLA",
        "alternate_repository": REPO_ROOT / "third_party" / "finevla",
        "checkpoint_dir": REPO_ROOT / "third_party" / "FineVLA" / "checkpoints",
    }
    exists = {name: candidate.exists() for name, candidate in candidates.items()}
    official_ready = bool((exists["official_repository"] or exists["alternate_repository"]) and exists["checkpoint_dir"])
    result = {
        "method": "AFID-VLA",
        "stage": "0",
        "closest_prior": "FineVLA",
        "closest_prior_primary_source": "https://arxiv.org/html/2605.27284v1",
        "policy_2_label": "official_finevla" if official_ready else "finevla_action_factor_proxy",
        "official_ready": official_ready,
        "asset_exists": exists,
        "checked_paths": {name: str(candidate) for name, candidate in candidates.items()},
        "proxy_deviations": []
        if official_ready
        else [
            "official FineVLA repository/checkpoint assets are not locally verified",
            "Stage 0 fixes policy 2 as a transparent action-factor proxy under the frozen SmolVLA backbone",
        ],
        "comparison_position": 2,
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    if DEFAULT_CCIF_ACTION_SEMANTICS.is_file():
        source = _read_json(DEFAULT_CCIF_ACTION_SEMANTICS)
        semantics = dict(source)
        semantics.update(
            {
                "method": "AFID-VLA",
                "same_definition_applies_to_policies": list(POLICY_ROWS),
                "source_semantics": str(DEFAULT_CCIF_ACTION_SEMANTICS),
                "created_utc": _utc_now(),
            }
        )
    else:
        semantics = {
            "method": "AFID-VLA",
            "model_native_action_shape": [HORIZON, ACTION_DIM],
            "environment_action_shape": [ACTION_DIM],
            "postprocessor_or_unnormalizer_class": "cached SmolVLA Base chunks with existing official postprocessor",
            "environment_action_space_low_high_exposed": False,
            "environment_action_space_low": None,
            "environment_action_space_high": None,
            "gripper_convention": "LIBERO/SmolVLA checkpoint action dimension 6 after postprocessor",
            "finite_checks_required": True,
            "action_bound_validity_rule": "not_used_without_official_environment_bounds",
            "final_action_validity_definition": "valid iff action chunk has shape [50,7], finite entries, and the same cached SmolVLA action semantics are used for every policy",
            "same_definition_applies_to_policies": list(POLICY_ROWS),
            "created_utc": _utc_now(),
        }
    _write_json(path, semantics)
    return semantics


def _source_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        int(row["demo_id"]),
        int(row["frame_index"]),
    )


def _action_chunk(actions: Any, frame_index: int) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    frame = int(frame_index)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTION_DIM}], got {value.shape}")
    if frame < 0 or frame + HORIZON > len(value):
        raise ValueError("frame must have a complete action chunk")
    chunk = value[frame : frame + HORIZON]
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains nonfinite values")
    return chunk


def _expert_chunk(source_path: str, demo_id: int, frame_index: int) -> np.ndarray:
    import h5py

    with h5py.File(str(_local_path(source_path)), "r") as handle:
        demo = handle["data"][f"demo_{int(demo_id)}"]
        return _action_chunk(np.asarray(demo["actions"], dtype=np.float64), frame_index)


def _load_base_records(
    ccif_manifest_path: Path,
    ccif_partial_path: Path,
    *,
    max_sources: int | None = None,
) -> list[dict[str, Any]]:
    ccif_manifest = _read_json(ccif_manifest_path)
    ccif_partial = _read_json(ccif_partial_path)
    manifest_rows = [
        row
        for row in ccif_manifest.get("rows", [])
        if row.get("model_or_probe") == "smolvla_base" and row.get("task_identity") in ALLOWED_TASKS
    ]
    partial_by_source = {
        _source_key(row): row for row in ccif_partial.get("rows", []) if row.get("model_or_probe") == "smolvla_base"
    }
    records: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        key = _source_key(manifest)
        partial = partial_by_source.get(key)
        if partial is None:
            continue
        base_cache = _local_path(str(partial["base_chunk_cache_path"]))
        if not base_cache.is_file():
            continue
        if _sha256(base_cache) != str(partial["base_chunk_cache_sha256"]).upper():
            raise RuntimeError(f"base cache hash mismatch for {manifest['row_key']}")
        base = np.asarray(_read_npz_array(base_cache, "base_chunk"), dtype=np.float64)
        base = chunk_matrix(base, "base_chunk")[0]
        expert = _expert_chunk(str(manifest["source_path"]), int(manifest["demo_id"]), int(manifest["frame_index"]))
        if _array_sha256(expert) != str(partial["action_chunk_sha256"]).upper():
            raise RuntimeError(f"demo action hash mismatch for {manifest['row_key']}")
        records.append(
            {
                "split": str(manifest["partition"]),
                "task_suite": str(manifest["suite"]),
                "task_id": str(manifest["task_identity"]),
                "source_edge_sha256": str(manifest["source_edge_sha256"]),
                "source_path": str(manifest["source_path"]),
                "task_language": str(manifest.get("task_language", "")),
                "demo_id": int(manifest["demo_id"]),
                "window_start": int(manifest["frame_index"]),
                "phase": float(manifest.get("phase", 0.0)),
                "phase_bin": int(manifest.get("phase_bin", 0)),
                "episode_length": int(manifest.get("episode_length", HORIZON)),
                "base_chunk": base,
                "expert_chunk": expert,
                "base_chunk_cache_path": str(partial["base_chunk_cache_path"]),
                "base_chunk_cache_sha256": str(partial["base_chunk_cache_sha256"]).upper(),
                "base_chunk_sha256": str(partial["base_chunk_sha256"]).upper(),
                "action_chunk_sha256": str(partial["action_chunk_sha256"]).upper(),
            }
        )
        if max_sources is not None and len(records) >= int(max_sources):
            break
    return records


def _phase_key(record: Mapping[str, Any]) -> str:
    return f"phase:{int(record['phase_bin'])}"


def _task_phase_key(record: Mapping[str, Any]) -> str:
    return f"{record['task_id']}|phase:{int(record['phase_bin'])}"


def _manifest_rows(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        factor_key = keys[index] if index < len(keys) else FACTOR_KEY
        for policy in POLICY_ROWS:
            row = {
                "split": record["split"],
                "task_suite": record["task_suite"],
                "task_id": record["task_id"],
                "demo_id": record["demo_id"],
                "window_start": record["window_start"],
                "factor_key": factor_key,
                "policy": policy,
                "policy_probe": POLICY_PROBE,
                "record_index": index,
                "source_edge_sha256": record["source_edge_sha256"],
                "phase_bin": record["phase_bin"],
                "task_language": record["task_language"],
            }
            row["row_key"] = afid_row_key(row)
            rows.append(row)
    return rows


def _task_phase_mask_baseline(records: Sequence[Mapping[str, Any]], train: np.ndarray, target: np.ndarray) -> np.ndarray:
    flat_target = target.reshape(len(target), -1)
    train_keys = [_task_phase_key(record) for index, record in enumerate(records) if train[index]]
    all_keys = [_task_phase_key(record) for record in records]
    default = float(np.mean(flat_target[train])) >= 0.5
    group_values: dict[str, bool] = {}
    for key in sorted(set(train_keys)):
        indexes = [index for index, value in enumerate(train_keys) if value == key]
        group_values[key] = bool(np.mean(flat_target[train][indexes]) >= 0.5)
    return np.asarray([np.full(target.shape[1:], group_values.get(key, default), dtype=bool) for key in all_keys])


def _materialize_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = np.asarray([record["base_chunk"] for record in records], dtype=np.float64)
    expert = np.asarray([record["expert_chunk"] for record in records], dtype=np.float64)
    discovery = np.asarray([record["split"] == "discovery" for record in records], dtype=bool)
    validation = np.asarray([record["split"] == "validation" for record in records], dtype=bool)
    if not discovery.any() or not validation.any():
        raise ValueError("AFID Stage 0 requires both discovery and validation rows")
    residual = expert - base
    residual_scale = fit_residual_scale(residual[discovery])
    true_mask = factor_mask(residual, residual_scale)
    labels = extract_action_factors(base, expert)
    keys = factor_keys(labels)
    predictor = fit_linear_factor_predictor(base[discovery], true_mask[discovery])
    confidence = predict_factor_confidence(predictor, base)
    predicted_mask = (confidence >= TAU_CONF).astype(np.float64)
    group_model = fit_group_mean(residual[discovery], [keys[index] for index, keep in enumerate(discovery) if keep])
    factor_residual = predict_group_mean(group_model, keys)
    phase_model = fit_group_mean(residual[discovery], [_task_phase_key(record) for index, record in enumerate(records) if discovery[index]])
    task_phase_residual = predict_group_mean(phase_model, [_task_phase_key(record) for record in records])
    standard_model = fit_group_mean(residual[discovery], ["global" for _ in range(int(np.sum(discovery)))])
    standard_residual = predict_group_mean(standard_model, ["global" for _ in records])
    finevla = apply_finevla_proxy(base, factor_residual, confidence)
    afid, gate = apply_afid_gate(base, factor_residual, predicted_mask, confidence)
    no_factor = apply_finevla_proxy(base, task_phase_residual, confidence)
    standard = base + group_clip(standard_residual)
    oracle = base + true_mask * group_clip(residual)
    task_phase = base + group_clip(task_phase_residual)
    mask_only = base + true_mask * group_clip(factor_residual)
    predictions = {
        "smolvla_base": base,
        "finevla_action_factor_proxy": finevla,
        "afid_full": afid,
        "afid_no_factor_ablation": no_factor,
        "standard_lora": standard,
        "factor_conditioned_oracle_diagnostic": oracle,
        "task_phase_residual_diagnostic": task_phase,
        "mask_only_residual_diagnostic": mask_only,
    }
    return {
        "base": base,
        "expert": expert,
        "residual": residual,
        "discovery": discovery,
        "validation": validation,
        "residual_scale": residual_scale,
        "true_mask": true_mask,
        "predicted_mask": predicted_mask,
        "labels": labels,
        "factor_keys": keys,
        "predictor": predictor,
        "confidence": confidence,
        "gate": gate,
        "predictions": predictions,
        "task_phase_mask_baseline": _task_phase_mask_baseline(records, discovery, true_mask),
    }


def _partial_rows(
    records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    completed: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions = arrays["predictions"]
    true_mask = np.asarray(arrays["true_mask"], dtype=np.float64)
    predicted_mask = np.asarray(arrays["predicted_mask"], dtype=np.float64)
    confidence = np.asarray(arrays["confidence"], dtype=np.float64)
    gate = np.asarray(arrays["gate"], dtype=np.float64)
    for manifest in manifest_rows:
        key = str(manifest["row_key"])
        if key in completed:
            continue
        index = int(manifest["record_index"])
        policy = str(manifest["policy"])
        prediction = np.asarray(predictions[policy][index], dtype=np.float64)
        base = np.asarray(arrays["base"][index], dtype=np.float64)
        expert = np.asarray(arrays["expert"][index], dtype=np.float64)
        delta = prediction - base
        rows.append(
            {
                "row_key": key,
                "split": manifest["split"],
                "task_suite": manifest["task_suite"],
                "task_id": manifest["task_id"],
                "demo_id": int(manifest["demo_id"]),
                "window_start": int(manifest["window_start"]),
                "factor_key": manifest["factor_key"],
                "policy": policy,
                "policy_probe": POLICY_PROBE,
                "prediction_finite": bool(np.isfinite(prediction).all()),
                "prediction_shape": list(prediction.shape),
                "target_huber": mean_huber(prediction.reshape(1, HORIZON, ACTION_DIM), expert.reshape(1, HORIZON, ACTION_DIM)),
                "delta_abs_max": float(np.max(np.abs(delta))),
                "mask_positive_fraction": float(np.mean(true_mask[index] > 0.5)),
                "predicted_mask_positive_fraction": float(np.mean(predicted_mask[index] > 0.5)),
                "confidence_mean": float(np.mean(confidence[index])),
                "gate_activation_fraction": float(np.mean(gate[index] > 0.0)),
            }
        )
    return rows


def _partial_payload(
    manifest_hash: str,
    planned_rows: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "AFID-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": int(planned_rows),
        "completed_model_row_count": len(rows),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "rows": list(rows),
        "updated_utc": _utc_now(),
    }


def _load_resume(partial_path: Path, manifest_rows: Sequence[Mapping[str, Any]], manifest_hash: str) -> tuple[list[dict[str, Any]], int, str | None]:
    if not partial_path.is_file():
        return [], 0, None
    partial = _read_json(partial_path)
    if partial.get("manifest_hash") != manifest_hash:
        return [], 0, None
    valid_keys = {str(row["row_key"]) for row in manifest_rows}
    rows = [dict(row) for row in partial.get("rows", []) if str(row.get("row_key")) in valid_keys]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["row_key"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique, int(partial.get("exception_count", 0)), partial.get("last_exception")


def _write_identity_checkpoint(path: Path, base: np.ndarray, residual: np.ndarray, mask: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    identity, _ = apply_afid_gate(base, residual, np.zeros_like(mask), np.zeros_like(confidence))
    checkpoint = path / "identity_fixture.npz"
    np.savez_compressed(checkpoint, base=base.astype(np.float32), identity=identity.astype(np.float32))
    with np.load(checkpoint, allow_pickle=False) as payload:
        reloaded_identity = np.asarray(payload["identity"], dtype=np.float64)
    error = float(np.max(np.abs(reloaded_identity - base)))
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_reload_ok": bool(error <= 1e-6),
        "reloaded_identity_max_abs_error": error,
    }


def _result_from_rows(
    records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    validation = np.asarray(arrays["validation"], dtype=bool)
    target = arrays["expert"][validation]
    huber_by_policy = {
        policy: mean_huber(np.asarray(arrays["predictions"][policy])[validation], target) for policy in POLICY_ROWS
    }
    base_score = huber_by_policy["smolvla_base"]
    finevla_score = huber_by_policy["finevla_action_factor_proxy"]
    afid_score = huber_by_policy["afid_full"]
    no_factor_score = huber_by_policy["afid_no_factor_ablation"]
    standard_score = huber_by_policy["standard_lora"]
    oracle_score = huber_by_policy["factor_conditioned_oracle_diagnostic"]
    labels_health = factor_label_health(arrays["labels"])
    validation_tasks = [str(record["task_id"]) for index, record in enumerate(records) if validation[index]]
    validation_mask_health = mask_health(np.asarray(arrays["true_mask"])[validation], task_ids=validation_tasks)
    predictor_metrics = binary_prediction_metrics(
        np.asarray(arrays["predicted_mask"])[validation] > 0.5,
        np.asarray(arrays["true_mask"])[validation] > 0.5,
    )
    task_phase_metrics = binary_prediction_metrics(
        np.asarray(arrays["task_phase_mask_baseline"])[validation],
        np.asarray(arrays["true_mask"])[validation] > 0.5,
    )
    clean = clean_retention_summary(
        arrays["base"],
        apply_afid_gate(arrays["base"], arrays["residual"], np.zeros_like(arrays["true_mask"]), np.zeros_like(arrays["confidence"]))[0],
        apply_afid_gate(arrays["base"], arrays["residual"], np.zeros_like(arrays["true_mask"]), np.ones_like(arrays["confidence"]))[0],
    )
    identity = _write_identity_checkpoint(paths["identity_dir"], arrays["base"], arrays["residual"], arrays["true_mask"], arrays["confidence"])
    gradient = gradient_smoke(arrays["base"], arrays["predictions"]["afid_full"] - arrays["base"], arrays["gate"], arrays["expert"])
    delta = action_delta_summary(arrays["base"], arrays["predictions"]["afid_full"])
    validation_records = [record for index, record in enumerate(records) if validation[index]]
    validation_counts = Counter(str(record["task_id"]) for record in validation_records)
    total_validation = sum(validation_counts.values())
    split_counts = Counter(str(record["split"]) for record in records)
    task_counts = Counter(str(record["task_id"]) for record in records)
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=_proposal_hash_text() == PROPOSAL_HASH,
        serializer_preflight_ok=bool(_read_json(paths["serializer_preflight"]).get("passed", False)),
        official_prior_asset_check_persisted=paths["official_prior_asset_check"].is_file(),
        manifest_integrity_ok=bool(
            manifest_summary["duplicate_manifest_key_count"] == 0
            and manifest_summary["duplicate_partial_key_count"] == 0
            and manifest_summary["missing_manifest_key_count"] == 0
            and manifest_summary["extra_partial_key_count"] == 0
            and manifest_summary["split_overlap_key_count"] == 0
            and manifest_summary["key_sets_equal"]
        ),
        source_alignment_ok=True,
        action_semantics_ok=paths["action_semantics"].is_file(),
        base_chunks_valid=True,
        factor_labels_noncollapsed=bool(labels_health["usable_factor_count"] > 0),
        usable_factor_count=int(labels_health["usable_factor_count"]),
        factor_mask_global_positive_fraction=float(validation_mask_health["factor_mask_global_positive_fraction"]),
        validation_task_mask_fraction_min=float(validation_mask_health.get("validation_task_mask_fraction_min", 0.0)),
        validation_task_mask_fraction_max=float(validation_mask_health.get("validation_task_mask_fraction_max", 1.0)),
        factor_predictor_beats_majority=bool(
            max(
                predictor_metrics["accuracy_improvement_over_majority"],
                predictor_metrics["macro_f1_improvement_over_majority"],
            )
            >= 0.05
        ),
        factor_predictor_beats_task_phase=bool(
            predictor_metrics["accuracy"] >= task_phase_metrics["accuracy"] + 0.05
            or predictor_metrics["macro_f1"] >= task_phase_metrics["macro_f1"] + 0.05
        ),
        factor_conditioned_oracle_reduction=relative_improvement(base_score, oracle_score),
        finevla_proxy_residual_headroom=relative_improvement(finevla_score, oracle_score),
        afid_differs_from_base=bool(np.mean(np.abs(arrays["predictions"]["afid_full"] - arrays["base"])) > 1e-12),
        afid_differs_from_finevla_proxy=bool(
            np.mean(np.abs(arrays["predictions"]["afid_full"] - arrays["predictions"]["finevla_action_factor_proxy"])) > 1e-12
        ),
        afid_differs_from_no_factor=bool(
            np.mean(np.abs(arrays["predictions"]["afid_full"] - arrays["predictions"]["afid_no_factor_ablation"])) > 1e-12
        ),
        afid_differs_from_standard_lora=bool(
            np.mean(np.abs(arrays["predictions"]["afid_full"] - arrays["predictions"]["standard_lora"])) > 1e-12
        ),
        identity_max_abs_error=max(float(clean["identity_max_abs_error"]), float(identity["reloaded_identity_max_abs_error"])),
        inactive_gate_max_abs_error=float(clean["inactive_gate_max_abs_error"]),
        finite_objectives_and_gradients=bool(gradient["finite_objectives_and_gradients"]),
        expected_parameter_gradient_nonzero=bool(gradient["expected_parameter_gradient_nonzero"]),
        frozen_base_gradient_count=int(gradient["frozen_base_gradient_count"]),
        weighted_gradient_norm_ratio_max=float(gradient["weighted_gradient_norm_ratio_max"]),
        gate_activation_fraction=float(np.mean(arrays["gate"] > 0.0)),
        action_deltas_bounded=bool(delta["action_deltas_bounded"]),
        action_validity_ok=bool(all(row.get("prediction_finite") for row in partial_rows)),
        clean_retention_ok=bool(clean["clean_retention_ok"] and identity["checkpoint_reload_ok"]),
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=0,
    )
    final_decision = classify_stage0(decision_inputs)
    return {
        "method": "AFID-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "final_decision": final_decision,
        "completed_model_row_count": len(partial_rows),
        "planned_model_row_count": len(manifest_rows),
        "exception_count": 0,
        **manifest_summary,
        "proposal_hash_ok": decision_inputs.proposal_hash_ok,
        "serializer_preflight_ok": decision_inputs.serializer_preflight_ok,
        "preflight_passed": True,
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "confirmatory_records_read": 0,
        "training_happened": False,
        "validation_search_happened": False,
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "validation_task_counts": dict(validation_counts),
        "maximum_validation_task_fraction": max((count / total_validation for count in validation_counts.values()), default=1.0),
        "factor_label_health": labels_health,
        "mask_health": validation_mask_health,
        "factor_predictor_metrics": predictor_metrics,
        "task_phase_mask_baseline_metrics": task_phase_metrics,
        "factor_conditioned_oracle_reduction": decision_inputs.factor_conditioned_oracle_reduction,
        "finevla_proxy_residual_headroom": decision_inputs.finevla_proxy_residual_headroom,
        "gate_activation_fraction": decision_inputs.gate_activation_fraction,
        "identity_max_abs_error": decision_inputs.identity_max_abs_error,
        "inactive_gate_max_abs_error": decision_inputs.inactive_gate_max_abs_error,
        "expected_parameter_gradient_nonzero": decision_inputs.expected_parameter_gradient_nonzero,
        "frozen_base_gradient_count": decision_inputs.frozen_base_gradient_count,
        "weighted_gradient_norm_ratio_max": decision_inputs.weighted_gradient_norm_ratio_max,
        "action_deltas_bounded": decision_inputs.action_deltas_bounded,
        "action_validity_ok": decision_inputs.action_validity_ok,
        "clean_retention_ok": decision_inputs.clean_retention_ok,
        "afid_differs_from_base": decision_inputs.afid_differs_from_base,
        "afid_differs_from_finevla_proxy": decision_inputs.afid_differs_from_finevla_proxy,
        "afid_differs_from_no_factor": decision_inputs.afid_differs_from_no_factor,
        "afid_differs_from_standard_lora": decision_inputs.afid_differs_from_standard_lora,
        "huber_by_policy": huber_by_policy,
        "residual_scale": arrays["residual_scale"],
        "identity": identity,
        "gradient": gradient,
        "decision_inputs": asdict(decision_inputs),
        "stage_0_is_closed_loop_scientific_kill": False,
        "valid_scientific_result": False,
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "created_utc": _utc_now(),
    }


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# AFID-VLA Stage 0 Result",
            "",
            f"Decision: `{result['final_decision']}`",
            "",
            f"Rows: `{result['completed_model_row_count']}/{result['planned_model_row_count']}`",
            f"Factor predictor accuracy: `{result['factor_predictor_metrics']['accuracy']}`",
            f"Factor-conditioned oracle reduction: `{result['factor_conditioned_oracle_reduction']}`",
            f"FineVLA residual headroom: `{result['finevla_proxy_residual_headroom']}`",
            f"Gate activation: `{result['gate_activation_fraction']}`",
            f"Exceptions: `{result['exception_count']}`",
            f"Duplicate partial keys: `{result['duplicate_partial_key_count']}`",
            f"Missing manifest keys: `{result['missing_manifest_key_count']}`",
            "",
            "This is a development-only Stage 0 audit, not a closed-loop scientific result.",
            "Timing and resource-use evidence is not paper-eligible.",
            "",
        ]
    )
    _write_text(path, text)


def _write_adjudication(path: Path, result: Mapping[str, Any]) -> None:
    if result["final_decision"] == "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        next_step = "Proceed to frozen bounded validation search with at most six configurations."
    else:
        next_step = "Archive this Stage 0 development stop class and continue without rescue unless a pre-result implementation blocker is identified."
    text = "\n".join(
        [
            "# AFID-VLA Stage 0 Adjudication",
            "",
            f"Decision: `{result['final_decision']}`",
            "",
            next_step,
            "",
            "No confirmatory-test records, simulator rollouts, reward rows, success flags, or done flags were read.",
            "",
        ]
    )
    _write_text(path, text)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _existing_worker_summary(paths: Mapping[str, Path]) -> dict[str, Any]:
    pid = None
    if paths["pid"].is_file():
        text = paths["pid"].read_text(encoding="utf-8").strip()
        if text.isdigit():
            pid = int(text)
    status = _read_json(paths["status"]) if paths["status"].is_file() else None
    heartbeat = _read_json(paths["heartbeat"]) if paths["heartbeat"].is_file() else None
    partial_summary = None
    partial_parse_error = None
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
    return {
        "pid": pid,
        "pid_alive": _pid_alive(pid or -1),
        "status": status,
        "heartbeat": heartbeat,
        "partial_summary": partial_summary,
        "partial_parse_error": partial_parse_error,
        "result_exists": paths["result_json"].is_file(),
        "exit_code_exists": paths["exit_code"].is_file(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    existing = _existing_worker_summary(paths)
    if paths["result_json"].is_file() and not args.force:
        result = _read_json(paths["result_json"])
        _write_json(
            paths["status"],
            {
                "method": "AFID-VLA",
                "status": "completed_existing_result_reused",
                "final_decision": result.get("final_decision"),
                "updated_utc": _utc_now(),
            },
        )
        return result
    if existing["pid_alive"] and not args.force:
        raise RuntimeError(f"existing AFID Stage 0 worker is alive: {existing}")
    if existing["partial_parse_error"]:
        raise RuntimeError(f"existing AFID partial does not parse: {existing['partial_parse_error']}")

    state: dict[str, Any] = {
        "method": "AFID-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "pid": os.getpid(),
        "status": "running",
        "phase": "startup",
        "started_utc": _utc_now(),
    }
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["status"], state)
    _write_json(paths["heartbeat"], {**state, "updated_utc": _utc_now()})
    try:
        serializer = _serializer_preflight(paths["serializer_preflight"])
        prior = _official_prior_asset_check(paths["official_prior_asset_check"])
        action_semantics = _write_action_semantics(paths["action_semantics"])
        preflight = {
            "method": "AFID-VLA",
            "proposal_hash_ok": _proposal_hash_text() == PROPOSAL_HASH,
            "serializer_preflight_ok": bool(serializer["passed"]),
            "official_prior_asset_check_persisted": bool(prior),
            "action_semantics_persisted": bool(action_semantics),
            "checkpoint_argument": str(paths["checkpoint"]),
            "data_root_argument": str(paths["data_root"]),
            "ccif_manifest": str(args.ccif_manifest),
            "ccif_partial": str(args.ccif_partial),
            "cached_base_source": "verified CCIF Stage 0 Base chunks",
            "no_confirmatory_records_read": True,
            "existing_worker_summary": existing,
            "created_utc": _utc_now(),
        }
        _write_json(paths["preflight"], preflight)
        if not preflight["proposal_hash_ok"]:
            raise RuntimeError("frozen AFID proposal hash mismatch")
        records = _load_base_records(Path(args.ccif_manifest), Path(args.ccif_partial), max_sources=args.max_sources)
        arrays = _materialize_arrays(records)
        manifest_rows = _manifest_rows(records, arrays["factor_keys"])
        manifest_payload = {
            "method": "AFID-VLA",
            "stage": "0",
            "policy_probe": POLICY_PROBE,
            "proposal_hash": PROPOSAL_HASH,
            "policy_rows": list(POLICY_ROWS),
            "planned_model_row_count": len(manifest_rows),
            "unique_observation_row_count": len(records),
            "rows": manifest_rows,
            "frozen_data": {
                "development_tasks": sorted(ALLOWED_TASKS),
                "discovery_demos": "0..7",
                "validation_demos": "8..9",
                "confirmatory_identities_read": 0,
            },
            "created_utc": _utc_now(),
        }
        manifest_hash = canonical_json_sha256({key: value for key, value in manifest_payload.items() if key != "created_utc"})
        manifest_payload["manifest_hash"] = manifest_hash
        _write_json(paths["manifest"], manifest_payload)
        partial_rows, previous_exception_count, previous_last_exception = _load_resume(paths["partial"], manifest_rows, manifest_hash)
        completed = {str(row["row_key"]) for row in partial_rows}
        partial_rows.extend(_partial_rows(records, manifest_rows, arrays, completed))
        ordered = {str(row["row_key"]): row for row in partial_rows}
        partial_rows = [ordered[str(row["row_key"])] for row in manifest_rows]
        _write_json(
            paths["partial"],
            _partial_payload(
                manifest_hash,
                len(manifest_rows),
                partial_rows,
                exception_count=previous_exception_count,
                last_exception=previous_last_exception,
            ),
        )
        manifest_summary = validate_manifest(manifest_rows, partial_rows)
        result = _result_from_rows(records, manifest_rows, partial_rows, arrays, manifest_summary, paths)
        result["manifest_hash"] = manifest_hash
        result["resume_exception_count"] = previous_exception_count
        result["resume_last_exception"] = previous_last_exception
        result["official_prior_asset_check"] = prior
        result["action_semantics"] = action_semantics
        _write_json(paths["result_json"], result)
        _write_result_markdown(paths["result_md"], result)
        _write_adjudication(paths["adjudication"], result)
        elapsed = time.time() - started
        _write_json(
            paths["status"],
            {
                **state,
                "status": "completed",
                "phase": "complete",
                "elapsed_seconds": elapsed,
                "final_decision": result["final_decision"],
                "completed_model_row_count": result["completed_model_row_count"],
                "planned_model_row_count": result["planned_model_row_count"],
                "updated_utc": _utc_now(),
            },
        )
        _write_json(
            paths["heartbeat"],
            {
                **state,
                "status": "completed",
                "phase": "complete",
                "final_decision": result["final_decision"],
                "updated_utc": _utc_now(),
            },
        )
        _write_text(paths["exit_code"], "0\n")
        return result
    except BaseException as exc:
        blocker = {
            "method": "AFID-VLA",
            "stage": "0",
            "proposal_hash": PROPOSAL_HASH,
            "status": "implementation_blocker",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "created_utc": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        _write_json(paths["status"], {**state, "status": "failed", "phase": "implementation_blocker", "updated_utc": _utc_now()})
        _write_json(paths["heartbeat"], {**state, "status": "failed", "phase": "implementation_blocker", "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "1\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "afid_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "afid_vla" / "stage0"))
    parser.add_argument("--ccif-manifest", default=str(DEFAULT_CCIF_MANIFEST))
    parser.add_argument("--ccif-partial", default=str(DEFAULT_CCIF_PARTIAL))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"AFID serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0
    result = run(args)
    print(json.dumps({"final_decision": result.get("final_decision"), "result": str(paths["result_json"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
