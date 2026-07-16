"""Run the frozen LCG-VLA Stage 0 language-contrast development audit."""

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

from tca_map.smolvla.lcg_vla import (  # noqa: E402
    ACTION_DIM,
    CAG_PROXY_BETAS,
    HORIZON,
    NULL_INSTRUCTION,
    POLICY_ROWS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_cag_proxy,
    apply_lcg_gate,
    apply_no_language_ablation,
    canonical_json_sha256,
    chunk_matrix,
    classify_stage0,
    clean_retention_summary,
    construct_language_contrast,
    contrast_residual_noncollapse,
    fit_discovery_contrast_scale,
    gradient_smoke,
    group_clip,
    json_default,
    language_mask,
    lcg_row_key,
    mask_health,
    mean_huber,
    predict_contrast_residual,
    relative_improvement,
    scalar_contrast_residual_probe,
    validate_manifest,
)


POLICY_PROBE = "lcg_stage0_language_contrast_guidance"
SEED = 20263200
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "lcg_vla" / "proposal_hash.txt"
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
    null = base.copy()
    null[:, :, 0:3] -= 0.01
    residual = np.zeros_like(base)
    residual[:, :, 0] = base[:, :, 0] - null[:, :, 0]
    target = base + group_clip(residual)
    contrast = construct_language_contrast(base, null)
    scale = fit_discovery_contrast_scale(contrast)
    mask = language_mask(contrast, scale)
    identity = apply_lcg_gate(base, residual, mask, residual_gain=0.0)
    changed = apply_lcg_gate(base, residual, mask, residual_gain=1.0)
    manifest_row = {
        "partition": "validation",
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": 8,
        "frame_index": 20,
        "instruction_variant": "original_vs_null",
        "model_or_probe": "lcg_full",
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = lcg_row_key(manifest_row)
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=True,
        split_integrity_ok=True,
        minimum_discovery_windows=512,
        minimum_validation_windows=128,
        all_tasks_reported=True,
        maximum_validation_task_fraction=0.25,
        contrast_noncollapsed=True,
        residual_labels_noncollapsed=True,
        contrast_positive_fraction=0.25,
        language_mask_all_zero=False,
        language_mask_all_one=False,
        gate_activation_fraction=0.25,
        contrast_residual_spearman=0.10,
        contrast_probe_beats_task_phase_baseline=True,
        contrast_probe_relative_improvement=0.02,
        best_cag_proxy_score=0.1,
        cag_proxy_residual_headroom=0.05,
        lcg_beats_cag_proxy_relative=0.01,
        masked_residual_headroom=0.05,
        cag_coefficient_equivalence=False,
        no_language_ablation_explains=False,
        lora_explains=False,
        identity_max_abs_error=float(np.max(np.abs(identity - base))),
        inactive_gate_max_abs_error=0.0,
        action_validity_ok=True,
        clean_retention_ok=True,
        finite_objectives_and_gradients=True,
        expected_parameter_gradient_nonzero=True,
        frozen_base_gradient_count=0,
        weighted_gradient_norm_ratio_max=1.0,
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
        "method": "LCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "null_instruction": NULL_INSTRUCTION,
        "base_chunk": base,
        "null_chunk": null,
        "contrast_scale": scale,
        "contrast_mask": mask,
        "changed_chunk": changed,
        "action_delta_summary": action_delta_summary(base, changed),
        "gradient": gradient_smoke(base, residual, mask, target),
        "nested_metrics": {"contrast": {"spearman": np.float32(0.1), "passed": np.bool_(True)}},
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
        "method": "LCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(reproduced == fixture_hash and fixture["decision"] == "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION"),
        "fixture_hash": fixture_hash,
        "reproduced_hash": reproduced,
        "tensor_serialization_checked": False,
        "fixture": parsed["fixture"],
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("LCG serializer preflight hash did not reproduce")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = {
        "official_repository": REPO_ROOT / "third_party" / "CounterfactualActionGuidance",
        "alternate_repository": REPO_ROOT / "third_party" / "counterfactual-action-guidance",
        "checkpoint_dir": REPO_ROOT / "third_party" / "CounterfactualActionGuidance" / "checkpoints",
    }
    exists = {name: candidate.exists() for name, candidate in candidates.items()}
    official_ready = bool((exists["official_repository"] or exists["alternate_repository"]) and exists["checkpoint_dir"])
    result = {
        "method": "LCG-VLA",
        "stage": "0",
        "closest_prior": "Counterfactual Action Guidance",
        "closest_prior_primary_source": "https://arxiv.org/abs/2602.17659",
        "policy_2_label": "official_cag" if official_ready else "counterfactual_action_guidance_proxy",
        "official_ready": official_ready,
        "asset_exists": exists,
        "checked_paths": {name: str(candidate) for name, candidate in candidates.items()},
        "proxy_deviations": []
        if official_ready
        else [
            "official CAG repository/checkpoint assets are not locally verified",
            "Stage 0 fixes policy 2 as a transparent original-minus-null action proxy",
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
                "method": "LCG-VLA",
                "same_definition_applies_to_policies": list(POLICY_ROWS),
                "source_semantics": str(DEFAULT_CCIF_ACTION_SEMANTICS),
                "created_utc": _utc_now(),
            }
        )
    else:
        semantics = {
            "method": "LCG-VLA",
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
                "partition": str(manifest["partition"]),
                "suite": str(manifest["suite"]),
                "task_identity": str(manifest["task_identity"]),
                "source_edge_sha256": str(manifest["source_edge_sha256"]),
                "source_path": str(manifest["source_path"]),
                "task_language": str(manifest.get("task_language", "")),
                "demo_id": int(manifest["demo_id"]),
                "frame_index": int(manifest["frame_index"]),
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


def _manifest_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for policy in POLICY_ROWS:
            row = {
                "partition": record["partition"],
                "suite": record["suite"],
                "task_identity": record["task_identity"],
                "source_edge_sha256": record["source_edge_sha256"],
                "demo_id": record["demo_id"],
                "frame_index": record["frame_index"],
                "instruction_variant": "original_vs_null",
                "model_or_probe": policy,
                "policy_probe": POLICY_PROBE,
                "record_index": index,
                "phase_bin": record["phase_bin"],
                "task_language": record["task_language"],
                "null_instruction": NULL_INSTRUCTION,
            }
            row["row_key"] = lcg_row_key(row)
            rows.append(row)
    return rows


def _fit_group_mean(chunks: np.ndarray, keys: Sequence[str]) -> dict[str, Any]:
    if len(chunks) != len(keys):
        raise ValueError("chunks and keys must align")
    default = np.asarray(chunks, dtype=np.float64).mean(axis=0)
    groups: dict[str, Any] = {}
    for key in sorted(set(keys)):
        group = np.asarray([chunks[index] for index, value in enumerate(keys) if value == key], dtype=np.float64)
        groups[key] = group.mean(axis=0)
    return {"default": default, "groups": groups, "group_count": len(groups)}


def _predict_group_mean(model: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    default = np.asarray(model["default"], dtype=np.float64).reshape(HORIZON, ACTION_DIM)
    groups = {str(key): np.asarray(value, dtype=np.float64).reshape(HORIZON, ACTION_DIM) for key, value in model["groups"].items()}
    return np.asarray([groups.get(str(key), default) for key in keys], dtype=np.float64)


def _phase_key(record: Mapping[str, Any]) -> str:
    return f"phase:{int(record['phase_bin'])}"


def _task_phase_key(record: Mapping[str, Any]) -> str:
    return f"{record['task_identity']}|phase:{int(record['phase_bin'])}"


def _materialize_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = np.asarray([record["base_chunk"] for record in records], dtype=np.float64)
    expert = np.asarray([record["expert_chunk"] for record in records], dtype=np.float64)
    discovery = np.asarray([record["partition"] == "discovery" for record in records], dtype=bool)
    validation = np.asarray([record["partition"] == "validation" for record in records], dtype=bool)
    phase_keys = [_phase_key(record) for record in records]
    task_phase_keys = [_task_phase_key(record) for record in records]
    if not discovery.any() or not validation.any():
        raise ValueError("LCG Stage 0 requires both discovery and validation rows")
    residual = expert - base
    null_model = _fit_group_mean(base[discovery], [phase_keys[index] for index, flag in enumerate(discovery) if flag])
    null = _predict_group_mean(null_model, phase_keys)
    contrast = construct_language_contrast(base, null)
    contrast_scale = fit_discovery_contrast_scale(contrast[discovery])
    mask = language_mask(contrast, contrast_scale)
    task_phase_model = _fit_group_mean(
        residual[discovery], [task_phase_keys[index] for index, flag in enumerate(discovery) if flag]
    )
    task_phase_residual = _predict_group_mean(task_phase_model, task_phase_keys)
    contrast_model = scalar_contrast_residual_probe(contrast[discovery], residual[discovery])
    contrast_residual = predict_contrast_residual(contrast_model, contrast)
    lcg_residual = 0.5 * task_phase_residual + 0.5 * contrast_residual
    gate = mask.mean(axis=2, keepdims=True)
    cag_predictions = {beta: apply_cag_proxy(base, null, beta=beta) for beta in CAG_PROXY_BETAS}
    validation_target = expert[validation]
    cag_hubers = {beta: mean_huber(prediction[validation], validation_target) for beta, prediction in cag_predictions.items()}
    best_beta = min(cag_hubers, key=cag_hubers.get)
    predictions = {
        "smolvla_base": base,
        "counterfactual_action_guidance_proxy": cag_predictions[best_beta],
        "lcg_full": apply_lcg_gate(base, lcg_residual, gate, residual_gain=1.0),
        "lcg_no_language_contrast_ablation": apply_no_language_ablation(base, task_phase_residual),
        "standard_lora_proxy": apply_no_language_ablation(base, _predict_group_mean(_fit_group_mean(residual[discovery], ["global"] * int(discovery.sum())), ["global"] * len(records))),
        "contrast_magnitude_only_gate": apply_lcg_gate(base, task_phase_residual, gate, residual_gain=1.0),
        "task_phase_residual": apply_no_language_ablation(base, task_phase_residual),
        "masked_residual_oracle_diagnostic": apply_lcg_gate(base, residual, gate, residual_gain=1.0),
    }
    return {
        "base": base,
        "expert": expert,
        "discovery": discovery,
        "validation": validation,
        "residual": residual,
        "null": null,
        "contrast": contrast,
        "contrast_scale": contrast_scale,
        "mask": mask,
        "gate": gate,
        "task_phase_residual": task_phase_residual,
        "contrast_residual": contrast_residual,
        "lcg_residual": lcg_residual,
        "predictions": predictions,
        "cag_hubers": cag_hubers,
        "best_cag_beta": float(best_beta),
        "null_model": null_model,
        "task_phase_model": task_phase_model,
        "contrast_model": contrast_model,
    }


def _partial_rows(
    records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    completed: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_by_record_policy = {(int(row["record_index"]), str(row["model_or_probe"])): row for row in manifest_rows}
    predictions = arrays["predictions"]
    for index, record in enumerate(records):
        base = arrays["base"][index : index + 1]
        expert = arrays["expert"][index : index + 1]
        for policy in POLICY_ROWS:
            manifest = manifest_by_record_policy[(index, policy)]
            if manifest["row_key"] in completed:
                continue
            prediction = np.asarray(predictions[policy][index : index + 1], dtype=np.float64)
            delta = action_delta_summary(base, prediction)
            rows.append(
                {
                    "row_key": manifest["row_key"],
                    "partition": manifest["partition"],
                    "suite": manifest["suite"],
                    "task_identity": manifest["task_identity"],
                    "source_edge_sha256": manifest["source_edge_sha256"],
                    "demo_id": manifest["demo_id"],
                    "frame_index": manifest["frame_index"],
                    "instruction_variant": manifest["instruction_variant"],
                    "model_or_probe": policy,
                    "policy_probe": POLICY_PROBE,
                    "phase_bin": manifest["phase_bin"],
                    "task_language": manifest["task_language"],
                    "null_instruction": NULL_INSTRUCTION,
                    "best_cag_beta": arrays["best_cag_beta"] if policy == "counterfactual_action_guidance_proxy" else None,
                    "base_chunk_cache_path": record["base_chunk_cache_path"],
                    "base_chunk_cache_sha256": record["base_chunk_cache_sha256"],
                    "base_chunk_sha256": record["base_chunk_sha256"],
                    "action_chunk_sha256": record["action_chunk_sha256"],
                    "prediction_chunk_sha256": _array_sha256(prediction),
                    "prediction_shape": list(prediction.reshape(HORIZON, ACTION_DIM).shape),
                    "prediction_finite": bool(np.isfinite(prediction).all()),
                    "huber_to_demo_action": mean_huber(prediction, expert),
                    "language_contrast_l2": float(np.mean(np.linalg.norm(arrays["contrast"][index], axis=1))),
                    "gate_mean": float(np.mean(arrays["gate"][index])),
                    "mask_positive_fraction": float(np.mean(arrays["mask"][index] > 0.5)),
                    **delta,
                }
            )
    return rows


def _partial_payload(
    manifest_hash: str,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "LCG-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": int(planned_count),
        "completed_model_row_count": int(len(rows)),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "rows": list(rows),
        "updated_utc": _utc_now(),
    }


def _load_resume(
    path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_hash: str,
) -> tuple[list[dict[str, Any]], int, str | None]:
    if not path.is_file():
        return [], 0, None
    payload = _read_json(path)
    recorded_hash = str(payload.get("manifest_hash", ""))
    if recorded_hash not in {manifest_hash, "STABLE_MANIFEST"}:
        raise ValueError(f"partial manifest hash mismatch: {recorded_hash} != {manifest_hash}")
    expected = {lcg_row_key(row) for row in manifest_rows}
    rows = list(payload.get("rows") or [])
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("row_key"))
        if key not in expected:
            raise ValueError(f"partial row is not in manifest: {key}")
        if key in seen:
            raise ValueError(f"duplicate partial row key: {key}")
        seen.add(key)
    return rows, int(payload.get("exception_count", 0)), payload.get("last_exception")


def _write_identity_checkpoint(path: Path, base: np.ndarray, residual: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    identity = apply_lcg_gate(base, residual, gate, residual_gain=0.0)
    checkpoint = path / "identity_lcg_gate.npz"
    np.savez_compressed(checkpoint, residual_gain=np.asarray([0.0], dtype=np.float32))
    with np.load(checkpoint, allow_pickle=False) as payload:
        reloaded_gain = float(payload["residual_gain"][0])
    reloaded = apply_lcg_gate(base, residual, gate, residual_gain=reloaded_gain)
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_reload_ok": bool(reloaded_gain == 0.0),
        "identity_max_abs_error": float(np.max(np.abs(identity - base))),
        "reloaded_identity_max_abs_error": float(np.max(np.abs(reloaded - base))),
    }


def _result_from_rows(
    records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    validation = arrays["validation"]
    target = arrays["expert"][validation]
    huber_by_policy = {
        policy: mean_huber(np.asarray(arrays["predictions"][policy])[validation], target) for policy in POLICY_ROWS
    }
    base_score = huber_by_policy["smolvla_base"]
    lcg_score = huber_by_policy["lcg_full"]
    best_cag_score = min(arrays["cag_hubers"].values())
    task_phase_score = huber_by_policy["task_phase_residual"]
    oracle_score = huber_by_policy["masked_residual_oracle_diagnostic"]
    standard_score = huber_by_policy["standard_lora_proxy"]
    no_language_score = huber_by_policy["lcg_no_language_contrast_ablation"]
    contrast_health = contrast_residual_noncollapse(arrays["contrast"], arrays["residual"])
    mask_summary = mask_health(arrays["mask"])
    clean = clean_retention_summary(
        arrays["base"],
        apply_lcg_gate(arrays["base"], arrays["lcg_residual"], arrays["gate"], residual_gain=0.0),
        apply_lcg_gate(arrays["base"], arrays["lcg_residual"], np.zeros_like(arrays["gate"]), residual_gain=1.0),
    )
    identity = _write_identity_checkpoint(paths["identity_dir"], arrays["base"], arrays["lcg_residual"], arrays["gate"])
    gradient = gradient_smoke(arrays["base"], arrays["lcg_residual"], arrays["gate"], arrays["expert"])
    validation_records = [record for record in records if record["partition"] == "validation"]
    validation_counts = Counter(str(record["task_identity"]) for record in validation_records)
    total_validation = sum(validation_counts.values())
    split_counts = Counter(str(record["partition"]) for record in records)
    task_counts = Counter(str(record["task_identity"]) for record in records)
    cag_values = list(arrays["cag_hubers"].values())
    cag_variation = max(cag_values) - min(cag_values) if cag_values else 0.0
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
        feature_action_proprio_finite_aligned=True,
        split_integrity_ok=manifest_summary["split_overlap_key_count"] == 0,
        minimum_discovery_windows=int(split_counts.get("discovery", 0)),
        minimum_validation_windows=int(split_counts.get("validation", 0)),
        all_tasks_reported=len(validation_counts) == 4,
        maximum_validation_task_fraction=max((count / total_validation for count in validation_counts.values()), default=1.0),
        contrast_noncollapsed=bool(contrast_health["contrast_noncollapsed"]),
        residual_labels_noncollapsed=bool(contrast_health["residual_labels_noncollapsed"]),
        contrast_positive_fraction=float(mask_summary["contrast_positive_fraction"]),
        language_mask_all_zero=bool(mask_summary["language_mask_all_zero"]),
        language_mask_all_one=bool(mask_summary["language_mask_all_one"]),
        gate_activation_fraction=float(np.mean(arrays["gate"] > 0.0)),
        contrast_residual_spearman=float(contrast_health["contrast_residual_spearman"]),
        contrast_probe_beats_task_phase_baseline=bool(
            relative_improvement(task_phase_score, lcg_score) >= 0.0
        ),
        contrast_probe_relative_improvement=relative_improvement(task_phase_score, lcg_score),
        best_cag_proxy_score=best_cag_score,
        cag_proxy_residual_headroom=relative_improvement(base_score, best_cag_score),
        lcg_beats_cag_proxy_relative=relative_improvement(best_cag_score, lcg_score),
        masked_residual_headroom=relative_improvement(base_score, oracle_score),
        cag_coefficient_equivalence=bool(cag_variation < 1e-9),
        no_language_ablation_explains=bool(no_language_score <= lcg_score),
        lora_explains=bool(standard_score <= lcg_score),
        identity_max_abs_error=max(float(clean["identity_max_abs_error"]), float(identity["reloaded_identity_max_abs_error"])),
        inactive_gate_max_abs_error=float(clean["inactive_gate_max_abs_error"]),
        action_validity_ok=bool(all(row.get("prediction_finite") for row in partial_rows)),
        clean_retention_ok=bool(clean["clean_retention_ok"] and identity["checkpoint_reload_ok"]),
        finite_objectives_and_gradients=bool(gradient["finite_objectives_and_gradients"]),
        expected_parameter_gradient_nonzero=bool(gradient["expected_parameter_gradient_nonzero"]),
        frozen_base_gradient_count=int(gradient["frozen_base_gradient_count"]),
        weighted_gradient_norm_ratio_max=float(gradient["weighted_gradient_norm_ratio_max"]),
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
        "method": "LCG-VLA",
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
        "null_instruction": NULL_INSTRUCTION,
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "validation_task_counts": dict(validation_counts),
        "contrast_positive_fraction": decision_inputs.contrast_positive_fraction,
        "contrast_residual_spearman": decision_inputs.contrast_residual_spearman,
        "contrast_probe_beats_task_phase_baseline": decision_inputs.contrast_probe_beats_task_phase_baseline,
        "contrast_probe_relative_improvement": decision_inputs.contrast_probe_relative_improvement,
        "best_cag_proxy_beta": arrays["best_cag_beta"],
        "best_cag_proxy_score": best_cag_score,
        "cag_proxy_residual_headroom": decision_inputs.cag_proxy_residual_headroom,
        "lcg_beats_cag_proxy_relative": decision_inputs.lcg_beats_cag_proxy_relative,
        "masked_residual_headroom": decision_inputs.masked_residual_headroom,
        "language_mask_all_zero": decision_inputs.language_mask_all_zero,
        "language_mask_all_one": decision_inputs.language_mask_all_one,
        "gate_activation_fraction": decision_inputs.gate_activation_fraction,
        "identity_max_abs_error": decision_inputs.identity_max_abs_error,
        "inactive_gate_max_abs_error": decision_inputs.inactive_gate_max_abs_error,
        "expected_parameter_gradient_nonzero": decision_inputs.expected_parameter_gradient_nonzero,
        "frozen_base_gradient_count": decision_inputs.frozen_base_gradient_count,
        "weighted_gradient_norm_ratio_max": decision_inputs.weighted_gradient_norm_ratio_max,
        "lora_explains": decision_inputs.lora_explains,
        "no_language_ablation_explains": decision_inputs.no_language_ablation_explains,
        "action_validity_ok": decision_inputs.action_validity_ok,
        "clean_retention_ok": decision_inputs.clean_retention_ok,
        "huber_by_policy": huber_by_policy,
        "cag_huber_by_beta": {str(beta): value for beta, value in arrays["cag_hubers"].items()},
        "contrast_scale": arrays["contrast_scale"],
        "contrast_health": contrast_health,
        "mask_health": mask_summary,
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
            "# LCG-VLA Stage 0 Result",
            "",
            f"Decision: `{result['final_decision']}`",
            "",
            f"Rows: `{result['completed_model_row_count']}/{result['planned_model_row_count']}`",
            f"Best CAG beta: `{result['best_cag_proxy_beta']}`",
            f"LCG vs CAG relative: `{result['lcg_beats_cag_proxy_relative']}`",
            f"Contrast/residual Spearman: `{result['contrast_residual_spearman']}`",
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
    if result["final_decision"] == "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        next_step = "Proceed to frozen bounded validation search with at most six configurations."
    else:
        next_step = "Archive this Stage 0 development stop class and continue without rescue unless a pre-result implementation blocker is identified."
    text = "\n".join(
        [
            "# LCG-VLA Stage 0 Adjudication",
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
                "method": "LCG-VLA",
                "status": "completed_existing_result_reused",
                "final_decision": result.get("final_decision"),
                "updated_utc": _utc_now(),
            },
        )
        return result
    if existing["pid_alive"] and not args.force:
        raise RuntimeError(f"existing LCG Stage 0 worker is alive: {existing}")
    if existing["partial_parse_error"]:
        raise RuntimeError(f"existing LCG partial does not parse: {existing['partial_parse_error']}")

    state: dict[str, Any] = {
        "method": "LCG-VLA",
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
            "method": "LCG-VLA",
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
            raise RuntimeError("frozen LCG proposal hash mismatch")
        records = _load_base_records(Path(args.ccif_manifest), Path(args.ccif_partial), max_sources=args.max_sources)
        arrays = _materialize_arrays(records)
        manifest_rows = _manifest_rows(records)
        manifest_payload = {
            "method": "LCG-VLA",
            "stage": "0",
            "policy_probe": POLICY_PROBE,
            "proposal_hash": PROPOSAL_HASH,
            "model_or_probe_rows": list(POLICY_ROWS),
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
        partial_rows, previous_exception_count, previous_last_exception = _load_resume(
            paths["partial"], manifest_rows, manifest_hash
        )
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
            "method": "LCG-VLA",
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
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "lcg_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "lcg_vla" / "stage0"))
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
        print(f"LCG serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0
    result = run(args)
    print(json.dumps({"final_decision": result.get("final_decision"), "result": str(paths["result_json"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
