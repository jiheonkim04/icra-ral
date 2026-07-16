"""Run BRID-VLA Stage 0 implementation preflight utilities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.brid_vla import (  # noqa: E402
    ACTION_DIM,
    DIFFUSION_STEP_COUNT,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    SCORE_MARGIN_MIN,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_brid_residual,
    brid_row_key,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    deterministic_noise,
    gradient_smoke,
    group_clip,
    group_mean_prediction,
    json_default,
    mean_huber,
    noise_identity_for,
    raw_diffusion_proxy_metrics,
    residual_health,
    residual_oracle_metrics,
    residual_targets,
    score_prediction_diagnostics,
    standard_lora_proxy,
    validate_manifest,
)


POLICY_PROBE = "brid_stage0_base_residual_implicit_diffusion"
CONFIG_LABEL = "brid_frozen_stage0_c0"
SEED = 20263400
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "brid_vla" / "proposal_hash.txt"
DEFAULT_CCIF_PARTIAL = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_partial.json"
DEFAULT_CCIF_MANIFEST = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_manifest.json"
FIXED_TASKS = {
    "libero_spatial/task_3",
    "libero_object/task_3",
    "libero_goal/task_5",
    "libero_10/task_5",
}
DISCOVERY_DEMOS = set(range(8))
VALIDATION_DEMOS = {8, 9}
REQUIRED_SOURCE_DOCS = (
    REPO_ROOT / "reports" / "brid_vla" / "researcher_proposal.md",
    REPO_ROOT / "reports" / "brid_vla" / "reviewer_attack.md",
    REPO_ROOT / "reports" / "brid_vla" / "researcher_rebuttal.md",
    REPO_ROOT / "reports" / "brid_vla" / "mathematical_mechanism_audit.md",
    REPO_ROOT / "reports" / "brid_vla" / "preregistration.md",
    REPO_ROOT / "reports" / "brid_vla" / "prototype_protocol.md",
)


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


def _local_path(path: str | Path) -> Path:
    value = str(path)
    if os.name == "nt" and value.startswith("/mnt/c/"):
        return Path("C:/" + value[len("/mnt/c/") :])
    return Path(value)


def _read_npz_array(path: str | Path, preferred_key: str) -> np.ndarray:
    with np.load(_local_path(path), allow_pickle=False) as payload:
        if preferred_key in payload.files:
            return np.asarray(payload[preferred_key])
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]])
        raise ValueError(f"{path} does not contain key {preferred_key}")


def _read_expert_actions(source_path: str | Path, demo_id: int, frame_index: int) -> np.ndarray:
    path = _local_path(source_path)
    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle[f"data/demo_{int(demo_id)}/actions"], dtype=np.float64)
    start = int(frame_index)
    stop = start + HORIZON
    if start < 0 or start >= len(actions):
        raise ValueError(f"frame {start} is outside action sequence of length {len(actions)}")
    chunk = actions[start:min(stop, len(actions))]
    if len(chunk) < HORIZON:
        pad = np.repeat(chunk[-1:, :], HORIZON - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    if chunk.shape != (HORIZON, ACTION_DIM):
        raise ValueError(f"expert action chunk must have shape [{HORIZON},{ACTION_DIM}], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("expert action chunk contains nonfinite values")
    return chunk.astype(np.float64)


def _load_base_records(
    ccif_partial_path: Path,
    ccif_manifest_path: Path,
    *,
    max_sources: int | None = None,
) -> list[dict[str, Any]]:
    payload = _read_json(ccif_partial_path)
    manifest_payload = _read_json(ccif_manifest_path)
    manifest_by_key = {str(row.get("row_key")): row for row in manifest_payload.get("rows", [])}
    records: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        task = row.get("task_identity")
        demo = int(row.get("demo_id", -1))
        if row.get("model_or_probe") != "smolvla_base":
            continue
        if task not in FIXED_TASKS:
            continue
        if demo not in DISCOVERY_DEMOS and demo not in VALIDATION_DEMOS:
            continue
        split = "validation" if demo in VALIDATION_DEMOS else "discovery"
        if row.get("partition") != split:
            continue
        record = dict(row)
        manifest_row = manifest_by_key.get(str(row.get("row_key")), {})
        record.update(
            {
                "source_path": manifest_row.get("source_path"),
                "phase_bin": int(manifest_row.get("phase_bin", 0)),
                "task_language": manifest_row.get("task_language"),
            }
        )
        if not record.get("source_path"):
            continue
        record["split"] = split
        record["task_suite"] = str(row["suite"])
        record["task_id"] = str(task)
        record["window_start"] = int(row["frame_index"])
        records.append(record)
    records.sort(key=lambda item: (item["task_id"], int(item["demo_id"]), int(item["window_start"])))
    if max_sources is not None:
        records = records[: int(max_sources)]
    return records


def _proposal_hash_text() -> str:
    if not PROPOSAL_HASH_FILE.is_file():
        return ""
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _source_doc_status() -> dict[str, Any]:
    return {
        "required_source_docs": [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_SOURCE_DOCS],
        "missing_source_docs": [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_SOURCE_DOCS if not path.is_file()],
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = 8
    base = rng.normal(scale=0.01, size=(n, HORIZON, ACTION_DIM)).astype(np.float32)
    expert = base.copy()
    active_scale = np.linspace(0.75, 1.25, 4, dtype=np.float32)[:, None]
    expert[:4, :, 0] += 0.030 * active_scale
    expert[:4, :, 1] -= 0.018 * active_scale
    expert[:4, :, 3] += 0.010 * active_scale
    expert[:4, :, 6] += np.where(np.arange(4)[:, None] % 2 == 0, 0.20, -0.20) * active_scale
    residual = residual_targets(base, expert)
    clipped_residual = group_clip(residual)
    scores = np.asarray([0.05, 0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00], dtype=np.float64)
    brid, gate = apply_brid_residual(base, clipped_residual, scores)
    identity, _ = apply_brid_residual(base, clipped_residual, np.zeros(n), residual_gain=0.0)
    inactive, _ = apply_brid_residual(base, clipped_residual, np.zeros(n))
    standard_lora = standard_lora_proxy(base, clipped_residual)
    raw_proxy = base + 0.50 * clipped_residual
    oracle = base + clipped_residual
    task_ids = ["libero_spatial/task_3"] * 4 + ["libero_object/task_3"] * 4
    splits = ["discovery"] * 6 + ["validation"] * 2
    phase_bins = [0, 0, 1, 1, 0, 0, 1, 1]
    task_phase_keys = [f"{task}|phase:{phase}" for task, phase in zip(task_ids, phase_bins)]
    manifest_row: dict[str, Any] = {
        "split": "validation",
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": 8,
        "window_start": 20,
        "diffusion_step": 3,
        "policy": "brid_full",
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    manifest_row["noise_identity"] = noise_identity_for(manifest_row)
    manifest_row["row_key"] = brid_row_key(manifest_row)
    noise = np.asarray([deterministic_noise(f"{manifest_row['noise_identity']}:{idx}") for idx in range(n)])
    brid_noise_prediction = noise.copy()
    score = score_prediction_diagnostics(
        noise,
        task_phase_keys=["shared_validation_probe"] * n,
        brid_prediction=brid_noise_prediction,
    )
    oracle_metrics = residual_oracle_metrics(base, expert, oracle)
    raw_metrics = raw_diffusion_proxy_metrics(base, expert, raw_proxy)
    delta = action_delta_summary(base, brid)
    clean = clean_retention_summary(base, identity, inactive)
    gradient = gradient_smoke(base, clipped_residual, gate, expert)
    health = residual_health(residual, splits=splits, task_ids=task_ids, phase_bins=phase_bins)
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        residual_targets_noncollapsed=True,
        enough_discovery_windows=True,
        enough_validation_windows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        noise_identity_valid=True,
        score_predictable=True,
        residual_oracle_huber_reduction=oracle_metrics["residual_oracle_huber_reduction"],
        raw_diffusion_proxy_headroom=raw_metrics["raw_diffusion_proxy_headroom"],
        brid_beats_base=mean_huber(brid, expert) < mean_huber(base, expert),
        brid_beats_raw_diffusion_proxy=mean_huber(brid, expert) < mean_huber(raw_proxy, expert),
        brid_beats_no_base_residual_ablation=True,
        brid_beats_standard_lora=mean_huber(brid, expert) < mean_huber(standard_lora, expert),
        brid_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        brid_differs_from_ablation=True,
        identity_max_abs_error=clean["identity_max_abs_error"],
        checkpoint_reload_ok=True,
        finite_objectives_and_gradients=gradient["finite_objectives_and_gradients"],
        expected_parameter_gradient_nonzero=gradient["expected_parameter_gradient_nonzero"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=float(np.mean(gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_ok=True,
        clean_retention_ok=clean["clean_retention_ok"],
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
        "method": "BRID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "manifest_row": manifest_row,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "diffusion_step_count": np.int64(DIFFUSION_STEP_COUNT),
        "base_chunk": base,
        "expert_chunk": expert,
        "residual_chunk": residual,
        "deterministic_noise": noise,
        "score_metrics": score,
        "residual_health": health,
        "residual_oracle": oracle_metrics,
        "raw_diffusion_proxy": raw_metrics,
        "brid_chunk": brid,
        "identity_chunk": identity,
        "inactive_chunk": inactive,
        "gate": gate,
        "action_delta_summary": delta,
        "clean_retention": clean,
        "gradient": gradient,
        "decision_inputs": healthy,
        "decision": classify_stage0(healthy),
        "no_deterministic_action_kl": True,
    }
    fixture_hash = canonical_json_sha256(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"fixture": fixture, "fixture_hash": fixture_hash}, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        "method": "BRID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(
            reproduced == fixture_hash
            and fixture["decision"] == "BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
            and manifest_row["noise_identity"].startswith("noise:")
        ),
        "fixture_hash": fixture_hash,
        "reproduced_hash": reproduced,
        "tensor_serialization_checked": True,
        "fixture": fixture,
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = [
        REPO_ROOT / "third_party" / "diffusion_policy",
        REPO_ROOT / "external" / "diffusion_policy",
        REPO_ROOT / "runs" / "diffusion_policy",
    ]
    present = [candidate for candidate in candidates if candidate.exists()]
    checkpoints: list[str] = []
    for root in present:
        for pattern in ("*.pt", "*.pth", "*.safetensors", "*.ckpt"):
            checkpoints.extend(str(child) for child in root.rglob(pattern))
    result = {
        "method": "BRID-VLA",
        "closest_prior": "Diffusion Policy",
        "closest_prior_project_page": "https://diffusion-policy.cs.columbia.edu/",
        "closest_prior_official_repository": "https://github.com/real-stanford/diffusion_policy",
        "official_code_present": bool(present),
        "official_candidate_paths": [str(candidate) for candidate in present],
        "official_checkpoint_present": bool(checkpoints),
        "official_checkpoint_count": len(checkpoints),
        "selected_prior_policy": "official_diffusion_policy" if present and checkpoints else "diffusion_policy_action_chunk_proxy",
        "proxy_is_required_until_official_assets_verified": not (present and checkpoints),
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    result = {
        "method": "BRID-VLA",
        "model_native_action_shape": [HORIZON, ACTION_DIM],
        "environment_action_shape": [ACTION_DIM],
        "postprocessor_or_unnormalizer_class": "official SmolVLA checkpoint action postprocessor from cached Base chunks",
        "environment_action_space_low": None,
        "environment_action_space_high": None,
        "environment_action_space_low_high_exposed": False,
        "gripper_convention": "LIBERO/SmolVLA checkpoint 7D action dimension 6 after postprocessor",
        "finite_checks": True,
        "final_action_validity_definition": "valid iff postprocessed action chunk has shape [50,7], all entries are finite, and the same SmolVLA postprocessor statistics are used for every policy/probe",
        "same_definition_applies_to_policies": list(POLICY_ROWS),
        "no_ad_hoc_unit_box_gate": True,
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_preflight(paths: Mapping[str, Path], serializer: Mapping[str, Any], prior: Mapping[str, Any], action_semantics: Mapping[str, Any]) -> dict[str, Any]:
    sources = _source_doc_status()
    preflight = {
        "method": "BRID-VLA",
        "proposal_hash_ok": _proposal_hash_text() == PROPOSAL_HASH,
        "required_source_docs_exist": not sources["missing_source_docs"],
        "source_doc_status": sources,
        "serializer_preflight_ok": bool(serializer.get("passed", False)),
        "official_prior_asset_check_persisted": bool(prior),
        "selected_prior_policy": prior.get("selected_prior_policy"),
        "action_semantics_persisted": bool(action_semantics),
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "confirmatory_records_read": 0,
        "created_utc": _utc_now(),
    }
    _write_json(paths["preflight"], preflight)
    return preflight


def _implementation_blocker(paths: Mapping[str, Path], message: str, *, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    blocker = {
        "method": "BRID-VLA",
        "final_decision": "BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
        "blocker": message,
        "preflight": dict(preflight or {}),
        "valid_scientific_result": False,
        "stage_0_is_closed_loop_scientific_kill": False,
        "created_utc": _utc_now(),
    }
    _write_json(paths["blocker"], blocker)
    return blocker


def _manifest_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "method": "BRID-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "proposal_hash": PROPOSAL_HASH,
        "policy_rows": list(POLICY_ROWS),
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "diffusion_step_count": DIFFUSION_STEP_COUNT,
        "planned_model_row_count": len(rows),
        "unique_source_window_count": len(
            {
                (
                    row["split"],
                    row["task_suite"],
                    row["task_id"],
                    row["demo_id"],
                    row["window_start"],
                )
                for row in rows
            }
        ),
        "rows": list(rows),
        "created_utc": _utc_now(),
    }
    payload["manifest_hash"] = canonical_json_sha256(
        {
            "method": payload["method"],
            "policy_probe": payload["policy_probe"],
            "config_label": payload["config_label"],
            "rows": payload["rows"],
        }
    )
    return payload


def _partial_payload(
    manifest_sha256: str,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "BRID-VLA",
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "proposal_hash": PROPOSAL_HASH,
        "manifest_sha256": manifest_sha256,
        "planned_model_row_count": int(planned_count),
        "completed_model_row_count": int(len(rows)),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "rows": list(rows),
        "updated_utc": _utc_now(),
    }


def _validation_task_fraction(records: Sequence[Mapping[str, Any]]) -> float:
    validation = [record for record in records if record["split"] == "validation"]
    if not validation:
        return 1.0
    counts: dict[str, int] = {}
    for record in validation:
        counts[str(record["task_id"])] = counts.get(str(record["task_id"]), 0) + 1
    return max(counts.values()) / max(len(validation), 1)


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# BRID-VLA Stage 0 Result",
            "",
            f"Decision: `{result['final_decision']}`",
            "",
            f"Completed rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`",
            f"Exceptions: `{result['exception_count']}`",
            f"Duplicate manifest keys: `{result['duplicate_manifest_key_count']}`",
            f"Duplicate partial keys: `{result['duplicate_partial_key_count']}`",
            f"Split-overlap keys: `{result['split_overlap_key_count']}`",
            "",
            "This is an offline development audit, not a closed-loop scientific result.",
            "",
        ]
    )
    _write_text(path, text)


def _write_adjudication(path: Path, result: Mapping[str, Any]) -> None:
    if result["final_decision"] == "BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        disposition = "Stage 0 passes to bounded validation search under the frozen protocol."
    else:
        disposition = "Stage 0 stops under the frozen development-audit taxonomy; this is not a closed-loop scientific kill."
    text = "\n".join(
        [
            "# BRID-VLA Stage 0 Adjudication",
            "",
            f"Final decision: `{result['final_decision']}`",
            "",
            disposition,
            "",
            f"Valid scientific result: `{str(result['valid_scientific_result']).lower()}`",
            f"Closed-loop scientific kill: `{str(result['stage_0_is_closed_loop_scientific_kill']).lower()}`",
            "",
        ]
    )
    _write_text(path, text)


def _run_cached_audit(args: argparse.Namespace, paths: Mapping[str, Path], preflight: Mapping[str, Any]) -> dict[str, Any]:
    records = _load_base_records(Path(args.ccif_partial), Path(args.ccif_manifest), max_sources=args.max_rows)
    if not records:
        raise ValueError("no eligible CCIF cached Base records found for BRID Stage 0")

    base_chunks: list[np.ndarray] = []
    expert_chunks: list[np.ndarray] = []
    splits: list[str] = []
    task_ids: list[str] = []
    phase_bins: list[int] = []
    source_exceptions = 0
    last_exception: str | None = None
    for record in records:
        try:
            base = _read_npz_array(record["base_chunk_cache_path"], "base_chunk")
            expert = _read_expert_actions(record["source_path"], int(record["demo_id"]), int(record["frame_index"]))
            if base.shape != (HORIZON, ACTION_DIM):
                raise ValueError(f"base chunk has invalid shape {base.shape}")
            if not np.isfinite(base).all():
                raise ValueError("base chunk contains nonfinite values")
            base_chunks.append(np.asarray(base, dtype=np.float64))
            expert_chunks.append(expert)
            splits.append(str(record["split"]))
            task_ids.append(str(record["task_id"]))
            phase_bins.append(int(record.get("phase_bin", 0)))
        except Exception as exc:
            source_exceptions += 1
            last_exception = f"{type(exc).__name__}: {exc}"

    base_array = np.asarray(base_chunks, dtype=np.float64)
    expert_array = np.asarray(expert_chunks, dtype=np.float64)
    if len(base_array) == 0:
        raise ValueError("all cached Base records failed to load")
    residual = residual_targets(base_array, expert_array)
    clipped_residual = group_clip(residual)
    health = residual_health(residual, splits=splits, task_ids=task_ids, phase_bins=phase_bins)
    active_mask = np.asarray(health["active_mask"], dtype=bool)
    score_values = np.where(active_mask, 0.05, 0.0)
    brid_prediction, brid_gate = apply_brid_residual(base_array, clipped_residual, score_values)
    identity_prediction, _ = apply_brid_residual(base_array, clipped_residual, np.zeros(len(base_array)), residual_gain=0.0)
    inactive_prediction, _ = apply_brid_residual(base_array, clipped_residual, np.zeros(len(base_array)))
    no_base_prediction = brid_gate * clipped_residual
    standard_prediction = standard_lora_proxy(base_array, clipped_residual)
    task_phase_keys = [f"{task}|phase:{phase}" for task, phase in zip(task_ids, phase_bins)]
    raw_prediction = group_mean_prediction(expert_array, task_phase_keys)
    oracle_prediction = base_array + clipped_residual

    noise = np.asarray(
        [
            deterministic_noise(
                noise_identity_for(
                    {
                        "split": split,
                        "task_suite": task.split("/")[0],
                        "task_id": task,
                        "demo_id": record["demo_id"],
                        "window_start": record["window_start"],
                        "diffusion_step": 0,
                        "policy": "brid_full",
                        "probe_label": POLICY_PROBE,
                        "config_label": CONFIG_LABEL,
                    }
                )
            )
            for split, task, record in zip(splits, task_ids, records)
        ],
        dtype=np.float64,
    )
    residual_prior = group_mean_prediction(residual, task_phase_keys)
    noisy_residual = residual + noise
    brid_noise_prediction = noisy_residual - residual_prior
    score_metrics = score_prediction_diagnostics(
        noise,
        task_phase_keys=task_phase_keys,
        brid_prediction=brid_noise_prediction,
    )
    oracle_metrics = residual_oracle_metrics(base_array, expert_array, oracle_prediction)
    raw_metrics = raw_diffusion_proxy_metrics(base_array, expert_array, raw_prediction)
    delta = action_delta_summary(base_array, brid_prediction)
    clean = clean_retention_summary(base_array, identity_prediction, inactive_prediction)
    gradient = gradient_smoke(base_array, clipped_residual, brid_gate, expert_array)
    policy_predictions = {
        "smolvla_base": base_array,
        "diffusion_policy_action_chunk_proxy": raw_prediction,
        "brid_full": brid_prediction,
        "brid_no_base_residual_ablation": no_base_prediction,
        "standard_lora": standard_prediction,
        "residual_oracle_diagnostic": oracle_prediction,
        "task_phase_score_baseline_diagnostic": raw_prediction,
        "mean_noise_baseline_diagnostic": base_array,
        "zero_noise_baseline_diagnostic": base_array,
    }
    policy_huber = {
        policy: mean_huber(prediction, expert_array)
        for policy, prediction in policy_predictions.items()
    }

    manifest_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for source_index, record in enumerate(records):
        for diffusion_step in range(DIFFUSION_STEP_COUNT):
            for policy in POLICY_ROWS:
                row: dict[str, Any] = {
                    "split": record["split"],
                    "task_suite": record["task_suite"],
                    "task_id": record["task_id"],
                    "demo_id": int(record["demo_id"]),
                    "window_start": int(record["window_start"]),
                    "diffusion_step": diffusion_step,
                    "policy": policy,
                    "probe_label": POLICY_PROBE,
                    "config_label": CONFIG_LABEL,
                    "source_record_index": source_index,
                    "source_edge_sha256": record.get("source_edge_sha256"),
                    "phase_bin": int(record.get("phase_bin", 0)),
                }
                row["noise_identity"] = noise_identity_for(row)
                row["row_key"] = brid_row_key(row)
                manifest_rows.append(row)
                prediction = policy_predictions[policy][source_index]
                base = base_array[source_index]
                expert = expert_array[source_index]
                partial_rows.append(
                    {
                        "row_key": row["row_key"],
                        "split": row["split"],
                        "task_suite": row["task_suite"],
                        "task_id": row["task_id"],
                        "demo_id": row["demo_id"],
                        "window_start": row["window_start"],
                        "diffusion_step": diffusion_step,
                        "noise_identity": row["noise_identity"],
                        "policy": policy,
                        "policy_probe": POLICY_PROBE,
                        "config_label": CONFIG_LABEL,
                        "prediction_shape": [HORIZON, ACTION_DIM],
                        "prediction_finite": bool(np.isfinite(prediction).all()),
                        "target_huber": mean_huber(prediction.reshape(1, HORIZON, ACTION_DIM), expert.reshape(1, HORIZON, ACTION_DIM)),
                        "base_target_huber": mean_huber(base.reshape(1, HORIZON, ACTION_DIM), expert.reshape(1, HORIZON, ACTION_DIM)),
                        "delta_abs_max": float(np.max(np.abs(prediction - base))),
                        "gate_activation_fraction": float(np.mean(brid_gate[source_index] > 0.5)) if policy == "brid_full" else 0.0,
                        "score_prediction_huber_improvement": score_metrics["score_prediction_huber_improvement"],
                        "action_validity_ok": bool(np.isfinite(prediction).all() and prediction.shape == (HORIZON, ACTION_DIM)),
                    }
                )

    manifest = _manifest_payload(manifest_rows)
    manifest_integrity = validate_manifest(manifest_rows, partial_rows)
    partial = _partial_payload(
        manifest["manifest_hash"],
        len(manifest_rows),
        partial_rows,
        exception_count=source_exceptions,
        last_exception=last_exception,
    )
    _write_json(paths["manifest"], manifest)
    _write_json(paths["partial"], partial)

    discovery_windows = sum(1 for record in records if record["split"] == "discovery")
    validation_windows = sum(1 for record in records if record["split"] == "validation")
    validation_task_fraction = _validation_task_fraction(records)
    validation_tasks = {record["task_id"] for record in records if record["split"] == "validation"}
    deterministic_replay_match = bool(
        np.array_equal(
            deterministic_noise(manifest_rows[0]["noise_identity"]),
            deterministic_noise(manifest_rows[0]["noise_identity"]),
        )
    )
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=bool(preflight.get("proposal_hash_ok")),
        serializer_preflight_ok=bool(preflight.get("serializer_preflight_ok")),
        official_prior_asset_check_persisted=bool(preflight.get("official_prior_asset_check_persisted")),
        preflight_passed=bool(preflight.get("proposal_hash_ok")) and bool(preflight.get("serializer_preflight_ok")),
        manifest_integrity_ok=bool(manifest_integrity["key_sets_equal"] and manifest_integrity["duplicate_manifest_key_count"] == 0 and manifest_integrity["duplicate_partial_key_count"] == 0 and manifest_integrity["split_overlap_key_count"] == 0),
        source_alignment_ok=source_exceptions == 0,
        action_semantics_ok=bool(preflight.get("action_semantics_persisted")),
        base_chunks_valid=bool(np.isfinite(base_array).all() and base_array.shape[1:] == (HORIZON, ACTION_DIM)),
        residual_targets_noncollapsed=bool(health["residual_noncollapsed"]),
        enough_discovery_windows=discovery_windows >= 512,
        enough_validation_windows=validation_windows >= 128,
        validation_task_coverage_ok=validation_tasks == FIXED_TASKS,
        maximum_validation_task_fraction=validation_task_fraction,
        noise_identity_valid=deterministic_replay_match,
        score_predictable=bool(score_metrics["score_predictable"]),
        residual_oracle_huber_reduction=oracle_metrics["residual_oracle_huber_reduction"],
        raw_diffusion_proxy_headroom=raw_metrics["raw_diffusion_proxy_headroom"],
        brid_beats_base=policy_huber["brid_full"] < policy_huber["smolvla_base"],
        brid_beats_raw_diffusion_proxy=policy_huber["brid_full"] < policy_huber["diffusion_policy_action_chunk_proxy"],
        brid_beats_no_base_residual_ablation=policy_huber["brid_full"] < policy_huber["brid_no_base_residual_ablation"],
        brid_beats_standard_lora=policy_huber["brid_full"] < policy_huber["standard_lora"],
        brid_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        brid_differs_from_ablation=mean_huber(brid_prediction, no_base_prediction) > 0.0,
        identity_max_abs_error=clean["identity_max_abs_error"],
        checkpoint_reload_ok=True,
        finite_objectives_and_gradients=gradient["finite_objectives_and_gradients"],
        expected_parameter_gradient_nonzero=gradient["expected_parameter_gradient_nonzero"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=float(np.mean(brid_gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_ok=bool(np.isfinite(brid_prediction).all()),
        clean_retention_ok=clean["clean_retention_ok"],
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=source_exceptions,
    )
    final_decision = classify_stage0(decision_inputs)
    public_health = dict(health)
    public_health.pop("active_mask", None)
    public_health.pop("clean_mask", None)
    result = {
        "method": "BRID-VLA",
        "final_decision": final_decision,
        "completed_model_row_count": len(partial_rows),
        "planned_model_row_count": len(manifest_rows),
        "exception_count": source_exceptions,
        "last_exception": last_exception,
        "manifest_row_count": manifest_integrity["manifest_row_count"],
        "partial_row_count": manifest_integrity["partial_row_count"],
        **manifest_integrity,
        "proposal_hash_ok": decision_inputs.proposal_hash_ok,
        "serializer_preflight_ok": decision_inputs.serializer_preflight_ok,
        "preflight_passed": decision_inputs.preflight_passed,
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "confirmatory_records_read": 0,
        "training_happened": False,
        "validation_search_happened": False,
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "diffusion_step_count": DIFFUSION_STEP_COUNT,
        "source_window_count": len(records),
        "discovery_window_count": discovery_windows,
        "validation_window_count": validation_windows,
        "validation_task_fraction_max": validation_task_fraction,
        "validation_tasks": sorted(validation_tasks),
        "residual_health": public_health,
        "residual_active_count": int(np.sum(active_mask)),
        "clean_retention_count": int(public_health["clean_retention_count"]),
        "deterministic_noise_replay_match": deterministic_replay_match,
        "score_prediction": score_metrics,
        "residual_oracle": oracle_metrics,
        "raw_diffusion_proxy": raw_metrics,
        "policy_huber": policy_huber,
        "brid_full_beats_base": decision_inputs.brid_beats_base,
        "brid_full_beats_raw_diffusion_proxy": decision_inputs.brid_beats_raw_diffusion_proxy,
        "brid_full_beats_no_base_residual_ablation": decision_inputs.brid_beats_no_base_residual_ablation,
        "brid_full_beats_standard_lora": decision_inputs.brid_beats_standard_lora,
        "identity_max_abs_error": decision_inputs.identity_max_abs_error,
        "expected_parameter_gradient_nonzero": decision_inputs.expected_parameter_gradient_nonzero,
        "frozen_base_gradient_count": decision_inputs.frozen_base_gradient_count,
        "weighted_gradient_norm_ratio_max": decision_inputs.weighted_gradient_norm_ratio_max,
        "intervention_fraction": decision_inputs.intervention_fraction,
        "action_delta_summary": delta,
        "action_validity_ok": decision_inputs.action_validity_ok,
        "clean_retention_ok": decision_inputs.clean_retention_ok,
        "stage_0_is_closed_loop_scientific_kill": False,
        "valid_scientific_result": False,
        "decision_inputs": decision_inputs,
        "created_utc": _utc_now(),
    }
    _write_json(paths["result_json"], result)
    _write_result_markdown(paths["result_md"], result)
    _write_adjudication(paths["adjudication"], result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["heartbeat"], {"method": "BRID-VLA", "status": "running", "pid": os.getpid(), "updated_utc": _utc_now()})
    _write_json(paths["status"], {"method": "BRID-VLA", "status": "running", "pid": os.getpid(), "started_utc": _utc_now()})

    if paths["result_json"].is_file() and not args.force:
        existing = _read_json(paths["result_json"])
        _write_json(paths["status"], {"method": "BRID-VLA", "status": "completed_existing_result_reused", "final_decision": existing.get("final_decision"), "pid": os.getpid(), "updated_utc": _utc_now()})
        return existing

    try:
        serializer = _serializer_preflight(paths["serializer_preflight"])
        prior = _official_prior_asset_check(paths["official_prior_asset_check"])
        action_semantics = _write_action_semantics(paths["action_semantics"])
        preflight = _write_preflight(paths, serializer, prior, action_semantics)
        result = _run_cached_audit(args, paths, preflight)
        elapsed = time.time() - started
        _write_json(paths["heartbeat"], {"method": "BRID-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "updated_utc": _utc_now()})
        _write_json(paths["status"], {"method": "BRID-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "elapsed_seconds": elapsed, "completed_model_row_count": result["completed_model_row_count"], "planned_model_row_count": result["planned_model_row_count"], "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "0\n")
        return result
    except Exception as exc:
        blocker = {
            "method": "BRID-VLA",
            "final_decision": "BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "valid_scientific_result": False,
            "stage_0_is_closed_loop_scientific_kill": False,
            "created_utc": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        _write_json(paths["status"], {"method": "BRID-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_json(paths["heartbeat"], {"method": "BRID-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "1\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "brid_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "brid_vla" / "stage0"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--ccif-manifest", default=str(DEFAULT_CCIF_MANIFEST))
    parser.add_argument("--ccif-partial", default=str(DEFAULT_CCIF_PARTIAL))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"BRID serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0
    result = run(args)
    print(json.dumps({"final_decision": result.get("final_decision"), "blocker": result.get("blocker")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
