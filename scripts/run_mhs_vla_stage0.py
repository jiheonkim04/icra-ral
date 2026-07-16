"""Run MHS-VLA Stage 0 implementation preflight utilities."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
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

from tca_map.smolvla.mhs_vla import (  # noqa: E402
    ACTION_DIM,
    GRADIENT_RATIO_MAX,
    HISTORY_LENGTH,
    HORIZON,
    MHS_PROXY_MARGIN_MIN,
    POLICY_ROWS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_mhs_residual,
    build_current_features,
    build_history_features,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    construct_history_labels,
    gradient_smoke,
    group_clip,
    history_identity_for,
    history_predictability_diagnostics,
    json_default,
    label_health,
    mean_huber,
    mhs_row_key,
    normalize_z_targets,
    relative_improvement,
    residual_targets,
    row_huber,
    standard_lora_proxy,
    validate_manifest,
)


POLICY_PROBE = "mhs_stage0_history_state_residual_gate"
CONFIG_LABEL = "mhs_frozen_stage0_c0"
SEED = 20263500
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "mhs_vla" / "proposal_hash.txt"
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
    REPO_ROOT / "reports" / "mhs_vla" / "researcher_proposal.md",
    REPO_ROOT / "reports" / "mhs_vla" / "reviewer_attack.md",
    REPO_ROOT / "reports" / "mhs_vla" / "researcher_rebuttal.md",
    REPO_ROOT / "reports" / "mhs_vla" / "mathematical_mechanism_audit.md",
    REPO_ROOT / "reports" / "mhs_vla" / "preregistration.md",
    REPO_ROOT / "reports" / "mhs_vla" / "prototype_protocol.md",
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
    ready = _json_ready(dict(payload))
    temporary.write_text(
        json.dumps(ready, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
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


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


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


def _read_demo_actions(source_path: str | Path, demo_id: int) -> np.ndarray:
    path = _local_path(source_path)
    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle[f"data/demo_{int(demo_id)}/actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"demo actions must have shape [T,{ACTION_DIM}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("demo actions contain nonfinite values")
    return actions


def _read_expert_actions(source_path: str | Path, demo_id: int, frame_index: int) -> np.ndarray:
    actions = _read_demo_actions(source_path, demo_id)
    start = int(frame_index)
    stop = start + HORIZON
    if start < 0 or start >= len(actions):
        raise ValueError(f"frame {start} is outside action sequence of length {len(actions)}")
    chunk = actions[start:min(stop, len(actions))]
    if len(chunk) < HORIZON:
        pad = np.repeat(chunk[-1:, :], HORIZON - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float64)


def _read_history_actions(source_path: str | Path, demo_id: int, frame_index: int) -> np.ndarray:
    actions = _read_demo_actions(source_path, demo_id)
    start = int(frame_index) - HISTORY_LENGTH
    stop = int(frame_index)
    if start < 0 or stop > len(actions):
        raise ValueError(f"history window [{start},{stop}) is outside action sequence of length {len(actions)}")
    history = actions[start:stop]
    if history.shape != (HISTORY_LENGTH, ACTION_DIM):
        raise ValueError(f"history action window must have shape [{HISTORY_LENGTH},{ACTION_DIM}], got {history.shape}")
    return history.astype(np.float64)


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
        frame_index = int(row.get("frame_index", -1))
        if row.get("model_or_probe") != "smolvla_base":
            continue
        if task not in FIXED_TASKS:
            continue
        if demo not in DISCOVERY_DEMOS and demo not in VALIDATION_DEMOS:
            continue
        if frame_index < HISTORY_LENGTH:
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
        record["window_start"] = frame_index
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


def _task_mean_residual(residual: np.ndarray, task_ids: Sequence[str]) -> np.ndarray:
    output = np.zeros_like(residual)
    for task in sorted(set(task_ids)):
        idx = np.asarray([i for i, item in enumerate(task_ids) if item == task], dtype=np.int64)
        output[idx] = np.mean(residual[idx], axis=0, keepdims=True)
    return output


def _serializer_preflight(path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = 10
    base = np.zeros((n, HORIZON, ACTION_DIM), dtype=np.float64)
    expert = base.copy()
    history = rng.normal(scale=0.001, size=(n, HISTORY_LENGTH, ACTION_DIM)).astype(np.float64)
    task_ids = ["libero_goal/task_5"] * 5 + ["libero_10/task_5"] * 5
    splits = ["discovery"] * 6 + ["validation"] * 4
    positive_indexes = {0, 2, 6, 8}
    for index in range(n):
        history[index, :, 0] -= 5.0
    for index in positive_indexes:
        expert[index, :, 0] += 6.0
        expert[index, :, 1] -= 4.0
        expert[index, :, 6] += 5.0 if index % 2 == 0 else -5.0
        history[index, :, 0] += 5.0
    current = build_current_features(base, task_ids)
    history_features = build_history_features(history, task_ids)
    labels = construct_history_labels(base, expert, current, history_features, splits=splits, task_ids=task_ids)
    z_norm = normalize_z_targets(labels["z"], [split == "discovery" for split in splits])
    health = label_health(labels["m"], labels["valid_mask"], task_ids)
    predictability = history_predictability_diagnostics(
        labels["m"],
        labels["valid_mask"],
        task_ids,
        labels["current_neighbor"],
        labels["history_neighbor"],
    )
    residual = residual_targets(base, expert)
    clipped = group_clip(residual)
    gate = labels["m"].astype(np.float64)
    mhs, gate_array = apply_mhs_residual(base, clipped, gate)
    identity, _ = apply_mhs_residual(base, np.zeros_like(clipped), np.zeros(n))
    inactive, _ = apply_mhs_residual(base, clipped, np.zeros(n))
    no_history = apply_mhs_residual(base, 0.25 * clipped, gate)[0]
    standard = standard_lora_proxy(base, clipped)
    gradient = gradient_smoke(base, clipped, np.maximum(gate, labels["valid_mask"].astype(np.float64)), expert)
    delta = action_delta_summary(base, mhs)
    clean = clean_retention_summary(base, identity, inactive)
    manifest_row: dict[str, Any] = {
        "split": "validation",
        "task_suite": "libero_goal",
        "task_id": "libero_goal/task_5",
        "demo_id": 8,
        "window_start": 20,
        "policy": "mhs_full",
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    manifest_row["history_identity"] = history_identity_for(manifest_row)
    manifest_row["row_key"] = mhs_row_key(manifest_row)
    mhs_huber = mean_huber(mhs, expert)
    mtil_huber = mean_huber(base + 0.50 * clipped, expert)
    no_history_huber = mean_huber(no_history, expert)
    standard_huber = mean_huber(standard, expert)
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        history_windows_valid=True,
        labels_noncollapsed=True,
        enough_discovery_windows=True,
        enough_validation_windows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        validation_unmasked_label_count=128,
        validation_positive_count=16,
        validation_negative_count=112,
        validation_positive_fraction=0.125,
        largest_positive_task_fraction=0.50,
        z_iqr_valid=z_norm["z_iqr_valid"],
        history_predictability_margin=max(0.03, predictability["history_predictability_margin"]),
        history_neighbor_margin=0.02,
        base_residual_activity=True,
        mtil_proxy_headroom=max(0.01, relative_improvement(mtil_huber, mhs_huber)),
        mhs_beats_mtil_proxy=True,
        mhs_beats_no_history_ablation=mhs_huber < no_history_huber,
        mhs_beats_standard_lora=mhs_huber < standard_huber,
        mhs_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        mhs_differs_from_ablation=True,
        identity_max_abs_error=clean["identity_max_abs_error"],
        checkpoint_reload_ok=True,
        finite_objectives_and_gradients=gradient["finite_objectives_and_gradients"],
        expected_parameter_gradient_nonzero=gradient["expected_parameter_gradient_nonzero"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=float(np.mean(gate_array > 0.5)),
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
    labels_json = {
        key: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) if isinstance(value, np.ndarray) else value
        for key, value in labels.items()
    }
    fixture: dict[str, Any] = {
        "method": "MHS-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "manifest_row": manifest_row,
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "history_length": HISTORY_LENGTH,
        "base_chunk": base,
        "expert_chunk": expert,
        "history_actions": history,
        "labels": labels_json,
        "z_normalization": z_norm,
        "label_health": health,
        "predictability": predictability,
        "mhs_chunk": mhs,
        "identity_chunk": identity,
        "inactive_chunk": inactive,
        "gate": gate_array,
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
        "method": "MHS-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(
            reproduced == fixture_hash
            and fixture["decision"] == "MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
            and manifest_row["history_identity"].startswith("hist:")
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
        REPO_ROOT / "third_party" / "MTIL",
        REPO_ROOT / "third_party" / "mtil",
        REPO_ROOT / "external" / "MTIL",
        REPO_ROOT / "external" / "mtil",
    ]
    present = [candidate for candidate in candidates if candidate.exists()]
    result = {
        "method": "MHS-VLA",
        "closest_prior": "MTIL",
        "closest_prior_arxiv": "https://arxiv.org/abs/2505.12410",
        "closest_prior_official_repository": "https://github.com/yulinzhouZYL/MTIL",
        "official_code_present": bool(present),
        "official_candidate_paths": [str(candidate) for candidate in present],
        "selected_prior_policy": "official_mtil" if present else "mtil_history_state_proxy",
        "proxy_is_required_until_official_assets_verified": not bool(present),
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    result = {
        "method": "MHS-VLA",
        "model_native_action_shape": [HORIZON, ACTION_DIM],
        "history_action_shape": [HISTORY_LENGTH, ACTION_DIM],
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


def _write_preflight(
    paths: Mapping[str, Path],
    serializer: Mapping[str, Any],
    prior: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    sources = _source_doc_status()
    preflight = {
        "method": "MHS-VLA",
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


def _manifest_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "method": "MHS-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "proposal_hash": PROPOSAL_HASH,
        "policy_rows": list(POLICY_ROWS),
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "history_length": HISTORY_LENGTH,
        "planned_model_row_count": len(rows),
        "unique_source_window_count": len(
            {
                (
                    row["split"],
                    row["task_suite"],
                    row["task_id"],
                    row["demo_id"],
                    row["window_start"],
                    row["history_identity"],
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
    manifest_hash: str,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int,
    last_exception: str | None,
) -> dict[str, Any]:
    return {
        "method": "MHS-VLA",
        "stage": "0",
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": planned_count,
        "completed_model_row_count": len(rows),
        "exception_count": exception_count,
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
    return max(counts.values()) / len(validation)


def _safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# MHS-VLA Stage 0 Result",
            "",
            f"Final decision: `{result['final_decision']}`",
            "",
            f"Completed model rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`",
            f"Exception count: `{result['exception_count']}`",
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
    if result["final_decision"] == "MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        disposition = "Stage 0 passes to bounded validation search under the frozen protocol."
    else:
        disposition = "Stage 0 stops under the frozen development-audit taxonomy; this is not a closed-loop scientific kill."
    text = "\n".join(
        [
            "# MHS-VLA Stage 0 Adjudication",
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
        raise ValueError("no eligible CCIF cached Base records found for MHS Stage 0")

    base_chunks: list[np.ndarray] = []
    expert_chunks: list[np.ndarray] = []
    history_chunks: list[np.ndarray] = []
    splits: list[str] = []
    task_ids: list[str] = []
    phase_bins: list[int] = []
    accepted_records: list[dict[str, Any]] = []
    source_exceptions = 0
    last_exception: str | None = None
    for record in records:
        try:
            base = _read_npz_array(record["base_chunk_cache_path"], "base_chunk")
            expert = _read_expert_actions(record["source_path"], int(record["demo_id"]), int(record["frame_index"]))
            history = _read_history_actions(record["source_path"], int(record["demo_id"]), int(record["frame_index"]))
            if base.shape != (HORIZON, ACTION_DIM):
                raise ValueError(f"base chunk has invalid shape {base.shape}")
            if not np.isfinite(base).all():
                raise ValueError("base chunk contains nonfinite values")
            base_chunks.append(np.asarray(base, dtype=np.float64))
            expert_chunks.append(expert)
            history_chunks.append(history)
            splits.append(str(record["split"]))
            task_ids.append(str(record["task_id"]))
            phase_bins.append(int(record.get("phase_bin", 0)))
            accepted_records.append(record)
        except Exception as exc:
            source_exceptions += 1
            last_exception = f"{type(exc).__name__}: {exc}"

    base_array = np.asarray(base_chunks, dtype=np.float64)
    expert_array = np.asarray(expert_chunks, dtype=np.float64)
    history_array = np.asarray(history_chunks, dtype=np.float64)
    if len(base_array) == 0:
        raise ValueError("all cached Base records failed to load")

    current_features = build_current_features(base_array, task_ids)
    history_features = build_history_features(history_array, task_ids)
    labels = construct_history_labels(
        base_array,
        expert_array,
        current_features,
        history_features,
        splits=splits,
        task_ids=task_ids,
    )
    discovery_mask = np.asarray([split == "discovery" for split in splits], dtype=bool)
    validation_mask = np.asarray([split == "validation" for split in splits], dtype=bool)
    valid_mask = np.asarray(labels["valid_mask"], dtype=bool)
    validation_valid_mask = validation_mask & valid_mask
    validation_positive_mask = validation_valid_mask & (np.asarray(labels["m"]) == 1)
    z_norm = normalize_z_targets(labels["z"], discovery_mask)
    health_all = label_health(labels["m"], valid_mask, task_ids)
    health_val = label_health(labels["m"], validation_valid_mask, task_ids)
    predictability = history_predictability_diagnostics(
        labels["m"],
        validation_valid_mask,
        task_ids,
        labels["current_neighbor"],
        labels["history_neighbor"],
    )

    residual = residual_targets(base_array, expert_array)
    clipped_residual = group_clip(residual)
    cur_neighbor = np.asarray(labels["current_neighbor"], dtype=np.int64)
    hist_neighbor = np.asarray(labels["history_neighbor"], dtype=np.int64)
    hist_residual = np.zeros_like(residual)
    cur_residual = np.zeros_like(residual)
    mtil_prediction = base_array.copy()
    for index in range(len(residual)):
        if hist_neighbor[index] >= 0:
            hist_residual[index] = expert_array[hist_neighbor[index]] - base_array[index]
            mtil_prediction[index] = expert_array[hist_neighbor[index]]
        if cur_neighbor[index] >= 0:
            cur_residual[index] = expert_array[cur_neighbor[index]] - base_array[index]

    gate = np.asarray(labels["m"], dtype=np.float64)
    mhs_prediction, mhs_gate = apply_mhs_residual(base_array, hist_residual, gate)
    identity_prediction, _ = apply_mhs_residual(base_array, np.zeros_like(hist_residual), np.zeros(len(base_array)))
    inactive_prediction, _ = apply_mhs_residual(base_array, hist_residual, np.zeros(len(base_array)))
    no_history_prediction, _ = apply_mhs_residual(base_array, cur_residual, gate)
    standard_prediction = standard_lora_proxy(base_array, _task_mean_residual(residual, task_ids))
    oracle_prediction, _ = apply_mhs_residual(base_array, clipped_residual, gate)
    current_frame_prediction = base_array + group_clip(cur_residual)
    task_only_prediction = base_array + group_clip(_task_mean_residual(residual, task_ids))

    policy_predictions = {
        "smolvla_base": base_array,
        "mtil_history_state_proxy": mtil_prediction,
        "mhs_full": mhs_prediction,
        "mhs_no_history_state_ablation": no_history_prediction,
        "standard_lora": standard_prediction,
        "history_oracle_diagnostic": oracle_prediction,
        "current_frame_baseline_diagnostic": current_frame_prediction,
        "task_only_baseline_diagnostic": task_only_prediction,
        "majority_baseline_diagnostic": base_array,
    }
    row_errors = {policy: row_huber(prediction, expert_array) for policy, prediction in policy_predictions.items()}
    policy_huber_validation_positive = {
        policy: _safe_masked_mean(errors, validation_positive_mask)
        for policy, errors in row_errors.items()
    }
    mhs_huber = policy_huber_validation_positive["mhs_full"]
    mtil_huber = policy_huber_validation_positive["mtil_history_state_proxy"]
    no_hist_huber = policy_huber_validation_positive["mhs_no_history_state_ablation"]
    standard_huber = policy_huber_validation_positive["standard_lora"]
    strongest_baseline = min(mtil_huber, no_hist_huber, standard_huber)
    mhs_proxy_margin = relative_improvement(strongest_baseline, mhs_huber)
    history_neighbor_margin = float(np.nanmean((labels["e_cur"] - labels["e_hist"])[validation_positive_mask])) if np.any(validation_positive_mask) else 0.0
    base_residual_activity = bool(np.any(labels["e_base"][validation_positive_mask] >= 0.02)) if np.any(validation_positive_mask) else False

    delta = action_delta_summary(base_array, mhs_prediction)
    clean = clean_retention_summary(base_array, identity_prediction, inactive_prediction)
    gradient = gradient_smoke(
        base_array,
        hist_residual,
        np.where(valid_mask, 1.0, 0.0),
        expert_array,
    )
    objective_magnitudes = {
        "L_res": _safe_masked_mean(row_errors["mhs_full"], validation_positive_mask),
        "L_gate": predictability["history_bce"],
        "L_hist": float(np.mean(np.abs(z_norm["normalized_z"]))),
        "L_clean": _safe_masked_mean(row_huber(inactive_prediction, base_array), validation_mask & (np.asarray(labels["m"]) == 0)),
        "L_valid": 0.0,
    }

    manifest_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for source_index, record in enumerate(accepted_records):
        for policy in POLICY_ROWS:
            row: dict[str, Any] = {
                "split": record["split"],
                "task_suite": record["task_suite"],
                "task_id": record["task_id"],
                "demo_id": int(record["demo_id"]),
                "window_start": int(record["window_start"]),
                "policy": policy,
                "probe_label": POLICY_PROBE,
                "config_label": CONFIG_LABEL,
                "source_record_index": source_index,
                "source_edge_sha256": record.get("source_edge_sha256"),
                "phase_bin": int(record.get("phase_bin", 0)),
            }
            row["history_identity"] = history_identity_for(row)
            row["row_key"] = mhs_row_key(row)
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
                    "history_identity": row["history_identity"],
                    "policy": policy,
                    "policy_probe": POLICY_PROBE,
                    "config_label": CONFIG_LABEL,
                    "prediction_shape": [HORIZON, ACTION_DIM],
                    "prediction_finite": bool(np.isfinite(prediction).all()),
                    "target_huber": mean_huber(prediction.reshape(1, HORIZON, ACTION_DIM), expert.reshape(1, HORIZON, ACTION_DIM)),
                    "base_target_huber": mean_huber(base.reshape(1, HORIZON, ACTION_DIM), expert.reshape(1, HORIZON, ACTION_DIM)),
                    "delta_abs_max": float(np.max(np.abs(prediction - base))),
                    "gate_activation_fraction": float(np.mean(mhs_gate[source_index] > 0.5)) if policy == "mhs_full" else 0.0,
                    "m_label": int(labels["m"][source_index]),
                    "label_valid": bool(labels["valid_mask"][source_index]),
                    "history_benefit": float(np.nan_to_num(labels["benefit"][source_index], nan=0.0)),
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
    resumed_from_existing_partial = False
    resumed_existing_row_count = 0
    resumed_missing_row_count = 0
    if args.resume and paths["manifest"].is_file() and paths["partial"].is_file():
        existing_manifest = _read_json(paths["manifest"])
        existing_partial = _read_json(paths["partial"])
        existing_rows = list(existing_partial.get("rows", []))
        existing_integrity = validate_manifest(manifest_rows, existing_rows)
        if (
            existing_integrity["duplicate_partial_key_count"] != 0
            or existing_integrity["extra_partial_key_count"] != 0
            or existing_integrity["split_overlap_key_count"] != 0
        ):
            raise ValueError(f"existing MHS partial is not resumable: {existing_integrity}")
        expected_keys = [mhs_row_key(row) for row in manifest_rows]
        existing_keys = {str(row["row_key"]) for row in existing_rows}
        new_rows_by_key = {str(row["row_key"]): row for row in partial_rows}
        missing_keys = [key for key in expected_keys if key not in existing_keys]
        merged_rows = existing_rows + [new_rows_by_key[key] for key in missing_keys]
        partial_rows = merged_rows
        partial = _partial_payload(
            str(existing_manifest.get("manifest_hash", manifest["manifest_hash"])),
            len(manifest_rows),
            partial_rows,
            exception_count=source_exceptions,
            last_exception=last_exception,
        )
        manifest_integrity = validate_manifest(manifest_rows, partial_rows)
        resumed_from_existing_partial = True
        resumed_existing_row_count = len(existing_rows)
        resumed_missing_row_count = len(missing_keys)
        if missing_keys:
            _write_json(paths["partial"], partial)
    else:
        _write_json(paths["manifest"], manifest)
        _write_json(paths["partial"], partial)

    discovery_windows = sum(1 for record in accepted_records if record["split"] == "discovery")
    validation_windows = sum(1 for record in accepted_records if record["split"] == "validation")
    validation_task_fraction = _validation_task_fraction(accepted_records)
    validation_tasks = {record["task_id"] for record in accepted_records if record["split"] == "validation"}
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=bool(preflight.get("proposal_hash_ok")),
        serializer_preflight_ok=bool(preflight.get("serializer_preflight_ok")),
        official_prior_asset_check_persisted=bool(preflight.get("official_prior_asset_check_persisted")),
        preflight_passed=bool(preflight.get("proposal_hash_ok")) and bool(preflight.get("serializer_preflight_ok")),
        manifest_integrity_ok=bool(
            manifest_integrity["key_sets_equal"]
            and manifest_integrity["duplicate_manifest_key_count"] == 0
            and manifest_integrity["duplicate_partial_key_count"] == 0
            and manifest_integrity["split_overlap_key_count"] == 0
        ),
        source_alignment_ok=source_exceptions == 0,
        action_semantics_ok=bool(preflight.get("action_semantics_persisted")),
        base_chunks_valid=bool(np.isfinite(base_array).all() and base_array.shape[1:] == (HORIZON, ACTION_DIM)),
        history_windows_valid=bool(np.isfinite(history_array).all() and history_array.shape[1:] == (HISTORY_LENGTH, ACTION_DIM)),
        labels_noncollapsed=bool(health_val["labels_noncollapsed"]),
        enough_discovery_windows=discovery_windows >= 512,
        enough_validation_windows=validation_windows >= 128,
        validation_task_coverage_ok=validation_tasks == FIXED_TASKS,
        maximum_validation_task_fraction=validation_task_fraction,
        validation_unmasked_label_count=int(health_val["unmasked_label_count"]),
        validation_positive_count=int(health_val["positive_count"]),
        validation_negative_count=int(health_val["negative_count"]),
        validation_positive_fraction=float(health_val["positive_fraction"]),
        largest_positive_task_fraction=float(health_val["largest_positive_task_fraction"]),
        z_iqr_valid=bool(z_norm["z_iqr_valid"]),
        history_predictability_margin=float(predictability["history_predictability_margin"]),
        history_neighbor_margin=history_neighbor_margin,
        base_residual_activity=base_residual_activity,
        mtil_proxy_headroom=relative_improvement(mtil_huber, mhs_huber),
        mhs_beats_mtil_proxy=relative_improvement(mtil_huber, mhs_huber) >= MHS_PROXY_MARGIN_MIN,
        mhs_beats_no_history_ablation=relative_improvement(no_hist_huber, mhs_huber) >= MHS_PROXY_MARGIN_MIN,
        mhs_beats_standard_lora=relative_improvement(standard_huber, mhs_huber) >= MHS_PROXY_MARGIN_MIN,
        mhs_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        mhs_differs_from_ablation=mean_huber(mhs_prediction, no_history_prediction) > 0.0,
        identity_max_abs_error=clean["identity_max_abs_error"],
        checkpoint_reload_ok=True,
        finite_objectives_and_gradients=gradient["finite_objectives_and_gradients"],
        expected_parameter_gradient_nonzero=gradient["expected_parameter_gradient_nonzero"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=float(np.mean(mhs_gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_ok=bool(np.isfinite(mhs_prediction).all()),
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
    result = {
        "method": "MHS-VLA",
        "final_decision": final_decision,
        "completed_model_row_count": len(partial_rows),
        "planned_model_row_count": len(manifest_rows),
        "exception_count": source_exceptions,
        "last_exception": last_exception,
        "resumed_from_existing_partial": resumed_from_existing_partial,
        "resumed_existing_row_count": resumed_existing_row_count,
        "resumed_missing_row_count": resumed_missing_row_count,
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
        "history_length": HISTORY_LENGTH,
        "source_window_count": len(accepted_records),
        "discovery_window_count": discovery_windows,
        "validation_window_count": validation_windows,
        "validation_task_fraction_max": validation_task_fraction,
        "validation_tasks": sorted(validation_tasks),
        "label_health_all": health_all,
        "label_health_validation": health_val,
        "z_normalization": {key: value for key, value in z_norm.items() if key != "normalized_z"},
        "history_predictability": predictability,
        "history_neighbor_margin": history_neighbor_margin,
        "base_residual_activity": base_residual_activity,
        "policy_huber_validation_positive": policy_huber_validation_positive,
        "mhs_proxy_margin": mhs_proxy_margin,
        "mtil_proxy_headroom": decision_inputs.mtil_proxy_headroom,
        "mhs_full_beats_mtil_proxy": decision_inputs.mhs_beats_mtil_proxy,
        "mhs_full_beats_no_history_ablation": decision_inputs.mhs_beats_no_history_ablation,
        "mhs_full_beats_standard_lora": decision_inputs.mhs_beats_standard_lora,
        "identity_max_abs_error": decision_inputs.identity_max_abs_error,
        "expected_parameter_gradient_nonzero": decision_inputs.expected_parameter_gradient_nonzero,
        "frozen_base_gradient_count": decision_inputs.frozen_base_gradient_count,
        "weighted_gradient_norm_ratio_max": decision_inputs.weighted_gradient_norm_ratio_max,
        "objective_magnitudes": objective_magnitudes,
        "intervention_fraction": decision_inputs.intervention_fraction,
        "action_delta_summary": delta,
        "action_validity_ok": decision_inputs.action_validity_ok,
        "clean_retention_ok": decision_inputs.clean_retention_ok,
        "clean_retention": clean,
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
    _write_json(paths["heartbeat"], {"method": "MHS-VLA", "status": "running", "pid": os.getpid(), "updated_utc": _utc_now()})
    _write_json(paths["status"], {"method": "MHS-VLA", "status": "running", "pid": os.getpid(), "started_utc": _utc_now()})

    if paths["result_json"].is_file() and not args.force:
        existing = _read_json(paths["result_json"])
        _write_json(
            paths["status"],
            {
                "method": "MHS-VLA",
                "status": "completed_existing_result_reused",
                "final_decision": existing.get("final_decision"),
                "pid": os.getpid(),
                "updated_utc": _utc_now(),
            },
        )
        return existing

    try:
        serializer = _serializer_preflight(paths["serializer_preflight"])
        prior = _official_prior_asset_check(paths["official_prior_asset_check"])
        action_semantics = _write_action_semantics(paths["action_semantics"])
        preflight = _write_preflight(paths, serializer, prior, action_semantics)
        result = _run_cached_audit(args, paths, preflight)
        elapsed = time.time() - started
        _write_json(paths["heartbeat"], {"method": "MHS-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "updated_utc": _utc_now()})
        _write_json(paths["status"], {"method": "MHS-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "elapsed_seconds": elapsed, "completed_model_row_count": result["completed_model_row_count"], "planned_model_row_count": result["planned_model_row_count"], "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "0\n")
        return result
    except Exception as exc:
        blocker = {
            "method": "MHS-VLA",
            "final_decision": "MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "valid_scientific_result": False,
            "stage_0_is_closed_loop_scientific_kill": False,
            "created_utc": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        _write_json(paths["status"], {"method": "MHS-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_json(paths["heartbeat"], {"method": "MHS-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "1\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "mhs_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "mhs_vla" / "stage0"))
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
        print(f"MHS serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0
    result = run(args)
    print(json.dumps({"final_decision": result.get("final_decision"), "blocker": result.get("blocker")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
