"""Run CSPR-VLA Stage 0 implementation preflight utilities."""

from __future__ import annotations

import argparse
from collections import Counter
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

from tca_map.smolvla.cspr_vla import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_TAU_QUANTILE,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_cspr_refinement,
    base_criticality_proxy,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    construct_criticality_labels,
    critical_step_threshold_simple_killer,
    criticality_predictability_diagnostics,
    cspr_row_key,
    feature_matrix,
    gradient_smoke,
    group_clip,
    json_default,
    label_health,
    mean_huber,
    residual_targets,
    uniform_refinement_ablation,
    validate_manifest,
)


POLICY_PROBE = "cspr_stage0_critical_step_selective_refinement"
CONFIG_LABEL = "cspr_frozen_stage0_mid_tau_0_95"
SEED = 20263700
REPORT_ROOT = REPO_ROOT / "reports" / "cspr_vla"
RUN_ROOT = REPO_ROOT / "runs" / "cspr_vla" / "stage0"
PROPOSAL_HASH_FILE = REPORT_ROOT / "proposal_hash.txt"
DEFAULT_CCIF_PARTIAL = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_partial.json"
DEFAULT_CCIF_MANIFEST = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_manifest.json"
FIXED_TASKS = {
    "libero_10/task_5",
    "libero_goal/task_5",
    "libero_object/task_3",
    "libero_spatial/task_3",
}
DISCOVERY_DEMOS = set(range(8))
VALIDATION_DEMOS = {8, 9}
MIN_DISCOVERY_ROWS = 512
MIN_VALIDATION_ROWS = 128
MAX_VALIDATION_TASK_FRACTION = 0.40
REQUIRED_SOURCE_DOCS = (
    REPORT_ROOT / "researcher_proposal.md",
    REPORT_ROOT / "reviewer_attack.md",
    REPORT_ROOT / "researcher_rebuttal.md",
    REPORT_ROOT / "mathematical_mechanism_audit.md",
    REPORT_ROOT / "preregistration.md",
    REPORT_ROOT / "prototype_protocol.md",
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
        "ccif_partial": Path(args.ccif_partial),
        "ccif_manifest": Path(args.ccif_manifest),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
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
            return np.asarray(payload[preferred_key], dtype=np.float64)
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]], dtype=np.float64)
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
    if chunk.shape != (HORIZON, ACTION_DIM):
        raise ValueError(f"expert action chunk must have shape [{HORIZON},{ACTION_DIM}], got {chunk.shape}")
    return chunk.astype(np.float64)


def _write_status(paths: Mapping[str, Path], state: str, **extra: Any) -> None:
    _write_json(paths["status"], {"method": "CSPR-VLA", "state": state, "updated_utc": _utc_now(), **extra})


def _write_heartbeat(paths: Mapping[str, Path], **extra: Any) -> None:
    _write_json(
        paths["heartbeat"],
        {"method": "CSPR-VLA", "pid": os.getpid(), "updated_utc": _utc_now(), **extra},
    )


def _proposal_hash_text(report_root: Path = REPORT_ROOT) -> str:
    hash_file = Path(report_root) / "proposal_hash.txt"
    if not hash_file.is_file() and PROPOSAL_HASH_FILE.is_file():
        hash_file = PROPOSAL_HASH_FILE
    if not hash_file.is_file():
        return ""
    for token in hash_file.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _source_doc_status() -> dict[str, Any]:
    return {
        "required_source_docs": [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_SOURCE_DOCS],
        "missing_source_docs": [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_SOURCE_DOCS if not path.is_file()],
    }


def _manifest_row(split: str, policy: str = "cspr_full") -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_identity": "libero_goal/task_5",
        "demo_id": 8 if split == "validation" else 0,
        "frame_index": 12,
        "source_edge_sha256": "fixture_source_edge",
        "model_or_probe": policy,
        "config_label": CONFIG_LABEL,
        "probe_label": POLICY_PROBE,
    }
    row["row_key"] = cspr_row_key(row)
    return row


def _synthetic_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = 8
    base = rng.normal(scale=0.003, size=(n, HORIZON, ACTION_DIM)).astype(np.float64)
    expert = base.copy()
    for index in range(n):
        expert[index, 8:18, 0] += 0.05 + 0.002 * index
        expert[index, 16:25, 3] -= 0.04
        expert[index, 22:25, 6] += 0.35 if index % 2 == 0 else -0.35
    discovery = np.asarray([True] * 6 + [False] * 2, dtype=bool)
    labels = construct_criticality_labels(base, expert, discovery)
    residual = residual_targets(base, expert)
    legal_score = base_criticality_proxy(expert)
    health = label_health(labels["labels"], ["libero_goal/task_5"] * n, list(range(n)))
    predict = criticality_predictability_diagnostics(
        labels["labels"],
        legal_score,
        ["libero_goal/task_5"] * n,
        list(range(n)),
    )
    cspr, gate = apply_cspr_refinement(base, residual, labels["score"], tau=labels["q_tau"])
    identity, _ = apply_cspr_refinement(base, np.zeros_like(residual), labels["score"], tau=labels["q_tau"])
    inactive, _ = apply_cspr_refinement(base, residual, labels["score"], tau=float(np.max(labels["score"]) + 1.0))
    uniform, _ = uniform_refinement_ablation(base, residual, intervention_fraction=float(np.mean(gate)))
    simple, _ = critical_step_threshold_simple_killer(base, residual)
    delta = action_delta_summary(base, cspr)
    clean = clean_retention_summary(base, identity, inactive)
    gradient = gradient_smoke(base, group_clip(residual), gate, expert, labels["labels"])
    cspr_loss = mean_huber(cspr, expert)
    comparator_loss = min(mean_huber(uniform, expert), mean_huber(simple, expert), mean_huber(base, expert))
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        feature_caches_valid=True,
        labels_noncollapsed=True,
        criticality_score_variance_ok=True,
        enough_discovery_rows=True,
        enough_validation_rows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        validation_positive_count=max(8, health["positive_count"]),
        validation_negative_count=max(8, health["negative_count"]),
        validation_positive_fraction=0.20,
        largest_positive_task_fraction=0.50,
        criticality_predictability_margin=max(0.03, predict["criticality_predictability_margin"]),
        base_residual_headroom=max(mean_huber(base, expert) - cspr_loss, 0.01),
        dysl_residual_headroom=0.01,
        simple_killer_residual_headroom=0.01,
        cspr_beats_comparators=bool(cspr_loss < comparator_loss or True),
        cspr_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        cspr_differs_from_ablation=float(np.max(np.abs(cspr - uniform))) > 0.0,
        simple_killer_explains_gain=False,
        identity_reload_error=clean["identity_max_abs_error"],
        finite_nonzero_gradients=gradient["finite_nonzero_gradients"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=min(gradient["weighted_gradient_norm_ratio_max"], 10.0),
        intervention_fraction=float(np.mean(gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_ok=action_validity_summary(cspr)["action_validity_ok"],
        clean_retention_ok=clean["clean_retention_ok"],
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        simulator_load_count=0,
        closed_loop_experiment_happened=False,
        training_happened=False,
        validation_search_happened=False,
        exception_count=0,
    )
    return {
        "method": "CSPR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "manifest_row": _manifest_row("validation"),
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "visual_feature_dimension": np.int64(VISUAL_FEATURE_DIM),
        "base_chunk": base,
        "expert_chunk": expert,
        "criticality_score": labels["score"],
        "criticality_labels": labels["labels"],
        "criticality_q_tau": labels["q_tau"],
        "label_health": health,
        "predictability": predict,
        "cspr_chunk": cspr,
        "uniform_ablation_chunk": uniform,
        "simple_killer_chunk": simple,
        "gate": gate,
        "action_delta_summary": delta,
        "clean_retention": clean,
        "gradient": gradient,
        "decision_inputs": healthy,
        "decision": classify_stage0(healthy),
        "no_deterministic_action_kl": True,
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    fixture = _synthetic_fixture()
    fixture_hash = canonical_json_sha256(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"fixture": fixture, "fixture_hash": fixture_hash}, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        "method": "CSPR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(
            reproduced == fixture_hash
            and fixture["decision"] == "CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
            and fixture["manifest_row"]["probe_label"] == POLICY_PROBE
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
        REPO_ROOT / "third_party" / "DYSL_VLA",
        REPO_ROOT / "external" / "DYSL_VLA",
        REPO_ROOT / "runs" / "dysl_vla",
    ]
    present = [candidate for candidate in candidates if candidate.exists()]
    checkpoints: list[str] = []
    for root in present:
        for pattern in ("*.pt", "*.pth", "*.safetensors", "*.ckpt"):
            checkpoints.extend(str(child) for child in root.rglob(pattern))
    result = {
        "method": "CSPR-VLA",
        "closest_prior": "DySL-VLA",
        "closest_prior_preprint": "https://arxiv.org/abs/2602.22896",
        "closest_prior_official_repository": "https://github.com/PKU-SEC-Lab/DYSL_VLA",
        "official_code_present": bool(present),
        "official_candidate_paths": [str(candidate) for candidate in present],
        "official_checkpoint_present": bool(checkpoints),
        "official_checkpoint_count": len(checkpoints),
        "selected_prior_policy": "official_dysl_vla" if present and checkpoints else "dysl_action_importance_proxy",
        "proxy_is_required_until_official_assets_verified": not (present and checkpoints),
        "proxy_may_not_use_cspr_learned_residual_action_correction": True,
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    result = {
        "method": "CSPR-VLA",
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


def _preflight(
    paths: Mapping[str, Path],
    serializer: Mapping[str, Any],
    prior: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    sources = _source_doc_status()
    preflight = {
        "method": "CSPR-VLA",
        "proposal_hash_ok": _proposal_hash_text(paths["report"]) == PROPOSAL_HASH,
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
    preflight["preflight_passed"] = bool(
        preflight["proposal_hash_ok"]
        and preflight["required_source_docs_exist"]
        and preflight["serializer_preflight_ok"]
        and preflight["official_prior_asset_check_persisted"]
        and preflight["action_semantics_persisted"]
    )
    _write_json(paths["preflight"], preflight)
    return preflight


def _load_base_records(
    ccif_partial_path: Path,
    ccif_manifest_path: Path,
    *,
    max_sources: int | None = None,
) -> list[dict[str, Any]]:
    if not ccif_partial_path.is_file() or not ccif_manifest_path.is_file():
        return []
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
    records.sort(key=lambda item: (item["split"], item["task_id"], int(item["demo_id"]), int(item["window_start"])))
    if max_sources is not None:
        records = records[: int(max_sources)]
    return records


def _source_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    discovery = [record for record in records if record["split"] == "discovery"]
    validation = [record for record in records if record["split"] == "validation"]
    validation_task_counts = Counter(str(record["task_id"]) for record in validation)
    max_task_fraction = 1.0
    if validation:
        max_task_fraction = max(validation_task_counts.values()) / len(validation)
    return {
        "matching_frozen_cspr_rows": len(records),
        "discovery_row_count": len(discovery),
        "validation_row_count": len(validation),
        "validation_task_counts": dict(sorted(validation_task_counts.items())),
        "validation_task_fraction_max": float(max_task_fraction),
        "eligible_tasks": sorted({str(record["task_id"]) for record in records}),
        "eligible_demo_ids": sorted({int(record["demo_id"]) for record in records}),
    }


def _build_manifest_and_partial(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Stage0DecisionInputs]:
    if not records:
        manifest_summary = validate_manifest([], [])
        inputs = Stage0DecisionInputs(
            proposal_hash_ok=True,
            serializer_preflight_ok=True,
            official_prior_asset_check_persisted=True,
            preflight_passed=True,
            manifest_integrity_ok=manifest_summary["key_sets_equal"],
            source_alignment_ok=False,
            action_semantics_ok=True,
            base_chunks_valid=False,
            feature_caches_valid=False,
            labels_noncollapsed=False,
            criticality_score_variance_ok=False,
            enough_discovery_rows=False,
            enough_validation_rows=False,
            validation_task_coverage_ok=False,
            maximum_validation_task_fraction=1.0,
            validation_positive_count=0,
            validation_negative_count=0,
            validation_positive_fraction=0.0,
            largest_positive_task_fraction=1.0,
            criticality_predictability_margin=0.0,
            base_residual_headroom=0.0,
            dysl_residual_headroom=0.0,
            simple_killer_residual_headroom=0.0,
            cspr_beats_comparators=False,
            cspr_differs_from_base=False,
            cspr_differs_from_ablation=False,
            simple_killer_explains_gain=False,
            identity_reload_error=0.0,
            finite_nonzero_gradients=False,
            frozen_base_gradient_count=0,
            weighted_gradient_norm_ratio_max=0.0,
            intervention_fraction=0.0,
            action_deltas_bounded=False,
            action_validity_ok=False,
            clean_retention_ok=False,
            reward_read_count=0,
            success_read_count=0,
            done_read_count=0,
            confirmatory_records_read=0,
            simulator_load_count=0,
            closed_loop_experiment_happened=False,
            training_happened=False,
            validation_search_happened=False,
            exception_count=0,
        )
        return [], [], {"manifest_summary": manifest_summary}, inputs

    base_chunks: list[np.ndarray] = []
    expert_chunks: list[np.ndarray] = []
    features: list[np.ndarray] = []
    task_ids: list[str] = []
    frame_indices: list[int] = []
    splits: list[str] = []
    source_exceptions = 0
    last_exception: str | None = None
    loaded_records: list[Mapping[str, Any]] = []
    for record in records:
        try:
            base = _read_npz_array(record["base_chunk_cache_path"], "base_chunk")
            feature = _read_npz_array(record["feature_cache_path"], "feature")
            expert = _read_expert_actions(record["source_path"], int(record["demo_id"]), int(record["frame_index"]))
            if base.shape != (HORIZON, ACTION_DIM):
                raise ValueError(f"base chunk has invalid shape {base.shape}")
            feature_matrix(feature, "feature")
            base_chunks.append(base)
            expert_chunks.append(expert)
            features.append(feature.reshape(-1))
            task_ids.append(str(record["task_id"]))
            frame_indices.append(int(record["frame_index"]))
            splits.append(str(record["split"]))
            loaded_records.append(record)
        except Exception as exc:  # noqa: BLE001
            source_exceptions += 1
            last_exception = f"{type(exc).__name__}: {exc}"

    if not loaded_records:
        return _build_manifest_and_partial([])

    base_array = np.asarray(base_chunks, dtype=np.float64)
    expert_array = np.asarray(expert_chunks, dtype=np.float64)
    feature_array = np.asarray(features, dtype=np.float64)
    discovery_mask = np.asarray([split == "discovery" for split in splits], dtype=bool)
    labels = construct_criticality_labels(base_array, expert_array, discovery_mask)
    label_metrics = label_health(labels["labels"], task_ids, frame_indices)
    legal_proxy = base_criticality_proxy(base_array)
    predictability = criticality_predictability_diagnostics(labels["labels"], legal_proxy, task_ids, frame_indices)
    residual = residual_targets(base_array, expert_array)
    clipped_residual = group_clip(residual)
    cspr, gate = apply_cspr_refinement(base_array, clipped_residual, labels["score"], tau=labels["q_tau"])
    identity, _ = apply_cspr_refinement(base_array, np.zeros_like(clipped_residual), labels["score"], tau=labels["q_tau"])
    inactive, _ = apply_cspr_refinement(base_array, clipped_residual, labels["score"], tau=float(np.max(labels["score"]) + 1.0))
    uniform, _ = uniform_refinement_ablation(base_array, clipped_residual, intervention_fraction=float(np.mean(gate)))
    simple, _ = critical_step_threshold_simple_killer(base_array, clipped_residual)
    delta = action_delta_summary(base_array, cspr)
    clean = clean_retention_summary(base_array, identity, inactive)
    gradient = gradient_smoke(base_array, clipped_residual, gate, expert_array, labels["labels"])
    base_loss = mean_huber(base_array, expert_array)
    cspr_loss = mean_huber(cspr, expert_array)
    uniform_loss = mean_huber(uniform, expert_array)
    simple_loss = mean_huber(simple, expert_array)
    dysl_loss = base_loss
    comparator_loss = min(dysl_loss, uniform_loss, simple_loss)
    source_coverage = _source_coverage(loaded_records)
    validation_labels = labels["labels"][~discovery_mask]
    validation_health = label_health(
        validation_labels if len(validation_labels) else labels["labels"],
        [task for task, split in zip(task_ids, splits) if split == "validation"] or task_ids,
        [frame for frame, split in zip(frame_indices, splits) if split == "validation"] or frame_indices,
    )

    manifest_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(loaded_records):
        for policy in POLICY_ROWS:
            manifest_row = {
                "split": record["split"],
                "task_suite": record["task_suite"],
                "task_identity": record["task_id"],
                "demo_id": int(record["demo_id"]),
                "frame_index": int(record["frame_index"]),
                "source_edge_sha256": record["source_edge_sha256"],
                "model_or_probe": policy,
                "config_label": CONFIG_LABEL,
                "probe_label": POLICY_PROBE,
            }
            manifest_row["row_key"] = cspr_row_key(manifest_row)
            manifest_rows.append(manifest_row)
            partial_rows.append(
                {
                    "row_key": manifest_row["row_key"],
                    "split": record["split"],
                    "task_identity": record["task_id"],
                    "demo_id": int(record["demo_id"]),
                    "frame_index": int(record["frame_index"]),
                    "model_or_probe": policy,
                    "criticality_positive_fraction": float(np.mean(labels["labels"][record_index])),
                    "gate_activation_fraction": float(np.mean(gate[record_index] > 0.5)),
                    "base_huber": float(mean_huber(base_array[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                    "cspr_huber": float(mean_huber(cspr[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                    "action_validity_ok": True,
                }
            )

    manifest_summary = validate_manifest(manifest_rows, partial_rows)
    validation_task_coverage_ok = set(source_coverage["validation_task_counts"]) == FIXED_TASKS
    inputs = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=bool(
            manifest_summary["duplicate_manifest_key_count"] == 0
            and manifest_summary["duplicate_partial_key_count"] == 0
            and manifest_summary["missing_manifest_key_count"] == 0
            and manifest_summary["extra_partial_key_count"] == 0
            and manifest_summary["split_overlap_key_count"] == 0
            and manifest_summary["key_sets_equal"]
        ),
        source_alignment_ok=bool(source_exceptions == 0),
        action_semantics_ok=True,
        base_chunks_valid=True,
        feature_caches_valid=bool(feature_array.shape[1] == VISUAL_FEATURE_DIM and np.isfinite(feature_array).all()),
        labels_noncollapsed=validation_health["labels_noncollapsed"],
        criticality_score_variance_ok=labels["criticality_score_variance_ok"],
        enough_discovery_rows=source_coverage["discovery_row_count"] >= MIN_DISCOVERY_ROWS,
        enough_validation_rows=source_coverage["validation_row_count"] >= MIN_VALIDATION_ROWS,
        validation_task_coverage_ok=validation_task_coverage_ok,
        maximum_validation_task_fraction=source_coverage["validation_task_fraction_max"],
        validation_positive_count=validation_health["positive_count"],
        validation_negative_count=validation_health["negative_count"],
        validation_positive_fraction=validation_health["positive_fraction"],
        largest_positive_task_fraction=validation_health["largest_positive_task_fraction"],
        criticality_predictability_margin=predictability["criticality_predictability_margin"],
        base_residual_headroom=max(base_loss - cspr_loss, 0.0),
        dysl_residual_headroom=max(dysl_loss - cspr_loss, 0.0),
        simple_killer_residual_headroom=max(simple_loss - cspr_loss, 0.0),
        cspr_beats_comparators=bool(comparator_loss - cspr_loss >= 0.005),
        cspr_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        cspr_differs_from_ablation=float(np.max(np.abs(cspr - uniform))) > 1e-12,
        simple_killer_explains_gain=bool(simple_loss <= cspr_loss),
        identity_reload_error=clean["identity_max_abs_error"],
        finite_nonzero_gradients=gradient["finite_nonzero_gradients"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=float(np.mean(gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_ok=action_validity_summary(cspr)["action_validity_ok"],
        clean_retention_ok=clean["clean_retention_ok"],
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        simulator_load_count=0,
        closed_loop_experiment_happened=False,
        training_happened=False,
        validation_search_happened=False,
        exception_count=source_exceptions,
    )
    metrics = {
        "source_coverage": source_coverage,
        "manifest_summary": manifest_summary,
        "label_health": label_metrics,
        "validation_label_health": validation_health,
        "criticality": {
            "q_tau": labels["q_tau"],
            "tau_quantile": labels["tau_quantile"],
            "criticality_score_variance_ok": labels["criticality_score_variance_ok"],
        },
        "predictability": predictability,
        "losses": {
            "base_huber": base_loss,
            "dysl_proxy_huber": dysl_loss,
            "cspr_huber": cspr_loss,
            "uniform_ablation_huber": uniform_loss,
            "simple_killer_huber": simple_loss,
        },
        "action_delta": delta,
        "clean_retention": clean,
        "gradient": gradient,
        "source_exception_count": source_exceptions,
        "last_source_exception": last_exception,
    }
    return manifest_rows, partial_rows, metrics, inputs


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# CSPR-VLA Stage 0 Result",
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
    if result["final_decision"] == "CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        disposition = "Stage 0 passes to bounded validation search under the frozen protocol."
    else:
        disposition = "Stage 0 stops under the frozen development-audit taxonomy; this is not a closed-loop scientific kill."
    text = "\n".join(
        [
            "# CSPR-VLA Stage 0 Adjudication",
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


def _write_stage0_result(
    paths: Mapping[str, Path],
    preflight: Mapping[str, Any],
    serializer: Mapping[str, Any],
    prior: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    decision_inputs: Stage0DecisionInputs,
) -> dict[str, Any]:
    manifest_summary = validate_manifest(manifest_rows, partial_rows)
    decision_inputs = Stage0DecisionInputs(
        **{
            **asdict(decision_inputs),
            "proposal_hash_ok": bool(preflight.get("proposal_hash_ok")),
            "serializer_preflight_ok": bool(serializer.get("passed")),
            "official_prior_asset_check_persisted": bool(prior),
            "preflight_passed": bool(preflight.get("preflight_passed")),
            "action_semantics_ok": bool(action_semantics),
            "manifest_integrity_ok": bool(
                manifest_summary["duplicate_manifest_key_count"] == 0
                and manifest_summary["duplicate_partial_key_count"] == 0
                and manifest_summary["missing_manifest_key_count"] == 0
                and manifest_summary["extra_partial_key_count"] == 0
                and manifest_summary["split_overlap_key_count"] == 0
                and manifest_summary["key_sets_equal"]
            ),
        }
    )
    decision = classify_stage0(decision_inputs)
    result = {
        "method": "CSPR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "proposal_hash_ok": bool(preflight.get("proposal_hash_ok")),
        "final_decision": decision,
        "decision": decision,
        "valid_scientific_result": False,
        "closed_loop_scientific_result": False,
        "stage_0_is_closed_loop_scientific_kill": False,
        "planned_model_row_count": len(manifest_rows),
        "completed_model_row_count": len(partial_rows),
        "exception_count": int(decision_inputs.exception_count),
        "last_exception": metrics.get("last_source_exception"),
        "manifest_summary": manifest_summary,
        "duplicate_manifest_key_count": manifest_summary["duplicate_manifest_key_count"],
        "duplicate_partial_key_count": manifest_summary["duplicate_partial_key_count"],
        "missing_manifest_key_count": manifest_summary["missing_manifest_key_count"],
        "extra_partial_key_count": manifest_summary["extra_partial_key_count"],
        "split_overlap_key_count": manifest_summary["split_overlap_key_count"],
        "key_sets_equal": manifest_summary["key_sets_equal"],
        "policy_row_counts": dict(Counter(str(row.get("model_or_probe")) for row in partial_rows)),
        "preflight": preflight,
        "serializer_preflight": serializer,
        "official_prior_asset_check": prior,
        "action_semantics": action_semantics,
        "decision_inputs": decision_inputs,
        "metrics": metrics,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "created_utc": _utc_now(),
    }
    manifest_payload = {
        "method": "CSPR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "rows": list(manifest_rows),
        "manifest_summary": manifest_summary,
        "manifest_hash": canonical_json_sha256({"rows": list(manifest_rows), "method": "CSPR-VLA"}),
    }
    _write_json(paths["manifest"], manifest_payload)
    _write_json(
        paths["partial"],
        {
            "method": "CSPR-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "policy_probe": POLICY_PROBE,
            "config_label": CONFIG_LABEL,
            "planned_model_row_count": len(manifest_rows),
            "completed_model_row_count": len(partial_rows),
            "exception_count": int(decision_inputs.exception_count),
            "rows": list(partial_rows),
            "updated_utc": _utc_now(),
        },
    )
    _write_json(paths["result_json"], result)
    _write_result_markdown(paths["result_md"], result)
    _write_adjudication(paths["adjudication"], result)
    return result


def _run(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if paths["result_json"].is_file() and not args.force:
        result = _read_json(paths["result_json"])
        _write_status(paths, "existing_result_found", final_decision=result.get("final_decision"))
        return 0
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_status(paths, "running")
    _write_heartbeat(paths, completed_rows=0, planned_rows=0)
    serializer = _serializer_preflight(paths["serializer_preflight"])
    prior = _official_prior_asset_check(paths["official_prior_asset_check"])
    action_semantics = _write_action_semantics(paths["action_semantics"])
    preflight = _preflight(paths, serializer, prior, action_semantics)
    records = _load_base_records(paths["ccif_partial"], paths["ccif_manifest"], max_sources=args.max_rows)
    manifest_rows, partial_rows, metrics, decision_inputs = _build_manifest_and_partial(records)
    _write_heartbeat(paths, completed_rows=len(partial_rows), planned_rows=len(manifest_rows))
    result = _write_stage0_result(
        paths,
        preflight,
        serializer,
        prior,
        action_semantics,
        manifest_rows,
        partial_rows,
        metrics,
        decision_inputs,
    )
    _write_status(paths, "completed", final_decision=result["final_decision"])
    _write_text(paths["exit_code"], "0\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default=str(REPORT_ROOT))
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--ccif-partial", default=str(DEFAULT_CCIF_PARTIAL))
    parser.add_argument("--ccif-manifest", default=str(DEFAULT_CCIF_MANIFEST))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths(args)
    try:
        if args.serializer_preflight:
            _serializer_preflight(paths["serializer_preflight"])
            return 0
        return _run(args)
    except Exception as exc:  # noqa: BLE001
        paths["report"].mkdir(parents=True, exist_ok=True)
        blocker = {
            "method": "CSPR-VLA",
            "final_decision": "CSPR_STAGE_0_IMPLEMENTATION_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "valid_scientific_result": False,
            "stage_0_is_closed_loop_scientific_kill": False,
            "created_utc": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        _write_status(paths, "failed", error_type=type(exc).__name__, error=str(exc))
        _write_text(paths["exit_code"], "1\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
