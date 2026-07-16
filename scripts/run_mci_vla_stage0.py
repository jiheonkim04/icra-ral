"""Run MCI-VLA Stage 0 implementation preflight and development audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from tca_map.smolvla.mci_vla import (  # noqa: E402
    ACTION_DIM,
    COMPARATOR_MARGIN_MIN,
    HORIZON,
    LATENT_DIM_VALUES,
    POLICY_ROWS,
    PROPOSAL_HASH,
    PROPRIO_DIM,
    TRANSFORMATION_FAMILIES,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_mci_adapter,
    augmentation_only_lora_killer,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    consistency_code,
    consistency_observability_diagnostics,
    identity_passthrough,
    json_default,
    mean_huber,
    mci_no_consistency_code_ablation,
    mci_row_key,
    objective_gradient_smoke,
    representation_health,
    residual_targets,
    rovla_multiconsistency_proxy,
    transformed_inputs,
    validate_manifest,
)


POLICY_PROBE = "mci_stage0_multi_consistency_invariance"
CONFIG_LABEL = "mci_frozen_stage0_lc050_dz16"
SEED = 20263800
REPORT_ROOT = REPO_ROOT / "reports" / "mci_vla"
RUN_ROOT = REPO_ROOT / "runs" / "mci_vla" / "stage0"
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
    chunk = actions[start : min(stop, len(actions))]
    if len(chunk) < HORIZON:
        pad = np.repeat(chunk[-1:, :], HORIZON - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    if chunk.shape != (HORIZON, ACTION_DIM):
        raise ValueError(f"expert action chunk must have shape [{HORIZON},{ACTION_DIM}], got {chunk.shape}")
    return chunk.astype(np.float64)


def _write_status(paths: Mapping[str, Path], state: str, **extra: Any) -> None:
    _write_json(paths["status"], {"method": "MCI-VLA", "state": state, "updated_utc": _utc_now(), **extra})


def _write_heartbeat(paths: Mapping[str, Path], **extra: Any) -> None:
    _write_json(
        paths["heartbeat"],
        {"method": "MCI-VLA", "pid": os.getpid(), "updated_utc": _utc_now(), **extra},
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


def _manifest_row(split: str, family: str = "instruction", policy: str = "mci_full") -> dict[str, Any]:
    row = {
        "split": split,
        "task_suite": "libero_goal",
        "task_identity": "libero_goal/task_5",
        "demo_id": 8 if split == "validation" else 0,
        "window_start": 12,
        "transform_family": family,
        "policy": policy,
        "config_label": CONFIG_LABEL,
        "probe_label": POLICY_PROBE,
    }
    row["row_key"] = mci_row_key(row)
    return row


def _synthetic_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = 12
    base = rng.normal(scale=0.003, size=(n, HORIZON, ACTION_DIM)).astype(np.float64)
    expert = base.copy()
    for index in range(n):
        expert[index, 8:18, 0] += 0.04 + 0.002 * index
        expert[index, 18:26, 4] -= 0.03
        expert[index, 24:28, 6] += 0.25 if index % 2 == 0 else -0.25
    features = rng.normal(size=(n, VISUAL_FEATURE_DIM)).astype(np.float64)
    proprio = rng.normal(scale=0.05, size=(n, PROPRIO_DIM)).astype(np.float64)
    tasks = ["libero_goal/task_5"] * n
    code = consistency_code(features, proprio, tasks, base, latent_dim=LATENT_DIM_VALUES[0])
    transformed_feature, transformed_prop, transformed_tasks, transformed_base, _ = transformed_inputs(
        features,
        proprio,
        tasks,
        base,
        family="observation_proprioception",
    )
    transformed_code = consistency_code(transformed_feature, transformed_prop, transformed_tasks, transformed_base)
    residual = residual_targets(base, expert)
    mci, gate, _ = apply_mci_adapter(base, residual, code)
    identity, _ = identity_passthrough(base)
    ablation, _ = mci_no_consistency_code_ablation(base, residual, intervention_fraction=float(np.mean(gate > 0.5)))
    rovla = rovla_multiconsistency_proxy(base)
    killer = augmentation_only_lora_killer(base, residual)
    gradient = objective_gradient_smoke(base, expert, transformed_base, code, transformed_code, mci, gate)
    rep = representation_health(code)
    clean = clean_retention_summary(base, identity, base)
    delta = action_delta_summary(base, mci)
    manifest = [_manifest_row("discovery"), _manifest_row("validation")]
    partial = [{"row_key": row["row_key"]} for row in manifest]
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        feature_caches_valid=True,
        transformations_noncollapsed=True,
        enough_discovery_rows=True,
        enough_validation_rows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        minimum_validation_pairs_per_family=32,
        positive_contrast_count=32,
        negative_contrast_count=32,
        representation_dims_fraction_above_floor=max(0.80, rep["dims_fraction_above_floor"]),
        consistency_predictability_margin=0.03,
        base_transformed_pair_headroom=max(mean_huber(base, expert) - mean_huber(mci, expert), 0.01),
        rovla_residual_headroom=max(mean_huber(rovla, expert) - mean_huber(mci, expert), 0.01),
        augmentation_residual_headroom=max(mean_huber(killer, expert) - mean_huber(mci, expert), 0.01),
        mci_beats_comparators=True,
        mci_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        mci_differs_from_rovla=float(np.max(np.abs(mci - rovla))) > 0.0,
        mci_differs_from_ablation=float(np.max(np.abs(mci - ablation))) > 0.0,
        mci_differs_from_augmentation_only_lora=float(np.max(np.abs(mci - killer))) > 0.0,
        exact_base_passthrough_ok=clean["identity_max_abs_error"] <= 1e-7,
        identity_reload_error=clean["identity_max_abs_error"],
        finite_nonzero_gradients=gradient["finite_nonzero_gradients"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=min(gradient["weighted_gradient_norm_ratio_max"], 10.0),
        intervention_fraction=float(np.mean(gate > 0.5)),
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_rate=1.0,
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
        "method": "MCI-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "manifest_row": manifest[0],
        "manifest_summary": validate_manifest(manifest, partial),
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "visual_feature_dimension": VISUAL_FEATURE_DIM,
        "proprio_dimension": PROPRIO_DIM,
        "policy_rows": POLICY_ROWS,
        "transformation_families": TRANSFORMATION_FAMILIES,
        "code": code,
        "transformed_code": transformed_code,
        "gate_activation_fraction": float(np.mean(gate > 0.5)),
        "mci_chunk": mci,
        "rovla_proxy_chunk": rovla,
        "no_code_ablation_chunk": ablation,
        "augmentation_only_lora_chunk": killer,
        "action_delta_summary": delta,
        "clean_retention": clean,
        "representation_health": rep,
        "gradient": gradient,
        "decision_inputs": decision_inputs,
        "decision": classify_stage0(decision_inputs),
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
        "method": "MCI-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(
            reproduced == fixture_hash
            and fixture["decision"] == "MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
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
        REPO_ROOT / "third_party" / "RoVLA",
        REPO_ROOT / "external" / "RoVLA",
        REPO_ROOT / "runs" / "rovla",
    ]
    present = [candidate for candidate in candidates if candidate.exists()]
    checkpoints: list[str] = []
    for root in present:
        for pattern in ("*.pt", "*.pth", "*.safetensors", "*.ckpt"):
            checkpoints.extend(str(child) for child in root.rglob(pattern))
    result = {
        "method": "MCI-VLA",
        "closest_prior": "RoVLA",
        "closest_prior_preprint": "https://arxiv.org/abs/2605.19678",
        "closest_prior_official_repository": "https://github.com/HCPLab-SYSU/RoVLA",
        "official_code_present": bool(present),
        "official_candidate_paths": [str(candidate) for candidate in present],
        "official_checkpoint_present": bool(checkpoints),
        "official_checkpoint_count": len(checkpoints),
        "selected_prior_policy": "official_rovla" if present and checkpoints else "rovla_multiconsistency_proxy",
        "transparent_proxy_mismatch_list": []
        if present and checkpoints
        else [
            "official RoVLA checkpoint/assets unavailable locally",
            "using transparent deterministic multi-consistency proxy",
            "proxy preserves instruction, observation/proprioception, and action-evolution consistency axes",
        ],
        "proxy_is_required_until_official_assets_verified": not (present and checkpoints),
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    result = {
        "method": "MCI-VLA",
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
        "method": "MCI-VLA",
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
        manifest_row = manifest_by_key.get(str(row.get("row_key")), {})
        if not row.get("base_chunk_cache_path") or not row.get("feature_cache_path"):
            continue
        if not manifest_row.get("source_path"):
            continue
        records.append(
            {
                "split": split,
                "task_suite": str(row["suite"]),
                "task_id": str(task),
                "demo_id": demo,
                "frame_index": int(row["frame_index"]),
                "source_edge_sha256": row["source_edge_sha256"],
                "source_path": manifest_row["source_path"],
                "task_language": manifest_row.get("task_language"),
                "phase_bin": int(manifest_row.get("phase_bin", 0)),
                "base_chunk_cache_path": row["base_chunk_cache_path"],
                "base_chunk_cache_sha256": row.get("base_chunk_cache_sha256"),
                "feature_cache_path": row["feature_cache_path"],
                "feature_cache_sha256": row.get("feature_cache_sha256"),
                "proprio_dim": int(row.get("proprio_dim", PROPRIO_DIM)),
                "proprio_finite_fraction": float(row.get("proprio_finite_fraction", 1.0)),
            }
        )
    records.sort(key=lambda item: (item["split"], item["task_id"], int(item["demo_id"]), int(item["frame_index"])))
    if max_sources is not None:
        records = records[: int(max_sources)]
    return records


def _source_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    discovery = [record for record in records if record["split"] == "discovery"]
    validation = [record for record in records if record["split"] == "validation"]
    validation_counts = Counter(str(record["task_id"]) for record in validation)
    validation_total = len(validation)
    return {
        "source_row_count": len(records),
        "discovery_row_count": len(discovery),
        "validation_row_count": len(validation),
        "discovery_demo_ids": sorted({int(record["demo_id"]) for record in discovery}),
        "validation_demo_ids": sorted({int(record["demo_id"]) for record in validation}),
        "discovery_task_counts": dict(Counter(str(record["task_id"]) for record in discovery)),
        "validation_task_counts": dict(validation_counts),
        "validation_task_fraction_max": max(validation_counts.values()) / validation_total
        if validation_total and validation_counts
        else 1.0,
    }


def _empty_decision_inputs(manifest_summary: Mapping[str, Any], *, exception_count: int = 0) -> Stage0DecisionInputs:
    return Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=bool(manifest_summary.get("key_sets_equal", True)),
        source_alignment_ok=False,
        action_semantics_ok=True,
        base_chunks_valid=False,
        feature_caches_valid=False,
        transformations_noncollapsed=False,
        enough_discovery_rows=False,
        enough_validation_rows=False,
        validation_task_coverage_ok=False,
        maximum_validation_task_fraction=1.0,
        minimum_validation_pairs_per_family=0,
        positive_contrast_count=0,
        negative_contrast_count=0,
        representation_dims_fraction_above_floor=0.0,
        consistency_predictability_margin=0.0,
        base_transformed_pair_headroom=0.0,
        rovla_residual_headroom=0.0,
        augmentation_residual_headroom=0.0,
        mci_beats_comparators=False,
        mci_differs_from_base=False,
        mci_differs_from_rovla=False,
        mci_differs_from_ablation=False,
        mci_differs_from_augmentation_only_lora=False,
        exact_base_passthrough_ok=False,
        identity_reload_error=0.0,
        finite_nonzero_gradients=False,
        frozen_base_gradient_count=0,
        weighted_gradient_norm_ratio_max=0.0,
        intervention_fraction=0.0,
        action_deltas_bounded=False,
        action_validity_rate=0.0,
        clean_retention_ok=False,
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        simulator_load_count=0,
        closed_loop_experiment_happened=False,
        training_happened=False,
        validation_search_happened=False,
        exception_count=exception_count,
    )


def _build_manifest_and_partial(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Stage0DecisionInputs]:
    if not records:
        manifest_summary = validate_manifest([], [])
        return [], [], {"source_coverage": _source_coverage([]), "manifest_summary": manifest_summary}, _empty_decision_inputs(manifest_summary)

    base_chunks: list[np.ndarray] = []
    expert_chunks: list[np.ndarray] = []
    features: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    loaded_records: list[Mapping[str, Any]] = []
    source_exceptions = 0
    last_exception: str | None = None
    for record in records:
        try:
            base = _read_npz_array(record["base_chunk_cache_path"], "base_chunk")
            feature = _read_npz_array(record["feature_cache_path"], "feature")
            expert = _read_expert_actions(record["source_path"], int(record["demo_id"]), int(record["frame_index"]))
            if base.shape != (HORIZON, ACTION_DIM):
                raise ValueError(f"base chunk has invalid shape {base.shape}")
            if feature.shape != (VISUAL_FEATURE_DIM,):
                feature = feature.reshape(-1)
            if feature.shape != (VISUAL_FEATURE_DIM,):
                raise ValueError(f"visual feature has invalid shape {feature.shape}")
            if int(record.get("proprio_dim", PROPRIO_DIM)) != PROPRIO_DIM:
                raise ValueError("record proprio dimension is not 8")
            base_chunks.append(base)
            expert_chunks.append(expert)
            features.append(feature)
            proprio.append(np.zeros(PROPRIO_DIM, dtype=np.float64))
            loaded_records.append(record)
        except Exception as exc:  # noqa: BLE001
            source_exceptions += 1
            last_exception = f"{type(exc).__name__}: {exc}"

    if not loaded_records:
        manifest_summary = validate_manifest([], [])
        metrics = {
            "source_coverage": _source_coverage([]),
            "manifest_summary": manifest_summary,
            "source_exception_count": source_exceptions,
            "last_source_exception": last_exception,
        }
        return [], [], metrics, _empty_decision_inputs(manifest_summary, exception_count=source_exceptions)

    base_array = np.asarray(base_chunks, dtype=np.float64)
    expert_array = np.asarray(expert_chunks, dtype=np.float64)
    feature_array = np.asarray(features, dtype=np.float64)
    proprio_array = np.asarray(proprio, dtype=np.float64)
    task_ids = [str(record["task_id"]) for record in loaded_records]
    frame_indices = [int(record["frame_index"]) for record in loaded_records]
    splits = [str(record["split"]) for record in loaded_records]
    discovery_mask = np.asarray([split == "discovery" for split in splits], dtype=bool)
    validation_mask = ~discovery_mask
    code = consistency_code(feature_array, proprio_array, task_ids, base_array, latent_dim=LATENT_DIM_VALUES[0])
    residual = residual_targets(base_array, expert_array)
    mci, gate, capped_residual = apply_mci_adapter(base_array, residual, code)
    intervention_fraction = float(np.mean(gate > 0.5))
    ablation, _ = mci_no_consistency_code_ablation(base_array, residual, intervention_fraction=intervention_fraction)
    rovla = rovla_multiconsistency_proxy(base_array)
    killer = augmentation_only_lora_killer(base_array, residual)
    identity, _ = identity_passthrough(base_array)
    clean = clean_retention_summary(base_array, identity, base_array)
    delta = action_delta_summary(base_array, mci)
    rep = representation_health(code[validation_mask] if np.any(validation_mask) else code)

    family_metrics: dict[str, Any] = {}
    observability_scores: list[float] = []
    observability_targets: list[int] = []
    observability_tasks: list[str] = []
    observability_frames: list[int] = []
    observability_magnitudes: list[float] = []
    observability_families: list[str] = []
    transformed_for_gradient = base_array.copy()
    transformed_code_for_gradient = code.copy()
    for family in TRANSFORMATION_FAMILIES:
        tf, tp, tt, tb, metadata = transformed_inputs(feature_array, proprio_array, task_ids, base_array, family=family)
        tz = consistency_code(tf, tp, tt, tb, latent_dim=LATENT_DIM_VALUES[0])
        if family == "observation_proprioception":
            transformed_for_gradient = tb
            transformed_code_for_gradient = tz
        distances = np.linalg.norm(code - tz, axis=1)
        shuffled = np.roll(code, 1, axis=0)
        negative_distances = np.linalg.norm(tz - shuffled, axis=1)
        validation_distances = distances[validation_mask] if np.any(validation_mask) else distances
        validation_negative = negative_distances[validation_mask] if np.any(validation_mask) else negative_distances
        family_metrics[family] = {
            "metadata": metadata,
            "pair_count": int(len(distances)),
            "validation_pair_count": int(len(validation_distances)),
            "positive_contrast_count": int(len(validation_distances)),
            "negative_contrast_count": int(len(validation_negative)),
            "positive_code_distance_mean": float(np.mean(validation_distances)) if len(validation_distances) else 0.0,
            "negative_code_distance_mean": float(np.mean(validation_negative)) if len(validation_negative) else 0.0,
            "noncollapsed": bool(np.var(distances) > 0.0 or np.var(negative_distances) > 0.0),
        }
        selected_indexes = np.where(validation_mask)[0] if np.any(validation_mask) else np.arange(len(code))
        action_magnitude = np.mean(np.abs(base_array), axis=(1, 2))
        for index in selected_indexes:
            observability_scores.append(float(-distances[index]))
            observability_targets.append(1)
            observability_tasks.append(task_ids[index])
            observability_frames.append(frame_indices[index])
            observability_magnitudes.append(float(action_magnitude[index]))
            observability_families.append(family)
            observability_scores.append(float(-negative_distances[index]))
            observability_targets.append(0)
            observability_tasks.append(task_ids[index])
            observability_frames.append(frame_indices[index])
            observability_magnitudes.append(float(action_magnitude[index]))
            observability_families.append(family)

    observability = consistency_observability_diagnostics(
        observability_scores,
        observability_targets,
        observability_tasks,
        observability_frames,
        observability_magnitudes,
        observability_families,
    )
    gradient = objective_gradient_smoke(
        base_array,
        expert_array,
        transformed_for_gradient,
        code,
        transformed_code_for_gradient,
        mci,
        gate,
    )
    base_loss = mean_huber(base_array, expert_array)
    mci_loss = mean_huber(mci, expert_array)
    rovla_loss = mean_huber(rovla, expert_array)
    ablation_loss = mean_huber(ablation, expert_array)
    killer_loss = mean_huber(killer, expert_array)
    comparator_best = min(rovla_loss, ablation_loss, killer_loss)
    source_coverage = _source_coverage(loaded_records)
    validation_task_coverage_ok = set(source_coverage["validation_task_counts"]) == FIXED_TASKS

    manifest_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    action_valid_flags: list[bool] = []
    policy_actions = {
        "smolvla_base": base_array,
        "rovla_multiconsistency_proxy": rovla,
        "mci_full": mci,
        "mci_no_consistency_code_ablation": ablation,
        "augmentation_only_lora_killer": killer,
        "transformation_label_health_diagnostic": base_array,
        "consistency_observability_diagnostic": base_array,
        "identity_passthrough_reload_diagnostic": identity,
        "objective_gradient_scale_diagnostic": mci,
    }
    for record_index, record in enumerate(loaded_records):
        for family in TRANSFORMATION_FAMILIES:
            for policy in POLICY_ROWS:
                manifest_row = {
                    "split": record["split"],
                    "task_suite": record["task_suite"],
                    "task_identity": record["task_id"],
                    "demo_id": int(record["demo_id"]),
                    "window_start": int(record["frame_index"]),
                    "transform_family": family,
                    "policy": policy,
                    "config_label": CONFIG_LABEL,
                    "probe_label": POLICY_PROBE,
                    "source_edge_sha256": record["source_edge_sha256"],
                    "base_chunk_cache_sha256": record.get("base_chunk_cache_sha256"),
                    "feature_cache_sha256": record.get("feature_cache_sha256"),
                }
                manifest_row["row_key"] = mci_row_key(manifest_row)
                action = policy_actions[policy][record_index : record_index + 1]
                validity = action_validity_summary(action)
                action_valid_flags.append(bool(validity["action_validity_ok"]))
                partial_rows.append(
                    {
                        "row_key": manifest_row["row_key"],
                        "split": record["split"],
                        "task_identity": record["task_id"],
                        "demo_id": int(record["demo_id"]),
                        "window_start": int(record["frame_index"]),
                        "transform_family": family,
                        "policy": policy,
                        "action_validity_ok": bool(validity["action_validity_ok"]),
                        "code_std_mean": rep["std_mean"],
                        "gate_activation_fraction": float(np.mean(gate[record_index] > 0.5)),
                        "base_huber": float(mean_huber(base_array[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                        "mci_huber": float(mean_huber(mci[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                        "rovla_proxy_huber": float(mean_huber(rovla[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                        "no_code_ablation_huber": float(mean_huber(ablation[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                        "augmentation_only_lora_huber": float(mean_huber(killer[record_index : record_index + 1], expert_array[record_index : record_index + 1])),
                    }
                )
                manifest_rows.append(manifest_row)

    manifest_summary = validate_manifest(manifest_rows, partial_rows)
    minimum_validation_pairs = min(
        (metrics["validation_pair_count"] for metrics in family_metrics.values()),
        default=0,
    )
    positive_count = int(sum(metrics["positive_contrast_count"] for metrics in family_metrics.values()))
    negative_count = int(sum(metrics["negative_contrast_count"] for metrics in family_metrics.values()))
    transformations_noncollapsed = all(bool(metrics["noncollapsed"]) for metrics in family_metrics.values())
    action_validity_rate = float(np.mean(action_valid_flags)) if action_valid_flags else 0.0
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
        source_alignment_ok=source_exceptions == 0,
        action_semantics_ok=True,
        base_chunks_valid=True,
        feature_caches_valid=bool(feature_array.shape[1] == VISUAL_FEATURE_DIM and np.isfinite(feature_array).all()),
        transformations_noncollapsed=transformations_noncollapsed,
        enough_discovery_rows=source_coverage["discovery_row_count"] >= MIN_DISCOVERY_ROWS,
        enough_validation_rows=source_coverage["validation_row_count"] >= MIN_VALIDATION_ROWS,
        validation_task_coverage_ok=validation_task_coverage_ok,
        maximum_validation_task_fraction=source_coverage["validation_task_fraction_max"],
        minimum_validation_pairs_per_family=int(minimum_validation_pairs),
        positive_contrast_count=positive_count,
        negative_contrast_count=negative_count,
        representation_dims_fraction_above_floor=rep["dims_fraction_above_floor"],
        consistency_predictability_margin=observability["consistency_predictability_margin"],
        base_transformed_pair_headroom=max(base_loss - mci_loss, 0.0),
        rovla_residual_headroom=max(rovla_loss - mci_loss, 0.0),
        augmentation_residual_headroom=max(killer_loss - mci_loss, 0.0),
        mci_beats_comparators=bool(comparator_best - mci_loss >= COMPARATOR_MARGIN_MIN),
        mci_differs_from_base=delta["changed_cell_fraction"] > 0.0,
        mci_differs_from_rovla=float(np.max(np.abs(mci - rovla))) > 1e-12,
        mci_differs_from_ablation=float(np.max(np.abs(mci - ablation))) > 1e-12,
        mci_differs_from_augmentation_only_lora=float(np.max(np.abs(mci - killer))) > 1e-12,
        exact_base_passthrough_ok=clean["identity_max_abs_error"] <= 1e-7,
        identity_reload_error=clean["identity_max_abs_error"],
        finite_nonzero_gradients=gradient["finite_nonzero_gradients"],
        frozen_base_gradient_count=gradient["frozen_base_gradient_count"],
        weighted_gradient_norm_ratio_max=gradient["weighted_gradient_norm_ratio_max"],
        intervention_fraction=intervention_fraction,
        action_deltas_bounded=delta["action_deltas_bounded"],
        action_validity_rate=action_validity_rate,
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
        "proprioception_source": "zero_vector_placeholder_because_ccif_rows_expose_dim_and_finite_fraction_not_raw_proprio",
        "transformation_pair_counts": family_metrics,
        "representation_health": rep,
        "observability": observability,
        "losses": {
            "base_huber": base_loss,
            "rovla_proxy_huber": rovla_loss,
            "mci_huber": mci_loss,
            "no_code_ablation_huber": ablation_loss,
            "augmentation_only_lora_huber": killer_loss,
        },
        "headroom": {
            "base_transformed_pair_headroom": max(base_loss - mci_loss, 0.0),
            "rovla_residual_headroom": max(rovla_loss - mci_loss, 0.0),
            "augmentation_residual_headroom": max(killer_loss - mci_loss, 0.0),
        },
        "action_delta": delta,
        "clean_retention": clean,
        "gradient": gradient,
        "source_exception_count": source_exceptions,
        "last_source_exception": last_exception,
    }
    return manifest_rows, partial_rows, metrics, inputs


def _merge_resume_partial(paths: Mapping[str, Path], manifest_rows: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]], resume: bool) -> list[dict[str, Any]]:
    if not resume or not paths["partial"].is_file() or paths["result_json"].is_file():
        return list(partial_rows)
    existing_payload = _read_json(paths["partial"])
    existing_rows = [dict(row) for row in existing_payload.get("rows", [])]
    existing_keys = {str(row.get("row_key")) for row in existing_rows}
    manifest_keys = {mci_row_key(row) for row in manifest_rows}
    accepted_existing = [row for row in existing_rows if str(row.get("row_key")) in manifest_keys]
    accepted_keys = {str(row.get("row_key")) for row in accepted_existing}
    missing_rows = [dict(row) for row in partial_rows if str(row.get("row_key")) not in accepted_keys]
    return accepted_existing + missing_rows


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# MCI-VLA Stage 0 Result",
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
    if result["final_decision"] == "MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        disposition = "Stage 0 passes to bounded validation search under the frozen protocol."
    else:
        disposition = "Stage 0 stops under the frozen development-audit taxonomy; this is not a closed-loop scientific kill."
    text = "\n".join(
        [
            "# MCI-VLA Stage 0 Adjudication",
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
        "method": "MCI-VLA",
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
        "policy_row_counts": dict(Counter(str(row.get("policy")) for row in partial_rows)),
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
        "resource_contention_intervals": [],
        "created_utc": _utc_now(),
    }
    manifest_payload = {
        "method": "MCI-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
        "rows": list(manifest_rows),
        "manifest_summary": manifest_summary,
        "manifest_hash": canonical_json_sha256({"rows": list(manifest_rows), "method": "MCI-VLA"}),
    }
    _write_json(paths["manifest"], manifest_payload)
    _write_json(
        paths["partial"],
        {
            "method": "MCI-VLA",
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
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
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
    manifest_rows, candidate_partial_rows, metrics, decision_inputs = _build_manifest_and_partial(records)
    partial_rows = _merge_resume_partial(paths, manifest_rows, candidate_partial_rows, args.resume)
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
            "method": "MCI-VLA",
            "final_decision": "MCI_STAGE_0_IMPLEMENTATION_FAILURE",
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
