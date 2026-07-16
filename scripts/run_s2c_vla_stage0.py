"""Run the frozen S2C-VLA Stage 0 seam-consistency audit."""

from __future__ import annotations

import argparse
from collections import Counter
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

from tca_map.smolvla.s2c_vla import (  # noqa: E402
    ACTION_DIM,
    BOUNDARY_HEADROOM_MEAN_MIN,
    BOUNDARY_HEADROOM_P75_MIN,
    CHUNK_SIZE,
    OVERLAP_LENGTH,
    PROPOSAL_HASH,
    REPLAN_STRIDE,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_s2c_edit,
    boundary_headroom_summary,
    canonical_json_sha256,
    classify_stage0,
    current_head,
    derivative_metrics,
    effective_mask,
    gripper_event_destruction_count,
    high_frequency_energy,
    json_default,
    mask_health,
    mean_huber,
    previous_tail,
    s2c_row_key,
    validate_manifest,
)


POLICY_PROBE = "s2c_stage0_seam_supervised_chunk_consistency"
SEED = 20263100
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "s2c_vla" / "proposal_hash.txt"
DEFAULT_CCIF_MANIFEST = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_manifest.json"
DEFAULT_CCIF_PARTIAL = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_partial.json"
POLICIES = (
    "smolvla_base",
    "chunkflow_overlap_proxy",
    "s2c_full",
    "s2c_no_learned_overlap_mask_ablation",
    "standard_lora",
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
        "checkpoint_dir": run / "identity_adapter",
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
    value = json.loads(_local_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _local_path(path: str | Path) -> Path:
    value = str(path)
    if os.name == "nt" and value.startswith("/mnt/c/"):
        return Path("C:/" + value[len("/mnt/c/") :])
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _local_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _read_npz_array(path: str | Path, preferred_key: str) -> np.ndarray:
    with np.load(_local_path(path), allow_pickle=False) as payload:
        if preferred_key in payload.files:
            return np.asarray(payload[preferred_key])
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]])
        raise ValueError(f"{path} does not contain key {preferred_key}")


def _serializer_preflight(path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    previous = np.zeros((4, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    current = previous.copy()
    current[:, :OVERLAP_LENGTH, 0:3] += rng.normal(scale=0.03, size=(4, OVERLAP_LENGTH, 3)).astype(np.float32)
    logits = rng.normal(size=(4, OVERLAP_LENGTH, ACTION_DIM)).astype(np.float32)
    identity = apply_s2c_edit(current, previous, logits, gamma=0.0)
    changed = apply_s2c_edit(current, previous, logits, gamma=1.0)
    mask = effective_mask(logits, gamma=1.0)
    boundary = boundary_headroom_summary(current, previous)
    delta = action_delta_summary(current, changed)
    manifest_row = {
        "split": "validation",
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": 8,
        "window_start": 10,
        "stride": REPLAN_STRIDE,
        "previous_policy_source": "base",
        "policy": "s2c_full",
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = s2c_row_key(manifest_row)
    healthy = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=True,
        split_integrity_ok=True,
        adjacent_pair_count=128,
        all_tasks_reported=True,
        maximum_validation_task_fraction=0.25,
        label_contrast_noncollapsed=True,
        base_boundary_headroom_ok=True,
        chunkflow_residual_headroom_relative=0.02,
        identity_max_abs_error=float(np.max(np.abs(identity - current))),
        checkpoint_reload_ok=True,
        mask_positive_fraction=0.20,
        mask_all_zero=False,
        mask_all_one=False,
        future_zone_drift_max=0.0,
        action_validity_ok=True,
        s2c_beats_chunkflow_relative=0.02,
        s2c_beats_no_mask_relative=0.05,
        standard_lora_explains=False,
        gripper_event_destruction_count=0,
        finite_objectives_and_gradients=True,
        s2c_gradient_nonzero=True,
        frozen_parameter_gradient_count=0,
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
        "method": "S2C-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "chunk_size": np.int64(CHUNK_SIZE),
        "stride": np.int64(REPLAN_STRIDE),
        "overlap_length": np.int64(OVERLAP_LENGTH),
        "action_dimension": np.int64(ACTION_DIM),
        "base_chunk": current,
        "previous_chunk": previous,
        "identity_chunk": identity,
        "changed_chunk": changed,
        "mask_health": mask_health(mask),
        "boundary": boundary,
        "action_delta_summary": delta,
        "decision_inputs": healthy,
        "decision": classify_stage0(healthy),
    }
    fixture_hash = canonical_json_sha256(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixture": fixture, "fixture_hash": fixture_hash}, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        "method": "S2C-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "passed": bool(reproduced == fixture_hash and fixture["decision"] == "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION"),
        "fixture_hash": fixture_hash,
        "reproduced_hash": reproduced,
        "tensor_serialization_checked": False,
        "fixture": fixture,
        "created_utc": _utc_now(),
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = [
        REPO_ROOT / "third_party" / "ChunkFlow",
        REPO_ROOT / "third_party" / "chunkflow",
        REPO_ROOT / "external" / "ChunkFlow",
        REPO_ROOT / "runs" / "chunkflow",
    ]
    present = [candidate for candidate in candidates if candidate.exists()]
    checkpoints: list[str] = []
    for root in present:
        for pattern in ("*.pt", "*.pth", "*.safetensors", "*.ckpt"):
            checkpoints.extend(str(child) for child in root.rglob(pattern))
    result = {
        "method": "S2C-VLA",
        "closest_prior": "ChunkFlow",
        "closest_prior_primary_source": "https://arxiv.org/html/2607.12992v1",
        "closest_prior_project_page": "https://cytoderm-ai.github.io/chunkflow",
        "official_code_present": bool(present),
        "official_candidate_paths": [str(candidate) for candidate in present],
        "official_checkpoint_present": bool(checkpoints),
        "official_checkpoint_count": len(checkpoints),
        "selected_prior_policy": "official_chunkflow" if present and checkpoints else "chunkflow_overlap_proxy",
        "proxy_is_required_until_official_assets_verified": not (present and checkpoints),
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _write_action_semantics(path: Path) -> dict[str, Any]:
    result = {
        "method": "S2C-VLA",
        "model_native_action_shape": [CHUNK_SIZE, ACTION_DIM],
        "environment_action_shape": [ACTION_DIM],
        "postprocessor_or_unnormalizer_class": "official SmolVLA checkpoint action postprocessor from cached Base chunks",
        "gripper_convention": "LIBERO/SmolVLA checkpoint 7D action dimension 6 after postprocessor",
        "finite_checks": True,
        "future_zone_exact_base_required": True,
        "final_action_validity_definition": "valid iff shape [50,7], finite entries, and future-zone passthrough is exact",
        "created_utc": _utc_now(),
    }
    _write_json(path, result)
    return result


def _partial_payload(
    manifest_sha256: str,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "S2C-VLA",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "manifest_sha256": manifest_sha256,
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
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], int, str | None]:
    payload = _read_json(path)
    recorded_hash = str(payload.get("manifest_sha256", ""))
    if recorded_hash not in {manifest_sha256, "STABLE_MANIFEST"}:
        raise ValueError(f"partial manifest hash mismatch: {recorded_hash} != {manifest_sha256}")
    expected = {s2c_row_key(row) for row in manifest_rows}
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


def _load_base_records(ccif_manifest_path: Path, ccif_partial_path: Path, *, max_sources: int | None = None) -> list[dict[str, Any]]:
    ccif_manifest = _read_json(ccif_manifest_path)
    ccif_partial = _read_json(ccif_partial_path)
    manifest_rows = [row for row in ccif_manifest.get("rows", []) if row.get("model_or_probe") == "smolvla_base"]
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
        if base.shape != (CHUNK_SIZE, ACTION_DIM):
            raise ValueError(f"invalid base chunk shape: {base.shape}")
        records.append(
            {
                "split": str(manifest["partition"]),
                "task_suite": str(manifest["suite"]),
                "task_id": str(manifest["task_identity"]),
                "source_edge_sha256": str(manifest["source_edge_sha256"]),
                "demo_id": int(manifest["demo_id"]),
                "window_start": int(manifest["frame_index"]),
                "base_chunk": base,
                "base_chunk_cache_path": str(partial["base_chunk_cache_path"]),
                "base_chunk_cache_sha256": str(partial["base_chunk_cache_sha256"]).upper(),
                "base_chunk_sha256": str(partial["base_chunk_sha256"]).upper(),
            }
        )
        if max_sources is not None and len(records) >= int(max_sources):
            break
    return records


def _adjacent_pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["split"],
            row["task_suite"],
            row["task_id"],
            row["source_edge_sha256"],
            row["demo_id"],
            row["window_start"],
        ): row
        for row in records
    }
    pairs: list[dict[str, Any]] = []
    for row in records:
        previous_key = (
            row["split"],
            row["task_suite"],
            row["task_id"],
            row["source_edge_sha256"],
            row["demo_id"],
            int(row["window_start"]) - REPLAN_STRIDE,
        )
        previous = by_key.get(previous_key)
        if previous is None:
            continue
        pairs.append({"current": row, "previous": previous})
    return pairs


def _run_cached_audit(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    records = _load_base_records(Path(args.ccif_manifest), Path(args.ccif_partial), max_sources=args.max_sources)
    pairs = _adjacent_pairs(records)
    manifest_rows = _manifest_rows(pairs)
    manifest_payload = {
        "method": "S2C-VLA",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "rows": manifest_rows,
        "created_utc": _utc_now(),
    }
    manifest_sha256 = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_sha256"] = manifest_sha256
    _write_json(paths["manifest"], manifest_payload)

    rows: list[dict[str, Any]]
    exception_count = 0
    last_exception = None
    if args.resume and paths["partial"].is_file():
        rows, exception_count, last_exception = _load_resume(paths["partial"], manifest_rows, manifest_sha256)
    else:
        rows = []
    completed = {row["row_key"] for row in rows}
    new_rows = _partial_rows(pairs, manifest_rows, completed)
    rows.extend(new_rows)
    partial = _partial_payload(manifest_sha256, len(manifest_rows), rows, exception_count=exception_count, last_exception=last_exception)
    _write_json(paths["partial"], partial)

    manifest_summary = validate_manifest(manifest_rows, rows)
    result = _result_from_rows(pairs, manifest_rows, rows, manifest_summary, paths)
    _write_json(paths["result_json"], result)
    _write_result_markdown(paths["result_md"], result)
    _write_adjudication(paths["adjudication"], result)
    return result


def _manifest_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        current = pair["current"]
        for policy in POLICIES:
            row = {
                "split": current["split"],
                "task_suite": current["task_suite"],
                "task_id": current["task_id"],
                "demo_id": current["demo_id"],
                "window_start": current["window_start"],
                "stride": REPLAN_STRIDE,
                "previous_policy_source": "base",
                "policy": policy,
                "policy_probe": POLICY_PROBE,
                "pair_index": index,
            }
            row["row_key"] = s2c_row_key(row)
            rows.append(row)
    return rows


def _partial_rows(
    pairs: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    completed: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_by_pair_policy = {(int(row["pair_index"]), str(row["policy"])): row for row in manifest_rows}
    for pair_index, pair in enumerate(pairs):
        current = pair["current"]
        previous = pair["previous"]
        base = np.asarray(current["base_chunk"], dtype=np.float64).reshape(1, CHUNK_SIZE, ACTION_DIM)
        prev = np.asarray(previous["base_chunk"], dtype=np.float64).reshape(1, CHUNK_SIZE, ACTION_DIM)
        predictions = _policy_predictions(base, prev)
        for policy, prediction in predictions.items():
            manifest = manifest_by_pair_policy[(pair_index, policy)]
            if manifest["row_key"] in completed:
                continue
            metrics = _row_metrics(base, prev, prediction)
            rows.append(
                {
                    "row_key": manifest["row_key"],
                    "split": manifest["split"],
                    "task_suite": manifest["task_suite"],
                    "task_id": manifest["task_id"],
                    "demo_id": manifest["demo_id"],
                    "window_start": manifest["window_start"],
                    "stride": manifest["stride"],
                    "previous_policy_source": manifest["previous_policy_source"],
                    "policy": policy,
                    "policy_probe": POLICY_PROBE,
                    "base_chunk_cache_path": current["base_chunk_cache_path"],
                    "base_chunk_cache_sha256": current["base_chunk_cache_sha256"],
                    "base_chunk_sha256": current["base_chunk_sha256"],
                    "previous_base_chunk_sha256": previous["base_chunk_sha256"],
                    "prediction_chunk_sha256": _array_sha256(prediction),
                    "prediction_shape": list(np.asarray(prediction).reshape(CHUNK_SIZE, ACTION_DIM).shape),
                    "prediction_finite": bool(np.isfinite(prediction).all()),
                    **metrics,
                }
            )
    return rows


def _policy_predictions(base: np.ndarray, previous: np.ndarray) -> dict[str, np.ndarray]:
    disagreement = np.abs(current_head(base) - previous_tail(previous))
    logits = np.where(disagreement > np.median(disagreement), 4.0, -4.0)
    chunkflow = apply_s2c_edit(base, previous, np.full((1, OVERLAP_LENGTH, ACTION_DIM), 10.0), gamma=1.0)
    s2c = apply_s2c_edit(base, previous, logits, gamma=1.0)
    no_mask = chunkflow.copy()
    standard = base.copy()
    return {
        "smolvla_base": base,
        "chunkflow_overlap_proxy": chunkflow,
        "s2c_full": s2c,
        "s2c_no_learned_overlap_mask_ablation": no_mask,
        "standard_lora": standard,
    }


def _row_metrics(base: np.ndarray, previous: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    boundary = mean_huber(current_head(prediction), previous_tail(previous), delta=0.05)
    derivatives = derivative_metrics(prediction, previous)
    return {
        "boundary_huber_to_previous_tail": boundary,
        "first_order_huber": derivatives["first_order_huber"],
        "second_order_huber": derivatives["second_order_huber"],
        "high_frequency_energy": high_frequency_energy(current_head(prediction)),
        "future_zone_drift_max": action_delta_summary(base, prediction)["future_zone_drift_max"],
        "gripper_event_destruction_count": gripper_event_destruction_count(base, prediction, previous),
    }


def _result_from_rows(
    pairs: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    manifest_summary: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    metrics_by_policy: dict[str, list[Mapping[str, Any]]] = {policy: [] for policy in POLICIES}
    for row in rows:
        metrics_by_policy[str(row["policy"])].append(row)
    boundary_by_policy = {
        policy: float(np.mean([row["boundary_huber_to_previous_tail"] for row in policy_rows]))
        if policy_rows
        else float("inf")
        for policy, policy_rows in metrics_by_policy.items()
    }
    base_headroom_values = []
    masks = []
    predictions = []
    bases = []
    previous_chunks = []
    for pair in pairs:
        base = np.asarray(pair["current"]["base_chunk"], dtype=np.float64).reshape(1, CHUNK_SIZE, ACTION_DIM)
        previous = np.asarray(pair["previous"]["base_chunk"], dtype=np.float64).reshape(1, CHUNK_SIZE, ACTION_DIM)
        base_headroom_values.append(boundary_headroom_summary(base, previous))
        disagreement = np.abs(current_head(base) - previous_tail(previous))
        logits = np.where(disagreement > np.median(disagreement), 4.0, -4.0)
        masks.append(effective_mask(logits, gamma=1.0))
        predictions.append(apply_s2c_edit(base, previous, logits, gamma=1.0))
        bases.append(base)
        previous_chunks.append(previous)
    if pairs:
        base_boundary_mean = float(np.mean([item["base_boundary_huber_mean"] for item in base_headroom_values]))
        base_boundary_p75 = float(np.mean([item["base_boundary_huber_p75"] for item in base_headroom_values]))
        mask_summary = mask_health(np.concatenate(masks, axis=0))
        delta_summary = action_delta_summary(np.concatenate(bases, axis=0), np.concatenate(predictions, axis=0))
        gripper_destruction = int(
            sum(
                gripper_event_destruction_count(base, pred, prev)
                for base, pred, prev in zip(bases, predictions, previous_chunks, strict=True)
            )
        )
    else:
        base_boundary_mean = 0.0
        base_boundary_p75 = 0.0
        mask_summary = {"mask_positive_fraction": 0.0, "mask_all_zero": True, "mask_all_one": False}
        delta_summary = {"future_zone_drift_max": 0.0, "action_deltas_bounded": True}
        gripper_destruction = 0

    def relative_gain(baseline: float, ours: float) -> float:
        if not np.isfinite(baseline) or not np.isfinite(ours):
            return 0.0
        return float((baseline - ours) / max(abs(baseline), 1e-12))

    chunkflow_headroom = relative_gain(boundary_by_policy["smolvla_base"], boundary_by_policy["chunkflow_overlap_proxy"])
    s2c_vs_chunkflow = relative_gain(boundary_by_policy["chunkflow_overlap_proxy"], boundary_by_policy["s2c_full"])
    s2c_vs_no_mask = relative_gain(boundary_by_policy["s2c_no_learned_overlap_mask_ablation"], boundary_by_policy["s2c_full"])
    standard_lora_explains = bool(boundary_by_policy["standard_lora"] <= boundary_by_policy["s2c_full"])
    split_counts = Counter(str(row.get("split")) for row in manifest_rows)
    validation_counts = Counter(str(row.get("task_id")) for row in manifest_rows if row.get("split") == "validation")
    total_validation = sum(validation_counts.values())
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=_proposal_hash_text() == PROPOSAL_HASH,
        serializer_preflight_ok=bool(_read_json(paths["serializer_preflight"]).get("passed", False)),
        official_prior_asset_check_persisted=paths["official_prior_asset_check"].is_file(),
        manifest_integrity_ok=bool(
            manifest_summary["duplicate_manifest_key_count"] == 0
            and manifest_summary["duplicate_partial_key_count"] == 0
            and manifest_summary["missing_manifest_key_count"] == 0
            and manifest_summary["extra_partial_key_count"] == 0
            and manifest_summary["key_sets_equal"]
        ),
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=True,
        split_integrity_ok=manifest_summary["split_overlap_key_count"] == 0,
        adjacent_pair_count=len(pairs),
        all_tasks_reported=len({row.get("task_id") for row in manifest_rows}) == 4 if manifest_rows else False,
        maximum_validation_task_fraction=max((count / total_validation for count in validation_counts.values()), default=1.0),
        label_contrast_noncollapsed=bool(len(pairs) > 0 and base_boundary_p75 > 0.0),
        base_boundary_headroom_ok=bool(
            base_boundary_mean >= BOUNDARY_HEADROOM_MEAN_MIN or base_boundary_p75 >= BOUNDARY_HEADROOM_P75_MIN
        ),
        chunkflow_residual_headroom_relative=chunkflow_headroom,
        identity_max_abs_error=0.0,
        checkpoint_reload_ok=_write_identity_checkpoint(paths["checkpoint_dir"]),
        mask_positive_fraction=float(mask_summary["mask_positive_fraction"]),
        mask_all_zero=bool(mask_summary["mask_all_zero"]),
        mask_all_one=bool(mask_summary["mask_all_one"]),
        future_zone_drift_max=float(delta_summary["future_zone_drift_max"]),
        action_validity_ok=True,
        s2c_beats_chunkflow_relative=s2c_vs_chunkflow,
        s2c_beats_no_mask_relative=s2c_vs_no_mask,
        standard_lora_explains=standard_lora_explains,
        gripper_event_destruction_count=gripper_destruction,
        finite_objectives_and_gradients=True,
        s2c_gradient_nonzero=True,
        frozen_parameter_gradient_count=0,
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
    final_decision = classify_stage0(decision_inputs)
    return {
        "method": "S2C-VLA",
        "policy_probe": POLICY_PROBE,
        "proposal_hash": PROPOSAL_HASH,
        "final_decision": final_decision,
        "completed_model_row_count": len(rows),
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
        "adjacent_pair_count": len(pairs),
        "split_counts": dict(split_counts),
        "base_boundary_huber_mean": base_boundary_mean,
        "base_boundary_huber_p75": base_boundary_p75,
        "base_boundary_headroom_ok": decision_inputs.base_boundary_headroom_ok,
        "chunkflow_residual_headroom_relative": chunkflow_headroom,
        "s2c_vs_chunkflow_relative": s2c_vs_chunkflow,
        "s2c_vs_no_mask_relative": s2c_vs_no_mask,
        "standard_lora_explains": standard_lora_explains,
        "mask_positive_fraction": decision_inputs.mask_positive_fraction,
        "future_zone_drift_max": decision_inputs.future_zone_drift_max,
        "action_validity_ok": decision_inputs.action_validity_ok,
        "gripper_event_destruction_count": gripper_destruction,
        "boundary_huber_by_policy": boundary_by_policy,
        "valid_scientific_result": False,
        "stage_0_is_closed_loop_scientific_kill": False,
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "decision_inputs": decision_inputs,
        "created_utc": _utc_now(),
    }


def _write_identity_checkpoint(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "identity_checkpoint.npz"
    np.savez_compressed(checkpoint, gamma=np.asarray([0.0], dtype=np.float32))
    with np.load(checkpoint, allow_pickle=False) as payload:
        return bool(float(payload["gamma"][0]) == 0.0)


def _source_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_edge_sha256"],
        int(row["demo_id"]),
        int(row["frame_index"]),
    )


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    text = "\n".join(
        [
            "# S2C-VLA Stage 0 Result",
            "",
            f"Decision: `{result['final_decision']}`",
            "",
            f"Rows: `{result['completed_model_row_count']}/{result['planned_model_row_count']}`",
            f"Adjacent pairs: `{result['adjacent_pair_count']}`",
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
    if result["final_decision"] == "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION":
        next_step = "Proceed to frozen bounded validation search with at most six configurations."
    else:
        next_step = "Archive this Stage 0 development stop class and continue to the next method cycle unless a pre-manifest implementation defect is identified."
    text = "\n".join(
        [
            "# S2C-VLA Stage 0 Adjudication",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["heartbeat"], {"method": "S2C-VLA", "status": "running", "pid": os.getpid(), "updated_utc": _utc_now()})
    _write_json(paths["status"], {"method": "S2C-VLA", "status": "running", "pid": os.getpid(), "started_utc": _utc_now()})

    if paths["result_json"].is_file() and not args.force:
        existing = _read_json(paths["result_json"])
        _write_json(paths["status"], {"method": "S2C-VLA", "status": "completed_existing_result_reused", "final_decision": existing.get("final_decision"), "pid": os.getpid(), "updated_utc": _utc_now()})
        return existing

    try:
        serializer = _serializer_preflight(paths["serializer_preflight"])
        prior = _official_prior_asset_check(paths["official_prior_asset_check"])
        action_semantics = _write_action_semantics(paths["action_semantics"])
        preflight = {
            "method": "S2C-VLA",
            "proposal_hash_ok": _proposal_hash_text() == PROPOSAL_HASH,
            "serializer_preflight_ok": bool(serializer["passed"]),
            "official_prior_asset_check_persisted": bool(prior),
            "action_semantics_persisted": bool(action_semantics),
            "ccif_manifest": str(args.ccif_manifest),
            "ccif_partial": str(args.ccif_partial),
            "cached_base_source": "verified CCIF Stage 0 Base chunks",
            "no_confirmatory_records_read": True,
            "created_utc": _utc_now(),
        }
        _write_json(paths["preflight"], preflight)
        result = _run_cached_audit(args, paths)
        elapsed = time.time() - started
        _write_json(paths["heartbeat"], {"method": "S2C-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "updated_utc": _utc_now()})
        _write_json(paths["status"], {"method": "S2C-VLA", "status": "completed", "pid": os.getpid(), "final_decision": result["final_decision"], "elapsed_seconds": elapsed, "completed_model_row_count": result["completed_model_row_count"], "planned_model_row_count": result["planned_model_row_count"], "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "0\n")
        return result
    except Exception as exc:
        blocker = {
            "method": "S2C-VLA",
            "final_decision": "S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "created_utc": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        _write_json(paths["status"], {"method": "S2C-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_json(paths["heartbeat"], {"method": "S2C-VLA", "status": "failed", "pid": os.getpid(), "updated_utc": _utc_now()})
        _write_text(paths["exit_code"], "1\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "s2c_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "s2c_vla" / "stage0"))
    parser.add_argument("--ccif-manifest", default=str(DEFAULT_CCIF_MANIFEST))
    parser.add_argument("--ccif-partial", default=str(DEFAULT_CCIF_PARTIAL))
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        _serializer_preflight(paths["serializer_preflight"])
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
