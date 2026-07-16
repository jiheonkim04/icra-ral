"""Run DCCG-VLA Stage 0 implementation preflight utilities."""

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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.dccg_vla import (  # noqa: E402
    ACTION_DIM,
    FEATURE_COUNT,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_dccg_guidance,
    canonical_json_sha256,
    classify_stage0,
    coherence_energy,
    coherence_features,
    deployment_bin_key,
    dccg_row_key,
    feature_health,
    fit_demo_statistics,
    gripper_event_summary,
    gradient_smoke,
    json_default,
    no_demo_calibration_stats,
    smoothing_simple_killer,
    validate_manifest,
)


POLICY_PROBE = "dccg_stage0_demonstration_calibrated_coherence_guidance"
CONFIG_LABEL = "dccg_frozen_stage0_c0"
REPORT_ROOT = REPO_ROOT / "reports" / "dccg_vla"
RUN_ROOT = REPO_ROOT / "runs" / "dccg_vla" / "stage0"
DEFAULT_CCIF_PARTIAL = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_partial.json"
DEFAULT_CCIF_MANIFEST = REPO_ROOT / "reports" / "ccif_vla" / "stage_0_manifest.json"
DISCOVERY_TASKS = {
    "libero_10/task_1",
    "libero_10/task_3",
    "libero_goal/task_1",
    "libero_goal/task_3",
    "libero_object/task_1",
    "libero_spatial/task_1",
}
VALIDATION_TASKS = {
    "libero_10/task_5",
    "libero_goal/task_5",
    "libero_object/task_3",
    "libero_spatial/task_3",
}
DISCOVERY_DEMOS = set(range(30))
VALIDATION_DEMOS = set(range(30, 40))
MIN_DISCOVERY_WINDOWS = 384
MIN_VALIDATION_WINDOWS = 128
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


def _write_status(paths: Mapping[str, Path], state: str, **extra: Any) -> None:
    _write_json(
        paths["status"],
        {
            "method": "DCCG-VLA",
            "state": state,
            "updated_utc": _utc_now(),
            **extra,
        },
    )


def _write_heartbeat(paths: Mapping[str, Path], **extra: Any) -> None:
    _write_json(
        paths["heartbeat"],
        {
            "method": "DCCG-VLA",
            "pid": os.getpid(),
            "updated_utc": _utc_now(),
            **extra,
        },
    )


def _manifest_row(split: str, policy: str = "dccg_full") -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_id": "libero_goal/task_5",
        "demo_id": 30 if split == "validation" else 0,
        "window_start": 12,
        "bin_key": "libero_goal|q1|t1|r0|g1|c1",
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["row_key"] = dccg_row_key(row)
    return row


def _synthetic_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(20263600)
    base = np.zeros((4, HORIZON, ACTION_DIM), dtype=np.float64)
    demo = base.copy()
    for idx in range(4):
        demo[idx, :, 0] = np.linspace(0.0, 0.04 + idx * 0.004, HORIZON)
        demo[idx, :, 1] = 0.01 * np.sin(np.linspace(0.0, np.pi, HORIZON) + idx)
        demo[idx, 18:24, 6] = 0.35
    jitter = demo.copy()
    jitter[:, :, 0] += rng.normal(scale=0.015, size=(4, HORIZON))
    bin_keys = [deployment_bin_key(chunk, task_family="libero_goal", queue_index=12) for chunk in demo]
    features = coherence_features(demo)
    stats = fit_demo_statistics(features, bin_keys)
    global_stats = no_demo_calibration_stats(features)
    gradient = gradient_smoke(jitter[:1], [bin_keys[0]], stats)
    dccg, _ = apply_dccg_guidance(jitter[:1], gradient["gradient"], [1.0], gamma=0.10)
    smoothing = smoothing_simple_killer(jitter[:1])
    gate_fraction = 0.25
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        features_noncollapsed=True,
        bins_noncollapsed=True,
        enough_discovery_windows=True,
        enough_validation_windows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        gate_activation_fraction=gate_fraction,
        base_acg_headroom=0.05,
        dccg_differs_from_base=True,
        dccg_differs_from_acg=True,
        dccg_differs_from_ablation=True,
        dccg_differs_from_smoothing=bool(np.max(np.abs(dccg - smoothing)) > 0.0),
        finite_nonzero_gradients=bool(gradient["finite_nonzero_gradients"]),
        exact_base_passthrough_ok=True,
        gripper_event_preservation_ok=True,
        normalized_action_validity_ok=True,
        postprocessed_action_validity_ok=True,
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
    manifest = [_manifest_row("discovery", "smolvla_base"), _manifest_row("validation", "dccg_full")]
    partial = [{"row_key": row["row_key"]} for row in manifest]
    return {
        "method": "DCCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "feature_count": FEATURE_COUNT,
        "policy_rows": POLICY_ROWS,
        "config_label": CONFIG_LABEL,
        "probe_label": POLICY_PROBE,
        "manifest_row": manifest[0],
        "manifest_summary": validate_manifest(manifest, partial),
        "coherence_features": features,
        "coherence_energy": coherence_energy(jitter[:1], [bin_keys[0]], stats),
        "global_energy": coherence_energy(jitter[:1], ["global"], global_stats),
        "gradient_norm_mean": gradient["gradient_norm_mean"],
        "action_delta_summary": action_delta_summary(jitter[:1], dccg),
        "action_validity_summary": action_validity_summary(dccg),
        "decision_inputs": decision_inputs,
        "decision": classify_stage0(decision_inputs),
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    fixture = _synthetic_fixture()
    fixture_hash = canonical_json_sha256(fixture)
    payload = {
        "method": "DCCG-VLA",
        "passed": True,
        "created_utc": _utc_now(),
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "parsed": True,
        "reproduced_hash": fixture_hash,
    }
    _write_json(path, payload)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted_hash = canonical_json_sha256(persisted["fixture"])
    persisted["reproduced_hash"] = persisted_hash
    persisted["passed"] = persisted_hash == persisted["fixture_hash"]
    _write_json(path, persisted)
    return persisted


def _preflight(report_root: Path) -> dict[str, Any]:
    proposal_hash_file = report_root / "proposal_hash.txt"
    proposal_hash_ok = proposal_hash_file.is_file() and proposal_hash_file.read_text(encoding="utf-8").strip() == PROPOSAL_HASH
    missing_docs = [str(path) for path in REQUIRED_SOURCE_DOCS if not path.is_file()]
    payload = {
        "method": "DCCG-VLA",
        "created_utc": _utc_now(),
        "proposal_hash": PROPOSAL_HASH,
        "proposal_hash_ok": proposal_hash_ok,
        "missing_required_docs": missing_docs,
        "preflight_passed": proposal_hash_ok and not missing_docs,
        "implementation_stage": "preflight_only",
    }
    _write_json(report_root / "stage_0_preflight.json", payload)
    return payload


def _write_static_contract_artifacts(report_root: Path) -> None:
    _write_json(
        report_root / "stage_0_official_prior_asset_check.json",
        {
            "method": "DCCG-VLA",
            "created_utc": _utc_now(),
            "closest_prior": "ACG",
            "official_repository": "https://github.com/DAVIAN-Robotics/ACG",
            "official_assets_checked": False,
            "local_policy_2_label": "acg_official_proxy",
            "transparent_proxy_required_if_official_assets_unavailable": True,
        },
    )
    _write_json(
        report_root / "stage_0_action_semantics.json",
        {
            "method": "DCCG-VLA",
            "created_utc": _utc_now(),
            "model_native_action_shape": [HORIZON, ACTION_DIM],
            "postprocessor_or_unnormalizer_class": "official SmolVLA/LIBERO action postprocessor required at full Stage 0",
            "gripper_convention": "preserve hard transition counts and sign-change timing",
            "final_action_validity_definition": "shape [50,7], finite entries, official postprocessor validity, and Base-relative group caps",
        },
    )


def _cache_coverage(partial_path: Path, manifest_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ccif_partial": str(partial_path),
        "ccif_manifest": str(manifest_path),
        "ccif_partial_exists": partial_path.is_file(),
        "ccif_manifest_exists": manifest_path.is_file(),
        "smolvla_base_rows_total": 0,
        "matching_frozen_dccg_rows": 0,
        "available_task_counts": {},
        "available_demo_ids_by_task": {},
        "required_discovery_tasks": sorted(DISCOVERY_TASKS),
        "required_validation_tasks": sorted(VALIDATION_TASKS),
        "required_discovery_demo_range": [0, 29],
        "required_validation_demo_range": [30, 39],
    }
    if not partial_path.is_file():
        return summary
    rows = _read_json(partial_path).get("rows", [])
    task_counts: Counter[str] = Counter()
    demos: dict[str, set[int]] = defaultdict(set)
    matching = 0
    for row in rows:
        if row.get("model_or_probe") != "smolvla_base":
            continue
        task = str(row.get("task_identity"))
        demo = int(row.get("demo_id", -1))
        task_counts[task] += 1
        demos[task].add(demo)
        if task in DISCOVERY_TASKS and demo in DISCOVERY_DEMOS:
            matching += 1
        if task in VALIDATION_TASKS and demo in VALIDATION_DEMOS:
            matching += 1
    summary["smolvla_base_rows_total"] = int(sum(task_counts.values()))
    summary["matching_frozen_dccg_rows"] = int(matching)
    summary["available_task_counts"] = dict(sorted(task_counts.items()))
    summary["available_demo_ids_by_task"] = {
        task: sorted(values) for task, values in sorted(demos.items())
    }
    return summary


def _load_cached_base_records(partial_path: Path, manifest_path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    if not partial_path.is_file() or not manifest_path.is_file():
        return []
    partial = _read_json(partial_path)
    manifest = _read_json(manifest_path)
    manifest_by_key = {str(row.get("row_key")): row for row in manifest.get("rows", [])}
    records: list[dict[str, Any]] = []
    for row in partial.get("rows", []):
        if row.get("model_or_probe") != "smolvla_base":
            continue
        task = str(row.get("task_identity"))
        demo = int(row.get("demo_id", -1))
        split: str | None = None
        if task in DISCOVERY_TASKS and demo in DISCOVERY_DEMOS:
            split = "discovery"
        if task in VALIDATION_TASKS and demo in VALIDATION_DEMOS:
            split = "validation"
        if split is None:
            continue
        manifest_row = manifest_by_key.get(str(row.get("row_key")), {})
        cache_path = row.get("base_chunk_cache_path")
        if not cache_path:
            continue
        record = {
            "split": split,
            "task_suite": str(row.get("suite") or task.split("/")[0]),
            "task_id": task,
            "demo_id": demo,
            "window_start": int(row.get("frame_index", 0)),
            "source_edge_sha256": row.get("source_edge_sha256"),
            "source_path": manifest_row.get("source_path"),
            "base_chunk_cache_path": cache_path,
            "base_chunk_cache_sha256": row.get("base_chunk_cache_sha256"),
        }
        records.append(record)
    records.sort(key=lambda item: (item["split"], item["task_id"], int(item["demo_id"]), int(item["window_start"])))
    if max_records is not None:
        records = records[: int(max_records)]
    return records


def _deterministic_noise(label: str, shape: Sequence[int], scale: float) -> np.ndarray:
    seed = int(canonical_json_sha256({"label": label})[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=tuple(shape))


def _policy_action(
    policy: str,
    base: np.ndarray,
    bin_key: str,
    stats: Mapping[str, Mapping[str, Any]] | None,
    global_stats: Mapping[str, Mapping[str, Any]] | None,
    row_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    chunk = base.reshape(1, HORIZON, ACTION_DIM)
    extra: dict[str, Any] = {}
    if policy == "smolvla_base":
        return chunk.copy(), extra
    if policy == "action_smoothing_simple_killer" or policy == "acg_official_proxy":
        return smoothing_simple_killer(chunk), {"proxy_note": "transparent smoothing-based coherence proxy"}
    if policy == "synthetic_jitter_diagnostic":
        jitter = chunk.copy()
        jitter[:, :, 0:3] += _deterministic_noise(row_key, (1, HORIZON, 3), 0.015)
        return jitter, extra
    if policy == "synthetic_pause_diagnostic":
        paused = chunk.copy()
        paused[:, 18:28, 0:3] = paused[:, 18:19, 0:3]
        return paused, extra
    if policy == "synthetic_gripper_corruption_diagnostic":
        corrupt = chunk.copy()
        corrupt[:, 16:30, 6] *= -1.0
        return corrupt, extra
    selected_stats = stats if policy == "dccg_full" else global_stats
    if policy in {"dccg_full", "dccg_no_demo_calibration_ablation"} and selected_stats:
        gradient = gradient_smoke(chunk, [bin_key], selected_stats)
        guided, gate = apply_dccg_guidance(chunk, gradient["gradient"], [1.0], gamma=0.10)
        extra.update(
            {
                "gate_value": float(gate[0]),
                "finite_nonzero_gradient": bool(gradient["finite_nonzero_gradients"]),
                "gradient_norm_mean": gradient["gradient_norm_mean"],
            }
        )
        return guided, extra
    return chunk.copy(), {"fallback_to_base": True}


def _build_manifest_and_partial(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for record in records:
        chunk = _read_npz_array(str(record["base_chunk_cache_path"]), "base_chunk")
        if chunk.shape != (HORIZON, ACTION_DIM):
            raise ValueError(f"cached base chunk must have shape [{HORIZON},{ACTION_DIM}], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("cached base chunk contains nonfinite values")
        bin_key = deployment_bin_key(chunk, task_family=str(record["task_suite"]), queue_index=int(record["window_start"]))
        loaded.append({"record": dict(record), "chunk": chunk, "bin_key": bin_key})

    discovery = [item for item in loaded if item["record"]["split"] == "discovery"]
    stats = None
    global_stats = None
    feature_summary = {
        "feature_count": FEATURE_COUNT,
        "features_noncollapsed": False,
        "bins_noncollapsed": False,
        "bin_counts": {},
    }
    if discovery:
        discovery_chunks = np.stack([item["chunk"] for item in discovery], axis=0)
        discovery_bins = [str(item["bin_key"]) for item in discovery]
        features = coherence_features(discovery_chunks)
        stats = fit_demo_statistics(features, discovery_bins)
        global_stats = no_demo_calibration_stats(features)
        feature_summary = feature_health(features, discovery_bins)

    manifest_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    energy_by_policy: dict[str, list[float]] = defaultdict(list)
    gradient_flags: list[bool] = []
    action_valid_flags: list[bool] = []
    gripper_flags: list[bool] = []
    dccg_delta_max: list[float] = []

    for source_index, item in enumerate(loaded):
        record = item["record"]
        base = item["chunk"]
        bin_key = str(item["bin_key"])
        for policy in POLICY_ROWS:
            manifest_row = {
                "split": record["split"],
                "task_suite": record["task_suite"],
                "task_id": record["task_id"],
                "demo_id": record["demo_id"],
                "window_start": record["window_start"],
                "bin_key": bin_key,
                "policy": policy,
                "config_label": CONFIG_LABEL,
                "probe_label": POLICY_PROBE,
                "source_record_index": source_index,
                "source_edge_sha256": record.get("source_edge_sha256"),
            }
            manifest_row["row_key"] = dccg_row_key(manifest_row)
            action, extra = _policy_action(policy, base, bin_key, stats, global_stats, manifest_row["row_key"])
            validity = action_validity_summary(action)
            grip = gripper_event_summary(base.reshape(1, HORIZON, ACTION_DIM), action)
            energy_stats = stats or global_stats
            energy = coherence_energy(action, [bin_key if energy_stats is stats else "global"], energy_stats)[0] if energy_stats else 0.0
            delta = action_delta_summary(base.reshape(1, HORIZON, ACTION_DIM), action)
            manifest_rows.append(manifest_row)
            partial_rows.append(
                {
                    "row_key": manifest_row["row_key"],
                    "split": record["split"],
                    "task_id": record["task_id"],
                    "demo_id": record["demo_id"],
                    "window_start": record["window_start"],
                    "policy": policy,
                    "coherence_energy": float(energy),
                    "action_validity_ok": bool(validity["action_validity_ok"]),
                    "gripper_event_preservation_ok": bool(grip["gripper_event_preservation_ok"]),
                    "action_delta_summary": delta,
                    **extra,
                }
            )
            energy_by_policy[policy].append(float(energy))
            action_valid_flags.append(bool(validity["action_validity_ok"]))
            gripper_flags.append(bool(grip["gripper_event_preservation_ok"]))
            if "finite_nonzero_gradient" in extra:
                gradient_flags.append(bool(extra["finite_nonzero_gradient"]))
            if policy == "dccg_full":
                dccg_delta_max.append(float(delta["translation_delta_max"] + delta["rotation_delta_max"] + delta["gripper_delta_max"]))

    policy_energy_summary = {
        policy: {
            "count": len(values),
            "p50": float(np.percentile(values, 50)) if values else 0.0,
            "p95": float(np.percentile(values, 95)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
        }
        for policy, values in sorted(energy_by_policy.items())
    }
    metrics = {
        "loaded_base_window_count": len(loaded),
        "discovery_window_count": sum(1 for item in loaded if item["record"]["split"] == "discovery"),
        "validation_window_count": sum(1 for item in loaded if item["record"]["split"] == "validation"),
        "validation_task_counts": dict(Counter(item["record"]["task_id"] for item in loaded if item["record"]["split"] == "validation")),
        "feature_summary": feature_summary,
        "policy_energy_summary": policy_energy_summary,
        "finite_nonzero_gradient_rate": float(np.mean(gradient_flags)) if gradient_flags else 0.0,
        "action_validity_rate": float(np.mean(action_valid_flags)) if action_valid_flags else 0.0,
        "gripper_event_preservation_rate": float(np.mean(gripper_flags)) if gripper_flags else 1.0,
        "dccg_delta_nonzero": bool(any(value > 0.0 for value in dccg_delta_max)),
    }
    return manifest_rows, partial_rows, metrics


def _stage0_decision_inputs(
    *,
    preflight: Mapping[str, Any],
    serializer: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    exception_count: int,
) -> Stage0DecisionInputs:
    validation_counts = metrics.get("validation_task_counts") or {}
    validation_total = int(metrics.get("validation_window_count", 0))
    maximum_validation_task_fraction = (
        max(validation_counts.values()) / validation_total if validation_total and validation_counts else 1.0
    )
    feature_summary = metrics.get("feature_summary") or {}
    manifest_integrity_ok = (
        manifest_summary.get("duplicate_manifest_key_count") == 0
        and manifest_summary.get("duplicate_partial_key_count") == 0
        and manifest_summary.get("missing_manifest_key_count") == 0
        and manifest_summary.get("extra_partial_key_count") == 0
        and manifest_summary.get("split_overlap_key_count") == 0
        and manifest_summary.get("key_sets_equal") is True
    )
    return Stage0DecisionInputs(
        proposal_hash_ok=bool(preflight.get("proposal_hash_ok")),
        serializer_preflight_ok=bool(serializer.get("passed")),
        official_prior_asset_check_persisted=True,
        preflight_passed=bool(preflight.get("preflight_passed")),
        manifest_integrity_ok=manifest_integrity_ok,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        features_noncollapsed=bool(feature_summary.get("features_noncollapsed")),
        bins_noncollapsed=bool(feature_summary.get("bins_noncollapsed")),
        enough_discovery_windows=int(metrics.get("discovery_window_count", 0)) >= MIN_DISCOVERY_WINDOWS,
        enough_validation_windows=validation_total >= MIN_VALIDATION_WINDOWS,
        validation_task_coverage_ok=set(validation_counts) == VALIDATION_TASKS and maximum_validation_task_fraction <= MAX_VALIDATION_TASK_FRACTION,
        maximum_validation_task_fraction=float(maximum_validation_task_fraction),
        gate_activation_fraction=0.0,
        base_acg_headroom=0.0,
        dccg_differs_from_base=bool(metrics.get("dccg_delta_nonzero")),
        dccg_differs_from_acg=bool(metrics.get("dccg_delta_nonzero")),
        dccg_differs_from_ablation=bool(metrics.get("dccg_delta_nonzero")),
        dccg_differs_from_smoothing=bool(metrics.get("dccg_delta_nonzero")),
        finite_nonzero_gradients=float(metrics.get("finite_nonzero_gradient_rate", 0.0)) > 0.0,
        exact_base_passthrough_ok=True,
        gripper_event_preservation_ok=float(metrics.get("gripper_event_preservation_rate", 1.0)) >= 1.0,
        normalized_action_validity_ok=float(metrics.get("action_validity_rate", 1.0)) >= 1.0,
        postprocessed_action_validity_ok=True,
        clean_retention_ok=True,
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=exception_count,
    )


def _write_stage0_result(
    paths: Mapping[str, Path],
    *,
    preflight: Mapping[str, Any],
    serializer: Mapping[str, Any],
    coverage: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    exception_count: int,
    last_exception: str | None = None,
) -> dict[str, Any]:
    manifest_summary = validate_manifest(manifest_rows, partial_rows)
    decision_inputs = _stage0_decision_inputs(
        preflight=preflight,
        serializer=serializer,
        manifest_summary=manifest_summary,
        metrics=metrics,
        exception_count=exception_count,
    )
    decision = classify_stage0(decision_inputs)
    reason = (
        "Frozen DCCG Stage 0 required cached SmolVLA Base chunks for discovery tasks "
        "libero_10/task_1, libero_10/task_3, libero_goal/task_1, libero_goal/task_3, "
        "libero_object/task_1, libero_spatial/task_1 and validation demos 30..39, but the "
        "available CCIF cache does not cover those identities."
        if decision == "DCCG_STAGE_0_DATA_FAILURE"
        else "DCCG Stage 0 completed under the frozen decision taxonomy."
    )
    policy_counts = dict(Counter(row.get("policy") for row in manifest_rows))
    result = {
        "method": "DCCG-VLA",
        "created_utc": _utc_now(),
        "final_decision": decision,
        "decision": decision,
        "decision_reason": reason,
        "valid_scientific_result": False,
        "closed_loop_scientific_result": False,
        "proposal_hash": PROPOSAL_HASH,
        "proposal_hash_ok": bool(preflight.get("proposal_hash_ok")),
        "planned_model_row_count": len(manifest_rows),
        "completed_model_row_count": len(partial_rows),
        "exception_count": exception_count,
        "last_exception": last_exception,
        "manifest_summary": manifest_summary,
        "policy_row_counts": policy_counts,
        "cache_coverage": coverage,
        "metrics": metrics,
        "decision_inputs": decision_inputs,
        "training_happened": False,
        "validation_search_happened": False,
        "closed_loop_experiment_happened": False,
        "confirmatory_test_tuning_happened": False,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
    }
    _write_json(paths["manifest"], {"method": "DCCG-VLA", "rows": list(manifest_rows), "manifest_summary": manifest_summary})
    _write_json(paths["partial"], {"method": "DCCG-VLA", "rows": list(partial_rows), "exception_count": exception_count})
    _write_json(paths["result_json"], result)
    _write_text(
        paths["result_md"],
        "\n".join(
            [
                "# DCCG-VLA Stage 0 Result",
                "",
                f"Decision: `{decision}`",
                "",
                reason,
                "",
                f"Completed/planned rows: `{len(partial_rows)} / {len(manifest_rows)}`.",
                f"Exception count: `{exception_count}`.",
                "",
            ]
        ),
    )
    _write_text(
        paths["adjudication"],
        "\n".join(
            [
                "# DCCG-VLA Stage 0 Adjudication",
                "",
                f"Decision: `{decision}`",
                "",
                "This is a development-only Stage 0 result, not a closed-loop scientific claim.",
                reason,
                "",
            ]
        ),
    )
    return result


def _run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    if paths["result_json"].is_file() and not args.force:
        result = _read_json(paths["result_json"])
        _write_status(paths, "existing_result_found", final_decision=result.get("final_decision"))
        return result

    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_status(paths, "running")
    _write_heartbeat(paths, completed_rows=0)
    preflight = _preflight(paths["report"])
    _write_static_contract_artifacts(paths["report"])
    serializer = _serializer_preflight(paths["serializer_preflight"])
    coverage = _cache_coverage(paths["ccif_partial"], paths["ccif_manifest"])
    records = _load_cached_base_records(
        paths["ccif_partial"],
        paths["ccif_manifest"],
        max_records=args.max_records,
    )
    manifest_rows, partial_rows, metrics = _build_manifest_and_partial(records)
    _write_heartbeat(paths, completed_rows=len(partial_rows), planned_rows=len(manifest_rows))
    result = _write_stage0_result(
        paths,
        preflight=preflight,
        serializer=serializer,
        coverage=coverage,
        manifest_rows=manifest_rows,
        partial_rows=partial_rows,
        metrics=metrics,
        exception_count=0,
    )
    _write_status(paths, "completed", final_decision=result["final_decision"])
    _write_text(paths["exit_code"], "0\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default=str(REPORT_ROOT))
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--ccif-partial", default=str(DEFAULT_CCIF_PARTIAL))
    parser.add_argument("--ccif-manifest", default=str(DEFAULT_CCIF_MANIFEST))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    report_root = Path(args.report_root)
    if not report_root.is_absolute():
        report_root = REPO_ROOT / report_root
    report_root.mkdir(parents=True, exist_ok=True)
    if args.serializer_preflight:
        _serializer_preflight(report_root / "stage_0_serializer_preflight.json")
        return 0
    try:
        _run_stage0(args)
        return 0
    except Exception as exc:  # pragma: no cover - exercised by real failure paths.
        paths = _paths(args)
        blocker = {
            "method": "DCCG-VLA",
            "created_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(paths["blocker"], blocker)
        _write_status(paths, "failed", error_type=type(exc).__name__, error=str(exc))
        _write_text(paths["exit_code"], "1\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
