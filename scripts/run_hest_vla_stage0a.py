"""Run the frozen HEST-VLA Stage 0A action-source and algebra audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.hest_vla import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    CURVATURE_LAMBDA,
    HORIZON,
    PROPOSAL_HASH,
    Stage0ADecisionInputs,
    canonical_json_sha256,
    chunk_sha256,
    classify_stage0a,
    cumulative_arm_energy,
    gripper_transition,
    hest_transform,
    moving_average_control,
    no_endpoint_ablation,
    parse_sha256_registry,
    spline_proxy,
    support_bounds,
    support_valid,
    validate_manifest,
)


EXPECTED_ROWS = 160
EXPECTED_DISCOVERY_ROWS = 128
EXPECTED_VALIDATION_ROWS = 32
WINDOW_FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
PROPOSAL_FILE = REPO_ROOT / "reports" / "hest_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "hest_vla" / "proposal_hash.txt"

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "chunks": run / "chunks",
        "data": Path(args.data_root),
        "pid": report / "stage_0a_pid.txt",
        "heartbeat": report / "stage_0a_heartbeat.json",
        "status": report / "stage_0a_status.json",
        "partial": report / "stage_0a_partial.json",
        "manifest": report / "stage_0a_pair_manifest.json",
        "preflight": report / "stage_0a_preflight.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "blocker": report / "stage_0a_implementation_blocker.json",
    }


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(5.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def _proposal_hash_from_registry() -> str:
    return parse_sha256_registry(PROPOSAL_HASH_FILE.read_text(encoding="utf-8"))


def _selected_starts(episode_length: int) -> list[int]:
    maximum = int(episode_length) - HORIZON
    if maximum < 0:
        return []
    selected: list[int] = []
    for fraction in WINDOW_FRACTIONS:
        start = int(np.floor(float(fraction) * maximum + 0.5))
        if start not in selected:
            selected.append(start)
    return selected


def _window_key(row: Mapping[str, Any]) -> str:
    fields = (
        row["partition"],
        row["suite"],
        row["task_identity"],
        row["source_path"],
        row["demo_id"],
        row["start"],
        row["stop"],
    )
    return "|".join(str(value) for value in fields)


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for suite, task_identity, relative in TASK_SOURCES:
        source = data_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_record = {
            "suite": suite,
            "task_identity": task_identity,
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "edge_sha256": _edge_hash(source),
        }
        with h5py.File(source, "r") as handle:
            demos = handle["data"]
            for partition, demo_ids in (("discovery", range(0, 8)), ("validation", range(8, 10))):
                for demo_id in demo_ids:
                    demo_key = f"demo_{demo_id}"
                    if demo_key not in demos:
                        raise KeyError(f"missing {demo_key} in {source}")
                    actions = np.asarray(demos[demo_key]["actions"], dtype=np.float64)
                    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
                        raise ValueError(f"invalid actions for {source}:{demo_key}: {actions.shape}")
                    starts = _selected_starts(int(actions.shape[0]))
                    if len(starts) != 4:
                        raise ValueError(
                            f"{source}:{demo_key} has {len(starts)} unique frozen starts, expected four"
                        )
                    for slot, start in enumerate(starts):
                        stop = start + HORIZON
                        action_chunk = actions[start:stop]
                        row: dict[str, Any] = {
                            "partition": partition,
                            "suite": suite,
                            "task_identity": task_identity,
                            "source_path": str(source),
                            "source_edge_sha256": source_record["edge_sha256"],
                            "demo_id": demo_id,
                            "demo_key": demo_key,
                            "episode_length": int(actions.shape[0]),
                            "window_slot": slot,
                            "start": int(start),
                            "stop": int(stop),
                            "action_shape": [HORIZON, ACTION_DIM],
                            "action_sha256": chunk_sha256(action_chunk),
                        }
                        row["window_key"] = _window_key(row)
                        rows.append(row)
        sources.append(source_record)
    expected_counts = {
        "discovery": sum(row["partition"] == "discovery" for row in rows),
        "validation": sum(row["partition"] == "validation" for row in rows),
    }
    if len(rows) != EXPECTED_ROWS or expected_counts != {
        "discovery": EXPECTED_DISCOVERY_ROWS,
        "validation": EXPECTED_VALIDATION_ROWS,
    }:
        raise RuntimeError(f"unexpected manifest counts: total={len(rows)}, partitions={expected_counts}")
    return rows, sources


def _read_chunk(row: Mapping[str, Any]) -> np.ndarray:
    with h5py.File(str(row["source_path"]), "r") as handle:
        actions = handle["data"][str(row["demo_key"])]["actions"]
        chunk = np.asarray(actions[int(row["start"]) : int(row["stop"])], dtype=np.float64)
    if chunk_sha256(chunk) != row["action_sha256"]:
        raise RuntimeError(f"source action hash mismatch for {row['window_key']}")
    return chunk


def _load_resume(path: Path, *, manifest_hash: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    partial = _read_json(path)
    if partial.get("method") != "HEST-VLA" or partial.get("stage") != "0A":
        raise RuntimeError("partial method/stage mismatch")
    if partial.get("proposal_hash") != PROPOSAL_HASH:
        raise RuntimeError("partial proposal hash mismatch")
    if partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial manifest hash mismatch")
    rows = list(partial.get("rows") or [])
    keys = [str(row["window_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("partial contains duplicate window keys")
    return rows


def _persist_partial(
    path: Path,
    *,
    manifest_hash: str,
    rows: Sequence[Mapping[str, Any]],
    exception_count: int = 0,
    last_exception: str | None = None,
) -> None:
    _write_json(
        path,
        {
            "method": "HEST-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "manifest_hash": manifest_hash,
            "planned_window_count": EXPECTED_ROWS,
            "completed_window_count": len(rows),
            "completed_window_keys": [str(row["window_key"]) for row in rows],
            "exception_count": int(exception_count),
            "last_exception": last_exception,
            "rows": list(rows),
            "updated_at": _utc_now(),
        },
    )


def _process_row(
    row: Mapping[str, Any],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    chunk_dir: Path,
) -> dict[str, Any]:
    base = _read_chunk(row)
    hest, fallback = hest_transform(base, alpha=1.0, lower=lower, upper=upper)
    prior = spline_proxy(base, alpha=1.0)
    no_endpoint = no_endpoint_ablation(base, alpha=1.0)
    moving = moving_average_control(base)

    key_hash = hashlib.sha256(str(row["window_key"]).encode("utf-8")).hexdigest().upper()
    chunk_path = chunk_dir / f"{key_hash}.npy"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(chunk_path, hest, allow_pickle=False)
    reloaded = np.load(chunk_path, allow_pickle=False)

    base_energy = cumulative_arm_energy(base)
    hest_energy = cumulative_arm_energy(hest)
    reduction = 0.0 if base_energy <= 1e-30 else 1.0 - hest_energy / base_energy
    endpoint_error = float(np.max(np.abs(hest[:, :ARM_DIM].sum(axis=0) - base[:, :ARM_DIM].sum(axis=0))))
    first_action_error = float(np.max(np.abs(hest[0, :ARM_DIM] - base[0, :ARM_DIM])))
    gripper_error = float(np.max(np.abs(hest[:, 6] - base[:, 6])))
    outputs = {
        "base": base,
        "spline_proxy": prior,
        "hest": hest,
        "no_endpoint": no_endpoint,
        "moving_average": moving,
    }
    return {
        "window_key": row["window_key"],
        "partition": row["partition"],
        "suite": row["suite"],
        "task_identity": row["task_identity"],
        "source_path": row["source_path"],
        "demo_id": row["demo_id"],
        "start": row["start"],
        "stop": row["stop"],
        "source_action_sha256": row["action_sha256"],
        "output_sha256": {name: chunk_sha256(value) for name, value in outputs.items()},
        "source_shape_valid": list(base.shape) == [HORIZON, ACTION_DIM],
        "source_finite_fraction": float(np.isfinite(base).mean()),
        "gripper_transition": gripper_transition(base),
        "fallback_reason": fallback,
        "endpoint_max_abs_error": endpoint_error,
        "first_action_max_abs_error": first_action_error,
        "gripper_max_abs_error": gripper_error,
        "support_valid": {name: support_valid(value, lower, upper) for name, value in outputs.items()},
        "base_energy": base_energy,
        "hest_energy": hest_energy,
        "hest_energy_reduction": float(reduction),
        "hest_base_arm_max_abs_delta": float(np.max(np.abs(hest[:, :ARM_DIM] - base[:, :ARM_DIM]))),
        "hest_spline_proxy_max_abs_delta": float(np.max(np.abs(hest - prior))),
        "hest_no_endpoint_max_abs_delta": float(np.max(np.abs(hest - no_endpoint))),
        "hest_moving_average_max_abs_delta": float(np.max(np.abs(hest - moving))),
        "roundtrip_max_abs_error": float(np.max(np.abs(reloaded - hest))),
        "persisted_chunk_path": str(chunk_path),
        "persisted_chunk_size_bytes": chunk_path.stat().st_size,
        "persisted_chunk_sha256": _sha256(chunk_path),
    }


def _summarize(
    manifest_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    proposal_hash_ok: bool,
    exception_count: int,
) -> tuple[dict[str, Any], Stage0ADecisionInputs]:
    audit = validate_manifest(manifest_rows, rows)
    validation = [row for row in rows if row["partition"] == "validation"]
    transitions = sum(bool(row["gripper_transition"]) for row in validation)
    acting_fraction = float(np.mean([float(row["hest_base_arm_max_abs_delta"]) > 1e-8 for row in validation]))
    energy_reductions = [float(row["hest_energy_reduction"]) for row in validation]
    comparator_max = {
        "spline_proxy": max(float(row["hest_spline_proxy_max_abs_delta"]) for row in validation),
        "no_endpoint": max(float(row["hest_no_endpoint_max_abs_delta"]) for row in validation),
        "moving_average": max(float(row["hest_moving_average_max_abs_delta"]) for row in validation),
    }
    source_ok = all(bool(row["source_shape_valid"]) and float(row["source_finite_fraction"]) == 1.0 for row in rows)
    all_support = all(all(bool(value) for value in row["support_valid"].values()) for row in rows)
    manifest_ok = (
        audit["manifest_row_count"] == EXPECTED_ROWS
        and audit["partial_row_count"] == EXPECTED_ROWS
        and audit["duplicate_manifest_key_count"] == 0
        and audit["duplicate_partial_key_count"] == 0
        and audit["missing_manifest_key_count"] == 0
        and audit["extra_partial_key_count"] == 0
        and audit["partition_overlap_count"] == 0
        and bool(audit["key_sets_equal"])
    )
    inputs = Stage0ADecisionInputs(
        proposal_hash_ok=proposal_hash_ok,
        manifest_audit_ok=manifest_ok,
        source_finite_shape_ok=source_ok,
        arm_support_noncollapsed=bool(np.all((upper[:ARM_DIM] - lower[:ARM_DIM]) > 1e-8)),
        validation_transition_count=transitions,
        endpoint_max_error=max(float(row["endpoint_max_abs_error"]) for row in rows),
        first_action_max_error=max(float(row["first_action_max_abs_error"]) for row in rows),
        gripper_max_error=max(float(row["gripper_max_abs_error"]) for row in rows),
        all_variant_support_valid=all_support,
        acting_fraction=acting_fraction,
        median_energy_reduction=float(np.median(energy_reductions)),
        comparator_distinct=all(value > 1e-10 for value in comparator_max.values()),
        roundtrip_max_error=max(float(row["roundtrip_max_abs_error"]) for row in rows),
        exception_count=exception_count,
    )
    summary = {
        "manifest_audit": audit,
        "source_finite_shape_ok": source_ok,
        "discovery_support_lower": lower,
        "discovery_support_upper": upper,
        "discovery_arm_ranges": upper[:ARM_DIM] - lower[:ARM_DIM],
        "arm_support_noncollapsed": inputs.arm_support_noncollapsed,
        "validation_transition_count": transitions,
        "validation_nontransition_count": len(validation) - transitions,
        "endpoint_max_error": inputs.endpoint_max_error,
        "first_action_max_error": inputs.first_action_max_error,
        "gripper_max_error": inputs.gripper_max_error,
        "all_variant_support_valid": all_support,
        "fallback_counts": {
            str(reason): sum(row["fallback_reason"] == reason for row in rows)
            for reason in sorted({row["fallback_reason"] for row in rows}, key=lambda value: str(value))
        },
        "acting_fraction": acting_fraction,
        "median_energy_reduction": inputs.median_energy_reduction,
        "min_energy_reduction": min(energy_reductions),
        "comparator_max_abs_delta": comparator_max,
        "comparator_distinct": inputs.comparator_distinct,
        "roundtrip_max_error": inputs.roundtrip_max_error,
    }
    return summary, inputs


def _result_markdown(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    audit = summary["manifest_audit"]
    return "\n".join(
        [
            "# HEST-VLA Stage 0A Result",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            f"Proposal hash: `{result['proposal_hash']}`.",
            "",
            f"Windows completed: `{result['completed_window_count']} / {result['planned_window_count']}`.",
            "",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            f"Duplicate partial keys: `{audit['duplicate_partial_key_count']}`; missing: `{audit['missing_manifest_key_count']}`; extra: `{audit['extra_partial_key_count']}`.",
            "",
            f"Validation gripper-transition windows: `{summary['validation_transition_count']} / {EXPECTED_VALIDATION_ROWS}`.",
            "",
            f"HEST acting fraction: `{summary['acting_fraction']}`.",
            "",
            f"Median cumulative-arm energy reduction: `{summary['median_energy_reduction']}`.",
            "",
            f"Endpoint / first-action / gripper max errors: `{summary['endpoint_max_error']}` / `{summary['first_action_max_error']}` / `{summary['gripper_max_error']}`.",
            "",
            f"All variant support valid: `{summary['all_variant_support_valid']}`.",
            "",
            "This CPU-only gate read no confirmatory reset identity, reward, success, done flag, video, or SmolVLA output. It is not a closed-loop scientific result.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    _write_text(paths["pid"], f"{pid}\n")

    proposal_hash_recomputed = _sha256(PROPOSAL_FILE)
    proposal_hash_registry = _proposal_hash_from_registry()
    proposal_hash_ok = proposal_hash_recomputed == PROPOSAL_HASH == proposal_hash_registry
    if not proposal_hash_ok:
        raise RuntimeError("frozen proposal hash mismatch")
    if paths["result_json"].is_file():
        existing = _read_json(paths["result_json"])
        if existing.get("final_decision"):
            raise RuntimeError("completed Stage 0A result already exists; refusing duplicate execution")

    manifest_rows, sources = _build_manifest(paths["data"])
    manifest_payload = {
        "method": "HEST-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "planned_window_count": EXPECTED_ROWS,
        "partition_counts": {"discovery": EXPECTED_DISCOVERY_ROWS, "validation": EXPECTED_VALIDATION_ROWS},
        "sources": sources,
        "rows": manifest_rows,
    }
    manifest_hash = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_hash"] = manifest_hash
    _write_json(paths["manifest"], manifest_payload)

    discovery_chunks = [_read_chunk(row) for row in manifest_rows if row["partition"] == "discovery"]
    lower, upper = support_bounds(discovery_chunks)
    rows = _load_resume(paths["partial"], manifest_hash=manifest_hash)
    completed = {str(row["window_key"]) for row in rows}

    state = {
        "method": "HEST-VLA",
        "stage": "0A",
        "pid": pid,
        "status": "running",
        "phase": "action_audit",
        "planned_window_count": EXPECTED_ROWS,
        "completed_window_count": len(rows),
        "exception_count": 0,
    }
    _write_json(paths["preflight"], {
        "proposal_hash_recomputed": proposal_hash_recomputed,
        "proposal_hash_registry": proposal_hash_registry,
        "proposal_hash_ok": proposal_hash_ok,
        "manifest_hash": manifest_hash,
        "planned_window_count": EXPECTED_ROWS,
        "resumed_window_count": len(rows),
        "confirmatory_inputs_read": 0,
        "smolvla_loaded": False,
        "cuda_loaded": False,
        "simulator_loaded": False,
        "started_at": _utc_now(),
    })
    _write_json(paths["status"], {**state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})

    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], state, stop), daemon=True)
    heartbeat.start()
    try:
        for index, manifest_row in enumerate(manifest_rows, start=1):
            if manifest_row["window_key"] in completed:
                continue
            processed = _process_row(manifest_row, lower=lower, upper=upper, chunk_dir=paths["chunks"])
            rows.append(processed)
            completed.add(str(processed["window_key"]))
            state["completed_window_count"] = len(rows)
            _persist_partial(paths["partial"], manifest_hash=manifest_hash, rows=rows)
            _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
            print(f"[hest-stage0a] completed {len(rows)}/{EXPECTED_ROWS}", flush=True)

        summary, decision_inputs = _summarize(
            manifest_rows,
            rows,
            lower=lower,
            upper=upper,
            proposal_hash_ok=proposal_hash_ok,
            exception_count=0,
        )
        decision = classify_stage0a(decision_inputs)
        result = {
            "method": "HEST-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "manifest_hash": manifest_hash,
            "worker_pid": pid,
            "worker_alive_at_adjudication": True,
            "planned_window_count": EXPECTED_ROWS,
            "completed_window_count": len(rows),
            "resumed_window_count": int(_read_json(paths["preflight"])["resumed_window_count"]),
            "exception_count": 0,
            "final_decision": decision,
            "valid_scientific_result": False,
            "scientific_kill": False,
            "stage_0b_allowed": decision == "HEST_STAGE_0A_PASS_STAGE_0B_ALLOWED",
            "summary": summary,
            "confirmatory_reset_identity_read_count": 0,
            "reward_read_count": 0,
            "success_read_count": 0,
            "done_read_count": 0,
            "video_read_count": 0,
            "smolvla_load_count": 0,
            "simulator_load_count": 0,
            "cuda_load_count": 0,
            "timing_throughput_resource_evidence_eligible_for_paper": False,
            "completed_at": _utc_now(),
        }
        validation = {
            "proposal_hash_recomputed": proposal_hash_ok,
            "manifest_json_parsed": True,
            "partial_json_parsed": True,
            "manifest_hash_matches_partial": _read_json(paths["partial"])["manifest_hash"] == manifest_hash,
            **summary["manifest_audit"],
            "exception_count": 0,
            "final_decision": decision,
        }
        _write_json(paths["result_json"], result)
        _write_text(paths["result_md"], _result_markdown(result))
        _write_json(paths["validation"], validation)
        state.update({"status": "completed", "phase": "complete", "completed_window_count": len(rows)})
        _write_json(paths["status"], {**state, "completed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        return result
    except Exception as exc:
        detail = traceback.format_exc()
        _persist_partial(
            paths["partial"],
            manifest_hash=manifest_hash,
            rows=rows,
            exception_count=1,
            last_exception=detail,
        )
        blocker = {
            "method": "HEST-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": detail,
            "completed_window_count": len(rows),
            "planned_window_count": EXPECTED_ROWS,
            "scientific_kill": False,
            "failed_at": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        state.update({"status": "failed", "phase": "failed", "exception_count": 1})
        _write_json(paths["status"], {**state, "failed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "hest_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "hest_vla" / "stage0a"))
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
