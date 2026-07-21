#!/usr/bin/env python3
"""Integrity-only wrapper for untouched Epoch 9E joint schedule keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9e_joint_certification as frozen
from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9e_joint_certification_protocol.json"
ORIGINAL_RESULT_PATH = REPORTS / "epoch9e_joint_certification/result.json"
SEAL_PATH = REPORTS / "epoch9e_joint_continuation_execution_seal.json"
SCHEDULE_PATH = REPORTS / "epoch9e_continuation_schedule_audit.json"
OUTPUT_ROOT = REPORTS / "epoch9e_joint_continuation"
RESULT_PATH = OUTPUT_ROOT / "result.json"
MISSING_PREFIX = "RuntimeError: trace does not contain the frozen five-step response window: "
MISSING_CLASS = "FROZEN_RESPONSE_WINDOW_MISSING_SCIENTIFIC_MISS"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_key(kind: str, manifest: dict[str, Any]) -> str:
    return f"primary:{manifest['scene_id']}" if kind == "primary" else f"sham:{manifest['sham_id']}"


def is_missing_response_failure(row: dict[str, Any]) -> bool:
    return bool(
        row.get("completed") is False
        and isinstance(row.get("exception"), str)
        and row["exception"].startswith(MISSING_PREFIX)
    )


def trace_window_audit(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        actions = np.asarray(trace["action"], dtype=np.float64)
        contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
        verify = np.asarray(trace["rgb_displacement_pixels"], dtype=np.float64)[phase == "contact_verify_observe"]
    response_steps = int(np.count_nonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response"])))
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "steps": int(len(phase)),
        "response_window_steps": response_steps,
        "response_window_valid": response_steps == 5,
        "sampled_physical_contact": bool(np.any(contact)),
        "contact_verify_rgb_displacement_pixels_min": float(np.min(verify)) if len(verify) else None,
        "contact_verify_rgb_displacement_pixels_max": float(np.max(verify)) if len(verify) else None,
        "actions_finite_and_bounded": bool(np.isfinite(actions).all() and np.all(np.abs(actions) <= 1.0)),
    }


def structure_missing_response_failure(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if not is_missing_response_failure(row):
        raise ValueError("row is not the specifically authorized missing-response condition")
    expected_error_path = Path(row["exception"][len(MISSING_PREFIX):])
    if not expected_error_path.is_absolute():
        raise RuntimeError("classified missing response did not carry an absolute trace path")
    scene_id = manifest["scene_id"]
    trace_paths = [frozen.TRACE_ROOT / f"{scene_id}_{slot}.npz" for slot in manifest["probe_order"]]
    if any(not path.is_file() for path in trace_paths):
        raise RuntimeError("classified missing response did not preserve every attempted probe trace")
    audits = [trace_window_audit(path) for path in trace_paths]
    invalid = [audit for audit in audits if not audit["response_window_valid"]]
    if len(invalid) != 1 or (ROOT / invalid[0]["path"]).resolve() != expected_error_path.resolve():
        raise RuntimeError("missing-response classification does not match exactly one invalid trace")
    return {
        **row,
        "failure_class": MISSING_CLASS,
        "scientific_key_status": "FAILED_FINAL_NO_RECOMPUTE",
        "failure_is_infrastructure": False,
        "failure_is_controller_or_threshold_repairable": False,
        "probe_trace_audits": audits,
        "fixed_denominator_handling": {
            "assignment_rank": "incorrect",
            "completion_oracle": "failure_not_executed",
            "exact_pair_flip": "nonflip",
            "physical_contrast": "missing_not_imputed",
        },
    }


def process_schedule(
    schedule: Iterable[tuple[str, dict[str, Any]]],
    existing_keys: set[str],
    execute: Callable[[str, dict[str, Any]], dict[str, Any]],
    persist: Callable[[dict[str, Any]], None],
) -> None:
    """Run missing keys; continue only the exact frozen-window scientific miss."""

    for kind, manifest in schedule:
        key = row_key(kind, manifest)
        if key in existing_keys:
            continue
        row = execute(kind, manifest)
        if row.get("row_key") != key:
            raise RuntimeError(f"executor returned wrong scientific key: {row.get('row_key')} != {key}")
        if row.get("completed"):
            persist(row)
            existing_keys.add(key)
            continue
        if kind == "primary" and is_missing_response_failure(row):
            structured = structure_missing_response_failure(row, manifest)
            persist(structured)
            existing_keys.add(key)
            continue
        persist({**row, "failure_class": "UNAUTHORIZED_OR_INFRASTRUCTURE_FAILURE", "scientific_key_status": "FAILED_FINAL_NO_RECOMPUTE"})
        existing_keys.add(key)
        raise RuntimeError(f"non-authorized continuation failure stops safely: {key}: {row.get('exception')}")


def validate_seal() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seal, protocol, schedule = load(SEAL_PATH), load(PROTOCOL_PATH), load(SCHEDULE_PATH)
    bindings = {
        "protocol": sha256(PROTOCOL_PATH) == seal["joint_protocol_sha256"],
        "schedule": sha256(SCHEDULE_PATH) == seal["continuation_schedule_sha256"],
        "original_result": sha256(ORIGINAL_RESULT_PATH) == seal["interrupted_result_sha256"],
        "original_runner": sha256(ROOT / seal["original_runner_path"]) == seal["original_runner_sha256"],
        "controller": sha256(ROOT / seal["controller_path"]) == seal["controller_sha256"],
        "continuation_runner": sha256(Path(__file__)) == seal["continuation_runner_sha256"],
    }
    for trace in seal["immutable_existing_traces"]:
        bindings[f"trace:{trace['path']}"] = sha256(ROOT / trace["path"]) == trace["sha256"]
    if not all(bindings.values()):
        raise RuntimeError(f"continuation execution seal mismatch: {bindings}")
    if seal["integrity_only_repair_allowance_consumed"] is not True:
        raise RuntimeError("continuation repair allowance was not consumed by the seal")
    return seal, protocol, schedule


def manifest_rows(protocol: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    primary = [("primary", row) for row in protocol["assignments"] if int(row["base_identity_id"]) != 20261134]
    sham = [("sham", row) for row in protocol["sham_control"]["manifest"]]
    rows = primary + sham
    if len(primary) != 22 or len(sham) != 12:
        raise RuntimeError("continuation manifest is not the frozen untouched schedule")
    return rows


def summarize(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, int]:
    status = {row["row_key"]: row for row in rows}
    return {
        "scheduled_rows": len(manifest),
        "recorded_rows": len(rows),
        "completed_rows": sum(bool(row.get("completed")) for row in rows),
        "classified_missing_response_rows": sum(row.get("failure_class") == MISSING_CLASS for row in rows),
        "other_failed_rows": sum(not row.get("completed") and row.get("failure_class") != MISSING_CLASS for row in rows),
        "unexecuted_rows": sum(row["status"] == "NOT_EXECUTED" for row in manifest),
        "unique_recorded_keys": len(status),
    }


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--resume", action="store_true")
    resume = args.parse_args().resume
    seal, protocol, schedule_audit = validate_seal()
    del seal
    frozen_original = frozen.load(frozen.ORIGINAL_PROTOCOL_PATH)
    calibration = frozen.campaign.load_calibration()
    config = frozen.campaign.ControllerConfig(**protocol["controller_contract"]["base_controller_config"])
    bases = frozen.base_lookup(protocol)
    scheduled = manifest_rows(protocol)
    expected_keys = [row_key(kind, row) for kind, row in scheduled]
    if expected_keys != [row["row_key"] for row in schedule_audit["pending_primary_rows"] + schedule_audit["pending_sham_rows"]]:
        raise RuntimeError("continuation runner order differs from the committed schedule audit")
    if RESULT_PATH.exists():
        if not resume:
            raise FileExistsError("continuation result exists; missing-key resume flag is required")
        result = load(RESULT_PATH)
        if result["protocol_sha256"] != sha256(PROTOCOL_PATH) or result["execution_seal_sha256"] != sha256(SEAL_PATH):
            raise RuntimeError("continuation resume binding mismatch")
    else:
        if resume:
            raise FileNotFoundError("resume requested without a continuation result")
        result = {
            "schema_version": "epoch9e.joint_continuation_result.v1",
            "started_at": timestamp(),
            "pid": os.getpid(),
            "protocol_path": relative(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "execution_seal_path": relative(SEAL_PATH),
            "execution_seal_sha256": sha256(SEAL_PATH),
            "continuation_runner_sha256": sha256(Path(__file__)),
            "original_runner_path": relative(Path(frozen.__file__)),
            "original_runner_sha256": sha256(Path(frozen.__file__)),
            "interrupted_result_path": relative(ORIGINAL_RESULT_PATH),
            "interrupted_result_sha256": sha256(ORIGINAL_RESULT_PATH),
            "integrity_only_repair_allowance_consumed": True,
            "never_recompute_keys": schedule_audit["never_recompute_keys"],
            "manifest": [
                {
                    "row_key": row_key(kind, row),
                    "row_type": "PRIMARY_ASSIGNMENT" if kind == "primary" else "SHAM_CONTROL",
                    "base_identity_id": row["base_identity_id"],
                    "assignment": row["assignment"],
                    "status": "NOT_EXECUTED",
                }
                for kind, row in scheduled
            ],
            "rows": [],
            "resource_monitor": {
                "process_max_rss_bytes": 0,
                "wsl_mem_used_peak_bytes": 0,
                "wsl_swap_used_peak_bytes": 0,
                "gpu_initial": frozen.gpu_sample(),
            },
            "validation_accessed": False,
            "confirmation_accessed": False,
        }
        result["summary"] = summarize(result["rows"], result["manifest"])
        atomic_write_json(RESULT_PATH, result)
    keys = [row["row_key"] for row in result["rows"]]
    if len(keys) != len(set(keys)) or any(key not in expected_keys for key in keys):
        raise RuntimeError("continuation result contains duplicate or foreign keys")
    existing_keys = set(keys)
    env_class = frozen.campaign.load_env_class()

    def execute(kind: str, manifest: dict[str, Any]) -> dict[str, Any]:
        base = bases[int(manifest["base_identity_id"])]
        if kind == "primary":
            return frozen.run_primary(env_class, manifest, base, config, calibration, frozen_original, protocol["controller_contract"])
        return frozen.run_sham(env_class, manifest, base, config, calibration, frozen_original)

    def persist(row: dict[str, Any]) -> None:
        result["rows"].append(row)
        manifest_row = next(value for value in result["manifest"] if value["row_key"] == row["row_key"])
        manifest_row["status"] = "COMPLETED" if row.get("completed") else row.get("failure_class", "FAILED")
        frozen.update_resource_peaks(result)
        result["summary"] = summarize(result["rows"], result["manifest"])
        atomic_write_json(RESULT_PATH, result)
        if int(result["resource_monitor"]["wsl_swap_used_peak_bytes"]) != 0:
            raise RuntimeError("WSL swap use detected; continuation stopped with recorded key")

    process_schedule(scheduled, existing_keys, execute, persist)
    result["completed_at"] = timestamp()
    result["resource_monitor"]["gpu_final"] = frozen.gpu_sample()
    result["summary"] = summarize(result["rows"], result["manifest"])
    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
