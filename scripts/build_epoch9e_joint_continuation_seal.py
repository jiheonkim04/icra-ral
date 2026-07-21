#!/usr/bin/env python3
"""Seal the integrity-only Epoch 9E continuation before opening pending outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "epoch9e_joint_continuation_execution_seal.json"
SCOPE_JSON = REPORTS / "epoch9e_failfast_repair_scope_justification.json"
SCOPE_MD = REPORTS / "epoch9e_failfast_repair_scope_justification.md"
AUTHORITY = REPORTS / "epoch9e_failfast_continuation_authorization.json"
CORRECTION = REPORTS / "epoch9e_failfast_root_cause_and_scope_correction.json"
SENSITIVITY = REPORTS / "epoch9e_missing_pair_sensitivity_protocol.json"
SCHEDULE = REPORTS / "epoch9e_continuation_schedule_audit.json"
PROTOCOL = REPORTS / "epoch9e_joint_certification_protocol.json"
ORIGINAL_RESULT = REPORTS / "epoch9e_joint_certification/result.json"
ORIGINAL_ADJUDICATION = REPORTS / "epoch9e_joint_certification_adjudication.json"
ORIGINAL_SEAL = REPORTS / "epoch9e_joint_execution_seal.json"
CONTROLLER = ROOT / "scripts/epoch9e_nondrag_controller.py"
ORIGINAL_RUNNER = ROOT / "scripts/run_epoch9e_joint_certification.py"
CONTINUATION_RUNNER = ROOT / "scripts/run_epoch9e_joint_continuation.py"
CONTINUATION_ADJUDICATOR = ROOT / "scripts/adjudicate_epoch9e_joint_continuation.py"
CONTINUATION_HOST = ROOT / "scripts/run_epoch9e_joint_continuation_host.ps1"


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


def committed_sha256(path: Path) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative(path)}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest().upper()


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if any(path.exists() for path in (OUTPUT, SCOPE_JSON, SCOPE_MD)):
        raise FileExistsError("refusing to overwrite continuation seal or scope report")
    forbidden = [REPORTS / "epoch9e_joint_continuation/result.json", REPORTS / "epoch9e_joint_continuation_adjudication.json"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("cannot seal after a continuation result or adjudication exists")
    inputs = (AUTHORITY, CORRECTION, SENSITIVITY, SCHEDULE, PROTOCOL, ORIGINAL_RESULT, ORIGINAL_ADJUDICATION, ORIGINAL_SEAL, CONTROLLER, ORIGINAL_RUNNER, CONTINUATION_RUNNER, CONTINUATION_ADJUDICATOR, CONTINUATION_HOST)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != committed_sha256(path):
            raise RuntimeError(f"seal input differs from HEAD: {relative(path)}")
    authority, correction, sensitivity, schedule, protocol = load(AUTHORITY), load(CORRECTION), load(SENSITIVITY), load(SCHEDULE), load(PROTOCOL)
    if authority["allowance_consumed"] is not False or authority["controller_or_scientific_change_authorized"] is not False:
        raise RuntimeError("invalid explicit authorization boundary")
    if correction["frozen_hashes"]["controller"]["sha256"] != sha256(CONTROLLER):
        raise RuntimeError("controller changed after interruption")
    if correction["frozen_hashes"]["original_runner"]["sha256"] != sha256(ORIGINAL_RUNNER):
        raise RuntimeError("original scientific runner changed after interruption")
    if correction["frozen_hashes"]["interrupted_result"]["sha256"] != sha256(ORIGINAL_RESULT):
        raise RuntimeError("interrupted result changed")
    pending_trace_names = [path.name for path in (REPORTS / "epoch9e_joint_certification/traces").glob("*.npz") if any(str(identity) in path.name for identity in range(20261135, 20261146))]
    if pending_trace_names:
        raise RuntimeError(f"pending outcomes were opened before continuation seal: {pending_trace_names}")
    existing_traces = correction["frozen_hashes"]["existing_traces"]
    if any(sha256(ROOT / row["path"]) != row["sha256"] for row in existing_traces):
        raise RuntimeError("immutable existing trace changed")
    config = protocol["controller_contract"]["base_controller_config"]
    scientific_contract = {
        "controller_name": protocol["controller_contract"]["name"],
        "controller_config": config,
        "contact_transition_threshold_pixels": protocol["controller_contract"]["contact_trigger"]["visual_displacement_pixels_at_least"],
        "primary_score": protocol["controller_contract"]["primary_score"],
        "response_window": protocol["controller_contract"]["response_window"],
        "joint_go": protocol["joint_go"],
        "paired_test": protocol["paired_test"],
        "position_order_control": protocol["position_order_control"],
        "sham_required": protocol["sham_control"]["required"],
    }
    scientific_contract_sha = hashlib.sha256(json.dumps(scientific_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()
    source_checkpoint = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    seal = {
        "schema_version": "epoch9e.joint_continuation_execution_seal.v1",
        "sealed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "starting_checkpoint": authority["starting_checkpoint"],
        "source_checkpoint": source_checkpoint,
        "explicit_authority_path": relative(AUTHORITY),
        "explicit_authority_sha256": sha256(AUTHORITY),
        "authority_document_sha256": authority["authority_sha256"],
        "integrity_only_repair_allowance_consumed": True,
        "controller_or_scientific_repair_used": False,
        "joint_protocol_path": relative(PROTOCOL),
        "joint_protocol_sha256": sha256(PROTOCOL),
        "scientific_contract": scientific_contract,
        "scientific_contract_sha256": scientific_contract_sha,
        "controller_path": relative(CONTROLLER),
        "controller_sha256": sha256(CONTROLLER),
        "original_runner_path": relative(ORIGINAL_RUNNER),
        "original_runner_sha256": sha256(ORIGINAL_RUNNER),
        "continuation_runner_path": relative(CONTINUATION_RUNNER),
        "continuation_runner_sha256": sha256(CONTINUATION_RUNNER),
        "continuation_adjudicator_path": relative(CONTINUATION_ADJUDICATOR),
        "continuation_adjudicator_sha256": sha256(CONTINUATION_ADJUDICATOR),
        "continuation_host_path": relative(CONTINUATION_HOST),
        "continuation_host_sha256": sha256(CONTINUATION_HOST),
        "continuation_schedule_path": relative(SCHEDULE),
        "continuation_schedule_sha256": sha256(SCHEDULE),
        "missing_pair_sensitivity_path": relative(SENSITIVITY),
        "missing_pair_sensitivity_sha256": sha256(SENSITIVITY),
        "root_cause_correction_path": relative(CORRECTION),
        "root_cause_correction_sha256": sha256(CORRECTION),
        "interrupted_result_path": relative(ORIGINAL_RESULT),
        "interrupted_result_sha256": sha256(ORIGINAL_RESULT),
        "interrupted_adjudication_path": relative(ORIGINAL_ADJUDICATION),
        "interrupted_adjudication_sha256": sha256(ORIGINAL_ADJUDICATION),
        "original_execution_seal_path": relative(ORIGINAL_SEAL),
        "original_execution_seal_sha256": sha256(ORIGINAL_SEAL),
        "immutable_existing_traces": [{"path": row["path"], "sha256": row["sha256"]} for row in existing_traces],
        "authorized_exception_prefix": "RuntimeError: trace does not contain the frozen five-step response window: ",
        "pending_primary_row_keys": [row["row_key"] for row in schedule["pending_primary_rows"]],
        "pending_sham_row_keys": [row["row_key"] for row in schedule["pending_sham_rows"]],
        "never_recompute_keys": schedule["never_recompute_keys"],
        "runtime": {"serial": True, "environments_at_once": 1, "models_at_once": 0, "host_ram_ceiling_percent": 82.0, "wsl_swap_used_peak_bytes": 0, "missing_key_resume": True},
        "pending_outcomes_accessed_before_seal": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT, seal)
    scope = {
        "schema_version": "epoch9e.failfast_repair_scope_justification.v1",
        "recorded_at": seal["sealed_at"],
        "starting_checkpoint": authority["starting_checkpoint"],
        "authority_freeze_checkpoint": "db1b10b",
        "wrapper_repair_checkpoint": "7368b56",
        "seal_source_checkpoint": source_checkpoint,
        "new_runtime_files": [relative(CONTINUATION_RUNNER), relative(CONTINUATION_ADJUDICATOR), relative(CONTINUATION_HOST)],
        "historical_files_modified": [],
        "scientific_hashes_unchanged": {"controller": seal["controller_sha256"], "original_runner": seal["original_runner_sha256"], "protocol": seal["joint_protocol_sha256"], "scientific_contract": seal["scientific_contract_sha256"]},
        "allowed_behavior_change": "finalize exactly classified missing frozen response windows and continue subsequent untouched keys",
        "other_exceptions_swallowed": False,
        "base_20261134_rerun": False,
        "pending_outcomes_accessed_before_seal": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(SCOPE_JSON, scope)
    atomic_write_text(SCOPE_MD, f"""# Epoch 9E Fail-Fast Repair Scope Justification

The wrapper-only repair is committed at `7368b56` and sealed from source checkpoint `{source_checkpoint}`. It adds three new runtime files and modifies no historical scientific file. The controller, original scientific runner, protocol, threshold, score, response window, endpoints, gates, identities, interrupted result, and four existing traces retain their recorded hashes.

The only behavior change is that the exact missing-frozen-response-window condition becomes a finalized scientific miss and the wrapper advances to the next untouched committed key. Every other returned failure or raised infrastructure exception stops safely. Base `20261134` is never rerun. The one explicit integrity-repair allowance is consumed by `{relative(OUTPUT)}`.
""")
    print(json.dumps({"output": relative(OUTPUT), "source_checkpoint": source_checkpoint, "controller": seal["controller_sha256"], "scientific_contract": scientific_contract_sha, "pending_primary": len(seal["pending_primary_row_keys"]), "pending_sham": len(seal["pending_sham_row_keys"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
