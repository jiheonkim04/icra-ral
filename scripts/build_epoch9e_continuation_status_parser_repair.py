#!/usr/bin/env python3
"""Seal the non-scientific `0n` exit-status serialization/parser correction."""

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
RUN_DIR = REPORTS / "epoch9e_joint_continuation"
MONITOR = RUN_DIR / "host_resource_monitor_attempt_1.json"
RAW_STATUS = RUN_DIR / "runner_exit_code_attempt_1.txt"
STDOUT = RUN_DIR / "runner_stdout_attempt_1.log"
STDERR = RUN_DIR / "runner_stderr_attempt_1.log"
RESULT = RUN_DIR / "result.json"
SEAL = REPORTS / "epoch9e_joint_continuation_execution_seal.json"
ADJUDICATOR = ROOT / "scripts/adjudicate_epoch9e_joint_continuation.py"
HOST = ROOT / "scripts/run_epoch9e_joint_continuation_host.ps1"
OUTPUT_STATUS = REPORTS / "epoch9e_continuation_host_exit_status_correction.json"
OUTPUT_STATUS_MD = REPORTS / "epoch9e_continuation_host_exit_status_correction.md"
OUTPUT_PARSER = REPORTS / "epoch9e_continuation_adjudicator_parser_repair.json"
OUTPUT_PARSER_MD = REPORTS / "epoch9e_continuation_adjudicator_parser_repair.md"


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


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    outputs = (OUTPUT_STATUS, OUTPUT_STATUS_MD, OUTPUT_PARSER, OUTPUT_PARSER_MD)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite continuation status/parser repair")
    for path in (MONITOR, RAW_STATUS, STDOUT, STDERR, RESULT, SEAL, ADJUDICATOR, HOST):
        if not path.is_file():
            raise FileNotFoundError(path)
    monitor, result, seal = load(MONITOR), load(RESULT), load(SEAL)
    raw_status = RAW_STATUS.read_text(encoding="utf-8")
    stdout = STDOUT.read_text(encoding="utf-8")
    stderr = STDERR.read_text(encoding="utf-8")
    expected_summary = {
        "classified_missing_response_rows": 0,
        "completed_rows": 34,
        "other_failed_rows": 0,
        "recorded_rows": 34,
        "scheduled_rows": 34,
        "unexecuted_rows": 0,
        "unique_recorded_keys": 34,
    }
    if result["summary"] != expected_summary or result.get("completed_at") is None:
        raise RuntimeError("continuation runner did not complete the frozen schedule")
    if json.loads(stdout.strip()) != expected_summary:
        raise RuntimeError("runner stdout does not prove normal completed return")
    if raw_status != "0n":
        raise RuntimeError(f"unexpected raw status serialization: {raw_status!r}")
    if monitor["wsl_process_exit_code"] != 0 or monitor["scientific_result_sha256_after_runner"] != sha256(RESULT):
        raise RuntimeError("host process/result provenance does not support exit correction")
    if "Traceback" in stderr or "RuntimeError" in stderr or "MemoryError" in stderr:
        raise RuntimeError("stderr contains a genuine runner failure")
    if seal["continuation_adjudicator_sha256"] == sha256(ADJUDICATOR):
        raise RuntimeError("adjudicator parser repair was not applied")
    status = {
        "schema_version": "epoch9e.continuation_host_exit_status_correction.v1",
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "attempt": 1,
        "host_monitor_path": relative(MONITOR),
        "host_monitor_sha256": sha256(MONITOR),
        "raw_status_path": relative(RAW_STATUS),
        "raw_status_sha256": sha256(RAW_STATUS),
        "raw_status_text": raw_status,
        "recorded_authoritative_runner_exit_code": monitor["authoritative_runner_exit_code"],
        "wsl_process_exit_code": monitor["wsl_process_exit_code"],
        "runner_stdout_path": relative(STDOUT),
        "runner_stdout_sha256": sha256(STDOUT),
        "runner_stdout_summary": expected_summary,
        "runner_stderr_path": relative(STDERR),
        "runner_stderr_sha256": sha256(STDERR),
        "runner_stderr_contains_exception": False,
        "scientific_result_path": relative(RESULT),
        "scientific_result_sha256": sha256(RESULT),
        "scientific_result_completed_at": result["completed_at"],
        "corrected_authoritative_runner_exit_code": 0,
        "correction_basis": "raw `0n` is the normal numeric zero followed by a literal n from the host printf format; WSL process exit, runner stdout, final manifest, and result hash independently prove normal runner completion",
        "scientific_result_changed": False,
        "outcomes_recomputed_or_rerun": False,
        "host_ram_ceiling_breached": monitor["host_ram_ceiling_breached"],
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_STATUS, status)
    parser = {
        "schema_version": "epoch9e.continuation_adjudicator_parser_repair.v1",
        "recorded_at": status["recorded_at"],
        "source_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "original_sealed_adjudicator_sha256": seal["continuation_adjudicator_sha256"],
        "repaired_adjudicator_path": relative(ADJUDICATOR),
        "repaired_adjudicator_sha256": sha256(ADJUDICATOR),
        "executed_sealed_host_sha256": seal["continuation_host_sha256"],
        "future_repaired_host_path": relative(HOST),
        "future_repaired_host_sha256": sha256(HOST),
        "status_correction_path": relative(OUTPUT_STATUS),
        "status_correction_sha256": sha256(OUTPUT_STATUS),
        "repair_scope": "accept only the hash-bound attempt-1 raw status text `0n` as exit 0 when WSL process exit, completed stdout summary, result hash, and absence of exception all agree",
        "scientific_fields_changed": False,
        "threshold_score_endpoint_gate_or_missing_pair_rule_changed": False,
        "outcomes_recomputed_or_rerun": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_PARSER, parser)
    atomic_write_text(OUTPUT_STATUS_MD, """# Epoch 9E Continuation Host Exit-Status Correction

The frozen continuation completed all 34 scheduled rows and returned normally, but the host status command serialized numeric zero as the two characters `0n`. The PowerShell integer parser therefore recorded 255. WSL process exit 0, the exact 34/34 stdout summary, completed result timestamp, matching result hash, and exception-free stderr independently prove runner exit 0.

This append-only correction changes no scientific result and authorizes no rerun.
""")
    atomic_write_text(OUTPUT_PARSER_MD, f"""# Epoch 9E Continuation Adjudicator Parser Repair

The sealed adjudicator hash `{seal['continuation_adjudicator_sha256']}` is preserved. The repaired hash `{sha256(ADJUDICATOR)}` adds only a hash-bound parser exception for attempt 1's exact `0n` status artifact. No scientific field, threshold, endpoint, score, gate, trace, result, or sensitivity rule changed.

The host wrapper now writes the status with `printf "%s"` for any future infrastructure-only resume. The frozen schedule is already complete and is not rerun.
""")
    print(json.dumps({"corrected_exit": 0, "result_sha256": status["scientific_result_sha256"], "original_adjudicator": parser["original_sealed_adjudicator_sha256"], "repaired_adjudicator": parser["repaired_adjudicator_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
