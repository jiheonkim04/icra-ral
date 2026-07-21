#!/usr/bin/env python3
"""Seal the sole Epoch 9E joint panel before any joint outcome exists."""

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
OUTPUT = REPORTS / "epoch9e_joint_execution_seal.json"
PROTOCOL = REPORTS / "epoch9e_joint_certification_protocol.json"
MECHANICS_SEAL = REPORTS / "epoch9e_mechanics_execution_seal.json"
MECHANICS_ADJUDICATION = REPORTS / "epoch9e_mechanics_smoke_adjudication.json"
RUNNER = ROOT / "scripts/run_epoch9e_joint_certification.py"
ADJUDICATOR = ROOT / "scripts/adjudicate_epoch9e_joint_certification.py"
HOST = ROOT / "scripts/run_epoch9e_joint_certification_host.ps1"
CONTROLLER = ROOT / "scripts/epoch9e_nondrag_controller.py"
ORIGINAL_RUNNER = ROOT / "scripts/run_epoch9b_dynamic_nudge.py"


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
    payload = subprocess.check_output(
        ["git", "show", f"HEAD:{relative(path)}"], cwd=ROOT
    )
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite joint execution seal")
    forbidden_outcomes = [
        REPORTS / "epoch9e_joint_certification/result.json",
        REPORTS / "epoch9e_joint_certification_adjudication.json",
        REPORTS / "epoch9e_joint_certification_adjudication.md",
    ]
    if any(path.exists() for path in forbidden_outcomes):
        raise RuntimeError("cannot seal after any joint outcome or adjudication exists")
    paths = (PROTOCOL, MECHANICS_SEAL, MECHANICS_ADJUDICATION, RUNNER, ADJUDICATOR, HOST, CONTROLLER, ORIGINAL_RUNNER)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != committed_sha256(path):
            raise RuntimeError(f"sealed input is not byte-identical to HEAD: {relative(path)}")
    protocol = load(PROTOCOL)
    mechanics_seal = load(MECHANICS_SEAL)
    mechanics = load(MECHANICS_ADJUDICATION)
    if mechanics.get("decision") != "MECHANICS_SMOKE_PASS_FREEZE_CONTROLLER" or mechanics.get("mechanics_smoke_pass") is not True:
        raise RuntimeError("mechanics smoke did not freeze the controller")
    if sha256(CONTROLLER) != mechanics_seal["controller_sha256"]:
        raise RuntimeError("controller changed after the passing mechanics smoke")
    if protocol["one_shot"] != {
        "panels": 1,
        "near_miss_rerun": False,
        "row_replacement": False,
        "endpoint_repair_after_outcome": False,
    }:
        raise RuntimeError("joint protocol is not the frozen one-shot contract")
    seal = {
        "schema_version": "epoch9e.joint_execution_seal.v1",
        "sealed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "joint_outcomes_accessed_before_seal": False,
        "controller_frozen_after_passing_mechanics_smoke": True,
        "joint_protocol_path": relative(PROTOCOL),
        "joint_protocol_sha256": sha256(PROTOCOL),
        "mechanics_execution_seal_path": relative(MECHANICS_SEAL),
        "mechanics_execution_seal_sha256": sha256(MECHANICS_SEAL),
        "mechanics_adjudication_path": relative(MECHANICS_ADJUDICATION),
        "mechanics_adjudication_sha256": sha256(MECHANICS_ADJUDICATION),
        "runner_path": relative(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "adjudicator_path": relative(ADJUDICATOR),
        "adjudicator_sha256": sha256(ADJUDICATOR),
        "host_wrapper_path": relative(HOST),
        "host_wrapper_sha256": sha256(HOST),
        "controller_path": relative(CONTROLLER),
        "controller_sha256": sha256(CONTROLLER),
        "original_runner_path": relative(ORIGINAL_RUNNER),
        "original_runner_sha256": sha256(ORIGINAL_RUNNER),
        "one_shot": protocol["one_shot"],
        "joint_go": protocol["joint_go"],
        "success_decision": protocol["success_decision"],
        "failure_decision": protocol["failure_decision"],
        "runtime": {
            "serial": True,
            "environments_at_once": 1,
            "models_at_once": 0,
            "host_ram_ceiling_percent": 82.0,
            "wsl_swap_used_peak_bytes": 0,
            "resume_authorized": False,
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT, seal)
    print(json.dumps({"output": relative(OUTPUT), "source_checkpoint": seal["source_checkpoint"], "runner": seal["runner_sha256"], "controller": seal["controller_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
