#!/usr/bin/env python3
"""Bind causal runner, adjudicator, monitor, protocol, and frozen controller."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "epoch9d_causal_execution_seal.json"
PROTOCOL = REPORTS / "epoch9d_causal_panel_protocol.json"
CONTROLLER = REPORTS / "epoch9b_dynamic_nudge/controller_freeze.json"
CALIBRATION = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"
RUNNER = ROOT / "scripts/run_epoch9d_causal_panel.py"
ADJUDICATOR = ROOT / "scripts/adjudicate_epoch9d_causal_panel.py"
HOST_WRAPPER = ROOT / "scripts/run_epoch9d_causal_panel_host.ps1"
ORIGINAL_RUNNER = ROOT / "scripts/run_epoch9b_dynamic_nudge.py"


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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite causal execution seal")
    result_path = REPORTS / "epoch9d_causal_panel/result.json"
    if result_path.exists():
        raise RuntimeError("cannot seal after causal outcomes exist")
    for path in (PROTOCOL, CONTROLLER, CALIBRATION, RUNNER, ADJUDICATOR, HOST_WRAPPER, ORIGINAL_RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["validation_accessed"] or protocol["confirmation_accessed"]:
        raise RuntimeError("sealed-stage contamination")
    seal = {
        "schema_version": "epoch9d.causal_execution_seal.v1",
        "sealed_at": timestamp(),
        "branch": git("branch", "--show-current"),
        "source_checkpoint": git("rev-parse", "HEAD"),
        "outcomes_accessed_before_seal": False,
        "causal_protocol_path": relative(PROTOCOL),
        "causal_protocol_sha256": sha256(PROTOCOL),
        "runner_path": relative(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "adjudicator_path": relative(ADJUDICATOR),
        "adjudicator_sha256": sha256(ADJUDICATOR),
        "host_wrapper_path": relative(HOST_WRAPPER),
        "host_wrapper_sha256": sha256(HOST_WRAPPER),
        "original_epoch9b_runner_path": relative(ORIGINAL_RUNNER),
        "original_epoch9b_runner_sha256": sha256(ORIGINAL_RUNNER),
        "original_controller_freeze_path": relative(CONTROLLER),
        "original_controller_freeze_sha256": sha256(CONTROLLER),
        "calibration_path": relative(CALIBRATION),
        "calibration_sha256": sha256(CALIBRATION),
        "runtime": {
            "wsl_distribution": "Ubuntu-22.04",
            "python": "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python",
            "mujoco_gl": "egl",
            "simulator_environments_at_once": 1,
            "resident_vla_models": 0,
            "serial": True,
            "output": "reports/epoch9d_causal_panel/result.json",
            "resume": "missing row_key only",
        },
        "frozen_counts": {
            "base_states": protocol["base_state_count"],
            "primary_assignments": protocol["assignment_scene_count"],
            "candidate_probes": protocol["candidate_probe_count"],
            "sham_rows": protocol["sham_control"]["sham_row_count"],
        },
        "frozen_score": {
            "name": protocol["primary_score"]["name"],
            "threshold_m": protocol["primary_score"]["threshold_m"],
            "secondary_score": protocol["secondary_score"],
        },
        "frozen_decisions": {
            "go": protocol["success_decision"],
            "fail": protocol["failure_decision"],
            "near_miss": protocol["near_miss_replication"],
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT, seal)
    print(json.dumps({
        "output": relative(OUTPUT),
        "protocol": seal["causal_protocol_sha256"],
        "runner": seal["runner_sha256"],
        "adjudicator": seal["adjudicator_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
