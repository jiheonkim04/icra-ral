#!/usr/bin/env python3
"""Freeze the single open-tool back-contact calibration for Epoch 9."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_repair3.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_repair4.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    repair = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    repair.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.repair4.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_OPEN_TOOL_BACK_CONTACT_CALIBRATION_BEFORE_REPAIR4_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_repair3.json",
            "development_repair4_basis": {
                "diagnostic_run": (
                    "reports/epoch9_relational_probe_dataset/development/repair3_demo31_33_diagnostic/result.json"
                ),
                "observation": (
                    "With the official -1 open command, all eight front probes contacted safely while all eight back probes "
                    "missed and produced at most micrometer-scale back-bowl motion."
                ),
                "calibration": (
                    "Mirror the successful front tool-to-center clearance in scene coordinates: set the back EEF contact y "
                    "to +0.017 m instead of -0.0012 m, retaining x=-0.1593 m, z=0.9218 m, and the 12 mm inward push."
                ),
                "repair_budget": (
                    "This is the sole post-semantics Cartesian contact recalibration. A failure closes this paired fixed-slot "
                    "probe route; no reset-specific waypoint tuning is authorized."
                ),
            },
        }
    )
    repair["paired_probe_controller"] = {
        "controller_id": "paired_v15_open_gripper_back_contact_calibrated",
        "push_scale_by_slot": {"front": 2.0 / 3.0, "back": 2.0 / 3.0},
        "gripper_command": -1.0,
        "contact_override_by_slot": {
            "front": [0.050, 0.169, 0.926],
            "back": [-0.1593, 0.017, 0.9218],
        },
        "runner_path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": sha256(RUNNER),
    }
    atomic_write_json(OUTPUT, repair)
    print(json.dumps({"output": str(OUTPUT), "runner_sha256": sha256(RUNNER)}, indent=2))


if __name__ == "__main__":
    main()
