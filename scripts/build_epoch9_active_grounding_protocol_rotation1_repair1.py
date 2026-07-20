#!/usr/bin/env python3
"""Freeze the one safety repair for the front-reference rotation."""

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

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_rotation1.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_rotation1_repair1.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    protocol = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    protocol.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.rotation1.repair1.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_FRONT_REFERENCE_SHORT_PUSH_BEFORE_REPAIR_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_rotation1.json",
            "rotation_repair_basis": {
                "failed_run": (
                    "reports/epoch9_relational_probe_dataset/development/rotation1_front_reference_v1/result.json"
                ),
                "observation": (
                    "The first 32 physical pairs remained within gate; demo 37 at front factor 1x displaced 0.04290 m, "
                    "while its observed 2x/4x/8x counterparts displaced 0.02152-0.02212 m."
                ),
                "repair": (
                    "Shorten the sole front inward push from 2/3 to 4/9 of the original 18 mm delta (12 mm to 8 mm). "
                    "Keep the open gripper, contact waypoint, return, identities, assignments, model, and all gates unchanged."
                ),
                "repair_budget": (
                    "This is the only controller repair authorized for the rotated front-reference method. A repeated "
                    "execution-gate failure closes the rotation before model fitting."
                ),
            },
        }
    )
    protocol["paired_probe_controller"]["controller_id"] = "front_reference_v2_short_open_push"
    protocol["paired_probe_controller"]["push_scale_by_slot"]["front"] = 4.0 / 9.0
    protocol["paired_probe_controller"]["runner_sha256"] = sha256(RUNNER)
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"output": str(OUTPUT), "front_push_scale": 4.0 / 9.0}, indent=2))


if __name__ == "__main__":
    main()
