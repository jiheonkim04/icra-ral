#!/usr/bin/env python3
"""Freeze the short back-slot push after repair1's 3 cm boundary failure."""

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

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_repair1.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_repair2.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    repair = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    repair.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.repair2.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_SHORT_BACK_PUSH_REPAIR_BEFORE_REPAIR2_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_repair1.json",
            "development_repair2_basis": {
                "failed_run": (
                    "reports/epoch9_relational_probe_dataset/development/repair1_front_first_v1/result.json"
                ),
                "observation": (
                    "The fixed-order repair removed catastrophic sequential interference, but demo 33 with front/back "
                    "mass factors 8x/1x ended with 0.03215 m back-candidate displacement, above the unchanged 0.03 m gate."
                ),
                "repair": (
                    "Shorten only the back-slot inward push from 2/3 to 4/9 of its original 18 mm calibration delta. "
                    "Keep the front probe, contact approach, hold, return path, splits, assignments, model inputs, and gates fixed."
                ),
                "label_blind_rationale": (
                    "The repair responds to a safety displacement metric. It does not use response features or prediction "
                    "correctness and is applied identically to every back-slot mass condition."
                ),
            },
        }
    )
    repair["paired_probe_controller"] = {
        "controller_id": "paired_v13_back_short_push",
        "front_push_scale": 2.0 / 3.0,
        "back_push_scale": 4.0 / 9.0,
        "push_scale_by_slot": {"front": 2.0 / 3.0, "back": 4.0 / 9.0},
        "runner_path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": sha256(RUNNER),
    }
    atomic_write_json(OUTPUT, repair)
    print(json.dumps({"output": str(OUTPUT), "runner_sha256": sha256(RUNNER)}, indent=2))


if __name__ == "__main__":
    main()
