#!/usr/bin/env python3
"""Adjudicate and close the failed paired fixed-slot probe route."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

OUTPUT = ROOT / "reports/epoch9_paired_probe_route_adjudication.json"
RUNS = {
    "v1_balanced_order": ROOT / "reports/epoch9_relational_probe_dataset/development/paired_v1/result.json",
    "repair1_fixed_order": ROOT
    / "reports/epoch9_relational_probe_dataset/development/repair1_front_first_v1/result.json",
    "repair2_short_back_push": ROOT
    / "reports/epoch9_relational_probe_dataset/development/repair2_full_v1/result.json",
    "repair3_open_gripper": ROOT
    / "reports/epoch9_relational_probe_dataset/development/repair3_demo31_33_diagnostic/result.json",
    "repair4_open_back_contact": ROOT
    / "reports/epoch9_relational_probe_dataset/development/repair4_demo31_33_diagnostic/result.json",
}


def main() -> None:
    rows = []
    for attempt, path in RUNS.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "attempt": attempt,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "status": result.get("status"),
                "summary": result["summary"],
            }
        )
    report = {
        "schema_version": "epoch9.paired_fixed_slot_probe.adjudication.v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_class": "DEVELOPMENT_ROUTE_ADJUDICATION",
        "attempts": rows,
        "failure_chain": [
            "Balanced order exposed catastrophic back-bowl sweep after a back-first probe.",
            "Fixed order removed that interaction but exceeded the unchanged 3 cm displacement gate on a new reset.",
            "A shorter back push exposed accidental grasp/lift because the inherited gripper sign was wrong.",
            "The official open-gripper correction restored safety but missed every back contact in the targeted diagnostic.",
            "The sole open-tool back contact recalibration reached only 62.5% aggregate contact and failed the 100% gate.",
        ],
        "decision": "PAIRED_FIXED_SLOT_PROBE_ROUTE_CLOSED_ROTATE_TO_CONTROLLED_REFERENCE",
        "closed_scope": (
            "This closes the exact sequential paired fixed-slot Cartesian controller and reset-by-reset waypoint repair. "
            "It does not close active physical grounding, visual contact feedback, learned exploration, or the separately "
            "preregistered complementary front-reference task."
        ),
        "confirmation_accessed": False,
        "validation_accessed": False,
    }
    atomic_write_json(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
