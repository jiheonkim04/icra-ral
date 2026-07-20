#!/usr/bin/env python3
"""Freeze the final zero-travel contact/hold impedance probe rotation."""

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

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_rotation2.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_rotation3.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    protocol = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    protocol.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.rotation3.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_ZERO_TRAVEL_CONTACT_HOLD_BEFORE_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_rotation2.json",
            "method_rotation3": {
                "closed_method": "finite-travel open-gripper lateral tap",
                "closure_evidence": (
                    "reports/epoch9_relational_probe_dataset/development/rotation2_demo37_diagnostic/result.json"
                ),
                "new_mechanism": (
                    "Use the same legal contact-boundary approach but command zero inward travel, hold for eight feedback "
                    "steps to expose contact impedance, then use the clearance-first return."
                ),
                "terminal_rule": (
                    "This is the final fixed-slot contact mechanism. Any contact or 3 cm disturbance failure closes active "
                    "contact probing for the local scene before model fitting and leaves validation/confirmation sealed."
                ),
            },
        }
    )
    controller = protocol["paired_probe_controller"]
    controller["controller_id"] = "front_reference_v4_zero_travel_contact_hold"
    controller["push_scale_by_slot"]["front"] = 0.0
    controller["runner_sha256"] = sha256(RUNNER)
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"output": str(OUTPUT), "front_push_scale": 0.0}, indent=2))


if __name__ == "__main__":
    main()
