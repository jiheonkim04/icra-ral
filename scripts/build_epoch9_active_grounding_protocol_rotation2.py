#!/usr/bin/env python3
"""Freeze the clearance-first return mechanism rotation."""

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

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_rotation1_repair1.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_rotation2.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    protocol = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    protocol.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.rotation2.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_CLEARANCE_FIRST_RETURN_BEFORE_OUTCOMES",
            "supersedes_for_future_collection": (
                "reports/epoch9_active_grounding_protocol_rotation1_repair1.json"
            ),
            "method_rotation2": {
                "closed_method": "legacy low-clearance front-reference return",
                "closure_evidence": [
                    "reports/epoch9_relational_probe_dataset/development/rotation1_front_reference_v1/result.json",
                    "reports/epoch9_relational_probe_dataset/development/rotation1_repair1_demo37_diagnostic/result.json",
                ],
                "diagnosis": (
                    "The offending bowl moved +0.0409 m in x despite a y-axis probe; shortening the probe by 4 mm did not "
                    "remove the sweep. The low-clearance lift/traverse is therefore the suspected disturbance source."
                ),
                "new_mechanism": (
                    "After reversing the probe, retreat another 40 mm away from the candidate at contact height, then lift "
                    "to z=1.02 m before any central traverse. This is a clearance-first return, not another contact waypoint tune."
                ),
                "budget": (
                    "One targeted development diagnostic and, only on pass, one complete development grid. A repeat safety "
                    "failure closes active fixed-slot probing before temporal-model fitting."
                ),
            },
        }
    )
    controller = protocol["paired_probe_controller"]
    controller["controller_id"] = "front_reference_v3_clearance_first_return"
    controller["return_variant"] = "clearance_first"
    controller["runner_sha256"] = sha256(RUNNER)
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"output": str(OUTPUT), "return_variant": "clearance_first"}, indent=2))


if __name__ == "__main__":
    main()
