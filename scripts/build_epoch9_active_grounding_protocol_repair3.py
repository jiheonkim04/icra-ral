#!/usr/bin/env python3
"""Freeze the corrected open-gripper paired probe after repair2 diagnosis."""

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
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_repair3.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    repair = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    repair.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.repair3.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_OPEN_GRIPPER_REPAIR_BEFORE_REPAIR3_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_repair2.json",
            "development_repair3_basis": {
                "failed_runs": [
                    "reports/epoch9_relational_probe_dataset/development/repair1_front_first_v1/result.json",
                    "reports/epoch9_relational_probe_dataset/development/repair2_full_v1/result.json",
                ],
                "diagnostic_evidence": (
                    "The repair2 back probe lifted the light bowl from z=0.8984 m to z=1.1182 m. In the official HDF5 "
                    "demonstration, gripper action -1 is held throughout approach and changes to +1 only at grasp closure."
                ),
                "root_cause": (
                    "The inherited controller used +1 during approach/return and 0 during contact under the mistaken "
                    "description 'neutral gripper'; the paired reset geometry allowed the closing gripper to catch a bowl."
                ),
                "repair": (
                    "Hold the official open command -1 during every approach, contact, hold, withdrawal, and return step. "
                    "Restore the original 2/3 push scale for both slots; keep fixed front-to-back order and every split, "
                    "assignment, model input, and 3 cm/5 cm/5 px gate unchanged."
                ),
                "label_blind_rationale": (
                    "The change corrects verified action semantics and removes accidental grasping; it does not use "
                    "response features or classification correctness."
                ),
            },
        }
    )
    repair["paired_probe_controller"] = {
        "controller_id": "paired_v14_open_gripper",
        "push_scale_by_slot": {"front": 2.0 / 3.0, "back": 2.0 / 3.0},
        "gripper_command": -1.0,
        "gripper_semantics": "official demonstration open command, held throughout non-grasping probe",
        "runner_path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": sha256(RUNNER),
    }
    atomic_write_json(OUTPUT, repair)
    print(json.dumps({"output": str(OUTPUT), "runner_sha256": sha256(RUNNER)}, indent=2))


if __name__ == "__main__":
    main()
