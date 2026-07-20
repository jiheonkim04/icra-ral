#!/usr/bin/env python3
"""Repair only the oracle grasp pose using observed expert first contact."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

SOURCE = ROOT / "reports/epoch9b_dynamic_nudge/controller_calibration.json"
DIAGNOSTIC = ROOT / "reports/epoch9_controller_development/expert_contact_diagnostic.json"
OUTPUT = ROOT / "reports/epoch9b_dynamic_nudge/controller_calibration_repair1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    result["schema_version"] = "epoch9b.dynamic_nudge_calibration.repair1.v1"
    result["frozen_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["source_calibration"] = {
        "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(SOURCE),
    }
    result["repair_scope"] = (
        "oracle-only grasp offset: use the development expert's first physical contact pose rather than "
        "the earlier gripper-close transition; RGB calibration and probe controller are unchanged"
    )
    result["preserved_failed_attempt"] = (
        "reports/epoch9b_dynamic_nudge/development/d1_guarded_lateral_nudge/result.json"
    )
    for slot in ("front", "back"):
        row = next(
            value for value in diagnostic["rows"] if value["slot"] == slot and int(value["demo_index"]) == 30
        )
        contact = row["first_contact"]
        initial = np.asarray(row["initial_target_eval_only"], dtype=np.float64)
        eef = np.asarray(contact["eef_pos"], dtype=np.float64)
        oracle = result["pose_adaptive_oracle_calibration"][slot]
        oracle["historical_close_transition_grasp_offset_xyz_m"] = oracle["grasp_eef_minus_object_xyz_m"]
        oracle["grasp_eef_minus_object_xyz_m"] = (eef - initial).tolist()
        oracle["grasp_eef_quat_xyzw"] = contact["eef_quat"]
        oracle["grasp_pose_source"] = "expert_contact_diagnostic first target-contact step"
        oracle["first_contact_step"] = int(contact["step"])
    atomic_write_json(OUTPUT, result)
    print(json.dumps({"status": "ORACLE_CALIBRATION_REPAIR1_FROZEN", "probe_controller_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
