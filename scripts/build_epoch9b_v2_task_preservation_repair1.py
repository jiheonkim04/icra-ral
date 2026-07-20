#!/usr/bin/env python3
"""Freeze a fresh v2 manifest for the centered-contact repair."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


BASE = ROOT / "reports/epoch9b_v2_task_preservation_protocol.json"
PARENT_PANEL = ROOT / "reports/epoch9b_dynamic_nudge/feasibility_panel_result.json"
OUTPUT = ROOT / "reports/epoch9b_v2_task_preservation_protocol_repair1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def fresh_manifest() -> list[dict[str, Any]]:
    rows = []
    for index in range(24):
        cell = index % 8
        heavy = "front" if cell < 4 else "back"
        light = "back" if heavy == "front" else "front"
        order = ["front", "back"] if cell % 4 < 2 else ["back", "front"]
        instruction = "heaviest" if cell % 2 == 0 else "lightest"
        rows.append(
            {
                "scene_id": f"epoch9b_repair1_scene_{index:03d}",
                "partition": "DEVELOPMENT_FEASIBILITY_REPAIR1",
                "source_state_demo_index": 33,
                "generator_seed": 911000 + index,
                "fresh_scene_construction": "replace both candidate free-joint XY positions with new frozen interior-lane coordinates before settling",
                "candidate_initial_xy_m": {
                    "front": [
                        0.078 + 0.0045 * ((7 * index + 3) % 16),
                        0.123 + 0.0035 * ((5 * index + 1) % 13),
                    ],
                    "back": [
                        -0.176 + 0.0048 * ((11 * index + 2) % 16),
                        0.024 + 0.0038 * ((3 * index + 4) % 15),
                    ],
                },
                "mass_factor": {heavy: 8.0, light: 1.0},
                "heavy_slot": heavy,
                "probe_order": order,
                "instruction_property": instruction,
                "completion_target_slot_eval_only": heavy if instruction == "heaviest" else light,
            }
        )
    return rows


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    protocol = json.loads(BASE.read_text(encoding="utf-8"))
    panel = json.loads(PARENT_PANEL.read_text(encoding="utf-8"))
    if panel["summary"]["minimal_feasibility_panel_go"]:
        raise RuntimeError("repair protocol is only valid after the parent panel NO-GO")
    manifest = fresh_manifest()
    protocol.update(
        {
            "schema_version": "epoch9b.v2_task_preservation.repair1.v1",
            "status": "FROZEN_BEFORE_REPAIR1_CONTROLLER_OUTCOME",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "parent_protocol_path": str(BASE.relative_to(ROOT)).replace("\\", "/"),
            "parent_protocol_sha256": sha256(BASE),
            "parent_panel_no_go": {
                "path": str(PARENT_PANEL.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PARENT_PANEL),
                "reason": "one front candidate exceeded the frozen y-lane ceiling by 0.004117 m; all other minimum feasibility thresholds passed",
                "parent_panel_summary": panel["summary"],
            },
            "repair_scope": "center the front contact after fixture-clear transit; preserve action bounds, impulse, response threshold, lanes, 0.05 m displacement rule, and all GO thresholds",
            "feasibility_manifest": manifest,
            "manifest_balance_audit": {
                "scene_count": len(manifest),
                "heavy_front": sum(row["heavy_slot"] == "front" for row in manifest),
                "heavy_back": sum(row["heavy_slot"] == "back" for row in manifest),
                "front_first": sum(row["probe_order"] == ["front", "back"] for row in manifest),
                "back_first": sum(row["probe_order"] == ["back", "front"] for row in manifest),
                "heaviest_instruction": sum(row["instruction_property"] == "heaviest" for row in manifest),
                "lightest_instruction": sum(row["instruction_property"] == "lightest" for row in manifest),
                "all_scene_ids_new": not bool(
                    {row["scene_id"] for row in manifest}
                    & {row["scene_id"] for row in panel["manifest"]}
                ),
                "sealed_identity_used": False,
            },
        }
    )
    atomic_write_json(OUTPUT, protocol)
    return protocol


def main() -> int:
    result = build()
    print(json.dumps({"status": result["status"], **result["manifest_balance_audit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
