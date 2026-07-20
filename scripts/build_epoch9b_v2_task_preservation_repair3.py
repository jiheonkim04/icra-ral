#!/usr/bin/env python3
"""Freeze the interior-margin task repair with the original mirrored probe."""

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


BASE = ROOT / "reports/epoch9b_v2_task_preservation_protocol_repair2.json"
PARENT_DEVELOPMENT = ROOT / "reports/epoch9b_dynamic_nudge/development/d19_inward_contact_balanced/result.json"
OUTPUT = ROOT / "reports/epoch9b_v2_task_preservation_protocol_repair3.json"


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
                "scene_id": f"epoch9b_repair3_scene_{index:03d}",
                "partition": "DEVELOPMENT_FEASIBILITY_REPAIR3",
                "source_state_demo_index": 35,
                "generator_seed": 913000 + index,
                "fresh_scene_construction": "replace both candidate free-joint XY positions with new frozen coordinates at least 0.010 m inside every lateral lane boundary before settling",
                "candidate_initial_xy_m": {
                    "front": [
                        0.080 + 0.0045 * ((3 * index + 1) % 17),
                        0.126 + 0.0031 * ((5 * index + 2) % 13),
                    ],
                    "back": [
                        -0.174 + 0.0045 * ((7 * index + 3) % 17),
                        0.027 + 0.0035 * ((11 * index + 1) % 14),
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
    development = json.loads(PARENT_DEVELOPMENT.read_text(encoding="utf-8"))
    if development["summary"]["oracle_completion_success_scenes"] >= 6:
        raise RuntimeError("repair3 requires the documented repair2 oracle-headroom failure")
    manifest = fresh_manifest()
    prior_ids = {row["scene_id"] for row in protocol["feasibility_manifest"]}
    protocol.update(
        {
            "schema_version": "epoch9b.v2_task_preservation.repair3.v1",
            "status": "FROZEN_BEFORE_REPAIR3_CONTROLLER_OUTCOME",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "parent_protocol_path": str(BASE.relative_to(ROOT)).replace("\\", "/"),
            "parent_protocol_sha256": sha256(BASE),
            "parent_repair2_development_no_go": {
                "path": str(PARENT_DEVELOPMENT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PARENT_DEVELOPMENT),
                "reason": "inward contact passed probe validity and ranking but achieved only five of eight oracle completions",
                "summary": development["summary"],
            },
            "repair_scope": "restore the original mirrored fixture-clear probe and sample procedural initial centers at least 0.010 m inside the unchanged frozen lanes; preserve impulse, response threshold, 0.05 m displacement rule, and every GO threshold",
            "procedural_initial_center_margin_m": 0.010,
            "safe_center_lanes_unchanged": True,
            "feasibility_manifest": manifest,
            "manifest_balance_audit": {
                "scene_count": len(manifest),
                "heavy_front": sum(row["heavy_slot"] == "front" for row in manifest),
                "heavy_back": sum(row["heavy_slot"] == "back" for row in manifest),
                "front_first": sum(row["probe_order"] == ["front", "back"] for row in manifest),
                "back_first": sum(row["probe_order"] == ["back", "front"] for row in manifest),
                "heaviest_instruction": sum(row["instruction_property"] == "heaviest" for row in manifest),
                "lightest_instruction": sum(row["instruction_property"] == "lightest" for row in manifest),
                "all_scene_ids_new": not bool(prior_ids & {row["scene_id"] for row in manifest}),
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
