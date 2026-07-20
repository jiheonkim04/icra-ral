#!/usr/bin/env python3
"""Freeze the development-only slot response calibration for Epoch 9B."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


SOURCE = (
    ROOT
    / "reports/epoch9b_dynamic_nudge/development/"
    / "d14_balanced_mirrored_event_ballistic/result.json"
)
OUTPUT = ROOT / "reports/epoch9b_dynamic_nudge/back_response_threshold_calibration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [row for row in source["rows"] if row.get("completed")]
    values: dict[str, list[float]] = {"heavy": [], "light": []}
    audit_rows = []
    for row in rows:
        probe = next(value for value in row["probes"] if value["slot"] == "back")
        validity = {
            "contact_or_excitation": bool(probe["intended_target_contact_or_excitation_eval_only"]),
            "finite_bounded_actions": bool(probe["finite_bounded_actions"]),
            "lane_reachability": bool(probe["lane_and_reachability_continuous_pass"]),
            "distance_limit": bool(probe["maximum_intended_displacement_limit_pass"]),
            "tracker_quality": bool(probe["response_tracker_quality_min"] >= 0.50),
            "no_candidate_pair_collision": not bool(probe["candidate_pair_collision_eval_only"]),
            "no_distractor_collision": not bool(probe["candidate_distractor_collision_eval_only"]),
        }
        if not all(validity.values()):
            raise RuntimeError(f"invalid back-slot calibration probe in {row['scene_id']}: {validity}")
        label = "heavy" if row["scene"]["heavy_slot"] == "back" else "light"
        response = float(probe["response_estimated_displacement_m"])
        values[label].append(response)
        audit_rows.append(
            {
                "scene_id": row["scene_id"],
                "probe_order": row["scene"]["probe_order"],
                "label_development_only": label,
                "back_response_m": response,
                "validity": validity,
            }
        )
    if len(values["heavy"]) != 4 or len(values["light"]) != 4:
        raise RuntimeError("expected a balanced four-heavy/four-light back-slot calibration")
    heavy_max = max(values["heavy"])
    light_min = min(values["light"])
    if not heavy_max < light_min:
        raise RuntimeError("back-slot development responses are not separable")
    threshold = 0.5 * (heavy_max + light_min)
    result = {
        "schema_version": "epoch9b.dynamic_nudge.back_response_threshold.v1",
        "status": "FROZEN_BEFORE_SAFE_PATH_REPEATABILITY_AND_24_SCENE_PANEL",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_class": "DEVELOPMENT_SCORE_CALIBRATION",
        "source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "source_back_probes_only": True,
        "source_back_probes_all_v2_valid": True,
        "sealed_validation_or_confirmation_accessed": False,
        "heavy_back_response_m": values["heavy"],
        "light_back_response_m": values["light"],
        "heavy_back_summary_m": {
            "mean": float(np.mean(values["heavy"])),
            "max": float(heavy_max),
        },
        "light_back_summary_m": {
            "mean": float(np.mean(values["light"])),
            "min": float(light_min),
        },
        "back_heavy_threshold_m": float(threshold),
        "development_separation_margin_m": float(light_min - heavy_max),
        "ranking_rule": {
            "back_response_at_or_below_threshold": "predict back candidate heavy",
            "back_response_above_threshold": "predict front candidate heavy",
            "front_response_role": "reported but not used because front development response was less separable",
        },
        "audit_rows": audit_rows,
    }
    atomic_write_json(OUTPUT, result)
    return result


def main() -> int:
    result = build()
    print(json.dumps({
        "status": result["status"],
        "threshold_m": result["back_heavy_threshold_m"],
        "margin_m": result["development_separation_margin_m"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
