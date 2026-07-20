#!/usr/bin/env python3
"""Freeze the fixed-order repair after the v1 sequential-interference failure."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_repair1.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    repair = copy.deepcopy(source)
    repair.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.repair1.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_FIXED_ORDER_REPAIR_BEFORE_REPAIR_PAIRED_PROBE_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol.json",
            "development_repair_basis": {
                "failed_run": "reports/epoch9_relational_probe_dataset/development/paired_v1/result.json",
                "observation": (
                    "Three demo-30 back-first episodes completed legal contact but the later front probe displaced the back bowl "
                    "0.241-0.251 m; every observed front-first episode remained within 0.03 m."
                ),
                "diagnosis": (
                    "The unconstrained-orientation return left a configuration-dependent sequential interaction. "
                    "The failure was order-specific and not monotonic in the latent mass label."
                ),
                "repair": (
                    "Fix the controller and deployment order to front then back. No waypoint, action, visual threshold, "
                    "mass assignment, split identity, model input, or mechanism threshold changes."
                ),
                "claim_limit": "No probe-order invariance claim is authorized.",
            },
        }
    )
    repair["factorial_design"]["probe_orders"] = [["front", "back"]]
    repair["factorial_design"]["balance"] = (
        "each reset identity contains both latent-property directions and 8x/1x and 4x/2x contrasts; "
        "probe order is the fixed deployable front-to-back controller order"
    )
    repair["validation_gates"].pop("probe_order_consistency_fraction_min", None)
    repair["validation_gates"]["fixed_front_then_back_order_required"] = True
    for partition, rows in repair["episode_specs"].items():
        selected = [row for row in rows if row["probe_order"] == ["front", "back"]]
        repair["episode_specs"][partition] = selected
        repair["partitions"][partition]["physical_pair_episodes"] = len(selected)
    atomic_write_json(OUTPUT, repair)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "counts": {partition: len(rows) for partition, rows in repair["episode_specs"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
