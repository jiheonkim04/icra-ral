#!/usr/bin/env python3
"""Freeze the controlled front-reference active comparison rotation."""

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

SOURCE = ROOT / "reports/epoch9_active_grounding_protocol_repair3.json"
OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol_rotation1.json"
RUNNER = ROOT / "scripts/run_epoch9_relational_probe_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    protocol = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    protocol.update(
        {
            "schema_version": "epoch9.active_grounding.protocol.rotation1.v1",
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FROZEN_FRONT_REFERENCE_ROTATION_BEFORE_OUTCOMES",
            "supersedes_for_future_collection": "reports/epoch9_active_grounding_protocol_repair4.json",
            "method_rotation": {
                "closed_route": "paired fixed-slot sequential probing",
                "closure_evidence": [
                    "reports/epoch9_relational_probe_dataset/development/paired_v1/result.json",
                    "reports/epoch9_relational_probe_dataset/development/repair1_front_first_v1/result.json",
                    "reports/epoch9_relational_probe_dataset/development/repair2_full_v1/result.json",
                    "reports/epoch9_relational_probe_dataset/development/repair3_demo31_33_diagnostic/result.json",
                    "reports/epoch9_relational_probe_dataset/development/repair4_demo31_33_diagnostic/result.json",
                ],
                "new_controlled_residual": (
                    "Probe only the front candidate with the verified open-gripper controller. The task generator guarantees "
                    "one member of the front/back pair is heavier, so the back belief is the complement of the inferred "
                    "front belief. This is an active reference-comparison task, not general multi-object system identification."
                ),
                "scientific_question": (
                    "Can a temporal legal response representation ground relative heaviest/lightest selection under a "
                    "controlled complementary-pair intervention, beyond RGB-only and non-temporal controls?"
                ),
            },
        }
    )
    protocol["task"]["primary_unit"] = (
        "one physical complementary front/back mass assignment with one active front reference probe; inverse language "
        "queries are not counted as independent trials"
    )
    protocol["task"]["controlled_pair_constraint"] = (
        "exactly one candidate is heavier for every declared assignment; deployment is restricted to this generator"
    )
    protocol["factorial_design"]["probe_orders"] = [["front"]]
    protocol["factorial_design"]["balance"] = (
        "each reset identity contains both latent-property directions and 8x/1x and 4x/2x contrasts; only the front "
        "reference candidate is actively probed"
    )
    for partition, rows in protocol["episode_specs"].items():
        for row in rows:
            row["probe_order"] = ["front"]
    protocol["paired_probe_controller"] = {
        "controller_id": "front_reference_v1_open_gripper",
        "push_scale_by_slot": {"front": 2.0 / 3.0, "back": 2.0 / 3.0},
        "gripper_command": -1.0,
        "contact_override_by_slot": {"front": [0.050, 0.169, 0.926]},
        "runner_path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": sha256(RUNNER),
    }
    protocol["validation_gates"].pop("fixed_front_then_back_order_required", None)
    protocol["validation_gates"]["front_reference_probe_only_required"] = True
    protocol["planned_controls"] = [
        "balanced no-probe prior",
        "initial RGB-only control",
        "single mean controller-error feature",
        "Epoch 8 shared agentview aggregate features",
        "endpoint-only response model",
        "temporally shuffled response model",
        "probe classifier without complementary relational selection",
        "offline oracle mass-label headroom",
    ]
    protocol["claim_boundary"] = (
        "A pass supports active relative-mass grounding only for this complementary two-candidate LIBERO generator with "
        "one safe reference probe. It does not support arbitrary candidate sets, two-probe order invariance, general "
        "material understanding, real-robot transfer, or unrestricted VLA physical reasoning."
    )
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"output": str(OUTPUT), "episodes": {k: len(v) for k, v in protocol["episode_specs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
