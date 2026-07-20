#!/usr/bin/env python3
"""Materialize the preregistered Epoch 9 split and decision protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json
from tca_map.epoch9_active_grounding import (
    LEGAL_TRACE_FIELDS,
    MASS_ASSIGNMENTS,
    PARTITION_DEMO_INDICES,
    PROBE_ORDERS,
    build_episode_specs,
)

OUTPUT = ROOT / "reports/epoch9_active_grounding_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    controller = ROOT / "scripts/run_epoch9_probe_controller_development.py"
    source_hdf5 = Path(
        "/mnt/c/assets/data/libero/libero_90/"
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_demo.hdf5"
    )
    source_bddl = Path(
        "/mnt/c/assets/repos/LIBERO/libero/libero/bddl_files/libero_90/"
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.bddl"
    )
    specs = {partition: build_episode_specs(partition) for partition in PARTITION_DEMO_INDICES}
    protocol = {
        "schema_version": "epoch9.active_grounding.protocol.v1",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "FROZEN_BEFORE_SAME_SCENE_PAIRED_PROBE_OUTCOMES",
        "source_checkpoint": "f8314938e440aaefb6399d80594c78431043aa7f",
        "task": {
            "scene": "LIBERO Kitchen Scene 2 with visually identical front, middle, and back black bowls",
            "candidate_slots": ["front", "back"],
            "middle_slot_role": "excluded candidate and untouched distractor because Epoch 8 exposed middle-slot competence confounding",
            "queries": ["select the heaviest candidate", "select the lightest candidate"],
            "primary_unit": "one physical front/back mass assignment and probe order; inverse language queries are not counted as independent trials",
            "property_labels": "simulator mass interventions are offline training/evaluation labels only and are forbidden online policy inputs",
        },
        "partitions": {
            name: {
                "demo_indices": list(indices),
                "physical_pair_episodes": len(specs[name]),
                "outcome_access": (
                    "development allowed"
                    if name == "development"
                    else "only after full system freeze"
                    if name == "validation"
                    else "sealed until validation gate passes"
                ),
            }
            for name, indices in PARTITION_DEMO_INDICES.items()
        },
        "factorial_design": {
            "mass_assignments_front_back": [list(values) for values in MASS_ASSIGNMENTS],
            "probe_orders": [list(values) for values in PROBE_ORDERS],
            "balance": "each reset identity contains both latent-property directions, 8x/1x and 4x/2x contrasts, and both probe orders",
        },
        "frozen_probe_controller": {
            "attempt_id": "v12_visual_tolerance_calibration",
            "source_sha256": sha256(controller),
            "development_evidence": [
                "reports/epoch9_controller_development/v11_front_contact_depth/result.json",
                "reports/epoch9_controller_development/v11_front_contact_depth_scale1/result.json",
                "reports/epoch9_controller_development/v12_visual_tolerance_calibration/result.json",
            ],
            "online_inputs": [
                "fixed candidate-slot calibration",
                "agentview RGB",
                "end-effector position and quaternion",
                "executed action and controller-command history",
                "elapsed controller phase",
            ],
            "legal_recorded_trace_fields": list(LEGAL_TRACE_FIELDS),
            "forbidden_action_inputs": [
                "mass or inertia",
                "simulator object pose or identity lookup",
                "segmentation",
                "force/torque",
                "reward, success, done, future observations, or expert actions",
            ],
        },
        "source_artifacts": {
            "hdf5": str(source_hdf5),
            "hdf5_sha256": sha256(source_hdf5),
            "bddl": str(source_bddl),
            "bddl_sha256": sha256(source_bddl),
        },
        "model_freeze_rule": (
            "architecture, legal feature transform, temporal length, optimizer, seed ensemble, checkpoint rule, and all gates "
            "must be fixed using development identities only before validation simulation starts"
        ),
        "planned_controls": [
            "balanced no-probe prior",
            "initial RGB-only control",
            "single mean controller-error feature",
            "Epoch 8 fixed seven-feature LDA",
            "endpoint-only response model",
            "temporally shuffled response model",
            "independent per-probe scoring without relational context",
            "offline oracle mass-label headroom",
        ],
        "validation_gates": {
            "execution_exception_count_max": 0,
            "contact_fraction": 1.0,
            "bounded_action_fraction": 1.0,
            "final_each_candidate_displacement_m_max": 0.03,
            "final_eef_displacement_m_max": 0.05,
            "visual_return_residual_pixels_max": 5.0,
            "physical_pair_accuracy_min": 0.80,
            "exact_binomial_one_sided_p_vs_half_max": 0.05,
            "mass_contrast_accuracy_each_min": 0.70,
            "probe_order_consistency_fraction_min": 0.90,
            "must_beat_no_probe_prior": True,
            "must_beat_initial_rgb_control": True,
            "must_not_underperform_best_non_temporal_control": True,
        },
        "confirmation_rule": (
            "confirmation may run once, without repair, only if every validation execution and mechanism gate passes; "
            "the same thresholds apply and validation plus confirmation are reported separately"
        ),
        "claim_boundary": (
            "A pass supports active relative-mass grounding for this controlled two-candidate LIBERO scene and legal sensor stack; "
            "it does not establish general material understanding, real-robot transfer, or unrestricted VLA physical reasoning"
        ),
        "episode_specs": specs,
    }
    atomic_write_json(OUTPUT, protocol)
    print(json.dumps({"output": str(OUTPUT), "counts": {k: len(v) for k, v in specs.items()}}, indent=2))


if __name__ == "__main__":
    main()
