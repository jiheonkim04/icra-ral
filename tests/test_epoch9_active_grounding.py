from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from tca_map.epoch9_active_grounding import (
    MASS_ASSIGNMENTS,
    PARTITION_DEMO_INDICES,
    PROBE_ORDERS,
    build_episode_specs,
    validate_partition_disjointness,
)


def test_epoch9_partitions_are_identity_disjoint() -> None:
    validate_partition_disjointness()
    values = list(PARTITION_DEMO_INDICES.values())
    assert not (set(values[0]) & set(values[1]))
    assert not (set(values[0]) & set(values[2]))
    assert not (set(values[1]) & set(values[2]))


def test_epoch9_factorial_grid_is_balanced_per_identity() -> None:
    rows = build_episode_specs("development")
    assert len(rows) == len(PARTITION_DEMO_INDICES["development"]) * len(MASS_ASSIGNMENTS) * len(PROBE_ORDERS)
    for demo_index in PARTITION_DEMO_INDICES["development"]:
        subset = [row for row in rows if row["demo_index"] == demo_index]
        assert Counter(tuple(row["probe_order"]) for row in subset) == {
            order: len(MASS_ASSIGNMENTS) for order in PROBE_ORDERS
        }
        assert Counter(row["heavier_slot_training_label"] for row in subset) == {"front": 4, "back": 4}


def test_epoch9_inverse_query_labels_are_consistent() -> None:
    for row in build_episode_specs("validation"):
        assert row["heavier_slot_training_label"] != row["lighter_slot_training_label"]
        expected = "front" if row["front_mass_factor"] > row["back_mass_factor"] else "back"
        assert row["heavier_slot_training_label"] == expected


def test_epoch9_repair_protocol_fixes_deployable_order_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_repair1.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for rows in protocol["episode_specs"].values():
        assert rows
        assert all(row["probe_order"] == ["front", "back"] for row in rows)


def test_epoch9_relational_head_is_antisymmetric() -> None:
    import torch

    from scripts.run_epoch9_temporal_relational_model import build_model

    torch.manual_seed(9)
    paired = torch.randn(2, 2, 12, 128)
    for variant in ("relational", "independent"):
        model = build_model(128, variant).eval()
        with torch.no_grad():
            forward = model(paired)
            swapped = model(paired[:, [1, 0]])
        assert torch.allclose(forward, -swapped, atol=1e-6)


def test_epoch9_reference_belief_uses_front_temporal_sequence() -> None:
    import torch

    from scripts.run_epoch9_temporal_relational_model import build_model

    torch.manual_seed(90)
    model = build_model(128, "reference").eval()
    paired = torch.zeros(2, 2, 12, 128)
    paired[1, 0] = 1.0
    with torch.no_grad():
        logits = model(paired)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert logits[0] != logits[1]


def test_epoch9_smoke_trace_has_only_legal_fields_if_materialized() -> None:
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    path = root / (
        "reports/epoch9_relational_probe_dataset/development/mechanical_smoke_v1/traces/"
        "development_demo30_front8_back1_front-first_front.npz"
    )
    if not path.exists():
        return
    with np.load(path) as trace:
        assert set(trace.files) == {
            "phase",
            "action",
            "eef_pos",
            "eef_quat",
            "controller_goal_pos",
            "controller_error",
            "rgb_diff_32",
        }


def test_epoch9_repair2_shortens_only_back_push_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_repair2.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    scales = protocol["paired_probe_controller"]["push_scale_by_slot"]
    assert scales["front"] == 2.0 / 3.0
    assert scales["back"] == 4.0 / 9.0
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_repair3_uses_official_open_gripper_command_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_repair3.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    controller = protocol["paired_probe_controller"]
    assert controller["gripper_command"] == -1.0
    assert controller["push_scale_by_slot"] == {"front": 2.0 / 3.0, "back": 2.0 / 3.0}
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_repair4_preserves_gates_and_has_one_back_calibration_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_repair4.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    controller = protocol["paired_probe_controller"]
    assert controller["gripper_command"] == -1.0
    assert controller["contact_override_by_slot"]["back"] == [-0.1593, 0.017, 0.9218]
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_rotation1_is_front_reference_only_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_rotation1.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["paired_probe_controller"]["gripper_command"] == -1.0
    assert protocol["validation_gates"]["front_reference_probe_only_required"] is True
    for rows in protocol["episode_specs"].values():
        assert all(row["probe_order"] == ["front"] for row in rows)


def test_epoch9_rotation1_repair1_only_shortens_front_push_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_rotation1_repair1.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    controller = protocol["paired_probe_controller"]
    assert controller["gripper_command"] == -1.0
    assert controller["push_scale_by_slot"]["front"] == 4.0 / 9.0
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_rotation2_changes_return_not_safety_gate_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_rotation2.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    controller = protocol["paired_probe_controller"]
    assert controller["return_variant"] == "clearance_first"
    assert controller["gripper_command"] == -1.0
    assert controller["push_scale_by_slot"]["front"] == 4.0 / 9.0
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_rotation3_is_zero_travel_and_keeps_gate_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_active_grounding_protocol_rotation3.json"
    if not path.exists():
        return
    protocol = json.loads(path.read_text(encoding="utf-8"))
    controller = protocol["paired_probe_controller"]
    assert controller["push_scale_by_slot"]["front"] == 0.0
    assert controller["return_variant"] == "clearance_first"
    assert protocol["validation_gates"]["final_each_candidate_displacement_m_max"] == 0.03


def test_epoch9_terminal_state_keeps_sealed_partitions_if_materialized() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/epoch9_campaign_state.json"
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["epoch8_status"] == "EPOCH8_EXACT_METHODS_FINISHED_CONTINUATION_REQUIRED"
    assert state["paper_status"] == "PAPER_NOT_AUTHORIZED"
    assert state["program_status"] == "CONTINUATION_REQUIRED"
    assert state["validation_accessed"] is False
    assert state["confirmation_accessed"] is False
