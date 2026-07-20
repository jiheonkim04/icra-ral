from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = REPO_ROOT / "scripts/build_epoch9b_v2_task_preservation_protocol.py"
    spec = importlib.util.spec_from_file_location("epoch9b_protocol_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_balanced_fresh_and_sealed_safe() -> None:
    builder = _builder()
    rows = builder.feasibility_manifest()
    audit = builder.balance_audit(rows)
    assert audit["scene_count"] == 24
    assert audit["unique_scene_count"] == 24
    assert audit["heavy_slot_counts"] == {"back": 12, "front": 12}
    assert audit["probe_order_counts"] == {"back/front": 12, "front/back": 12}
    assert audit["instruction_property_counts"] == {"heaviest": 12, "lightest": 12}
    assert audit["all_eight_factor_cells_equal"]
    assert not audit["sealed_integer_demo_indices_present"]


def test_v2_distance_rule_is_subordinate_to_task_preservation() -> None:
    builder = _builder()
    value = builder.protocol()
    assert value["v2_absolute_displacement_rule"]["limit_m"] == 0.05
    assert value["v2_absolute_displacement_rule"]["legacy_v1_reference_m_reported_only"] == 0.03
    assert not value["v2_absolute_displacement_rule"]["threshold_set_from_epoch9_v1_0419m_outcome"]
    assert "official env.check_success" in value["per_scene_task_preservation"]["required"][-1]
    assert value["minimal_feasibility_go"]["post_probe_oracle_completion"] == ">=20/24 scenes"


def test_lane_extent_covers_clean_resets_plus_intended_nudge() -> None:
    builder = _builder()
    value = builder.protocol()
    reset = value["clean_reset_basis"]["by_slot"]
    lanes = value["safe_center_lanes_m"]
    for slot in ("front", "back"):
        x_min, y_min = reset[slot]["center_xyz_min_m"][:2]
        x_max, y_max = reset[slot]["center_xyz_max_m"][:2]
        assert lanes[slot]["x"][0] <= x_min
        assert lanes[slot]["x"][1] >= x_max + 0.05
        assert lanes[slot]["y"][0] <= y_min
        assert lanes[slot]["y"][1] >= y_max
