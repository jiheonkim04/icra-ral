from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_epoch9b_dynamic_nudge.py"
    spec = importlib.util.spec_from_file_location("epoch9b_dynamic_nudge", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_controller_config_is_fixed_impulse_and_information_safe() -> None:
    module = _module()
    config = module.ControllerConfig()
    assert config.impulse_action_x == 0.65
    assert config.impulse_steps == 3
    assert config.coast_steps == 2
    assert config.gripper_closed_command == 1.0
    assert config.approach_increment_m == 0.003
    assert config.contact_verify_retract_m > 0
    assert module.paddle_y_offset("front", config) == -config.paddle_y_offset_m
    assert module.paddle_y_offset("back", config) == config.paddle_y_offset_m
    assert module.approach_y("front", 0.17, -0.08, config) == config.front_clear_approach_y_m
    assert module.approach_y("front", 0.17, -0.05, config) == 0.17 - config.paddle_y_offset_m
    centered = module.ControllerConfig(front_centered_contact=True)
    assert module.approach_y("front", 0.17, -0.05, centered) == 0.17
    inward = module.ControllerConfig(front_inward_contact=True)
    assert module.approach_y("front", 0.17, -0.05, inward) > 0.17
    assert module.approach_y("front", 0.13, -0.05, inward) < 0.13
    assert module.calibrated_rank_scores({"front": 0.0, "back": 0.004}, 0.005)["back"] < 0
    assert module.calibrated_rank_scores({"front": 0.0, "back": 0.008}, 0.005)["front"] < 0


def test_development_manifest_is_outside_panel_and_balanced() -> None:
    module = _module()
    rows = module.development_manifest(8)
    assert len({row["scene_id"] for row in rows}) == 8
    assert all(row["source_state_demo_index"] == 31 for row in rows)
    assert sum(row["heavy_slot"] == "front" for row in rows) == 4
    assert sum(row["probe_order"] == ["front", "back"] for row in rows) == 4


def test_lane_contains_uses_frozen_lane_and_reachability() -> None:
    module = _module()
    import json

    protocol = json.loads(module.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert module.lane_contains(protocol, "front", [0.10, 0.14, 0.90])
    assert not module.lane_contains(protocol, "front", [0.20, 0.14, 0.90])
    assert module.lane_contains(protocol, "back", [-0.14, 0.05, 0.90])


def test_panel_manifest_has_exact_frozen_counts() -> None:
    module = _module()
    import json

    protocol = json.loads(module.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert len(protocol["feasibility_manifest"]) == 24
    assert sum(row["heavy_slot"] == "front" for row in protocol["feasibility_manifest"]) == 12


def test_edge_stress_manifest_is_balanced_and_near_front_lane_ceiling() -> None:
    module = _module()
    rows = module.edge_stress_manifest(4)
    assert sum(row["heavy_slot"] == "front" for row in rows) == 2
    assert all(0.17 <= row["candidate_initial_xy_m"]["front"][1] <= 0.174 for row in rows)
