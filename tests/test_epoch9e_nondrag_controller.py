from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.epoch9e_nondrag_controller import inward_approach_y, vertical_liftoff_action


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def test_vertical_liftoff_has_exact_zero_planar_command() -> None:
    action = vertical_liftoff_action(campaign.ControllerConfig(), 0.65)
    assert action.shape == (7,)
    assert action[0] == 0.0
    assert action[1] == 0.0
    assert action[2] == 0.65
    assert action[3] == action[4] == action[5] == 0.0
    assert action[6] == 1.0


def test_inward_orientation_is_mirrored_from_rgb_lane_geometry_only() -> None:
    protocol = json.loads((REPORTS / "epoch9b_v2_task_preservation_protocol.json").read_text(encoding="utf-8"))
    config = campaign.ControllerConfig()
    parameters = set(inspect.signature(inward_approach_y).parameters)
    assert "mass" not in parameters and "property" not in parameters and "score" not in parameters
    upper, upper_event = inward_approach_y("front", 0.170, -0.055, config, protocol)
    lower, lower_event = inward_approach_y("front", 0.120, -0.055, config, protocol)
    back_upper, back_upper_event = inward_approach_y("back", 0.085, -0.055, config, protocol)
    back_lower, back_lower_event = inward_approach_y("back", 0.020, -0.055, config, protocol)
    assert upper > 0.170 and upper_event["nearest_boundary"] == "upper"
    assert lower < 0.120 and lower_event["nearest_boundary"] == "lower"
    assert back_upper > 0.085 and back_upper_event["nearest_boundary"] == "upper"
    assert back_lower < 0.020 and back_lower_event["nearest_boundary"] == "lower"
    transit, event = inward_approach_y("front", 0.170, -0.080, config, protocol)
    assert transit == config.front_clear_approach_y_m
    assert event["mode"] == "front_high_clear_transit"
