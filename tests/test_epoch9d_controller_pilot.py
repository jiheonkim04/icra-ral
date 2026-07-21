from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.adjudicate_epoch9d_controller_pilot import pilot_gate_from_counts
from scripts.run_epoch9d_controller_pilot import lane_guard_approach_y


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def test_controller_pilot_manifest_is_fresh_and_balanced() -> None:
    protocol = json.loads((REPORTS / "epoch9d_controller_development_protocol.json").read_text(encoding="utf-8"))
    inventory = json.loads((REPORTS / "epoch9d_identity_seed_inventory.json").read_text(encoding="utf-8"))
    state = json.loads((REPORTS / "epoch9d_campaign_state.json").read_text(encoding="utf-8"))
    rows = protocol["variant1_pilot_manifest"]
    assert len(rows) == 12
    assert [row["generated_identity_id"] for row in rows] == list(range(72, 84))
    assert [row["generator_seed"] for row in rows] == list(range(914200, 914212))
    assert {row["partition"] for row in rows} == {"CONTROLLER_PILOT_DEVELOPMENT"}
    assert sum(row["heavy_slot"] == "front" for row in rows) == 6
    assert sum(row["heavy_slot"] == "back" for row in rows) == 6
    assert sum(row["instruction_property"] == "heaviest" for row in rows) == 6
    assert sum(row["probe_order"] == ["front", "back"] for row in rows) == 6
    assert all(row["outcomes_accessed_during_construction"] == [] for row in rows)
    assert all(row["mass_assignment_applied_during_construction"] is False for row in rows)
    assert state["identity_and_seed_allocations"]["controller_pilot_generated_identity_ids"] == list(range(72, 96))
    assert set(range(914200, 914212)).isdisjoint(inventory["seed_values"])


def test_lane_guard_is_rgb_geometry_only_and_points_inward() -> None:
    protocol = json.loads((REPORTS / "epoch9b_v2_task_preservation_protocol.json").read_text(encoding="utf-8"))
    development = json.loads((REPORTS / "epoch9d_controller_development_protocol.json").read_text(encoding="utf-8"))
    guard = development["variant1"]["guard"]
    config = campaign.ControllerConfig(**development["variant1"]["base_controller_config"])
    parameters = set(inspect.signature(lane_guard_approach_y).parameters)
    assert "mass" not in parameters
    assert "property" not in parameters
    front_lane = protocol["safe_center_lanes_m"]["front"]["y"]
    upper_y = front_lane[1] - 0.005
    value, event = lane_guard_approach_y("front", upper_y, -0.055, config, protocol, guard, upper_y - 0.018)
    assert event is not None and event["nearest_boundary"] == "upper"
    assert value > upper_y
    lower_y = front_lane[0] + 0.005
    value, event = lane_guard_approach_y("front", lower_y, -0.055, config, protocol, guard, lower_y - 0.018)
    assert event is not None and event["nearest_boundary"] == "lower"
    assert value < lower_y
    untouched, event = lane_guard_approach_y("front", upper_y, -0.080, config, protocol, guard, 0.105)
    assert untouched == 0.105 and event is None


def test_pilot_gate_is_exact_and_one_failure_does_not_pass() -> None:
    counts = {
        "complete_scenes": 12,
        "finite_bounded_actions": 24,
        "intended_contact_or_excitation": 23,
        "lane_and_reachability": 24,
        "collisions": 0,
        "identity_swaps": 0,
        "falls": 0,
        "workspace_exits": 0,
        "rank_correct": 10,
        "rank_by_heavy_position": {"front": {"correct": 5}, "back": {"correct": 5}},
        "oracle_completion": 10,
    }
    assert all(pilot_gate_from_counts(counts).values())
    counts["lane_and_reachability"] = 23
    gates = pilot_gate_from_counts(counts)
    assert gates["lane_and_reachability_24_of_24"] is False
    assert sum(not value for value in gates.values()) == 1


def test_controller_pilot_execution_seal_binds_every_executable_after_build() -> None:
    path = REPORTS / "epoch9d_controller_pilot_execution_seal.json"
    if not path.exists():
        return
    seal = json.loads(path.read_text(encoding="utf-8"))
    assert seal["outcomes_accessed_before_seal"] is False
    assert seal["frozen_counts"] == {"scenes": 12, "candidate_probes": 24}
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("runner_path", "runner_sha256"),
        ("adjudicator_path", "adjudicator_sha256"),
        ("host_wrapper_path", "host_wrapper_sha256"),
        ("original_epoch9b_runner_path", "original_epoch9b_runner_sha256"),
        ("original_controller_freeze_path", "original_controller_freeze_sha256"),
        ("calibration_path", "calibration_sha256"),
    ):
        assert sha256(ROOT / seal[path_key]) == seal[hash_key]
