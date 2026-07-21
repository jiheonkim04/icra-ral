from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def test_epoch9e_joint_manifest_is_exact_fresh_balanced_and_one_shot() -> None:
    protocol = json.loads((REPORTS / "epoch9e_joint_certification_protocol.json").read_text(encoding="utf-8"))
    identities = json.loads((REPORTS / "epoch9e_fresh_identity_manifest.json").read_text(encoding="utf-8"))
    allocation = identities["allocations"]["joint_certification_base_pairs"]
    bases = protocol["base_states"]
    assignments = protocol["assignments"]
    assert len(bases) == 12
    assert len(assignments) == 24
    assert [row["base_identity_id"] for row in bases] == allocation["identity_ids"]
    assert [row["generator_seed"] for row in bases] == allocation["generator_seeds"]
    assert len({row["base_state_vector_sha256"] for row in bases}) == 12
    assert len({row["first_agentview_rgb_sha256"] for row in bases}) == 12
    assert min(value for row in bases for value in row["candidate_initial_lane_margin_m_eval_only"].values()) >= 0.0049
    assert Counter(tuple(row["probe_order"]) for row in bases) == {("front", "back"): 6, ("back", "front"): 6}
    assert Counter(row["instruction_property"] for row in bases) == {"heaviest": 6, "lightest": 6}
    assert Counter(row["heavy_slot_eval_only"] for row in assignments) == {"front": 12, "back": 12}
    assert Counter(row["instruction_property"] for row in assignments) == {"heaviest": 12, "lightest": 12}
    pairs: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in assignments:
        pairs[row["base_identity_id"]][row["assignment"]] = row
    assert all(set(pair) == {"A", "B"} for pair in pairs.values())
    assert all(pair["A"]["mass_factor"] == {"front": 1.0, "back": 8.0} for pair in pairs.values())
    assert all(pair["B"]["mass_factor"] == {"front": 8.0, "back": 1.0} for pair in pairs.values())
    assert protocol["one_shot"] == {
        "panels": 1,
        "near_miss_rerun": False,
        "row_replacement": False,
        "endpoint_repair_after_outcome": False,
    }
    assert protocol["outcomes_accessed_during_protocol_construction"] == []
    assert protocol["validation_accessed"] is False
    assert protocol["confirmation_accessed"] is False


def test_epoch9e_controller_contract_changes_only_nondrag_disengagement() -> None:
    protocol = json.loads((REPORTS / "epoch9e_joint_certification_protocol.json").read_text(encoding="utf-8"))
    controller = protocol["controller_contract"]
    assert controller["only_replaced_phase"] == "pre-response contact_verify_retract"
    assert controller["nondrag_lift"]["commanded_planar_component"] == 0.0
    assert controller["nondrag_lift"]["direction_world_xyz"] == [0.0, 0.0, 1.0]
    assert controller["inward_orientation"]["mirrored_front_back"] is True
    assert controller["inward_orientation"]["mass_or_label_input"] is False
    assert controller["response_window"]["unchanged"] is True
    assert controller["response_window"]["action_sequence_x"] == [0.65, 0.65, 0.65, 0.0, 0.0]
    assert controller["primary_score"]["threshold_m"] == 0.005219466062047384
    assert set(controller["forbidden_action_inputs"]) == {
        "mass", "property label", "response-score sign", "simulator pose", "force", "reward", "success", "segmentation", "oracle identity"
    }


def test_epoch9e_joint_gates_and_shams_match_authority() -> None:
    protocol = json.loads((REPORTS / "epoch9e_joint_certification_protocol.json").read_text(encoding="utf-8"))
    gate = protocol["joint_go"]
    assert gate["finite_bounded_actions"] == {"minimum": 48, "denominator": 48}
    assert gate["intended_contact_or_excitation"]["minimum"] == 46
    assert gate["both_candidates_excited"]["minimum"] == 22
    assert gate["full_trajectory_lane_reachable"]["minimum"] == 48
    assert gate["rank_correct"] == {"minimum": 20, "denominator": 24}
    assert gate["rank_correct_each_heavy_position"] == {"minimum": 10, "denominator": 12}
    assert gate["exact_pair_both_assignments_flip_correctly"] == {"minimum": 9, "denominator": 12}
    assert gate["completion_oracle"] == {"minimum": 20, "denominator": 24}
    assert gate["completion_oracle_each_heavy_position"] == {"minimum": 9, "denominator": 12}
    assert protocol["paired_test"]["p_strictly_less_than"] == 0.01
    assert protocol["position_order_control"]["first_rgb_exact_pairs"] == 12
    assert protocol["sham_control"]["base_state_count"] == 6
    assert protocol["sham_control"]["row_count"] == 12
    assert len(protocol["sham_control"]["manifest"]) == 12
    assert protocol["success_decision"] == "EPOCH9E_JOINT_CERTIFICATION_GO"
    assert protocol["failure_decision"] == "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED"


def test_epoch9e_protocol_builder_resource_record_binds_protocol() -> None:
    monitor = json.loads((REPORTS / "epoch9e_joint_protocol_builder_resource.json").read_text(encoding="utf-8-sig"))
    protocol_path = REPORTS / "epoch9e_joint_certification_protocol.json"
    assert monitor["runner_exit_code"] == 0
    assert monitor["host_ram_ceiling_breached"] is False
    assert monitor["peak_host_ram_percent"] < 82.0
    assert monitor["scientific_outcomes_accessed"] is False
    assert monitor["protocol_sha256"] == sha256(protocol_path)


def test_epoch9e_exact_pair_preflight_is_outcome_suppressed_and_exact() -> None:
    result_path = REPORTS / "epoch9e_exact_pair_preflight.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    monitor = json.loads((REPORTS / "epoch9e_exact_pair_preflight_host_resource.json").read_text(encoding="utf-8-sig"))
    assert result["summary"] == {
        "assignment_rows": 24,
        "first_rgb_exact_rows": 24,
        "initial_localization_exact_rows": 24,
        "a_b_rgb_exact_pairs": 12,
        "a_b_localization_exact_pairs": 12,
    }
    assert all(row["actions_executed"] == 0 for row in result["rows"])
    assert all(row["reward_done_success_accessed"] is False for row in result["rows"])
    assert result["resource"]["wsl_swap_used_peak_bytes"] == 0
    assert result["scientific_outcomes_accessed"] is False
    assert result["validation_accessed"] is False
    assert result["confirmation_accessed"] is False
    assert monitor["runner_exit_code"] == 0
    assert monitor["host_ram_ceiling_breached"] is False
    assert monitor["peak_host_ram_percent"] < 82.0
    assert monitor["result_sha256"] == sha256(result_path)


def test_epoch9e_mechanics_smoke_protocol_is_fresh_label_blind_and_unscored() -> None:
    smoke_path = REPORTS / "epoch9e_mechanics_smoke_protocol.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    identities = json.loads((REPORTS / "epoch9e_fresh_identity_manifest.json").read_text(encoding="utf-8"))
    allocation = identities["allocations"]["mechanics_smoke"]
    monitor = json.loads((REPORTS / "epoch9e_mechanics_smoke_protocol_host_resource.json").read_text(encoding="utf-8-sig"))
    assert smoke["scene_count"] == 8
    assert smoke["candidate_probe_count"] == 16
    assert [row["generated_identity_id"] for row in smoke["manifest"]] == allocation["identity_ids"]
    assert [row["generator_seed"] for row in smoke["manifest"]] == allocation["generator_seeds"]
    assert all(row["mass_factor"] == {"front": 1.0, "back": 1.0} for row in smoke["manifest"])
    assert all(row["mass_rank_authorized"] is False for row in smoke["manifest"])
    assert all(row["oracle_success_authorized"] is False for row in smoke["manifest"])
    assert smoke["must_not_compute_or_reveal"] == ["mass rank", "mass-conditioned response", "oracle task success"]
    assert smoke["scientific_outcomes_accessed"] is False
    assert monitor["runner_exit_code"] == 0
    assert monitor["host_ram_ceiling_breached"] is False
    assert monitor["result_sha256"] == sha256(smoke_path)
