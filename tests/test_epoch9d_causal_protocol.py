from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_epoch9d_raw_trace_diagnostic_reproduces_frozen_ranking() -> None:
    diagnostic = load("epoch9d_existing_trace_causal_diagnostic.json")
    score = diagnostic["frozen_primary_score"]
    assert score["threshold_m"] == 0.005219466062047384
    assert score["secondary_score_frozen"] is False
    assert diagnostic["campaign_summaries"]["original_frozen_panel"]["rank_correct"] == 22
    assert diagnostic["campaign_summaries"]["repair3_frozen_panel"]["rank_correct"] == 19
    assert len(diagnostic["all_scene_rows"]) == 64
    for row in diagnostic["all_scene_rows"]:
        back = row["primary_back_response_m"]
        reconstructed = min(
            {"front": score["threshold_m"] - back, "back": back - score["threshold_m"]},
            key={"front": score["threshold_m"] - back, "back": back - score["threshold_m"]}.get,
        )
        assert reconstructed == row["predicted_heavy_slot"]
        assert all((ROOT / metrics["trace_path"]).exists() for metrics in row["probe_metrics_by_slot"].values())


def test_epoch9d_causal_protocol_is_exact_balanced_and_sealed() -> None:
    protocol = load("epoch9d_causal_panel_protocol.json")
    state = load("epoch9d_campaign_state.json")
    inventory = load("epoch9d_identity_seed_inventory.json")
    assert protocol["base_state_count"] == 16
    assert protocol["assignment_scene_count"] == 32
    assert protocol["candidate_probe_count"] == 64
    assert protocol["validation_accessed"] is False
    assert protocol["confirmation_accessed"] is False
    assert protocol["secondary_score"] is None
    assert protocol["primary_score"]["name"] == "original_back_slot_rgb_response_threshold_score"
    bases = protocol["base_states"]
    assert [row["base_identity_id"] for row in bases] == list(range(56, 72))
    assert [row["generator_seed"] for row in bases] == list(range(914100, 914116))
    assert set(row["generator_seed"] for row in bases).isdisjoint(inventory["seed_values"])
    assert set(row["base_identity_id"] for row in bases).isdisjoint(range(40, 50))
    assert Counter(row["spatial_stratum"] for row in bases) == Counter({f"spatial_{i}": 4 for i in range(4)})
    assert Counter(row["probe_order"][0] for row in bases) == Counter({"front": 8, "back": 8})
    assert len({row["base_state_vector_sha256"] for row in bases}) == 16
    for row in bases:
        vector = np.asarray(row["base_state_vector_float64"], dtype=np.float64)
        assert array_sha256(vector) == row["base_state_vector_sha256"]
        assert row["mass_assignment_applied_during_construction"] is False
        assert row["outcomes_accessed_during_construction"] == []
        assert min(row["candidate_initial_lane_margin_m_eval_only"].values()) >= 0.010
        assert min(value["quality"] for value in row["initial_rgb_localization_audit"].values()) >= 0.50
    assignments = protocol["assignments"]
    assert Counter(row["heavy_slot_eval_only"] for row in assignments) == Counter({"front": 16, "back": 16})
    for identity in range(56, 72):
        pair = [row for row in assignments if row["base_identity_id"] == identity]
        assert [row["assignment"] for row in pair] == ["A", "B"]
        assert pair[0]["mass_factor"] == {"front": 1.0, "back": 8.0}
        assert pair[1]["mass_factor"] == {"front": 8.0, "back": 1.0}
        assert pair[0]["probe_order"] == pair[1]["probe_order"]
    assert state["phase_status"]["B_mass_swap_causal_panel"] == "FROZEN_READY_TO_RUN"


def test_epoch9d_causal_protocol_hashes_and_gates_are_fixed() -> None:
    protocol = load("epoch9d_causal_panel_protocol.json")
    state = load("epoch9d_campaign_state.json")
    for record in (
        protocol["phase_a_diagnostic"],
        protocol["original_controller_freeze"],
    ):
        assert sha256(ROOT / record["path"]) == record["sha256"]
    phase_a = state["phase_a_evidence"]
    assert sha256(ROOT / phase_a["diagnostic_json"]["path"]) == phase_a["diagnostic_json"]["sha256"]
    assert sha256(ROOT / phase_a["diagnostic_markdown"]["path"]) == phase_a["diagnostic_markdown"]["sha256"]
    assert sha256(ROOT / phase_a["causal_panel_protocol"]["path"]) == phase_a["causal_panel_protocol"]["sha256"]
    gates = protocol["causal_signal_go"]
    assert gates["correct_heavy_light_rank_scenes"] == {"minimum": 26, "denominator": 32}
    assert gates["correct_rank_each_heavy_position_stratum"] == {"minimum": 12, "denominator": 16}
    assert gates["exact_state_pairs_both_assignments_flip_correctly"] == {"minimum": 12, "denominator": 16}
    assert gates["mass_intervention_test"]["one_sided_exact_sign_test_p_strictly_less_than"] == 0.01
    assert protocol["near_miss_replication"]["pooling"] is False
    assert protocol["near_miss_replication"]["second_replication"] is False
    sham = protocol["sham_control"]
    assert sham["base_state_count"] == 8
    assert sham["sham_row_count"] == 16
    assert len(sham["manifest"]) == 16
