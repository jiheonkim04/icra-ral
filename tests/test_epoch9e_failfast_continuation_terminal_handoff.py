from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DECISION = "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_continuation_terminal_state_and_fixed_denominators_are_exact() -> None:
    handoff = load("epoch9e_failfast_continuation_terminal_handoff.json")
    assert handoff["terminal_state"] == DECISION
    assert handoff["joint_certification_go"] is False
    assert handoff["fixed_denominator_accounting"]["primary_assignments"] == {
        "planned": 24,
        "completed": 23,
        "failed_missing_response": 1,
        "invalid_other": 0,
        "unexecuted": 0,
    }
    assert handoff["fixed_denominator_accounting"]["shams"] == {
        "planned": 12,
        "completed": 12,
        "failed": 0,
        "invalid_other": 0,
        "unexecuted": 0,
    }


def test_continuation_terminal_20261134_and_sensitivity_are_conservative() -> None:
    handoff = load("epoch9e_failfast_continuation_terminal_handoff.json")
    handling = handoff["base_20261134_endpoint_handling"]
    assert handling["binary_pair"] == "adverse/nonflip in fixed 12-pair flip and sign endpoints"
    assert handling["continuous_pair"] == "missing and excluded from the 11-pair observed mean/CI"
    science = handoff["scientific_results"]
    assert science["complete_case_physical"]["n"] == 11
    assert science["fixed_denominator_sign"] == {
        "n": 12,
        "positive": 10,
        "nonpositive_or_missing": 2,
        "one_sided_exact_p": 0.019287109375,
    }
    assert science["worst_case_sensitivity"]["student_t_95_interval_m"][0] > 0
    assert science["worst_case_sensitivity"]["hc3_95_interval_m"][0] > 0
    assert science["twelve_pair_observed_physical_mean_or_ci_reported"] is False


def test_continuation_terminal_failed_gates_close_the_route_and_keep_seals() -> None:
    handoff = load("epoch9e_failfast_continuation_terminal_handoff.json")
    assert handoff["failed_gates"] == [
        "rank_at_least_20_of_24",
        "rank_each_heavy_position_at_least_10_of_12",
        "exact_pair_flips_at_least_9_of_12_with_missing_adverse",
        "fixed_denominator_one_sided_sign_p_below_0_01",
    ]
    assert handoff["validation_accessed"] is False
    assert handoff["confirmation_accessed"] is False
    assert handoff["estimator_development_started"] is False
    assert handoff["official_evaluation_started"] is False
    assert handoff["paper_status"] == "PAPER_NOT_AUTHORIZED"
    assert handoff["paper_paths"] == []


def test_continuation_terminal_hashes_and_protected_manifests_remain_exact() -> None:
    handoff = load("epoch9e_failfast_continuation_terminal_handoff.json")
    frozen = handoff["frozen_hash_proof"]
    assert sha256(ROOT / frozen["controller"]["path"]) == frozen["controller"]["sha256"]
    assert sha256(ROOT / frozen["historical_interrupted_result"]["path"]) == frozen["historical_interrupted_result"]["sha256"]
    protected = {row["path"]: row for row in handoff["protected_untracked_manifests"]}
    assert protected["rollouts/2026_07_17/"]["manifest_sha256"] == "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"
    assert protected["rollouts/2026_07_18/"]["manifest_sha256"] == "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"


def test_continuation_evidence_index_v2_hashes_every_entry() -> None:
    index = load("epoch9e_failfast_continuation_evidence_index_v2.json")
    assert index["terminal_decision"] == DECISION
    assert index["entry_count"] == len(index["entries"])
    assert index["validation_accessed"] is False
    assert index["confirmation_accessed"] is False
    for row in index["entries"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
