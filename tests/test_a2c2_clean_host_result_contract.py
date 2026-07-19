from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "a2c2_prior"


def load(name: str) -> dict:
    return json.loads((REPORT_ROOT / name).read_text(encoding="utf-8"))


def test_clean_host_resource_result_selects_smallest_passing_cap() -> None:
    report = load("clean_host_resource_smoke_result.json")

    assert [row["cap_gib"] for row in report["caps"]] == [8, 10, 12]
    assert [row["final_decision"] for row in report["caps"]] == [
        "A2C2_RESOURCE_SMOKE_FAIL_MEMORY_LEAK",
        "A2C2_RESOURCE_SMOKE_FAIL_MEMORY_LEAK",
        "A2C2_RESOURCE_SMOKE_PASS",
    ]
    assert report["smallest_passing_cap_gib"] == 12
    assert report["optional_14gb_executed"] is False


def test_clean_host_scientific_result_is_complete_matched_and_valid() -> None:
    report = load("clean_host_prior_verification_result.json")

    panel = report["frozen_panel"]
    assert panel["completed_scientific_rows"] == panel["planned_scientific_rows"] == 45
    assert panel["duplicate_keys"] == 0
    assert panel["exceptions"] == 0
    assert panel["all_actions_finite"] is True
    assert panel["delayed_base_prior_identity_match"] is True
    assert report["execution"]["expert_action_at_live_inference"] is False


def test_clean_host_decision_preserves_frozen_gate_semantics() -> None:
    report = load("clean_host_prior_verification_result.json")

    assert report["frozen_panel"]["base_standard"]["successes"] == 10
    assert report["frozen_panel"]["base_delayed"]["successes"] == 4
    assert report["frozen_panel"]["prior_delayed"]["successes"] == 3
    assert report["frozen_gates"]["base_competent"] is True
    assert report["frozen_gates"]["repeatable_delay_gap"] is True
    assert report["frozen_gates"]["prior_improves"] is False
    assert report["frozen_adjudicator_decision"] == "NO_DIAGNOSTIC_HEADROOM"
    assert report["final_decision"] == "A2C2_PRIOR_NO_LOCAL_IMPROVEMENT"
    assert report["ours_designed_or_executed"] is False
