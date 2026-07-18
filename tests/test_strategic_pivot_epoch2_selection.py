from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((REPO_ROOT / "reports" / name).read_text(encoding="utf-8"))


def test_pivot_epoch2_has_exactly_two_rejected_candidates() -> None:
    report = _load("strategic_pivot_epoch2_selection_result.json")

    assert report["execution_type"] == "REPORT_ONLY"
    assert report["epoch"] == "PIVOT_EPOCH_2"
    assert report["candidate_count"] == 2
    assert len(report["candidates"]) == 2
    assert report["selected_thesis_id"] is None
    assert report["decision"] == "NO_DEFENSIBLE_PIVOT_FOUND"
    assert report["terminal_campaign_status"] == "NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND"
    assert all(not item["hard_filter_pass"] for item in report["candidates"])
    assert report["selection_adjudication"]["hard_filter_passing_candidates"] == []
    assert report["selection_adjudication"]["materially_different_from_pivot_epoch_1"] is True
    assert report["selection_adjudication"]["materially_different_from_wrist_dropout"] is True


def test_candidate_totals_match_frozen_formula() -> None:
    report = _load("strategic_pivot_epoch2_selection_result.json")

    for candidate in report["candidates"]:
        scores = candidate["scores"]
        total = (
            2 * scores["N"]
            + 2 * scores["R"]
            + 2 * scores["H"]
            + 2 * scores["F"]
            + 1.5 * scores["P"]
            + 1.5 * scores["C"]
            + scores["G"]
            + scores["A"]
            + scores["D"]
        )
        assert total == candidate["total"]


def test_terminal_report_forbids_unauthorized_progression() -> None:
    report = _load("strategic_pivot_final_decision.json")

    assert report["final_decision"] == "NO_DEFENSIBLE_LOCAL_RESEARCH_PATH_FOUND"
    assert report["paper_package_status"] == "NOT_AUTHORIZED"
    assert report["pivot_epoch_1"]["ours_authorized"] is False
    assert report["pivot_epoch_2"]["candidate_count"] == 2
    assert report["pivot_epoch_2"]["hard_filter_passing_candidates"] == 0
    assert all(report["prohibitions"].values())
