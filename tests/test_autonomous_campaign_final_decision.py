from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def test_autonomous_campaign_final_decision_is_explicit() -> None:
    final = (REPORTS / "autonomous_campaign_final_decision.md").read_text(encoding="utf-8")

    assert "Final decision: `NO_METHOD_AFTER_3_VALID_CYCLES`" in final
    assert "Paper-ready status: `false`" in final
    assert final.count("Cycle 01") >= 1
    assert final.count("Cycle 02") >= 1
    assert final.count("Cycle 03") >= 1


def test_autonomous_campaign_state_records_resource_bounds() -> None:
    state = (REPORTS / "autonomous_campaign_state.md").read_text(encoding="utf-8")

    assert "new downloads: `0 GiB`" in state
    assert "active GPU time in this batch: `0 h`" in state
    assert "WSL Git caveat" in state


def test_core_ledgers_reference_autonomous_campaign_reports() -> None:
    project_state = (REPORTS / "project_state.md").read_text(encoding="utf-8")
    next_actions = (REPORTS / "next_actions.md").read_text(encoding="utf-8")
    decision_log = (REPORTS / "decision_log.md").read_text(encoding="utf-8")

    assert "autonomous_campaign_final_decision.md" in project_state
    assert "NO_METHOD_AFTER_3_VALID_CYCLES" in next_actions
    assert "Autonomous Dual-Review RA-L Campaign" in decision_log
