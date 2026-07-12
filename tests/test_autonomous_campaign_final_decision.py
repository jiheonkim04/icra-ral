import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def test_active_campaign_final_decision_is_nonterminal_pivot() -> None:
    final = (REPORTS / "autonomous_until_paper_final_decision.md").read_text(encoding="utf-8")

    assert "Current campaign decision: `EPOCH_1_COMPLETED_PIVOT_REQUIRED`" in final
    assert "This is not a terminal decision." in final
    assert "READY_TO_DRAFT_RAL_PAPER_PACKAGE" in final
    assert "Start Epoch 2 candidate generation" in final


def test_active_campaign_state_records_governance_v2() -> None:
    state = json.loads((REPORTS / "autonomous_until_paper_state.json").read_text(encoding="utf-8"))

    assert state["governance_file"] == "reports/current_research_governance.md"
    assert state["current_decision"] == "EPOCH_1_COMPLETED_PIVOT_REQUIRED"
    assert state["current_epoch"] == 2
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False


def test_core_ledgers_reference_current_governance() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    manual = (REPORTS / "codex_delegation_manual.md").read_text(encoding="utf-8")
    governance = (REPORTS / "current_research_governance.md").read_text(encoding="utf-8")

    assert "reports/current_research_governance.md" in agents
    assert "Multi-stage autonomous research is permitted" in manual
    assert "There is no finite global method-cycle limit." in governance
