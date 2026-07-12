import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_active_state_is_epoch_2_pivot_without_cycle_cap() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8"))

    assert state["current_epoch"] == 2
    assert state["current_cycle"] == 0
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "EPOCH_1_COMPLETED_PIVOT_REQUIRED"
    assert state["valid_final_states"] == ALLOWED_FINAL_STATES


def test_epoch_1_corrected_adjudication_records_all_cycles() -> None:
    adjudication = (REPO_ROOT / "reports" / "epoch_1_corrected_adjudication.md").read_text(encoding="utf-8")

    assert "DICD-VLA" in adjudication
    assert "UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED" in adjudication
    assert "FEDO-VLA" in adjudication
    assert "VALID_CURRENT_FORMULATION_KILL" in adjudication
    assert "GCAP-VLA" in adjudication
    assert "UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED" in adjudication
