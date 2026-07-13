import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_active_state_is_epoch_4_cycle_1_pivot_without_cycle_cap() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 1
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED"
    assert state["valid_final_states"] == ALLOWED_FINAL_STATES
    assert state["epoch_2_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    assert state["epoch_2_cycle_2_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    assert state["epoch_2_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_2_cycle_3_outcome"]["ocfn_full_successes"] == 26
    assert state["epoch_2_cycle_3_outcome"]["zero_noise_smolvla_successes"] == 27
    assert state["epoch_2_cycle_3_outcome"]["paired_upper_ci_vs_strongest_baseline"] == 0.0625
    assert state["epoch_3_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    assert state["epoch_3_cycle_1_outcome"]["cbfd_full_successes"] == 0
    assert state["epoch_3_cycle_1_outcome"]["frozen_smolvla_successes"] == 7
    assert state["epoch_3_cycle_2_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_2_outcome"]["scvc_full_successes"] == 11
    assert state["epoch_3_cycle_2_outcome"]["shifted_frozen_smolvla_successes"] == 20
    assert state["epoch_3_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_3_outcome"]["pse_full_successes"] == 50
    assert state["epoch_3_cycle_3_outcome"]["bright_single_successes"] == 51
    assert state["epoch_3_cycle_3_outcome"]["validation_unique_keys"] == 400
    assert state["epoch_3_synthesis"]["next_epoch"] == 4


def test_epoch_1_corrected_adjudication_records_all_cycles() -> None:
    adjudication = (REPO_ROOT / "reports" / "epoch_1_corrected_adjudication.md").read_text(encoding="utf-8")

    assert "DICD-VLA" in adjudication
    assert "UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED" in adjudication
    assert "FEDO-VLA" in adjudication
    assert "VALID_CURRENT_FORMULATION_KILL" in adjudication
    assert "GCAP-VLA" in adjudication
    assert "UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED" in adjudication
