import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def test_active_campaign_final_decision_is_nonterminal_pivot() -> None:
    final = (REPORTS / "autonomous_until_paper_final_decision.md").read_text(encoding="utf-8")

    assert "Current campaign decision: `EPOCH_4_CYCLE_3_FANG_STAGE_B_VALID_KILL_NEXT_METHOD_REQUIRED`" in final
    assert "This is not a terminal decision." in final
    assert "READY_TO_DRAFT_RAL_PAPER_PACKAGE" in final
    assert "FANG-VLA" in final
    assert "fang_c01" in final
    assert "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT" in final
    assert "11 / 40" in final
    assert "16 / 40" in final
    assert "CAVM-VLA" in final
    assert "24 / 58" in final
    assert "23 / 58" in final


def test_active_campaign_state_records_governance_v2() -> None:
    state = json.loads((REPORTS / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["governance_file"] == "reports/current_research_governance.md"
    assert state["current_decision"] == "EPOCH_4_CYCLE_3_FANG_STAGE_B_VALID_KILL_NEXT_METHOD_REQUIRED"
    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 4
    assert state["current_stage"] == "epoch_4_cycle_4_candidate_search_pending"
    assert state["method"] == "NEXT_METHOD_UNSELECTED"
    assert state["proposal_hash"] is None
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["epoch_2_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    assert state["epoch_3_cycle_2_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_4_cycle_1_outcome"]["final_decision"] == "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_4_cycle_1_outcome"]["rcv_full_successes"] == 20
    assert state["epoch_4_cycle_1_outcome"]["rcv_no_context_ablation_successes"] == 24
    assert state["epoch_4_cycle_2_outcome"]["final_decision"] == "STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION"
    assert state["epoch_4_cycle_2_outcome"]["cavm_full_successes"] == 24
    assert state["epoch_4_cycle_2_outcome"]["nearest_success_replay_successes"] == 23
    assert state["next_action"].startswith("Begin Epoch 4 Cycle 4")
    assert "post_pse_research_design_governance_applied" in state["completed_stages"]
    assert "epoch_4_cycle_1_rcv_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "post_cavm_performance_governance_applied" in state["completed_stages"]
    assert "epoch_4_cycle_3_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert state["epoch_4_cycle_3_outcome"]["final_decision"] == "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    assert state["epoch_4_cycle_3_outcome"]["fang_full_successes"] == 11
    assert state["epoch_4_cycle_3_outcome"]["base_smolvla_successes"] == 16


def test_core_ledgers_reference_current_governance() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    manual = (REPORTS / "codex_delegation_manual.md").read_text(encoding="utf-8")
    governance = (REPORTS / "current_research_governance.md").read_text(encoding="utf-8")

    assert "reports/current_research_governance.md" in agents
    assert "Multi-stage autonomous research is permitted" in manual
    assert "There is no finite global method-cycle limit." in governance
    assert "Post-CAVM Performance-Oriented Research Design Governance" in governance
