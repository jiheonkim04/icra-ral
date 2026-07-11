import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_implementation_v2_terminal_decision_requires_two_implemented_kills():
    state = _json(REPORTS / "implementation_v2_campaign_state.json")

    assert state["previous_terminal_decision_reclassified_as"] == "PREMATURE_LITERATURE_ONLY_TERMINATION"
    assert state["final_decision"] == "TWO_IMPLEMENTED_METHODS_KILLED"
    assert len(state["implemented_cycles"]) == 2
    assert all(cycle["training_happened"] for cycle in state["implemented_cycles"])
    assert all(cycle["closed_loop_experiment_happened"] for cycle in state["implemented_cycles"])
    assert all(cycle["go"] is False for cycle in state["implemented_cycles"])


def test_phase_barrier_and_censor_credit_results_are_valid_kills():
    phase = _json(REPORTS / "phase_barrier_vla_prototype_result.json")
    censor = _json(REPORTS / "censor_credit_vla_prototype_result.json")

    assert phase["final_decision"] == "PHASE_BARRIER_VALID_KILL"
    assert phase["training_happened"] is True
    assert phase["closed_loop_experiment_happened"] is True
    assert phase["summary"]["passes_prototype_go"] is False

    assert censor["final_decision"] == "CENSOR_CREDIT_VALID_KILL"
    assert censor["training_happened"] is True
    assert censor["closed_loop_experiment_happened"] is True
    assert censor["summary"]["passes_prototype_go"] is False
    assert (
        censor["summary"]["full_task_balanced_success_rate"]
        == censor["summary"]["ablation_task_balanced_success_rate"]
    )


def test_core_ledgers_reference_implementation_v2_decision():
    project_state = (REPORTS / "project_state.md").read_text(encoding="utf-8")
    next_actions = (REPORTS / "next_actions.md").read_text(encoding="utf-8")
    decision_log = (REPORTS / "decision_log.md").read_text(encoding="utf-8")

    assert "implementation_v2_final_decision.md" in project_state
    assert "TWO_IMPLEMENTED_METHODS_KILLED" in next_actions
    assert "Autonomous RA-L Research Implementation V2" in decision_log
