import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def _json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_phase_barrier_result_counts_match_postmortem():
    result = _json("phase_barrier_vla_prototype_result.json")
    postmortem = (REPORTS / "phase_barrier_empirical_postmortem.md").read_text(encoding="utf-8")

    assert result["train"]["training_state_count"] == 5
    assert result["train"]["training_record_count"] == 20
    assert result["train"]["positive_label_count"] == 8
    assert result["train"]["negative_label_count"] == 12
    assert len(result["episodes"]) == 10
    assert result["summary"]["by_variant"]["phase_barrier_full"]["successes"] == 0
    assert result["summary"]["by_variant"]["phase_barrier_full"]["total"] == 2
    assert result["summary"]["by_variant"]["phase_barrier_full"]["mean_action_delta_norm"] == 0.111434
    assert "UNDERPOWERED_PROTOTYPE_INCONCLUSIVE" in postmortem


def test_censor_credit_label_collapse_matches_postmortem():
    result = _json("censor_credit_vla_prototype_result.json")
    postmortem = (REPORTS / "censor_credit_empirical_postmortem.md").read_text(encoding="utf-8")

    rows = result["train"]["rows"]
    assert result["train"]["training_state_count"] == 6
    assert result["train"]["training_record_count"] == 24
    assert result["train"]["censored_positive_count"] == 4
    assert result["train"]["uncensored_positive_count"] == 4
    assert all(row["censored_label"] == row["uncensored_label"] for row in rows)
    assert result["train"]["censored_model"]["weights"] == result["train"]["uncensored_model"]["weights"]
    assert result["summary"]["full_task_balanced_success_rate"] == 0.5
    assert result["summary"]["ablation_task_balanced_success_rate"] == 0.5
    assert "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE" in postmortem


def test_final_postmortem_decision_is_not_terminal_two_kills():
    final_decision = (REPORTS / "final_method_decision.md").read_text(encoding="utf-8")
    comparison = (REPORTS / "two_method_failure_comparison.md").read_text(encoding="utf-8")

    assert "PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM" in final_decision
    assert "TWO_IMPLEMENTED_METHODS_KILLED" in comparison
    assert "UNDERPOWERED_PROTOTYPE_INCONCLUSIVE" in comparison
    assert "IMPLEMENTATION_OR_OPTIMIZATION_FAILURE" in comparison
