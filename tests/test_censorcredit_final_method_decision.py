import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_censorcredit_failure_is_label_collapse_and_forbids_repair() -> None:
    original = _load_json("reports/censor_credit_vla_prototype_result.json")
    repair = _load_json("reports/censor_credit_repair_result.json")
    train = original["train"]
    rows = train["rows"]

    assert train["training_record_count"] == 24
    assert train["training_state_count"] == 6
    assert train["censored_positive_count"] == 4
    assert train["uncensored_positive_count"] == 4
    assert all(not row["prefix_success"] for row in rows)
    assert all(not row["recovered_success"] for row in rows)
    assert sum(row["censored_label"] != row["uncensored_label"] for row in rows) == 0
    assert train["censored_model"]["weights"] == train["uncensored_model"]["weights"]

    assert repair["exact_failure_classification"] == "LABEL_OR_DATA_FAILURE"
    assert repair["repair_allowed"] is False
    assert repair["repair_attempted"] is False
    assert repair["training_run"] is False
    assert repair["closed_loop_rollout_run"] is False
    assert repair["final_decision"] == "CENSORCREDIT_NO_VALID_REPAIR"
    assert repair["evidence"]["rows_with_label_disagreement"] == 0
    assert repair["evidence"]["censored_uncensored_model_weights_equal"] is True


def test_final_method_is_killed_before_implementation() -> None:
    final_method = _load_json("reports/final_distinct_method_result.json")
    final_decision = _load_json("reports/final_autonomous_method_decision.json")

    assert final_method["status"] == "FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION"
    assert final_method["implementation_run"] is False
    assert final_method["training_run"] is False
    assert final_method["closed_loop_rollout_run"] is False
    assert "NEAR_EXACT_PRIOR_ART_DUPLICATION" in final_method["kill_grounds"]
    assert "HARD_UNAVAILABLE_RESOURCE" in final_method["kill_grounds"]
    assert final_method["final_campaign_decision"] == "NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED"

    assert final_decision["censorcredit_decision"] == "CENSORCREDIT_NO_VALID_REPAIR"
    assert final_decision["final_method_status"] == "FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION"
    assert final_decision["final_campaign_decision"] == "NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED"
