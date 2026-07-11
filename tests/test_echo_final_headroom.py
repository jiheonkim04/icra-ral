from tca_map.smolvla.echo_final_headroom import (
    choose_final_decision,
    downstream_headroom_metrics,
    summarize_candidate_diversity,
    summarize_diversity_across_states,
)


def _candidate(index, *, success=False, offset=0.0):
    return {
        "candidate_index": index,
        "postprocessed_action_chunk": [[offset, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] for _ in range(4)],
        "downstream_success": success,
        "effect_compatibility": offset,
    }


def test_candidate_diversity_marks_identical_candidates_impoverished():
    summary = summarize_candidate_diversity([_candidate(index, offset=0.0) for index in range(8)])
    assert summary["state_candidates_impoverished"] is True
    assert summary["effective_distinct_candidates"] == 1
    aggregate = summarize_diversity_across_states([summary] * 8 + [summarize_candidate_diversity([_candidate(0, offset=0.0), _candidate(1, offset=1.0)])] * 4)
    assert aggregate["policy_candidates_impoverished"] is True


def test_downstream_headroom_requires_non_relaxed_original_thresholds_and_two_tasks():
    groups = []
    for group_index in range(10):
        groups.append(
            {
                "group_id": f"g{group_index}",
                "task_key": "task_a" if group_index < 5 else "task_b",
                "phase": "approach" if group_index % 2 == 0 else "transport",
                "random_candidate_index": 1,
                "candidates": [
                    _candidate(0, success=False),
                    _candidate(1, success=group_index in {0, 1, 5}),
                ],
            }
        )
    metrics = downstream_headroom_metrics(groups)
    assert metrics["oracle_improvement_pp"] >= 10.0
    assert metrics["recoverable_default_failure_rate"] >= 0.15
    assert metrics["tasks_with_recovery_count"] == 2
    assert metrics["passes_final_gate"] is True


def test_downstream_headroom_fails_when_gain_is_one_task_only():
    groups = []
    for group_index in range(10):
        groups.append(
            {
                "group_id": f"g{group_index}",
                "task_key": "task_a" if group_index < 5 else "task_b",
                "phase": "approach" if group_index % 2 == 0 else "transport",
                "random_candidate_index": 1,
                "candidates": [
                    _candidate(0, success=False),
                    _candidate(1, success=group_index in {0, 1, 2}),
                ],
            }
        )
    metrics = downstream_headroom_metrics(groups)
    assert metrics["passes_original_thresholds"] is True
    assert metrics["headroom_spans_multiple_tasks"] is False
    assert metrics["passes_final_gate"] is False


def test_final_decision_prioritizes_impoverished_official_policy_candidates():
    decision = choose_final_decision(
        measurement_valid=True,
        official_diversity={"policy_candidates_impoverished": True},
        official_metrics={"passes_final_gate": False},
        structured_metrics={"passes_final_gate": True},
    )
    assert decision == "ECHO_POLICY_CANDIDATES_IMPOVERISHED"

