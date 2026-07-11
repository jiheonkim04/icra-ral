import pytest

from tca_map.smolvla.echo_vla import (
    CandidateRecord,
    assert_no_privileged_deployment_inputs,
    build_candidate_record,
    candidate_headroom_metrics,
    compatibility_score,
    pairwise_ranking_pairs,
    required_effect_for_phase,
    validate_counterfactual_group,
)


def _candidate(index, *, success=False, effect=None):
    return build_candidate_record(
        group_id="g0",
        candidate_index=index,
        start_state=[1.0, 2.0, 3.0],
        start_observation={"eef": [0.0, 0.0, 0.0], "image_hash": "abc"},
        action_chunk=[[0.01 * index] * 7, [0.02 * index] * 7],
        horizon=2,
        phase="transport",
        realized_effect=effect or {"object_goal_delta": index * 0.4, "object_retained": 0.8},
        success=success,
        source="unit_test",
    )


def test_counterfactual_group_requires_identical_start_state_and_observation():
    group = [_candidate(0), _candidate(1)]
    proof = validate_counterfactual_group(group)
    assert proof["valid"] is True
    changed = CandidateRecord(
        **{
            **group[1].to_json(),
            "start_state_hash": "different",
        }
    )
    proof = validate_counterfactual_group([group[0], changed])
    assert proof["valid"] is False


def test_no_privileged_deployment_inputs_rejects_effect_leakage():
    assert_no_privileged_deployment_inputs({"observation": [0.0], "instruction": "pick"})
    with pytest.raises(ValueError, match="privileged"):
        assert_no_privileged_deployment_inputs({"observation": [0.0], "realized_effect": [1.0]})
    with pytest.raises(ValueError, match="privileged"):
        assert_no_privileged_deployment_inputs({"sim_state": [1.0], "candidate": [0.0]})


def test_phase_required_effect_and_compatibility_are_interpretable():
    required = required_effect_for_phase("transport")
    assert required["object_goal_delta"] > 0
    good = compatibility_score({"object_goal_delta": 1.0, "object_retained": 1.0}, "transport")
    bad = compatibility_score({"object_goal_delta": -1.0, "object_retained": -1.0}, "transport")
    assert good > bad


def test_headroom_metric_uses_default_vs_oracle_candidate():
    groups = [
        [_candidate(0, success=False, effect={"object_goal_delta": 0.0}), _candidate(1, success=True, effect={"object_goal_delta": 1.0, "object_retained": 1.0})],
        [_candidate(0, success=False, effect={"object_goal_delta": 0.0}), _candidate(1, success=True, effect={"object_goal_delta": 1.0, "object_retained": 1.0})],
        [_candidate(0, success=True, effect={"object_goal_delta": 0.8, "object_retained": 1.0}), _candidate(1, success=True, effect={"object_goal_delta": 1.0, "object_retained": 1.0})],
    ]
    metrics = candidate_headroom_metrics(groups)
    assert metrics["oracle_improvement_pp"] > 10
    assert metrics["default_failure_recoverable_rate"] >= 0.15
    assert metrics["passes_headroom_gate"] is True


def test_pairwise_ranking_pairs_compare_same_state_candidates():
    group = [_candidate(0, effect={"object_goal_delta": 0.0}), _candidate(1, effect={"object_goal_delta": 1.0, "object_retained": 1.0})]
    pairs = pairwise_ranking_pairs(group)
    assert pairs
    assert pairs[0][0] == 1
    assert pairs[0][1] == 0
