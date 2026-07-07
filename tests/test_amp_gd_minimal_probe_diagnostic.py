from argparse import Namespace

import numpy as np

from tca_map.amp_gd.minimal_probe_diagnostic import (
    choose_amp_probe,
    generate_trial,
    initial_belief,
    observe_after_probe,
    run_diagnostic,
    run_policy,
)


def test_informative_probe_reduces_belief_entropy():
    trial = generate_trial(11, 0)
    belief = initial_belief(trial)
    probe_end, meta = choose_amp_probe(trial, belief)

    assert probe_end is not None
    posterior, obs = observe_after_probe(trial, probe_end, belief)

    assert meta["probe_selected"] is True
    assert obs["cue_revealed"] is True
    assert obs["belief_entropy_after"] < obs["belief_entropy_before"]
    assert int(np.argmax(posterior)) == trial.intended_index


def test_amp_probe_policy_commits_to_intended_target_on_seeded_trial():
    trial = generate_trial(23, 4)
    record = run_policy(trial, "amp_gd_micro_probe", np.random.default_rng(123))

    assert record["probe_used"] is True
    assert record["target_disambiguation_correct"] is True
    assert record["wrong_target"] is False
    assert record["success"] is True


def test_state1_diagnostic_beats_simple_baselines():
    report = run_diagnostic(trial_count=30, seed_values=(11, 23, 37))
    metrics = report["metrics"]
    comparison = report["comparison"]

    assert report["trial_count"] == 30
    assert metrics["amp_gd_micro_probe"]["wrong_target_rate"] < metrics["no_probe_greedy"]["wrong_target_rate"]
    assert metrics["amp_gd_micro_probe"]["wrong_target_rate"] < metrics["random_probe"]["wrong_target_rate"]
    assert metrics["amp_gd_micro_probe"]["wrong_target_rate"] < metrics["safety_only_clipping"]["wrong_target_rate"]
    assert comparison["utility_cost_bounded"] is True


def test_build_report_rejects_too_few_trials():
    from tca_map.amp_gd.minimal_probe_diagnostic import build_report

    report = build_report(
        Namespace(
            trials=19,
            seeds="11,23,37",
            report_json="unused.json",
            report_md="unused.md",
            state1_md="unused_state.md",
        )
    )

    assert report["result"]["passed"] is False
    assert "at least 20" in report["result"]["blocked_reason"]

