from __future__ import annotations

import math

import pytest

from scripts import run_epoch10b_stage0 as stage0


def test_windows_checkpoint_path_maps_to_wsl() -> None:
    assert (
        stage0._windows_to_wsl(r"C:\assets\checkpoints\panel\seed_101\step_0030")
        == "/mnt/c/assets/checkpoints/panel/seed_101/step_0030"
    )


def test_permutation_maps_are_fixed_point_free_and_unpaired_stays_in_task() -> None:
    states = [
        {"suite": suite, "state_id": f"{suite}|state_{index}"}
        for suite in ("a", "b")
        for index in range(4)
    ]
    unpaired_a, shuffled_a = stage0._permutation_maps(states)
    unpaired_b, shuffled_b = stage0._permutation_maps(list(reversed(states)))
    assert unpaired_a == unpaired_b
    assert shuffled_a == shuffled_b
    assert all(target != source for target, source in unpaired_a.items())
    assert all(target != source for target, source in shuffled_a.items())
    assert all(target.split("|")[0] == source.split("|")[0] for target, source in unpaired_a.items())


def test_concordance_excludes_same_lineage_and_performance_ties() -> None:
    checkpoints = ["a30", "a100", "b30", "b100"]
    lineages = {"a30": "a", "a100": "a", "b30": "b", "b100": "b"}
    performance = {
        ("a30", "task"): 0.2,
        ("a100", "task"): 0.2,
        ("b30", "task"): 0.8,
        ("b100", "task"): 0.8,
    }
    metric = {
        ("a30", "task"): 2.0,
        ("a100", "task"): 2.1,
        ("b30", "task"): 1.0,
        ("b100", "task"): 1.1,
    }
    value, report = stage0._concordance(metric, performance, lineages, ["task"], checkpoints)
    assert value == 1.0
    assert report["task"]["informative_cross_lineage_pairs"] == 4
    assert report["task"]["performance_ties_excluded"] == 0


def test_concordance_predictor_tie_scores_half() -> None:
    performance = {("a", "task"): 0.0, ("b", "task"): 1.0}
    metric = {("a", "task"): 1.0, ("b", "task"): 1.0}
    value, _ = stage0._concordance(metric, performance, {"a": "a", "b": "b"}, ["task"], ["a", "b"])
    assert value == 0.5


def test_extract_successes_requires_exact_frozen_panel() -> None:
    successes = [index % 2 == 0 for index in range(15)]
    metrics = {
        "per_task": [
            {"task_group": "libero_goal", "task_id": 0, "metrics": {"successes": successes}}
        ]
    }
    assert stage0._extract_successes(metrics, "libero_goal") == successes
    metrics["per_task"][0]["metrics"]["successes"] = successes[:-1]
    with pytest.raises(stage0.Stage0Error, match="ROLLOUT_EPISODE_CARDINALITY"):
        stage0._extract_successes(metrics, "libero_goal")


def test_weighted_concordance_keeps_nested_stages_as_clustered_repeats() -> None:
    # Two lineage occurrences, each with two stages.  All four cross-lineage
    # comparisons are concordant; the two within-lineage comparisons are not
    # eligible.
    scores = stage0.np.asarray([[2.0, 2.2, 1.0, 1.2]])
    performance = stage0.np.asarray([[0.2, 0.3, 0.8, 0.7]])
    assert stage0._weighted_concordance(scores, performance, [0, 0, 1, 1]) == 1.0


def test_centered_rank_expected_direction_is_negative_metric() -> None:
    tasks = ["x", "y"]
    checkpoints = ["a", "b", "c"]
    metric = {}
    performance = {}
    for task in tasks:
        for index, checkpoint in enumerate(checkpoints):
            metric[(checkpoint, task)] = float(3 - index)
            performance[(checkpoint, task)] = float(index)
    ranks = stage0._centered_ranks(metric, performance, tasks, checkpoints)
    assert math.isclose(ranks["spearman"], 1.0)
    assert math.isclose(ranks["kendall_tau_b"], 1.0)


def test_branch_materialization_requires_valid_finite_primary_score() -> None:
    row = {"valid": True, "bounded_recovery_cost_by_horizon": {"4": 0.25}}
    assert stage0._branch_materialized(row)
    row["valid"] = False
    assert not stage0._branch_materialized(row)
    row["valid"] = True
    row["bounded_recovery_cost_by_horizon"]["4"] = float("nan")
    assert not stage0._branch_materialized(row)


def test_step_caps_match_frozen_official_lerobot_defaults() -> None:
    assert stage0.SUITE_STEP_CAPS == {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    assert 16 * 15 * sum(stage0.SUITE_STEP_CAPS.values()) == 331_200
