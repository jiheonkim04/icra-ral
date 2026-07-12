import numpy as np
import pytest

from tca_map.smolvla.ocfn_vla import (
    OCFNConfig,
    assert_no_privileged_inference_fields,
    build_all_selections,
    full_equals_baseline,
    make_noise_bank,
    noise_sha256,
    stage_a_decision,
    stage_b_decision,
    zero_noise,
)


def test_noise_bank_is_deterministic_and_nonzero():
    config = OCFNConfig(noise_count=4, chunk_size=3, max_action_dim=2, seed_base=11)
    first = make_noise_bank(config)
    second = make_noise_bank(config)
    assert set(first) == {0, 1, 2, 3}
    assert first[0].shape == (1, 3, 2)
    assert noise_sha256(first[2]) == noise_sha256(second[2])
    assert not np.allclose(first[0], first[1])
    assert np.allclose(zero_noise(config), 0.0)


def test_privileged_inference_fields_rejected():
    assert_no_privileged_inference_fields(["task_key", "suite", "instruction"])
    with pytest.raises(ValueError, match="success"):
        assert_no_privileged_inference_fields(["task_key", "success"])


def test_task_global_and_shuffled_selection_are_separate():
    rows = [
        {"task_key": "task_a", "noise_id": 0, "success": False, "episode_steps": 10, "reward_sum": 0.0},
        {"task_key": "task_a", "noise_id": 1, "success": True, "episode_steps": 5, "reward_sum": 1.0},
        {"task_key": "task_b", "noise_id": 0, "success": False, "episode_steps": 10, "reward_sum": 0.0},
        {"task_key": "task_b", "noise_id": 2, "success": True, "episode_steps": 4, "reward_sum": 1.0},
    ]
    selections = build_all_selections(rows, ["task_a", "task_b"], OCFNConfig(noise_count=4, task_shuffle_seed=7))
    assert selections["ocfn_full"]["task_a"].noise_id == 1
    assert selections["ocfn_full"]["task_b"].noise_id == 2
    assert selections["global_success_noise_prior"]["task_a"].noise_id in {1, 2}
    assert not full_equals_baseline(selections, "global_success_noise_prior", ["task_a", "task_b"])
    assert selections["task_shuffled_noise_prior"]["task_a"].source == "task_shuffled_noise_prior"


def test_stage_a_trivial_equivalence_kill():
    summary = {
        "frozen_smolvla": {"task_balanced_success_rate": 0.4, "successes": 4},
        "zero_noise_smolvla": {"task_balanced_success_rate": 0.4, "successes": 4},
        "global_success_noise_prior": {"task_balanced_success_rate": 0.4, "successes": 4},
        "task_shuffled_noise_prior": {"task_balanced_success_rate": 0.4, "successes": 4},
        "ocfn_full": {"task_balanced_success_rate": 0.4, "successes": 4},
    }
    assert (
        stage_a_decision(
            summary,
            full_global_equivalent=True,
            full_shuffled_equivalent=True,
            full_action_delta_vs_global=0.0,
            full_action_delta_vs_shuffled=0.0,
        )
        == "STAGE_A_PERMANENT_KILL_TRIVIAL_EQUIVALENCE"
    )


def test_stage_a_positive_to_stage_b():
    summary = {
        "frozen_smolvla": {"task_balanced_success_rate": 0.3, "successes": 3},
        "zero_noise_smolvla": {"task_balanced_success_rate": 0.2, "successes": 2},
        "global_success_noise_prior": {"task_balanced_success_rate": 0.3, "successes": 3},
        "task_shuffled_noise_prior": {"task_balanced_success_rate": 0.2, "successes": 2},
        "ocfn_full": {"task_balanced_success_rate": 0.5, "successes": 5},
    }
    assert (
        stage_a_decision(
            summary,
            full_global_equivalent=False,
            full_shuffled_equivalent=False,
            full_action_delta_vs_global=0.3,
            full_action_delta_vs_shuffled=0.2,
        )
        == "STAGE_A_POSITIVE_TO_STAGE_B"
    )


def test_stage_b_prototype_go_requires_strong_full_result():
    summary = {
        "frozen_smolvla": {"task_balanced_success_rate": 0.35},
        "zero_noise_smolvla": {"task_balanced_success_rate": 0.3},
        "global_success_noise_prior": {"task_balanced_success_rate": 0.35},
        "task_shuffled_noise_prior": {"task_balanced_success_rate": 0.25},
        "ocfn_full": {"task_balanced_success_rate": 0.5},
    }
    paired = {
        "frozen_smolvla": {"paired_bootstrap_ci": [0.02, 0.28], "failure_rate_reduction": 0.23},
        "zero_noise_smolvla": {"paired_bootstrap_ci": [0.05, 0.31], "failure_rate_reduction": 0.29},
        "global_success_noise_prior": {"paired_bootstrap_ci": [0.02, 0.28], "failure_rate_reduction": 0.23},
        "task_shuffled_noise_prior": {"paired_bootstrap_ci": [0.07, 0.35], "failure_rate_reduction": 0.33},
    }
    assert (
        stage_b_decision(
            summary,
            paired,
            mechanism_active=True,
            complete=True,
            exception_count=0,
            pairs_per_policy=40,
        )
        == "STAGE_B_PROTOTYPE_GO"
    )


def test_stage_b_ablation_explains_method_kill():
    summary = {
        "frozen_smolvla": {"task_balanced_success_rate": 0.35},
        "zero_noise_smolvla": {"task_balanced_success_rate": 0.4},
        "global_success_noise_prior": {"task_balanced_success_rate": 0.4},
        "task_shuffled_noise_prior": {"task_balanced_success_rate": 0.4},
        "ocfn_full": {"task_balanced_success_rate": 0.35},
    }
    paired = {
        "zero_noise_smolvla": {"paired_bootstrap_ci": [-0.18, 0.08], "failure_rate_reduction": -0.08},
        "task_shuffled_noise_prior": {"paired_bootstrap_ci": [-0.18, 0.08], "failure_rate_reduction": -0.08},
    }
    assert (
        stage_b_decision(
            summary,
            paired,
            mechanism_active=True,
            complete=True,
            exception_count=0,
            pairs_per_policy=40,
        )
        == "STAGE_B_PERMANENT_KILL_ABLATION_EXPLAINS_METHOD"
    )
