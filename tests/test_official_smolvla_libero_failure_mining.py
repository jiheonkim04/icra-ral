from tca_map.smolvla.official_libero_failure_mining import choose_method_readiness_decision


def test_failure_mining_selects_control_stability_when_l2_improves_loss_worsens():
    report = {
        "aggregate_metrics": {
            "frozen_base": {"sample_count": 200, "action_l2_mean": 0.10, "eval_loss_mean": 0.01},
            "rank4_lora": {"sample_count": 200, "action_l2_mean": 0.08, "eval_loss_mean": 0.02},
            "mean_action_prior": {"sample_count": 200, "action_l2_mean": 0.30},
        },
        "comparison": {
            "lora_help_count": 150,
            "lora_hurt_count": 50,
            "lora_eval_loss_worse_count": 170,
        },
    }

    assert choose_method_readiness_decision(report) == "GO_METHOD_DESIGN_CONTROL_STABILITY"


def test_failure_mining_rejects_when_mean_prior_explains():
    report = {
        "aggregate_metrics": {
            "frozen_base": {"sample_count": 200, "action_l2_mean": 0.10, "eval_loss_mean": 0.01},
            "rank4_lora": {"sample_count": 200, "action_l2_mean": 0.08, "eval_loss_mean": 0.02},
            "mean_action_prior": {"sample_count": 200, "action_l2_mean": 0.07},
        },
        "comparison": {"lora_help_count": 150, "lora_hurt_count": 50, "lora_eval_loss_worse_count": 170},
    }

    assert choose_method_readiness_decision(report) == "NO_METHOD_WORTHY_GAP"


def test_failure_mining_requires_enough_samples():
    report = {
        "aggregate_metrics": {
            "frozen_base": {"sample_count": 20, "action_l2_mean": 0.10, "eval_loss_mean": 0.01},
            "rank4_lora": {"sample_count": 20, "action_l2_mean": 0.08, "eval_loss_mean": 0.02},
            "mean_action_prior": {"sample_count": 20, "action_l2_mean": 0.30},
        },
        "comparison": {"lora_help_count": 15, "lora_hurt_count": 5, "lora_eval_loss_worse_count": 17},
    }

    assert choose_method_readiness_decision(report) == "NEED_LONGER_OFFICIAL_BASELINE_REPRO"
