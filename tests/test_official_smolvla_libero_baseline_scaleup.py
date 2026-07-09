from tca_map.smolvla.official_libero_baseline_scaleup import choose_final_decision


def test_official_scaleup_decision_detects_cpu_fallback():
    assert choose_final_decision({"cpu_fallback_occurred": True}) == "CPU_FALLBACK_BUG"


def test_official_scaleup_decision_requests_longer_repro_when_lora_worse():
    report = {
        "training": {
            "training_completed": True,
            "completed_steps": 100,
            "loss_decrease_fraction": 0.5,
        },
        "evaluation": {
            "frozen_base": {"action_l2_mean": 0.1},
            "lora_after_training": {"action_l2_mean": 0.2, "eval_loss_mean": 0.01},
        },
        "runtime": {"cuda": {"max_allocated_mb": 1000}, "total_elapsed_sec": 100},
    }
    assert choose_final_decision(report) == "READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO"


def test_official_scaleup_decision_allows_method_design_when_green():
    report = {
        "training": {
            "training_completed": True,
            "completed_steps": 100,
            "loss_decrease_fraction": 0.5,
        },
        "evaluation": {
            "frozen_base": {"action_l2_mean": 0.1},
            "lora_after_training": {"action_l2_mean": 0.099, "eval_loss_mean": 0.01},
        },
        "runtime": {"cuda": {"max_allocated_mb": 1000}, "total_elapsed_sec": 100},
    }
    assert choose_final_decision(report) == "READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA"
