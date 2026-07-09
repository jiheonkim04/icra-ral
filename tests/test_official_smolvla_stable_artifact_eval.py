from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _bootstrap_ci,
    _choose_decision,
    _manifest_samples,
    _stability_analysis,
)


def _mini_manifest():
    splits = {"train": [], "val": [], "test": []}
    for task in range(40):
        for split, episodes, frames in [
            ("train", [task * 5, task * 5 + 1], 15),
            ("val", [task * 5 + 2], 10),
            ("test", [task * 5 + 3, task * 5 + 4], 15),
        ]:
            for episode in episodes:
                for frame in range(frames):
                    splits[split].append(
                        {
                            "sample_id": f"{split}_{task}_{episode}_{frame}",
                            "split": split,
                            "task_index": task,
                            "task": f"task {task}",
                            "episode_index": episode,
                            "frame_index": frame,
                            "episode_length": 100,
                            "dataset_global_index": episode * 100 + frame,
                            "normalized_phase": frame / 99,
                        }
                    )
    return {
        "summary": {
            "frame_counts": {"train": 1200, "val": 400, "test": 1200},
            "leakage_checks": {
                "episode_disjoint_train_val": True,
                "episode_disjoint_train_test": True,
                "episode_disjoint_val_test": True,
            },
        },
        "splits": splits,
    }


def test_manifest_samples_preserve_fixed_split_counts_and_local_offsets():
    selected_episodes, split_samples, all_samples = _manifest_samples(_mini_manifest())

    assert len(selected_episodes) == 200
    assert len(split_samples["train"]) == 1200
    assert len(split_samples["val"]) == 400
    assert len(split_samples["test"]) == 1200
    assert len(all_samples) == 2800
    assert split_samples["train"][0]["dataset_local_index"] == 0
    assert split_samples["val"][0]["split"] == "val"


def test_bootstrap_ci_is_deterministic_for_group_means():
    rows = [
        {"task_index": 0, "episode_index": 0, "action_l2": 1.0},
        {"task_index": 0, "episode_index": 0, "action_l2": 3.0},
        {"task_index": 1, "episode_index": 1, "action_l2": 5.0},
        {"task_index": 1, "episode_index": 1, "action_l2": 7.0},
    ]

    first = _bootstrap_ci(rows, group_key="task_index", seed=7, iterations=20)
    second = _bootstrap_ci(rows, group_key="task_index", seed=7, iterations=20)

    assert first == second
    assert first["group_count"] == 2
    assert first["low"] <= first["mean"] <= first["high"]


def test_decision_prefers_frame_oracle_headroom_after_static_when_stable():
    analysis = {
        "are_metrics_stable_enough_for_method_design_later": True,
        "is_method_worthy_gap_left_after_simple_static_baselines": True,
        "does_task_oracle_remain_weak": True,
        "does_frame_oracle_headroom_remain_meaningful": True,
        "does_frame_oracle_remain_after_static": True,
        "is_rank4_lora_robustly_better_than_frozen_base": False,
        "is_rank4_lora_robustly_worse_than_frozen_base": False,
    }

    assert _choose_decision(analysis) == "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_STATIC"


def test_stability_analysis_marks_simple_baseline_gap_when_static_near_oracle():
    def metric(value):
        return {
            "action_l2_mean": value,
            "per_task": {str(i): {"action_l2_mean": value} for i in range(40)},
            "task_bootstrap_ci95_action_l2": {"low": value - 0.01, "high": value + 0.01},
        }

    evaluation = {
        "metrics": {
            "frozen_base": metric(0.10),
            "rank4_lora": metric(0.09),
            "mean_action_prior": metric(1.0),
            "frame_oracle": metric(0.087),
            "task_oracle": metric(0.099),
            "moira_style_instruction_task_router": metric(0.10),
            "static_mix_val_selected": metric(0.089),
        },
        "split_summary": {"test": {"frame_count": 1200}},
        "win_counts_by_task": {"counts": {"static_mix_val_selected": 40}},
    }

    analysis = _stability_analysis(evaluation)

    assert analysis["are_metrics_stable_enough_for_method_design_later"]
    assert not analysis["is_method_worthy_gap_left_after_simple_static_baselines"]
