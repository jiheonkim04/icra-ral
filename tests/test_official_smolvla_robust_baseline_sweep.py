from tca_map.smolvla.official_libero_robust_baseline_sweep import _choose_decision, make_episode_folds


def _record(episode: int, task: int, frame: int) -> dict:
    return {
        "episode_index": episode,
        "frame_index": frame,
        "task_index": task,
        "base_action": [0.0] * 7,
        "lora_action": [0.1] * 7,
        "mean_action": [0.5] * 7,
        "target_action": [0.0] * 7,
        "base_action_l2": 0.0,
        "lora_action_l2": 0.1,
    }


def test_make_episode_folds_are_episode_disjoint():
    episodes = [1, 4, 2, 3, 7, 9, 8, 13, 14, 15]
    artifact = {"dataset": {"selected_episodes": episodes}}
    records = [_record(ep, idx // 2, frame) for idx, ep in enumerate(episodes) for frame in range(2)]

    folds = make_episode_folds(records, artifact, fold_count=5)

    assert len(folds) == 5
    for fold in folds:
        train = set(fold["train_episodes"])
        val = set(fold["val_episodes"])
        test = set(fold["test_episodes"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert len(fold["split_records"]["test"]) == 4


def test_decision_marks_split_instability_before_new_method_design():
    answers = {
        "is_standard_lora_robustly_better_than_frozen_base": False,
        "is_standard_lora_split_dependent": True,
        "is_method_worthy_frame_gap_left_after_static_merge": True,
        "does_frame_oracle_headroom_remain_large": True,
        "are_simple_baselines_enough": False,
    }
    wins = {"realistic": {"rank4_lora": 2, "static_mix_val_selected": 1}}

    assert _choose_decision(answers, wins, 5) == "METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD"


def test_decision_requires_minimum_fold_count():
    answers = {
        "is_standard_lora_robustly_better_than_frozen_base": True,
        "is_standard_lora_split_dependent": False,
        "is_method_worthy_frame_gap_left_after_static_merge": False,
        "does_frame_oracle_headroom_remain_large": False,
        "are_simple_baselines_enough": True,
    }
    wins = {"realistic": {"rank4_lora": 2}}

    assert _choose_decision(answers, wins, 2) == "NEED_LONGER_OFFICIAL_BASELINE_REPRO"
