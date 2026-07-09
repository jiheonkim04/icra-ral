from tca_map.smolvla.official_libero_lora_seed_repro import (
    _aggregate_seed_summaries,
    _choose_decision,
    _parse_seeds,
)


def _metric(value):
    return {
        "action_l2_mean": value,
        "task_balanced_action_l2_mean": value,
        "translation_l2_mean": value,
        "rotation_l2_mean": value,
        "gripper_abs_mean": value,
    }


def _seed_summary(seed, *, base=0.10, lora=0.12, static=0.09, frame=0.07, task=0.085, moira=0.11):
    return {
        "seed": seed,
        "metrics": {
            "frozen_base": _metric(base),
            "rank4_lora": _metric(lora),
            "mean_action_prior": _metric(1.0),
            "frame_oracle": _metric(frame),
            "task_oracle": _metric(task),
            "moira_style_instruction_task_router": _metric(moira),
            "static_mix_val_selected": _metric(static),
        },
        "rank_order_realistic": [{"baseline": "static_mix_val_selected", "action_l2": static}],
        "win_counts_by_task": {"counts": {"static_mix_val_selected": 40}},
        "analysis": {
            "lora_beats_base": lora < base,
            "lora_beats_static": lora < static,
            "static_is_best_realistic": static < min(base, lora, 1.0, moira),
            "frame_oracle_headroom_after_static": static - frame,
            "task_oracle_headroom_over_base": base - task,
        },
    }


def test_parse_seeds_accepts_comma_list():
    assert _parse_seeds("11,22, 33") == [11, 22, 33]


def test_static_merge_decision_when_static_wins_all_reproduced_seeds():
    aggregate = _aggregate_seed_summaries([_seed_summary(11), _seed_summary(22), _seed_summary(33)])

    assert aggregate["answers"]["static_merge_remains_strongest_realistic_baseline"]
    assert _choose_decision(aggregate) == "STATIC_MERGE_ROBUST_BASELINE_READY"


def test_lora_instability_decision_when_lora_rank_changes_and_static_not_consistent():
    seeds = [
        _seed_summary(11, lora=0.09, static=0.11, moira=0.12),
        _seed_summary(22, lora=0.12, static=0.13, moira=0.125),
        _seed_summary(33, lora=0.10, static=0.13, moira=0.12),
    ]
    for seed in seeds:
        seed["rank_order_realistic"] = [{"baseline": "rank4_lora", "action_l2": seed["metrics"]["rank4_lora"]["action_l2_mean"]}]
        seed["win_counts_by_task"] = {"counts": {"rank4_lora": 20, "frozen_base": 20}}
        seed["analysis"]["static_is_best_realistic"] = False
    aggregate = _aggregate_seed_summaries(seeds)

    assert aggregate["answers"]["lora_instability_confirmed"]
    assert _choose_decision(aggregate) == "LORA_SEED_INSTABILITY_CONFIRMED"
