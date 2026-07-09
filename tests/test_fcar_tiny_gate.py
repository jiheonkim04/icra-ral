from tca_map.smolvla.fcar_tiny_gate import (
    build_feature_matrix,
    choose_final_decision,
    feature_schema,
    split_records_by_episode,
)


def _record(episode: int, task: int, frame: int = 0) -> dict:
    return {
        "dataset_local_index": frame,
        "episode_index": episode,
        "frame_index": frame,
        "episode_length": 20,
        "task_index": task,
        "task": f"task_{task}",
        "phase": "early",
        "normalized_phase": frame / 19,
        "state": [float(task)] * 8,
        "base_action": [0.0] * 7,
        "lora_action": [0.1] * 7,
        "mean_action": [0.5] * 7,
        "target_action": [0.0] * 7,
        "base_eval_loss": 0.01,
        "lora_eval_loss": 0.02,
        "base_action_l2": 0.0,
        "lora_action_l2": 0.1,
        "oracle_help_label": 0,
    }


def test_split_records_by_episode_is_disjoint_and_nonempty():
    selected_tasks = [
        {"task_index": 1, "episodes": [1, 4]},
        {"task_index": 2, "episodes": [2, 3]},
        {"task_index": 4, "episodes": [7, 9]},
        {"task_index": 5, "episodes": [8, 13]},
        {"task_index": 8, "episodes": [14, 15]},
    ]
    records = [_record(ep, task["task_index"], frame) for task in selected_tasks for ep in task["episodes"] for frame in range(2)]

    split = split_records_by_episode(records, selected_tasks)
    episodes = {name: {row["episode_index"] for row in rows} for name, rows in split.items()}

    assert split["train"]
    assert split["val"]
    assert split["test"]
    assert episodes["train"].isdisjoint(episodes["val"])
    assert episodes["train"].isdisjoint(episodes["test"])
    assert episodes["val"].isdisjoint(episodes["test"])


def test_feature_schema_excludes_target_and_oracle_labels():
    rows = [_record(1, 1), _record(2, 2)]
    matrix, mean, std = build_feature_matrix(rows)
    names = feature_schema()

    assert matrix.shape == (2, len(names))
    assert mean.shape[0] == len(names)
    assert std.shape[0] == len(names)
    assert not any("target" in name for name in names)
    assert not any("oracle" in name for name in names)


def test_decision_requires_hard_gain_and_static_moira_beats():
    assert (
        choose_final_decision(
            base_l2=0.10,
            lora_l2=0.12,
            mean_l2=1.0,
            moira_l2=0.11,
            static_l2=0.11,
            fcar_l2=0.094,
            train_l2=0.09,
            oracle_inputs_used=False,
        )
        == "GO_FCAR_SCALEUP"
    )
    assert (
        choose_final_decision(
            base_l2=0.10,
            lora_l2=0.12,
            mean_l2=1.0,
            moira_l2=0.095,
            static_l2=0.11,
            fcar_l2=0.096,
            train_l2=0.09,
            oracle_inputs_used=False,
        )
        == "FCAR_KILLED_BY_STATIC_BASELINE"
    )
    assert (
        choose_final_decision(
            base_l2=0.10,
            lora_l2=0.12,
            mean_l2=1.0,
            moira_l2=0.11,
            static_l2=0.11,
            fcar_l2=0.101,
            train_l2=0.09,
            oracle_inputs_used=False,
        )
        == "NO_FCAR_GAIN_OVER_BASE"
    )
