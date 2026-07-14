from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.mtf_vla import (
    MTFConfig,
    _frameskip_proxy_records,
    audit_mtf_records,
    build_score_records,
    compute_mtf_scores,
    run_validation_search,
    validate_inference_fields,
)


def _prediction_record(split: str, task: int, episode: int, frame: int, action: list[float], base: list[float] | None = None) -> dict:
    index = task * 100000 + episode * 1000 + frame
    return {
        "sample_id": f"{split}_task{task}_episode{episode}_frame{frame}",
        "split": split,
        "task_index": task,
        "task": f"task {task}",
        "episode_index": episode,
        "frame_index": frame,
        "dataset_global_index": index,
        "eval_seed": 101,
        "normalized_phase": frame / 9.0,
        "target_action": action,
        "base_action": base or action,
    }


def test_validate_inference_fields_rejects_privileged_identity() -> None:
    with pytest.raises(ValueError, match="privileged MTF inference fields"):
        validate_inference_fields({"state": [0.0] * 8, "identity": 2026})


def test_compute_scores_marks_transition_frame_higher() -> None:
    records = []
    state_by_index = {}
    for frame in range(10):
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        if frame >= 5:
            action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        record = _prediction_record("train", 0, 0, frame, action)
        records.append(record)
        state_by_index[record["dataset_global_index"]] = [float(frame >= 5)] * 8

    scored = compute_mtf_scores(
        [
            {
                **record,
                "key": record["sample_id"],
                "frame_key": (record["split"], record["task_index"], record["episode_index"], record["frame_index"], record["eval_seed"]),
                "phase": record["normalized_phase"],
                "target_action": np.asarray(record["target_action"], dtype=np.float64),
                "base_action": np.asarray(record["base_action"], dtype=np.float64),
                "state": np.asarray(state_by_index[record["dataset_global_index"]], dtype=np.float64),
            }
            for record in records
        ],
        MTFConfig(min_scoreable_records=1, min_task_count=1),
    )

    assert scored[5]["score"] > scored[0]["score"]
    assert scored[5]["gripper_transition"] == 1.0


def test_sample_key_preserves_eval_seed_identity() -> None:
    first = _prediction_record("val", 0, 0, 0, [0.0] * 7)
    second = {**first, "eval_seed": 202}

    rows = build_score_records([first, second])

    assert rows[0]["key"] != rows[1]["key"]
    assert rows[0]["frame_key"] != rows[1]["frame_key"]


def test_build_score_records_uses_embedded_state() -> None:
    record = _prediction_record("train", 0, 0, 0, [0.0] * 7)
    record["state"] = [1.0] * 8

    rows = build_score_records([record])

    assert rows[0]["state"].shape == (8,)
    assert float(rows[0]["state"][0]) == 1.0


def test_audit_passes_on_noncollapsed_synthetic_records() -> None:
    records = []
    state_by_index = {}
    for split in ("train", "val", "test"):
        for task in range(3):
            episode = task + (0 if split == "train" else 10 if split == "val" else 20)
            for frame in range(20):
                action = [0.01 * frame, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                if frame >= 10:
                    action = [0.5 + 0.02 * frame, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                record = _prediction_record(split, task, episode, frame, action)
                records.append(record)
                state_by_index[record["dataset_global_index"]] = [0.01 * frame + float(frame >= 10)] * 8
    summary = {"summary": {"policy_summary": {"frozen_base": {"task_balanced_success_rate": 0.74}}}}
    report = audit_mtf_records(
        records,
        state_by_index=state_by_index,
        base_headroom_summary=summary,
        config=MTFConfig(
            min_scoreable_records=20,
            min_task_count=3,
            min_records_per_selected_task=10,
            min_high_low_score_gap=0.05,
            min_phase_bins_per_selected_task=2,
        ),
    )

    assert report["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert report["split_overlap"]["train_reserved"] == 0
    assert report["base_retention_target_manifest"]["reloadable"] is True


def test_validation_search_freezes_one_of_six_configs() -> None:
    records = []
    state_by_index = {}
    for split in ("train", "val", "test"):
        for task in range(3):
            episode = task + (0 if split == "train" else 10 if split == "val" else 20)
            for frame in range(100):
                action = [0.01 * frame, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                if frame % 20 in {8, 9, 10, 11, 12}:
                    sign = 1.0 if frame % 2 else -1.0
                    action = [sign * (2.0 + 0.02 * frame), 0.0, 0.0, 0.0, 0.0, 0.0, sign]
                base = [value * 0.5 for value in action]
                record = _prediction_record(split, task, episode, frame, action, base=base)
                record["normalized_phase"] = frame / 99.0
                record["lora_action"] = [value * 0.75 for value in action]
                records.append(record)
                state_by_index[record["dataset_global_index"]] = [0.01 * frame + float(frame >= 10)] * 8
    summary = {"summary": {"policy_summary": {"frozen_base": {"task_balanced_success_rate": 0.74}}}}

    report = run_validation_search(records, state_by_index=state_by_index, base_headroom_summary=summary)

    assert report["tried_config_count"] == 6
    assert report["final_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert report["selected_training_manifest"]["checkpoint_required_before_stage_a"] is True


def test_frameskip_proxy_selects_action_variation_from_full_train_group() -> None:
    records = [
        {"key": "mtf_high_due_gripper", "task_index": 0, "phase_bin": 0, "action_variation": 0.1},
        {"key": "frameskip_top_a", "task_index": 0, "phase_bin": 0, "action_variation": 5.0},
        {"key": "frameskip_top_b", "task_index": 0, "phase_bin": 0, "action_variation": 4.0},
        {"key": "low_motion", "task_index": 0, "phase_bin": 0, "action_variation": 0.0},
    ]

    selected = _frameskip_proxy_records(records, 0.5)

    assert [record["key"] for record in selected] == ["frameskip_top_a", "frameskip_top_b"]
