from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.mtf_vla import PROPOSAL_HASH
from tca_map.smolvla.mtf_vla_training import (
    MTFTrainArgs,
    MTFTrainingError,
    _all_stage_a_variants_verified,
    _override_current_action,
    build_training_jobs,
)


def _split_row(split: str, task: int, episode: int, frame: int, index: int) -> dict:
    return {
        "sample_id": f"{split}_task{task}_episode{episode}_frame{frame}",
        "split": split,
        "task": f"task {task}",
        "task_index": task,
        "episode_index": episode,
        "episode_length": 100,
        "frame_index": frame,
        "dataset_global_index": index,
        "normalized_phase": frame / 99.0,
    }


def _stable_record(row: dict) -> dict:
    return {
        **row,
        "target_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
        "base_action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    }


def _fixtures() -> tuple[dict, dict, dict]:
    train0 = _split_row("train", 0, 10, 1, 1001)
    train1 = _split_row("train", 0, 10, 2, 1002)
    train2 = _split_row("train", 1, 20, 3, 2003)
    val = _split_row("val", 0, 30, 4, 3004)
    test = _split_row("test", 0, 40, 5, 4005)
    split_manifest = {"splits": {"train": [train0, train1, train2], "val": [val], "test": [test]}}
    stable_artifact = {"records": [_stable_record(row) for row in [train0, train1, train2, val, test]]}
    selected_manifest = {
        "method": "MTF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "config_id": "mtf_r20_ret100",
        "retained_high_frame_ratio": 0.2,
        "retention_coefficient": 1.0,
        "confirmatory_test_identities_used": False,
        "variants": {
            "mtf_full": {
                "high_milestone_frames": [{**train0, "score": 0.9}, {**train1, "score": 0.8}],
                "base_retention_frames": [{**train2, "score": 0.0}],
                "retention_coefficient": 1.0,
            },
            "mtf_no_retention_ablation": {
                "high_milestone_frames": [{**train0, "score": 0.9}, {**train1, "score": 0.8}],
                "base_retention_frames": [],
                "retention_coefficient": 0.0,
            },
            "frameskip_proxy_lora": {"selected_frames": [{**train1, "score": 0.8}]},
            "uniform_retained_ratio_lora": {"selected_frames": [{**train2, "score": 0.0}]},
        },
    }
    return selected_manifest, split_manifest, stable_artifact


def test_build_training_jobs_preserves_mtf_retention_contract() -> None:
    selected_manifest, split_manifest, stable_artifact = _fixtures()

    plan = build_training_jobs(
        selected_manifest=selected_manifest,
        split_manifest=split_manifest,
        stable_artifact=stable_artifact,
        train_args=MTFTrainArgs(steps=12, seed=7),
    )

    assert plan["final_decision"] == "MTF_ADAPTER_TRAINING_PLAN_READY"
    assert plan["confirmatory_test_identities_used"] is False
    full = next(job for job in plan["jobs"] if job["variant"] == "mtf_full")
    assert full["event_count"] == 3
    assert full["demo_action_chunk_event_count"] == 2
    assert full["base_current_action_retention_event_count"] == 1
    retention_events = [event for event in full["events"] if event["objective"] == "base_current_action_retention"]
    assert retention_events[0]["base_action"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    assert "current 7D action only" in retention_events[0]["retention_target_scope"]
    assert retention_events[0]["phase"] == "early"
    assert plan["stage_a_allowed"] is False


def test_build_training_jobs_blocks_non_train_selected_frame() -> None:
    selected_manifest, split_manifest, stable_artifact = _fixtures()
    selected_manifest["variants"]["uniform_retained_ratio_lora"]["selected_frames"] = [
        {**split_manifest["splits"]["test"][0], "score": 0.5}
    ]

    plan = build_training_jobs(
        selected_manifest=selected_manifest,
        split_manifest=split_manifest,
        stable_artifact=stable_artifact,
        train_args=MTFTrainArgs(steps=12, seed=7, variants=("uniform_retained_ratio_lora",)),
    )

    assert plan["final_decision"] == "MTF_ADAPTER_TRAINING_PLAN_BLOCKED"
    assert any("non-train frame selected" in reason for reason in plan["hard_stop_reasons"])


def test_override_current_action_replaces_only_first_action_step() -> None:
    raw = {"action": np.zeros((3, 7), dtype=np.float32)}

    updated = _override_current_action(raw, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0])

    np.testing.assert_allclose(updated["action"][0], np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(updated["action"][1:], np.zeros((2, 7), dtype=np.float32))
    np.testing.assert_allclose(raw["action"], np.zeros((3, 7), dtype=np.float32))


def test_build_training_jobs_rejects_extra_variant() -> None:
    selected_manifest, split_manifest, stable_artifact = _fixtures()

    with pytest.raises(MTFTrainingError, match="invalid requested variants"):
        build_training_jobs(
            selected_manifest=selected_manifest,
            split_manifest=split_manifest,
            stable_artifact=stable_artifact,
            train_args=MTFTrainArgs(variants=("mtf_full", "new_unfrozen_variant")),
        )


def test_stage_a_ready_requires_all_four_verified_trainable_policies() -> None:
    one_verified = [{"variant": "mtf_no_retention_ablation", "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED"}]
    all_verified = [
        {"variant": "mtf_full", "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED"},
        {"variant": "mtf_no_retention_ablation", "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED"},
        {"variant": "frameskip_proxy_lora", "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED"},
        {"variant": "uniform_retained_ratio_lora", "final_decision": "MTF_ADAPTER_CHECKPOINT_VERIFIED"},
    ]

    assert _all_stage_a_variants_verified(one_verified) is False
    assert _all_stage_a_variants_verified(all_verified) is True
