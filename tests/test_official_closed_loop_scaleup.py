import argparse

from tca_map.smolvla.official_closed_loop_scaleup import (
    RESET_SEEDS,
    annotate_failures,
    build_episode_manifest,
    choose_final_decision,
    _extract_single_env,
    select_evenly_spaced_task_ids,
    summarize_scaleup,
    wilson_interval,
)


def test_evenly_spaced_task_ids_for_ten_task_suite():
    assert select_evenly_spaced_task_ids(10, 5) == [0, 2, 4, 6, 8]


def test_episode_manifest_is_complete_for_scaleup_shape():
    task_manifest = {
        "tasks": [
            {"suite": f"suite_{suite}", "task_id": task, "instruction": "do it"}
            for suite in range(4)
            for task in [0, 2, 4, 6, 8]
        ]
    }
    args = argparse.Namespace(date="2026-07-11")

    manifest = build_episode_manifest(args, task_manifest)

    assert manifest["planned_episode_count"] == 4 * 5 * len(RESET_SEEDS) * 4
    assert len(manifest["episodes"]) == manifest["planned_episode_count"]
    assert len({episode["episode_id"] for episode in manifest["episodes"]}) == manifest["planned_episode_count"]


def test_wilson_interval_bounds_are_valid():
    low, high = wilson_interval(8, 10)

    assert 0.0 <= low <= high <= 1.0


def test_extract_single_env_from_official_nested_make_env_shape():
    env = object()

    assert _extract_single_env({"suite": {2: env}}, "suite", 2) is env


def test_paired_summary_counts_win_loss_tie():
    rows = [
        {"policy": "frozen_base", "suite": "s", "task_id": 0, "reset_seed": 1, "success": False, "episode_length": 1},
        {
            "policy": "rank4_lora_seed_11",
            "suite": "s",
            "task_id": 0,
            "reset_seed": 1,
            "success": True,
            "episode_length": 1,
        },
        {"policy": "frozen_base", "suite": "s", "task_id": 0, "reset_seed": 2, "success": True, "episode_length": 1},
        {
            "policy": "rank4_lora_seed_11",
            "suite": "s",
            "task_id": 0,
            "reset_seed": 2,
            "success": False,
            "episode_length": 1,
        },
        {"policy": "frozen_base", "suite": "s", "task_id": 0, "reset_seed": 3, "success": True, "episode_length": 1},
        {
            "policy": "rank4_lora_seed_11",
            "suite": "s",
            "task_id": 0,
            "reset_seed": 3,
            "success": True,
            "episode_length": 1,
        },
    ]

    summary = summarize_scaleup({"episodes": rows})

    assert summary["paired_vs_frozen_base"]["reset_level"]["rank4_lora_seed_11"] == {
        "loss": 1,
        "tie": 1,
        "win": 1,
    }


def test_failure_annotation_is_conservative_without_visual_evidence():
    row = {
        "episode_id": "frozen_base|suite|task_0|seed_1",
        "policy": "frozen_base",
        "suite": "suite",
        "task_id": 0,
        "reset_seed": 1,
        "success": False,
        "failure_status": "unsuccessful",
        "action_validity": {"finite": True, "shape_ok": True},
        "video_path": None,
    }

    annotations = annotate_failures({"episodes": [row]})

    assert annotations["annotations"][0]["dominant_failure_phase"] == "ambiguous_or_unclassified"
    assert annotations["bounded_review_queue"]


def test_infrastructure_failure_decision_has_priority():
    decision = choose_final_decision(
        {"planned_episode_count": 1, "completed_episode_count": 0, "infrastructure_failure_count": 1},
        {"training_seed_variance": {}, "offline_online": {}},
        {"failure_count": 1, "category_counts": {"environment_or_infrastructure_failure": 1}},
    )

    assert decision == "ROLLOUT_INFRASTRUCTURE_FAILURE"
