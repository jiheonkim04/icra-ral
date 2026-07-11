import json
from pathlib import Path

from tca_map.smolvla.closed_loop_failure_visual_review import (
    MAX_REVIEW_EPISODES,
    SELECTED_REVIEW_EPISODES,
    _episode_id,
)


def test_visual_review_episode_set_is_bounded_and_unique():
    assert MAX_REVIEW_EPISODES == 24
    assert len(SELECTED_REVIEW_EPISODES) == MAX_REVIEW_EPISODES

    episode_ids = [
        _episode_id(item["policy"], item["suite"], int(item["task_id"]), int(item["reset_seed"]))
        for item in SELECTED_REVIEW_EPISODES
    ]

    assert len(set(episode_ids)) == MAX_REVIEW_EPISODES


def test_visual_review_episode_set_is_restricted_to_predeclared_hard_slices():
    allowed_policies = {"frozen_base", "rank4_lora_seed_11", "rank4_lora_seed_22", "rank4_lora_seed_33"}
    allowed_tasks = {("libero_spatial", 4), ("libero_10", 4)}

    assert {item["policy"] for item in SELECTED_REVIEW_EPISODES} == allowed_policies
    assert {(item["suite"], int(item["task_id"])) for item in SELECTED_REVIEW_EPISODES} == allowed_tasks


def test_visual_annotation_report_uses_allowed_failure_phases():
    report = json.loads(Path("reports/closed_loop_failure_visual_annotations.json").read_text(encoding="utf-8"))
    allowed = set(report["allowed_phase_labels"])

    assert report["selected_episode_count"] == MAX_REVIEW_EPISODES
    assert report["videos_reviewed"] == MAX_REVIEW_EPISODES
    assert report["rerun_errors"] == 0
    assert report["cross_task_common_mechanism_supported"] is False

    for annotation in report["annotations"]:
        phase = annotation["failure_phase"]
        assert phase is None or phase in allowed
        assert annotation["same_suite_task_policy_reset_as_scaleup"] is True
