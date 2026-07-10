from __future__ import annotations

import json
from pathlib import Path

from tca_map.smolvla.official_lora_drift_audit import (
    _compare_repeats,
    add_regenerated_artifact_reference_alignment,
    compare_artifact_alignment,
    choose_final_decision,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(*, split: str, lora: list[float]) -> dict:
    return {
        "split": split,
        "sample_id": f"{split}_task0_episode1_frame2",
        "episode_index": 1,
        "frame_index": 2,
        "task_index": 0,
        "dataset_local_index": 42,
        "episode_length": 10,
        "task": "dummy task",
        "phase": "mid",
        "target_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "base_action": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "mean_action": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "lora_action": lora,
    }


def test_artifact_alignment_proves_labels_base_and_flags_lora_difference(tmp_path: Path) -> None:
    old_path = tmp_path / "old_seed_11.json"
    new_path = tmp_path / "new_seed_11.json"
    _write(old_path, {"records": [_record(split="test", lora=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])]})
    _write(new_path, {"records": [_record(split="test", lora=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])]})

    result = compare_artifact_alignment(
        seeds=[11],
        old_pattern=str(tmp_path / "old_seed_{seed}.json"),
        regenerated_pattern=str(tmp_path / "new_seed_{seed}.json"),
    )

    row = result["per_seed"][0]
    assert result["protocol_drift_from_artifact_alignment"] is False
    assert row["aligned_for_split_labels_targets_and_base_predictions"] is True
    assert row["max_target_abs_diff"] == 0.0
    assert row["max_base_prediction_abs_diff"] == 0.0
    assert row["max_lora_action_l2_diff"] == 0.3


def test_repeat_comparison_accepts_exact_disk_eval_repeat() -> None:
    row = {
        "episode_index": 1,
        "frame_index": 2,
        "task_index": 0,
        "split": "test",
        "pred_preview": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
    }
    first = {
        "seed": 11,
        "pass_index": 1,
        "rank4_lora_action_l2_mean": 0.1,
        "static_mix_action_l2_mean": 0.08,
        "selected_alpha": 0.5,
        "_lora_rows": [row],
    }
    second = {
        "seed": 11,
        "pass_index": 2,
        "rank4_lora_action_l2_mean": 0.1,
        "static_mix_action_l2_mean": 0.08,
        "selected_alpha": 0.5,
        "_lora_rows": [dict(row)],
    }

    result = _compare_repeats(first, second)

    assert result["deterministic_within_tolerance"] is True
    assert result["max_per_action_l2_diff"] == 0.0
    assert result["rank4_action_l2_metric_diff"] == 0.0
    assert result["selected_alpha_identical"] is True


def test_final_decision_protocol_drift_over_canonicalization() -> None:
    report = {
        "checkpoint_integrity": {"all_complete_verified": True},
        "deterministic_evaluation": {"performed": True, "all_deterministic_within_tolerance": True},
        "artifact_alignment": {"protocol_drift_from_artifact_alignment": False},
        "source_protocol_diff": {
            "old_wrap_assignment": False,
            "regenerated_wrap_assignment": True,
        },
        "config_diff": [
            {"field": "prediction and postprocessing code", "classification": "DIFFERENT"},
        ],
        "historical_reproducibility_status": {"old_learned_policy_identity_reconstructable": False},
    }

    assert choose_final_decision(report) == "PROTOCOL_DRIFT_FOUND"


def test_reference_alignment_flags_fixed_seed_reeval_metric_mismatch() -> None:
    report = {
        "deterministic_evaluation": {
            "performed": True,
            "per_seed": [
                {
                    "seed": 22,
                    "passes": [
                        {
                            "rank4_lora_action_l2_mean": 0.086,
                            "static_mix_action_l2_mean": 0.078,
                            "selected_alpha": 0.5,
                        }
                    ],
                }
            ],
        }
    }
    regenerated = {
        "seed_summaries": [
            {
                "seed": 22,
                "metrics": {
                    "rank4_lora": {"action_l2_mean": 0.089},
                    "static_mix_val_selected": {"action_l2_mean": 0.081},
                },
                "static_selection": {"selected_weight": 0.25},
            }
        ]
    }

    add_regenerated_artifact_reference_alignment(report, regenerated)

    eval_report = report["deterministic_evaluation"]
    reference = eval_report["per_seed"][0]["regenerated_artifact_reference"]
    assert eval_report["regenerated_artifact_matches_fixed_seed_reeval"] is False
    assert eval_report["regenerated_artifact_mismatch_seeds"] == [22]
    assert reference["selected_alpha_identical_to_fixed_seed_reeval"] is False
    assert reference["matches_fixed_seed_reeval_within_tolerance"] is False
