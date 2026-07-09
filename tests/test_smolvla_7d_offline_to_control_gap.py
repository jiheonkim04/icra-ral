import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import offline_to_control_gap as gap


def test_offline_to_control_decision_set_is_exact():
    assert gap.FINAL_DECISIONS == {
        "FEATURE_PATH_MISMATCH",
        "GRIPPER_PHASE_FAILURE",
        "TRANSLATION_DIRECTION_FAILURE",
        "OPEN_LOOP_ACTION_SEQUENCE_FAILURE",
        "CLOSED_LOOP_COMPOUNDING_FAILURE",
        "ACTION_VALIDITY_RANGE_FAILURE",
        "READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS",
        "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
    }


def test_offline_to_control_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(gap.RUN_GATE, raising=False)
    args = argparse.Namespace(
        data_root=str(tmp_path / "data"),
        libero_root=str(tmp_path / "LIBERO"),
        robosuite_root=str(tmp_path / "robosuite"),
        adapter_dir=str(tmp_path / "adapter"),
        exact_init_report_path=str(tmp_path / "exact.json"),
        output_dir=str(tmp_path / "runs"),
        report_path=str(tmp_path / "report.json"),
        max_tasks=2,
        train_demos_per_task=5,
        eval_demos_per_task=2,
        records_per_demo=8,
        max_replay_steps=8,
        post_signal_margin=0,
        camera_size=32,
        feature_mismatch_threshold=0.1,
    )

    report, code = gap.build_report(args)

    assert code != 0
    assert report["decision"] == "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP"
    assert gap.RUN_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["training_performed"] is False


def test_gripper_timing_reports_transition_error():
    expert = np.zeros((5, 7), dtype=np.float32)
    pred = np.zeros((5, 7), dtype=np.float32)
    expert[:, 6] = [-1, -1, 1, 1, 1]
    pred[:, 6] = [-1, -1, -1, 1, 1]

    timing = gap._gripper_timing(pred, expert)

    assert timing["expert_first_nonnegative_index"] == 2
    assert timing["pred_first_nonnegative_index"] == 3
    assert timing["first_nonnegative_timing_error"] == 1


def test_translation_cosine_detects_opposite_direction():
    expert = np.zeros((2, 7), dtype=np.float32)
    pred = np.zeros((2, 7), dtype=np.float32)
    expert[:, 0] = [1.0, 1.0]
    pred[:, 0] = [-1.0, -1.0]

    result = gap._translation_cosine(pred, expert)

    assert result["mean"] == -1.0
    assert result["negative_rate"] == 1.0


def test_decide_prioritizes_feature_path_mismatch():
    report = {
        "state1_feature_path_audit": {"feature_path_mismatch_found": True},
        "state2_teacher_forced_sequence": {
            "aggregate": {
                "smolvla_7d_adapter": {
                    "translation_cosine_negative_rate_mean": 1.0,
                    "gripper_error_mean": 2.0,
                }
            }
        },
        "state3_open_loop_action_replay": {"executed": True, "aggregate": {"smolvla_7d_adapter": {"success_count": 0}}},
    }

    decision, category, next_step = gap._decide(report)

    assert decision == "FEATURE_PATH_MISMATCH"
    assert category == "FEATURE_PATH_MISMATCH"
    assert "feature schema" in next_step
