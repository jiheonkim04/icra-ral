import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import action_range_fix as range_fix


def test_action_range_fix_decision_set_is_exact():
    assert range_fix.FINAL_DECISIONS == {
        "READY_FOR_METHOD_AFTER_RANGE_FIX",
        "RANGE_FIXED_BUT_CONTROL_GAP_REMAINS",
        "GRIPPER_CONVENTION_FAILURE",
        "NORMALIZATION_STILL_INVALID",
        "CLIP_ONLY_BASELINE_DOMINATES",
        "TOO_HEAVY_LOCAL",
    }


def test_gripper_distribution_identifies_signed_binary_labels():
    actions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    dist = range_fix._gripper_distribution(actions)

    assert dist["binary_signed"] is True
    assert dist["sign_based"] is True
    assert dist["negative"] == 2
    assert dist["positive"] == 1


def test_validity_reports_gripper_as_dominant_clip_dim():
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3],
        ],
        dtype=np.float32,
    )

    validity = range_fix._validity(actions)

    assert validity["dominant_clip_dim"] == 6
    assert validity["gripper_clip_rate"] == 1.0
    assert validity["controller_valid_rate_proxy"] == 0.0


def test_affine_calibrator_uses_train_mapping_and_clips():
    train_features = np.arange(21, dtype=np.float32).reshape(3, 7)
    train_labels = np.ones((3, 7), dtype=np.float32) * 0.5

    def base_predict(features):
        return np.ones((np.asarray(features).shape[0], 7), dtype=np.float32) * 2.0

    predictor = range_fix._fit_affine_calibrator(train_features, train_labels, base_predict)
    pred = predictor.predict(train_features)

    assert predictor.report["uses_eval_labels"] is False
    assert predictor.report["not_method_success"] is True
    assert pred.shape == (3, 7)
    assert np.all(pred <= 1.0)
    assert np.all(pred >= -1.0)


def test_decide_clip_only_dominates():
    report = {
        "state1_action_range_and_clipping_audit": {
            "gripper_convention": {"dominant_clip_dimension_is_gripper": True}
        },
        "state3_offline_after_range_fix": {
            "baselines": {
                "previous_unfixed_adapter": {
                    "eval_metrics": {"action_l2": 1.0},
                    "action_validity": {"clip_rate_step": 0.5, "controller_valid_rate_proxy": 0.5},
                },
                "range_fixed_smolvla_7d_adapter": {
                    "eval_metrics": {"action_l2": 1.1},
                    "action_validity": {"clip_rate_step": 0.0, "controller_valid_rate_proxy": 1.0},
                    "gripper_accuracy": {"sign_accuracy": 0.9},
                },
            }
        },
        "state4_replay_after_range_fix": {
            "aggregate": {
                "expert": {"success_count": 1, "case_count": 1},
                "mean_action": {"success_count": 0, "progress_proxy_mean": 0.0},
                "ridge": {"success_count": 0, "progress_proxy_mean": 0.0},
                "small_mlp": {"success_count": 0, "progress_proxy_mean": 0.0},
                "previous_unfixed_adapter": {"success_count": 0, "progress_proxy_mean": -0.1},
                "previous_unfixed_adapter_clip_only": {"success_count": 0, "progress_proxy_mean": 0.2},
                "range_fixed_smolvla_7d_adapter": {"success_count": 0, "progress_proxy_mean": 0.1},
            }
        },
    }

    decision, _next = range_fix._decide(report)

    assert decision == "CLIP_ONLY_BASELINE_DOMINATES"


def test_action_range_fix_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(range_fix.RUN_GATE, raising=False)
    args = argparse.Namespace(
        data_root=str(tmp_path / "data"),
        libero_root=str(tmp_path / "LIBERO"),
        robosuite_root=str(tmp_path / "robosuite"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        adapter_dir=str(tmp_path / "adapters"),
        exact_init_report_path=str(tmp_path / "exact.json"),
        output_dir=str(tmp_path / "runs"),
        report_path=str(tmp_path / "report.json"),
        max_tasks=2,
        train_demos_per_task=5,
        eval_demos_per_task=2,
        records_per_demo=8,
        adapter_steps=1,
        adapter_hidden_dim=8,
        mlp_steps=1,
        mlp_hidden_dim=8,
        learning_rate=1e-3,
        lora_learning_rate=1e-3,
        max_replay_steps=8,
        post_signal_margin=0,
        camera_size=32,
    )

    report, code = range_fix.build_report(args)

    assert code != 0
    assert report["decision"] == "TOO_HEAVY_LOCAL"
    assert range_fix.RUN_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["uses_hard_coded_gripper_fill"] is False
