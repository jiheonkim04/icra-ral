import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import libero_ee_state_features as ee_features
from tca_map.smolvla_lora_baseline import live_feature_schema_fix as fix


def test_live_feature_fix_decision_set_is_exact():
    assert fix.FINAL_DECISIONS == {
        "READY_FOR_METHOD_AFTER_FEATURE_FIX",
        "FEATURE_FIXED_BUT_CONTROL_GAP_REMAINS",
        "FEATURE_CONVENTION_UNRESOLVED",
        "FEATURE_PATH_STILL_MISMATCHED",
        "ACTION_VALIDITY_RANGE_FAILURE",
        "TOO_HEAVY_LOCAL",
    }


def test_xyzw_axis_angle_preserves_hdf5_above_pi_branch():
    quat_xyzw = np.asarray([0.9998494017, -0.0094100408, -0.0126762808, -0.0072067264], dtype=np.float32)

    axis_angle = ee_features.quat_xyzw_to_hdf5_axis_angle(quat_xyzw)

    assert axis_angle[0] > np.pi
    assert np.allclose(axis_angle, np.asarray([3.156396, -0.029714, -0.040026], dtype=np.float32), atol=1e-3)


def test_live_feature_builder_does_not_use_quat_first3():
    obs = {
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.9998494, -0.00941004, -0.01267628, -0.00720673], dtype=np.float32),
    }

    feature, meta = ee_features.build_live_feature(obs, 0.25)

    assert feature.shape == (7,)
    assert meta["uses_quat_first3_fallback"] is False
    assert meta["orientation_convention"] == ee_features.ORIENTATION_CONVENTION
    assert not np.allclose(feature[3:6], obs["robot0_eef_quat"][:3])


def test_old_quat_first3_is_explicitly_marked_bad():
    obs = {
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
    }

    _feature, meta = ee_features.old_quat_first3_feature(obs, 0.0)

    assert meta["uses_quat_first3_fallback"] is True
    assert "legacy" in meta["source"]


def test_live_feature_fix_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(fix.RUN_GATE, raising=False)
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
        feature_match_threshold=0.15,
    )

    report, code = fix.build_report(args)

    assert code != 0
    assert report["decision"] == "TOO_HEAVY_LOCAL"
    assert fix.RUN_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["training_performed"] is False


def test_decide_feature_fixed_but_control_gap_remains():
    report = {
        "state3_live_hdf5_feature_alignment": {"feature_mismatch_fixed": True},
        "state4_teacher_forced_after_fix": {
            "aggregate": {"feature_fix_materially_improves_teacher_forced": True}
        },
        "state5_replay_after_fix": {
            "cases": [
                {
                    "results": {
                        "smolvla_7d_adapter_fixed_live": {
                            "action_validity": {
                                "clip_rate_step": 0.0,
                                "controller_valid_rate_proxy": 1.0,
                            }
                        }
                    }
                }
            ],
            "aggregate": {
                "expert": {"success_count": 1, "case_count": 1},
                "mean_action": {"success_count": 0, "progress_proxy_mean": 0.1},
                "ridge": {"success_count": 0, "progress_proxy_mean": 0.2},
                "smolvla_7d_adapter_fixed_live": {"success_count": 0, "progress_proxy_mean": 0.05},
            },
        },
    }

    decision, _next = fix._decide(report)

    assert decision == "FEATURE_FIXED_BUT_CONTROL_GAP_REMAINS"
