import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import diagnosis


def test_diagnosis_decision_set_is_exact():
    assert diagnosis.FINAL_DECISIONS == {
        "READY_FOR_REAL_METHOD_AFTER_BASELINE",
        "ACTION_INTERFACE_BUG",
        "DATA_TOO_SMALL_OR_LOW_VARIANCE",
        "LORA_CAPACITY_OR_TARGET_MODULE_BLOCKED",
        "KILL_SMOLVLA_LORA_BASELINE",
    }


def test_diagnosis_requires_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS_TRAINING", raising=False)
    args = argparse.Namespace(
        hdf5_path=str(tmp_path / "missing.hdf5"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        hf_home=str(tmp_path / "hf_home"),
        report_path=str(tmp_path / "report.json"),
        device="cuda",
        lora_rank=4,
        overfit_steps=10,
        capacity_steps=10,
        learning_rate=1e-3,
    )

    report, code = diagnosis.build_report(args)

    assert code != 0
    assert report["decision"] == "ACTION_INTERFACE_BUG"
    assert report["policy"]["training_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["summary"]["exact_next_step"]


def test_action_stats_reports_variance_blocks():
    actions = np.asarray(
        [
            [0.0, 1.0, 2.0, 0.0, 0.1, 0.2, -1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    stats = diagnosis._action_stats(actions)

    assert stats["count"] == 2
    assert stats["action_dim"] == 7
    assert len(stats["variance"]) == 7
    assert stats["translation_variance_mean"] > 0
    assert stats["rotation_variance_mean"] > 0
    assert stats["gripper_variance"] == 1.0


def test_sample_timesteps_are_unique_and_bounded():
    values = diagnosis._sample_timesteps(length=10, count=4, start=2, stop=9)

    assert values == sorted(set(values))
    assert min(values) >= 2
    assert max(values) < 9
