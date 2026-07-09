import argparse

import numpy as np
import pytest

from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix


def test_interface_fix_decision_set_is_exact():
    assert fix.FINAL_DECISIONS == {
        "READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX",
        "INTERFACE_FIXED_BUT_LORA_WEAK",
        "ACTION_INTERFACE_STILL_BROKEN",
        "DATA_LOW_VARIANCE_OR_SPLIT_BAD",
        "TOO_HEAVY_LOCAL",
    }


def test_interface_fix_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_SMOLVLA_LIBERO_7D_INTERFACE_FIX", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LIBERO_7D_INTERFACE_TRAINING", raising=False)
    args = argparse.Namespace(
        hdf5_path=str(tmp_path / "missing.hdf5"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        report_path=str(tmp_path / "report.json"),
        one_sample_steps=1,
        one_demo_steps=1,
        adapter_steps=1,
        baseline_mlp_steps=1,
        adapter_hidden_dim=8,
        learning_rate=1e-2,
    )

    report, code = fix.build_report(args)

    assert code != 0
    assert report["decision"] == "ACTION_INTERFACE_STILL_BROKEN"
    assert report["policy"]["training_performed"] is False
    assert "ALLOW_SMOLVLA_LIBERO_7D_INTERFACE_FIX" in report["summary"]["exact_next_step"]


def test_libero_7d_normalizer_uses_seven_dimensions_and_round_trips():
    labels = np.asarray(
        [
            [0.0, 1.0, 2.0, 0.1, 0.2, 0.3, -1.0],
            [1.0, 2.0, 3.0, 0.2, 0.3, 0.4, 1.0],
        ],
        dtype=np.float32,
    )

    normalizer = fix.Libero7DNormalizer.fit(labels)
    normalized = normalizer.normalize(labels)
    restored = normalizer.unnormalize(normalized)

    assert normalizer.mean.shape == (7,)
    assert normalizer.std.shape == (7,)
    assert normalizer.report()["uses_so100_stats"] is False
    assert normalizer.report()["uses_eval_labels"] is False
    np.testing.assert_allclose(restored, labels, atol=1e-6)


def test_shape_guards_reject_six_dimensional_labels():
    bad = np.zeros((2, 6), dtype=np.float32)

    with pytest.raises(ValueError, match=r"\[N, 7\]"):
        fix.Libero7DNormalizer.fit(bad)


def test_action_stats_marks_libero_schema_blocks():
    actions = np.asarray(
        [
            [0.0, 1.0, 2.0, 0.0, 0.1, 0.2, -1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    stats = fix._action_stats(actions)

    assert stats["action_dim"] == 7
    assert stats["translation_dims"] == [0, 1, 2]
    assert stats["rotation_dims"] == [3, 4, 5]
    assert stats["gripper_dim"] == 6
    assert stats["gripper_values"] == [-1.0, 1.0]
