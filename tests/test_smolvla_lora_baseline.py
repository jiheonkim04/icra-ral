import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import diagnostic


def test_decision_set_is_exact():
    assert diagnostic.FINAL_DECISIONS == {
        "READY_FOR_METHOD_ON_TOP_OF_SMOLVLA_LORA",
        "KILL_NO_REAL_LORA_LEARNING",
        "KILL_MEAN_BASELINE_DOMINATED",
        "KILL_FROZEN_BASELINE_DOMINATED",
        "TOO_HEAVY_LOCAL",
        "ENV_BLOCKED",
    }


def test_runner_requires_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LORA_BASELINE", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LORA_BASELINE_TRAINING", raising=False)
    args = argparse.Namespace(
        hdf5_path=str(tmp_path / "missing.hdf5"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        hf_home=str(tmp_path / "hf_home"),
        report_path=str(tmp_path / "report.json"),
        device="cuda",
        lora_rank=4,
        max_steps=60,
        max_train_demos=3,
        max_eval_demos=2,
        records_per_demo=3,
        learning_rate=1e-3,
    )

    report, code = diagnostic.build_report(args)

    assert code != 0
    assert report["decision"] == "ENV_BLOCKED"
    assert report["policy"]["training_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["summary"]["exact_next_step"]


def test_metrics_from_predictions_reports_required_fields():
    predictions = [
        np.asarray([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, -1.0], dtype=np.float32),
    ]
    experts = [
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    ]

    metrics = diagnostic._metrics_from_predictions(predictions, experts)

    assert metrics["sample_count"] == 2
    assert metrics["action_l2"] > 0
    assert metrics["translation_l2"] > 0
    assert metrics["rotation_l2"] > 0
    assert metrics["gripper_error"] == 0.0
    assert metrics["gripper_accuracy"] == 1.0
    assert len(metrics["per_dim_mae"]) == 7
    assert metrics["worst_action_dimensions"][0]["mae"] >= metrics["worst_action_dimensions"][-1]["mae"]


def test_demo_sort_key_orders_numeric_suffixes():
    names = ["demo_10", "demo_2", "demo_1"]

    assert sorted(names, key=diagnostic._demo_sort_key) == ["demo_1", "demo_2", "demo_10"]
