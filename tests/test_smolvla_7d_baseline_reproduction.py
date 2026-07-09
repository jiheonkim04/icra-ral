import argparse

from tca_map.smolvla_lora_baseline import libero_7d_baseline_reproduction as repro


def test_baseline_reproduction_decision_set_is_exact():
    assert repro.FINAL_DECISIONS == {
        "READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D",
        "READY_FOR_METHOD_BUT_NEEDS_STRONGER_HEAD",
        "BASELINE_STILL_MLP_DOMINATED",
        "BASELINE_STILL_MEAN_DOMINATED",
        "DATA_SPLIT_NOT_MEANINGFUL",
        "TOO_HEAVY_LOCAL",
    }


def test_baseline_reproduction_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_SMOLVLA_LIBERO_7D_BASELINE_REPRODUCTION", raising=False)
    monkeypatch.delenv("ALLOW_SMOLVLA_LIBERO_7D_BASELINE_TRAINING", raising=False)
    args = argparse.Namespace(
        hdf5_path=str(tmp_path / "missing.hdf5"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        report_path=str(tmp_path / "report.json"),
        adapter_steps=1,
        small_mlp_steps=1,
        adapter_hidden_dim=8,
        learning_rate=5e-3,
        lora_learning_rate=1e-3,
    )

    report, code = repro.build_report(args)

    assert code != 0
    assert report["decision"] == "DATA_SPLIT_NOT_MEANINGFUL"
    assert report["policy"]["training_performed"] is False
    assert "ALLOW_SMOLVLA_LIBERO_7D_BASELINE_REPRODUCTION" in report["summary"]["exact_next_step"]


def test_target_module_audit_excludes_native_6d_action_modules():
    audit = repro._target_module_audit()

    current = audit["current_projection_modules"]
    assert current["executed_for_fixed_7d"] == ["state_proj"]
    assert "action_in_proj" in current["not_executed"]
    assert "action_out_proj" in current["not_executed"]
    assert "old hard-coded gripper" in audit["strict_boundary"] or "SO100 action normalizer" in audit["strict_boundary"]


def test_split_leakage_detects_record_overlap():
    train = [{"hdf5_path": "a.hdf5", "demo_name": "demo_0", "timestep": 1}]
    eval_records = [{"hdf5_path": "a.hdf5", "demo_name": "demo_0", "timestep": 1}]

    leakage = repro._split_leakage(train, eval_records)

    assert leakage["exact_record_overlap"] == 1
    assert leakage["has_exact_record_leakage"] is True
    assert leakage["has_demo_overlap"] is True


def test_sample_timesteps_are_unique_and_bounded():
    values = repro._sample_timesteps(length=20, count=5, start=3, stop=18)

    assert values == sorted(set(values))
    assert min(values) >= 3
    assert max(values) < 18
