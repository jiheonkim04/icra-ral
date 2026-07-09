import argparse
import json
from pathlib import Path

import numpy as np

from tca_map.patchguard_vla import diagnostic
from tca_map.patchguard_vla import state1b


def test_patch_variant_changes_agentview_without_state_leakage():
    obs = {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "agentview_rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }

    patched, metadata = diagnostic.apply_patch_variant(obs, diagnostic.VARIANT_FIXED_VISIBLE_PATCH, seed=1)

    assert metadata["agentview_modified"] is True
    assert metadata["eye_in_hand_modified"] is False
    assert metadata["state_modified"] is False
    assert np.any(patched["agentview_image"] != obs["agentview_image"])
    assert np.array_equal(patched["robot0_eef_pos"], obs["robot0_eef_pos"])
    assert np.array_equal(patched["robot0_eef_quat"], obs["robot0_eef_quat"])


def test_state1_decision_priority():
    assert (
        diagnostic._state1_decision(
            patch_effect_nontrivial=False,
            kinematic_signal_available=True,
            baseline_dominated=False,
            real_vla_used=True,
            local_adapter_path_feasible_now=True,
        )
        == "KILL_ATTACK_NOT_REPRODUCIBLE"
    )
    assert (
        diagnostic._state1_decision(
            patch_effect_nontrivial=True,
            kinematic_signal_available=True,
            baseline_dominated=True,
            real_vla_used=True,
            local_adapter_path_feasible_now=True,
        )
        == "KILL_BASELINE_DOMINATED"
    )
    assert (
        diagnostic._state1_decision(
            patch_effect_nontrivial=True,
            kinematic_signal_available=True,
            baseline_dominated=False,
            real_vla_used=True,
            local_adapter_path_feasible_now=False,
        )
        == "TOO_HEAVY_LOCAL"
    )


def test_patchguard_runner_requires_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_PATCHGUARD_VLA_STATE1", raising=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"inputs": {"hdf5_path": str(tmp_path / "missing.hdf5")}}), encoding="utf-8")
    ckpt = tmp_path / "smolvla"
    hf_home = tmp_path / "hf"
    checkpoint_root = tmp_path / "checkpoints"
    ckpt.mkdir()
    hf_home.mkdir()
    checkpoint_root.mkdir()

    args = argparse.Namespace(
        plan_report=str(plan),
        smolvla_ckpt=str(ckpt),
        checkpoint_root=str(checkpoint_root),
        hf_home=str(hf_home),
        hdf5_path="",
        task="perform the task",
        report_path=str(tmp_path / "report.json"),
        device="cpu",
        seed=7,
        require_bitsandbytes_for_lora=False,
    )

    report, code = diagnostic.build_report(args, loader=lambda *args: None)

    assert code != 0
    assert report["decision"] == "stop"
    assert report["policy"]["model_load_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["recommended_next_step"]


def test_state1b_runner_requires_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_PATCHGUARD_VLA_STATE1B", raising=False)
    monkeypatch.delenv("ALLOW_PATCHGUARD_TINY_LORA_TRAINING", raising=False)
    args = argparse.Namespace(
        plan_report=str(tmp_path / "missing_plan.json"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        hf_home=str(tmp_path / "hf_home"),
        hdf5_path="",
        report_path=str(tmp_path / "report.json"),
        device="cuda",
        lora_rank=4,
        max_steps=10,
        max_train_samples=1,
        learning_rate=1e-3,
        seed=7,
        dependency_install_happened=False,
    )

    report, code = state1b.build_report(args)

    assert code != 0
    assert report["decision"] == "TOO_HEAVY_LOCAL"
    assert report["policy"]["training_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["summary"]["exact_next_step"]


def test_state1b_decision_set_is_exact():
    assert {
        "READY_FOR_PATCHGUARD_LORA_STATE2",
        "QLORA_BLOCKED_BUT_LORA_POSSIBLE",
        "ENV_BLOCKED_INSTALL_FAILED",
        "KILL_NO_ADAPTER_PATH",
        "KILL_BASELINE_DOMINATED",
        "TOO_HEAVY_LOCAL",
    } == state1b.FINAL_DECISIONS

