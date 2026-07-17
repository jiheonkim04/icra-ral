from __future__ import annotations

from pathlib import Path

import pytest

from tca_map.xvla_spatial_task5 import training_gate
from tca_map.xvla_spatial_task5.training_gate import TrainingGateConfig, run_training_gate


def test_training_gate_runs_frozen_arms_then_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_train(config):
        calls.append(config.arm_id)
        return {"success": True, "decision": "FAKE_ARM_COMPLETE", "arm_id": config.arm_id}

    def fake_offline(config):
        calls.append("offline")
        return {"success": True, "decision": "R2P_XVLA_OFFLINE_PASS_BEATS_UNIFORM_ABLATION"}

    monkeypatch.setattr(training_gate, "run_training_arm", fake_train)
    monkeypatch.setattr(training_gate, "run_offline_validation", fake_offline)

    result = run_training_gate(
        TrainingGateConfig(
            output_root=tmp_path / "training",
            offline_output=tmp_path / "offline.json",
        )
    )

    assert result["status"] == "COMPLETE"
    assert result["success"] is True
    assert calls == [
        "r2p_xvla_rank8_phase_weights_lr1e4_steps64",
        "uniform_task5_xvla_rank8_lambda0_lr1e4_steps64",
        "offline",
    ]
    assert (tmp_path / "training" / "gate_result.json").is_file()
    assert (tmp_path / "training" / "gate_heartbeat.json").is_file()


def test_training_gate_skips_offline_for_debug_max_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_train(config):
        calls.append(config.arm_id)
        return {"success": True, "decision": "FAKE_ARM_COMPLETE", "arm_id": config.arm_id}

    def fake_offline(config):  # pragma: no cover - must not be called
        raise AssertionError("offline validation should be skipped for debug step override")

    monkeypatch.setattr(training_gate, "run_training_arm", fake_train)
    monkeypatch.setattr(training_gate, "run_offline_validation", fake_offline)

    result = run_training_gate(
        TrainingGateConfig(
            output_root=tmp_path / "training",
            offline_output=tmp_path / "offline.json",
            max_steps_override=1,
        )
    )

    assert result["decision"] == "R2P_XVLA_TRAINING_DEBUG_COMPLETE_OFFLINE_SKIPPED"
    assert result["success"] is True
    assert calls == [
        "r2p_xvla_rank8_phase_weights_lr1e4_steps64",
        "uniform_task5_xvla_rank8_lambda0_lr1e4_steps64",
    ]


def test_training_gate_rejects_downloads_before_runtime_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="downloads are not allowed"):
        run_training_gate(
            TrainingGateConfig(
                output_root=tmp_path / "training",
                offline_output=tmp_path / "offline.json",
                local_files_only=False,
            )
        )
