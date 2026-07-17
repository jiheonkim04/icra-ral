from __future__ import annotations

from pathlib import Path

from tca_map.xvla_task6 import training_gate
from tca_map.xvla_task6.training_gate import TrainingGateConfig, run_training_gate


def test_task6_training_gate_runs_exactly_frozen_two_arms_then_offline(monkeypatch, tmp_path: Path) -> None:
    trained: list[str] = []

    def fake_train(config):
        trained.append(config.arm_id)
        return {
            "success": True,
            "status": "COMPLETE",
            "arm_id": config.arm_id,
            "optimizer_steps_completed": config.max_steps_override or 64,
        }

    def fake_offline(config):
        return {
            "status": "COMPLETE",
            "success": False,
            "decision": "MPR_XVLA_OFFLINE_SELECTION_NOT_PASSED",
            "closed_loop_ours_evaluation_happened": False,
            "output_path": str(config.output_path),
        }

    monkeypatch.setattr(training_gate, "run_training_arm", fake_train)
    monkeypatch.setattr(training_gate, "run_offline_validation", fake_offline)

    result = run_training_gate(
        TrainingGateConfig(
            output_root=tmp_path / "training",
            offline_output=tmp_path / "offline.json",
            max_steps_override=None,
        )
    )

    assert trained == ["mpr_xvla_rank8_lambda2_lr1e4_steps64", "uniform_task6_xvla_rank8_lambda0_lr1e4_steps64"]
    assert result["status"] == "COMPLETE"
    assert result["success"] is False
    assert result["decision"] == "MPR_XVLA_OFFLINE_SELECTION_NOT_PASSED"
    assert result["closed_loop_ours_evaluation_happened"] is False
