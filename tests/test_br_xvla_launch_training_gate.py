from __future__ import annotations

from pathlib import Path

from tca_map.xvla_task1 import launch_training_gate as launch_module
from tca_map.xvla_task1.launch_training_gate import LaunchGateConfig, build_launch_command, launch_training_gate


def test_build_launch_command_preserves_offline_env_and_gate_module(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launch_module, "_run_capture", lambda _cmd: "/mnt/c/Users/jiheo/tca_map")
    config = LaunchGateConfig(
        repo_root=tmp_path,
        output_root=tmp_path / "runs" / "train",
        offline_output=tmp_path / "runs" / "offline.json",
        max_steps_override=1,
        num_validation_chunks=4,
        denoise_steps=2,
        dry_run=True,
    )

    launch = build_launch_command(config)

    assert "TRANSFORMERS_OFFLINE=1" in launch["inner_command"]
    assert "HF_DATASETS_OFFLINE=1" in launch["inner_command"]
    assert "HF_HUB_OFFLINE=1" in launch["inner_command"]
    assert "-m tca_map.xvla_task1.training_gate" in launch["inner_command"]
    assert "--max-steps-override 1" in launch["inner_command"]
    assert "gate_result.json" in launch["paths"]["gate_result"]


def test_dry_run_launch_writes_manifest_without_training(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launch_module, "_run_capture", lambda _cmd: "/mnt/c/Users/jiheo/tca_map")

    result = launch_training_gate(
        LaunchGateConfig(
            repo_root=tmp_path,
            output_root=tmp_path / "runs" / "train",
            offline_output=tmp_path / "runs" / "offline.json",
            max_steps_override=1,
            dry_run=True,
        )
    )

    manifest = Path(result["manifest_path"])
    assert manifest.is_file()
    assert result["status"] == "DRY_RUN"
    assert result["training_happened_at_launch_manifest_write"] is False
    assert result["optimizer_step_happened_at_launch_manifest_write"] is False
    assert result["closed_loop_ours_evaluation_happened_at_launch_manifest_write"] is False
    assert (tmp_path / "runs" / "train" / "gate_exact_resume_command.txt").is_file()
