from __future__ import annotations

from pathlib import Path

from tca_map.xvla_task1 import launch_closed_loop_residual as launch_module
from tca_map.xvla_task1.launch_closed_loop_residual import (
    LaunchClosedLoopConfig,
    build_launch_command,
    launch_closed_loop_residual,
)


def test_build_launch_command_freezes_exact_residual_module(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launch_module, "_run_capture", lambda _cmd: "/mnt/c/Users/jiheo/tca_map")
    config = LaunchClosedLoopConfig(repo_root=tmp_path, output_root=tmp_path / "runs" / "closed_loop", dry_run=True)

    launch = build_launch_command(config)

    assert "TRANSFORMERS_OFFLINE=1" in launch["inner_command"]
    assert "HF_DATASETS_OFFLINE=1" in launch["inner_command"]
    assert "HF_HUB_OFFLINE=1" in launch["inner_command"]
    assert "-m tca_map.xvla_task1.closed_loop_residual_eval" in launch["inner_command"]
    assert "--identities 20260727" in launch["inner_command"]
    assert "--policies xvla_prior_base,br_xvla_primary,uniform_xvla_ablation" in launch["inner_command"]
    assert "closed_loop_result.json" in launch["paths"]["eval_result"]


def test_dry_run_launch_writes_manifest_without_eval(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launch_module, "_run_capture", lambda _cmd: "/mnt/c/Users/jiheo/tca_map")

    result = launch_closed_loop_residual(
        LaunchClosedLoopConfig(repo_root=tmp_path, output_root=tmp_path / "runs" / "closed_loop", dry_run=True)
    )

    manifest = Path(result["manifest_path"])
    assert manifest.is_file()
    assert result["status"] == "DRY_RUN"
    assert result["identities"] == [20260727]
    assert result["training_happened_at_launch_manifest_write"] is False
    assert result["optimizer_step_happened_at_launch_manifest_write"] is False
    assert result["checkpoint_written_at_launch_manifest_write"] is False
    assert result["closed_loop_ours_evaluation_happened_at_launch_manifest_write"] is False
    assert result["retuning_from_result_allowed"] is False
    assert (tmp_path / "runs" / "closed_loop" / "closed_loop_exact_resume_command.txt").is_file()
