import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tca_map.actionmap_anchor.diagnostic import build_actionmap_anchor_diagnostic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "210_actionmap_anchor_diagnostic.ps1"


def _write_demo(path: Path, *, offset: float, task: str) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 48
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["language"] = f"put the {task} on the plate"
        actions = demo.create_dataset("actions", shape=(steps, 7), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(steps, 3), dtype="f4")
        ee_ori = obs.create_dataset("ee_ori", shape=(steps, 3), dtype="f4")
        joints = obs.create_dataset("joint_states", shape=(steps, 7), dtype="f4")
        grip = obs.create_dataset("gripper_states", shape=(steps, 2), dtype="f4")
        for step in range(steps):
            phase = step / float(steps - 1)
            ee = np.asarray([phase + offset, phase * phase, 0.2 + 0.1 * np.sin(np.pi * phase)], dtype=np.float64)
            action = np.asarray(
                [
                    np.sin(np.pi * phase + offset),
                    np.cos(np.pi * phase) * 0.4,
                    phase - 0.5,
                    0.3 * np.sin(2 * np.pi * phase),
                    -0.2 + offset,
                    0.15 * np.cos(np.pi * phase),
                    1.0 if phase > 0.55 else -1.0,
                ],
                dtype=np.float64,
            )
            actions[step, :] = np.clip(action, -1.0, 1.0)
            ee_pos[step, :] = ee
            ee_ori[step, :] = [0.1, 0.2 * phase, -0.1]
            joints[step, :] = np.linspace(0.0, 1.0, 7) * (phase + offset)
            grip[step, :] = [0.01, -0.01] if phase > 0.55 else [0.04, -0.04]


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "libero"
    _write_demo(root / "libero_10" / "demo_a.hdf5", offset=0.0, task="mug")
    _write_demo(root / "libero_10" / "demo_b.hdf5", offset=0.2, task="bowl")
    return root


def test_actionmap_anchor_diagnostic_runs_required_baselines(tmp_path):
    root = _fixture_root(tmp_path)

    report = build_actionmap_anchor_diagnostic(
        libero_data_root=root,
        max_demos=2,
        max_action_steps=40,
        feature_width=24,
        max_steps=12,
        learning_rate=0.15,
        trans_bins=5,
        rot_bins=5,
    )

    assert report["schema_version"] == "actionmap-anchor-diagnostic-v1"
    assert report["policy"]["training_performed"] is True
    assert report["model"]["loss_computed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["official_actionmap_reproduction_attempted"] is False
    assert report["policy"]["new_extension_implemented"] is False
    assert report["policy"]["failure_mining_performed"] is False
    assert report["data"]["real_hdf5_metric_produced"] is True
    assert {"mean_action_baseline", "linear_l1_action_head", "oracle_nearest_action_candidate_upper_bound"}.issubset(
        report["variants"]
    )
    if report["model"]["oracle_gate_passed"]:
        assert {"simple_mlp_action_head", "actionmap_heatmap_candidate_head"}.issubset(report["variants"])
        actionmap = report["variants"]["actionmap_heatmap_candidate_head"]
        assert actionmap["candidate_diagnostics"]["heatmap_nll"] is not None
        assert actionmap["candidate_diagnostics"]["translation_top3_accuracy"] is not None
        assert actionmap["candidate_diagnostics"]["unique_translation_bins"] >= 1
    else:
        assert "simple_mlp_action_head" not in report["variants"]
        assert "actionmap_heatmap_candidate_head" not in report["variants"]
        assert report["decision"]["decision"] == "kill"
    for payload in report["variants"].values():
        assert payload["metrics"]["action_l2"] is not None
        assert payload["metrics"]["translation_l2"] is not None
        assert payload["metrics"]["rotation_l2"] is not None
        assert payload["metrics"]["gripper_error"] is not None
        assert payload["per_phase"]
    assert report["decision"]["decision"] in {"continue", "kill", "blocked"}
    assert "actionmap_action_l2" in report["decision"]


def test_actionmap_anchor_runner_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ActionMap anchor runner tests")
    root = _fixture_root(tmp_path)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoDataRoot",
            str(root),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_actionmap_anchor_runner_outputs_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ActionMap anchor runner tests")
    root = _fixture_root(tmp_path)
    report_json = tmp_path / "report.json"

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoDataRoot",
            str(root),
            "-MaxDemos",
            "2",
            "-MaxActionSteps",
            "32",
            "-MaxSteps",
            "8",
            "-TransBins",
            "5",
            "-RotBins",
            "5",
            "-JsonReportPath",
            str(report_json),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_TINY_TRAINING": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_json.read_text(encoding="utf-8-sig"))
    assert data["schema_version"] == "actionmap-anchor-diagnostic-v1"
    assert "actionmap_heatmap_candidate_head" in data["variants"]
