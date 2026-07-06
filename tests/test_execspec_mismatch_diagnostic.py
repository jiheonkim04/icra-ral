import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.execspec import mismatch_diagnostic as diag


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "163_execspec_mismatch_diagnostic.ps1"


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 20
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(20, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(20,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(20,), dtype="i1")
        for step in range(20):
            phase = step / 19.0
            actions[step, :] = [
                0.10 + 0.20 * phase,
                -0.05 + 0.10 * phase,
                0.02 + 0.08 * phase,
                0.12 * phase,
                -0.10 * phase,
                0.06 * phase,
                -1.0 if step < 8 else 1.0,
            ]
            rewards[step] = 1.0 if step == 15 else 0.0
            dones[step] = 1 if step == 15 else 0


def _args(tmp_path: Path, demo: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=str(tmp_path / "missing_manifest.json"),
        demo_path=str(demo),
        max_steps=20,
        substantial_drift_threshold=0.10,
        gripper_mismatch_threshold=0.25,
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
    )


def test_execspec_mismatch_report_reproduces_drift_and_repair(tmp_path, monkeypatch):
    for gate in diag.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    demo = tmp_path / "demo.hdf5"
    _write_demo(demo)

    report = diag.build_report(_args(tmp_path, demo))

    assert report["result"]["passed"] is True
    assert report["summary"]["mismatch_reproduced"] is True
    assert report["summary"]["continue_or_kill"] == "continue"
    assert report["summary"]["simple_repair_baseline_beaten"] is True
    assert "gripper_sign_flip" in report["summary"]["substantial_mismatch_variants"]
    assert report["variants"]["correct_7d_expert_action_replay"]["metrics"]["action_l2_mean"] == 0.0
    assert report["variants"]["gripper_sign_flip"]["metrics"]["gripper_mismatch_rate"] == 1.0
    assert report["variants"]["per_dimension_scale_mismatch"]["repair"]["diagonal_affine_beats_simple_baselines"] is True
    assert report["policy"]["replay_or_rollout_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_execspec_mismatch_report_refuses_forbidden_gate(tmp_path, monkeypatch):
    demo = tmp_path / "demo.hdf5"
    _write_demo(demo)
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report = diag.build_report(_args(tmp_path, demo))

    assert report["result"]["passed"] is False
    assert report["summary"]["continue_or_kill"] == "blocked"
    assert "ALLOW_ROLLOUTS" in report["result"]["blocked_reason"]


def test_execspec_mismatch_script_writes_reports(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ExecSpec script tests")
    demo = tmp_path / "demo.hdf5"
    report_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    _write_demo(demo)
    env = os.environ.copy()
    for gate in diag.FORBIDDEN_GATES:
        env.pop(gate, None)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-DemoPath",
            str(demo),
            "-JsonReportPath",
            str(report_path),
            "-MarkdownReportPath",
            str(md_path),
            "-MaxSteps",
            "20",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert data["result"]["passed"] is True
    assert data["summary"]["mismatch_reproduced"] is True
    assert md_path.exists()
