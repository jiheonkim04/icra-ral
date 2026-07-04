import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "61_bounded_simulator_reset_step_smoke.ps1"


def _run_script(tmp_path, extra_env=None, extra_args=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the bounded simulator reset/step smoke")

    json_report = tmp_path / "bounded_simulator_reset_step_smoke_report.json"
    md_report = tmp_path / "bounded_simulator_reset_step_smoke_report.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
    ):
        env.pop(key, None)
    env.update(extra_env or {})

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
            *(extra_args or []),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_bounded_reset_step_smoke_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["bounded_simulator_reset_step_smoke_passed"] is False
    assert report["policy"]["task_local_gate_required"] == "ALLOW_SIMULATOR_RESET_STEP=1"
    assert report["policy"]["reset_step_smoke_attempted"] is False
    assert report["policy"]["reset_step_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_bounded_reset_step_smoke_refuses_rollout_gate(tmp_path):
    result, report, _, _ = _run_script(
        tmp_path,
        extra_env={
            "ALLOW_SIMULATOR_RESET_STEP": "1",
            "ALLOW_TINY_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_TINY_ROLLOUT" in report["reason"]
    assert report["policy"]["reset_step_smoke_attempted"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_bounded_reset_step_smoke_rejects_unbounded_step_count(tmp_path):
    result, report, _, _ = _run_script(tmp_path, extra_args=["-MaxSteps", "6"])

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "MaxSteps" in report["reason"]
    assert report["policy"]["reset_step_smoke_attempted"] is False
