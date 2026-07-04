import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "57_setup_wsl_simulator_deps.ps1"


def test_wsl_simulator_dependency_setup_dry_run_is_safe(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the WSL simulator dependency setup check")

    json_report = tmp_path / "wsl_simulator_dependency_setup_report.json"
    md_report = tmp_path / "wsl_simulator_dependency_setup_report.md"
    env = os.environ.copy()
    env.pop("ALLOW_WSL_SIM_DEPS", None)
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
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    report = json.loads(result.stdout[start:])

    assert report["policy"]["bounded_wsl_python_packaging_setup"] is True
    assert report["policy"]["apt_installs_performed"] is False
    assert report["policy"]["sudo_used"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["simulator_environment_steps_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["tokens_read_or_written"] is False
    assert report["policy"]["paper_grade_claims_made"] is False
    assert report["execution"]["requested"] is False
    assert report["execution"]["executed"] is False
    assert report["risk_assessment"]["expected_vram_gb"] == 0
    assert report["risk_assessment"]["token_license_payment_needed"] is False
    assert report["risk_assessment"]["cuda_driver_system_graphics_changes"] is False
    assert "recommended_next_step" in report
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_simulator_dependency_setup_execute_requires_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the WSL simulator dependency setup check")

    json_report = tmp_path / "wsl_simulator_dependency_setup_report.json"
    md_report = tmp_path / "wsl_simulator_dependency_setup_report.md"
    env = os.environ.copy()
    env.pop("ALLOW_WSL_SIM_DEPS", None)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Execute",
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    report = json.loads(result.stdout[start:])

    assert report["execution"]["requested"] is True
    assert report["execution"]["gate_present"] is False
    assert report["execution"]["executed"] is False
    assert "ALLOW_WSL_SIM_DEPS=1" in report["recommended_next_step"]
