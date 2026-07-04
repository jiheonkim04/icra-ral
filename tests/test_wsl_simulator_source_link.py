import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "60_link_wsl_simulator_sources.ps1"


def _run_script(tmp_path, extra_args=None, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the WSL simulator source link check")

    json_report = tmp_path / "wsl_simulator_source_link_report.json"
    md_report = tmp_path / "wsl_simulator_source_link_report.md"
    env = os.environ.copy()
    env.pop("ALLOW_WSL_SIM_SOURCE_LINK", None)
    env.update(extra_env or {})

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PathsFile",
            str(tmp_path / "missing_paths.local.yaml"),
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


def test_wsl_simulator_source_link_dry_run_is_safe(tmp_path):
    result, report, json_report, md_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["policy"]["source_link_only"] is True
    assert report["policy"]["uses_existing_wsl_venv"] is True
    assert report["policy"]["creates_repo_local_venv"] is False
    assert report["policy"]["pip_no_index"] is True
    assert report["policy"]["pip_no_deps"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["reset_step_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["execution"]["requested"] is False
    assert report["execution"]["executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_simulator_source_link_execute_requires_gate(tmp_path):
    robosuite_root = tmp_path / "robosuite"
    libero_root = tmp_path / "LIBERO"
    for root in (robosuite_root, libero_root):
        root.mkdir()
        (root / "setup.py").write_text("from setuptools import setup\nsetup(name='dummy')\n", encoding="utf-8")

    result, report, _, _ = _run_script(
        tmp_path,
        extra_args=["-Execute"],
        extra_env={
            "ROBOSUITE_ROOT": str(robosuite_root),
            "LIBERO_ROOT": str(libero_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["execution"]["requested"] is True
    assert report["execution"]["gate_present"] is False
    assert report["execution"]["executed"] is False
    assert "ALLOW_WSL_SIM_SOURCE_LINK=1" in report["recommended_next_step"]
