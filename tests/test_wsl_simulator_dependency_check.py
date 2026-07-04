import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "56_check_wsl_simulator_deps.ps1"


def test_wsl_simulator_dependency_check_is_check_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the WSL simulator dependency check")

    json_report = tmp_path / "wsl_simulator_dependency_report.json"
    md_report = tmp_path / "wsl_simulator_dependency_report.md"
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

    assert report["policy"]["check_only"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["simulator_environment_steps_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["tokens_read_or_written"] is False
    assert report["policy"]["paper_grade_claims_made"] is False
    assert report["decision"] in {"proceed", "stop"}
    assert report["ready_for_user_level_pip_install"] in {True, False}
    assert report["ready_for_simulator_import_retry"] in {True, False}
    assert "recommended_next_step" in report
    assert json_report.exists()
    assert md_report.exists()
