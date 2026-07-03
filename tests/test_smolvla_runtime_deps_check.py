import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "17_check_smolvla_runtime_deps.ps1"


def test_runtime_dependency_checker_is_check_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the runtime dependency checker")

    report_path = tmp_path / "runtime_deps_report.json"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ReportPath",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout
    report = json.loads(result.stdout[start:])

    assert report["policy"]["check_only"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["heavy_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert "torch" in report["runtime_dependencies"]["required"]
    assert "num2words" in report["runtime_dependencies"]["required"]
