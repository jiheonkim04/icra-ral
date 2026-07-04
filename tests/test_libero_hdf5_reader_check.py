import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "50_check_libero_hdf5_reader.ps1"


def test_hdf5_reader_check_is_check_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for HDF5 reader checker tests")

    json_report = tmp_path / "hdf5_reader.json"
    md_report = tmp_path / "hdf5_reader.md"
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
    assert start >= 0, result.stdout
    report = json.loads(result.stdout[start:])
    assert report["policy"]["check_only"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert isinstance(report["h5py"]["available"], bool)
    assert json_report.exists()
    assert md_report.exists()


def test_hdf5_reader_check_refuses_execution_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for HDF5 reader checker tests")

    env = os.environ.copy()
    env["ALLOW_DOWNLOADS"] = "1"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-JsonReportPath",
            str(tmp_path / "hdf5_reader.json"),
            "-MarkdownReportPath",
            str(tmp_path / "hdf5_reader.md"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "ALLOW_DOWNLOADS" in (result.stdout + result.stderr)
