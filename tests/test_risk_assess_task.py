import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "41_risk_assess_task.ps1"


def _run_risk(tmp_path, args):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the risk assessment script")
    json_report = tmp_path / "risk_assessment_report.json"
    md_report = tmp_path / "risk_assessment_report.md"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            *args,
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
    return result, json_report, md_report


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_risk_assessment_allows_small_official_download(tmp_path):
    result, json_report, md_report = _run_risk(
        tmp_path,
        [
            "-Task",
            "tiny metadata acquisition",
            "-Category",
            "download",
            "-Source",
            "official/example",
            "-TargetPath",
            r"C:\assets\example",
            "-ExpectedSizeGb",
            "0.01",
            "-ExpectedRuntimeMinutes",
            "1",
            "-OfficialSource",
        ],
    )
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["policy"]["risk_assessment_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["decision"] == "proceed"
    assert json_report.exists()
    assert md_report.exists()


def test_risk_assessment_stops_unknown_download_size(tmp_path):
    result, _, _ = _run_risk(
        tmp_path,
        [
            "-Task",
            "unknown dataset",
            "-Category",
            "download",
            "-Source",
            "official/example",
            "-TargetPath",
            r"C:\assets\example",
            "-OfficialSource",
        ],
    )
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["decision"] == "stop"
    assert any("expected size is unknown" in reason for reason in report["stop_reasons"])


def test_risk_assessment_stops_openvla_execution(tmp_path):
    result, _, _ = _run_risk(
        tmp_path,
        [
            "-Task",
            "openvla smoke",
            "-Category",
            "gpu",
            "-Source",
            "openvla",
            "-OpenVlaRequired",
        ],
    )
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["decision"] == "stop"
    assert report["gates"]["openvla_required"] is True
    assert any("OpenVLA-OFT" in reason for reason in report["stop_reasons"])
