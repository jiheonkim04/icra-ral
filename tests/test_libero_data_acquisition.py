import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_data_acquisition import (
    LIBERO_DOWNLOAD_BUDGET_GB,
    MIN_FREE_AFTER_GB,
    OFFICIAL_SOURCE_URL,
    build_risk_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "49_acquire_libero_data.ps1"


def test_libero_data_acquisition_budget_is_limited_to_official_source(tmp_path):
    report = build_risk_report(target=tmp_path / "libero", cache=tmp_path / "hf_home")

    assert report["source"]["source_url"] == OFFICIAL_SOURCE_URL
    assert report["source"]["official_or_documented"] is True
    assert report["source"]["token_required"] is False
    assert report["source"]["login_required"] is False
    assert report["source"]["license_click_through_required"] is False
    assert report["source"]["payment_required"] is False
    assert report["budgets"]["libero_download_budget_gb"] == LIBERO_DOWNLOAD_BUDGET_GB
    assert report["budgets"]["min_free_after_gb"] == MIN_FREE_AFTER_GB
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert any("target path is outside" in reason for reason in report["stop_reasons"])


def test_libero_data_acquisition_script_is_plan_only_by_default(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for acquisition script tests")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-TargetPath",
            "C:\\assets\\data\\libero",
            "-CachePath",
            "C:\\assets\\hf_home",
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

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0
    report = json.loads(result.stdout[start:])
    assert report["policy"]["downloads_performed"] is False
    assert report["decision"] in {"proceed", "stop"}
    assert (tmp_path / "report.json").exists()


def test_libero_data_acquisition_requires_download_gate_for_acquire(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for acquisition script tests")

    env = os.environ.copy()
    env.pop("ALLOW_DOWNLOADS", None)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Acquire",
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_DOWNLOADS=1" in (result.stdout + result.stderr)
