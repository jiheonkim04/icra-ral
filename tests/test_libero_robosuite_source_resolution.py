import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "45_resolve_libero_robosuite_sources.ps1"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "46_prepare_libero_robosuite_sources.ps1"


def _run_resolution(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for source resolution")
    json_report = tmp_path / "source_resolution.json"
    md_report = tmp_path / "source_resolution.md"
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
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_libero_robosuite_source_resolution_splits_repo_and_dataset_decisions(tmp_path):
    result, report, json_report, md_report = _run_resolution(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_repo_setup"] is True
    assert report["ready_for_full_dataset_download"] is True
    assert report["sources"]["libero_repo"]["source_url"] == "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    assert report["sources"]["robosuite_repo"]["source_url"] == "https://github.com/ARISE-Initiative/robosuite.git"
    assert report["sources"]["libero_full_dataset"]["source_url"] == "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
    assert report["sources"]["libero_full_dataset"]["expected_size_gb"] == 100.0
    assert report["sources"]["libero_full_dataset"]["decision"] == "proceed"
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["simulator_executed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_libero_robosuite_setup_requires_download_gate():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'ALLOW_DOWNLOADS") -eq "1"' in content
    assert "git\", \"clone\", \"--depth\", \"1\"" not in content
    assert "clone\", \"--depth\", \"1\"" in content
    assert "full_dataset_downloaded = $false" in content
