import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_interface import build_offline_interface_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "48_plan_libero_offline_interface_smoke.ps1"


def test_offline_interface_gate_stops_when_no_files_exist(tmp_path):
    report = build_offline_interface_report(tmp_path)

    assert report["decision"] == "stop"
    assert report["dataset_files_detected"] is False
    assert report["ready_for_offline_interface_smoke"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["simulator_executed"] is False
    assert report["ready_for_rollout"] is False


def test_offline_interface_gate_accepts_tiny_jsonl_fixture(tmp_path):
    sample_path = tmp_path / "tiny_libero_style.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "instruction": "pick up the mug",
                "target": {"name": "mug"},
                "expert_action": [0.1, 0.0, 0.2, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_offline_interface_report(tmp_path)

    assert report["decision"] == "proceed"
    assert report["dataset_files_detected"] is True
    assert report["ready_for_offline_interface_smoke"] is True
    assert report["file_inspections"][0]["reader"] == "jsonl"
    assert report["file_inspections"][0]["interface_ready"] is True
    assert report["policy"]["training_performed"] is False


def test_offline_interface_script_refuses_execution_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline interface gate script tests")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_TINY_TRAINING": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing offline interface smoke gate" in (result.stdout + result.stderr)


def test_offline_interface_script_reports_stop_for_empty_temp_data_root(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline interface gate script tests")

    paths_file = tmp_path / "paths.local.yaml"
    paths_file.write_text(f"assets:\n  libero_data_root: \"{tmp_path}\"\n", encoding="utf-8")
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-PathsFile",
            str(paths_file),
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
    assert report["decision"] == "stop"
    assert report["ready_for_offline_interface_smoke"] is False
