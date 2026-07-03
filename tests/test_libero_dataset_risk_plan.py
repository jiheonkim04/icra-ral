import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "42_plan_libero_dataset_risk.ps1"


def _run_planner(tmp_path, extra_args=None, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the LIBERO dataset risk planner")
    json_report = tmp_path / "libero_dataset_risk_report.json"
    md_report = tmp_path / "libero_dataset_risk_report.md"
    env = os.environ.copy()
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


def test_libero_dataset_risk_planner_stops_when_source_and_paths_missing(tmp_path):
    result, report, json_report, md_report = _run_planner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert any("dataset source is missing" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()


def test_libero_dataset_risk_planner_detects_local_tiny_subset(tmp_path):
    libero_root = tmp_path / "LIBERO"
    libero_data_root = tmp_path / "libero_data"
    robosuite_root = tmp_path / "robosuite"
    for path in (libero_root, libero_data_root, robosuite_root):
        path.mkdir()
    (libero_data_root / "tiny_demo.hdf5").write_text("dummy", encoding="utf-8")

    result, report, _, _ = _run_planner(
        tmp_path,
        extra_env={
            "LIBERO_ROOT": str(libero_root),
            "LIBERO_DATA_ROOT": str(libero_data_root),
            "ROBOSUITE_ROOT": str(robosuite_root),
            "DATA_ROOT": str(tmp_path / "data"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_libero_path_check"] is True
    assert report["ready_for_libero_offline_subset"] is True
    assert report["ready_for_libero_rollout"] is False
    assert report["dataset_probe"]["data_files_detected"] is True


def test_libero_dataset_risk_planner_allows_green_future_acquisition(tmp_path):
    target = tmp_path / "libero_target"
    result, report, _, _ = _run_planner(
        tmp_path,
        extra_args=[
            "-Source",
            "official/example-libero-source",
            "-OfficialSource",
            "-ExpectedSizeGb",
            "0.01",
            "-TargetPath",
            str(target),
        ],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_libero_dataset_acquisition"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["warnings"]
