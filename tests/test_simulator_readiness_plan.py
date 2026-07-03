import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "43_plan_simulator_readiness.ps1"


def _run_planner(tmp_path, extra_args=None, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the simulator readiness planner")
    json_report = tmp_path / "simulator_readiness_plan_report.json"
    md_report = tmp_path / "simulator_readiness_plan_report.md"
    env = os.environ.copy()
    for key in ("LIBERO_ROOT", "LIBERO_DATA_ROOT", "ROBOSUITE_ROOT", "DATA_ROOT"):
        env[key] = ""
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


def test_simulator_readiness_stops_when_paths_missing(tmp_path):
    result, report, json_report, md_report = _run_planner(
        tmp_path,
        extra_args=["-RuntimePlatform", "linux"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["simulator_imports_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert "LIBERO_ROOT is missing or does not exist" in report["stop_reasons"]
    assert "\x00" not in report["wsl"]["status"]["output"]
    assert "\x00" not in report["wsl"]["distros"]["output"]
    assert json_report.exists()
    assert md_report.exists()


def test_simulator_readiness_allows_import_smoke_plan_on_linux_with_paths(tmp_path):
    libero_root = tmp_path / "LIBERO"
    libero_data_root = tmp_path / "libero_data"
    robosuite_root = tmp_path / "robosuite"
    for path in (libero_root, libero_data_root, robosuite_root):
        path.mkdir()

    result, report, _, _ = _run_planner(
        tmp_path,
        extra_args=["-RuntimePlatform", "linux"],
        extra_env={
            "LIBERO_ROOT": str(libero_root),
            "LIBERO_DATA_ROOT": str(libero_data_root),
            "ROBOSUITE_ROOT": str(robosuite_root),
            "DATA_ROOT": str(tmp_path / "data"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_simulator_path_check"] is True
    assert report["ready_for_dataset_path_check"] is True
    assert report["ready_for_simulator_import_smoke"] is True
    assert report["ready_for_simulator_render_smoke"] is False
    assert report["ready_for_libero_rollout"] is False
    assert report["policy"]["render_smoke_performed"] is False


def test_simulator_readiness_blocks_native_windows_import_smoke(tmp_path):
    libero_root = tmp_path / "LIBERO"
    robosuite_root = tmp_path / "robosuite"
    for path in (libero_root, robosuite_root):
        path.mkdir()

    result, report, _, _ = _run_planner(
        tmp_path,
        extra_args=["-RuntimePlatform", "windows"],
        extra_env={
            "LIBERO_ROOT": str(libero_root),
            "ROBOSUITE_ROOT": str(robosuite_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_simulator_path_check"] is True
    assert report["ready_for_simulator_import_smoke"] is False
    assert any("native Windows" in warning for warning in report["warnings"])
    assert "simulator import smoke should run in WSL2/Linux, not native Windows" in report["stop_reasons"]
