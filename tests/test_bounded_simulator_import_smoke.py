import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "55_bounded_simulator_import_smoke.ps1"


def _run_script(tmp_path, extra_args=None, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the bounded simulator import smoke")

    json_report = tmp_path / "bounded_simulator_import_smoke_report.json"
    md_report = tmp_path / "bounded_simulator_import_smoke_report.md"
    env = os.environ.copy()
    env.pop("ALLOW_SIMULATOR_IMPORT_SMOKE", None)
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


def test_bounded_simulator_import_smoke_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["bounded_simulator_import_smoke_passed"] is False
    assert report["policy"]["task_local_gate_required"] == "ALLOW_SIMULATOR_IMPORT_SMOKE=1"
    assert report["policy"]["simulator_imports_attempted"] is False
    assert report["policy"]["simulator_imports_performed"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_bounded_simulator_import_smoke_stops_on_windows_runtime_before_import(tmp_path):
    libero_root = tmp_path / "LIBERO"
    robosuite_root = tmp_path / "robosuite"
    libero_data_root = tmp_path / "libero_data"
    for path in (libero_root, robosuite_root, libero_data_root):
        path.mkdir()

    result, report, _, _ = _run_script(
        tmp_path,
        extra_args=["-RuntimePlatform", "windows"],
        extra_env={
            "ALLOW_SIMULATOR_IMPORT_SMOKE": "1",
            "LIBERO_ROOT": str(libero_root),
            "ROBOSUITE_ROOT": str(robosuite_root),
            "LIBERO_DATA_ROOT": str(libero_data_root),
            "DATA_ROOT": str(tmp_path / "data"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["bounded_simulator_import_smoke_passed"] is False
    assert report["planner"]["ready_for_simulator_path_check"] is True
    assert report["planner"]["ready_for_simulator_import_smoke"] is False
    assert report["policy"]["simulator_imports_attempted"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["ready_for_rollout"] is False
