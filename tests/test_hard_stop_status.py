import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "27_summarize_hard_stop_status.ps1"


def test_hard_stop_status_summary_is_check_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the hard-stop status summary")

    report_path = tmp_path / "hard_stop_status_report.json"
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
        env=os.environ.copy(),
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

    assert report["policy"]["summary_only"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["hard_stop_reached"] is True
    assert isinstance(report["assets"]["ready_for_smolvla_adapter_smoke"], bool)
    assert isinstance(report["assets"]["ready_for_openvla_oft_smoke"], bool)
    assert isinstance(report["assets"]["ready_for_libero_rollout"], bool)
    gates = {item["gate"] for item in report["approval_requests"]}
    assert "runtime_install" in gates
    assert "smolvla_load_only_heavy_import" in gates
    assert "tiny_head_only_training" in gates
    blocking_gates = {
        item["gate"]
        for item in report["approval_requests"]
        if item["current_blocker"]
    }
    assert set(report["current_blocking_gates"]) == blocking_gates
    runtime_request = next(
        item for item in report["approval_requests"] if item["gate"] == "runtime_install"
    )
    if runtime_request["current_blocker"]:
        assert "runtime installation" in report["hard_stop_reason"]
    else:
        assert "runtime installation" not in report["hard_stop_reason"]
    assert report_path.exists()
