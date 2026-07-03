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
    assert isinstance(report["hard_stop_reached"], bool)
    assert isinstance(report["assets"]["ready_for_smolvla_adapter_smoke"], bool)
    assert isinstance(report["assets"]["ready_for_openvla_oft_smoke"], bool)
    assert isinstance(report["assets"]["ready_for_libero_rollout"], bool)
    gates = {item["gate"] for item in report["risk_gate_requests"]}
    assert "runtime_install" in gates
    assert "smolvla_load_only_heavy_import" in gates
    assert "tiny_head_only_training" in gates
    assert "single_sample_interface_smoke" in gates
    blocking_gates = {
        item["gate"]
        for item in report["risk_gate_requests"]
        if item["current_blocker"]
    }
    assert set(report["current_blocking_gates"]) == blocking_gates
    runtime_request = next(
        item for item in report["risk_gate_requests"] if item["gate"] == "runtime_install"
    )
    load_only_request = next(
        item for item in report["risk_gate_requests"] if item["gate"] == "smolvla_load_only_heavy_import"
    )
    tiny_request = next(
        item for item in report["risk_gate_requests"] if item["gate"] == "tiny_head_only_training"
    )
    interface_request = next(
        item for item in report["risk_gate_requests"] if item["gate"] == "single_sample_interface_smoke"
    )
    assert load_only_request["risk_assessed_autonomy"] is True
    assert tiny_request["risk_assessed_autonomy"] is True
    assert interface_request["risk_assessed_autonomy"] is True
    assert load_only_request["current_blocker"] is False
    assert tiny_request["current_blocker"] is False
    assert interface_request["current_blocker"] is False
    assert "ready_for_autonomous_tiny_training_smoke" in report["tiny_head_only"]
    assert "smoke_passed" in report["tiny_head_only"]
    assert "decision" in report["go_no_go"]
    assert "single_sample_interface_passed" in report["smolvla_smokes"]
    assert "eval_smoke_passed" in report["feature_cache"]
    assert "num2words" in report["runtime"]["required"]
    if runtime_request["current_blocker"]:
        assert "runtime installation" in report["hard_stop_reason"]
    else:
        assert report["hard_stop_reason"] is None
        assert report["hard_stop_reached"] is False
        if report["go_no_go"]["decision"]:
            assert "Go/no-go summary is complete" in report["recommended_next_step"]
        elif report["tiny_head_only"]["smoke_passed"]:
            assert "go/no-go summary" in report["recommended_next_step"]
        else:
            assert "Continue autonomous SmolVLA pilot" in report["recommended_next_step"]
    assert report_path.exists()
