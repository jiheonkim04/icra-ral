import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "31_generate_go_no_go_report.ps1"


def test_go_no_go_status_generator_is_summary_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the go/no-go status generator")

    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
    ]:
        env.pop(gate, None)

    json_report = tmp_path / "go_no_go_status_report.json"
    markdown_report = tmp_path / "go_no_go_status_report.md"
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
            str(json_report),
            "-MarkdownReportPath",
            str(markdown_report),
        ],
        cwd=REPO_ROOT,
        env=env,
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
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["decision"].startswith("no_go")
    assert report["ready_for_bounded_local_pilot"] in {True, False}
    assert report["blocked_for_larger_paper_grade_stage"] is True
    assert "lora_qlora_planning" in report
    assert report["lora_qlora_planning"]["qlora_safe_to_run_now"] is False
    assert "paper-grade empirical claims" in report["no_go_for"]
    assert (
        "bounded local SmolVLA pilot tasks inside standing approval" in report["go_for"]
        or report["all_safe_smokes_passed"] is False
    )
    assert json_report.exists()
    assert markdown_report.exists()


def test_go_no_go_status_generator_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the go/no-go status generator")

    env = os.environ.copy()
    env["ALLOW_DOWNLOADS"] = "1"
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
            str(tmp_path / "go_no_go_status_report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "go_no_go_status_report.md"),
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
    assert "dangerous gates" in (result.stdout + result.stderr)
