import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "34_plan_lora_comparison.ps1"


def _clean_env(extra=None):
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
    env.update(extra or {})
    return env


def _run_script(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the LoRA comparison planner")

    report_path = tmp_path / "lora_comparison_plan_report.json"
    return subprocess.run(
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
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ), report_path


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_lora_comparison_plan_is_planning_only(tmp_path):
    result, report_path = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["planning_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["adapter_construction_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["privileged_inference_allowed"] is False
    assert report["lora_comparison_plan_ready"] is True
    assert report["safe_to_run_lora_comparison_now"] is False
    assert any(arm["name"] == "ActionMap + LoRA" for arm in report["comparison_arms"])
    assert any(arm["name"] == "TCA-Map + LoRA + Distributional TCA-Select" for arm in report["comparison_arms"])
    assert any("TCA-Map + LoRA vs ActionMap + LoRA" in item["name"] for item in report["required_comparisons"])
    assert report_path.exists()


def test_lora_comparison_plan_refuses_execution_gate(tmp_path):
    result, _ = _run_script(tmp_path, {"ALLOW_GPU_TRAINING": "1"})

    assert result.returncode == 20
    assert "execution gates" in (result.stdout + result.stderr)
