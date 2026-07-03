import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "33_plan_lora_tiny_smoke.ps1"


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
        pytest.skip("PowerShell is required for the LoRA tiny smoke scaffold planner")

    report_path = tmp_path / "lora_tiny_smoke_scaffold_report.json"
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


def test_lora_tiny_smoke_scaffold_is_planning_only(tmp_path):
    result, report_path = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["planning_only"] is True
    assert report["policy"]["scaffold_only"] is True
    assert report["policy"]["required_lora_track"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["adapter_construction_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["bounds"]["max_steps_cap"] == 100
    assert report["bounds"]["max_runtime_cap_seconds"] == 900
    assert report["lora_tiny_smoke_scaffold_ready"] is True
    assert report["safe_to_execute_lora_tiny_smoke_now"] is False
    assert "ActionMap + LoRA" in report["required_experiment_tracks"]
    assert report_path.exists()


def test_lora_tiny_smoke_scaffold_refuses_execution_gate(tmp_path):
    result, _ = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})

    assert result.returncode == 20
    assert "execution gates" in (result.stdout + result.stderr)
