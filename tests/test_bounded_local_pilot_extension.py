import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "44_bounded_local_pilot_extension.ps1"


def _script_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
        "ALLOW_TINY_TRAINING",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _run_script(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the bounded local pilot extension")

    return subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-CacheDir",
            str(tmp_path / "cache"),
            "-HeadOnlyReportPath",
            str(tmp_path / "bounded_head_only_extension_report.json"),
            "-JsonReportPath",
            str(tmp_path / "bounded_local_pilot_extension_report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "bounded_local_pilot_extension_report.md"),
            "-MaxSteps",
            "5",
            "-PrepareDummyCache",
        ],
        cwd=REPO_ROOT,
        env=_script_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_bounded_local_pilot_extension_requires_training_gate(tmp_path):
    result = _run_script(tmp_path)

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_bounded_local_pilot_extension_runs_with_bounded_gate(tmp_path):
    result = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["bounded_local_pilot_extension_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["not_paper_grade"] is True
    assert report["bounds"]["local_policy_max_steps"] == 300
    assert report["bounds"]["runner_max_steps_cap"] == 100
    assert {head["head"] for head in report["heads"]} == {"actionmap", "tca_map"}
