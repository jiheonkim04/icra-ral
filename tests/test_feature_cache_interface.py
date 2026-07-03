import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.features.cache import FEATURE_CACHE_SCHEMA_VERSION, validate_feature_cache, write_dummy_feature_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "19_plan_feature_cache.ps1"


def test_dummy_feature_cache_round_trip(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = write_dummy_feature_cache(output_dir=output_dir, max_samples=3)
    validation = validate_feature_cache(output_dir)

    assert manifest["schema_version"] == FEATURE_CACHE_SCHEMA_VERSION
    assert manifest["record_count"] == 3
    assert validation["valid"] is True
    assert validation["record_count"] == 3
    assert manifest["policy"]["downloads_performed"] is False
    assert manifest["policy"]["model_load_performed"] is False
    assert manifest["policy"]["training_performed"] is False


def _run_planner(tmp_path, extra_env=None, write_dummy=False):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the feature-cache planner")

    report_path = tmp_path / "feature_cache_plan_report.json"
    output_dir = tmp_path / "dummy_cache"
    args = [
        powershell,
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-Python",
        sys.executable,
        "-ReportPath",
        str(report_path),
        "-OutputDir",
        str(output_dir),
    ]
    if write_dummy:
        args.append("-WriteDummyCache")

    env = os.environ.copy()
    env.update(extra_env or {})
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
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


def test_feature_cache_planner_is_safe_and_can_write_dummy_cache(tmp_path):
    result = _run_planner(tmp_path, write_dummy=True)
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["dummy_cache_written"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["dummy_validation"]["valid"] is True


def test_feature_cache_planner_refuses_dangerous_gates(tmp_path):
    result = _run_planner(tmp_path, extra_env={"ALLOW_HEAVY_IMPORT": "1"})
    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
