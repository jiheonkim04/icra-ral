import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.features.cache import write_dummy_feature_cache
from tca_map.features.cached_eval import evaluate_feature_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "25_eval_feature_cache_smoke.ps1"


def test_evaluate_feature_cache_from_dummy_records(tmp_path):
    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "eval_report.json"
    write_dummy_feature_cache(cache_dir, max_samples=4)

    report = evaluate_feature_cache(cache_dir=cache_dir, report_path=report_path)
    assert report["cache_valid"] is True
    assert report["metrics"]["mode"] == "feature_cache_eval_smoke"
    assert report["metrics"]["cache_record_count"] == 4
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report_path.exists()


def _run_script(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the feature-cache eval smoke")

    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "feature_cache_eval_report.json"
    env = os.environ.copy()
    env.update(extra_env or {})
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
            str(cache_dir),
            "-ReportPath",
            str(report_path),
            "-PrepareDummyCache",
        ],
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


def test_feature_cache_eval_script_is_eval_only(tmp_path):
    result = _run_script(tmp_path)
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["cache_valid"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_feature_cache_eval_script_refuses_dangerous_gates(tmp_path):
    result = _run_script(tmp_path, {"ALLOW_HEAVY_IMPORT": "1"})
    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
