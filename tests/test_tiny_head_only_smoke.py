import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.features.cache import write_dummy_feature_cache
from tca_map.features.tiny_head_only_smoke import (
    MAX_TINY_SMOKE_STEPS,
    TinyHeadOnlySmokeError,
    run_tiny_head_only_smoke,
    validate_smoke_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "29_tiny_head_only_smoke.ps1"


def test_tiny_head_only_smoke_trains_dummy_cached_heads(tmp_path):
    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "tiny_head_only_smoke_report.json"
    write_dummy_feature_cache(cache_dir, max_samples=4)

    report = run_tiny_head_only_smoke(
        cache_dir=cache_dir,
        report_path=report_path,
        max_steps=4,
        prepare_dummy_cache=False,
        require_training_gate=False,
    )

    assert report["tiny_head_only_smoke_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["safe_to_run_real_pilot"] is False
    assert {head["head"] for head in report["heads"]} == {"actionmap", "tca_map"}
    assert all(head["max_steps"] <= MAX_TINY_SMOKE_STEPS for head in report["heads"])
    assert report_path.exists()


def test_tiny_head_only_smoke_rejects_too_many_steps():
    with pytest.raises(TinyHeadOnlySmokeError):
        validate_smoke_bounds(max_steps=MAX_TINY_SMOKE_STEPS + 1, max_runtime_seconds=900)


def _script_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_TINY_TRAINING",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _run_script(tmp_path, extra_env=None, max_steps="3"):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the tiny head-only smoke script")

    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "tiny_head_only_smoke_report.json"
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
            "-MaxSteps",
            max_steps,
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


def test_tiny_head_only_smoke_script_requires_training_gate(tmp_path):
    result = _run_script(tmp_path)
    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_tiny_head_only_smoke_script_runs_with_bounded_gate(tmp_path):
    result = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["tiny_head_only_smoke_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
