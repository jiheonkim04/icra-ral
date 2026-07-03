import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.adapters.tiny_lora_smoke import (
    MAX_TINY_LORA_STEPS,
    TinyLoraSmokeError,
    run_tiny_lora_smoke,
    validate_smoke_bounds,
)
from tca_map.features.cache import write_dummy_feature_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "37_tiny_lora_smoke.ps1"


def test_tiny_lora_smoke_trains_dummy_cached_adapters(tmp_path):
    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "tiny_lora_smoke_report.json"
    write_dummy_feature_cache(cache_dir, max_samples=4)

    report = run_tiny_lora_smoke(
        cache_dir=cache_dir,
        report_path=report_path,
        max_steps=4,
        max_samples=4,
        rank=2,
        prepare_dummy_cache=False,
        require_training_gate=False,
    )

    assert report["tiny_lora_smoke_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["trainable_lora_adapter_weights_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["not_standard_success"] is True
    assert report["policy"]["not_paper_grade"] is True
    assert {arm["arm"] for arm in report["arms"]} == {
        "actionmap_lora",
        "tca_map_lora",
        "tca_map_lora_distributional_select",
    }
    assert all(arm["max_steps"] <= MAX_TINY_LORA_STEPS for arm in report["arms"])
    assert all(arm["trainable_lora_parameter_count"] > 0 for arm in report["arms"])
    assert report_path.exists()


def test_tiny_lora_smoke_rejects_too_many_steps():
    with pytest.raises(TinyLoraSmokeError):
        validate_smoke_bounds(max_steps=MAX_TINY_LORA_STEPS + 1, max_runtime_seconds=900, max_samples=4, rank=4)


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


def _run_script(tmp_path, extra_env=None, max_steps="3"):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the tiny LoRA smoke script")

    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "tiny_lora_smoke_report.json"
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
            "-MaxSamples",
            "4",
            "-Rank",
            "2",
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


def test_tiny_lora_smoke_script_requires_training_gate(tmp_path):
    result = _run_script(tmp_path)
    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_tiny_lora_smoke_script_runs_with_bounded_gate(tmp_path):
    result = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["tiny_lora_smoke_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_tiny_lora_smoke_script_refuses_dangerous_gate(tmp_path):
    result = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1", "ALLOW_HEAVY_IMPORT": "1"})
    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
