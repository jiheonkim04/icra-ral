import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "38_compare_tiny_lora_pilot.ps1"


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


def _arm(name, proxy, l1, target_acc, wrong_target, margin, latency, params):
    return {
        "arm": name,
        "trainable_lora_parameter_count": params,
        "metrics": {
            "offline_standard_proxy": proxy,
            "standard_proxy_score": proxy,
            "action_l1": l1,
            "action_mse": l1 * l1,
            "target_top1_accuracy": target_acc,
            "target_topk_accuracy": target_acc,
            "wrong_target_proxy_rate": wrong_target,
            "counterfactual_separation_margin": margin,
            "nuisance_stability_score": 1.0,
            "latency_ms": latency,
        },
    }


def _write_input(path: Path):
    payload = {
        "tiny_lora_smoke_passed": True,
        "cache_record_count": 4,
        "max_steps": 4,
        "max_steps_cap": 100,
        "max_samples": 4,
        "elapsed_seconds": 0.1,
        "arms": [
            _arm("actionmap_lora", 0.2, 0.5, 0.0, 1.0, 0.0, 1.0, 10),
            _arm("tca_map_lora", 0.7, 0.4, 0.75, 0.25, 0.1, 1.5, 20),
            _arm("tca_map_lora_distributional_select", 0.8, 0.35, 0.75, 0.25, 0.15, 1.8, 20),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_script(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the tiny LoRA comparison")

    input_report = tmp_path / "tiny_lora_smoke_report.json"
    json_report = tmp_path / "tiny_lora_comparison_report.json"
    markdown_report = tmp_path / "tiny_lora_comparison_report.md"
    _write_input(input_report)
    return subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-InputReportPath",
            str(input_report),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(markdown_report),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ), json_report, markdown_report


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_tiny_lora_comparison_is_proxy_only(tmp_path):
    result, json_report, markdown_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["bounded_local_pilot"] is True
    assert report["policy"]["offline_proxy_only"] is True
    assert report["policy"]["not_standard_success"] is True
    assert report["policy"]["not_paper_grade"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["tiny_lora_comparison_passed"] is True
    assert report["comparison"]["offline_standard_proxy_delta_tca_lora_minus_actionmap_lora"] == 0.5
    assert report["comparison"]["wrong_target_proxy_rate_delta_tca_lora_minus_actionmap_lora"] == -0.75
    assert report["comparison"]["offline_standard_proxy_delta_select_minus_tca_lora"] == 0.1
    assert json_report.exists()
    assert markdown_report.exists()


def test_tiny_lora_comparison_refuses_execution_gate(tmp_path):
    result, _, _ = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})

    assert result.returncode == 20
    assert "execution gates" in (result.stdout + result.stderr)
