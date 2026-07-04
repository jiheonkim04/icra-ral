import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "83_bounded_adapter_strategy_diagnostic.ps1"
MODULE = REPO_ROOT / "tca_map" / "smolvla" / "libero_learned_policy_rollout.py"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for adapter strategy diagnostic runner tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_ROLLOUTS",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _run_runner(tmp_path, extra_env=None, extra_args=None):
    json_report = tmp_path / "run.json"
    md_report = tmp_path / "run.md"
    args = [
        _powershell(),
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-JsonReportPath",
        str(json_report),
        "-MarkdownReportPath",
        str(md_report),
    ]
    if extra_args:
        args.extend(extra_args)
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_adapter_strategy_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["adapter_strategy_diagnostic_passed"] is False
    assert "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC=1" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_adapter_strategy_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_adapter_strategy_runner_refuses_oversized_variant_args(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={"ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC": "1"},
        extra_args=["-MaxStepsPerVariant", "11"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "1..3 strategy variants" in report["reason"]


def test_policy_module_accepts_adapter_strategy_gate_and_cli_arg():
    text = MODULE.read_text(encoding="utf-8")

    assert "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC" in text
    assert "--action-adapter-strategy" in text
    assert "ACTION_STRATEGY_GRIPPER_OPEN" in text
    assert "ACTION_STRATEGY_GRIPPER_CLOSE" in text
