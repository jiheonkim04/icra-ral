import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "84_plan_action_scale_diagnostic.ps1"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "85_bounded_action_scale_diagnostic.ps1"
MODULE = REPO_ROOT / "tca_map" / "smolvla" / "libero_learned_policy_rollout.py"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for action-scale diagnostic tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
        "ALLOW_ACTION_SCALE_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_plan(tmp_path, *, source_has_action_scale=True, extra_env=None):
    strategy_report = tmp_path / "adapter_strategy.json"
    source = tmp_path / "rollout.py"
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    _write_json(
        strategy_report,
        {
            "adapter_strategy_diagnostic_passed": True,
            "result": {
                "variants_completed": 3,
                "best_diagnostic_success_rate": 0.0,
                "best_reward_sum": 0.0,
            },
        },
    )
    source.write_text(
        (
            "parser.add_argument('--action-scale')\n"
            "adapt_policy_action_to_env_action(policy_action, action_dim, action_scale=args.action_scale)\n"
        )
        if source_has_action_scale
        else "adapt_policy_action_to_env_action(policy_action, action_dim)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PLAN_SCRIPT),
            "-AdapterStrategyReportPath",
            str(strategy_report),
            "-RolloutBridgeSourcePath",
            str(source),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
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


def _run_runner(tmp_path, extra_env=None, extra_args=None):
    json_report = tmp_path / "run.json"
    md_report = tmp_path / "run.md"
    args = [
        _powershell(),
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUNNER_SCRIPT),
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


def test_action_scale_plan_goes_green_with_strategy_result(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["action_scale_diagnostic_plan_passed"] is True
    assert report["ready_for_action_scale_diagnostic_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["diagnostic_plan"]["action_scale_variants"] == [0.25, 0.5, 1.0]
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_action_scale_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_ACTION_SCALE_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_ACTION_SCALE_DIAGNOSTIC" in reason for reason in report["stop_reasons"])


def test_action_scale_plan_requires_source_hook(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, source_has_action_scale=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("action-scale CLI" in reason for reason in report["stop_reasons"])


def test_action_scale_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["action_scale_diagnostic_passed"] is False
    assert "ALLOW_ACTION_SCALE_DIAGNOSTIC=1" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_action_scale_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_ACTION_SCALE_DIAGNOSTIC": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_action_scale_runner_refuses_out_of_budget_scale_args(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={"ALLOW_ACTION_SCALE_DIAGNOSTIC": "1"},
        extra_args=["-ActionScales", "3.0"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "finite and in the range" in report["reason"]


def test_policy_module_accepts_action_scale_gate_and_cli_arg():
    text = MODULE.read_text(encoding="utf-8")

    assert "ALLOW_ACTION_SCALE_DIAGNOSTIC" in text
    assert "--action-scale" in text
    assert "action_scale=args.action_scale" in text
