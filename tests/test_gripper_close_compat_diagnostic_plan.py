import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "96_plan_gripper_close_compat_diagnostic.ps1"
CLOSE = "policy_6d_delta_pose_plus_gripper_close"
ZERO_HOLD = "policy_6d_delta_pose_plus_gripper_zero_hold"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for gripper-close compatibility planner tests")
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
        "ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path, *, best_strategy=CLOSE, include_previous=False, previous_zero=True):
    repro = tmp_path / "repro.json"
    previous = tmp_path / "previous.json"
    source = tmp_path / "rollout.py"
    _write_json(
        repro,
        {
            "offline_adapter_reproduction_check_passed": True,
            "paths": {"hdf5_path": "C:/assets/data/libero/libero_10/demo.hdf5"},
            "reproduction": {
                "best_action_adapter_strategy_for_first_demo_action": best_strategy,
                "action_reproductions": {
                    CLOSE: {"l1_to_demo_first_action": 0.0},
                    ZERO_HOLD: {"l1_to_demo_first_action": 0.142857},
                },
            },
        },
    )
    if include_previous:
        _write_json(
            previous,
            {
                "variants": [
                    {
                        "strategy": CLOSE,
                        "passed": True,
                        "diagnostic_success_rate": 0.0 if previous_zero else 1.0,
                        "reward_sum": 0.0 if previous_zero else 1.0,
                    }
                ]
            },
        )
    source.write_text(
        f'parser.add_argument("--action-adapter-strategy", choices=["{ZERO_HOLD}", "{CLOSE}"])\n',
        encoding="utf-8",
    )
    return repro, previous, source


def _run_plan(tmp_path, *, best_strategy=CLOSE, include_previous=False, previous_zero=True, extra_env=None):
    repro, previous, source = _make_inputs(
        tmp_path,
        best_strategy=best_strategy,
        include_previous=include_previous,
        previous_zero=previous_zero,
    )
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-OfflineReproductionReportPath",
            str(repro),
            "-PreviousAdapterStrategyReportPath",
            str(previous),
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


def test_gripper_close_plan_goes_green_without_prior_duplicate(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["gripper_close_compat_plan_passed"] is True
    assert report["ready_for_gripper_close_compat_diagnostic_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["offline_evidence"]["best_strategy"] == CLOSE
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_gripper_close_plan_reduces_scope_after_duplicate_zero_signal(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, include_previous=True, previous_zero=True)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "reduce_scope"
    assert report["gripper_close_compat_plan_passed"] is True
    assert report["ready_for_gripper_close_compat_diagnostic_runner"] is False
    assert report["previous_diagnostic"]["duplicate_zero_signal"] is True
    assert "Do not rerun the identical" in report["recommended_next_step"]


def test_gripper_close_plan_refuses_wrong_offline_best_strategy(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, best_strategy=ZERO_HOLD)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["gripper_close_compat_plan_passed"] is False
    assert any("did not select the gripper-close" in reason for reason in report["stop_reasons"])


def test_gripper_close_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC" in reason for reason in report["stop_reasons"])
