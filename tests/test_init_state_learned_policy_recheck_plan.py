import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "101_plan_init_state_learned_policy_recheck.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for init-state recheck planner tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_ROLLOUTS",
        "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_DOWNLOADS",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_reports(tmp_path, *, hdf5_passed=True, set_init_state_ok=True):
    hdf5 = tmp_path / "hdf5_replay.json"
    policy = tmp_path / "policy_ready.json"
    single = tmp_path / "single_action.json"
    reduced = tmp_path / "reduced.json"
    _write_json(
        hdf5,
        {
            "bounded_hdf5_initial_state_replay_passed": hdf5_passed,
            "ready_for_learned_policy_rollout_recheck": hdf5_passed,
            "policy": {"task_suite": "libero_10", "task_id": 0},
            "replay_result": {
                "set_init_state_ok": set_init_state_ok,
                "steps_performed": 1,
                "hdf5_path": "/mnt/c/assets/data/libero/libero_10/demo.hdf5",
            },
        },
    )
    _write_json(policy, {"ready_for_tiny_learned_policy_rollout_execution": True})
    _write_json(single, {"wsl_smolvla_single_action_smoke_passed": True})
    _write_json(
        reduced,
        {
            "bounded_reduced_scope_learned_policy_rollout_passed": True,
            "rollout_result": {"result": {"reward_sum": 0.0, "diagnostic_success_rate": 0.0}},
        },
    )
    return hdf5, policy, single, reduced


def _run_plan(tmp_path, *, extra_env=None, extra_args=None, **report_kwargs):
    hdf5, policy, single, reduced = _make_reports(tmp_path, **report_kwargs)
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    args = [
        _powershell(),
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-Hdf5ReplayReportPath",
        str(hdf5),
        "-PolicyReadinessReportPath",
        str(policy),
        "-SingleActionReportPath",
        str(single),
        "-ReducedScopeReportPath",
        str(reduced),
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


def test_init_state_learned_policy_recheck_plan_goes_green(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_init_state_learned_policy_recheck_runner"] is True
    assert report["prerequisites"]["hdf5_replay"]["set_init_state_ok"] is True
    assert report["risk_assessment"]["learned_policy_inference_will_run"] is True
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["ready_for_paper_claim"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_init_state_recheck_plan_requires_hdf5_replay_success(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, hdf5_passed=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_bounded_init_state_learned_policy_recheck_runner"] is False
    assert any("HDF5 initial-state replay has not passed" in reason for reason in report["stop_reasons"])


def test_init_state_recheck_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_inference_performed"] is False


def test_init_state_recheck_plan_enforces_tiny_bounds(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, extra_args=["-TaskCount", "2", "-MaxStepsPerTask", "6"])

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("exactly one task" in reason for reason in report["stop_reasons"])
    assert any("1..5 steps" in reason for reason in report["stop_reasons"])
