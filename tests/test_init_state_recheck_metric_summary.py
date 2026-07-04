import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "103_generate_init_state_recheck_metric_summary.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for init-state metric summary tests")
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
        "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_rollout_report(path, *, passed_field, uses_hdf5=False, steps=3, reward=0.0, success=False):
    payload = {
        passed_field: True,
        "policy": {
            "action_adapter_strategy": "policy_6d_delta_pose_plus_gripper_close" if uses_hdf5 else "policy_6d_delta_pose_plus_gripper_zero_hold",
            "hdf5_init_state_set_in_environment": uses_hdf5,
        },
        "rollout_result": {
            "result": {"passed": True, "tasks_completed": 1, "total_steps_performed": steps},
            "tasks": [
                {
                    "task_name": "task_a",
                    "success_check": success,
                    "reward_sum": reward,
                    "done_seen": False,
                    "steps_performed": steps,
                    "policy_calls": steps,
                    "last_env_action_preview": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0 if uses_hdf5 else 0.0],
                    "last_action_adapter_metadata": {
                        "strategy": "policy_6d_delta_pose_plus_gripper_close" if uses_hdf5 else "policy_6d_delta_pose_plus_gripper_zero_hold"
                    },
                    "error": None,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path):
    init_report = tmp_path / "init.json"
    tiny_report = tmp_path / "tiny.json"
    reduced_report = tmp_path / "reduced.json"
    _write_rollout_report(init_report, passed_field="bounded_init_state_learned_policy_recheck_passed", uses_hdf5=True, steps=3)
    _write_rollout_report(tiny_report, passed_field="tiny_learned_policy_rollout_passed", uses_hdf5=False, steps=3)
    _write_rollout_report(reduced_report, passed_field="bounded_reduced_scope_learned_policy_rollout_passed", uses_hdf5=False, steps=10)
    return init_report, tiny_report, reduced_report


def _run_summary(tmp_path, *, extra_env=None, missing_init=False):
    init_report, tiny_report, reduced_report = _make_inputs(tmp_path)
    if missing_init:
        init_report.unlink()
    json_report = tmp_path / "summary.json"
    md_report = tmp_path / "summary.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-InitStateReportPath",
            str(init_report),
            "-TinyResetReportPath",
            str(tiny_report),
            "-ReducedResetReportPath",
            str(reduced_report),
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


def test_init_state_metric_summary_blocks_scaling_when_reward_stays_zero(tmp_path):
    result, report, json_report, md_report = _run_summary(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["init_state_recheck_metric_summary_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["metric_summary"]["positive_diagnostic_signal_found"] is False
    assert report["metric_summary"]["init_state_reward_sum"] == 0.0
    assert report["metric_summary"]["init_state_diagnostic_success"] is False
    assert report["metric_summary"]["hdf5_init_state_set_in_environment"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert len(report["metric_summary"]["scenarios"]) == 3
    assert json_report.exists()
    assert md_report.exists()


def test_init_state_metric_summary_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_summary(
        tmp_path,
        extra_env={"ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["init_state_recheck_metric_summary_passed"] is False
    assert "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK" in report["reason"]


def test_init_state_metric_summary_handles_missing_init_report(tmp_path):
    result, report, json_report, md_report = _run_summary(tmp_path, missing_init=True)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["init_state_recheck_metric_summary_passed"] is False
    assert "Missing or unreadable init-state recheck report" in report["reason"]
    assert json_report.exists()
    assert md_report.exists()
