import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "92_generate_learned_policy_diagnostic_synthesis.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for diagnostic synthesis tests")
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
        "ALLOW_PROMPT_FORMAT_DIAGNOSTIC",
        "ALLOW_CAMERA_SOURCE_DIAGNOSTIC",
        "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_reports(tmp_path, *, include_state=True, reward=0.0, success=0.0):
    zero = tmp_path / "zero.json"
    adapter = tmp_path / "adapter.json"
    scale = tmp_path / "scale.json"
    prompt = tmp_path / "prompt.json"
    camera = tmp_path / "camera.json"
    state = tmp_path / "state.json"

    _write_json(
        zero,
        {
            "zero_action_policy_diagnostic_comparison_passed": True,
            "ready_for_rollout_scaling": False,
            "comparison": {
                "zero_action": {"reward_sum": 0.0, "diagnostic_success_rate": 0.0},
                "learned_policy": {"reward_sum": reward, "diagnostic_success_rate": success},
            },
        },
    )
    _write_json(
        adapter,
        {
            "adapter_strategy_diagnostic_passed": True,
            "ready_for_rollout_scaling": False,
            "result": {
                "variants_completed": 3,
                "best_strategy": "policy_6d_delta_pose_plus_gripper_close",
                "best_diagnostic_success_rate": success,
                "best_reward_sum": reward,
            },
        },
    )
    _write_json(
        scale,
        {
            "action_scale_diagnostic_passed": True,
            "ready_for_rollout_scaling": False,
            "result": {
                "variants_completed": 3,
                "best_action_scale": 1.0,
                "best_diagnostic_success_rate": success,
                "best_reward_sum": reward,
            },
        },
    )
    _write_json(
        prompt,
        {
            "prompt_format_diagnostic_passed": True,
            "ready_for_rollout_scaling": False,
            "result": {
                "variants_completed": 3,
                "best_prompt_strategy": "bddl_language",
                "best_diagnostic_success_rate": success,
                "best_reward_sum": reward,
            },
        },
    )
    _write_json(
        camera,
        {
            "camera_source_diagnostic_passed": True,
            "ready_for_rollout_scaling": False,
            "result": {
                "variants_completed": 3,
                "best_camera_alias_strategy": "current_aliases",
                "best_diagnostic_success_rate": success,
                "best_reward_sum": reward,
            },
        },
    )
    if include_state:
        _write_json(
            state,
            {
                "state_sufficiency_diagnostic_passed": True,
                "ready_for_rollout_scaling": False,
                "result": {
                    "variants_completed": 3,
                    "best_state_adapter_strategy": "eef_pos_zero_rot",
                    "best_diagnostic_success_rate": success,
                    "best_reward_sum": reward,
                },
            },
        )
    return {
        "zero": zero,
        "adapter": adapter,
        "scale": scale,
        "prompt": prompt,
        "camera": camera,
        "state": state,
    }


def _run_synthesis(tmp_path, *, extra_env=None, include_state=True, reward=0.0, success=0.0):
    reports = _make_reports(tmp_path, include_state=include_state, reward=reward, success=success)
    json_report = tmp_path / "synthesis.json"
    md_report = tmp_path / "synthesis.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ZeroActionComparisonPath",
            str(reports["zero"]),
            "-AdapterStrategyReportPath",
            str(reports["adapter"]),
            "-ActionScaleReportPath",
            str(reports["scale"]),
            "-PromptFormatReportPath",
            str(reports["prompt"]),
            "-CameraSourceReportPath",
            str(reports["camera"]),
            "-StateSufficiencyReportPath",
            str(reports["state"]),
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


def test_synthesis_marks_zero_reward_ladder_no_go(tmp_path):
    result, report, json_report, md_report = _run_synthesis(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["learned_policy_diagnostic_synthesis_passed"] is True
    assert report["diagnostic_ladder_complete"] is True
    assert report["positive_diagnostic_signal_found"] is False
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert "nonzero reward" in report["no_go_for_rollout_scaling_reason"]
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_synthesis_stops_when_required_report_missing(tmp_path):
    result, report, _, _ = _run_synthesis(tmp_path, include_state=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["learned_policy_diagnostic_synthesis_passed"] is False
    assert any("Missing diagnostic report" in reason for reason in report["stop_reasons"])
    assert report["ready_for_rollout_scaling"] is False


def test_synthesis_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_synthesis(
        tmp_path,
        extra_env={"ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_load_performed"] is False
