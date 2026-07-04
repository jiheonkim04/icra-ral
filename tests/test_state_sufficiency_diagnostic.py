import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tca_map.smolvla import libero_learned_policy_rollout as rollout


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "90_plan_state_sufficiency_diagnostic.ps1"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "91_bounded_state_sufficiency_diagnostic.ps1"
MODULE = REPO_ROOT / "tca_map" / "smolvla" / "libero_learned_policy_rollout.py"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for state-sufficiency diagnostic tests")
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


def _obs():
    return {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }


def test_state_adapter_strategies_build_expected_vectors():
    first, first_meta = rollout._state_tensor(_obs(), 6, "cpu", "eef_pos_quat_first3")
    last, last_meta = rollout._state_tensor(_obs(), 6, "cpu", "eef_pos_quat_last3")
    zero, zero_meta = rollout._state_tensor(_obs(), 6, "cpu", "eef_pos_zero_rot")

    assert torch.allclose(first.cpu(), torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]))
    assert torch.allclose(last.cpu(), torch.tensor([[1.0, 2.0, 3.0, 0.2, 0.3, 0.4]]))
    assert torch.allclose(zero.cpu(), torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]))

    assert first_meta["adapter"] == "diagnostic_eef_pos_quat_xyz_6d_state_adapter"
    assert first_meta["state_adapter_strategy"] == "eef_pos_quat_first3"
    assert last_meta["adapter"] == "diagnostic_eef_pos_quat_last3_6d_state_adapter"
    assert zero_meta["adapter"] == "diagnostic_eef_pos_zero_rot_6d_state_adapter"
    assert zero_meta["uses_privileged_state"] is False

    with pytest.raises(ValueError, match="unsupported state adapter strategy"):
        rollout._state_tensor(_obs(), 6, "cpu", "bad_state")


def _run_plan(tmp_path, *, source_has_state=True, extra_env=None):
    camera_report = tmp_path / "camera.json"
    source = tmp_path / "rollout.py"
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    _write_json(
        camera_report,
        {
            "camera_source_diagnostic_passed": True,
            "result": {
                "variants_completed": 3,
                "best_camera_alias_strategy": "current_aliases",
                "best_diagnostic_success_rate": 0.0,
                "best_reward_sum": 0.0,
            },
        },
    )
    source.write_text(
        (
            "parser.add_argument('--state-adapter-strategy')\n"
            "STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_LAST3 = 'eef_pos_quat_last3'\n"
            "STATE_ADAPTER_STRATEGY_EEF_POS_ZERO_ROT = 'eef_pos_zero_rot'\n"
            "_build_batch(config, tokenizer_root, obs, task_language, args.device, args.camera_alias_strategy, args.state_adapter_strategy)\n"
        )
        if source_has_state
        else "_build_batch(config, tokenizer_root, obs, task_language, args.device, args.camera_alias_strategy)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PLAN_SCRIPT),
            "-CameraSourceReportPath",
            str(camera_report),
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


def test_state_sufficiency_plan_goes_green_with_camera_result(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["state_sufficiency_diagnostic_plan_passed"] is True
    assert report["ready_for_state_sufficiency_diagnostic_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["diagnostic_plan"]["state_adapter_strategy_variants"] == [
        "eef_pos_quat_first3",
        "eef_pos_quat_last3",
        "eef_pos_zero_rot",
    ]
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_state_sufficiency_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC" in reason for reason in report["stop_reasons"])


def test_state_sufficiency_plan_requires_source_hook(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, source_has_state=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("state-adapter-strategy CLI" in reason for reason in report["stop_reasons"])


def test_state_sufficiency_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["state_sufficiency_diagnostic_passed"] is False
    assert "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC=1" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_state_sufficiency_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_policy_module_accepts_state_sufficiency_gate_and_cli_arg():
    text = MODULE.read_text(encoding="utf-8")

    assert "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC" in text
    assert "--state-adapter-strategy" in text
    assert "STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_LAST3" in text
    assert "STATE_ADAPTER_STRATEGY_EEF_POS_ZERO_ROT" in text
