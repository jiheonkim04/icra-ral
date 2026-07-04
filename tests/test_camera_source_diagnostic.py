import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla import libero_learned_policy_rollout as rollout


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "88_plan_camera_source_diagnostic.ps1"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "89_bounded_camera_source_diagnostic.ps1"
MODULE = REPO_ROOT / "tca_map" / "smolvla" / "libero_learned_policy_rollout.py"


class _Feature:
    shape = (3, 8, 8)


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for camera-source diagnostic tests")
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
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_camera_alias_strategies_select_expected_sources():
    agentview = np.full((4, 4, 3), 255, dtype=np.uint8)
    eye = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = {"agentview_image": agentview, "robot0_eye_in_hand_image": eye}

    current = rollout._camera_aliases("current_aliases")
    _, source, metadata = rollout._image_tensor(obs, "observation.images.camera3", _Feature(), "cpu", aliases=current)
    assert source == "agentview_image"
    assert metadata["source_key"] == "agentview_image"

    camera3_eye = rollout._camera_aliases("camera3_eye_in_hand")
    _, source, metadata = rollout._image_tensor(obs, "observation.images.camera3", _Feature(), "cpu", aliases=camera3_eye)
    assert source == "robot0_eye_in_hand_image"
    assert metadata["source_key"] == "robot0_eye_in_hand_image"

    all_agent = rollout._camera_aliases("all_agentview")
    _, source, metadata = rollout._image_tensor(obs, "observation.images.camera2", _Feature(), "cpu", aliases=all_agent)
    assert source == "agentview_image"
    assert metadata["source_key"] == "agentview_image"

    with pytest.raises(ValueError, match="unsupported camera alias strategy"):
        rollout._camera_aliases("bad_camera")


def _run_plan(tmp_path, *, source_has_camera=True, extra_env=None):
    prompt_report = tmp_path / "prompt.json"
    source = tmp_path / "rollout.py"
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    _write_json(
        prompt_report,
        {
            "prompt_format_diagnostic_passed": True,
            "result": {
                "variants_completed": 3,
                "best_prompt_strategy": "bddl_language",
                "best_diagnostic_success_rate": 0.0,
                "best_reward_sum": 0.0,
            },
        },
    )
    source.write_text(
        (
            "parser.add_argument('--camera-alias-strategy')\n"
            "CAMERA_ALIAS_STRATEGY_CAMERA3_EYE_IN_HAND = 'camera3_eye_in_hand'\n"
            "def _camera_aliases(strategy): pass\n"
            "_build_batch(config, tokenizer_root, obs, task_language, args.device, args.camera_alias_strategy)\n"
        )
        if source_has_camera
        else "_build_batch(config, tokenizer_root, obs, task_language, args.device)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PLAN_SCRIPT),
            "-PromptFormatReportPath",
            str(prompt_report),
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


def test_camera_source_plan_goes_green_with_prompt_result(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["camera_source_diagnostic_plan_passed"] is True
    assert report["ready_for_camera_source_diagnostic_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["diagnostic_plan"]["camera_alias_strategy_variants"] == [
        "current_aliases",
        "camera3_eye_in_hand",
        "all_agentview",
    ]
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_camera_source_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_CAMERA_SOURCE_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_CAMERA_SOURCE_DIAGNOSTIC" in reason for reason in report["stop_reasons"])


def test_camera_source_plan_requires_source_hook(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, source_has_camera=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("camera-alias-strategy CLI" in reason for reason in report["stop_reasons"])


def test_camera_source_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["camera_source_diagnostic_passed"] is False
    assert "ALLOW_CAMERA_SOURCE_DIAGNOSTIC=1" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_camera_source_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_CAMERA_SOURCE_DIAGNOSTIC": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_policy_module_accepts_camera_source_gate_and_cli_arg():
    text = MODULE.read_text(encoding="utf-8")

    assert "ALLOW_CAMERA_SOURCE_DIAGNOSTIC" in text
    assert "--camera-alias-strategy" in text
    assert "CAMERA_ALIAS_STRATEGY_CAMERA3_EYE_IN_HAND" in text
    assert "_camera_aliases" in text
