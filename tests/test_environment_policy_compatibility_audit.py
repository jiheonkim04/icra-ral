import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "93_audit_environment_policy_compatibility.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for compatibility audit tests")
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


def _make_inputs(tmp_path, *, include_synthesis=True):
    smolvla = tmp_path / "smolvla"
    libero = tmp_path / "LIBERO"
    libero_data = tmp_path / "data" / "libero"
    robosuite = tmp_path / "robosuite"
    source = tmp_path / "rollout.py"
    synthesis = tmp_path / "synthesis.json"

    _write_json(
        smolvla / "config.json",
        {
            "type": "smolvla",
            "repo_id": None,
            "license": "apache-2.0",
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            "max_action_dim": 32,
            "chunk_size": 50,
            "n_action_steps": 50,
            "tokenizer_max_length": 48,
        },
    )
    bddl = libero / "libero" / "libero" / "bddl_files" / "libero_10" / "demo_task.bddl"
    bddl.parent.mkdir(parents=True, exist_ok=True)
    bddl.write_text("(:language pick up the mug)\n", encoding="utf-8")
    hdf5 = libero_data / "libero_10" / "demo_task_demo.hdf5"
    hdf5.parent.mkdir(parents=True, exist_ok=True)
    hdf5.write_bytes(b"marker")
    robosuite.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "config.load_vlm_weights = False\n"
        "adapt_policy_action_to_env_action(policy_action, action_dim)\n"
        "DEFAULT_IMAGE_ALIASES = {}\n"
        "def _camera_aliases(strategy): pass\n"
        "STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_LAST3 = 'eef_pos_quat_last3'\n",
        encoding="utf-8",
    )
    if include_synthesis:
        _write_json(
            synthesis,
            {
                "decision": "no_go_rollout_scaling",
                "diagnostic_ladder_complete": True,
                "positive_diagnostic_signal_found": False,
                "ready_for_rollout_scaling": False,
                "no_go_for_rollout_scaling_reason": "no positive signal",
            },
        )
    return smolvla, libero, libero_data, robosuite, source, synthesis


def _run_audit(tmp_path, *, include_synthesis=True, extra_env=None):
    smolvla, libero, libero_data, robosuite, source, synthesis = _make_inputs(
        tmp_path,
        include_synthesis=include_synthesis,
    )
    json_report = tmp_path / "audit.json"
    md_report = tmp_path / "audit.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-SmolVLACkpt",
            str(smolvla),
            "-LiberoRoot",
            str(libero),
            "-LiberoDataRoot",
            str(libero_data),
            "-RobosuiteRoot",
            str(robosuite),
            "-SynthesisReportPath",
            str(synthesis),
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


def test_environment_policy_audit_marks_rollout_scaling_no_go(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["environment_policy_compatibility_audit_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert report["high_severity_issue_count"] >= 3
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert any(issue["axis"] == "vlm_loading_policy" for issue in report["issues"])
    assert any(issue["axis"] == "action_convention" for issue in report["issues"])
    assert json_report.exists()
    assert md_report.exists()


def test_environment_policy_audit_stops_when_synthesis_missing(tmp_path):
    result, report, _, _ = _run_audit(tmp_path, include_synthesis=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["environment_policy_compatibility_audit_passed"] is False
    assert any("Missing JSON file" in reason for reason in report["stop_reasons"])
    assert report["ready_for_rollout_scaling"] is False


def test_environment_policy_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(
        tmp_path,
        extra_env={"ALLOW_CAMERA_SOURCE_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_CAMERA_SOURCE_DIAGNOSTIC" in reason for reason in report["stop_reasons"])
    assert report["policy"]["rollouts_performed"] is False
