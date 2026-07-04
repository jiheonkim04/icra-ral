import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "108_plan_vlm_loading_policy_action_normalization_audit.ps1"
PYTHON = r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for VLM/action-normalization audit tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path, *, include_config=True):
    ckpt = tmp_path / "smolvla"
    if include_config:
        _write_json(
            ckpt / "config.json",
            {
                "type": "smolvla",
                "repo_id": None,
                "license": None,
                "push_to_hub": True,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "load_vlm_weights": True,
                "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            },
        )
    _write_json(
        ckpt / "policy_preprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "tokenizer_processor",
                    "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                },
                {
                    "registry_name": "normalizer_processor",
                    "config": {
                        "norm_map": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                        "features": {
                            "observation.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "observation.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "observation.image3": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "action": {"type": "ACTION", "shape": [6]},
                        },
                    },
                    "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
                },
            ]
        },
    )
    _write_json(
        ckpt / "policy_postprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {
                        "norm_map": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                        "features": {"action": {"type": "ACTION", "shape": [6]}},
                    },
                    "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                }
            ]
        },
    )
    offline = tmp_path / "offline.json"
    summary = tmp_path / "summary.json"
    load_only = tmp_path / "load_only.json"
    _write_json(
        offline,
        {
            "offline_demo_action_decoding_passed": True,
            "policy": {"model_inference_performed": True},
            "files": {
                "external_tokenizer_dependency": {
                    "found": True,
                    "files_found": ["tokenizer.json", "processor_config.json", "config.json"],
                }
            },
            "sample": {"expert_action_shape": [7], "task": "turn on the stove"},
            "metrics": {
                "load_vlm_weights": False,
                "action_l1_to_expert": 0.375016,
                "action_mse_to_expert": 0.227689,
                "policy6_l1_to_expert_first6": 0.570848,
                "policy_action_shape": [6],
                "expert_action_shape": [7],
                "policy_action_preview": [-0.35, -0.06, 0.38, 1.8, 0.44, 0.31],
                "adapted_action_preview": [-0.35, -0.06, 0.38, 1.0, 0.44, 0.31, -1.0],
                "expert_action_preview": [0.0, 0.05625, -0.01875, 0.0, 0.0, 0.0, -1.0],
                "action_adapter_metadata": {
                    "strategy": "policy_6d_delta_pose_plus_gripper_close",
                    "gripper_value": -1.0,
                    "clipped_values": 1,
                },
                "batch_metadata": {
                    "image_sources": {
                        "observation.images.camera1": "agentview_image",
                        "observation.images.camera2": "robot0_eye_in_hand_image",
                        "observation.images.camera3": "agentview_image",
                    },
                    "image_adapters": {
                        "observation.images.camera1": {"resized": True},
                        "observation.images.camera2": {"resized": True},
                        "observation.images.camera3": {"resized": True},
                    },
                },
            },
        },
    )
    _write_json(
        summary,
        {
            "metrics": {
                "action_l1_to_expert": 0.375016,
                "action_mse_to_expert": 0.227689,
                "policy6_l1_to_expert_first6": 0.570848,
                "offline_alignment_signal": "weak",
            }
        },
    )
    _write_json(load_only, {"load": {"load_vlm_weights": False}})
    return ckpt, offline, summary, load_only


def _run_audit(tmp_path, *, include_config=True, extra_env=None):
    ckpt, offline, summary, load_only = _make_inputs(tmp_path, include_config=include_config)
    json_report = tmp_path / "audit.json"
    md_report = tmp_path / "audit.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            PYTHON,
            "-SmolVLACkpt",
            str(ckpt),
            "-OfflineDecodingReportPath",
            str(offline),
            "-OfflineSummaryPath",
            str(summary),
            "-LoadOnlyReportPath",
            str(load_only),
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


def test_vlm_action_normalization_audit_blocks_rollout_scaling(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["vlm_loading_policy_action_normalization_audit_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert report["ready_for_repeated_offline_decoding_plan"] is True
    assert report["checkpoint_summary"]["config_load_vlm_weights"] is True
    assert report["checkpoint_summary"]["observed_load_vlm_weights"] is False
    assert report["offline_alignment_summary"]["offline_alignment_signal"] == "weak"
    assert any(issue["axis"] == "vlm_loading_policy" for issue in report["issues"])
    assert any(issue["axis"] == "action_normalization" for issue in report["issues"])
    assert any(issue["axis"] == "action_adapter_clipping" for issue in report["issues"])
    assert report["policy"]["model_load_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_vlm_action_normalization_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(
        tmp_path,
        extra_env={"ALLOW_OFFLINE_DEMO_ACTION_DECODING": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["vlm_loading_policy_action_normalization_audit_passed"] is False
    assert any("ALLOW_OFFLINE_DEMO_ACTION_DECODING" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_inference_performed"] is False


def test_vlm_action_normalization_audit_stops_when_config_missing(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path, include_config=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["vlm_loading_policy_action_normalization_audit_passed"] is False
    assert any("config.json" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
