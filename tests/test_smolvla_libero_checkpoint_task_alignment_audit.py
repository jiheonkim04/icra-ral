import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "104_audit_smolvla_libero_checkpoint_task_alignment.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for checkpoint-task alignment audit tests")
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
        "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
        "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_SIMULATOR_IMPORT_SMOKE",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
        "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path, *, include_config=True):
    smolvla = tmp_path / "smolvla"
    libero = tmp_path / "LIBERO"
    libero_data = tmp_path / "data" / "libero"
    reports = tmp_path / "reports"
    if include_config:
        _write_json(
            smolvla / "config.json",
            {
                "type": "smolvla",
                "repo_id": None,
                "license": None,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "load_vlm_weights": True,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
            },
        )
    _write_json(
        smolvla / "policy_preprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "tokenizer_processor",
                    "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                },
                {
                    "registry_name": "normalizer_processor",
                    "config": {
                        "features": {
                            "observation.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "observation.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "observation.image3": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "action": {"type": "ACTION", "shape": [6]},
                        }
                    },
                },
            ]
        },
    )
    bddl = libero / "libero" / "libero" / "bddl_files" / "libero_10" / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl"
    bddl.parent.mkdir(parents=True, exist_ok=True)
    bddl.write_text("(:language turn on the stove and put the moka pot on it)\n", encoding="utf-8")
    _write_json(
        reports / "init.json",
        {
            "decision": "no_go_rollout_scaling",
            "metric_summary": {
                "positive_diagnostic_signal_found": False,
                "hdf5_init_state_set_in_environment": True,
                "init_state_vs_reset_3_step_reward_delta": 0.0,
                "scenarios": [
                    {
                        "task_name": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
                        "reward_sum": 0.0,
                        "diagnostic_success": False,
                    }
                ],
            },
        },
    )
    _write_json(
        reports / "hdf5.json",
        {
            "decision": "no_go_rollout_scaling",
            "hdf5_summary": {
                "action_dim": 7,
                "obs_shapes": {
                    "agentview_rgb": {"shape": [272, 128, 128, 3]},
                    "eye_in_hand_rgb": {"shape": [272, 128, 128, 3]},
                },
            },
            "paths": {
                "hdf5_path": str(
                    libero_data / "libero_10" / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
                )
            },
        },
    )
    _write_json(
        reports / "offline.json",
        {
            "decision": "no_go_rollout_scaling",
            "paths": {
                "hdf5_path": str(
                    libero_data / "libero_10" / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
                )
            },
            "reproduction": {
                "best_action_adapter_strategy_for_first_demo_action": "policy_6d_delta_pose_plus_gripper_close"
            },
        },
    )
    _write_json(reports / "load.json", {"load": {"load_vlm_weights": False}})
    return smolvla, libero, libero_data, reports


def _run_audit(tmp_path, *, include_config=True, extra_env=None):
    smolvla, libero, libero_data, reports = _make_inputs(tmp_path, include_config=include_config)
    json_report = tmp_path / "alignment.json"
    md_report = tmp_path / "alignment.md"
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
            "-InitStateSummaryPath",
            str(reports / "init.json"),
            "-Hdf5AuditPath",
            str(reports / "hdf5.json"),
            "-OfflineAdapterReportPath",
            str(reports / "offline.json"),
            "-LoadOnlyReportPath",
            str(reports / "load.json"),
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


def test_checkpoint_task_alignment_audit_blocks_rollout_scaling(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["smolvla_libero_checkpoint_task_alignment_audit_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert report["ready_for_offline_demonstration_conditioned_action_decoding_plan"] is True
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["task_summary"]["task_matches_bddl_filename"] is True
    assert any(issue["axis"] == "checkpoint_provenance" for issue in report["issues"])
    assert any(issue["axis"] == "vlm_loading_policy" for issue in report["issues"])
    assert any(issue["axis"] == "action_decoding_convention" for issue in report["issues"])
    assert json_report.exists()
    assert md_report.exists()


def test_checkpoint_task_alignment_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(
        tmp_path,
        extra_env={"ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["smolvla_libero_checkpoint_task_alignment_audit_passed"] is False
    assert any("ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK" in reason for reason in report["stop_reasons"])
    assert report["policy"]["rollouts_performed"] is False


def test_checkpoint_task_alignment_audit_stops_when_config_missing(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path, include_config=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["smolvla_libero_checkpoint_task_alignment_audit_passed"] is False
    assert any("config.json" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
