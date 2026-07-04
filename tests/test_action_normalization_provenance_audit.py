import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from tca_map.smolvla import action_normalization_provenance_audit as audit


def _write_summary(path: Path):
    path.write_text(
        json.dumps({"vlm_enabled_offline_decoding_summary_passed": True}),
        encoding="utf-8",
    )


def _write_vlm_report(path: Path):
    path.write_text(
        json.dumps(
            {
                "metrics": {"clipped_values_total": 3},
                "samples": [
                    {
                        "expert_action_preview": [0.0, 0.1, -0.1, 0.0, 0.0, 0.0, -1.0],
                        "adapted_action_preview": [-0.2, -0.3, 0.2, 1.0, 0.1, -0.1, -1.0],
                        "policy_action_preview": [-0.2, -0.3, 0.2, 1.8, 0.1, -0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_ckpt(root: Path):
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "normalization_mapping": {"ACTION": "MEAN_STD"},
                "output_features": {"action": {"shape": [6], "type": "ACTION"}},
                "input_features": {"observation.state": {"shape": [6], "type": "STATE"}},
                "load_vlm_weights": True,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "so100.buffer.action.mean": np.asarray([1.0, 120.0, 110.0, 50.0, -25.0, 12.0], dtype=np.float32),
        "so100.buffer.action.std": np.asarray([20.0, 50.0, 40.0, 35.0, 55.0, 18.0], dtype=np.float32),
    }
    save_file(tensors, root / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    save_file(tensors, root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")


def _args(tmp_path: Path):
    summary = tmp_path / "summary.json"
    vlm = tmp_path / "vlm.json"
    ckpt = tmp_path / "smolvla"
    _write_summary(summary)
    _write_vlm_report(vlm)
    _write_ckpt(ckpt)
    return argparse.Namespace(
        summary_report=str(summary),
        vlm_enabled_report=str(vlm),
        smolvla_ckpt=str(ckpt),
        report_path=str(tmp_path / "audit.json"),
        markdown_report_path=str(tmp_path / "audit.md"),
    )


def test_action_normalization_audit_flags_non_libero_stats(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = audit.build_report(_args(tmp_path))

    assert code == 0
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["action_normalization_provenance_audit_passed"] is True
    assert report["diagnosis"]["checkpoint_action_stats_appear_non_libero_scale"] is True
    assert report["diagnosis"]["checkpoint_action_stats_prefix_mismatch_risk"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["policy"]["model_load_performed"] is False


def test_action_normalization_audit_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")

    report, code = audit.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_HEAVY_IMPORT" in report["recommended_next_step"]
    assert report["policy"]["heavy_model_imports_performed"] is False


def test_action_normalization_audit_requires_summary_pass(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    Path(args.summary_report).write_text(
        json.dumps({"vlm_enabled_offline_decoding_summary_passed": False}),
        encoding="utf-8",
    )

    report, code = audit.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not pass" in report["recommended_next_step"]
