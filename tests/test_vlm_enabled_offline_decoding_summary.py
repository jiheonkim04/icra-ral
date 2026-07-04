import argparse
import json
from pathlib import Path

from tca_map.smolvla import vlm_enabled_offline_decoding_summary as summary


def _write_report(path: Path, *, load_vlm_weights: bool, l1: float, mse: float):
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "load_vlm_weights": load_vlm_weights,
                    "mean_action_l1_to_expert": l1,
                    "mean_action_mse_to_expert": mse,
                    "mean_policy6_l1_to_expert_first6": l1 + 0.1,
                    "offline_alignment_signal": "weak",
                    "clipped_values_total": 3,
                    "sample_count": 3,
                    "timesteps": [0, 136, 271],
                },
                "policy": {
                    "downloads_performed": False,
                    "rollouts_performed": False,
                    "training_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_config(root: Path):
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "load_vlm_weights": True,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "normalization_mapping": {"ACTION": "MEAN_STD"},
                "output_features": {"action": {"shape": [6], "type": "ACTION"}},
                "input_features": {
                    "observation.state": {"shape": [6], "type": "STATE"},
                    "observation.images.camera1": {"shape": [3, 256, 256], "type": "VISUAL"},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "device_processor", "config": {"device": "cpu"}}]}),
        encoding="utf-8",
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"registry_name": "unnormalizer_processor", "config": {"norm_map": {"ACTION": "MEAN_STD"}}},
                    {"registry_name": "device_processor", "config": {"device": "cpu"}},
                ]
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path):
    no_vlm = tmp_path / "no_vlm.json"
    vlm = tmp_path / "vlm.json"
    ckpt = tmp_path / "smolvla"
    _write_report(no_vlm, load_vlm_weights=False, l1=0.4, mse=0.2)
    _write_report(vlm, load_vlm_weights=True, l1=0.3, mse=0.15)
    _write_config(ckpt)
    return argparse.Namespace(
        no_vlm_report=str(no_vlm),
        vlm_enabled_report=str(vlm),
        smolvla_ckpt=str(ckpt),
        report_path=str(tmp_path / "summary.json"),
        markdown_report_path=str(tmp_path / "summary.md"),
    )


def test_summary_compares_vlm_enabled_against_no_vlm(tmp_path, monkeypatch):
    for gate in summary.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = summary.build_report(_args(tmp_path))

    assert code == 0
    assert report["decision"] == "summary_complete"
    assert report["comparison"]["vlm_enabled_improved_l1"] is True
    assert report["comparison"]["l1_delta"]["absolute"] == -0.1
    assert report["ready_for_rollout_scaling"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["diagnosis"]["ready_for_action_normalization_provenance_probe"] is True


def test_summary_refuses_forbidden_execution_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = summary.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["model_inference_performed"] is False


def test_summary_stops_when_required_report_missing(tmp_path, monkeypatch):
    for gate in summary.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    Path(args.vlm_enabled_report).unlink()

    report, code = summary.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "missing" in report["recommended_next_step"]
