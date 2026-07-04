import argparse
import json
from pathlib import Path

from tca_map.smolvla import vlm_enabled_repeated_offline_decoding as runner


def _write_plan(path: Path, *, ready=True):
    path.write_text(
        json.dumps(
            {
                "ready_for_bounded_vlm_enabled_repeated_offline_decoding_runner": ready,
                "inputs": {"hdf5_path": str(path.parent / "missing.hdf5"), "selected_timesteps": [0, 1, 2]},
                "planned_sample": {"hdf5": {"demo_name": "demo_0"}, "selected_task_text": "turn on the stove"},
            }
        ),
        encoding="utf-8",
    )


def _write_previous(path: Path):
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "mean_action_l1_to_expert": 0.4,
                    "mean_action_mse_to_expert": 0.2,
                    "load_vlm_weights": False,
                }
            }
        ),
        encoding="utf-8",
    )


def _make_layout(tmp_path: Path):
    smolvla = tmp_path / "smolvla"
    checkpoint_root = tmp_path / "checkpoints"
    hf_home = tmp_path / "hf_home"
    dep = hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
    smolvla.mkdir()
    checkpoint_root.mkdir()
    dep.mkdir(parents=True)
    (smolvla / "config.json").write_text("{}", encoding="utf-8")
    (smolvla / "model.safetensors").write_text("x", encoding="utf-8")
    (smolvla / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "processor_config.json"):
        (dep / name).write_text("x", encoding="utf-8")
    return smolvla, checkpoint_root, hf_home


def _args(tmp_path: Path):
    smolvla, checkpoint_root, hf_home = _make_layout(tmp_path)
    plan = tmp_path / "plan.json"
    previous = tmp_path / "previous.json"
    _write_plan(plan)
    _write_previous(previous)
    return argparse.Namespace(
        plan_report=str(plan),
        previous_report=str(previous),
        smolvla_ckpt=str(smolvla),
        checkpoint_root=str(checkpoint_root),
        hf_home=str(hf_home),
        hdf5_path="",
        task="turn on the stove",
        report_path=str(tmp_path / "report.json"),
        device="cpu",
    )


def _fake_loader(smolvla_ckpt, hf_home, external_dependency, device):
    class Config:
        load_vlm_weights = True

    class Policy:
        pass

    return Policy(), Config()


def test_vlm_enabled_repeated_offline_runner_requires_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING", raising=False)

    report, code = runner.build_report(_args(tmp_path), loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert report["policy"]["model_load_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["recommended_next_step"]


def test_vlm_enabled_repeated_offline_runner_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    monkeypatch.setenv("ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING", "1")
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = runner.build_report(_args(tmp_path), loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    monkeypatch.delenv("ALLOW_ROLLOUTS", raising=False)


def test_vlm_enabled_repeated_offline_runner_stops_when_plan_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    monkeypatch.setenv("ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING", "1")
    args = _args(tmp_path)
    _write_plan(Path(args.plan_report), ready=False)

    report, code = runner.build_report(args, loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
