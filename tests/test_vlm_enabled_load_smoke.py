import argparse
import json
from pathlib import Path

from tca_map.smolvla import vlm_enabled_load_smoke as smoke


def _write_plan(path: Path, *, ready=True):
    path.write_text(
        json.dumps(
            {
                "decision": "proceed" if ready else "stop",
                "ready_for_bounded_vlm_enabled_load_smoke_runner": ready,
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
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "processor_config.json", "model.safetensors"):
        (dep / name).write_text("x", encoding="utf-8")
    return smolvla, checkpoint_root, hf_home


def _args(tmp_path: Path) -> argparse.Namespace:
    smolvla, checkpoint_root, hf_home = _make_layout(tmp_path)
    plan = tmp_path / "plan.json"
    _write_plan(plan)
    return argparse.Namespace(
        plan_report=str(plan),
        smolvla_ckpt=str(smolvla),
        checkpoint_root=str(checkpoint_root),
        hf_home=str(hf_home),
        report_path=str(tmp_path / "report.json"),
        device="cpu",
    )


def _fake_loader(*, smolvla_ckpt, hf_home, external_dependency, device):
    assert Path(smolvla_ckpt).exists()
    assert Path(hf_home).exists()
    assert external_dependency["found"] is True
    assert device == "cpu"
    return {
        "load_elapsed_sec": 1.0,
        "rss_before_mb": 100.0,
        "rss_after_mb": 200.0,
        "gpu_before": {"available": False},
        "gpu_after": {"available": False},
        "cuda_max_allocated_mb": 0,
        "parameter_count": 123,
        "trainable_parameter_count": 0,
        "device": "cpu",
        "config_device": "cpu",
        "vlm_model_name": external_dependency["root"],
        "load_vlm_weights": True,
    }


def test_vlm_enabled_load_smoke_requires_both_task_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)
    monkeypatch.delenv("ALLOW_VLM_ENABLED_LOAD_SMOKE", raising=False)

    report, code = smoke.build_report(_args(tmp_path), loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert report["policy"]["model_load_performed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["recommended_next_step"]


def test_vlm_enabled_load_smoke_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    monkeypatch.setenv("ALLOW_VLM_ENABLED_LOAD_SMOKE", "1")
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = smoke.build_report(_args(tmp_path), loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    monkeypatch.delenv("ALLOW_ROLLOUTS", raising=False)


def test_vlm_enabled_load_smoke_fake_success_is_load_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    monkeypatch.setenv("ALLOW_VLM_ENABLED_LOAD_SMOKE", "1")

    report, code = smoke.build_report(_args(tmp_path), loader=_fake_loader)

    assert code == 0
    assert report["decision"] == "load_smoke_complete"
    assert report["vlm_enabled_load_smoke_passed"] is True
    assert report["policy"]["heavy_model_imports_performed"] is True
    assert report["policy"]["model_load_performed"] is True
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["load"]["load_vlm_weights"] is True


def test_vlm_enabled_load_smoke_stops_when_plan_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    monkeypatch.setenv("ALLOW_VLM_ENABLED_LOAD_SMOKE", "1")
    args = _args(tmp_path)
    _write_plan(Path(args.plan_report), ready=False)

    report, code = smoke.build_report(args, loader=_fake_loader)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
