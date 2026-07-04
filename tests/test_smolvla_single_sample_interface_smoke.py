import json
import os
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_ready_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    smolvla = tmp_path / "smolvla"
    checkpoint_root = tmp_path / "checkpoints"
    hf_home = tmp_path / "hf_home"
    dep = hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
    smolvla.mkdir()
    checkpoint_root.mkdir()
    dep.mkdir(parents=True)
    (smolvla / "config.json").write_text("{}", encoding="utf-8")
    (smolvla / "model.safetensors").write_text("", encoding="utf-8")
    (smolvla / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {
                            "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (dep / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return smolvla, checkpoint_root, hf_home


def _run_module(
    tmp_path: Path,
    allow_heavy_import: bool,
    allow_single_sample: bool,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    smolvla, checkpoint_root, hf_home = _make_ready_layout(tmp_path)
    report = tmp_path / "report.json"
    env = os.environ.copy()
    env.update(
        {
            "ALLOW_HEAVY_IMPORT": "1" if allow_heavy_import else "",
            "ALLOW_SINGLE_SAMPLE_INFERENCE": "1" if allow_single_sample else "",
            "SMOLVLA_CKPT": str(smolvla),
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "HF_HOME": str(hf_home),
        }
    )
    env.update(extra_env or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tca_map.smolvla.single_sample_interface_smoke",
            "--report-path",
            str(report),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _report(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_single_sample_interface_requires_heavy_import_gate(tmp_path):
    result = _run_module(tmp_path, allow_heavy_import=False, allow_single_sample=True)
    assert result.returncode == 2
    report = _report(result.stdout)
    assert report["policy"]["single_sample_interface_smoke"] is True
    assert report["policy"]["heavy_import_gate_set"] is False
    assert report["policy"]["single_sample_model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["real_rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_single_sample_interface_requires_single_sample_gate(tmp_path):
    result = _run_module(tmp_path, allow_heavy_import=True, allow_single_sample=False)
    assert result.returncode == 3
    report = _report(result.stdout)
    assert report["policy"]["heavy_import_gate_set"] is True
    assert report["policy"]["single_sample_inference_gate_set"] is False
    assert report["policy"]["single_sample_model_inference_performed"] is False


def test_single_sample_interface_blocks_forbidden_gates_before_loading(tmp_path):
    result = _run_module(
        tmp_path,
        allow_heavy_import=True,
        allow_single_sample=True,
        extra_env={"ALLOW_ROLLOUTS": "1"},
    )
    assert result.returncode == 4
    report = _report(result.stdout)
    assert report["policy"]["heavy_import_gate_set"] is True
    assert report["policy"]["single_sample_inference_gate_set"] is True
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["single_sample_model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["real_rollouts_performed"] is False


def test_synthetic_batch_records_state_and_image_adapter_metadata(monkeypatch, tmp_path):
    from tca_map.smolvla import single_sample_interface_smoke as smoke

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": torch.zeros((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }

    class FakeFeature:
        def __init__(self, shape):
            self.shape = shape

    class FakeConfig:
        tokenizer_max_length = 4
        input_features = {"observation.state": FakeFeature((6,))}
        image_features = {
            "observation.images.camera1": FakeFeature((3, 8, 8)),
            "observation.images.camera2": FakeFeature((3, 8, 8)),
        }

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    batch, metadata = smoke._build_synthetic_batch(FakeConfig(), tmp_path, "pick up the object", "cpu")

    assert batch["observation.state"].shape == (1, 6)
    assert metadata["state_adapter"]["adapter"] == "diagnostic_eef_pos_quat_xyz_6d_state_adapter"
    assert metadata["state_adapter"]["silent_truncation_performed"] is False
    assert metadata["image_adapters"]["observation.images.camera1"]["source_key"] == "agentview_image"
    assert metadata["image_adapters"]["observation.images.camera2"]["source_key"] == "robot0_eye_in_hand_image"
    assert metadata["synthetic_only"] is True
