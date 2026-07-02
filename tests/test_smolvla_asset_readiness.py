import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "11_check_real_assets.ps1"


def _run_real_asset_checker(
    tmp_path: Path,
    smolvla_dir: Path,
    checkpoint_root: Path,
    hf_home: Path | None = None,
) -> dict:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the real asset readiness script")

    env = os.environ.copy()
    env.update(
        {
            "SMOLVLA_CKPT": str(smolvla_dir),
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "HF_HOME": "" if hf_home is None else str(hf_home),
            "OPENVLA_OFT_CKPT": str(tmp_path / "missing_openvla"),
            "LIBERO_ROOT": str(tmp_path / "missing_libero"),
            "LIBERO_DATA_ROOT": str(tmp_path / "missing_libero_data"),
            "ROBOSUITE_ROOT": str(tmp_path / "missing_robosuite"),
            "DATA_ROOT": str(tmp_path / "missing_data"),
        }
    )
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PathsFile",
            str(tmp_path / "missing_paths.local.yaml"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout
    return json.loads(result.stdout[start:])


def test_empty_smolvla_directory_is_path_ready_not_adapter_ready(tmp_path):
    smolvla_dir = tmp_path / "smolvla"
    checkpoint_root = tmp_path / "checkpoints"
    smolvla_dir.mkdir()
    checkpoint_root.mkdir()

    report = _run_real_asset_checker(tmp_path, smolvla_dir, checkpoint_root)

    assert report["smolvla_path_configured"] is True
    assert report["smolvla_path_exists"] is True
    assert report["ready_for_smolvla_path_check"] is True
    assert report["smolvla_checkpoint_files_present"] is False
    assert report["ready_for_smolvla_adapter_smoke"] is False
    assert report["ready_for_smolvla_smoke"] is False


def test_dummy_smolvla_marker_files_complete_lightweight_readiness(tmp_path):
    smolvla_dir = tmp_path / "smolvla"
    checkpoint_root = tmp_path / "checkpoints"
    smolvla_dir.mkdir()
    checkpoint_root.mkdir()
    (smolvla_dir / "config.json").write_text("{}", encoding="utf-8")
    (smolvla_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (smolvla_dir / "model.safetensors").write_text("", encoding="utf-8")

    report = _run_real_asset_checker(tmp_path, smolvla_dir, checkpoint_root)

    assert report["smolvla_checkpoint_files_present"] is True
    assert report["ready_for_smolvla_path_check"] is True
    assert report["ready_for_smolvla_adapter_smoke"] is True
    assert report["ready_for_smolvla_smoke"] is True
    assert report["smolvla_expected_files"]["config_found"] == ["config.json"]
    assert report["smolvla_expected_files"]["tokenizer_found"] == ["tokenizer.json"]
    assert "model.safetensors" in report["smolvla_expected_files"]["weights_found"]


def test_external_smolvla_tokenizer_dependency_can_satisfy_readiness(tmp_path):
    smolvla_dir = tmp_path / "smolvla"
    checkpoint_root = tmp_path / "checkpoints"
    hf_home = tmp_path / "hf_home"
    dependency_dir = hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
    smolvla_dir.mkdir()
    checkpoint_root.mkdir()
    dependency_dir.mkdir(parents=True)

    (smolvla_dir / "config.json").write_text("{}", encoding="utf-8")
    (smolvla_dir / "model.safetensors").write_text("", encoding="utf-8")
    (smolvla_dir / "policy_preprocessor.json").write_text(
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
    (dependency_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    report = _run_real_asset_checker(tmp_path, smolvla_dir, checkpoint_root, hf_home)

    dependency = report["smolvla_expected_files"]["external_tokenizer_dependency"]
    assert report["smolvla_checkpoint_files_present"] is True
    assert report["ready_for_smolvla_adapter_smoke"] is True
    assert report["smolvla_expected_files"]["tokenizer_found"] == []
    assert dependency["name"] == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    assert dependency["found"] is True
    assert dependency["files_found"] == ["tokenizer_config.json"]
