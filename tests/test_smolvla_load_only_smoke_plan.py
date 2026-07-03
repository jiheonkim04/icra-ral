import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "15_plan_smolvla_load_only_smoke.ps1"


def _make_ready_smolvla_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    return smolvla_dir, checkpoint_root, hf_home


def _run_planner(tmp_path: Path, allow_heavy_import: bool = False) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the SmolVLA load-only smoke planner")

    smolvla_dir, checkpoint_root, hf_home = _make_ready_smolvla_layout(tmp_path)
    report_path = tmp_path / "load_only_plan_report.json"
    env = os.environ.copy()
    env.update(
        {
            "SMOLVLA_CKPT": str(smolvla_dir),
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "HF_HOME": str(hf_home),
            "ALLOW_HEAVY_IMPORT": "1" if allow_heavy_import else "",
        }
    )
    return subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PathsFile",
            str(tmp_path / "missing_paths.local.yaml"),
            "-ReportPath",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _json_from_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_smolvla_load_only_smoke_planner_is_planning_only(tmp_path):
    result = _run_planner(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["plan_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["next_gate"]["explicit_approval_required"] is False
    assert report["next_gate"]["required_gate"] == "ALLOW_HEAVY_IMPORT=1"
    assert report["next_gate"]["standing_approval"] == "SmolVLA autonomous pilot standing approval"


def test_smolvla_load_only_smoke_planner_refuses_heavy_import_gate(tmp_path):
    result = _run_planner(tmp_path, allow_heavy_import=True)
    assert result.returncode == 2
    report = _json_from_stdout(result.stdout)
    assert report["policy"]["plan_only"] is True
    assert report["policy"]["heavy_import_gate_set"] is True
    assert report["policy"]["heavy_model_imports_performed"] is False
