import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_head_comparison import build_offline_head_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "52_compare_libero_offline_actionmap_tca.ps1"


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(4, 7), dtype="f4")
        for row in range(4):
            actions[row, :] = offset + row * 0.01


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_object" / "pick_soup_demo.hdf5"
    counter = tmp_path / "data" / "libero_object" / "pick_milk_demo.hdf5"
    _write_demo(positive, 0.1)
    _write_demo(counter, 0.4)
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": [
            {
                "pair_id": "libero_object:pick_soup__vs__pick_milk",
                "positive_demo_file": str(positive),
                "positive_demo_relative_path": "libero_object/pick_soup_demo.hdf5",
                "counterfactual_demo_file": str(counter),
                "counterfactual_demo_relative_path": "libero_object/pick_milk_demo.hdf5",
                "positive_instruction": "pick soup",
                "counterfactual_instruction": "pick milk",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_offline_head_comparison_builds_proxy_arms(tmp_path):
    manifest = _write_manifest(tmp_path)
    report = build_offline_head_comparison(manifest, max_pairs=1, max_action_steps=4)

    assert report["libero_offline_head_comparison_passed"] is True
    assert report["ready_for_required_tiny_lora_comparison"] is True
    assert report["ready_for_rollout"] is False
    assert set(report["arms"]) == {
        "actionmap_head_only_proxy",
        "tca_map_head_only_proxy",
        "tca_map_distributional_select_proxy",
    }
    assert report["arms"]["tca_map_head_only_proxy"]["metrics"]["wrong_target_proxy_rate"] == 0.0
    assert report["arms"]["actionmap_head_only_proxy"]["metrics"]["wrong_target_proxy_rate"] == 1.0
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["model_inference_performed"] is False


def test_offline_head_comparison_script_refuses_execution_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline head comparison script tests")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ManifestPath",
            str(tmp_path / "missing.json"),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_TINY_TRAINING": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing offline comparison" in (result.stdout + result.stderr)


def test_offline_head_comparison_script_outputs_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline head comparison script tests")
    manifest = _write_manifest(tmp_path)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ManifestPath",
            str(manifest),
            "-MaxPairs",
            "1",
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0
    report = json.loads(result.stdout[start:])
    assert report["libero_offline_head_comparison_passed"] is True
    assert report["policy"]["rollouts_performed"] is False
