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


def _write_manifest(tmp_path: Path, pair_count: int = 2) -> Path:
    pairs = []
    for index in range(pair_count):
        positive = tmp_path / "data" / "libero_object" / f"pick_soup_{index}_demo.hdf5"
        counter = tmp_path / "data" / "libero_object" / f"pick_milk_{index}_demo.hdf5"
        _write_demo(positive, 0.1 + index * 0.05)
        _write_demo(counter, 0.4 + index * 0.05)
        pairs.append(
            {
                "pair_id": f"libero_object:pick_soup_{index}__vs__pick_milk_{index}",
                "suite": "libero_object",
                "positive_task_id": f"pick_soup_{index}",
                "counterfactual_task_id": f"pick_milk_{index}",
                "positive_demo_file": str(positive),
                "positive_demo_relative_path": f"libero_object/pick_soup_{index}_demo.hdf5",
                "counterfactual_demo_file": str(counter),
                "counterfactual_demo_relative_path": f"libero_object/pick_milk_{index}_demo.hdf5",
                "positive_instruction": "pick soup",
                "counterfactual_instruction": "pick milk",
                "swap_type": "metadata_target_or_goal_swap",
            }
        )
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": pairs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_offline_head_comparison_trains_proxy_arms(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=2)
    report = build_offline_head_comparison(manifest, max_pairs=2, max_action_steps=4, max_steps=8)

    assert report["libero_offline_head_training_comparison_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert set(report["arms"]) == {
        "actionmap_head_only",
        "tca_map_head_only",
        "tca_map_distributional_select",
    }
    for arm in report["arms"].values():
        assert arm["training_performed"] is True
        assert arm["batch_size"] == 1
        assert arm["steps"] == 8
        assert arm["initial_loss"] >= 0.0
        assert arm["final_loss"] >= 0.0
        assert arm["loss_curve"]
        assert "standard_proxy_score" in arm["evaluation_metrics"]
        assert "wrong_target_proxy_rate" in arm["evaluation_metrics"]
        assert "action_target_consistency_score" in arm["evaluation_metrics"]
    assert report["comparison"]["conclusion"] in {
        "supports_tca_map_and_tca_select",
        "supports_tca_map_but_tca_select_not_improved_in_this_tiny_proxy",
        "weakens_tca_map",
    }


def test_offline_head_comparison_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline head comparison script tests")

    manifest = _write_manifest(tmp_path, pair_count=1)
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

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_offline_head_comparison_script_refuses_dangerous_gates(tmp_path):
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
        env={**os.environ, "ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing offline training/eval" in (result.stdout + result.stderr)


def test_offline_head_comparison_script_outputs_training_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline head comparison script tests")
    manifest = _write_manifest(tmp_path, pair_count=2)

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
            "2",
            "-MaxSteps",
            "8",
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

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0
    report = json.loads(result.stdout[start:])
    assert report["libero_offline_head_training_comparison_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["arms"]["actionmap_head_only"]["loss_curve"]
