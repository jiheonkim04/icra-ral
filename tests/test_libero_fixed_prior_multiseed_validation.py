import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_fixed_prior_multiseed_validation import run_fixed_prior_multiseed_validation


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "135_validate_libero_fixed_prior_multiseed.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(4, 7), dtype="f4")
        for row in range(4):
            actions[row, :] = offset + row * 0.01


def _write_manifest(tmp_path: Path, pair_count: int = 8) -> Path:
    pairs = []
    names = [
        ("soup can", "milk carton"),
        ("black bowl", "moka pot"),
        ("yellow mug", "white mug"),
        ("book", "basket"),
    ]
    for index in range(pair_count):
        positive = tmp_path / "data" / "libero_10" / f"positive_{index}_demo.hdf5"
        counter = tmp_path / "data" / "libero_10" / f"counter_{index}_demo.hdf5"
        _write_demo(positive, 0.1 + index * 0.03)
        _write_demo(counter, 0.4 + index * 0.03)
        left, right = names[index % len(names)]
        pairs.append(
            {
                "pair_id": f"libero_10:positive_{index}__vs__counter_{index}",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
                "positive_instruction": f"pick the {left}",
                "counterfactual_instruction": f"pick the {right}",
            }
        )
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": pairs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_multiseed_validation_aggregates_fixed_split(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=8)
    report = run_fixed_prior_multiseed_validation(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        seeds=[11, 23, 37],
        max_pairs=8,
        max_action_steps=4,
        max_steps=4,
        max_runtime_seconds=900,
        max_samples=16,
        rank=2,
        require_training_gate=False,
    )

    assert report["fixed_prior_multiseed_validation_passed"] is True
    assert report["seed_count"] == 3
    assert report["record_count"] == 16
    assert report["split_consistent"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert len(report["per_seed_table"]) == 3
    assert "actionmap_lora" in report["aggregate_arms"]
    assert "tca_map_lora_fixed_learned_text_fusion" in report["aggregate_arms"]
    comparison = report["aggregate_comparison"]
    assert 0 <= comparison["fixed_prior_tca_lora_beats_actionmap_lora_count"] <= 3
    assert 0 <= comparison["tca_select_nontrivial_gain_count"] <= 3
    assert "mean" in comparison["fixed_prior_tca_lora_advantage"]


def test_multiseed_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for multiseed script tests")

    manifest = _write_manifest(tmp_path, pair_count=8)
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
            "-Seeds",
            "11,23,37",
            "-MaxActionSteps",
            "4",
            "-MaxSteps",
            "4",
            "-Rank",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_multiseed_script_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for multiseed script tests")

    manifest = _write_manifest(tmp_path, pair_count=8)
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
            "-Seeds",
            "11,23,37",
            "-MaxActionSteps",
            "4",
            "-MaxSteps",
            "4",
            "-Rank",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
