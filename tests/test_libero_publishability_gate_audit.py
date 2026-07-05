import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_publishability_gate_audit import run_publishability_gate_audit


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "136_audit_publishability_gate.ps1"


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


def test_publishability_gate_audit_outputs_leakage_breakdowns_and_selector_headroom(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=8)
    report = run_publishability_gate_audit(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        seeds=[11],
        max_pairs=8,
        max_action_steps=4,
        max_steps=4,
        max_runtime_seconds=900,
        max_samples=16,
        rank=2,
        require_training_gate=False,
    )

    assert report["publishability_gate_audit_passed"] is True
    assert report["record_count"] == 16
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is True
    assert report["policy"]["rollouts_performed"] is False

    prior = report["prior_source_leakage_audit"]
    assert prior["instruction_text_prior"]["classification"] == "A_valid_test_time_semantic_prior"
    assert prior["instruction_text_prior"]["uses_eval_labels"] is False
    assert prior["instruction_text_prior"]["uses_dataset_target_labels"] is False
    assert prior["fixed_learned_text_fusion"]["training_uses_dataset_target_labels"] is True
    assert prior["fixed_learned_text_fusion"]["uses_eval_labels"] is False
    assert prior["oracle_target_upper_bound"]["classification"] == "C_oracle_like_upper_bound"

    assert report["per_task_breakdown"]
    assert report["per_target_breakdown"]
    selector = report["selector_headroom_summary"]
    assert "oracle_selector_delta_over_nonselect_standard_proxy" in selector
    assert "candidate_diversity" in selector
    assert "score_diversity" in selector
    assert report["decision"]["selector_recommendation"]


def test_publishability_gate_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for publishability gate script tests")

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
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_publishability_gate_script_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for publishability gate script tests")

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
        env=_clean_env({"ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
