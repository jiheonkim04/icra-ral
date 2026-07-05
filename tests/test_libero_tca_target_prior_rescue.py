import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_tca_target_prior_rescue import run_tca_target_prior_rescue


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "56_debug_tca_target_prior_rescue.ps1"


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(4, 7), dtype="f4")
        for row in range(4):
            actions[row, :] = offset + row * 0.01


def _write_manifest(tmp_path: Path, pair_count: int = 4) -> Path:
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


def test_tca_target_prior_rescue_outputs_required_arms_and_metrics(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=4)
    report = run_tca_target_prior_rescue(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=4,
        max_action_steps=4,
        max_samples=8,
        max_steps=8,
    )

    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["target_head_training_sanity"]["initial_target_ce_loss"] >= report["target_head_training_sanity"]["final_target_ce_loss"]
    arms = {arm["arm"]: arm for arm in report["arms"]}
    assert set(arms) == {
        "learned_target_hard_tca",
        "oracle_target_tca_upper_bound",
        "soft_target_marginalized_tca",
        "soft_target_distributional_select",
        "instruction_text_prior_tca",
        "constant_majority_target_tca",
    }
    for arm in arms.values():
        metrics = arm["evaluation_metrics"]
        assert "standard_proxy_score" in metrics
        assert "wrong_target_proxy_rate" in metrics
        assert "action_target_consistency_score" in metrics
        assert "counterfactual_separation_margin" in metrics
    assert arms["oracle_target_tca_upper_bound"]["oracle"] is True
    assert report["comparison"]["recommended_next_milestone"]
    assert report["ready_for_rollout"] is False
    assert report["ready_for_paper_claim"] is False


def test_tca_target_prior_rescue_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for target-prior rescue script tests")

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


def test_tca_target_prior_rescue_script_refuses_dangerous_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for target-prior rescue script tests")

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
        env={**os.environ, "ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing TCA target-prior rescue diagnostic" in (result.stdout + result.stderr)
