import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_fixed_prior_lora_attribution import run_fixed_prior_lora_attribution


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "130_compare_libero_fixed_prior_lora.ps1"


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


def _write_manifest(tmp_path: Path, pair_count: int = 4) -> Path:
    pairs = []
    for index in range(pair_count):
        positive = tmp_path / "data" / "libero_10" / f"positive_{index}_demo.hdf5"
        counter = tmp_path / "data" / "libero_10" / f"counter_{index}_demo.hdf5"
        _write_demo(positive, 0.1 + index * 0.05)
        _write_demo(counter, 0.4 + index * 0.05)
        pairs.append(
            {
                "pair_id": f"libero_10:positive_{index}__vs__counter_{index}",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
                "positive_instruction": "pick the soup can",
                "counterfactual_instruction": "pick the milk carton",
            }
        )
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": pairs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_fixed_prior_lora_attribution_outputs_required_arms(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=4)
    report = run_fixed_prior_lora_attribution(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=4,
        max_action_steps=4,
        max_steps=4,
        max_samples=8,
        rank=2,
        fixed_head_report_path=tmp_path / "missing_fixed_head.json",
        previous_lora_report_path=tmp_path / "missing_lora.json",
        require_training_gate=False,
    )

    assert report["fixed_prior_lora_attribution_passed"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["record_count"] == 8
    arms = {arm["arm"]: arm for arm in report["arms"]}
    assert set(arms) == {
        "actionmap_lora",
        "tca_map_lora_hard_learned_target",
        "tca_map_lora_instruction_text_prior",
        "tca_map_lora_fixed_learned_text_fusion",
        "oracle_target_tca_lora_upper_bound",
        "tca_map_lora_fixed_fusion_tca_select_ablation",
    }
    assert arms["oracle_target_tca_lora_upper_bound"]["oracle"] is True
    assert arms["tca_map_lora_fixed_fusion_tca_select_ablation"]["tca_select_ablation"] is True
    for arm in arms.values():
        assert arm["trainable_lora_parameter_count"] > 0
        assert arm["loss_decreased"] is True
        metrics = arm["evaluation_metrics"]
        assert "standard_proxy_score" in metrics
        assert "wrong_target_proxy_rate" in metrics
        assert "gap_to_oracle_target_tca_lora_standard_proxy" in metrics
    assert "fixed_prior_tca_lora_vs_actionmap_lora" in report["comparison"]


def test_fixed_prior_lora_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for fixed-prior LoRA script tests")

    manifest = _write_manifest(tmp_path, pair_count=4)
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


def test_fixed_prior_lora_script_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for fixed-prior LoRA script tests")

    manifest = _write_manifest(tmp_path, pair_count=4)
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
