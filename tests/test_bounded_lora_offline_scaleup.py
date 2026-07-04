import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_lora_scaleup import run_bounded_libero_offline_lora_scaleup


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "126_bounded_lora_offline_scaleup.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
        "ALLOW_SIMULATOR_IMPORT_SMOKE",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_TINY_ROLLOUT",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(6, 7), dtype="f4")
        for row in range(6):
            actions[row, :] = offset + row * 0.01


def _write_manifest(tmp_path: Path, pair_count: int = 2) -> Path:
    pairs = []
    for index in range(pair_count):
        positive = tmp_path / "data" / f"positive_{index}.hdf5"
        counter = tmp_path / "data" / f"counter_{index}.hdf5"
        _write_demo(positive, 0.1 + index * 0.02)
        _write_demo(counter, 0.4 + index * 0.02)
        pairs.append(
            {
                "pair_id": f"pair_{index}",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
                "positive_instruction": f"pick target object {index}",
                "counterfactual_instruction": f"pick distractor object {index}",
            }
        )
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": pairs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_bounded_lora_offline_scaleup_builds_scaleup_report(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=2)
    report = run_bounded_libero_offline_lora_scaleup(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=2,
        max_action_steps=4,
        max_steps=3,
        max_samples=4,
        rank=2,
        require_training_gate=False,
    )

    assert report["bounded_lora_offline_scaleup_passed"] is True
    assert report["ready_for_offline_evidence_refresh"] is True
    assert report["ready_for_rollout"] is False
    assert report["ready_for_paper_claim"] is False
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["scaleup_limits"]["max_samples_cap"] == 64
    assert {arm["arm"] for arm in report["arms"]} == {
        "actionmap_lora",
        "tca_map_lora",
        "tca_map_lora_distributional_select",
    }


def test_bounded_lora_offline_scaleup_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for bounded LoRA scale-up script tests")

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


def test_bounded_lora_offline_scaleup_script_runs_with_bounded_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for bounded LoRA scale-up script tests")

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
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxPairs",
            "2",
            "-MaxActionSteps",
            "4",
            "-MaxSteps",
            "3",
            "-MaxSamples",
            "4",
            "-Rank",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["bounded_lora_offline_scaleup_passed"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_bounded_lora_offline_scaleup_script_rejects_over_budget_steps(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for bounded LoRA scale-up script tests")

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
            "-MaxSteps",
            "65",
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 13
    assert "MaxSteps" in (result.stdout + result.stderr)
