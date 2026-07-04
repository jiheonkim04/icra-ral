import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.smolvla.tca_select_ambiguity_stress import run_tca_select_ambiguity_stress


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "129_run_tca_select_ambiguity_stress_test.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
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


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "positive.hdf5"
    counter = tmp_path / "counter.hdf5"
    _write_demo(positive, 0.1)
    _write_demo(counter, 0.6)
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": [
            {
                "pair_id": "pair_0",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_tca_select_ambiguity_stress_improves_wrong_target_proxy(tmp_path):
    manifest = _write_manifest(tmp_path)
    report = run_tca_select_ambiguity_stress(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=1,
        max_records=2,
        candidate_count=4,
        max_runtime_seconds=60,
    )

    assert report["tca_select_ambiguity_stress_passed"] is True
    assert report["ready_for_selection_attribution_update"] is True
    assert report["ready_for_paper_claim"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["privileged_inference_used"] is False
    assert report["policy"]["external_verifier_used"] is False
    assert report["metrics"]["selection_wrong_target_proxy_delta_vs_top_heatmap"] < 0.0
    assert report["metrics"]["selection_action_l1_delta_vs_top_heatmap"] <= 0.0


def test_tca_select_ambiguity_stress_script_runs(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for TCA-Select ambiguity stress script tests")

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
            "1",
            "-MaxRecords",
            "2",
            "-CandidateCount",
            "4",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["tca_select_ambiguity_stress_passed"] is True
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_tca_select_ambiguity_stress_script_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for TCA-Select ambiguity stress script tests")

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
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_ROLLOUTS": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "ALLOW_ROLLOUTS" in (result.stdout + result.stderr)
