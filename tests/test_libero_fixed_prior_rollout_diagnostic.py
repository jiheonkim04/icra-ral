import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import build_fixed_prior_rollout_cases


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "139_bounded_fixed_prior_rollout_diagnostic.ps1"


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        actions = demo.create_dataset("actions", shape=(5, 7), dtype="f4")
        for row in range(5):
            actions[row, :] = offset + row * 0.01
            actions[row, 6] = -1.0


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_10" / "task_positive_demo.hdf5"
    counter = tmp_path / "data" / "libero_10" / "task_counter_demo.hdf5"
    _write_demo(positive, 0.1)
    _write_demo(counter, 0.3)
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": [
            {
                "pair_id": "libero_10:task_positive__vs__task_counter",
                "suite": "libero_10",
                "positive_task_id": "task_positive",
                "counterfactual_task_id": "task_counter",
                "positive_instruction": "put the moka pot on the stove",
                "counterfactual_instruction": "put the bowl in the drawer",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_green_readiness(path: Path) -> None:
    path.write_text(
        json.dumps({"risk_gate_status": "green", "rollout_diagnostic_authorized": True}),
        encoding="utf-8",
    )


def test_build_fixed_prior_rollout_cases_preserves_7d_actions(tmp_path):
    manifest = _write_manifest(tmp_path)
    cases = build_fixed_prior_rollout_cases(manifest, max_tasks=1, max_steps=5)

    assert len(cases) == 1
    case = cases[0]
    assert case["max_steps"] == 5
    assert case["action_diagnostics"]["positive_demo"]["shape"] == [5, 7]
    assert case["action_diagnostics"]["counterfactual_demo"]["shape"] == [5, 7]
    assert case["action_diagnostics"]["actionmap_style_target_agnostic_mean"]["shape"] == [5, 7]
    assert case["action_diagnostics"]["candidate_positive_vs_counter_l1"] > 0
    assert case["action_diagnostics"]["actionmap_l1_to_positive"] > 0
    assert case["action_diagnostics"]["fixed_prior_l1_to_positive"] == 0.0
    assert [variant["name"] for variant in case["variants"]] == [
        "actionmap_style_target_agnostic_mean",
        "fixed_semantic_target_prior_tca",
        "oracle_target_tca_upper_bound",
    ]


def test_fixed_prior_rollout_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for fixed-prior rollout script tests")
    manifest = _write_manifest(tmp_path)
    readiness = tmp_path / "readiness.json"
    _write_green_readiness(readiness)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ManifestPath",
            str(manifest),
            "-ReadinessReportPath",
            str(readiness),
            "-JsonReportPath",
            str(report),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["policy"]["diagnostic_rollouts_performed"] is False
    assert "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC=1 is required" in data["result"]["reason"]
