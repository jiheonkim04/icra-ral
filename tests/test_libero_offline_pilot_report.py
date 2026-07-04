import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_pilot_report import build_libero_offline_bounded_pilot_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "54_generate_libero_offline_bounded_pilot_report.ps1"


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


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_reports(tmp_path: Path):
    interface = _write_json(
        tmp_path / "interface.json",
        {
            "ready_for_offline_interface_smoke": True,
            "ready_for_rollout": False,
        },
    )
    split = _write_json(
        tmp_path / "split.json",
        {
            "ready_for_tiny_offline_counterfactual_split": True,
            "hdf5_inventory_count": 4,
            "counterfactual_pair_count": 2,
            "matched_task_count": 2,
            "suites": ["libero_10"],
        },
    )
    head = _write_json(
        tmp_path / "head.json",
        {
            "libero_offline_head_comparison_passed": True,
            "pair_count": 2,
            "arms": {
                "actionmap_head_only_proxy": {"metrics": {"action_l1": 0.2}},
                "tca_map_head_only_proxy": {"metrics": {"action_l1": 0.1}},
                "tca_map_distributional_select_proxy": {"metrics": {"action_l1": 0.05}},
            },
            "comparison": {
                "tca_map_vs_actionmap": {"action_l1_delta": -0.1},
                "tca_select_vs_tca_map": {"action_l1_delta": -0.05},
            },
        },
    )
    lora = _write_json(
        tmp_path / "lora.json",
        {
            "libero_offline_lora_comparison_passed": True,
            "record_count": 4,
            "action_prefix_dim": 4,
            "max_steps": 2,
            "lora_rank": 2,
            "arms": [
                {"arm": "actionmap_lora", "metrics": {"action_l1": 0.3, "wrong_target_proxy_rate": 1.0}},
                {"arm": "tca_map_lora", "metrics": {"action_l1": 0.2, "wrong_target_proxy_rate": 0.5}},
                {"arm": "tca_map_lora_distributional_select", "metrics": {"action_l1": 0.2}},
            ],
            "comparison": {
                "tca_lora_vs_actionmap_lora": {"action_l1_delta": -0.1},
                "tca_select_lora_vs_tca_lora": {"action_l1_delta": 0.0},
            },
        },
    )
    return interface, split, head, lora


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_libero_offline_bounded_pilot_report_summarizes_gates(tmp_path):
    interface, split, head, lora = _write_reports(tmp_path)
    report = build_libero_offline_bounded_pilot_report(interface, split, head, lora)

    assert report["libero_offline_bounded_pilot_report_passed"] is True
    assert report["ready_for_simulator_readiness_risk_assessment"] is True
    assert report["ready_for_rollout"] is False
    assert report["blocked_for_paper_grade_claims"] is True
    assert report["policy"]["summary_only"] is True
    assert report["policy"]["training_performed_by_this_report"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["required_lora_summary"]["action_l1_delta_tca_lora_minus_actionmap_lora"] == -0.1


def test_libero_offline_bounded_pilot_script_outputs_report(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for bounded pilot report script tests")

    interface, split, head, lora = _write_reports(tmp_path)
    json_report = tmp_path / "report.json"
    markdown_report = tmp_path / "report.md"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-InterfaceReportPath",
            str(interface),
            "-SplitReportPath",
            str(split),
            "-HeadReportPath",
            str(head),
            "-LoraReportPath",
            str(lora),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(markdown_report),
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
    assert report["libero_offline_bounded_pilot_report_passed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert json_report.exists()
    assert markdown_report.exists()


def test_libero_offline_bounded_pilot_script_refuses_execution_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for bounded pilot report script tests")

    interface, split, head, lora = _write_reports(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-InterfaceReportPath",
            str(interface),
            "-SplitReportPath",
            str(split),
            "-HeadReportPath",
            str(head),
            "-LoraReportPath",
            str(lora),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "execution gates" in (result.stdout + result.stderr)
