import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.smolvla.online_action_generation_bridge import build_online_action_source_inventory


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "143_online_action_generation_bridge.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for online action bridge script tests")
    return exe


def _clean_env():
    env = os.environ.copy()
    for key in (
        "ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT",
        "ALLOW_DOWNLOADS",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_ROLLOUT",
        "ALLOW_ROLLOUTS",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
    ):
        env.pop(key, None)
    return env


def _source(report, name):
    for item in report["sources"]:
        if item["source_name"] == name:
            return item
    raise AssertionError(f"missing source {name}")


def test_online_action_inventory_marks_actionmap_tca_as_not_7d_rollout_sources():
    inventory = build_online_action_source_inventory()

    actionmap = _source(inventory, "existing_actionmap_head_output")
    tca = _source(inventory, "existing_tca_map_head_output")
    native = _source(inventory, "native_smolvla_policy_output")
    hdf5 = _source(inventory, "hdf5_action_derived_candidates")

    assert native["online_generated_from_current_observation"] is True
    assert native["depends_on_future_hdf5_action"] is False
    assert native["action_dimension_produced"] == "6D_policy_delta_pose"
    assert native["can_be_mapped_to_libero_7d_without_silent_padding"] is True

    assert actionmap["action_dimension_produced"] == "4D"
    assert actionmap["valid_for_closed_loop_rollout_claim"] is False
    assert actionmap["can_be_mapped_to_libero_7d_without_silent_padding"] is False
    assert "unsupported action dimension mapping" in actionmap["adapter_probe"]["error"]

    assert tca["action_dimension_produced"] == "4D"
    assert tca["valid_for_closed_loop_rollout_claim"] is False
    assert tca["can_be_mapped_to_libero_7d_without_silent_padding"] is False
    assert hdf5["depends_on_future_hdf5_action"] is True
    assert hdf5["valid_for_closed_loop_rollout_claim"] is False
    assert inventory["valid_actionmap_tca_online_source_found"] is False


def test_online_action_bridge_script_without_gate_writes_inventory_only(tmp_path):
    report = tmp_path / "online_bridge.json"
    markdown = tmp_path / "online_bridge.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-JsonReportPath",
            str(report),
            "-MarkdownReportPath",
            str(markdown),
            "-MaxSteps",
            "5",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["decision"] == "inventory_only_blocked_before_rollout"
    assert data["policy"]["rollout_happened"] is False
    assert data["policy"]["model_load_performed"] is False
    assert data["inventory"]["valid_native_online_source_found"] is True
    assert data["inventory"]["valid_actionmap_tca_online_source_found"] is False
    assert "not closed-loop online ActionMap/TCA" in data["result"]["blocked_reason"]
    assert markdown.exists()
