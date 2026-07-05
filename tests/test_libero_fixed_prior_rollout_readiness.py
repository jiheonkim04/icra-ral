import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_fixed_prior_rollout_readiness import run_fixed_prior_rollout_readiness_gate
from tca_map.smolvla.interface_adapters import adapt_policy_action_to_env_action


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "138_gate_fixed_prior_rollout_readiness.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_GPU_TRAINING",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUT",
        "ALLOW_ROLLOUTS",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
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


def _write_manifest(tmp_path: Path, pair_count: int = 2) -> Path:
    pairs = []
    for index in range(pair_count):
        positive = tmp_path / "data" / "libero_10" / f"positive_{index}_demo.hdf5"
        counter = tmp_path / "data" / "libero_10" / f"counter_{index}_demo.hdf5"
        _write_demo(positive, 0.1 + index * 0.03)
        _write_demo(counter, 0.4 + index * 0.03)
        pairs.append(
            {
                "pair_id": f"libero_10:positive_{index}__vs__counter_{index}",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
                "positive_instruction": f"pick the soup can {index}",
                "counterfactual_instruction": f"pick the milk carton {index}",
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"ready_for_tiny_offline_counterfactual_split": True, "counterfactual_pairs": pairs}),
        encoding="utf-8",
    )
    return manifest_path


def _write_status_reports(tmp_path: Path):
    sim_import = tmp_path / "import.json"
    render = tmp_path / "render.json"
    reset = tmp_path / "reset.json"
    zero = tmp_path / "zero.json"
    action_stats = tmp_path / "action_stats.json"
    metadata = tmp_path / "metadata.json"
    alignment = tmp_path / "alignment.json"
    vlm = tmp_path / "vlm.json"
    sim_import.write_text(json.dumps({"bounded_simulator_import_smoke_passed": True}), encoding="utf-8")
    render.write_text(json.dumps({"bounded_simulator_render_smoke_passed": True}), encoding="utf-8")
    reset.write_text(json.dumps({"bounded_simulator_reset_step_smoke_passed": True}), encoding="utf-8")
    zero.write_text(json.dumps({"bounded_libero_robosuite_diagnostic_rollout_passed": True}), encoding="utf-8")
    action_stats.write_text(json.dumps({"decision": "no_go_rollout_scaling"}), encoding="utf-8")
    metadata.write_text(json.dumps({"high_priority_findings": ["action_dim_mismatch_explicit_adapter_in_use"]}), encoding="utf-8")
    alignment.write_text(json.dumps({"exists": True}), encoding="utf-8")
    vlm.write_text(json.dumps({"exists": True}), encoding="utf-8")
    return (
        {"import": sim_import, "render": render, "reset_step": reset, "zero_rollout": zero},
        {"action_stats": action_stats, "metadata": metadata, "alignment": alignment, "vlm_summary": vlm},
    )


def test_fixed_prior_rollout_gate_blocks_legacy_4d_proxy_action_bridge(tmp_path):
    manifest = _write_manifest(tmp_path)
    simulator_reports, previous_reports = _write_status_reports(tmp_path)
    report = run_fixed_prior_rollout_readiness_gate(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=2,
        max_action_steps=4,
        env_action_dim=7,
        record_action_dim=4,
        simulator_reports=simulator_reports,
        previous_reports=previous_reports,
    )

    assert report["risk_gate_status"] == "red"
    assert report["rollout_diagnostic_authorized"] is False
    assert report["rollout_happened"] is False
    assert report["training_happened"] is False
    assert report["loss_computed"] is False
    assert report["record_builder_mode"] == "legacy_prefix_or_custom_dim"
    assert report["simulator_status"]["environment_plumbing_ready_for_tiny_diagnostic"] is True
    assert report["target_prior_status"]["available_at_test_time_under_current_assumption"] is True
    assert report["action_bridge_status"]["offline_record_action_dims"] == [4]
    assert report["action_bridge_status"]["env_action_dim"] == 7
    assert report["action_bridge_status"]["existing_adapter_supports_current_proxy_action"] is False
    assert any("4D" in blocker and "7D" in blocker for blocker in report["blockers"])


def test_fixed_prior_rollout_gate_preserves_hdf5_7d_action_path(tmp_path):
    manifest = _write_manifest(tmp_path)
    simulator_reports, previous_reports = _write_status_reports(tmp_path)
    report = run_fixed_prior_rollout_readiness_gate(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        max_pairs=2,
        max_action_steps=4,
        env_action_dim=7,
        simulator_reports=simulator_reports,
        previous_reports=previous_reports,
    )

    assert report["risk_gate_status"] == "green"
    assert report["rollout_diagnostic_authorized"] is True
    assert report["record_action_dim"] == 7
    assert report["record_builder_mode"] == "preserve_full_env_action_dim"
    bridge = report["action_bridge_status"]
    assert bridge["offline_record_action_dims"] == [7]
    assert bridge["offline_candidate_action_dims"] == [7]
    assert bridge["source_hdf5_action_dims"] == [7]
    assert bridge["preserves_full_env_action_dim_from_hdf5"] is True
    assert bridge["existing_adapter_supports_current_proxy_action"] is True
    assert bridge["adapter_metadata"]["adapter_mode"] == "passthrough_same_dim"
    assert bridge["gripper_mapping_resolved"] is True
    assert bridge["rotation_mapping_resolved"] is True
    assert bridge["coordinate_convention_resolved"] is True
    assert bridge["clipping_expected_from_records"] is False
    assert bridge["actionmap_tca_candidate_actions_distinct"] is True
    assert report["rollout_happened"] is False
    assert report["training_happened"] is False
    assert report["loss_computed"] is False


def test_existing_adapter_preserves_explicit_6d_to_7d_bridge():
    result = adapt_policy_action_to_env_action([0.1, -0.2, 0.3, 0.0, 0.01, -0.01], 7)

    assert len(result.values) == 7
    assert result.values[:6] == pytest.approx([0.1, -0.2, 0.3, 0.0, 0.01, -0.01])
    assert result.values[6] == 0.0
    assert result.metadata["adapter_mode"] == "policy_6d_delta_pose_plus_gripper_zero_hold"
    assert result.metadata["implicit_padding_performed"] is False
    assert result.metadata["truncation_performed"] is False


def test_fixed_prior_rollout_gate_script_refuses_execution_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for fixed-prior rollout gate script tests")
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
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_ROLLOUT": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "execution gates are set" in (result.stdout + result.stderr)
