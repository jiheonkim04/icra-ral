from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from scripts import adjudicate_epoch9e_joint_certification as adjudicator
from scripts import build_epoch9e_joint_execution_seal as seal_builder
from scripts import run_epoch9e_joint_certification as runner


ROOT = Path(__file__).resolve().parents[1]


def passing_counts() -> dict:
    return {
        "finite_bounded_actions": 48,
        "intended_contact_or_excitation": 46,
        "both_candidates_excited": 22,
        "full_trajectory_lane_reachable": 48,
        "collisions": 0,
        "identity_swaps": 0,
        "falls": 0,
        "workspace_exits": 0,
        "unrecoverable_track_losses": 0,
        "rank_correct": 20,
        "rank_by_heavy_position": {
            "front": {"correct": 10, "total": 12},
            "back": {"correct": 10, "total": 12},
        },
        "exact_pair_correct_flips": 9,
        "completion_oracle": 20,
        "completion_by_heavy_position": {
            "front": {"success": 9, "total": 12},
            "back": {"success": 11, "total": 12},
        },
    }


def passing_statistics() -> dict:
    return {
        "one_sided_exact_sign_test_p": 0.009,
        "paired_student_t_95_interval_m": [0.001, 0.003],
        "adjusted_position_lane_order": {
            "valid": True,
            "estimate_m": 0.002,
            "hc3_95_interval_m": [0.0001, 0.004],
        },
    }


def test_joint_gates_encode_every_frozen_threshold_without_near_miss() -> None:
    controls = {"position_order_pass": True, "sham_pass": True}
    integrity = {
        "complete_unique_manifest": True,
        "nondrag_controller": True,
        "information_boundary": True,
        "trace_hashes": True,
        "execution_and_resource": True,
    }
    gates = adjudicator.joint_gates(passing_counts(), passing_statistics(), controls, integrity)
    assert all(gates.values())
    counts = passing_counts()
    counts["full_trajectory_lane_reachable"] = 47
    assert adjudicator.joint_gates(counts, passing_statistics(), controls, integrity)["full_trajectory_lane_reachable_48_of_48"] is False
    assert "near_miss" not in inspect.getsource(adjudicator.joint_gates)


def test_joint_primary_routes_only_through_frozen_nondrag_controller() -> None:
    source = inspect.getsource(runner.run_primary)
    assert "run_nondrag_probe" in source
    assert "campaign.probe_candidate" not in source
    assert source.index("run_nondrag_probe") < source.index("campaign.oracle_completion")
    assert "completion_target_slot_eval_only" in source


def test_joint_runner_is_one_shot_without_resume_surface() -> None:
    source = inspect.getsource(runner.main)
    assert "refusing to overwrite or resume" in source
    assert "--resume" not in source
    host = (ROOT / "scripts" / "run_epoch9e_joint_certification_host.ps1").read_text(encoding="utf-8")
    assert "resume = $false" in host
    assert "[switch]$Resume" not in host


def test_interval_helpers_fail_closed() -> None:
    assert adjudicator.paired_t_interval([]) == [None, None]
    assert adjudicator.interval_lower_positive([None, None]) is False
    assert adjudicator.interval_includes_zero([None, None]) is False
    assert adjudicator.interval_lower_positive([0.0, 1.0]) is False
    assert adjudicator.interval_includes_zero([-0.1, 0.1]) is True


def test_joint_seal_uses_byte_identity_to_head_not_client_worktree_metadata() -> None:
    path = ROOT / "scripts" / "run_epoch9e_joint_certification.py"
    working_digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    assert seal_builder.committed_sha256(path) == working_digest


def test_joint_execution_seal_binds_every_scientific_executable() -> None:
    seal = json.loads((ROOT / "reports" / "epoch9e_joint_execution_seal.json").read_text(encoding="utf-8"))
    bindings = {
        "joint_protocol": ROOT / seal["joint_protocol_path"],
        "mechanics_execution_seal": ROOT / seal["mechanics_execution_seal_path"],
        "mechanics_adjudication": ROOT / seal["mechanics_adjudication_path"],
        "runner": ROOT / seal["runner_path"],
        "adjudicator": ROOT / seal["adjudicator_path"],
        "host_wrapper": ROOT / seal["host_wrapper_path"],
        "controller": ROOT / seal["controller_path"],
        "original_runner": ROOT / seal["original_runner_path"],
    }
    for name, path in bindings.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == seal[f"{name}_sha256"]
    assert seal["joint_outcomes_accessed_before_seal"] is False
    assert seal["controller_frozen_after_passing_mechanics_smoke"] is True
    assert seal["one_shot"]["panels"] == 1
    assert seal["runtime"]["resume_authorized"] is False
