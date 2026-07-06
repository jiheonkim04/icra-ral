import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.execspec import baseline_dominance_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "167_execspec_baseline_dominance_audit.ps1"


def _result(variant: str, reward: float, success: bool) -> dict:
    return {
        "variant": variant,
        "reward_sum": reward,
        "final_success": success,
        "done_seen": success,
        "first_done_index": 10 if success else None,
    }


def _case(demo: str, mismatch: str, *, global_recovers: bool = False, gripper_recovers: bool = False) -> dict:
    return {
        "eval_demo_path": f"/tmp/{demo}.hdf5",
        "task_id": demo,
        "mismatch_type": mismatch,
        "summary": {
            "success_degraded": True,
            "reward_degraded": True,
            "simple_baseline_matches_full": global_recovers,
        },
        "replay_results": [
            _result("correct_7d_expert_action_replay", 1.0, True),
            _result("wrong_executable_spec_replay", 0.0, False),
            _result("identity_no_repair", 0.0, False),
            _result("clipping_only", 0.0, False),
            _result("global_affine_calibration", 1.0 if global_recovers else 0.0, global_recovers),
            _result("gripper_only_calibration", 1.0 if gripper_recovers else 0.0, gripper_recovers),
            _result("diagonal_affine_calibration", 1.0, True),
            _result("full_execspec_repair", 1.0, True),
        ],
    }


def _action_case(case: dict, recoveries: dict[str, float]) -> dict:
    return {
        "eval_demo_path": case["eval_demo_path"],
        "mismatch_type": case["mismatch_type"],
        "repair_methods": {
            method: {"recovery_fraction": recoveries.get(method, 0.0)}
            for method in audit.METHODS
        },
    }


def _state3_report() -> dict:
    cases = [
        _case("demo_global", "global_action_scale_mismatch", global_recovers=True),
        _case("demo_grip", "gripper_sign_flip", gripper_recovers=True),
        _case("demo_trans", "translation_scale_mismatch"),
    ]
    action_cases = []
    for case in cases:
        action_cases.append(
            _action_case(
                case,
                {
                    "diagonal_affine_calibration": 1.0,
                    "full_execspec_repair": 1.0,
                    "global_affine_calibration": 1.0 if case["mismatch_type"] == "global_action_scale_mismatch" else 0.0,
                    "gripper_only_calibration": 1.0 if case["mismatch_type"] == "gripper_sign_flip" else 0.0,
                },
            )
        )
    return {
        "evidence_label": "synthetic_state3",
        "exact_init_replay": {"cases": cases},
        "heldout_action_metrics": {"cases": action_cases},
    }


def test_diagonal_baseline_dominance_kills_broad_claim():
    report = audit.build_audit_report(_state3_report())

    assert report["inputs_summary"]["degraded_case_count"] == 3
    assert report["method_aggregates"]["full_execspec_repair"]["success_recovery_rate"] == 1.0
    assert report["method_aggregates"]["diagonal_affine_calibration"]["success_recovery_rate"] == 1.0
    assert report["decision"]["best_single_simple_baseline"] == "diagonal_affine_calibration"
    assert report["decision"]["final_decision"] == "kill"
    assert report["decision"]["simple_baselines_explain_result"] is True
    assert report["decision"]["repair_selector_routing_meaningful"] is False


def test_matched_cases_and_selector_are_reported():
    report = audit.build_audit_report(_state3_report())

    assert len(report["matched_cases"]) == 1
    assert report["matched_cases"][0]["mismatch_type"] == "global_action_scale_mismatch"
    assert report["matched_cases"][0]["simple_baseline_match_methods"] == ["global_affine_calibration"]
    assert report["mismatch_aware_selector"]["success_recovery_rate"] == 1.0
    assert report["repair_routing_opportunity"]["gripper_sign_flip"]["selector_rule"] == "gripper_only_calibration"


def test_audit_script_runs_report_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for audit script tests")
    state3 = tmp_path / "state3.json"
    state3.write_text(json.dumps(_state3_report()), encoding="utf-8")
    report_json = tmp_path / "audit.json"
    env = os.environ.copy()
    for gate in audit.FORBIDDEN_GATES:
        env.pop(gate, None)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-State3ReportPath",
            str(state3),
            "-JsonReportPath",
            str(report_json),
            "-MarkdownReportPath",
            str(tmp_path / "audit.md"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_json.read_text(encoding="utf-8-sig"))
    assert data["result"]["passed"] is True
    assert data["policy"]["new_replay_or_rollout_performed"] is False
    assert data["decision"]["final_decision"] == "kill"
