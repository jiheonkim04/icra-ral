import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import exact_init_replay_stabilization as stab


def test_exact_init_replay_decision_set_is_exact():
    assert stab.FINAL_DECISIONS == {
        "READY_FOR_METHOD_AFTER_STABLE_REPLAY_BASELINE",
        "READY_BUT_NEEDS_ACTION_VALIDITY_FIX",
        "OFFLINE_TO_CONTROL_GAP",
        "EXPERT_REPLAY_PROTOCOL_BLOCKED",
        "EXPERT_REPLAY_FIX_NEEDED",
        "TOO_HEAVY_LOCAL",
    }


def test_exact_init_replay_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(stab.RUN_GATE, raising=False)
    monkeypatch.delenv(stab.LEARNED_GATE, raising=False)
    args = argparse.Namespace(
        data_root=str(tmp_path / "data"),
        libero_root=str(tmp_path / "LIBERO"),
        robosuite_root=str(tmp_path / "robosuite"),
        adapter_dir=str(tmp_path / "adapter"),
        prior_result_path=str(tmp_path / "prior.json"),
        output_dir=str(tmp_path / "runs"),
        report_path=str(tmp_path / "report.json"),
        max_tasks=2,
        train_demos_per_task=5,
        eval_demos_per_task=2,
        records_per_demo=8,
        candidate_demos_per_task=4,
        max_replay_steps=8,
        post_signal_margin=0,
        camera_size=32,
    )

    report, code = stab.build_report(args)

    assert code != 0
    assert report["decision"] == "TOO_HEAVY_LOCAL"
    assert stab.RUN_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["replay_control_performed"] is False


def test_eligibility_requires_reward_or_success_and_finite_done():
    validity = {"shape_exactly_7d": True, "finite": True}
    no_done = {
        "reward_sum": 1.0,
        "final_success": True,
        "done_seen": False,
        "first_done_index": None,
        "passed": True,
        "error": None,
    }
    eligible, reasons = stab._eligibility(no_done, validity)

    assert eligible is False
    assert "finite done index missing" in reasons

    ok = {
        "reward_sum": 1.0,
        "final_success": True,
        "done_seen": True,
        "first_done_index": 12,
        "passed": True,
        "error": None,
    }
    eligible, reasons = stab._eligibility(ok, validity)

    assert eligible is True
    assert reasons == []


def test_learned_aggregate_uses_only_passed_cases_supplied_by_eligible_set():
    cases = [
        {
            "eligible_case": True,
            "results": {
                "expert": {"reward_sum": 1.0, "final_success": True, "done_seen": True},
                "mean_action": {"reward_sum": 0.0, "final_success": False, "done_seen": False},
            },
        }
    ]

    aggregate = stab._aggregate_learned(cases)

    assert aggregate["learned_aggregate_uses_only_eligible_cases"] is True
    assert aggregate["eligible_case_count"] == 1
    assert aggregate["expert"]["case_count"] == 1
    assert aggregate["mean_action"]["case_count"] == 1


def test_action_validity_audit_marks_adapter_gripper_clipping():
    cases = [
        {
            "task_name": "task",
            "demo_name": "demo_1",
            "offline_case_metrics": {
                "smolvla_7d_adapter": {
                    "action_validity": {
                        "clip_rate_element": 1.0 / 7.0,
                        "clip_rate_step": 1.0,
                        "controller_valid_rate_proxy": 0.0,
                        "dominant_clip_dim": 6,
                        "gripper_clip_rate": 1.0,
                    }
                }
            },
        }
    ]

    audit = stab._action_validity_audit({"cases": cases})

    assert np.isclose(audit["adapter_clip_rate_step_mean"], 1.0)
    assert audit["adapter_action_validity_fix_needed"] is True
    assert audit["mlp_replay_executed"] is False
