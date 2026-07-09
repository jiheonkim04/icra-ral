import argparse

import numpy as np

from tca_map.smolvla_lora_baseline import standard_replay_baseline as baseline


def test_standard_replay_decision_set_is_exact():
    assert baseline.FINAL_DECISIONS == {
        "READY_FOR_RA_L_METHOD_AFTER_STANDARD_BASELINE",
        "READY_BUT_NEEDS_ACTION_RANGE_FIX",
        "OFFLINE_TO_CONTROL_GAP",
        "MEAN_OR_MLP_REPLAY_DOMINATED",
        "EXPERT_REPLAY_UNSTABLE",
        "TOO_HEAVY_LOCAL",
    }


def test_standard_replay_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(baseline.RUN_GATE, raising=False)
    monkeypatch.delenv(baseline.TRAINING_GATE, raising=False)
    monkeypatch.delenv(baseline.REPLAY_GATE, raising=False)
    args = argparse.Namespace(
        data_root=str(tmp_path / "data"),
        libero_root=str(tmp_path / "LIBERO"),
        robosuite_root=str(tmp_path / "robosuite"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        output_dir=str(tmp_path / "runs"),
        report_path=str(tmp_path / "report.json"),
        max_tasks=2,
        train_demos_per_task=5,
        eval_demos_per_task=2,
        replay_demos_per_task=1,
        records_per_demo=8,
        adapter_steps=1,
        adapter_hidden_dim=8,
        mlp_steps=1,
        mlp_hidden_dim=8,
        learning_rate=5e-3,
        lora_learning_rate=1e-3,
        max_replay_steps=8,
        camera_size=32,
    )

    report, code = baseline.build_report(args)

    assert code != 0
    assert report["decision"] == "TOO_HEAVY_LOCAL"
    assert baseline.RUN_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["training_performed"] is False


def test_action_validity_reports_per_dimension_clip():
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],
            [1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        ],
        dtype=np.float32,
    )

    validity = baseline._action_validity(actions)

    assert validity["clip_rate_step"] == 1.0
    assert validity["per_dim_clip_rate"][0] == 0.5
    assert validity["per_dim_clip_rate"][6] == 0.5
    assert validity["gripper_clip_rate"] == 0.5


def test_replay_aggregate_uses_only_judgeable_cases_for_learned_policies():
    cases = [
        {
            "expert_ok_for_judging": False,
            "results": {
                "expert": {"reward_sum": 0.0, "final_success": False, "done_seen": False},
                "mean_action": {"reward_sum": 1.0, "final_success": True, "done_seen": True},
            },
        },
        {
            "expert_ok_for_judging": True,
            "results": {
                "expert": {"reward_sum": 1.0, "final_success": True, "done_seen": True},
                "mean_action": {"reward_sum": 0.0, "final_success": False, "done_seen": False},
            },
        },
    ]

    aggregate = baseline._aggregate_replay(cases)

    assert aggregate["expert"]["case_count"] == 2
    assert aggregate["mean_action"]["case_count"] == 1
    assert aggregate["expert_all_succeeded"] is False
