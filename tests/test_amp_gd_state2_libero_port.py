from argparse import Namespace

from tca_map.amp_gd.state2_libero_port import build_report, run_toy_robustness_audit


def test_toy_utility_drop_negative_means_amp_improvement():
    audit = run_toy_robustness_audit(20, (11, 23))

    utility = audit["utility_metric_audit"]

    assert utility["metric_bug_found"] is False
    assert utility["recomputed_utility_drop_vs_no_probe"] < 0.0
    assert "higher" in utility["state1_negative_drop_means"]


def test_toy_audit_marks_simple_informative_probe_match():
    audit = run_toy_robustness_audit(20, (11, 23))

    decision = audit["toy_route_decision"]

    assert decision["kill_toy_as_main_evidence"] is True
    assert decision["deterministic_heuristic_matches_amp"] is True


def test_state2_report_can_skip_libero_but_kills_route():
    args = Namespace(
        toy_trials=20,
        seeds="11,23",
        manifest="unused.json",
        case_index=0,
        libero_root="unused",
        robosuite_root="unused",
        camera_size=64,
        max_steps=5,
        seed=17,
        probe_scale=0.035,
        commit_scale=0.055,
        max_translation_norm=0.08,
        run_libero_probe=False,
        skip_libero=True,
        report_json="unused.json",
        report_md="unused.md",
        inventory_json="unused_inventory.json",
    )

    report = build_report(args)

    assert report["result"]["passed"] is True
    assert report["continue_or_kill"]["decision"] == "kill_or_reframe"
    assert report["policy"]["toy_rollout_control_metric_happened"] is True
    assert report["policy"]["libero_simulator_control_metric_happened"] is False
