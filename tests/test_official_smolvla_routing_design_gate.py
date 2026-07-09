from tca_map.smolvla.official_libero_routing_design_gate import choose_routing_decision


def test_routing_gate_rejects_tiny_frame_oracle_headroom():
    assert (
        choose_routing_decision(
            sample_count=200,
            frame_improvement_abs=0.004,
            frame_improvement_rel=0.10,
            task_improvement_abs=0.010,
            task_improvement_rel=0.10,
            moira_kills_task_only=False,
        )
        == "NO_ROUTING_HEADROOM"
    )


def test_routing_gate_prefers_frame_conditional_when_task_headroom_is_weak():
    assert (
        choose_routing_decision(
            sample_count=200,
            frame_improvement_abs=0.020,
            frame_improvement_rel=0.20,
            task_improvement_abs=0.001,
            task_improvement_rel=0.01,
            moira_kills_task_only=True,
        )
        == "GO_DESIGN_FRAME_CONDITIONAL_ROUTING"
    )


def test_routing_gate_marks_task_routing_killed_by_moira():
    assert (
        choose_routing_decision(
            sample_count=200,
            frame_improvement_abs=0.020,
            frame_improvement_rel=0.20,
            task_improvement_abs=0.010,
            task_improvement_rel=0.10,
            moira_kills_task_only=True,
        )
        == "ROUTING_NOVELTY_KILLED_BY_MOIRA"
    )
