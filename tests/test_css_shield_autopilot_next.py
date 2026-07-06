from tca_map.css_shield.autopilot_next import build_autopilot_state, select_next_milestone
from tca_map.css_shield.semantic_observability import resolve_semantic_targets


def test_autopilot_selects_state15_after_state1():
    selected = select_next_milestone({"last_completed_stage": "STATE 1"})

    assert selected["next_state"] == "STATE 1.5"
    assert selected["decision"] == "continue"


def test_autopilot_selects_state2_after_green_state15_report():
    report = {
        "result": {"passed": True},
        "policy": {"rollout_happened": True},
        "stage_c_controlled_diagnostic": {"decision": {"continue": True}},
    }

    selected = select_next_milestone({}, report)

    assert selected["next_state"] == "STATE 2"
    assert selected["decision"] == "continue"


def test_autopilot_state_records_diagnostic_only_policy():
    report = {
        "result": {"passed": True},
        "policy": {
            "rollout_happened": True,
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "gpu_jobs_performed": False,
            "downloads_performed": False,
            "heavy_model_imports_performed": False,
            "openvla_oft_executed": False,
        },
        "stage_c_controlled_diagnostic": {
            "decision": {"continue": True},
            "summary": {"comparison": {"full_vs_safety_wrong_target_delta": 1.0}},
        },
    }

    state = build_autopilot_state("abc123", report, {})

    assert state["current_main_commit"] == "abc123"
    assert state["rollout_happened"] is True
    assert state["training_happened"] is False
    assert state["loss_computed"] is False
    assert state["paper_grade_or_diagnostic"] == "diagnostic_only"


def test_semantic_resolver_uses_instruction_and_scene_names_only():
    resolved = resolve_semantic_targets(
        "turn on the stove and put the moka pot on it",
        ["moka_pot_1_pos", "chefmate_8_frypan_1_pos"],
        counterfactual_instruction="put the black bowl in the drawer",
    )

    assert resolved["intended_target"]["name"] == "moka_pot_1_pos"
    assert resolved["selected_distractor"]["name"] == "chefmate_8_frypan_1_pos"
    assert resolved["uses_bddl_metadata"] is False
    assert resolved["uses_eval_labels"] is False
