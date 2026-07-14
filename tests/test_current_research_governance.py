import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_active_state_records_closed_rac_stage_b_without_cycle_cap() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 7
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "DAGR_VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["current_stage"] == "epoch_4_cycle_7_dagr_adapter_training_pending"
    assert state["method"] == "DAGR-VLA"
    assert state["proposal_hash"] == "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89"
    assert state["prototype_protocol"] == "reports/dagr_vla/prototype_protocol.md"
    assert "epoch_4_cycle_3_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_development_audit_passed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_4_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_design_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_5_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_b_pending" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_stage_b_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_5_rac_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "post_rac_governance_update_pending" in state["completed_stages"]
    assert "post_rac_governance_update_installed" in state["completed_stages"]
    assert "epoch_4_cycle_6_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_6_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_selected_config_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_adapter_training_runner_validated" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_adapter_training_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_checkpoints_verified" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_a_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_a_policy_preflight_passed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_b_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_b_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_7_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_7_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_selected_config_frozen" in state["completed_stages"]
    assert state["task_reset_manifest"] == "reports/mtf_vla/stage_b_manifest.json"
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["paired_cases_per_policy"] == 10
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["reset_seeds"] == [20261201, 20261202]
    assert state["checkpoint_path"] == "runs/mtf_vla_checkpoints/mtf_r20_ret100"
    assert state["stage_a_result_json"] == "reports/mtf_vla/stage_a_result.json"
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["completed_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["final_decision"] == "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["policy_successes"]["mtf_full"]["successes"] == 7
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["policy_successes"]["frameskip_proxy_lora"]["successes"] == 8
    assert state["stage_b_result_json"] == "reports/mtf_vla/stage_b_result.json"
    assert state["epoch_4_cycle_6_mtf_stage_b_manifest"]["planned_episode_count"] == 200
    assert state["epoch_4_cycle_6_mtf_stage_b_manifest"]["paired_cases_per_policy"] == 40
    assert state["epoch_4_cycle_6_mtf_stage_b_manifest"]["reset_seeds"] == [20261203, 20261204]
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["final_decision"] == "MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["valid_current_formulation_kill"] is True
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["completed_episode_count"] == 200
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["mtf_full_successes"] == 26
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["mtf_no_retention_ablation_successes"] == 32
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["paired_delta_vs_no_retention_ablation"] == -0.15
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["paired_ci_vs_no_retention_ablation"] == [-0.275, -0.025]
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["simple_baseline_explains_method"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["selection_decision"] == "SELECT_DAGR_VLA"
    assert state["epoch_4_cycle_7_pre_stage_0"]["closest_prior"] == "DAM-VLA"
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_score"] == 89
    assert state["epoch_4_cycle_7_pre_stage_0"]["reviewer_attack"] == "reports/dagr_vla/reviewer_attack.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["epoch_4_cycle_7_pre_stage_0"]["researcher_rebuttal"] == "reports/dagr_vla/researcher_rebuttal.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["rebuttal_decision"] == "DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert state["epoch_4_cycle_7_pre_stage_0"]["mathematical_audit"] == "reports/dagr_vla/mathematical_mechanism_audit.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["preregistration"] == "reports/dagr_vla/preregistration.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["prototype_protocol"] == "reports/dagr_vla/prototype_protocol.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_7_pre_stage_0"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_config"] == "dagr_a020_route_mlp"
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_residual_alpha"] == 0.2
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_route_architecture"] == "mlp"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_allowed"] is False
    assert state["epoch_4_cycle_7_pre_stage_0"]["first_comparison_policies"] == [
        "frozen_smolvla",
        "dam_static_component_proxy",
        "dagr_full",
        "dagr_no_dynamic_route_ablation",
        "gripper_transition_heuristic",
    ]
    assert state["epoch_4_cycle_7_pre_stage_0"]["closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["selected_config"] == "dagr_a020_route_mlp"
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["scoreable_development_records"] == 1600
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["validation_any_route_fraction"] == 0.865
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["selected_action_validity"] == 1.0
    assert state["valid_final_states"] == ALLOWED_FINAL_STATES
    assert state["epoch_2_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    assert state["epoch_2_cycle_2_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    assert state["epoch_2_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_2_cycle_3_outcome"]["ocfn_full_successes"] == 26
    assert state["epoch_2_cycle_3_outcome"]["zero_noise_smolvla_successes"] == 27
    assert state["epoch_2_cycle_3_outcome"]["paired_upper_ci_vs_strongest_baseline"] == 0.0625
    assert state["epoch_3_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    assert state["epoch_3_cycle_1_outcome"]["cbfd_full_successes"] == 0
    assert state["epoch_3_cycle_1_outcome"]["frozen_smolvla_successes"] == 7
    assert state["epoch_3_cycle_2_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_2_outcome"]["scvc_full_successes"] == 11
    assert state["epoch_3_cycle_2_outcome"]["shifted_frozen_smolvla_successes"] == 20
    assert state["epoch_3_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_3_outcome"]["pse_full_successes"] == 50
    assert state["epoch_3_cycle_3_outcome"]["bright_single_successes"] == 51
    assert state["epoch_3_cycle_3_outcome"]["validation_unique_keys"] == 400
    assert state["epoch_3_synthesis"]["next_epoch"] == 4
    assert state["epoch_4_cycle_1_outcome"]["final_decision"] == "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_4_cycle_1_outcome"]["rcv_full_successes"] == 20
    assert state["epoch_4_cycle_1_outcome"]["rcv_no_context_ablation_successes"] == 24
    assert state["epoch_4_cycle_1_outcome"]["stateless_first_action_successes"] == 24
    assert state["epoch_4_cycle_1_outcome"]["valid_current_formulation_kill"] is True
    assert state["epoch_4_cycle_2_outcome"]["final_decision"] == "STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION"
    assert state["epoch_4_cycle_2_outcome"]["valid_current_formulation_non_go"] is True
    assert state["epoch_4_cycle_2_outcome"]["episode_count"] == 290
    assert state["epoch_4_cycle_2_outcome"]["cavm_full_successes"] == 24
    assert state["epoch_4_cycle_2_outcome"]["nearest_success_replay_successes"] == 23
    assert state["epoch_4_cycle_2_outcome"]["manifest_unique_variant_task_identity_keys"] == 290
    assert state["epoch_4_cycle_2_outcome"]["manifest_bad_pairs"] == 0
    assert state["epoch_4_cycle_3_outcome"]["final_decision"] == "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    assert state["epoch_4_cycle_3_outcome"]["valid_current_formulation_kill"] is True
    assert state["epoch_4_cycle_3_outcome"]["stage_b_completed"] is True
    assert state["epoch_4_cycle_3_outcome"]["episode_count"] == 200
    assert state["epoch_4_cycle_3_outcome"]["fang_full_successes"] == 11
    assert state["epoch_4_cycle_3_outcome"]["base_smolvla_successes"] == 16
    assert state["epoch_4_cycle_3_outcome"]["afil_local_proxy_successes"] == 15
    assert state["epoch_4_cycle_3_outcome"]["fang_no_failure_ablation_successes"] == 11
    assert state["epoch_4_cycle_3_outcome"]["paired_delta_vs_base_smolvla"] == -0.125
    assert state["epoch_4_cycle_4_pre_stage_0"]["method"] == "EvoState-VLA"
    assert state["epoch_4_cycle_4_pre_stage_0"]["selection_decision"] == "SELECT_EVOSTATE_VLA"
    assert state["epoch_4_cycle_4_pre_stage_0"]["candidate_generation"] == "reports/epoch_4_cycle_4_candidate_generation.md"
    assert state["epoch_4_cycle_4_pre_stage_0"]["mathematical_audit"] == "reports/evostate_vla/mathematical_mechanism_audit.md"
    assert state["epoch_4_cycle_4_outcome"]["final_decision"] == "AUDIT_STOP_DESIGN_FAILURE"
    assert state["epoch_4_cycle_4_outcome"]["closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_4_outcome"]["transition_pairs"] == 10769
    assert state["epoch_4_cycle_4_outcome"]["transition_improvement_vs_actionless"] == 0.024689372539669806
    assert state["epoch_4_cycle_4_outcome"]["required_transition_improvement_vs_actionless"] == 0.05
    assert state["epoch_4_cycle_5_pre_stage_a"]["method"] == "RAC-VLA"
    assert state["epoch_4_cycle_5_pre_stage_a"]["proposal_hash"] == "71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F"
    assert state["epoch_4_cycle_5_pre_stage_a"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_5_pre_stage_a"]["selected_config"] == "rac_h4_a0.05"
    assert state["epoch_4_cycle_5_pre_stage_a"]["selected_residual_alpha"] == 0.05
    assert state["epoch_4_cycle_5_stage_a_outcome"]["final_decision"] == "STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_5_stage_a_outcome"]["stage_a_completed"] is True
    assert state["epoch_4_cycle_5_stage_a_outcome"]["episode_count"] == 50
    assert state["epoch_4_cycle_5_stage_a_outcome"]["exceptions"] == 0
    assert state["epoch_4_cycle_5_stage_a_outcome"]["rac_full_successes"] == 0
    assert state["epoch_4_cycle_5_stage_a_outcome"]["base_smolvla_shifted_successes"] == 0
    assert state["epoch_4_cycle_5_stage_a_outcome"]["rac_no_consequence_ablation_successes"] == 0
    assert state["epoch_4_cycle_5_stage_a_outcome"]["reflective_history_proxy_successes"] == 1
    assert state["epoch_4_cycle_5_stage_a_outcome"]["online_diagonal_inverse_gain_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["final_decision"] == "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    assert state["epoch_4_cycle_5_stage_b_outcome"]["stage_b_completed"] is True
    assert state["epoch_4_cycle_5_stage_b_outcome"]["episode_count"] == 200
    assert state["epoch_4_cycle_5_stage_b_outcome"]["exceptions"] == 0
    assert state["epoch_4_cycle_5_stage_b_outcome"]["manifest_unique_variant_task_identity_keys"] == 200
    assert state["epoch_4_cycle_5_stage_b_outcome"]["manifest_duplicate_keys"] == 0
    assert state["epoch_4_cycle_5_stage_b_outcome"]["manifest_bad_pairs"] == 0
    assert state["epoch_4_cycle_5_stage_b_outcome"]["rac_full_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["base_smolvla_shifted_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["reflective_history_proxy_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["rac_no_consequence_ablation_successes"] == 2
    assert state["epoch_4_cycle_5_stage_b_outcome"]["online_diagonal_inverse_gain_successes"] == 2
    assert state["epoch_4_cycle_6_pre_stage_0"]["method"] == "MTF-VLA"
    assert state["epoch_4_cycle_6_pre_stage_0"]["selection_decision"] == "SELECT_MTF_VLA"
    assert state["epoch_4_cycle_6_pre_stage_0"]["proposal_hash"] == "11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31"
    assert state["epoch_4_cycle_6_pre_stage_0"]["prototype_protocol"] == "reports/mtf_vla/prototype_protocol.md"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["selected_config"] == "mtf_r20_ret100"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["selected_retention_coefficient"] == 1.0
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_runner_validated"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_happened"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_final_decision"] == "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_checkpoint_manifest"] == "reports/mtf_vla/adapter_checkpoint_manifest.json"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["final_decision"] == "MTF_ADAPTER_TRAINING_PLAN_READY"
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["stage_a_allowed"] is False
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["jobs"][0]["variant"] == "mtf_full"
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["jobs"][0]["event_count"] == 567
    assert state["epoch_4_cycle_6_mtf_adapter_checkpoint_outcome"]["final_decision"] == "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY"
    assert state["epoch_4_cycle_6_mtf_adapter_checkpoint_outcome"]["closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_6_mtf_adapter_checkpoint_outcome"]["frameskip_proxy_distinct_from_no_retention"] is True

    mtf_manifest = json.loads((REPO_ROOT / "reports" / "mtf_vla" / "selected_training_manifest.json").read_text(encoding="utf-8"))
    no_retention_keys = {
        row["key"] for row in mtf_manifest["variants"]["mtf_no_retention_ablation"]["high_milestone_frames"]
    }
    frameskip_keys = {
        row["key"] for row in mtf_manifest["variants"]["frameskip_proxy_lora"]["selected_frames"]
    }
    assert len(frameskip_keys) == 240
    assert frameskip_keys != no_retention_keys
    assert len(frameskip_keys & no_retention_keys) == 96


def test_epoch_1_corrected_adjudication_records_all_cycles() -> None:
    adjudication = (REPO_ROOT / "reports" / "epoch_1_corrected_adjudication.md").read_text(encoding="utf-8")

    assert "DICD-VLA" in adjudication
    assert "UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED" in adjudication
    assert "FEDO-VLA" in adjudication
    assert "VALID_CURRENT_FORMULATION_KILL" in adjudication
    assert "GCAP-VLA" in adjudication
    assert "UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED" in adjudication


def test_post_cavm_performance_research_design_governance_is_active() -> None:
    governance = (REPO_ROOT / "reports" / "current_research_governance.md").read_text(encoding="utf-8")

    assert "Post-CAVM Performance-Oriented Research Design Governance" in governance
    assert "Post-RAC Honest Positive-Result Governance" in governance
    assert "MAXIMIZE_THE_PROBABILITY_OF_AN_HONEST_PAPER_WORTHY_POSITIVE_RESULT" in governance
    assert "`DISCOVERY`" in governance
    assert "`VALIDATION`" in governance
    assert "`CONFIRMATORY_TEST`" in governance
    assert "DISCOVERY PARTITION" in governance
    assert "DEVELOPMENT / VALIDATION PARTITION" in governance
    assert "CONFIRMATORY TEST PARTITION" in governance
    assert "closest external prior" in governance
    assert "positive result that prior already demonstrates" in governance
    assert "positive external-prior anchor" in governance
    assert "bounded validation search" in governance
    assert "NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE" in governance
    assert "DATA_OR_SUPERVISION_FAILURE" in governance
    assert "GENUINE_METHOD_KILL" in governance
    assert "SIMPLE_BASELINE_EXPLAINS_METHOD" in governance
    assert "KEY_COMPONENT_NOT_USEFUL" in governance
    assert "detached durable execution" in governance
    assert "resume only missing evaluation keys" in governance
    assert "no more than `6` total configurations" in governance
    assert "No more than one mandatory simple killer baseline" in governance
    assert "AUTHOR_STATED" in governance
    assert "INDEPENDENTLY_INFERRED" in governance
    assert "CROSS_PAPER_SYNTHESIZED" in governance
    assert "future-work text" in governance
    assert "mathematical_mechanism_audit.md" in governance
    assert "identity-preserving integration audit" in governance
    assert "variables and tensor shapes" in governance
    assert "Confirmatory outcomes may not be used to retune the same method" in governance
    assert "Do not compute KL directly between deterministic 7D action vectors" in governance
