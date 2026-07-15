import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]
PESA_PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"
EAC_PROPOSAL_HASH = "A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E"
G3P_PROPOSAL_HASH = "BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71"


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_active_state_records_closed_rac_stage_b_without_cycle_cap() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 11
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["current_stage"] == "epoch_4_cycle_11_g3p_reviewer_attack_completed"
    assert state["method"] == "G3P-VLA"
    assert state["method_identity"] == "G3P-VLA"
    assert state["proposal_hash"] == G3P_PROPOSAL_HASH
    assert state["prototype_protocol"] is None
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
    assert "epoch_4_cycle_7_dagr_policy_identities_verified" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_a_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_a_policy_preflight_passed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_b_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_stage_b_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_7_dagr_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_8_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_8_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_selected_config_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_policy_identities_verified" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_a_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_a_policy_preflight_passed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_a_rollout_launched" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_stage_a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_8_marc_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_9_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_9_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_proposal_pending" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_9_pesa_design_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_10_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_10_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_proposal_pending" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_runtime_queue_check_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_selected_config_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_policy_preflight_passed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_runner_validated" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_rollout_launched" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_b_manifest_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_b_rollout_launched" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_stage_b_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_10_eac_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_11_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_11_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_reviewer_attack_completed" in state["completed_stages"]
    assert state["epoch_4_cycle_9_pre_stage_0"]["selection_decision"] == "SELECT_PESA_VLA"
    assert state["epoch_4_cycle_9_pre_stage_0"]["candidate_generation"] == "reports/epoch_4_cycle_9_candidate_generation.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["prior_mechanism_map"] == "reports/epoch_4_cycle_9_prior_mechanism_map.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["closest_prior"] == "PriorVLA"
    assert state["epoch_4_cycle_9_pre_stage_0"]["secondary_priors"] == ["LoRA-SP", "VLA-GSE"]
    assert state["epoch_4_cycle_9_pre_stage_0"]["selected_score"] == 90
    assert state["epoch_4_cycle_9_pre_stage_0"]["proposal"] == "reports/pesa_vla/researcher_proposal.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["proposal_hash"] == PESA_PROPOSAL_HASH
    assert state["epoch_4_cycle_9_pre_stage_0"]["reviewer_attack"] == "reports/pesa_vla/reviewer_attack.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["epoch_4_cycle_9_pre_stage_0"]["researcher_rebuttal"] == "reports/pesa_vla/researcher_rebuttal.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["rebuttal_decision"] == "PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert state["epoch_4_cycle_9_pre_stage_0"]["mathematical_audit"] == "reports/pesa_vla/mathematical_mechanism_audit.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["mathematical_audit_decision"] == "PESA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert state["epoch_4_cycle_9_pre_stage_0"]["preregistration"] == "reports/pesa_vla/preregistration.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["prototype_protocol"] == "reports/pesa_vla/prototype_protocol.md"
    assert state["epoch_4_cycle_9_pre_stage_0"]["development_audit"] == "reports/pesa_vla/development_audit.json"
    assert state["epoch_4_cycle_9_pre_stage_0"]["stage_0_decision"] == "DESIGN_FAILURE"
    assert state["epoch_4_cycle_9_pre_stage_0"]["stage_0_query_probe_accuracy_margin"] == -0.07750000000000001
    assert state["epoch_4_cycle_9_pre_stage_0"]["stage_0_training_happened"] is False
    assert state["epoch_4_cycle_9_pre_stage_0"]["stage_0_closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_9_pre_stage_0"]["first_comparison_policies"] == [
        "frozen_smolvla",
        "priorvla_style_proxy",
        "pesa_full",
        "pesa_no_spectral_no_prior_query_ablation",
        "standard_lora_or_clean_retention_baseline",
    ]
    assert state["epoch_4_cycle_9_pre_stage_0"]["closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_9_pre_stage_0"]["confirmatory_test_tuning_happened"] is False
    outcome = state["epoch_4_cycle_9_pesa_development_outcome"]
    assert outcome["final_decision"] == "DESIGN_FAILURE"
    assert outcome["stage_0_completed"] is True
    assert outcome["closed_loop_experiment_happened"] is False
    assert outcome["training_happened"] is False
    assert outcome["query_probe_accuracy"] == 0.5225
    assert outcome["query_probe_majority_accuracy"] == 0.6
    assert outcome["query_probe_accuracy_margin"] == -0.07750000000000001
    assert outcome["hard_stop_reasons"] == ["query probe accuracy margin below minimum: -0.077500"]
    assert outcome["valid_current_formulation_kill"] is False
    eac = state["epoch_4_cycle_10_pre_proposal"]
    assert eac["method"] == "EAC-VLA"
    assert eac["selection_decision"] == "SELECT_EAC_VLA"
    assert eac["candidate_generation"] == "reports/epoch_4_cycle_10_candidate_generation.md"
    assert eac["prior_mechanism_map"] == "reports/epoch_4_cycle_10_prior_mechanism_map.md"
    assert eac["candidate_count"] == 3
    assert eac["closest_prior"] == "Adaptive Action Chunking"
    assert eac["secondary_priors"] == ["AR-VLA", "AC2-VLA"]
    assert eac["selected_score"] == 93
    assert eac["selected_contribution_type"] == "PRIOR_EXTENSION"
    assert eac["first_comparison_policies"] == [
        "frozen_smolvla_fixed_queue",
        "aac_entropy_proxy",
        "eac_full",
        "eac_no_calibration_no_hysteresis_ablation",
        "fixed_short_replan_baseline",
    ]
    assert eac["stage_0_required"] is True
    assert eac["closed_loop_experiment_happened"] is False
    assert eac["confirmatory_test_tuning_happened"] is False
    assert eac["training_happened"] is False
    assert eac["proposal"] == "reports/eac_vla/researcher_proposal.md"
    assert eac["proposal_hash"] == EAC_PROPOSAL_HASH
    assert eac["reviewer_attack"] == "reports/eac_vla/reviewer_attack.md"
    assert eac["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert eac["researcher_rebuttal"] == "reports/eac_vla/researcher_rebuttal.md"
    assert eac["rebuttal_decision"] == "EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert eac["mathematical_audit"] == "reports/eac_vla/mathematical_mechanism_audit.md"
    assert eac["mathematical_audit_decision"] == "EAC_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert eac["preregistration"] == "reports/eac_vla/preregistration.md"
    assert eac["preregistration_decision"] == "EAC_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert eac["prototype_protocol"] == "reports/eac_vla/prototype_protocol.md"
    assert eac["prototype_protocol_decision"] == "EAC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert eac["stage_0_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert eac["stage_0_audit"] == "reports/eac_vla/stage_0_audit.json"
    assert eac["stage_0_completed"] is True
    assert eac["stage_0_scoreable_validation_records"] == 2000
    assert eac["stage_0_validation_unique_frames"] == 400
    assert eac["stage_0_reserved_records_not_used_for_tuning"] == 6000
    assert eac["stage_0_validation_reserved_frame_overlap"] == 0
    assert eac["stage_0_validation_reserved_sample_overlap"] == 0
    assert eac["stage_0_first_two_dispersion_p95"] == 0.0007983036317792467
    assert eac["stage_0_first_two_dispersion_nonzero_fraction"] == 1.0
    assert eac["stage_0_commitment_counts"] == {"2": 136, "8": 132, "50": 132}
    assert eac["stage_0_max_commitment_share"] == 0.34
    assert eac["stage_0_passthrough_max_abs_error"] == 5.07000000038449e-07
    assert eac["stage_0_queue_helper_present"] is True
    assert eac["stage_0_chunk_shape_ok"] is True
    assert eac["stage_0_full_chunk_values_available_in_artifact"] is False
    assert eac["stage_0_runtime_full_chunk_check_required_before_validation_search"] is True
    assert eac["stage_0_closed_loop_experiment_happened"] is False
    assert eac["stage_0_training_happened"] is False
    assert eac["stage_0_validation_search_happened"] is False
    assert eac["stage_0_confirmatory_test_tuning_happened"] is False
    assert eac["runtime_queue_check_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert eac["runtime_queue_check_chunk_shape"] == [50, 7]
    assert eac["runtime_queue_check_select_action_vs_chunk0_max_abs_diff"] == 0.0
    assert eac["runtime_queue_check_queue_owner_present"] is True
    assert eac["runtime_queue_check_queue_len_before_select_action"] == 0
    assert eac["runtime_queue_check_queue_len_after_select_action"] == 49
    assert eac["runtime_queue_check_all_prefixes_value_preserving"] is True
    assert eac["runtime_queue_check_max_prefix_abs_diff"] == 0.0
    assert eac["runtime_queue_check_max_queue_pop_abs_diff"] == 0.0
    assert eac["validation_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert eac["validation_search_happened"] is True
    assert eac["validation_confirmatory_records_used_for_tuning"] is False
    assert eac["tried_config_count"] == 6
    assert eac["selected_config"] == "eac_q33_aggressive_1_4_50"
    assert eac["selected_validation_score"] == 0.7530415186081504
    assert eac["selected_commitment_counts"] == {"1": 132, "4": 136, "50": 132}
    assert eac["selected_policy_calls_per_step_proxy"] == 0.4216
    assert eac["selected_risk_exposure_reduction_proxy"] == 0.9032794643799159
    assert eac["stage_a_manifest_allowed"] is True
    assert eac["stage_a_manifest_decision"] == "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING"
    assert eac["stage_a_manifest"] == "reports/eac_vla/stage_a_manifest.json"
    assert eac["stage_a_manifest_canonical_payload_sha256"] == "63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C"
    assert eac["stage_a_planned_episode_count"] == 50
    assert eac["stage_a_paired_cases_per_policy"] == 10
    assert eac["stage_a_reset_seeds"] == [20261211, 20261212]
    assert eac["stage_a_policy_order"] == [
        "frozen_smolvla_fixed_queue",
        "aac_entropy_proxy",
        "eac_full",
        "eac_no_calibration_no_hysteresis_ablation",
        "fixed_short_replan_baseline",
    ]
    assert eac["stage_a_preflight_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert eac["stage_a_preflight_policy_count"] == 5
    assert eac["stage_a_preflight_checkpoint_policy_count"] == 0
    assert eac["stage_a_preflight_cuda_ok"] is True
    assert eac["stage_a_preflight_policy_output_shape"] == [50, 7]
    assert eac["stage_a_preflight_all_policy_prefixes_value_preserving"] is True
    assert eac["stage_a_rollout_allowed"] is True
    assert eac["stage_a_runner_implementation_required"] is False
    assert eac["stage_a_runner_validation_decision"] == "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT"
    assert eac["stage_a_runner_policy_count"] == 5
    assert eac["stage_a_runner_runtime_samples_for_dynamic_schedulers"] == 2
    assert eac["stage_a_runner_all_policy_prefixes_value_preserving"] is True
    assert eac["stage_a_runner_any_action_values_modified"] is False
    eac_outcome = state["epoch_4_cycle_10_eac_development_outcome"]
    assert eac_outcome["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert eac_outcome["hard_stop_reasons"] == []
    assert eac_outcome["runtime_full_chunk_check_required_before_validation_search"] is True
    assert eac_outcome["runtime_queue_check_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert eac_outcome["validation_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert eac_outcome["selected_config"] == "eac_q33_aggressive_1_4_50"
    assert eac_outcome["stage_a_manifest_decision"] == "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING"
    assert eac_outcome["stage_a_preflight_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert eac_outcome["stage_a_preflight_all_policy_prefixes_value_preserving"] is True
    assert eac_outcome["valid_current_formulation_kill"] is False
    queue_check = state["epoch_4_cycle_10_eac_runtime_queue_check"]
    assert queue_check["final_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert queue_check["closed_loop_experiment_happened"] is False
    assert queue_check["training_happened"] is False
    assert queue_check["validation_search_happened"] is False
    assert queue_check["confirmatory_test_tuning_happened"] is False
    assert queue_check["chunk_shape"] == [50, 7]
    assert queue_check["raw_action_chunk_shape"] == [1, 50, 7]
    assert queue_check["select_action_vs_chunk0_max_abs_diff"] == 0.0
    assert queue_check["queue_len_before_select_action"] == 0
    assert queue_check["queue_len_after_select_action"] == 49
    assert queue_check["all_prefixes_value_preserving"] is True
    assert queue_check["hard_stop_reasons"] == []
    validation = state["epoch_4_cycle_10_eac_validation_search"]
    assert validation["final_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert validation["validation_search_happened"] is True
    assert validation["confirmatory_records_used_for_tuning"] is False
    assert validation["validation_frame_count"] == 400
    assert validation["tried_config_count"] == 6
    assert validation["selected_config"] == "eac_q33_aggressive_1_4_50"
    assert validation["selected_validation_score"] == 0.7530415186081504
    assert validation["selected_commitment_map"] == {"short": 1, "medium": 4, "long": 50}
    assert validation["selected_commitment_counts"] == {"1": 132, "4": 136, "50": 132}
    assert validation["stage_a_manifest_allowed"] is True
    assert validation["stage_a_rollout_allowed"] is False
    preflight = state["epoch_4_cycle_10_eac_stage_a_preflight"]
    assert preflight["final_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert preflight["closed_loop_experiment_happened"] is False
    assert preflight["training_happened"] is False
    assert preflight["confirmatory_test_tuning_happened"] is False
    assert preflight["planned_episode_count"] == 50
    assert preflight["paired_cases_per_policy"] == 10
    assert preflight["reset_seeds"] == [20261211, 20261212]
    assert preflight["policy_count"] == 5
    assert preflight["checkpoint_policy_count"] == 0
    assert preflight["cuda_ok"] is True
    assert preflight["policy_output_shape"] == [50, 7]
    assert preflight["all_policy_prefixes_value_preserving"] is True
    assert preflight["no_accidental_checkpoint_reuse"] is True
    assert preflight["old_custom_libero_7d_route_used"] is False
    assert preflight["errors"] == []
    assert preflight["stage_a_rollout_allowed"] is True
    assert preflight["stage_a_runner_implementation_required"] is False
    assert preflight["stage_a_runner_validation_decision"] == "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT"
    runner = state["epoch_4_cycle_10_eac_stage_a_runner_validation"]
    assert runner["final_decision"] == "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT"
    assert runner["closed_loop_experiment_happened"] is False
    assert runner["training_happened"] is False
    assert runner["confirmatory_test_tuning_happened"] is False
    assert runner["policy_count"] == 5
    assert runner["runtime_samples_for_dynamic_schedulers"] == 2
    assert runner["all_policy_prefixes_value_preserving"] is True
    assert runner["any_action_values_modified"] is False
    assert runner["stage_a_rollout_allowed"] is True
    launch = state["epoch_4_cycle_10_eac_stage_a_launch"]
    assert launch["final_decision"] == "EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert launch["run_dir"] == "runs/eac_vla_stage_a/20260714T194025Z"
    assert launch["child_pid"] == 403
    assert launch["planned_episode_count"] == 50
    assert launch["partial_result"] == "reports/eac_vla/stage_a_partial_result.json"
    outcome = state["epoch_4_cycle_10_eac_stage_a_outcome"]
    assert outcome["final_decision"] == "EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert outcome["stage_b_required"] is True
    assert outcome["valid_current_formulation_kill"] is False
    assert outcome["completed_episode_count"] == 50
    assert outcome["exception_count"] == 0
    assert outcome["eac_full_successes"] == 8
    assert outcome["aac_entropy_proxy_successes"] == 9
    assert outcome["frozen_smolvla_fixed_queue_successes"] == 7
    assert outcome["eac_no_calibration_no_hysteresis_ablation_successes"] == 7
    assert outcome["fixed_short_replan_baseline_successes"] == 7
    assert outcome["paired_delta_vs_frozen_smolvla_fixed_queue"] == 0.1
    assert outcome["paired_delta_vs_aac_entropy_proxy"] == -0.1
    assert outcome["eac_full_action_values_modified"] is False
    stage_b = state["epoch_4_cycle_10_eac_stage_b_manifest"]
    assert stage_b["final_decision"] == "EAC_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert stage_b["manifest"] == "reports/eac_vla/stage_b_manifest.json"
    assert stage_b["manifest_canonical_payload_sha256"] == "31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D"
    assert stage_b["planned_episode_count"] == 200
    assert stage_b["paired_cases_per_policy"] == 40
    assert stage_b["reset_seeds"] == [20261213, 20261214]
    assert stage_b["identity_overlap_verification"]["overlap_with_stage_a_reset_seeds"] == 0
    assert stage_b["partition_separation"]["stage_b_outcomes_used_for_retuning"] is False
    launch_b = state["epoch_4_cycle_10_eac_stage_b_launch"]
    assert launch_b["final_decision"] == "EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert launch_b["run_dir"] == "runs/eac_vla_stage_b/20260714T202334Z"
    assert launch_b["child_pid"] == 386
    assert launch_b["exit_code"] == 0
    assert launch_b["planned_episode_count"] == 200
    assert launch_b["completed_episode_count"] == 200
    assert launch_b["partial_result"] == "reports/eac_vla/stage_b_partial_result.json"
    outcome_b = state["epoch_4_cycle_10_eac_stage_b_outcome"]
    assert outcome_b["final_decision"] == "EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert outcome_b["valid_current_formulation_kill"] is True
    assert outcome_b["completed_episode_count"] == 200
    assert outcome_b["exception_count"] == 0
    assert outcome_b["frozen_smolvla_fixed_queue_successes"] == 30
    assert outcome_b["aac_entropy_proxy_successes"] == 30
    assert outcome_b["eac_full_successes"] == 29
    assert outcome_b["eac_no_calibration_no_hysteresis_ablation_successes"] == 30
    assert outcome_b["fixed_short_replan_baseline_successes"] == 29
    assert outcome_b["paired_delta_vs_fixed_short_replan_baseline"] == 0.0
    assert outcome_b["simple_baseline_explains_method"] is True
    g3p = state["epoch_4_cycle_11_pre_proposal"]
    assert g3p["method"] == "G3P-VLA"
    assert g3p["selection_decision"] == "SELECT_G3P_VLA"
    assert g3p["candidate_generation"] == "reports/epoch_4_cycle_11_candidate_generation.md"
    assert g3p["prior_mechanism_map"] == "reports/epoch_4_cycle_11_prior_mechanism_map.md"
    assert g3p["candidate_count"] == 3
    assert g3p["closest_prior"] == "Direct Action-Head Injection of A Grounded 3D Point"
    assert g3p["secondary_priors"] == ["RoboPoint", "RoboGround", "AffordanceVLA"]
    assert g3p["selected_score"] == 90
    assert g3p["selected_contribution_type"] == "PRIOR_EXTENSION"
    assert g3p["first_comparison_policies"] == [
        "frozen_smolvla",
        "g3p_3d_point_proxy",
        "g3p_full",
        "g3p_no_3d_no_injection_ablation",
        "simple_2d_phase_or_nearest_object_heuristic",
    ]
    assert g3p["source_gate_required"] is True
    assert g3p["proposal"] == "reports/g3p_vla/researcher_proposal.md"
    assert g3p["proposal_hash"] == G3P_PROPOSAL_HASH
    assert g3p["proposal_hash_file"] == "reports/g3p_vla/proposal_hash.txt"
    assert g3p["proposal_decision"] == "G3P_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert g3p["reviewer_attack"] == "reports/g3p_vla/reviewer_attack.md"
    assert g3p["reviewer_attack_pending"] is False
    assert g3p["reviewer_attack_completed"] is True
    assert g3p["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert g3p["rebuttal"] == "reports/g3p_vla/researcher_rebuttal.md"
    assert g3p["rebuttal_pending"] is True
    assert g3p["rollout_allowed"] is False
    assert g3p["closed_loop_experiment_happened"] is False
    assert g3p["training_happened"] is False
    assert g3p["confirmatory_test_tuning_happened"] is False
    g3p_proposal = state["epoch_4_cycle_11_g3p_proposal"]
    assert g3p_proposal["proposal"] == "reports/g3p_vla/researcher_proposal.md"
    assert g3p_proposal["proposal_hash"] == G3P_PROPOSAL_HASH
    assert g3p_proposal["final_decision"] == "G3P_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert g3p_proposal["reviewer_attack"] == "reports/g3p_vla/reviewer_attack.md"
    assert g3p_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert g3p_proposal["reviewer_attack_completed"] is True
    assert g3p_proposal["training_happened"] is False
    assert g3p_proposal["confirmatory_test_tuning_happened"] is False
    g3p_review = state["epoch_4_cycle_11_g3p_review"]
    assert g3p_review["reviewer_attack"] == "reports/g3p_vla/reviewer_attack.md"
    assert g3p_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert g3p_review["closest_prior_proxy_must_remain_transparent"] is True
    assert g3p_review["source_gate_required_before_rollout"] is True
    assert g3p_review["simple_heuristic_must_remain_live"] is True
    assert g3p_review["confirmatory_test_tuning_happened"] is False
    assert state["task_reset_manifest"] is None
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["paired_cases_per_policy"] == 10
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["reset_seeds"] == [20261201, 20261202]
    assert state["checkpoint_path"] is None
    assert state["stage_a_result_json"] is None
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["completed_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["final_decision"] == "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["policy_successes"]["mtf_full"]["successes"] == 7
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["policy_successes"]["frameskip_proxy_lora"]["successes"] == 8
    assert state["stage_b_manifest_json"] is None
    assert state["stage_b_partial_checkpoint"] is None
    assert state["stage_b_result_json"] is None
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
    assert state["epoch_4_cycle_7_pre_stage_0"]["policy_identity_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_7_pre_stage_0"]["checkpoint_root"] == "runs\\dagr_vla_checkpoints\\dagr_a020_route_mlp"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_manifest_decision"] == "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_planned_episode_count"] == 50
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_reset_seeds"] == [20261205, 20261206]
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_decision"] == "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_policy_count"] == 5
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_checkpoint_policy_count"] == 4
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_cuda_ok"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_checkpoint_checksum_matches"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_preflight_no_accidental_checkpoint_reuse"] is True
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
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["policy_identity_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["stage_a_preflight_decision"] == "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_policy_identity_outcome"]["final_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_7_dagr_policy_identity_outcome"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_manifest"]["final_decision"] == "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_7_dagr_stage_a_manifest"]["reset_seeds"] == [20261205, 20261206]
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["final_decision"] == "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["policy_count"] == 5
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["checkpoint_policy_count"] == 4
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["checkpoint_checksum_matches"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["cuda_ok"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["no_accidental_checkpoint_reuse"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["errors"] == []
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["final_decision"] == "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["completed_episode_count"] == 50
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["frozen_smolvla_successes"] == 8
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["gripper_transition_heuristic_successes"] == 7
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["dagr_full_successes"] == 6
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["dam_static_component_proxy_successes"] == 2
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["stage_b_required"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["catastrophic_stage_a_kill"] is False
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["final_decision"] == "DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["planned_episode_count"] == 200
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["paired_cases_per_policy"] == 40
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["reset_seeds"] == [20261207, 20261208]
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["manifest_canonical_payload_sha256"] == "2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B"
    outcome = state["epoch_4_cycle_7_dagr_stage_b_outcome"]
    assert outcome["final_decision"] == "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert outcome["valid_current_formulation_kill"] is True
    assert outcome["stage_b_completed"] is True
    assert outcome["completed_episode_count"] == 200
    assert outcome["exception_count"] == 0
    assert outcome["dagr_full_successes"] == 18
    assert outcome["frozen_smolvla_successes"] == 28
    assert outcome["gripper_transition_heuristic_successes"] == 24
    assert outcome["paired_delta_vs_frozen_smolvla"] == -0.25
    assert outcome["paired_ci_vs_frozen_smolvla"] == [-0.4, -0.1]
    assert outcome["paired_delta_vs_gripper_transition_heuristic"] == -0.15
    assert outcome["paired_ci_vs_gripper_transition_heuristic"] == [-0.3, 0.0]
    assert outcome["simple_baseline_explains_method"] is True
    assert state["epoch_4_cycle_8_pre_stage_a"]["selection_decision"] == "SELECT_MARC_VLA"
    assert state["epoch_4_cycle_8_pre_stage_a"]["closest_prior"] == "OpenVLA-OFT"
    assert state["epoch_4_cycle_8_pre_stage_a"]["proposal_hash"] == "D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A"
    assert state["epoch_4_cycle_8_pre_stage_a"]["reviewer_attack"] == "reports/marc_vla/reviewer_attack.md"
    assert state["epoch_4_cycle_8_pre_stage_a"]["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["epoch_4_cycle_8_pre_stage_a"]["researcher_rebuttal"] == "reports/marc_vla/researcher_rebuttal.md"
    assert state["epoch_4_cycle_8_pre_stage_a"]["rebuttal_decision"] == "MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert state["epoch_4_cycle_8_pre_stage_a"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_8_pre_stage_a"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_8_pre_stage_a"]["selected_config"] == "marc_a020_gate_mlp"
    assert state["epoch_4_cycle_8_pre_stage_a"]["selected_correction_alpha"] == 0.2
    assert state["epoch_4_cycle_8_pre_stage_a"]["selected_gate_architecture"] == "mlp"
    assert state["epoch_4_cycle_8_pre_stage_a"]["policy_identity_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_8_pre_stage_a"]["checkpoint_root"] == "runs\\marc_vla_checkpoints\\marc_a020_gate_mlp"
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_manifest_decision"] == "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_planned_episode_count"] == 50
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_paired_cases_per_policy"] == 10
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_reset_seeds"] == [20261209, 20261210]
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_manifest_canonical_payload_sha256"] == "3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E"
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_decision"] == "MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_policy_count"] == 5
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_checkpoint_policy_count"] == 4
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_cuda_ok"] is True
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_checkpoint_checksum_matches"] is True
    assert state["epoch_4_cycle_8_pre_stage_a"]["stage_a_preflight_no_accidental_checkpoint_reuse"] is True
    assert state["epoch_4_cycle_8_pre_stage_a"]["policy_identity_verified_count"] == 4
    assert state["epoch_4_cycle_8_pre_stage_a"]["first_comparison_policies"] == [
        "frozen_smolvla",
        "openvla_oft_l1_proxy",
        "marc_full",
        "marc_no_disagreement_gate_ablation",
        "static_l1_mixture_baseline",
    ]
    assert state["epoch_4_cycle_8_marc_development_outcome"]["scoreable_development_records"] == 1600
    assert state["epoch_4_cycle_8_marc_development_outcome"]["train_records"] == 1200
    assert state["epoch_4_cycle_8_marc_development_outcome"]["validation_records"] == 400
    assert state["epoch_4_cycle_8_marc_development_outcome"]["reserved_test_records_not_used"] == 1200
    assert state["epoch_4_cycle_8_marc_development_outcome"]["split_overlap"] == {
        "train_reserved": 0,
        "train_validation": 0,
        "validation_reserved": 0,
    }
    assert state["epoch_4_cycle_8_marc_development_outcome"]["train_disagreement_fraction"] == 0.4
    assert state["epoch_4_cycle_8_marc_development_outcome"]["validation_disagreement_fraction"] == 0.44
    assert state["epoch_4_cycle_8_marc_development_outcome"]["gate_probe_accuracy_margin"] == 0.04749999999999999
    assert state["epoch_4_cycle_8_marc_development_outcome"]["tried_config_count"] == 6
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_config"] == "marc_a020_gate_mlp"
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_validation_score"] == 0.5457964262366295
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_gate_accuracy_margin"] == 0.05249999999999999
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_gate_predicted_positive_fraction"] == 0.3325
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_delta_l2_p95"] == 0.011818917468190193
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_clean_delta_l2_p95"] == 0.010853752493858337
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_action_validity"] == 1.0
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_full_vs_l1_proxy_mean_l2"] == 0.007010325323790312
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_full_vs_static_mean_l2"] == 0.0019475044682621956
    assert state["epoch_4_cycle_8_marc_development_outcome"]["static_mixture_remains_live_reviewer_killer"] is True
    assert state["epoch_4_cycle_8_marc_development_outcome"]["policy_identity_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_8_marc_development_outcome"]["stage_a_allowed"] is True
    policy_outcome = state["epoch_4_cycle_8_marc_policy_identity_outcome"]
    assert policy_outcome["final_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert policy_outcome["stage_a_allowed"] is True
    assert policy_outcome["checkpoint_root"] == "runs\\marc_vla_checkpoints\\marc_a020_gate_mlp"
    assert policy_outcome["policy_identities"] == [
        "frozen_smolvla",
        "openvla_oft_l1_proxy",
        "marc_full",
        "marc_no_disagreement_gate_ablation",
        "static_l1_mixture_baseline",
    ]
    assert policy_outcome["variant_success_count"] == 4
    assert policy_outcome["marc_full_delta_l2_p95"] == 0.010693175718188286
    assert policy_outcome["openvla_oft_l1_proxy_delta_l2_p95"] == 0.2307613492012024
    assert policy_outcome["static_l1_mixture_delta_l2_p95"] == 0.07999999821186066
    assert policy_outcome["distinction"]["marc_full_vs_openvla_oft_l1_proxy_mean_l2"] == 0.08430124074220657
    assert policy_outcome["distinction"]["marc_full_vs_static_l1_mixture_baseline_mean_l2"] == 0.032826922833919525
    marc_manifest = state["epoch_4_cycle_8_marc_stage_a_manifest"]
    assert marc_manifest["final_decision"] == "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert marc_manifest["planned_episode_count"] == 50
    assert marc_manifest["paired_cases_per_policy"] == 10
    assert marc_manifest["reset_seeds"] == [20261209, 20261210]
    assert marc_manifest["manifest_canonical_payload_sha256"] == "3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E"
    marc_preflight = state["epoch_4_cycle_8_marc_stage_a_preflight"]
    assert marc_preflight["final_decision"] == "MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert marc_preflight["policy_count"] == 5
    assert marc_preflight["checkpoint_policy_count"] == 4
    assert marc_preflight["checkpoint_checksum_matches"] is True
    assert marc_preflight["cuda_ok"] is True
    assert marc_preflight["no_accidental_checkpoint_reuse"] is True
    assert marc_preflight["errors"] == []
    marc_launch = state["epoch_4_cycle_8_marc_stage_a_launch"]
    assert marc_launch["run_dir"] == "runs/marc_vla_stage_a/20260714T171356Z"
    assert marc_launch["child_pid"] == 414
    assert marc_launch["planned_episode_count"] == 50
    marc_outcome = state["epoch_4_cycle_8_marc_stage_a_outcome"]
    assert marc_outcome["final_decision"] == "MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
    assert marc_outcome["valid_current_formulation_kill"] is True
    assert marc_outcome["completed_episode_count"] == 50
    assert marc_outcome["exception_count"] == 0
    assert marc_outcome["frozen_smolvla_successes"] == 8
    assert marc_outcome["marc_full_successes"] == 0
    assert marc_outcome["marc_no_disagreement_gate_ablation_successes"] == 7
    assert marc_outcome["static_l1_mixture_baseline_successes"] == 7
    assert marc_outcome["paired_delta_vs_frozen_smolvla"] == -0.8
    assert marc_outcome["paired_delta_vs_no_disagreement_gate_ablation"] == -0.7
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
