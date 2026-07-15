import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]
PESA_PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"
EAC_PROPOSAL_HASH = "A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E"
G3P_PROPOSAL_HASH = "BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71"
CALA_PROPOSAL_HASH = "5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76"
RAR_PROPOSAL_HASH = "723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56"
COVI_PROPOSAL_HASH = "338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621"
LIFT_PROPOSAL_HASH = "3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F"
IARC_PROPOSAL_HASH = "A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408"
FAMR_PROPOSAL_HASH = "96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69"
PCAV_PROPOSAL_HASH = "E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA"
SPARC_PROPOSAL_HASH = "CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D"
NICE_PROPOSAL_HASH = "898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A"
HEST_PROPOSAL_HASH = "E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527"
HASTE_PROPOSAL_HASH = "5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6"
KITE_PROPOSAL_HASH = "FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8"
VDR_PROPOSAL_HASH = "0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72"
RAP_PROPOSAL_HASH = "E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008"


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_governance_freezes_false_negative_safeguard() -> None:
    governance = (REPO_ROOT / "reports" / "current_research_governance.md").read_text(encoding="utf-8")

    assert "False-Negative Safeguard For Pre-Rollout Decisions" in governance
    assert "FATAL_PREIMPLEMENTATION" in governance
    assert "ROBUST_EMPIRICAL_DESIGN_FAILURE" in governance
    assert "UNDERPOWERED_OR_UNRESOLVED" in governance
    assert "IMPLEMENTATION_OR_DATA_FAILURE" in governance
    assert "exactly one" in governance


def test_post_covi_lora_and_minimum_sufficient_governance_is_active() -> None:
    governance = (REPO_ROOT / "reports" / "current_research_governance.md").read_text(encoding="utf-8")

    assert "Post-COVI LoRA And Minimum-Sufficient Design Governance" in governance
    assert "SCIENTIFIC_METHOD" in governance
    assert "LOW_COMPUTE_PARAMETERIZATION" in governance
    assert "LoRA and QLoRA are compute-enabling implementation mechanisms" in governance
    assert "Conditional Standard-LoRA Control" in governance
    assert "The fifth policy is conditional, not mandatory" in governance
    assert "LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT" in governance
    assert "SmolVLA versus SmolVLA plus Ours" in governance
    assert "Quantized OpenVLA-OFT INT4 plus Ours" in governance


def test_active_state_records_rap_selection_and_vdr_stage_0a_failure() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 25
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "RAP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert state["current_stage"] == "epoch_4_cycle_25_rap_reviewer_attack_pending"
    assert state["method"] == "RAP-VLA"
    assert state["method_identity"] == "RAP-VLA"
    assert state["proposal_hash"] == RAP_PROPOSAL_HASH
    assert state["prototype_protocol"] is None
    assert "epoch_4_cycle_16_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_16_iarc_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_16_iarc_stage_0a_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_16_iarc_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_16_iarc_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_16_iarc_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_17_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_17_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_stage_0a_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_stage_0a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_endpoint_training_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_endpoint_training_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_endpoint_training_completed" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_endpoint_training_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_17_famr_endpoint_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_18_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_18_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_researcher_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_attempt_1_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_resumed_missing_keys_only" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_18_pcav_no_headroom_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_19_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_19_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_19_sparc_researcher_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_19_sparc_stage_0a_single_repair_consumed" in state["completed_stages"]
    assert "epoch_4_cycle_19_sparc_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_19_sparc_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_20_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_20_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_researcher_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0a_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0b1_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0b1_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0b1_completed" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_stage_0b1_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_20_nice_data_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_21_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_21_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_stage_0a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_stage_0a_pending" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_stage_0a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_21_hest_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_22_candidate_search_pending" in state["completed_stages"]
    hest = state["epoch_4_cycle_21_hest_stage_0a_outcome"]
    assert hest["final_decision"] == "HEST_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert hest["planned_window_count"] == 160
    assert hest["completed_window_count"] == 160
    assert hest["exception_count"] == 0
    assert hest["invalid_support_counts"]["base"] == 1
    assert hest["invalid_support_counts"]["hest"] == 1
    assert hest["stage_0b_allowed"] is False
    assert "epoch_4_cycle_22_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_stage_0a_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_stage_0a_runner_implemented" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_stage_0a_pending" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_stage_0a_launched" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_stage_0a_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_22_haste_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_23_candidate_search_pending" in state["completed_stages"]
    haste = state["epoch_4_cycle_22_haste_pre_stage_0a"]
    assert haste["candidate_count"] == 3
    assert haste["selected_score"] == 95
    assert haste["proposal_hash"] == HASTE_PROPOSAL_HASH
    assert haste["bounded_validation_search_max_configs"] == 6
    assert haste["implementation_commit"] == "3dd76f0"
    assert haste["real_checkpoint_interface_smoke_passed"] is True
    assert haste["zero_effect_identity_smoke_max_error"] == 0.0
    assert haste["stage_0a_pending"] is False
    outcome = state["epoch_4_cycle_22_haste_stage_0a_outcome"]
    assert outcome["final_decision"] == "HASTE_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert outcome["failure_class"] == "PRE_MANIFEST_IMPLEMENTATION_FAILURE"
    assert outcome["worker_pid"] == 295
    assert outcome["exit_code"] == 1
    assert outcome["persisted_row_count"] == 0
    assert outcome["manifest_persisted"] is False
    assert outcome["partial_persisted"] is False
    assert outcome["stage_0b_allowed"] is False
    assert outcome["rerun_allowed"] is False
    for stage in (
        "epoch_4_cycle_23_candidate_generation_completed",
        "epoch_4_cycle_23_kite_researcher_proposal_frozen",
        "epoch_4_cycle_23_kite_reviewer_attack_completed",
        "epoch_4_cycle_23_kite_rebuttal_completed",
        "epoch_4_cycle_23_kite_mathematical_audit_preregistered",
        "epoch_4_cycle_23_kite_preregistration_frozen",
        "epoch_4_cycle_23_kite_prototype_protocol_frozen",
        "epoch_4_cycle_23_kite_stage_0a_implementation_pending",
        "epoch_4_cycle_23_kite_stage_0a_runner_implemented",
        "epoch_4_cycle_23_kite_stage_0a_pending",
        "epoch_4_cycle_23_kite_stage_0a_launched",
        "epoch_4_cycle_23_kite_stage_0a_attempt_1_persistence_failure_recorded",
        "epoch_4_cycle_23_kite_stage_0a_resumed_missing_keys_only",
        "epoch_4_cycle_23_kite_stage_0a_completed",
        "epoch_4_cycle_23_kite_stage_0a_adjudicated",
        "epoch_4_cycle_23_kite_implementation_failure_recorded",
        "epoch_4_cycle_24_candidate_search_pending",
        "epoch_4_cycle_24_candidate_generation_completed",
        "epoch_4_cycle_24_vdr_researcher_proposal_frozen",
        "epoch_4_cycle_24_vdr_reviewer_attack_completed",
        "epoch_4_cycle_24_vdr_rebuttal_completed",
        "epoch_4_cycle_24_vdr_mathematical_audit_preregistered",
        "epoch_4_cycle_24_vdr_preregistration_frozen",
        "epoch_4_cycle_24_vdr_prototype_protocol_frozen",
        "epoch_4_cycle_24_vdr_stage_0a_pending",
        "epoch_4_cycle_24_vdr_stage_0a_completed",
        "epoch_4_cycle_24_vdr_stage_0a_adjudicated",
        "epoch_4_cycle_24_vdr_implementation_or_optimization_failure_recorded",
        "epoch_4_cycle_25_candidate_search_pending",
        "epoch_4_cycle_25_prior_mechanism_map_completed",
        "epoch_4_cycle_25_candidate_generation_completed",
        "epoch_4_cycle_25_rap_candidate_selected",
        "epoch_4_cycle_25_rap_researcher_proposal_pending",
        "epoch_4_cycle_25_rap_researcher_proposal_frozen",
        "epoch_4_cycle_25_rap_reviewer_attack_pending",
    ):
        assert stage in state["completed_stages"]
    rap = state["epoch_4_cycle_25_candidate_selection"]
    assert rap["candidate_count"] == 3
    assert rap["selected_score"] == 94
    assert rap["method"] == "RAP-VLA"
    assert rap["closest_prior"] == "OptimusVLA"
    assert rap["closest_prior_official_repository"] == "https://github.com/iLearn-Lab/CVPR26-OptimusVLA"
    assert rap["proposal_hash"] == RAP_PROPOSAL_HASH
    assert rap["policy_order"] == [
        "smolvla_base",
        "optimusvla_memory_prior_proxy",
        "rap_full",
        "rap_anchor_only_no_residual",
        "standard_lora",
    ]
    assert rap["standard_lora_required"] is True
    rap_proposal = state["epoch_4_cycle_25_rap_researcher_proposal"]
    assert rap_proposal["final_decision"] == "RAP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert rap_proposal["proposal"] == "reports/rap_vla/researcher_proposal.md"
    assert rap_proposal["proposal_hash"] == RAP_PROPOSAL_HASH
    vdr = state["epoch_4_cycle_24_candidate_selection"]
    assert vdr["candidate_count"] == 3
    assert vdr["selected_score"] == 92
    assert vdr["closest_prior"] == "FutureVLA"
    assert vdr["proposal_hash"] == VDR_PROPOSAL_HASH
    assert vdr["policy_order"] == [
        "smolvla_base",
        "futurevla_latent_alignment_proxy",
        "vdr_full",
        "vdr_no_action_residual",
        "standard_lora",
    ]
    assert vdr["standard_lora_required"] is True
    vdr_pre = state["epoch_4_cycle_24_vdr_pre_stage_0a"]
    assert vdr_pre["final_decision"] == "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert vdr_pre["stage_0a_pending"] is False
    assert vdr_pre["vdr_coefficients"] == [0.1, 0.3, 1.0]
    assert vdr_pre["runner"] == "scripts/run_vdr_vla_stage0a.py"
    assert vdr_pre["runner_validation"] == "reports/vdr_vla/stage_0a_serializer_preflight.json"
    assert vdr_pre["runner_unit_tests_passed"] == 10
    assert vdr_pre["serializer_preflight_passed"] is True
    vdr_outcome = state["epoch_4_cycle_24_vdr_stage_0a_outcome"]
    assert vdr_outcome["final_decision"] == "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert vdr_outcome["completed_model_row_count"] == 1536
    assert vdr_outcome["planned_model_row_count"] == 1536
    assert vdr_outcome["exception_count"] == 0
    assert vdr_outcome["duplicate_manifest_key_count"] == 0
    assert vdr_outcome["duplicate_partial_key_count"] == 0
    assert vdr_outcome["missing_manifest_key_count"] == 0
    assert vdr_outcome["extra_partial_key_count"] == 0
    assert vdr_outcome["split_overlap_key_count"] == 0
    assert vdr_outcome["key_sets_equal"] is True
    assert vdr_outcome["stage_0b_allowed"] is False
    assert vdr_outcome["valid_scientific_result"] is False
    assert vdr_outcome["scientific_kill"] is False
    assert vdr_outcome["action_validity_ok"] is False
    assert vdr_outcome["worker_pid"] == 411
    kite = state["epoch_4_cycle_23_kite_pre_stage_0a"]
    assert kite["candidate_count"] == 3
    assert kite["selected_score"] == 96
    assert kite["proposal_hash"] == KITE_PROPOSAL_HASH
    assert kite["horizons"] == [5, 20]
    assert kite["kite_coefficients"] == [0.1, 0.3, 1.0]
    assert kite["policy_order"] == [
        "smolvla_base",
        "geopredict_kinematics_proxy",
        "kite_full",
        "cumulative_action_target",
        "standard_lora",
    ]
    assert kite["implementation_commit"] == "62dbb75"
    assert kite["runner"] == "scripts/run_kite_vla_stage0a.py"
    assert kite["runner_validation"] == "reports/kite_vla/stage_0a_runner_validation.json"
    assert kite["stage_0a_pending"] is False
    outcome = state["epoch_4_cycle_23_kite_stage_0a_outcome"]
    assert outcome["final_decision"] == "KITE_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert outcome["completed_model_row_count"] == 128
    assert outcome["planned_model_row_count"] == 128
    assert outcome["resumed_model_row_count"] == 115
    assert outcome["exception_count"] == 1
    assert outcome["duplicate_partial_key_count"] == 0
    assert outcome["missing_manifest_key_count"] == 0
    assert outcome["bad_feature_cache_hash_count"] == 0
    assert outcome["headroom_passed"] is True
    assert outcome["kite_gradient_nonzero"] is True
    assert outcome["identity_max_abs_error"] == 0.0
    assert outcome["action_validity_ok"] is False
    assert outcome["invalid_action_row_count"] == 128
    assert outcome["stage_0b_allowed"] is False
    assert outcome["rerun_allowed"] is False
    selection = state["epoch_4_cycle_16_candidate_selection"]
    assert selection["candidate_count"] == 3
    assert selection["selected_score"] == 95
    assert selection["proposal_hash"] == IARC_PROPOSAL_HASH
    assert selection["policy_order"] == [
        "smolvla_base",
        "strong_vla_transparent_proxy",
        "iarc_vla_full",
        "iarc_unprojected_joint_replay_ablation",
        "standard_lora_clean_only",
    ]
    assert selection["bounded_validation_search_max_configs"] == 6
    assert selection["confirmatory_test_tuning_happened"] is False
    famr = state["epoch_4_cycle_17_candidate_selection"]
    assert famr["candidate_count"] == 3
    assert famr["selected_score"] == 93
    assert famr["proposal_hash"] == FAMR_PROPOSAL_HASH
    assert famr["closest_prior"] == "RETAIN"
    assert famr["bounded_validation_search_max_configs"] == 6
    assert famr["policy_order"] == [
        "smolvla_base",
        "retain_scalar_proxy",
        "famr_full",
        "famr_target_only",
        "standard_lora_new_task",
    ]
    pcav = state["epoch_4_cycle_18_candidate_selection"]
    assert pcav["candidate_count"] == 3
    assert pcav["selected_score"] == 95
    assert pcav["proposal_hash"] == PCAV_PROPOSAL_HASH
    assert pcav["closest_prior"] == "TACO"
    assert pcav["secondary_mechanism_prior"] == "ProgressVLA"
    assert pcav["bounded_validation_search_max_configs"] == 6
    assert pcav["policy_order"] == [
        "smolvla_base",
        "taco_support_proxy",
        "pcav_full",
        "pcav_progress_only",
        "standard_lora_new_task",
    ]
    assert pcav["training_happened"] is False
    assert pcav["validation_search_happened"] is False
    assert pcav["closed_loop_experiment_happened"] is False
    assert pcav["confirmatory_test_tuning_happened"] is False
    pcav_pre_stage = state["epoch_4_cycle_18_pcav_pre_stage_0a"]
    assert pcav_pre_stage["final_decision"] == "PCAV_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING"
    assert pcav_pre_stage["stage_0a_initial_rows"] == 24
    assert pcav_pre_stage["stage_0a_expansion_rows"] == 96
    assert pcav_pre_stage["candidate_count_per_row"] == 4
    assert pcav_pre_stage["confirmatory_observations_decoded_max"] == 0
    assert pcav_pre_stage["confirmatory_actions_computed_max"] == 0
    assert pcav_pre_stage["stage_0a_pending"] is False
    pcav_outcome = state["epoch_4_cycle_18_pcav_stage_0a_outcome"]
    assert pcav_outcome["final_decision"] == "PCAV_STAGE_0A_NO_USABLE_HEADROOM"
    assert pcav_outcome["failure_class"] == "NO_HEADROOM"
    assert pcav_outcome["valid_scientific_kill"] is False
    assert pcav_outcome["expanded_rows_completed"] == pcav_outcome["expanded_rows_planned"] == 96
    assert pcav_outcome["resume_preserved_initial_rows"] is True
    assert pcav_outcome["resume_new_rows"] == 72
    assert pcav_outcome["resume_repeated_completed_rows"] == 0
    assert pcav_outcome["exception_count"] == 0
    assert pcav_outcome["duplicate_key_count"] == 0
    assert pcav_outcome["missing_manifest_key_count"] == 0
    assert pcav_outcome["extra_result_key_count"] == 0
    assert pcav_outcome["source_health_passed"] is True
    assert pcav_outcome["partition_audit_passed"] is True
    assert pcav_outcome["manifest_health_passed"] is True
    assert pcav_outcome["all_base_candidates_valid"] is True
    assert pcav_outcome["materially_better_row_count"] == 7
    assert pcav_outcome["materially_better_fraction"] < 0.25
    assert pcav_outcome["median_oracle_relative_reduction_improvable"] < 0.05
    assert pcav_outcome["headroom_passed"] is False
    assert pcav_outcome["base_identity_max_abs_error"] == 0.0
    assert pcav_outcome["checkpoint_reload_max_abs_error"] == 0.0
    assert pcav_outcome["base_hash_unchanged"] is True
    assert pcav_outcome["confirmatory_observations_decoded"] == 0
    assert pcav_outcome["confirmatory_actions_computed"] == 0
    assert pcav_outcome["stage_0b_allowed"] is False
    pcav_validation = json.loads(
        (REPO_ROOT / "reports" / "pcav_vla" / "stage_0a_validation.json").read_text(encoding="utf-8")
    )
    assert pcav_validation["row_manifest_hash_recomputed"] is True
    assert pcav_validation["candidate_manifest_hash_recomputed"] is True
    assert pcav_validation["partial_candidate_keys_equal"] is True
    assert pcav_validation["candidate_expanded_key_sets_equal"] is True
    assert pcav_validation["accepted_without_rerun"] is True
    sparc = state["epoch_4_cycle_19_candidate_selection"]
    assert sparc["candidate_count"] == 3
    assert sparc["selected_score"] == 96
    assert sparc["proposal_hash"] == SPARC_PROPOSAL_HASH
    assert sparc["closest_prior"] == "COAST"
    assert sparc["bounded_validation_search_max_configs"] == 6
    assert sparc["policy_order"] == [
        "smolvla_base",
        "coast_single_source_transfer_proxy",
        "sparc_full",
        "sparc_source_failure_only",
        "standard_lora_target_success",
    ]
    assert sparc["training_happened"] is False
    assert sparc["validation_search_happened"] is False
    assert sparc["closed_loop_experiment_happened"] is False
    assert sparc["confirmatory_test_tuning_happened"] is False
    sparc_pre_stage = state["epoch_4_cycle_19_sparc_pre_stage_0a"]
    assert sparc_pre_stage["planned_observation_count"] == 2
    assert sparc_pre_stage["synthetic_unlabeled_operator_only"] is True
    assert sparc_pre_stage["confirmatory_records_read_max"] == 0
    assert sparc_pre_stage["stage_0a_pending"] is False
    sparc_outcome = state["epoch_4_cycle_19_sparc_stage_0a_outcome"]
    assert sparc_outcome["raw_final_decision"] == "SPARC_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert sparc_outcome["failure_class"] == "IMPLEMENTATION_OR_DATA_FAILURE"
    assert sparc_outcome["valid_scientific_kill"] is False
    assert sparc_outcome["single_allowed_repair_consumed"] is True
    assert sparc_outcome["final_worker_alive"] is False
    assert sparc_outcome["final_worker_status"] == "completed"
    assert sparc_outcome["exit_code"] == 0
    assert sparc_outcome["completed_observation_count"] == sparc_outcome["planned_observation_count"] == 2
    assert sparc_outcome["exception_count"] == 0
    assert sparc_outcome["duplicate_key_count"] == 0
    assert sparc_outcome["missing_manifest_key_count"] == 0
    assert sparc_outcome["extra_result_key_count"] == 0
    assert sparc_outcome["capture_identity_max_abs_error"] == 0.0
    assert sparc_outcome["configured_reload_max_abs_error"] == 0.0
    assert sparc_outcome["all_activation_rows_act"] is True
    assert sparc_outcome["all_action_rows_safe"] is False
    assert sparc_outcome["synthetic_unlabeled_operator_only"] is True
    assert sparc_outcome["labeled_activation_fit_happened"] is False
    assert sparc_outcome["confirmatory_records_read"] == 0
    assert sparc_outcome["stage_0b_allowed"] is False
    nice = state["epoch_4_cycle_20_candidate_selection"]
    assert nice["candidate_count"] == 3
    assert nice["selected_score"] == 96
    assert nice["proposal_hash"] == NICE_PROPOSAL_HASH
    assert nice["closest_prior"] == "VLA-Corrector"
    assert nice["closest_prior_source_commit"] == "9d23a0ba6fad562d3ed1a68fc52c8a12459abb41"
    assert nice["bounded_validation_search_max_configs"] == 6
    assert nice["policy_order"] == [
        "smolvla_base_fixed_horizon",
        "vla_corrector_official_proxy",
        "nice_full",
        "nice_mean_only_global_error_ablation",
        "fixed_short_horizon_replan",
    ]
    assert nice["training_happened"] is False
    assert nice["validation_search_happened"] is False
    assert nice["closed_loop_experiment_happened"] is False
    assert nice["confirmatory_test_tuning_happened"] is False
    nice_pre_stage = state["epoch_4_cycle_20_nice_pre_stage_0a"]
    assert nice_pre_stage["final_decision"] == "NICE_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING"
    assert nice_pre_stage["planned_pair_count"] == 128
    assert nice_pre_stage["k_step"] == 10
    assert nice_pre_stage["validation_records_read_max"] == 0
    assert nice_pre_stage["confirmatory_records_read_max"] == 0
    assert nice_pre_stage["stage_0a_pending"] is False
    nice_outcome = state["epoch_4_cycle_20_nice_stage_0a_outcome"]
    assert nice_outcome["final_decision"] == "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert nice_outcome["valid_scientific_result"] is False
    assert nice_outcome["worker_alive"] is False
    assert nice_outcome["exit_code"] == 0
    assert nice_outcome["completed_pair_count"] == nice_outcome["planned_pair_count"] == 128
    assert nice_outcome["exception_count"] == 0
    assert nice_outcome["duplicate_manifest_key_count"] == 0
    assert nice_outcome["duplicate_result_key_count"] == 0
    assert nice_outcome["missing_manifest_key_count"] == 0
    assert nice_outcome["extra_result_key_count"] == 0
    assert nice_outcome["latent_shape"] == [128, 960]
    assert nice_outcome["action_inside_fraction"] == 1.0
    assert nice_outcome["checkpoint_reload_max_abs_error"] == 0.0
    assert nice_outcome["base_action_identity_max_abs_error"] == 0.0
    assert nice_outcome["validation_records_read"] == 0
    assert nice_outcome["confirmatory_records_read"] == 0
    assert nice_outcome["simulator_rollout_count"] == 0
    assert nice_outcome["stage_0b1_allowed"] is True
    nice_stage0b1 = state["epoch_4_cycle_20_nice_stage_0b1_outcome"]
    assert nice_stage0b1["raw_worker_decision"] == "NICE_STAGE_0B1_IMPLEMENTATION_FAILURE"
    assert nice_stage0b1["final_decision"] == "NICE_STAGE_0B1_DATA_FAILURE_COLLAPSED_ACTION_REGIME_CONTRAST"
    assert nice_stage0b1["failure_class"] == "DATA_FAILURE"
    assert nice_stage0b1["valid_scientific_result"] is False
    assert nice_stage0b1["worker_alive"] is False
    assert nice_stage0b1["exit_code"] == 1
    assert nice_stage0b1["completed_pair_count"] == nice_stage0b1["planned_pair_count"] == 1792
    assert nice_stage0b1["exception_count"] == 1
    assert nice_stage0b1["duplicate_manifest_key_count"] == 0
    assert nice_stage0b1["duplicate_partial_key_count"] == 0
    assert nice_stage0b1["missing_manifest_key_count"] == 0
    assert nice_stage0b1["extra_partial_key_count"] == 0
    assert nice_stage0b1["frozen_gripper_deadband"] == 2.0
    assert nice_stage0b1["collapsed_validation_tasks"] == [
        "libero_object/task_3",
        "libero_spatial/task_3",
    ]
    assert nice_stage0b1["confirmatory_records_read"] == 0
    assert nice_stage0b1["simulator_rollout_count"] == 0
    assert nice_stage0b1["stage_0b2_allowed"] is False
    assert nice_stage0b1["nice_rescue_allowed"] is False
    pre_stage = state["epoch_4_cycle_17_famr_pre_stage_0a"]
    assert pre_stage["final_decision"] == "FAMR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING"
    assert pre_stage["confirmatory_observations_decoded_max"] == 0
    assert pre_stage["confirmatory_actions_computed_max"] == 0
    famr_stage_0a = state["epoch_4_cycle_17_famr_stage_0a"]
    assert famr_stage_0a["final_decision"] == "FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED"
    assert famr_stage_0a["valid_scientific_kill"] is False
    assert famr_stage_0a["micro_fit_steps_completed"] == famr_stage_0a["micro_fit_steps_planned"] == 20
    assert famr_stage_0a["fixed_subset_relative_reduction"] > 0.01
    assert famr_stage_0a["duplicate_key_count"] == 0
    assert famr_stage_0a["exception_count"] == 0
    assert famr_stage_0a["identity_max_abs_error"] == 0.0
    assert famr_stage_0a["checkpoint_reload_max_abs_error"] == 0.0
    assert famr_stage_0a["base_hash_unchanged"] is True
    assert famr_stage_0a["scaling_identity_passed"] is True
    assert famr_stage_0a["confirmatory_observations_decoded"] == 0
    assert famr_stage_0a["confirmatory_actions_computed"] == 0
    endpoint = state["epoch_4_cycle_17_famr_endpoint_training"]
    assert endpoint["final_decision"] == "FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert endpoint["failure_class"] == "IMPLEMENTATION_OR_DATA_FAILURE"
    assert endpoint["valid_scientific_kill"] is False
    assert endpoint["optimizer_steps_completed"] == endpoint["optimizer_steps_planned"] == 300
    assert endpoint["microbatches_completed"] == endpoint["microbatches_planned"] == 2400
    assert endpoint["task_counts"] == [800, 800, 800]
    assert endpoint["duplicate_key_count"] == 0
    assert endpoint["exception_count"] == 0
    assert endpoint["fixed_subset_relative_reduction"] > 0.75
    assert endpoint["action_effect_active_fraction"] == 1.0
    assert endpoint["outside_fraction"] > endpoint["outside_fraction_limit"]
    assert endpoint["p99_exceedance"] > endpoint["p99_exceedance_limit"]
    assert endpoint["checkpoint_reload_max_abs_error"] == 0.0
    assert endpoint["base_hash_unchanged"] is True
    assert endpoint["confirmatory_observations_decoded"] == 0
    assert endpoint["confirmatory_actions_computed"] == 0
    assert endpoint["headroom_allowed"] is False
    assert endpoint["validation_search_allowed"] is False
    outcome = state["epoch_4_cycle_16_iarc_stage_0a_outcome"]
    assert outcome["final_decision"] == "IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert outcome["valid_scientific_kill"] is False
    assert outcome["gradient_pairs_completed"] == outcome["gradient_pairs_planned"] == 40
    assert outcome["exception_count"] == 0
    assert outcome["duplicate_key_count"] == 0
    assert outcome["conflict_count"] == 18
    assert outcome["conflict_family_count"] == 4
    assert outcome["projection_constraint_pass_count"] == outcome["projected_row_count"] == 18
    assert outcome["agreeing_unchanged_count"] == outcome["agreeing_row_count"] == 22
    assert outcome["dataset_range_valid_fraction"] == 0.3
    assert outcome["invalid_validation_pair_count"] == 28
    assert outcome["confirmatory_observations_decoded"] == 0
    assert outcome["confirmatory_actions_computed"] == 0
    assert outcome["one_check_allowed"] is False
    assert outcome["stage_0b_allowed"] is False
    resource_audit = state["resource_contention_audit_20260715"]
    assert resource_audit["active_linux_research_worker_found"] is False
    assert resource_audit["completed_episode_count"] == resource_audit["planned_episode_count"] == 200
    assert resource_audit["exception_count"] == 0
    assert resource_audit["duplicate_key_count"] == 0
    assert resource_audit["missing_manifest_key_count"] == 0
    assert resource_audit["extra_result_key_count"] == 0
    assert resource_audit["simulator_synchronous"] is True
    assert resource_audit["performance_metrics_quarantined_if_overlap_unknown"] is True
    registry = json.loads(
        (REPO_ROOT / "reports" / "resource_contention_intervals.json").read_text(encoding="utf-8")
    )
    latest_interval = next(
        interval
        for interval in registry["intervals"]
        if interval["id"] == "windows_efficiency_mode_vmmemwsl_20260715_goal_pause_2_user_reported"
    )
    latest_run = latest_interval["durable_run_audit"]
    assert latest_interval["active_linux_research_worker_found_at_audit"] is False
    assert latest_run["pid"] == 387
    assert latest_run["pid_alive"] is False
    assert latest_run["partial_json_parsed"] is True
    assert latest_run["result_json_parsed"] is True
    assert latest_run["completed_optimizer_steps"] == latest_run["planned_optimizer_steps"] == 300
    assert latest_run["completed_microbatch_count"] == latest_run["planned_microbatch_count"] == 2400
    assert latest_run["exception_count"] == 0
    assert latest_run["duplicate_key_count"] == 0
    assert latest_run["closed_loop_rows_present"] is False
    assert latest_run["timing_and_resource_evidence_quarantined"] is True
    assert latest_run["resume_or_relaunch_performed"] is False
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
    assert "epoch_4_cycle_11_g3p_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_11_g3p_data_or_supervision_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_12_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_12_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_12_cala_design_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_13_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_13_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_13_rar_design_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_14_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_14_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_proposal_pending" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_14_covi_implementation_optimization_stop_recorded" in state["completed_stages"]
    assert "post_covi_lora_governance_installed" in state["completed_stages"]
    assert "epoch_4_cycle_15_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_15_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_prototype_protocol_frozen" in state["completed_stages"]
    covi_outcome = state["epoch_4_cycle_14_covi_stage_0_outcome"]
    assert covi_outcome["final_decision"] == "COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL"
    assert covi_outcome["objective_gradient_ratio"] == 1345.9529990435792
    assert covi_outcome["objective_gradient_ratio_max"] == 100.0
    assert covi_outcome["confirmatory_test_records_decoded"] == 0
    assert covi_outcome["scientific_method_kill"] is False
    lift = state["epoch_4_cycle_15_pre_proposal"]
    assert lift["method"] == "LIFT-VLA"
    assert lift["selection_decision"] == "SELECT_LIFT_VLA"
    assert lift["candidate_count"] == 3
    assert lift["selected_score"] == 90
    assert lift["selected_contribution_type"] == "CROSS_DOMAIN_MECHANISM_TRANSFER"
    assert lift["scientific_method"] == "pathwise conditional-minus-unconditional vector-field guidance through every SmolVLA flow step"
    assert lift["low_compute_parameterization"] == "frozen two-branch SmolVLA inference with no trainable parameters"
    assert lift["standard_lora_required"] is False
    assert lift["conditional_fifth_policy"] is None
    assert lift["bounded_validation_search_max_configs"] == 3
    assert lift["first_comparison_policies"] == [
        "frozen_smolvla",
        "training_free_cag_proxy",
        "lift_full_pathwise_guidance",
        "lift_last_step_only_ablation",
    ]
    assert lift["proposal_hash"] == LIFT_PROPOSAL_HASH
    assert lift["reviewer_attack_pending"] is False
    assert lift["reviewer_attack_completed"] is True
    assert lift["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert lift["rebuttal_completed"] is True
    assert lift["rebuttal_decision"] == "LIFT_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lift["rollout_allowed"] is False
    assert lift["confirmatory_test_tuning_happened"] is False
    lift_review = state["epoch_4_cycle_15_lift_review"]
    assert lift_review["final_decision"] == "LIFT_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lift_review["accepted_narrow_vla_flow_novelty"] is True
    assert lift_review["scoreable_counterfactual_benchmark_gate_required"] is True
    assert lift_review["native_flow_space_cag_required"] is True
    assert lift_review["same_noise_coupling_required"] is True
    assert lift_review["matched_compute_last_step_ablation_required"] is True
    assert lift_review["practical_equivalence_threshold_required"] is True
    assert lift_review["base_and_cag_headroom_required"] is True
    assert lift_review["one_chunk_memory_latency_gate_required"] is True
    assert lift_review["standard_lora_required"] is False
    assert lift_review["fifth_policy_required"] is False
    lift_math = state["epoch_4_cycle_15_lift_mathematical_audit"]
    assert lift_math["final_decision"] == "LIFT_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lift_math["native_chunk_shape"] == [1, 50, 32]
    assert lift_math["canonical_policy_chunk_shape"] == [1, 50, 7]
    assert lift_math["flow_steps"] == 10
    assert lift_math["matched_compute_field_evaluations"] == 20
    assert lift_math["practical_threshold_construction_frozen"] is True
    lift_prereg = state["epoch_4_cycle_15_lift_preregistration"]
    assert lift_prereg["final_decision"] == "LIFT_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert lift_prereg["discovery_target_tasks"] == [0, 1, 2, 3]
    assert lift_prereg["validation_target_tasks"] == [4, 5, 6]
    assert lift_prereg["confirmatory_target_tasks"] == [7, 8, 9]
    assert lift_prereg["guidance_scales"] == [1.25, 1.5, 2.0]
    lift_proto = state["epoch_4_cycle_15_lift_prototype_protocol"]
    assert lift_proto["final_decision"] == "LIFT_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert lift_proto["confirmatory_policy_observations_decoded_max"] == 0
    assert lift_proto["confirmatory_policy_actions_computed_max"] == 0
    assert lift_proto["stage_0_pending"] is True
    lift_stage_0 = state["epoch_4_cycle_15_lift_stage_0"]
    assert lift_stage_0["final_decision"] == "LIFT_COMPUTE_INFEASIBLE"
    assert lift_stage_0["manifest_rows_valid"] == 20
    assert lift_stage_0["manifest_rows_total"] == 20
    assert lift_stage_0["identity_native_max_abs_error"] == 0.0
    assert lift_stage_0["identity_postprocessed_max_abs_error"] == 0.0
    assert lift_stage_0["action_finite_fraction"] == 1.0
    assert lift_stage_0["action_range_valid_fraction"] == 0.8023809523809524
    assert lift_stage_0["confirmatory_policy_observations_decoded"] == 0
    assert lift_stage_0["confirmatory_policy_actions_computed"] == 0
    assert lift_stage_0["validation_search_happened"] is False
    assert lift_stage_0["closed_loop_experiment_happened"] is False
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
    assert g3p["rebuttal_pending"] is False
    assert g3p["rebuttal_completed"] is True
    assert g3p["rebuttal_decision"] == "G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert g3p["mathematical_audit"] == "reports/g3p_vla/mathematical_mechanism_audit.md"
    assert g3p["mathematical_audit_pending"] is False
    assert g3p["mathematical_audit_completed"] is True
    assert g3p["mathematical_audit_decision"] == "G3P_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert g3p["preregistration"] == "reports/g3p_vla/preregistration.md"
    assert g3p["prototype_protocol"] == "reports/g3p_vla/prototype_protocol.md"
    assert g3p["preregistration_completed"] is True
    assert g3p["prototype_protocol_completed"] is True
    assert g3p["prototype_protocol_decision"] == "G3P_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert g3p["stage_0_pending"] is False
    assert g3p["stage_0_completed"] is True
    assert g3p["stage_0_decision"] == "DATA_OR_SUPERVISION_FAILURE"
    assert g3p["stage_0_failure_class"] == "DATA_OR_SUPERVISION_FAILURE"
    assert g3p["validation_search_allowed"] is False
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
    assert g3p_proposal["rebuttal"] == "reports/g3p_vla/researcher_rebuttal.md"
    assert g3p_proposal["rebuttal_decision"] == "G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert g3p_proposal["rebuttal_completed"] is True
    assert g3p_proposal["training_happened"] is False
    assert g3p_proposal["confirmatory_test_tuning_happened"] is False
    g3p_review = state["epoch_4_cycle_11_g3p_review"]
    assert g3p_review["reviewer_attack"] == "reports/g3p_vla/reviewer_attack.md"
    assert g3p_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert g3p_review["closest_prior_proxy_must_remain_transparent"] is True
    assert g3p_review["source_gate_required_before_rollout"] is True
    assert g3p_review["simple_heuristic_must_remain_live"] is True
    assert g3p_review["confirmatory_test_tuning_happened"] is False
    g3p_rebuttal = state["epoch_4_cycle_11_g3p_rebuttal"]
    assert g3p_rebuttal["researcher_rebuttal"] == "reports/g3p_vla/researcher_rebuttal.md"
    assert g3p_rebuttal["final_decision"] == "G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert g3p_rebuttal["accepted_narrowed_novelty"] is True
    assert g3p_rebuttal["accepted_source_legality_gate"] is True
    assert g3p_rebuttal["accepted_simple_heuristic_killer"] is True
    assert g3p_rebuttal["accepted_identity_preserving_integration"] is True
    assert g3p_rebuttal["training_happened"] is False
    g3p_audit = state["epoch_4_cycle_11_g3p_mathematical_audit"]
    assert g3p_audit["mathematical_audit"] == "reports/g3p_vla/mathematical_mechanism_audit.md"
    assert g3p_audit["final_decision"] == "G3P_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert g3p_audit["kl_between_deterministic_actions_forbidden"] is True
    assert g3p_audit["identity_preserving_adapter_required"] is True
    assert g3p_audit["source_gate_required"] is True
    assert g3p_audit["bounded_validation_search_max_configs"] == 6
    assert g3p_audit["confirmatory_test_tuning_happened"] is False
    g3p_prereg = state["epoch_4_cycle_11_g3p_preregistration"]
    assert g3p_prereg["preregistration"] == "reports/g3p_vla/preregistration.md"
    assert g3p_prereg["final_decision"] == "G3P_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert g3p_prereg["bounded_validation_search_max_configs"] == 6
    assert g3p_prereg["stage_0_pending"] is True
    g3p_proto = state["epoch_4_cycle_11_g3p_prototype_protocol"]
    assert g3p_proto["prototype_protocol"] == "reports/g3p_vla/prototype_protocol.md"
    assert g3p_proto["final_decision"] == "G3P_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert g3p_proto["first_comparison_policies"] == [
        "frozen_smolvla",
        "g3p_3d_point_proxy",
        "g3p_full",
        "g3p_no_3d_no_injection_ablation",
        "simple_2d_phase_or_nearest_object_heuristic",
    ]
    assert g3p_proto["stage_0_decision"] == "DATA_OR_SUPERVISION_FAILURE"
    assert g3p_proto["stage_0_passed"] is False
    g3p_development = state["epoch_4_cycle_11_g3p_development_outcome"]
    assert g3p_development["final_decision"] == "DATA_OR_SUPERVISION_FAILURE"
    assert g3p_development["stage_0_completed"] is True
    assert g3p_development["stage_0_passed"] is False
    assert g3p_development["valid_closed_loop_scientific_kill"] is False
    assert g3p_development["closed_loop_experiment_happened"] is False
    assert g3p_development["training_happened"] is False
    assert g3p_development["validation_search_happened"] is False
    assert g3p_development["confirmatory_test_tuning_happened"] is False
    assert g3p_development["source_gate_passed"] is True
    assert g3p_development["train_material_point_fraction"] == 0.9982142857142857
    assert g3p_development["validation_material_point_fraction"] == 1.0
    assert "validation material point fraction collapsed" in " ".join(g3p_development["hard_stop_reasons"])
    cala = state["epoch_4_cycle_12_pre_proposal"]
    assert cala["method"] == "CALA-VLA"
    assert cala["selection_decision"] == "SELECT_CALA_VLA"
    assert cala["candidate_generation"] == "reports/epoch_4_cycle_12_candidate_generation.md"
    assert cala["prior_mechanism_map"] == "reports/epoch_4_cycle_12_prior_mechanism_map.md"
    assert cala["candidate_count"] == 3
    assert cala["closest_prior"] == "CAC-VLA"
    assert cala["closest_prior_url"] == "https://arxiv.org/abs/2607.04816"
    assert cala["secondary_priors"] == ["VLS", "World Pilot"]
    assert cala["selected_score"] == 94
    assert cala["selected_contribution_type"] == "PRIOR_EXTENSION"
    assert cala["first_comparison_policies"] == [
        "frozen_smolvla",
        "cac_vla_latent_action_proxy",
        "cala_full",
        "cala_no_context_gate_ablation",
        "task_mean_latent_action_baseline",
    ]
    assert cala["latent_label_health_gate_required"] is True
    assert cala["identity_preserving_gate_required"] is True
    assert cala["official_closest_prior_code_or_checkpoint_verified"] is False
    assert cala["rollout_allowed"] is False
    assert cala["closed_loop_experiment_happened"] is False
    assert cala["training_happened"] is False
    assert cala["validation_search_happened"] is False
    assert cala["confirmatory_test_tuning_happened"] is False
    assert cala["proposal"] == "reports/cala_vla/researcher_proposal.md"
    assert cala["proposal_hash"] == CALA_PROPOSAL_HASH
    assert cala["proposal_hash_file"] == "reports/cala_vla/proposal_hash.txt"
    assert cala["proposal_decision"] == "CALA_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert cala["reviewer_attack"] == "reports/cala_vla/reviewer_attack.md"
    assert cala["reviewer_attack_pending"] is False
    assert cala["reviewer_attack_completed"] is True
    assert cala["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert cala["rebuttal"] == "reports/cala_vla/researcher_rebuttal.md"
    assert cala["rebuttal_pending"] is False
    assert cala["rebuttal_completed"] is True
    assert cala["rebuttal_decision"] == "CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert cala["mathematical_audit"] == "reports/cala_vla/mathematical_mechanism_audit.md"
    assert cala["mathematical_audit_completed"] is True
    assert cala["mathematical_audit_decision"] == "CALA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cala["preregistration"] == "reports/cala_vla/preregistration.md"
    assert cala["prototype_protocol"] == "reports/cala_vla/prototype_protocol.md"
    assert cala["preregistration_completed"] is True
    assert cala["prototype_protocol_completed"] is True
    assert cala["prototype_protocol_decision"] == "CALA_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert cala["stage_0_pending"] is False
    assert cala["stage_0_audit"] == "reports/cala_vla/development_audit.json"
    assert cala["stage_0_decision"] == "DESIGN_FAILURE"
    assert cala["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert cala["validation_search_allowed"] is False
    assert cala["stage_0_training_happened"] is False
    assert cala["stage_0_closed_loop_experiment_happened"] is False
    assert cala["stage_0_confirmatory_test_tuning_happened"] is False
    assert cala["stage_0_hard_stop_reasons"] == ["latent predictability margin below minimum: -0.011718"]
    cala_proposal = state["epoch_4_cycle_12_cala_proposal"]
    assert cala_proposal["proposal"] == "reports/cala_vla/researcher_proposal.md"
    assert cala_proposal["proposal_hash"] == CALA_PROPOSAL_HASH
    assert cala_proposal["proposal_hash_file"] == "reports/cala_vla/proposal_hash.txt"
    assert cala_proposal["final_decision"] == "CALA_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert cala_proposal["closed_loop_experiment_happened"] is False
    assert cala_proposal["training_happened"] is False
    assert cala_proposal["validation_search_happened"] is False
    assert cala_proposal["confirmatory_test_tuning_happened"] is False
    assert cala_proposal["reviewer_attack"] == "reports/cala_vla/reviewer_attack.md"
    assert cala_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert cala_proposal["reviewer_attack_completed"] is True
    assert cala_proposal["researcher_rebuttal"] == "reports/cala_vla/researcher_rebuttal.md"
    assert cala_proposal["rebuttal_decision"] == "CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert cala_proposal["rebuttal_completed"] is True
    assert cala_proposal["mathematical_audit"] == "reports/cala_vla/mathematical_mechanism_audit.md"
    assert cala_proposal["mathematical_audit_decision"] == "CALA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cala_proposal["mathematical_audit_completed"] is True
    cala_review = state["epoch_4_cycle_12_cala_review"]
    assert cala_review["reviewer_attack"] == "reports/cala_vla/reviewer_attack.md"
    assert cala_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert cala_review["closest_prior_proxy_must_remain_transparent"] is True
    assert cala_review["future_action_leakage_gate_required"] is True
    assert cala_review["task_mean_baseline_must_remain_live"] is True
    assert cala_review["training_happened"] is False
    assert cala_review["confirmatory_test_tuning_happened"] is False
    assert cala_review["researcher_rebuttal"] == "reports/cala_vla/researcher_rebuttal.md"
    assert cala_review["rebuttal_decision"] == "CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert cala_review["rebuttal_completed"] is True
    assert cala_review["mathematical_audit"] == "reports/cala_vla/mathematical_mechanism_audit.md"
    assert cala_review["mathematical_audit_decision"] == "CALA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cala_review["mathematical_audit_completed"] is True
    cala_rebuttal = state["epoch_4_cycle_12_cala_rebuttal"]
    assert cala_rebuttal["researcher_rebuttal"] == "reports/cala_vla/researcher_rebuttal.md"
    assert cala_rebuttal["final_decision"] == "CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert cala_rebuttal["accepted_narrowed_novelty"] is True
    assert cala_rebuttal["accepted_future_action_source_gate"] is True
    assert cala_rebuttal["accepted_task_mean_simple_killer"] is True
    assert cala_rebuttal["accepted_identity_preserving_integration"] is True
    assert cala_rebuttal["training_happened"] is False
    assert cala_rebuttal["confirmatory_test_tuning_happened"] is False
    assert cala_rebuttal["mathematical_audit"] == "reports/cala_vla/mathematical_mechanism_audit.md"
    assert cala_rebuttal["mathematical_audit_decision"] == "CALA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cala_rebuttal["mathematical_audit_completed"] is True
    cala_audit = state["epoch_4_cycle_12_cala_mathematical_audit"]
    assert cala_audit["mathematical_audit"] == "reports/cala_vla/mathematical_mechanism_audit.md"
    assert cala_audit["final_decision"] == "CALA_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cala_audit["kl_between_deterministic_actions_forbidden"] is True
    assert cala_audit["identity_preserving_adapter_required"] is True
    assert cala_audit["future_action_inference_forbidden"] is True
    assert cala_audit["bounded_validation_search_max_configs"] == 6
    assert cala_audit["training_happened"] is False
    assert cala_audit["confirmatory_test_tuning_happened"] is False
    assert cala_audit["preregistration"] == "reports/cala_vla/preregistration.md"
    assert cala_audit["prototype_protocol"] == "reports/cala_vla/prototype_protocol.md"
    assert cala_audit["preregistration_completed"] is True
    assert cala_audit["prototype_protocol_completed"] is True
    cala_prereg = state["epoch_4_cycle_12_cala_preregistration"]
    assert cala_prereg["preregistration"] == "reports/cala_vla/preregistration.md"
    assert cala_prereg["final_decision"] == "CALA_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert cala_prereg["bounded_validation_search_max_configs"] == 6
    assert cala_prereg["stage_0_pending"] is True
    cala_proto = state["epoch_4_cycle_12_cala_prototype_protocol"]
    assert cala_proto["prototype_protocol"] == "reports/cala_vla/prototype_protocol.md"
    assert cala_proto["final_decision"] == "CALA_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert cala_proto["first_comparison_policies"] == [
        "frozen_smolvla",
        "cac_vla_latent_action_proxy",
        "cala_full",
        "cala_no_context_gate_ablation",
        "task_mean_latent_action_baseline",
    ]
    assert cala_proto["stage_0_next"] == "development_audit_completed_stop"
    assert cala_proto["stage_0_audit"] == "reports/cala_vla/development_audit.json"
    assert cala_proto["stage_0_decision"] == "DESIGN_FAILURE"
    assert cala_proto["stage_0_completed"] is True
    assert cala_proto["stage_0_passed"] is False
    assert cala_proto["validation_search_allowed"] is False
    cala_outcome = state["epoch_4_cycle_12_cala_development_outcome"]
    assert cala_outcome["final_decision"] == "DESIGN_FAILURE"
    assert cala_outcome["stage_0_completed"] is True
    assert cala_outcome["stage_0_passed"] is False
    assert cala_outcome["source_gate_passed"] is True
    assert cala_outcome["future_action_segments_used_at_inference"] is False
    assert cala_outcome["latent_labels_used_at_inference"] is False
    assert cala_outcome["latent_predictability_margin"] == -0.01171824382857035
    assert cala_outcome["best_trivial_baseline"] == "action_history_only"
    assert cala_outcome["validation_search_allowed"] is False
    assert cala_outcome["training_happened"] is False
    assert cala_outcome["closed_loop_experiment_happened"] is False
    cycle13 = state["epoch_4_cycle_13_pre_stage"]
    assert cycle13["candidate_search_pending"] is True
    assert cycle13["previous_method"] == "CALA-VLA"
    assert cycle13["must_not_rescue_previous_method"] is True
    rar = state["epoch_4_cycle_13_pre_proposal"]
    assert rar["method"] == "RAR-VLA"
    assert rar["selection_decision"] == "SELECT_RAR_VLA"
    assert rar["candidate_generation"] == "reports/epoch_4_cycle_13_candidate_generation.md"
    assert rar["prior_mechanism_map"] == "reports/epoch_4_cycle_13_prior_mechanism_map.md"
    assert rar["candidate_count"] == 3
    assert rar["closest_prior"] == "AR-VLA"
    assert rar["closest_prior_url"] == "https://arxiv.org/abs/2603.10126"
    assert rar["secondary_priors"] == ["ReactVLA", "DSWAM"]
    assert rar["selected_score"] == 91
    assert rar["selected_contribution_type"] == "PRIOR_EXTENSION"
    assert rar["first_comparison_policies"] == [
        "frozen_smolvla",
        "ar_vla_reanchored_expert_proxy",
        "rar_full",
        "rar_no_reanchor_memory_ablation",
        "ema_action_history_baseline",
    ]
    assert rar["causal_source_gate_required"] is True
    assert rar["action_history_simple_killer_required"] is True
    assert rar["identity_preserving_gate_required"] is True
    assert rar["official_closest_prior_code_or_checkpoint_verified"] is False
    assert rar["rollout_allowed"] is False
    assert rar["closed_loop_experiment_happened"] is False
    assert rar["training_happened"] is False
    assert rar["validation_search_happened"] is False
    assert rar["confirmatory_test_tuning_happened"] is False
    assert rar["must_not_rescue_previous_method"] is True
    assert rar["proposal"] == "reports/rar_vla/researcher_proposal.md"
    assert rar["proposal_hash"] == RAR_PROPOSAL_HASH
    assert rar["proposal_hash_file"] == "reports/rar_vla/proposal_hash.txt"
    assert rar["proposal_decision"] == "RAR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert rar["reviewer_attack"] == "reports/rar_vla/reviewer_attack.md"
    assert rar["reviewer_attack_pending"] is False
    assert rar["reviewer_attack_completed"] is True
    assert rar["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert rar["rebuttal"] == "reports/rar_vla/researcher_rebuttal.md"
    assert rar["rebuttal_pending"] is False
    assert rar["rebuttal_completed"] is True
    assert rar["rebuttal_decision"] == "RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert rar["mathematical_audit"] == "reports/rar_vla/mathematical_mechanism_audit.md"
    assert rar["mathematical_audit_completed"] is True
    assert rar["mathematical_audit_decision"] == "RAR_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert rar["preregistration"] == "reports/rar_vla/preregistration.md"
    assert rar["prototype_protocol"] == "reports/rar_vla/prototype_protocol.md"
    assert rar["preregistration_completed"] is True
    assert rar["prototype_protocol_completed"] is True
    assert rar["prototype_protocol_decision"] == "RAR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert rar["stage_0_pending"] is False
    assert rar["stage_0_audit"] == "reports/rar_vla/development_audit.json"
    assert rar["stage_0_audit_md"] == "reports/rar_vla/development_audit.md"
    assert rar["source_gate_manifest"] == "reports/rar_vla/source_gate_manifest.json"
    assert rar["history_feature_manifest"] == "reports/rar_vla/history_feature_manifest.json"
    assert rar["split_manifest"] == "reports/rar_vla/split_manifest.json"
    assert rar["stage_0_completed"] is True
    assert rar["stage_0_decision"] == "DESIGN_FAILURE"
    assert rar["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert rar["stage_0_hard_stop_reasons"] == ["residual predictability margin below minimum: -0.038376"]
    assert rar["stage_0_training_happened"] is False
    assert rar["stage_0_closed_loop_experiment_happened"] is False
    assert rar["stage_0_confirmatory_test_tuning_happened"] is False
    assert rar["validation_search_allowed"] is False
    rar_proposal = state["epoch_4_cycle_13_rar_proposal"]
    assert rar_proposal["proposal"] == "reports/rar_vla/researcher_proposal.md"
    assert rar_proposal["proposal_hash"] == RAR_PROPOSAL_HASH
    assert rar_proposal["proposal_hash_file"] == "reports/rar_vla/proposal_hash.txt"
    assert rar_proposal["final_decision"] == "RAR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert rar_proposal["closest_prior"] == "AR-VLA"
    assert rar_proposal["closed_loop_experiment_happened"] is False
    assert rar_proposal["training_happened"] is False
    assert rar_proposal["validation_search_happened"] is False
    assert rar_proposal["confirmatory_test_tuning_happened"] is False
    assert rar_proposal["reviewer_attack"] == "reports/rar_vla/reviewer_attack.md"
    assert rar_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert rar_proposal["reviewer_attack_completed"] is True
    rar_review = state["epoch_4_cycle_13_rar_review"]
    assert rar_review["reviewer_attack"] == "reports/rar_vla/reviewer_attack.md"
    assert rar_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert rar_review["closest_prior_proxy_must_remain_transparent"] is True
    assert rar_review["remac_tas_distinction_required"] is True
    assert rar_review["ema_action_history_baseline_must_remain_live"] is True
    assert rar_review["stage_0_inter_and_intra_chunk_diagnostics_required"] is True
    assert rar_review["training_happened"] is False
    assert rar_review["confirmatory_test_tuning_happened"] is False
    assert rar_review["researcher_rebuttal"] == "reports/rar_vla/researcher_rebuttal.md"
    assert rar_review["rebuttal_decision"] == "RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert rar_review["rebuttal_completed"] is True
    rar_rebuttal = state["epoch_4_cycle_13_rar_rebuttal"]
    assert rar_rebuttal["researcher_rebuttal"] == "reports/rar_vla/researcher_rebuttal.md"
    assert rar_rebuttal["final_decision"] == "RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert rar_rebuttal["accepted_narrowed_novelty"] is True
    assert rar_rebuttal["accepted_remac_tas_distinction"] is True
    assert rar_rebuttal["accepted_ema_action_history_killer"] is True
    assert rar_rebuttal["accepted_transparent_ar_proxy"] is True
    assert rar_rebuttal["accepted_identity_preserving_integration"] is True
    assert rar_rebuttal["training_happened"] is False
    assert rar_rebuttal["confirmatory_test_tuning_happened"] is False
    assert rar_rebuttal["mathematical_audit"] == "reports/rar_vla/mathematical_mechanism_audit.md"
    assert rar_rebuttal["mathematical_audit_decision"] == "RAR_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert rar_rebuttal["mathematical_audit_completed"] is True
    rar_audit = state["epoch_4_cycle_13_rar_mathematical_audit"]
    assert rar_audit["mathematical_audit"] == "reports/rar_vla/mathematical_mechanism_audit.md"
    assert rar_audit["final_decision"] == "RAR_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert rar_audit["kl_between_deterministic_actions_forbidden"] is True
    assert rar_audit["identity_preserving_adapter_required"] is True
    assert rar_audit["ema_action_history_baseline_required"] is True
    assert rar_audit["remac_tas_distinction_required"] is True
    assert rar_audit["bounded_validation_search_max_configs"] == 6
    assert rar_audit["training_happened"] is False
    assert rar_audit["confirmatory_test_tuning_happened"] is False
    assert rar_audit["preregistration"] == "reports/rar_vla/preregistration.md"
    assert rar_audit["prototype_protocol"] == "reports/rar_vla/prototype_protocol.md"
    rar_prereg = state["epoch_4_cycle_13_rar_preregistration"]
    assert rar_prereg["preregistration"] == "reports/rar_vla/preregistration.md"
    assert rar_prereg["final_decision"] == "RAR_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert rar_prereg["bounded_validation_search_max_configs"] == 6
    assert rar_prereg["stage_0_pending"] is True
    rar_proto = state["epoch_4_cycle_13_rar_prototype_protocol"]
    assert rar_proto["prototype_protocol"] == "reports/rar_vla/prototype_protocol.md"
    assert rar_proto["final_decision"] == "RAR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert rar_proto["first_comparison_policies"] == [
        "frozen_smolvla",
        "ar_vla_reanchored_expert_proxy",
        "rar_full",
        "rar_no_reanchor_memory_ablation",
        "ema_action_history_baseline",
    ]
    assert rar_proto["stage_0_next"] == "development_audit_completed_stop"
    assert rar_proto["stage_0_audit"] == "reports/rar_vla/development_audit.json"
    assert rar_proto["stage_0_decision"] == "DESIGN_FAILURE"
    assert rar_proto["stage_0_completed"] is True
    assert rar_proto["stage_0_passed"] is False
    assert rar_proto["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert rar_proto["validation_search_allowed"] is False
    rar_outcome = state["epoch_4_cycle_13_rar_development_outcome"]
    assert rar_outcome["final_decision"] == "DESIGN_FAILURE"
    assert rar_outcome["stage_0_completed"] is True
    assert rar_outcome["stage_0_passed"] is False
    assert rar_outcome["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert rar_outcome["source_gate_passed"] is True
    assert rar_outcome["future_actions_used_at_inference"] is False
    assert rar_outcome["cala_latents_used_at_inference"] is False
    assert rar_outcome["residual_predictability_margin"] == -0.03837609884238533
    assert rar_outcome["best_trivial_baseline"] == "zero_residual"
    assert rar_outcome["validation_search_allowed"] is False
    assert rar_outcome["training_happened"] is False
    assert rar_outcome["closed_loop_experiment_happened"] is False
    cycle14 = state["epoch_4_cycle_14_pre_stage"]
    assert cycle14["candidate_search_pending"] is True
    assert cycle14["previous_method"] == "RAR-VLA"
    assert cycle14["must_not_rescue_previous_method"] is True
    covi = state["epoch_4_cycle_14_pre_proposal"]
    assert covi["method"] == "COVI-VLA"
    assert covi["selection_decision"] == "SELECT_COVI_VLA"
    assert covi["candidate_generation"] == "reports/epoch_4_cycle_14_candidate_generation.md"
    assert covi["prior_mechanism_map"] == "reports/epoch_4_cycle_14_prior_mechanism_map.md"
    assert covi["candidate_count"] == 3
    assert covi["closest_prior"] == "LIBERO-Occ / Viewpoint Imagination"
    assert covi["closest_prior_url"] == "https://arxiv.org/abs/2606.10862"
    assert covi["closest_prior_code_url"] == "https://github.com/litsh/Libero-Occ"
    assert covi["secondary_priors"] == ["CamVLA", "STRONG-VLA"]
    assert covi["selected_score"] == 91
    assert covi["selected_contribution_type"] == "NEW_DEPLOYMENT_PROBLEM"
    assert covi["first_comparison_policies"] == [
        "frozen_smolvla_occluded",
        "vim_view_imagination_proxy",
        "covi_full",
        "covi_no_imagined_view_ablation",
        "random_cutout_clean_retention_baseline",
    ]
    assert covi["stage_0_required"] is True
    assert covi["occlusion_headroom_required"] is True
    assert covi["source_gate_required"] is True
    assert covi["view_completion_label_health_required"] is True
    assert covi["random_cutout_simple_killer_required"] is True
    assert covi["identity_preserving_visual_adapter_required"] is True
    assert covi["official_closest_prior_code_or_checkpoint_verified"] is False
    assert covi["rollout_allowed"] is False
    assert covi["closed_loop_experiment_happened"] is False
    assert covi["training_happened"] is False
    assert covi["validation_search_happened"] is False
    assert covi["confirmatory_test_tuning_happened"] is False
    assert covi["previous_method_stop"] == "RAR_STAGE_0_DESIGN_FAILURE"
    assert covi["must_not_rescue_previous_method"] is True
    assert covi["proposal"] == "reports/covi_vla/researcher_proposal.md"
    assert covi["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi["proposal_hash_file"] == "reports/covi_vla/proposal_hash.txt"
    assert covi["proposal_decision"] == "COVI_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert covi["reviewer_attack_pending"] is False
    assert covi["reviewer_attack"] == "reports/covi_vla/reviewer_attack.md"
    assert covi["reviewer_attack_completed"] is True
    assert covi["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert covi["rebuttal_pending"] is False
    assert covi["rebuttal"] == "reports/covi_vla/researcher_rebuttal.md"
    assert covi["rebuttal_completed"] is True
    assert covi["rebuttal_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi["mathematical_audit_pending"] is False
    assert covi["mathematical_audit_completed"] is True
    assert covi["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi["preregistration"] == "reports/covi_vla/preregistration.md"
    assert covi["prototype_protocol"] == "reports/covi_vla/prototype_protocol.md"
    assert covi["preregistration_completed"] is True
    assert covi["prototype_protocol_completed"] is True
    assert covi["false_negative_safeguard_required"] is True
    assert covi["stage_0_pending"] is True
    covi_proposal = state["epoch_4_cycle_14_covi_proposal"]
    assert covi_proposal["method"] == "COVI-VLA"
    assert covi_proposal["proposal"] == "reports/covi_vla/researcher_proposal.md"
    assert covi_proposal["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi_proposal["proposal_hash_file"] == "reports/covi_vla/proposal_hash.txt"
    assert covi_proposal["final_decision"] == "COVI_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert covi_proposal["reviewer_attack"] == "reports/covi_vla/reviewer_attack.md"
    assert covi_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert covi_proposal["reviewer_attack_completed"] is True
    assert covi_proposal["researcher_rebuttal"] == "reports/covi_vla/researcher_rebuttal.md"
    assert covi_proposal["rebuttal_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi_proposal["rebuttal_completed"] is True
    assert covi_proposal["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi_proposal["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi_proposal["mathematical_audit_completed"] is True
    assert covi_proposal["closed_loop_experiment_happened"] is False
    assert covi_proposal["training_happened"] is False
    assert covi_proposal["validation_search_happened"] is False
    assert covi_proposal["confirmatory_test_tuning_happened"] is False
    covi_review = state["epoch_4_cycle_14_covi_review"]
    assert covi_review["method"] == "COVI-VLA"
    assert covi_review["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi_review["reviewer_attack"] == "reports/covi_vla/reviewer_attack.md"
    assert covi_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert covi_review["vim_proxy_must_remain_transparent"] is True
    assert covi_review["direct_two_camera_fusion_diagnostic_required"] is True
    assert covi_review["random_cutout_simple_killer_must_remain_live"] is True
    assert covi_review["physical_occlusion_claim_must_be_validated"] is True
    assert covi_review["researcher_rebuttal"] == "reports/covi_vla/researcher_rebuttal.md"
    assert covi_review["rebuttal_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi_review["rebuttal_completed"] is True
    assert covi_review["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi_review["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi_review["mathematical_audit_completed"] is True
    covi_rebuttal = state["epoch_4_cycle_14_covi_rebuttal"]
    assert covi_rebuttal["method"] == "COVI-VLA"
    assert covi_rebuttal["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi_rebuttal["researcher_rebuttal"] == "reports/covi_vla/researcher_rebuttal.md"
    assert covi_rebuttal["final_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi_rebuttal["accepted_narrowed_novelty"] is True
    assert covi_rebuttal["accepted_transparent_vim_proxy"] is True
    assert covi_rebuttal["accepted_direct_two_camera_fusion_diagnostic"] is True
    assert covi_rebuttal["accepted_random_cutout_simple_killer"] is True
    assert covi_rebuttal["accepted_physical_occlusion_requirement"] is True
    assert covi_rebuttal["accepted_identity_preserving_integration"] is True
    assert covi_rebuttal["accepted_no_privileged_inference"] is True
    assert covi_rebuttal["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi_rebuttal["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi_rebuttal["mathematical_audit_completed"] is True
    covi_audit = state["epoch_4_cycle_14_covi_mathematical_audit"]
    assert covi_audit["method"] == "COVI-VLA"
    assert covi_audit["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi_audit["final_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi_audit["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi_audit["mathematical_audit_completed"] is True
    assert covi_audit["closed_loop_experiment_happened"] is False
    assert covi_audit["training_happened"] is False
    assert covi_audit["validation_search_happened"] is False
    assert covi_audit["confirmatory_test_tuning_happened"] is False
    assert covi_audit["kl_between_deterministic_actions_forbidden"] is True
    assert covi_audit["transparent_vim_proxy_required"] is True
    assert covi_audit["direct_two_camera_fusion_diagnostic_required"] is True
    assert covi_audit["random_cutout_simple_killer_required"] is True
    assert covi_audit["identity_preserving_visual_adapter_required"] is True
    assert covi_audit["bounded_validation_search_max_configs"] == 6
    assert covi_audit["preregistration_completed"] is True
    assert covi_audit["prototype_protocol_completed"] is True
    covi_prereg = state["epoch_4_cycle_14_covi_preregistration"]
    assert covi_prereg["final_decision"] == "COVI_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert covi_prereg["reviewer_status"] == "APPROVE_WITH_FIXED_EMPIRICAL_RISKS"
    assert covi_prereg["discovery_fit_records"] == 600
    assert covi_prereg["discovery_one_check_records"] == 600
    assert covi_prereg["validation_records"] == 400
    assert covi_prereg["reserved_confirmatory_records"] == 1200
    assert covi_prereg["visual_token_shape_per_stream"] == [64, 960]
    assert covi_prereg["false_negative_safeguard_required"] is True
    assert covi_prereg["one_unresolved_check_max"] == 1
    covi_proto = state["epoch_4_cycle_14_covi_prototype_protocol"]
    assert covi_proto["final_decision"] == "COVI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert covi_proto["stage_0_result"] == "reports/covi_vla/stage_0_result.json"
    assert covi_proto["implementation_blocker"] == "reports/covi_vla/implementation_blocker.json"
    assert covi_proto["stage_0_pending"] is True
    assert state["task_reset_manifest"] is None
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["paired_cases_per_policy"] == 10
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["reset_seeds"] == [20261201, 20261202]
    assert state["checkpoint_path"] == "/mnt/c/assets/checkpoints/smolvla_libero"
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
