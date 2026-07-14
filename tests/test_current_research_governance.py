import json
from pathlib import Path

from scripts.check_current_research_governance import ALLOWED_FINAL_STATES, validate


REPO_ROOT = Path(__file__).resolve().parents[1]
PESA_PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"


def test_current_research_governance_validator_passes() -> None:
    assert validate(REPO_ROOT) == []


def test_active_state_records_closed_rac_stage_b_without_cycle_cap() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 9
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "SELECT_PESA_VLA"
    assert state["current_stage"] == "epoch_4_cycle_9_pesa_mathematical_audit_pending"
    assert state["method"] == "PESA-VLA"
    assert state["proposal_hash"] == PESA_PROPOSAL_HASH
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
    assert state["epoch_4_cycle_9_pre_stage_0"]["mathematical_audit"] is None
    assert state["epoch_4_cycle_9_pre_stage_0"]["first_comparison_policies"] == [
        "frozen_smolvla",
        "priorvla_style_proxy",
        "pesa_full",
        "pesa_no_spectral_no_prior_query_ablation",
        "standard_lora_or_clean_retention_baseline",
    ]
    assert state["epoch_4_cycle_9_pre_stage_0"]["closed_loop_experiment_happened"] is False
    assert state["epoch_4_cycle_9_pre_stage_0"]["confirmatory_test_tuning_happened"] is False
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
