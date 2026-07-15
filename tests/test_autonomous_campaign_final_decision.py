import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
PESA_PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"
EAC_PROPOSAL_HASH = "A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E"
G3P_PROPOSAL_HASH = "BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71"
CALA_PROPOSAL_HASH = "5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76"
RAR_PROPOSAL_HASH = "723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56"


def test_active_campaign_final_decision_is_nonterminal_pivot() -> None:
    final = (REPORTS / "autonomous_until_paper_final_decision.md").read_text(encoding="utf-8")

    assert "Current campaign decision: `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`" in final
    assert "This is not a terminal decision." in final
    assert "READY_TO_DRAFT_RAL_PAPER_PACKAGE" in final
    assert "FANG-VLA" in final
    assert "fang_c01" in final
    assert "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT" in final
    assert "11 / 40" in final
    assert "16 / 40" in final
    assert "EvoState-VLA" in final
    assert "A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9" in final
    assert "AUDIT_STOP_DESIGN_FAILURE" in final
    assert "0.024689" in final
    assert "CAVM-VLA" in final
    assert "24 / 58" in final
    assert "23 / 58" in final
    assert "RAC-VLA" in final
    assert "71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F" in final
    assert "rac_h4_a0.05" in final
    assert "0.585745" in final
    assert "STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED" in final
    assert "RAC full reached `0 / 10`" in final
    assert "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT" in final
    assert "RAC full reached `1 / 40`" in final
    assert "no-consequence ablation reached `2 / 40`" in final
    assert "post-RAC governance update is installed" in final
    assert "MTF-VLA" in final
    assert "11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31" in final
    assert "FrameSkip proxy" in final
    assert "mtf_r20_ret100" in final
    assert "0.643663" in final
    assert "adapter-training runner is now implemented and dry-run validated" in final
    assert "zero train/validation/test frame overlap" in final
    assert "MTF adapter training is now complete for all four trainable Stage A policies" in final
    assert "adapter_checkpoint_manifest.json" in final
    assert "stage_a_manifest.json" in final
    assert "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED" in final
    assert "stage_b_manifest.json" in final
    assert "stage_b_result.json" in final
    assert "Stage B is required" in final
    assert "MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD" in final
    assert "MTF full reached `26 / 40`" in final
    assert "no-retention ablation reached `32 / 40`" in final
    assert "Full-minus-no-retention paired delta was `-0.15`" in final
    assert "Epoch 4 Cycle 7 generated exactly three post-MTF candidates" in final
    assert "DAGR-VLA" in final
    assert "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" in final
    assert "dagr_a020_route_mlp" in final
    assert "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING" in final
    assert "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY" in final
    assert "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT" in final
    assert "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT" in final
    assert "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED" in final
    assert "DAGR full `6 / 10`" in final
    assert "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD" in final
    assert "200` total episodes" in final
    assert "DAGR full reached `18 / 40`" in final
    assert "gripper-transition heuristic reached `24 / 40`" in final
    assert "Full-minus-Base paired delta was `-0.25`" in final
    assert "At preflight time, no DAGR closed-loop rollout or confirmatory-test tuning had happened" in final
    assert "Epoch 4 Cycle 8 generated exactly three post-DAGR candidates" in final
    assert "MARC-VLA" in final
    assert "D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A" in final
    assert "MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "marc_a020_gate_mlp" in final
    assert "0.5457964262366295" in final
    assert "gate accuracy margin `0.0525`" in final
    assert "static_l1_mixture_baseline" in final
    assert "0.0019475044682621956" in final
    assert "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY" in final
    assert "runs\\marc_vla_checkpoints\\marc_a020_gate_mlp" in final
    assert "0.010693175718188286" in final
    assert "0.2307613492012024" in final
    assert "0.032826922833919525" in final
    assert "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT" in final
    assert "3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E" in final
    assert "MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT" in final
    assert "MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE" in final
    assert "Epoch 4 Cycle 9 generated exactly three post-MARC candidates" in final
    assert "PESA-VLA" in final
    assert "Prior-Expert Spectral Adaptation" in final
    assert PESA_PROPOSAL_HASH in final
    assert "reports/pesa_vla/reviewer_attack.md" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "reports/pesa_vla/researcher_rebuttal.md" in final
    assert "PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "reports/pesa_vla/mathematical_mechanism_audit.md" in final
    assert "PESA_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "reports/pesa_vla/preregistration.md" in final
    assert "reports/pesa_vla/prototype_protocol.md" in final
    assert "reports/pesa_vla/development_audit.json" in final
    assert "Final PESA Stage 0 decision: `DESIGN_FAILURE`" in final
    assert "query probe accuracy margin below minimum: -0.077500" in final
    assert "Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`" in final
    assert "Epoch 4 Cycle 10 generated exactly three post-PESA candidates" in final
    assert "EAC-VLA" in final
    assert "Entropy-Calibrated Adaptive Chunking" in final
    assert "AAC entropy-only proxy" in final
    assert "fixed short-replan simple killer" in final
    assert EAC_PROPOSAL_HASH in final
    assert "reports/eac_vla/reviewer_attack.md" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "reports/eac_vla/researcher_rebuttal.md" in final
    assert "EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "reports/eac_vla/mathematical_mechanism_audit.md" in final
    assert "EAC_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "reports/eac_vla/preregistration.md" in final
    assert "reports/eac_vla/prototype_protocol.md" in final
    assert "EAC Stage 0 completed without training" in final
    assert "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH" in final
    assert "first-two dispersion p95 `0.0007983036317792467`" in final
    assert "commitment counts `2:136`, `8:132`, `50:132`" in final
    assert "passthrough max error `5.07000000038449e-07`" in final
    assert "EAC runtime queue check completed without training" in final
    assert "queue length `0 -> 49`" in final
    assert "EAC bounded validation search completed with exactly six configurations" in final
    assert "eac_q33_aggressive_1_4_50" in final
    assert "validation score `0.7530415186081504`" in final
    assert "EAC Stage A matched manifest is frozen in `reports/eac_vla/stage_a_manifest.json`" in final
    assert "63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C" in final
    assert "EAC Stage A policy preflight passed in `reports/eac_vla/stage_a_preflight.json`" in final
    assert "all policy prefixes preserved action values exactly" in final
    assert "EAC Stage A runner validation passed in `reports/eac_vla/stage_a_runner_validation.json`" in final
    assert "EAC Stage A completed `50 / 50` episodes with zero exceptions" in final
    assert "EAC Stage B matched manifest is frozen in `reports/eac_vla/stage_b_manifest.json`" in final
    assert "31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D" in final
    assert "EAC Stage B completed from the detached run `runs/eac_vla_stage_b/20260714T202334Z`" in final
    assert "EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD" in final
    assert "EAC full reached `29 / 40`" in final
    assert "AAC entropy proxy reached `30 / 40`" in final
    assert "fixed short-replan reached `29 / 40`" in final
    assert "Epoch 4 Cycle 11 generated exactly three post-EAC candidates" in final
    assert "G3P-VLA" in final
    assert "Grounded 3D Point Injection" in final
    assert G3P_PROPOSAL_HASH in final
    assert "reports/g3p_vla/reviewer_attack.md" in final
    assert "reports/g3p_vla/researcher_rebuttal.md" in final
    assert "reports/g3p_vla/mathematical_mechanism_audit.md" in final
    assert "reports/g3p_vla/preregistration.md" in final
    assert "reports/g3p_vla/prototype_protocol.md" in final
    assert "reports/g3p_vla/development_audit.json" in final
    assert "Final G3P Stage 0 decision: `DATA_OR_SUPERVISION_FAILURE`" in final
    assert "validation material fraction `1.0`" in final
    assert "CALA-VLA" in final
    assert "reports/epoch_4_cycle_12_candidate_generation.md" in final
    assert "reports/cala_vla/researcher_proposal.md" in final
    assert "reports/cala_vla/reviewer_attack.md" in final
    assert "reports/cala_vla/researcher_rebuttal.md" in final
    assert "CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "reports/cala_vla/mathematical_mechanism_audit.md" in final
    assert "CALA_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "reports/cala_vla/preregistration.md" in final
    assert "reports/cala_vla/prototype_protocol.md" in final
    assert "frozen-SmolVLA identity-preserving CAC-style latent-action adaptation" in final
    assert CALA_PROPOSAL_HASH in final
    assert "task-mean latent-action baseline" in final
    assert "CALA Stage 0 is complete" in final
    assert "final decision `DESIGN_FAILURE`" in final
    assert "action_history_only" in final
    assert "-0.01171824382857035" in final
    assert "Validation search, training, Stage A manifest freeze, and rollout are disallowed" in final
    assert "Epoch 4 Cycle 13 generated exactly three post-CALA candidates" in final
    assert "RAR-VLA" in final
    assert "reports/epoch_4_cycle_13_candidate_generation.md" in final
    assert "reports/epoch_4_cycle_13_prior_mechanism_map.md" in final
    assert "AR-VLA" in final
    assert "ema_action_history_baseline" in final
    assert "reports/rar_vla/researcher_proposal.md" in final
    assert RAR_PROPOSAL_HASH in final
    assert "reports/rar_vla/reviewer_attack.md" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "REMAC/TAS distinctions" in final
    assert "reports/rar_vla/researcher_rebuttal.md" in final
    assert "RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "reports/rar_vla/mathematical_mechanism_audit.md" in final
    assert "RAR_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "no deterministic-action KL" in final
    assert "Current stage: `epoch_4_cycle_13_rar_mathematical_audit_preregistered`" in final
    assert "runs/marc_vla_stage_a/20260714T171356Z" in final


def test_active_campaign_state_records_governance_v2() -> None:
    state = json.loads((REPORTS / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["governance_file"] == "reports/current_research_governance.md"
    assert state["current_decision"] == "RAR_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 13
    assert state["current_stage"] == "epoch_4_cycle_13_rar_mathematical_audit_preregistered"
    assert state["method"] == "RAR-VLA"
    assert state["method_identity"] == "RAR-VLA"
    assert state["proposal_hash"] == RAR_PROPOSAL_HASH
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["epoch_2_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_1_outcome"]["final_decision"] == "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    assert state["epoch_3_cycle_2_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_3_cycle_3_outcome"]["final_decision"] == "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_4_cycle_1_outcome"]["final_decision"] == "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    assert state["epoch_4_cycle_1_outcome"]["rcv_full_successes"] == 20
    assert state["epoch_4_cycle_1_outcome"]["rcv_no_context_ablation_successes"] == 24
    assert state["epoch_4_cycle_2_outcome"]["final_decision"] == "STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION"
    assert state["epoch_4_cycle_2_outcome"]["cavm_full_successes"] == 24
    assert state["epoch_4_cycle_2_outcome"]["nearest_success_replay_successes"] == 23
    assert state["next_action"].startswith("Freeze RAR-VLA preregistration and prototype protocol")
    assert state["task_reset_manifest"] is None
    assert state["stage_b_manifest_json"] is None
    assert state["stage_b_partial_checkpoint"] is None
    assert state["epoch_4_cycle_6_mtf_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["completed_episode_count"] == 50
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_6_mtf_stage_a_outcome"]["policy_successes"]["mtf_full"]["successes"] == 7
    assert state["epoch_4_cycle_6_mtf_stage_b_manifest"]["planned_episode_count"] == 200
    assert state["epoch_4_cycle_6_mtf_stage_b_manifest"]["paired_cases_per_policy"] == 40
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["final_decision"] == "MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["completed_episode_count"] == 200
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["mtf_full_successes"] == 26
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["mtf_no_retention_ablation_successes"] == 32
    assert state["epoch_4_cycle_6_mtf_stage_b_outcome"]["paired_delta_vs_no_retention_ablation"] == -0.15
    assert "post_pse_research_design_governance_applied" in state["completed_stages"]
    assert "epoch_4_cycle_1_rcv_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "post_cavm_performance_governance_applied" in state["completed_stages"]
    assert "epoch_4_cycle_3_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_a_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_stage_b_completed" in state["completed_stages"]
    assert "epoch_4_cycle_3_fang_valid_current_formulation_kill_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_4_evostate_design_failure_recorded" in state["completed_stages"]
    assert state["epoch_4_cycle_4_pre_stage_0"]["selection_decision"] == "SELECT_EVOSTATE_VLA"
    assert state["epoch_4_cycle_4_outcome"]["final_decision"] == "AUDIT_STOP_DESIGN_FAILURE"
    assert state["epoch_4_cycle_5_pre_stage_a"]["selection_decision"] == "SELECT_RAC_VLA"
    assert state["epoch_4_cycle_5_pre_stage_a"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG"
    assert state["epoch_4_cycle_5_pre_stage_a"]["selected_config"] == "rac_h4_a0.05"
    assert state["epoch_4_cycle_5_pre_stage_a"]["full_vs_best_baseline_accuracy_margin"] == 0.21126158232359227
    assert state["epoch_4_cycle_5_stage_a_outcome"]["final_decision"] == "STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_5_stage_a_outcome"]["episode_count"] == 50
    assert state["epoch_4_cycle_5_stage_a_outcome"]["rac_full_successes"] == 0
    assert state["epoch_4_cycle_5_stage_a_outcome"]["reflective_history_proxy_successes"] == 1
    assert state["epoch_4_cycle_5_stage_a_outcome"]["online_diagonal_inverse_gain_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["final_decision"] == "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    assert state["epoch_4_cycle_5_stage_b_outcome"]["episode_count"] == 200
    assert state["epoch_4_cycle_5_stage_b_outcome"]["manifest_duplicate_keys"] == 0
    assert state["epoch_4_cycle_5_stage_b_outcome"]["manifest_bad_pairs"] == 0
    assert state["epoch_4_cycle_5_stage_b_outcome"]["rac_full_successes"] == 1
    assert state["epoch_4_cycle_5_stage_b_outcome"]["rac_no_consequence_ablation_successes"] == 2
    assert state["epoch_4_cycle_5_stage_b_outcome"]["online_diagonal_inverse_gain_successes"] == 2
    assert "post_rac_governance_update_installed" in state["completed_stages"]
    assert "epoch_4_cycle_6_candidate_search_pending" in state["completed_stages"]
    assert "epoch_4_cycle_6_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_validation_search_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_selected_config_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_adapter_training_runner_validated" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_adapter_training_completed" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_checkpoints_verified" in state["completed_stages"]
    assert "epoch_4_cycle_6_mtf_stage_b_completed" in state["completed_stages"]
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
    assert state["epoch_4_cycle_9_pre_stage_0"]["method"] == "PESA-VLA"
    assert state["epoch_4_cycle_9_pre_stage_0"]["selection_decision"] == "SELECT_PESA_VLA"
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
    assert state["epoch_4_cycle_9_pre_stage_0"]["training_happened"] is False
    assert state["epoch_4_cycle_9_pre_stage_0"]["closed_loop_experiment_happened"] is False
    outcome = state["epoch_4_cycle_9_pesa_development_outcome"]
    assert outcome["final_decision"] == "DESIGN_FAILURE"
    assert outcome["query_probe_accuracy"] == 0.5225
    assert outcome["query_probe_majority_accuracy"] == 0.6
    assert outcome["query_probe_accuracy_margin"] == -0.07750000000000001
    assert outcome["training_happened"] is False
    assert outcome["closed_loop_experiment_happened"] is False
    eac = state["epoch_4_cycle_10_pre_proposal"]
    assert eac["selection_decision"] == "SELECT_EAC_VLA"
    assert eac["candidate_count"] == 3
    assert eac["closest_prior"] == "Adaptive Action Chunking"
    assert eac["selected_score"] == 93
    assert eac["first_comparison_policies"] == [
        "frozen_smolvla_fixed_queue",
        "aac_entropy_proxy",
        "eac_full",
        "eac_no_calibration_no_hysteresis_ablation",
        "fixed_short_replan_baseline",
    ]
    assert eac["closed_loop_experiment_happened"] is False
    assert eac["confirmatory_test_tuning_happened"] is False
    assert eac["proposal"] == "reports/eac_vla/researcher_proposal.md"
    assert eac["proposal_hash"] == EAC_PROPOSAL_HASH
    assert eac["reviewer_attack"] == "reports/eac_vla/reviewer_attack.md"
    assert eac["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert eac["researcher_rebuttal"] == "reports/eac_vla/researcher_rebuttal.md"
    assert eac["rebuttal_decision"] == "EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert eac["mathematical_audit"] == "reports/eac_vla/mathematical_mechanism_audit.md"
    assert eac["mathematical_audit_decision"] == "EAC_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert eac["preregistration"] == "reports/eac_vla/preregistration.md"
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
    assert eac["stage_0_passthrough_max_abs_error"] == 5.07000000038449e-07
    assert eac["stage_0_runtime_full_chunk_check_required_before_validation_search"] is True
    assert eac["runtime_queue_check_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert eac["runtime_queue_check_chunk_shape"] == [50, 7]
    assert eac["runtime_queue_check_select_action_vs_chunk0_max_abs_diff"] == 0.0
    assert eac["runtime_queue_check_queue_owner_present"] is True
    assert eac["runtime_queue_check_queue_len_before_select_action"] == 0
    assert eac["runtime_queue_check_queue_len_after_select_action"] == 49
    assert eac["runtime_queue_check_all_prefixes_value_preserving"] is True
    assert eac["validation_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert eac["validation_search_happened"] is True
    assert eac["validation_confirmatory_records_used_for_tuning"] is False
    assert eac["tried_config_count"] == 6
    assert eac["selected_config"] == "eac_q33_aggressive_1_4_50"
    assert eac["selected_validation_score"] == 0.7530415186081504
    assert eac["stage_a_manifest_decision"] == "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING"
    assert eac["stage_a_preflight_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert eac["stage_a_planned_episode_count"] == 50
    assert eac["stage_a_paired_cases_per_policy"] == 10
    assert eac["stage_a_reset_seeds"] == [20261211, 20261212]
    assert eac["stage_a_preflight_policy_count"] == 5
    assert eac["stage_a_preflight_checkpoint_policy_count"] == 0
    assert eac["stage_a_preflight_all_policy_prefixes_value_preserving"] is True
    assert eac["selected_commitment_counts"] == {"1": 132, "4": 136, "50": 132}
    eac_outcome = state["epoch_4_cycle_10_eac_development_outcome"]
    assert eac_outcome["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert eac_outcome["hard_stop_reasons"] == []
    assert eac_outcome["valid_current_formulation_kill"] is False
    queue_check = state["epoch_4_cycle_10_eac_runtime_queue_check"]
    assert queue_check["final_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert queue_check["chunk_shape"] == [50, 7]
    assert queue_check["raw_action_chunk_shape"] == [1, 50, 7]
    assert queue_check["select_action_vs_chunk0_max_abs_diff"] == 0.0
    assert queue_check["queue_len_after_select_action"] == 49
    assert queue_check["all_prefixes_value_preserving"] is True
    validation = state["epoch_4_cycle_10_eac_validation_search"]
    assert validation["final_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert validation["tried_config_count"] == 6
    assert validation["selected_config"] == "eac_q33_aggressive_1_4_50"
    assert validation["selected_validation_score"] == 0.7530415186081504
    eac_preflight = state["epoch_4_cycle_10_eac_stage_a_preflight"]
    assert eac_preflight["final_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert eac_preflight["planned_episode_count"] == 50
    assert eac_preflight["paired_cases_per_policy"] == 10
    assert eac_preflight["policy_count"] == 5
    assert eac_preflight["checkpoint_policy_count"] == 0
    assert eac_preflight["policy_output_shape"] == [50, 7]
    assert eac_preflight["all_policy_prefixes_value_preserving"] is True
    assert eac_preflight["errors"] == []
    eac_stage_b = state["epoch_4_cycle_10_eac_stage_b_manifest"]
    assert eac_stage_b["final_decision"] == "EAC_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert eac_stage_b["planned_episode_count"] == 200
    assert eac_stage_b["paired_cases_per_policy"] == 40
    assert eac_stage_b["reset_seeds"] == [20261213, 20261214]
    assert eac_stage_b["manifest_canonical_payload_sha256"] == "31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D"
    assert eac_stage_b["partition_separation"]["stage_b_outcomes_used_for_retuning"] is False
    eac_launch = state["epoch_4_cycle_10_eac_stage_b_launch"]
    assert eac_launch["final_decision"] == "EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert eac_launch["run_dir"] == "runs/eac_vla_stage_b/20260714T202334Z"
    assert eac_launch["child_pid"] == 386
    assert eac_launch["exit_code"] == 0
    assert eac_launch["planned_episode_count"] == 200
    assert eac_launch["completed_episode_count"] == 200
    assert eac_launch["partial_result"] == "reports/eac_vla/stage_b_partial_result.json"
    eac_stage_b_outcome = state["epoch_4_cycle_10_eac_stage_b_outcome"]
    assert eac_stage_b_outcome["final_decision"] == "EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert eac_stage_b_outcome["valid_current_formulation_kill"] is True
    assert eac_stage_b_outcome["completed_episode_count"] == 200
    assert eac_stage_b_outcome["exception_count"] == 0
    assert eac_stage_b_outcome["frozen_smolvla_fixed_queue_successes"] == 30
    assert eac_stage_b_outcome["aac_entropy_proxy_successes"] == 30
    assert eac_stage_b_outcome["eac_full_successes"] == 29
    assert eac_stage_b_outcome["eac_no_calibration_no_hysteresis_ablation_successes"] == 30
    assert eac_stage_b_outcome["fixed_short_replan_baseline_successes"] == 29
    assert eac_stage_b_outcome["paired_delta_vs_fixed_short_replan_baseline"] == 0.0
    assert eac_stage_b_outcome["simple_baseline_explains_method"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["selection_decision"] == "SELECT_DAGR_VLA"
    assert state["epoch_4_cycle_7_pre_stage_0"]["proposal_hash"] == "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89"
    assert state["epoch_4_cycle_7_pre_stage_0"]["reviewer_attack"] == "reports/dagr_vla/reviewer_attack.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["epoch_4_cycle_7_pre_stage_0"]["researcher_rebuttal"] == "reports/dagr_vla/researcher_rebuttal.md"
    assert state["epoch_4_cycle_7_pre_stage_0"]["rebuttal_decision"] == "DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert state["epoch_4_cycle_7_pre_stage_0"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_7_pre_stage_0"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_config"] == "dagr_a020_route_mlp"
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_residual_alpha"] == 0.2
    assert state["epoch_4_cycle_7_pre_stage_0"]["selected_route_architecture"] == "mlp"
    assert state["epoch_4_cycle_7_pre_stage_0"]["policy_identity_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_manifest_decision"] == "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_pre_stage_0"]["stage_a_manifest_sha256"] == "8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B"
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
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["scoreable_development_records"] == 1600
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["tried_config_count"] == 6
    assert state["epoch_4_cycle_7_dagr_development_outcome"]["selected_config"] == "dagr_a020_route_mlp"
    assert state["epoch_4_cycle_7_dagr_policy_identity_outcome"]["final_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_7_dagr_policy_identity_outcome"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_manifest"]["planned_episode_count"] == 50
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["final_decision"] == "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["policy_count"] == 5
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["checkpoint_policy_count"] == 4
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["cuda_ok"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_preflight"]["no_accidental_checkpoint_reuse"] is True
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["final_decision"] == "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["completed_episode_count"] == 50
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["exception_count"] == 0
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["frozen_smolvla_successes"] == 8
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["gripper_transition_heuristic_successes"] == 7
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["dagr_full_successes"] == 6
    assert state["epoch_4_cycle_7_dagr_stage_a_outcome"]["stage_b_required"] is True
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["final_decision"] == "DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["planned_episode_count"] == 200
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["paired_cases_per_policy"] == 40
    assert state["epoch_4_cycle_7_dagr_stage_b_manifest"]["reset_seeds"] == [20261207, 20261208]
    outcome = state["epoch_4_cycle_7_dagr_stage_b_outcome"]
    assert outcome["final_decision"] == "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert outcome["valid_current_formulation_kill"] is True
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
    assert state["epoch_4_cycle_8_pre_stage_a"]["secondary_priors"] == ["ReactVLA", "SnapFlow"]
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
    assert state["epoch_4_cycle_8_pre_stage_a"]["selected_validation_score"] == 0.5457964262366295
    assert state["epoch_4_cycle_8_pre_stage_a"]["selected_full_vs_static_mean_l2"] == 0.0019475044682621956
    assert state["epoch_4_cycle_8_pre_stage_a"]["static_mixture_remains_live_reviewer_killer"] is True
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
    assert state["epoch_4_cycle_8_pre_stage_a"]["first_comparison_policies"] == [
        "frozen_smolvla",
        "openvla_oft_l1_proxy",
        "marc_full",
        "marc_no_disagreement_gate_ablation",
        "static_l1_mixture_baseline",
    ]
    assert state["epoch_4_cycle_8_marc_development_outcome"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_8_marc_development_outcome"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
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
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_gate_predicted_positive_fraction"] == 0.3325
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_action_validity"] == 1.0
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_full_vs_l1_proxy_mean_l2"] == 0.007010325323790312
    assert state["epoch_4_cycle_8_marc_development_outcome"]["selected_full_vs_static_mean_l2"] == 0.0019475044682621956
    assert state["epoch_4_cycle_8_marc_development_outcome"]["linear_configs_stopped_for_collapsed_gate"] is True
    assert state["epoch_4_cycle_8_marc_development_outcome"]["policy_identity_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert state["epoch_4_cycle_8_marc_development_outcome"]["stage_a_allowed"] is True
    policy_outcome = state["epoch_4_cycle_8_marc_policy_identity_outcome"]
    assert policy_outcome["final_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert policy_outcome["stage_a_allowed"] is True
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
    assert marc_launch["partial_result"] == "reports/marc_vla/stage_a_partial_result.json"
    marc_outcome = state["epoch_4_cycle_8_marc_stage_a_outcome"]
    assert marc_outcome["final_decision"] == "MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
    assert marc_outcome["valid_current_formulation_kill"] is True
    assert marc_outcome["completed_episode_count"] == 50
    assert marc_outcome["exception_count"] == 0
    assert marc_outcome["frozen_smolvla_successes"] == 8
    assert marc_outcome["marc_full_successes"] == 0
    assert marc_outcome["marc_no_disagreement_gate_ablation_successes"] == 7
    assert marc_outcome["static_l1_mixture_baseline_successes"] == 7
    assert state["epoch_4_cycle_6_pre_stage_0"]["selection_decision"] == "SELECT_MTF_VLA"
    assert state["epoch_4_cycle_6_pre_stage_0"]["closest_prior"] == "FrameSkip"
    assert state["epoch_4_cycle_6_pre_stage_0"]["secondary_prior"] == "StructVLA"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["development_final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["validation_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["selected_config"] == "mtf_r20_ret100"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["stage_a_allowed"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_runner_validated"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_happened"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["adapter_training_final_decision"] == "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY"
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["frameskip_proxy_distinct_from_no_retention"] is True
    assert state["epoch_4_cycle_6_mtf_development_outcome"]["frameskip_proxy_train_events"] == 240
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["final_decision"] == "MTF_ADAPTER_TRAINING_PLAN_READY"
    assert state["epoch_4_cycle_6_mtf_adapter_training_plan"]["jobs"][0]["event_count"] == 567
    assert state["epoch_4_cycle_6_mtf_adapter_checkpoint_outcome"]["checkpoint_manifest"] == "reports/mtf_vla/adapter_checkpoint_manifest.json"
    assert state["epoch_4_cycle_3_outcome"]["final_decision"] == "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    assert state["epoch_4_cycle_3_outcome"]["fang_full_successes"] == 11
    assert state["epoch_4_cycle_3_outcome"]["base_smolvla_successes"] == 16


def test_core_ledgers_reference_current_governance() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    manual = (REPORTS / "codex_delegation_manual.md").read_text(encoding="utf-8")
    governance = (REPORTS / "current_research_governance.md").read_text(encoding="utf-8")

    assert "reports/current_research_governance.md" in agents
    assert "Multi-stage autonomous research is permitted" in manual
    assert "There is no finite global method-cycle limit." in governance
    assert "Post-CAVM Performance-Oriented Research Design Governance" in governance
    assert "Post-RAC Honest Positive-Result Governance" in governance
    assert "MAXIMIZE_THE_PROBABILITY_OF_AN_HONEST_PAPER_WORTHY_POSITIVE_RESULT" in governance
