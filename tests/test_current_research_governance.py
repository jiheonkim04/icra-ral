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
AMP_PROPOSAL_HASH = "67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4"
CFR_PROPOSAL_HASH = "9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE"
TSC_PROPOSAL_HASH = "0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941"
CCIF_PROPOSAL_HASH = "2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1"
URF_PROPOSAL_HASH = "E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532"
S2C_PROPOSAL_HASH = "399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3"
LCG_PROPOSAL_HASH = "F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E"
AFID_PROPOSAL_HASH = "B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC"


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


def test_active_state_records_amp_selection_and_rap_stage_0_failure() -> None:
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 33
    assert state["current_branch"] == "codex/autonomous-until-paper-governance-v2"
    assert state["maximum_method_cycles"] is None
    assert state["global_no_method_terminal_allowed"] is False
    assert state["current_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert state["current_stage"] == "epoch_4_cycle_33_afid_prototype_protocol_pending"
    assert state["method"] == "AFID-VLA"
    assert state["method_identity"] == "AFID-VLA"
    assert state["proposal_hash"] == AFID_PROPOSAL_HASH
    assert state["proposal_hash_file"] == "reports/afid_vla/proposal_hash.txt"
    assert state["researcher_proposal"] == "reports/afid_vla/researcher_proposal.md"
    assert state["reviewer_attack"] == "reports/afid_vla/reviewer_attack.md"
    assert state["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert state["rebuttal_decision"] == "AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert state["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert state["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert state["preregistration"] == "reports/afid_vla/preregistration.md"
    assert state["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
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
        "epoch_4_cycle_25_rap_reviewer_attack_completed",
        "epoch_4_cycle_25_rap_rebuttal_pending",
        "epoch_4_cycle_25_rap_rebuttal_completed",
        "epoch_4_cycle_25_rap_mathematical_audit_pending",
        "epoch_4_cycle_25_rap_mathematical_audit_preregistered",
        "epoch_4_cycle_25_rap_preregistration_pending",
        "epoch_4_cycle_25_rap_preregistration_frozen",
        "epoch_4_cycle_25_rap_prototype_protocol_pending",
        "epoch_4_cycle_25_rap_prototype_protocol_frozen",
        "epoch_4_cycle_25_rap_stage_0_pending",
        "epoch_4_cycle_25_rap_stage_0_launched",
        "epoch_4_cycle_25_rap_stage_0_completed",
        "epoch_4_cycle_25_rap_stage_0_adjudicated",
        "epoch_4_cycle_25_rap_implementation_or_optimization_failure_recorded",
        "epoch_4_cycle_26_candidate_search_pending",
        "epoch_4_cycle_26_prior_mechanism_map_completed",
        "epoch_4_cycle_26_candidate_generation_completed",
        "epoch_4_cycle_26_amp_candidate_selected",
        "epoch_4_cycle_26_amp_researcher_proposal_pending",
        "epoch_4_cycle_26_amp_researcher_proposal_frozen",
        "epoch_4_cycle_26_amp_reviewer_attack_pending",
        "epoch_4_cycle_26_amp_reviewer_attack_completed",
        "epoch_4_cycle_26_amp_rebuttal_pending",
        "epoch_4_cycle_26_amp_rebuttal_completed",
        "epoch_4_cycle_26_amp_mathematical_audit_pending",
        "epoch_4_cycle_26_amp_mathematical_audit_preregistered",
        "epoch_4_cycle_26_amp_preregistration_pending",
        "epoch_4_cycle_26_amp_preregistration_frozen",
        "epoch_4_cycle_26_amp_prototype_protocol_pending",
        "epoch_4_cycle_26_amp_prototype_protocol_frozen",
        "epoch_4_cycle_26_amp_stage_0_pending",
        "epoch_4_cycle_26_amp_stage_0_runner_implemented",
        "epoch_4_cycle_26_amp_stage_0_completed",
        "epoch_4_cycle_26_amp_implementation_failure_recorded",
        "epoch_4_cycle_27_candidate_search_pending",
        "epoch_4_cycle_27_prior_mechanism_map_completed",
        "epoch_4_cycle_27_candidate_generation_completed",
        "epoch_4_cycle_27_cfr_candidate_selected",
        "epoch_4_cycle_27_cfr_researcher_proposal_pending",
        "epoch_4_cycle_27_cfr_researcher_proposal_frozen",
        "epoch_4_cycle_27_cfr_reviewer_attack_pending",
        "epoch_4_cycle_27_cfr_reviewer_attack_completed",
        "epoch_4_cycle_27_cfr_rebuttal_pending",
        "epoch_4_cycle_27_cfr_rebuttal_completed",
        "epoch_4_cycle_27_cfr_mathematical_audit_pending",
        "epoch_4_cycle_27_cfr_mathematical_audit_preregistered",
        "epoch_4_cycle_27_cfr_preregistration_pending",
        "epoch_4_cycle_27_cfr_preregistration_frozen",
        "epoch_4_cycle_27_cfr_prototype_protocol_pending",
        "epoch_4_cycle_27_cfr_prototype_protocol_frozen",
        "epoch_4_cycle_27_cfr_stage_0_implementation_pending",
        "epoch_4_cycle_27_cfr_stage_0_runner_implemented",
        "epoch_4_cycle_27_cfr_stage_0_pending",
        "epoch_4_cycle_27_cfr_stage_0_launched",
        "epoch_4_cycle_27_cfr_stage_0_completed",
        "epoch_4_cycle_27_cfr_stage_0_adjudicated",
        "epoch_4_cycle_27_cfr_no_headroom_recorded",
        "epoch_4_cycle_28_candidate_search_pending",
        "epoch_4_cycle_28_prior_mechanism_map_completed",
        "epoch_4_cycle_28_candidate_generation_completed",
        "epoch_4_cycle_28_tsc_candidate_selected",
        "epoch_4_cycle_28_tsc_researcher_proposal_pending",
        "epoch_4_cycle_28_tsc_researcher_proposal_frozen",
        "epoch_4_cycle_28_tsc_reviewer_attack_pending",
        "epoch_4_cycle_28_tsc_reviewer_attack_completed",
        "epoch_4_cycle_28_tsc_rebuttal_pending",
        "epoch_4_cycle_28_tsc_rebuttal_completed",
        "epoch_4_cycle_28_tsc_mathematical_audit_pending",
        "epoch_4_cycle_28_tsc_mathematical_audit_preregistered",
        "epoch_4_cycle_28_tsc_preregistration_pending",
        "epoch_4_cycle_28_tsc_preregistration_frozen",
        "epoch_4_cycle_28_tsc_prototype_protocol_pending",
        "epoch_4_cycle_28_tsc_prototype_protocol_frozen",
        "epoch_4_cycle_28_tsc_stage_0_implementation_pending",
        "epoch_4_cycle_28_tsc_stage_0_runner_implemented",
        "epoch_4_cycle_28_tsc_stage_0_pending",
        "epoch_4_cycle_28_tsc_stage_0_launched",
        "epoch_4_cycle_28_tsc_stage_0_completed",
        "epoch_4_cycle_28_tsc_stage_0_adjudicated",
        "epoch_4_cycle_28_tsc_no_headroom_recorded",
        "epoch_4_cycle_29_candidate_search_pending",
        "epoch_4_cycle_29_prior_mechanism_map_completed",
        "epoch_4_cycle_29_candidate_generation_completed",
        "epoch_4_cycle_29_ccif_candidate_selected",
        "epoch_4_cycle_29_ccif_researcher_proposal_pending",
        "epoch_4_cycle_29_ccif_researcher_proposal_frozen",
        "epoch_4_cycle_29_ccif_reviewer_attack_pending",
        "epoch_4_cycle_29_ccif_reviewer_attack_completed",
        "epoch_4_cycle_29_ccif_rebuttal_pending",
        "epoch_4_cycle_29_ccif_rebuttal_completed",
        "epoch_4_cycle_29_ccif_mathematical_audit_pending",
        "epoch_4_cycle_29_ccif_mathematical_audit_preregistered",
        "epoch_4_cycle_29_ccif_preregistration_pending",
        "epoch_4_cycle_29_ccif_preregistration_frozen",
        "epoch_4_cycle_29_ccif_prototype_protocol_pending",
        "epoch_4_cycle_29_ccif_prototype_protocol_frozen",
        "epoch_4_cycle_29_ccif_stage_0_implementation_pending",
        "epoch_4_cycle_29_ccif_stage_0_implementation_validated",
        "epoch_4_cycle_29_ccif_stage_0_launch_pending",
        "epoch_4_cycle_29_ccif_stage_0_completed",
        "epoch_4_cycle_29_ccif_stage_0_adjudicated",
        "epoch_4_cycle_29_ccif_design_failure_recorded",
        "epoch_4_cycle_30_candidate_search_pending",
        "epoch_4_cycle_30_prior_mechanism_map_completed",
        "epoch_4_cycle_30_candidate_generation_completed",
        "epoch_4_cycle_30_urf_candidate_selected",
        "epoch_4_cycle_30_urf_researcher_proposal_pending",
        "epoch_4_cycle_30_urf_researcher_proposal_frozen",
        "epoch_4_cycle_30_urf_reviewer_attack_pending",
        "epoch_4_cycle_30_urf_reviewer_attack_completed",
        "epoch_4_cycle_30_urf_rebuttal_pending",
        "epoch_4_cycle_30_urf_rebuttal_completed",
        "epoch_4_cycle_30_urf_mathematical_audit_pending",
        "epoch_4_cycle_30_urf_mathematical_audit_preregistered",
        "epoch_4_cycle_30_urf_preregistration_pending",
        "epoch_4_cycle_30_urf_preregistration_frozen",
        "epoch_4_cycle_30_urf_prototype_protocol_pending",
        "epoch_4_cycle_30_urf_prototype_protocol_frozen",
        "epoch_4_cycle_30_urf_stage_0_implementation_pending",
        "epoch_4_cycle_30_urf_stage_0_implementation_validated",
        "epoch_4_cycle_30_urf_stage_0_launch_pending",
        "epoch_4_cycle_30_urf_stage_0_completed",
        "epoch_4_cycle_30_urf_stage_0_adjudicated",
        "epoch_4_cycle_30_urf_no_headroom_recorded",
        "epoch_4_cycle_31_candidate_search_pending",
        "epoch_4_cycle_31_prior_mechanism_map_completed",
        "epoch_4_cycle_31_candidate_generation_completed",
        "epoch_4_cycle_31_s2c_candidate_selected",
        "epoch_4_cycle_31_s2c_researcher_proposal_pending",
        "epoch_4_cycle_31_s2c_researcher_proposal_frozen",
        "epoch_4_cycle_31_s2c_reviewer_attack_pending",
        "epoch_4_cycle_31_s2c_reviewer_attack_completed",
        "epoch_4_cycle_31_s2c_rebuttal_pending",
        "epoch_4_cycle_31_s2c_rebuttal_completed",
        "epoch_4_cycle_31_s2c_mathematical_audit_pending",
        "epoch_4_cycle_31_s2c_mathematical_audit_preregistered",
        "epoch_4_cycle_31_s2c_preregistration_pending",
        "epoch_4_cycle_31_s2c_preregistration_frozen",
        "epoch_4_cycle_31_s2c_prototype_protocol_pending",
        "epoch_4_cycle_31_s2c_prototype_protocol_frozen",
        "epoch_4_cycle_31_s2c_stage_0_implementation_pending",
        "epoch_4_cycle_31_s2c_stage_0_implementation_validated",
        "epoch_4_cycle_31_s2c_stage_0_launch_pending",
        "epoch_4_cycle_31_s2c_stage_0_completed",
        "epoch_4_cycle_31_s2c_stage_0_adjudicated",
        "epoch_4_cycle_31_s2c_data_or_supervision_failure_recorded",
        "epoch_4_cycle_32_candidate_search_pending",
        "epoch_4_cycle_32_prior_mechanism_map_completed",
        "epoch_4_cycle_32_candidate_generation_completed",
        "epoch_4_cycle_32_lcg_candidate_selected",
        "epoch_4_cycle_32_lcg_researcher_proposal_pending",
        "epoch_4_cycle_32_lcg_researcher_proposal_frozen",
        "epoch_4_cycle_32_lcg_reviewer_attack_pending",
        "epoch_4_cycle_32_lcg_reviewer_attack_completed",
        "epoch_4_cycle_32_lcg_rebuttal_pending",
        "epoch_4_cycle_32_lcg_rebuttal_completed",
        "epoch_4_cycle_32_lcg_mathematical_audit_pending",
        "epoch_4_cycle_32_lcg_mathematical_audit_preregistered",
        "epoch_4_cycle_32_lcg_preregistration_pending",
        "epoch_4_cycle_32_lcg_preregistration_frozen",
        "epoch_4_cycle_32_lcg_prototype_protocol_pending",
        "epoch_4_cycle_32_lcg_prototype_protocol_frozen",
        "epoch_4_cycle_32_lcg_stage_0_implementation_pending",
        "epoch_4_cycle_32_lcg_stage_0_implementation_validated",
        "epoch_4_cycle_32_lcg_stage_0_ready",
        "epoch_4_cycle_32_lcg_stage_0_completed",
        "epoch_4_cycle_32_lcg_stage_0_adjudicated",
        "epoch_4_cycle_32_lcg_design_failure_recorded",
        "epoch_4_cycle_33_candidate_search_pending",
        "epoch_4_cycle_33_prior_mechanism_map_completed",
        "epoch_4_cycle_33_candidate_generation_completed",
        "epoch_4_cycle_33_afid_candidate_selected",
        "epoch_4_cycle_33_afid_researcher_proposal_pending",
        "epoch_4_cycle_33_afid_researcher_proposal_frozen",
        "epoch_4_cycle_33_afid_reviewer_attack_pending",
        "epoch_4_cycle_33_afid_reviewer_attack_completed",
        "epoch_4_cycle_33_afid_rebuttal_pending",
        "epoch_4_cycle_33_afid_rebuttal_completed",
        "epoch_4_cycle_33_afid_mathematical_audit_pending",
        "epoch_4_cycle_33_afid_mathematical_audit_preregistered",
        "epoch_4_cycle_33_afid_preregistration_pending",
        "epoch_4_cycle_33_afid_preregistration_frozen",
        "epoch_4_cycle_33_afid_prototype_protocol_pending",
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
    rap_review = state["epoch_4_cycle_25_rap_reviewer_attack"]
    assert rap_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert rap_review["reviewer_attack"] == "reports/rap_vla/reviewer_attack.md"
    assert rap_review["proposal_hash"] == RAP_PROPOSAL_HASH
    rap_rebuttal = state["epoch_4_cycle_25_rap_rebuttal"]
    assert rap_rebuttal["final_decision"] == "RAP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert rap_rebuttal["researcher_rebuttal"] == "reports/rap_vla/researcher_rebuttal.md"
    assert rap_rebuttal["accepted_reviewer_conditions"] is True
    rap_math = state["epoch_4_cycle_25_rap_mathematical_audit"]
    assert rap_math["final_decision"] == "RAP_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert rap_math["mathematical_audit"] == "reports/rap_vla/mathematical_mechanism_audit.md"
    assert rap_math["kl_between_deterministic_actions_used"] is False
    rap_preregistration = state["epoch_4_cycle_25_rap_preregistration"]
    assert rap_preregistration["final_decision"] == "RAP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert rap_preregistration["preregistration"] == "reports/rap_vla/preregistration.md"
    assert rap_preregistration["stage_0_allowed_next"] is True
    rap_protocol = state["epoch_4_cycle_25_rap_prototype_protocol"]
    assert rap_protocol["final_decision"] == "RAP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert rap_protocol["prototype_protocol"] == "reports/rap_vla/prototype_protocol.md"
    assert rap_protocol["stage_0_allowed_next"] is True
    assert rap_protocol["runner"] == "scripts/run_rap_vla_stage0.py"
    assert rap_protocol["stage_0_completed"] is True
    assert rap_protocol["stage_0_decision"] == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    rap_outcome = state["epoch_4_cycle_25_rap_stage_0_outcome"]
    assert rap_outcome["final_decision"] == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert rap_outcome["completed_model_row_count"] == 640
    assert rap_outcome["planned_model_row_count"] == 640
    assert rap_outcome["exception_count"] == 0
    assert rap_outcome["duplicate_manifest_key_count"] == 0
    assert rap_outcome["duplicate_partial_key_count"] == 0
    assert rap_outcome["missing_manifest_key_count"] == 0
    assert rap_outcome["extra_partial_key_count"] == 0
    assert rap_outcome["split_overlap_key_count"] == 0
    assert rap_outcome["key_sets_equal"] is True
    assert rap_outcome["official_prior_policy_2_label"] == "optimusvla_memory_prior_proxy"
    assert rap_outcome["action_validity_ok"] is False
    assert rap_outcome["base_action_in_bounds"] is False
    assert rap_outcome["anchor_relative_improvement"] == 0.23865551292280293
    assert rap_outcome["residual_probe_relative_improvement"] == -3.830674623085068
    assert rap_outcome["bounded_validation_allowed"] is False
    assert rap_outcome["rap_rescue_allowed"] is False
    cycle26 = state["epoch_4_cycle_26_candidate_search"]
    assert cycle26["candidate_search_pending"] is False
    assert cycle26["candidate_count_required"] == 3
    assert cycle26["candidate_count_generated"] == 3
    assert cycle26["rap_repair_allowed"] is False
    assert cycle26["selected_method"] == "AMP-VLA"
    assert cycle26["selected_score"] == 95
    assert cycle26["selection_decision"] == "AMP_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    amp = state["epoch_4_cycle_26_candidate_selection"]
    assert amp["candidate_count"] == 3
    assert amp["selected_score"] == 95
    assert amp["method"] == "AMP-VLA"
    assert amp["closest_prior"] == "ABot-M0"
    assert amp["closest_prior_primary_source"] == "https://arxiv.org/abs/2602.11236"
    assert amp["closest_prior_official_repository"] == "https://github.com/amap-cvlab/ABot-Manipulation"
    assert amp["policy_order"] == [
        "smolvla_base",
        "abot_m0_action_manifold_proxy",
        "amp_full",
        "amp_no_manifold_projection",
        "standard_lora",
    ]
    assert amp["standard_lora_required"] is True
    assert amp["first_serious_comparison_includes_closest_prior"] is True
    assert amp["rap_rescue_allowed"] is False
    assert amp["proposal"] == "reports/amp_vla/researcher_proposal.md"
    assert amp["proposal_hash"] == AMP_PROPOSAL_HASH
    assert amp["proposal_hash_file"] == "reports/amp_vla/proposal_hash.txt"
    assert amp["proposal_decision"] == "AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert amp["reviewer_attack"] == "reports/amp_vla/reviewer_attack.md"
    assert amp["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert amp["researcher_rebuttal"] == "reports/amp_vla/researcher_rebuttal.md"
    assert amp["rebuttal_decision"] == "AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert amp["mathematical_audit"] == "reports/amp_vla/mathematical_mechanism_audit.md"
    assert amp["math_audit_decision"] == "AMP_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert amp["preregistration"] == "reports/amp_vla/preregistration.md"
    assert amp["preregistration_decision"] == "AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert amp["prototype_protocol"] == "reports/amp_vla/prototype_protocol.md"
    assert amp["prototype_protocol_decision"] == "AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    amp_proposal = state["epoch_4_cycle_26_amp_researcher_proposal"]
    assert amp_proposal["final_decision"] == "AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert amp_proposal["proposal_hash"] == AMP_PROPOSAL_HASH
    assert amp_proposal["closed_loop_experiment_happened"] is False
    assert amp_proposal["confirmatory_test_tuning_happened"] is False
    amp_review = state["epoch_4_cycle_26_amp_reviewer_attack"]
    assert amp_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert amp_review["reviewer_attack"] == "reports/amp_vla/reviewer_attack.md"
    assert amp_review["proposal_hash"] == AMP_PROPOSAL_HASH
    assert "no-projection ablation and matched standard LoRA remain live" in amp_review["conditions"]
    amp_rebuttal = state["epoch_4_cycle_26_amp_rebuttal"]
    assert amp_rebuttal["final_decision"] == "AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert amp_rebuttal["researcher_rebuttal"] == "reports/amp_vla/researcher_rebuttal.md"
    assert amp_rebuttal["accepted_reviewer_conditions"] is True
    assert amp_rebuttal["accepted_clipping_diagnostic"] is True
    amp_math = state["epoch_4_cycle_26_amp_mathematical_audit"]
    assert amp_math["final_decision"] == "AMP_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert amp_math["mathematical_audit"] == "reports/amp_vla/mathematical_mechanism_audit.md"
    assert amp_math["kl_between_deterministic_actions_used"] is False
    assert amp_math["projection_vs_clipping_diagnostic_required"] is True
    amp_prereg = state["epoch_4_cycle_26_amp_preregistration"]
    assert amp_prereg["final_decision"] == "AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert amp_prereg["preregistration"] == "reports/amp_vla/preregistration.md"
    assert amp_prereg["stage_0_allowed_next"] is True
    assert amp_prereg["bounded_validation_search_max_configs"] == 6
    amp_protocol = state["epoch_4_cycle_26_amp_prototype_protocol"]
    assert amp_protocol["final_decision"] == "AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert amp_protocol["prototype_protocol"] == "reports/amp_vla/prototype_protocol.md"
    assert amp_protocol["stage_0_allowed_next"] is True
    assert amp_protocol["runner"] == "scripts/run_amp_vla_stage0.py"
    assert amp_protocol["stage_0_completed"] is True
    assert amp_protocol["stage_0_decision"] == "AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert amp_protocol["bounded_validation_allowed"] is False
    amp_outcome = state["epoch_4_cycle_26_amp_stage_0_outcome"]
    assert amp_outcome["final_decision"] == "AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert amp_outcome["completed_model_row_count"] == 1280
    assert amp_outcome["planned_model_row_count"] == 1280
    assert amp_outcome["exception_count"] == 0
    assert amp_outcome["duplicate_manifest_key_count"] == 0
    assert amp_outcome["duplicate_partial_key_count"] == 0
    assert amp_outcome["missing_manifest_key_count"] == 0
    assert amp_outcome["extra_partial_key_count"] == 0
    assert amp_outcome["split_overlap_key_count"] == 0
    assert amp_outcome["key_sets_equal"] is True
    assert amp_outcome["official_prior_policy_2_label"] == "abot_m0_action_manifold_proxy"
    assert amp_outcome["official_abot_ready"] is False
    assert amp_outcome["action_validity_ok"] is False
    assert amp_outcome["base_action_in_bounds"] is False
    assert amp_outcome["coordinate_probe_relative_improvement"] == -4.947553385520279
    assert amp_outcome["abot_proxy_headroom_relative_improvement"] == -2.663165575108502
    assert amp_outcome["bounded_validation_allowed"] is False
    assert amp_outcome["amp_rescue_allowed"] is False
    cycle27 = state["epoch_4_cycle_27_candidate_search"]
    assert cycle27["candidate_search_pending"] is False
    assert cycle27["candidate_count_required"] == 3
    assert cycle27["candidate_count_generated"] == 3
    assert cycle27["amp_repair_allowed"] is False
    assert cycle27["selected_method"] == "CFR-VLA"
    assert cycle27["selected_score"] == 92
    assert cycle27["selection_decision"] == "CFR_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    cfr = state["epoch_4_cycle_27_candidate_selection"]
    assert cfr["candidate_count"] == 3
    assert cfr["selected_score"] == 92
    assert cfr["method"] == "CFR-VLA"
    assert cfr["closest_prior"] == "DFM-VLA"
    assert cfr["closest_prior_primary_source"] == "https://arxiv.org/html/2603.26320v1"
    assert cfr["closest_prior_project_page"] == "https://chris1220313648.github.io/DFM-VLA/"
    assert cfr["contribution_type"] == "PRIOR_EXTENSION"
    assert cfr["policy_order"] == [
        "smolvla_base",
        "dfm_vla_continuous_refinement_proxy_or_official_dfm_vla_if_installed",
        "cfr_full",
        "cfr_no_iterative_refinement",
        "standard_lora",
    ]
    assert cfr["standard_lora_required"] is True
    assert cfr["first_serious_comparison_includes_closest_prior"] is True
    assert cfr["training_happened"] is False
    assert cfr["validation_search_happened"] is False
    assert cfr["closed_loop_experiment_happened"] is False
    assert cfr["confirmatory_test_tuning_happened"] is False
    assert cfr["amp_rescue_allowed"] is False
    assert cfr["proposal"] == "reports/cfr_vla/researcher_proposal.md"
    assert cfr["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr["proposal_hash_file"] == "reports/cfr_vla/proposal_hash.txt"
    assert cfr["proposal_decision"] == "CFR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    cfr_proposal = state["epoch_4_cycle_27_cfr_researcher_proposal"]
    assert cfr_proposal["final_decision"] == "CFR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert cfr_proposal["proposal"] == "reports/cfr_vla/researcher_proposal.md"
    assert cfr_proposal["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr_proposal["closed_loop_experiment_happened"] is False
    assert cfr_proposal["confirmatory_test_tuning_happened"] is False
    cfr_review = state["epoch_4_cycle_27_cfr_reviewer_attack"]
    assert cfr_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert cfr_review["reviewer_attack"] == "reports/cfr_vla/reviewer_attack.md"
    assert cfr_review["proposal_hash"] == CFR_PROPOSAL_HASH
    assert "DFM-VLA proxy or official DFM-VLA remains policy 2" in cfr_review["conditions"]
    assert "standard_lora remains the single simple reviewer-killer baseline" in cfr_review["conditions"]
    assert cfr_review["closed_loop_experiment_happened"] is False
    assert cfr_review["confirmatory_test_tuning_happened"] is False
    cfr_rebuttal = state["epoch_4_cycle_27_cfr_rebuttal"]
    assert cfr_rebuttal["final_decision"] == "CFR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert cfr_rebuttal["researcher_rebuttal"] == "reports/cfr_vla/researcher_rebuttal.md"
    assert cfr_rebuttal["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr_rebuttal["accepted_reviewer_conditions"] is True
    assert cfr_rebuttal["accepted_key_ablation"] == "cfr_no_iterative_refinement"
    assert cfr_rebuttal["accepted_simple_baseline"] == "standard_lora"
    assert cfr_rebuttal["accepted_official_action_validity_semantics"] is True
    assert cfr_rebuttal["accepted_no_privileged_inference_inputs"] is True
    cfr_math = state["epoch_4_cycle_27_cfr_mathematical_audit"]
    assert cfr_math["final_decision"] == "CFR_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert cfr_math["mathematical_audit"] == "reports/cfr_vla/mathematical_mechanism_audit.md"
    assert cfr_math["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr_math["kl_between_deterministic_actions_used"] is False
    assert cfr_math["deterministic_action_kl_forbidden"] is True
    assert cfr_math["official_action_validity_semantics_required"] is True
    assert cfr_math["dfm_proxy_policy_2_required"] is True
    assert "CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in cfr_math["stage_0_stop_classes"]
    cfr_prereg = state["epoch_4_cycle_27_cfr_preregistration"]
    assert cfr_prereg["final_decision"] == "CFR_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert cfr_prereg["preregistration"] == "reports/cfr_vla/preregistration.md"
    assert cfr_prereg["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr_prereg["stage_0_allowed_next"] is True
    assert cfr_prereg["bounded_validation_search_max_configs"] == 6
    cfr_protocol = state["epoch_4_cycle_27_cfr_prototype_protocol"]
    assert cfr_protocol["final_decision"] == "CFR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert cfr_protocol["prototype_protocol"] == "reports/cfr_vla/prototype_protocol.md"
    assert cfr_protocol["proposal_hash"] == CFR_PROPOSAL_HASH
    assert cfr_protocol["stage_0_allowed_next"] is True
    assert cfr_protocol["runner"] == "scripts/run_cfr_vla_stage0.py"
    assert cfr_protocol["stage_0_result"] == "reports/cfr_vla/stage_0_result.json"
    assert cfr_protocol["runner_implemented"] is True
    assert cfr_protocol["runner_unit_tests_passed"] == 8
    assert cfr_protocol["stage_0_pending"] is False
    assert cfr_protocol["stage_0_completed"] is True
    assert cfr_protocol["stage_0_decision"] == "CFR_STAGE_0_NO_USABLE_HEADROOM"
    assert cfr_protocol["bounded_validation_allowed"] is False
    cfr_prelaunch = state["epoch_4_cycle_27_cfr_stage_0_prelaunch"]
    assert cfr_prelaunch["final_decision"] == "CFR_STAGE_0_RUNNER_IMPLEMENTED_READY_TO_LAUNCH"
    assert cfr_prelaunch["runner"] == "scripts/run_cfr_vla_stage0.py"
    assert cfr_prelaunch["helper_module"] == "tca_map/smolvla/cfr_vla.py"
    assert cfr_prelaunch["unit_tests_passed"] == 8
    assert cfr_prelaunch["stage_0_action_semantics"] == "reports/cfr_vla/stage_0_action_semantics.json"
    cfr_outcome = state["epoch_4_cycle_27_cfr_stage_0_outcome"]
    assert cfr_outcome["final_decision"] == "CFR_STAGE_0_NO_USABLE_HEADROOM"
    assert cfr_outcome["failure_class"] == "NO_USABLE_HEADROOM"
    assert cfr_outcome["valid_scientific_result"] is False
    assert cfr_outcome["scientific_kill"] is False
    assert cfr_outcome["bounded_validation_allowed"] is False
    assert cfr_outcome["stage_a_allowed"] is False
    assert cfr_outcome["rerun_allowed"] is False
    assert cfr_outcome["cfr_rescue_allowed"] is False
    assert cfr_outcome["worker_completed"] is True
    assert cfr_outcome["exit_code_value"] == 0
    assert cfr_outcome["completed_model_row_count"] == 640
    assert cfr_outcome["planned_model_row_count"] == 640
    assert cfr_outcome["exception_count"] == 0
    assert cfr_outcome["manifest_row_count"] == 640
    assert cfr_outcome["partial_row_count"] == 640
    assert cfr_outcome["duplicate_manifest_key_count"] == 0
    assert cfr_outcome["duplicate_partial_key_count"] == 0
    assert cfr_outcome["missing_manifest_key_count"] == 0
    assert cfr_outcome["extra_partial_key_count"] == 0
    assert cfr_outcome["split_overlap_key_count"] == 0
    assert cfr_outcome["key_sets_equal"] is True
    assert cfr_outcome["proposal_hash_ok"] is True
    assert cfr_outcome["serializer_preflight_ok"] is True
    assert cfr_outcome["preflight_passed"] is True
    assert cfr_outcome["official_prior_policy_2_label"] == "dfm_vla_continuous_refinement_proxy"
    assert cfr_outcome["official_dfm_ready"] is False
    assert cfr_outcome["closed_loop_experiment_happened"] is False
    assert cfr_outcome["simulator_load_count"] == 0
    assert cfr_outcome["confirmatory_records_read"] == 0
    assert cfr_outcome["training_happened"] is False
    assert cfr_outcome["validation_search_happened"] is False
    assert cfr_outcome["feature_action_proprio_finite_aligned"] is True
    assert cfr_outcome["base_to_expert_residual_variance_all_positive"] is True
    assert cfr_outcome["residual_probe_relative_improvement"] == -6.04941221711208
    assert cfr_outcome["residual_probe_absolute_huber_improvement"] == -0.11968147462337628
    assert cfr_outcome["dfm_proxy_headroom_relative_improvement"] == -6.068176722319228
    assert cfr_outcome["dfm_proxy_headroom_absolute_huber_improvement"] == -0.11975307303185317
    assert cfr_outcome["cfr_prediction_huber"] == 0.141970899740119
    assert cfr_outcome["dfm_proxy_huber"] == 0.022217826708265838
    assert cfr_outcome["no_iterative_prediction_huber"] == 0.14195415377508339
    assert cfr_outcome["iterative_cfr_distinct_from_no_iterative"] is True
    assert cfr_outcome["action_validity_ok"] is True
    assert cfr_outcome["base_action_valid_under_official_semantics"] is True
    assert cfr_outcome["identity_max_abs_error"] == 0.0
    assert cfr_outcome["checkpoint_reload_ok"] is True
    assert cfr_outcome["finite_objectives_and_gradients"] is True
    assert cfr_outcome["cfr_gradient_nonzero"] is True
    assert cfr_outcome["frozen_parameter_gradient_count"] == 0
    assert cfr_outcome["resource_contention_interval_count"] == 3
    assert cfr_outcome["resource_overlap_interval_ids"] == []
    assert cfr_outcome["resource_unresolved_interval_ids"] == []
    assert cfr_outcome["timing_throughput_resource_evidence_eligible_for_paper"] is False
    cycle28 = state["epoch_4_cycle_28_candidate_search"]
    assert cycle28["candidate_search_pending"] is False
    assert cycle28["candidate_count_required"] == 3
    assert cycle28["candidate_count_generated"] == 3
    assert cycle28["previous_method"] == "CFR-VLA"
    assert cycle28["previous_decision"] == "CFR_STAGE_0_NO_USABLE_HEADROOM"
    assert cycle28["cfr_repair_allowed"] is False
    assert cycle28["cfr_rescue_allowed"] is False
    assert cycle28["selected_method"] == "TSC-VLA"
    assert cycle28["selected_score"] == 91
    assert cycle28["candidate_generation"] == "reports/epoch_4_cycle_28_candidate_generation.md"
    assert cycle28["prior_mechanism_map"] == "reports/epoch_4_cycle_28_prior_mechanism_map.md"
    assert cycle28["selection_decision"] == "TSC_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    tsc = state["epoch_4_cycle_28_candidate_selection"]
    assert tsc["candidate_count"] == 3
    assert tsc["selected_score"] == 91
    assert tsc["method"] == "TSC-VLA"
    assert tsc["closest_prior"] == "TS-Mask VLA"
    assert tsc["closest_prior_primary_source"] == "https://arxiv.org/abs/2607.09818"
    assert tsc["candidate_generation"] == "reports/epoch_4_cycle_28_candidate_generation.md"
    assert tsc["prior_mechanism_map"] == "reports/epoch_4_cycle_28_prior_mechanism_map.md"
    assert tsc["contribution_type"] == "PRIOR_EXTENSION"
    assert tsc["policy_order"] == [
        "smolvla_base",
        "ts_mask_continuous_proxy_or_official_ts_mask_vla_if_installed",
        "tsc_full",
        "tsc_no_targeted_mask_ablation",
        "standard_lora",
    ]
    assert tsc["standard_lora_required"] is True
    assert tsc["bounded_validation_search_max_configs"] == 6
    assert tsc["training_happened"] is False
    assert tsc["validation_search_happened"] is False
    assert tsc["closed_loop_experiment_happened"] is False
    assert tsc["confirmatory_test_tuning_happened"] is False
    assert tsc["first_serious_comparison_includes_closest_prior"] is True
    assert tsc["cfr_repair_allowed"] is False
    assert tsc["cfr_rescue_allowed"] is False
    assert tsc["selection_decision"] == "TSC_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    assert tsc["proposal"] == "reports/tsc_vla/researcher_proposal.md"
    assert tsc["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc["proposal_hash_file"] == "reports/tsc_vla/proposal_hash.txt"
    assert tsc["proposal_decision"] == "TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    tsc_proposal = state["epoch_4_cycle_28_tsc_researcher_proposal"]
    assert tsc_proposal["final_decision"] == "TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert tsc_proposal["proposal"] == "reports/tsc_vla/researcher_proposal.md"
    assert tsc_proposal["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_proposal["proposal_hash_file"] == "reports/tsc_vla/proposal_hash.txt"
    assert tsc_proposal["closest_prior"] == "TS-Mask VLA"
    assert tsc_proposal["training_happened"] is False
    assert tsc_proposal["validation_search_happened"] is False
    assert tsc_proposal["closed_loop_experiment_happened"] is False
    assert tsc_proposal["confirmatory_test_tuning_happened"] is False
    tsc_review = state["epoch_4_cycle_28_tsc_reviewer_attack"]
    assert tsc_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert tsc_review["reviewer_attack"] == "reports/tsc_vla/reviewer_attack.md"
    assert tsc_review["proposal_hash"] == TSC_PROPOSAL_HASH
    assert "ts_mask_continuous_proxy or official TS-Mask VLA remains policy 2" in tsc_review["conditions"]
    assert "no privileged inference input and no confirmatory-test tuning" in tsc_review["conditions"]
    tsc_rebuttal = state["epoch_4_cycle_28_tsc_rebuttal"]
    assert tsc_rebuttal["final_decision"] == "TSC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert tsc_rebuttal["researcher_rebuttal"] == "reports/tsc_vla/researcher_rebuttal.md"
    assert tsc_rebuttal["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_rebuttal["accepted_reviewer_conditions"] is True
    assert tsc_rebuttal["accepted_closest_prior_proxy"] == "ts_mask_continuous_proxy_or_official_ts_mask_vla_if_installed"
    assert tsc_rebuttal["accepted_key_ablation"] == "tsc_no_targeted_mask_ablation"
    assert tsc_rebuttal["accepted_simple_baseline"] == "standard_lora"
    assert tsc_rebuttal["accepted_no_privileged_inference_inputs"] is True
    tsc_math = state["epoch_4_cycle_28_tsc_mathematical_audit"]
    assert tsc_math["final_decision"] == "TSC_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert tsc_math["mathematical_audit"] == "reports/tsc_vla/mathematical_mechanism_audit.md"
    assert tsc_math["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_math["kl_between_deterministic_actions_used"] is False
    assert tsc_math["deterministic_action_kl_forbidden"] is True
    assert tsc_math["ts_mask_proxy_policy_2_required"] is True
    assert "TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in tsc_math["stage_0_stop_classes"]
    tsc_prereg = state["epoch_4_cycle_28_tsc_preregistration"]
    assert tsc_prereg["final_decision"] == "TSC_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert tsc_prereg["preregistration"] == "reports/tsc_vla/preregistration.md"
    assert tsc_prereg["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_prereg["stage_0_allowed_next"] is True
    assert tsc_prereg["bounded_validation_search_max_configs"] == 6
    assert "TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in tsc_prereg["stage_0_stop_classes"]
    assert tsc_prereg["prototype_protocol"] == "reports/tsc_vla/prototype_protocol.md"
    assert tsc_prereg["prototype_protocol_decision"] == "TSC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    tsc_protocol = state["epoch_4_cycle_28_tsc_prototype_protocol"]
    assert tsc_protocol["final_decision"] == "TSC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert tsc_protocol["prototype_protocol"] == "reports/tsc_vla/prototype_protocol.md"
    assert tsc_protocol["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_protocol["stage_0_allowed_next"] is True
    assert tsc_protocol["runner"] == "scripts/run_tsc_vla_stage0.py"
    assert tsc_protocol["stage_0_result"] == "reports/tsc_vla/stage_0_result.json"
    assert tsc_protocol["stage_0_partial"] == "reports/tsc_vla/stage_0_partial.json"
    assert tsc_protocol["runner_implemented"] is True
    assert tsc_protocol["helper_module"] == "tca_map/smolvla/tsc_vla.py"
    assert tsc_protocol["runner_validation"] == "tests/test_tsc_vla.py"
    assert tsc_protocol["runner_unit_tests_passed"] == 8
    assert tsc_protocol["py_compile_passed"] is True
    assert tsc_protocol["serializer_preflight"] == "reports/tsc_vla/stage_0_serializer_preflight.json"
    assert tsc_protocol["serializer_preflight_passed"] is True
    assert tsc_protocol["stage_0_pending"] is False
    assert tsc_protocol["stage_0_completed"] is True
    assert tsc_protocol["stage_0_decision"] == "TSC_STAGE_0_NO_USABLE_HEADROOM"
    assert tsc_protocol["bounded_validation_allowed"] is False
    tsc_prelaunch = state["epoch_4_cycle_28_tsc_stage_0_prelaunch"]
    assert tsc_prelaunch["final_decision"] == "TSC_STAGE_0_RUNNER_IMPLEMENTED_READY_TO_LAUNCH"
    assert tsc_prelaunch["runner"] == "scripts/run_tsc_vla_stage0.py"
    assert tsc_prelaunch["unit_tests_passed"] == 8
    assert tsc_prelaunch["serializer_preflight_passed"] is True
    assert tsc_prelaunch["stage_0_pending"] is True
    tsc_outcome = state["epoch_4_cycle_28_tsc_stage_0_outcome"]
    assert tsc_outcome["final_decision"] == "TSC_STAGE_0_NO_USABLE_HEADROOM"
    assert tsc_outcome["completed_model_row_count"] == 640
    assert tsc_outcome["planned_model_row_count"] == 640
    assert tsc_outcome["exception_count"] == 0
    assert tsc_outcome["duplicate_manifest_key_count"] == 0
    assert tsc_outcome["duplicate_partial_key_count"] == 0
    assert tsc_outcome["missing_manifest_key_count"] == 0
    assert tsc_outcome["extra_partial_key_count"] == 0
    assert tsc_outcome["split_overlap_key_count"] == 0
    assert tsc_outcome["key_sets_equal"] is True
    assert tsc_outcome["structured_mask_beats_trivial"] is False
    assert tsc_outcome["structured_mask_beats_magnitude"] is False
    assert tsc_outcome["bounded_validation_allowed"] is False
    assert tsc_outcome["scientific_kill"] is False
    cycle29 = state["epoch_4_cycle_29_candidate_search"]
    assert cycle29["candidate_search_pending"] is False
    assert cycle29["candidate_count_required"] == 3
    assert cycle29["candidate_count_generated"] == 3
    assert cycle29["previous_method"] == "TSC-VLA"
    assert cycle29["previous_decision"] == "TSC_STAGE_0_NO_USABLE_HEADROOM"
    assert cycle29["tsc_repair_allowed"] is False
    assert cycle29["tsc_rescue_allowed"] is False
    assert cycle29["selected_method"] == "CCIF-VLA"
    assert cycle29["selected_score"] == 92
    assert cycle29["selection_decision"] == "CCIF_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    ccif = state["epoch_4_cycle_29_candidate_selection"]
    assert ccif["candidate_count"] == 3
    assert ccif["selected_score"] == 92
    assert ccif["method"] == "CCIF-VLA"
    assert ccif["closest_prior"] == "Coarse-to-Control"
    assert ccif["closest_prior_primary_source"] == "https://arxiv.org/abs/2606.07107"
    assert ccif["policy_order"] == [
        "smolvla_base",
        "coarse_to_control_continuous_proxy",
        "ccif_full",
        "ccif_no_coarse_intent_ablation",
        "standard_lora",
    ]
    assert ccif["standard_lora_required"] is True
    assert ccif["training_happened"] is False
    assert ccif["validation_search_happened"] is False
    assert ccif["closed_loop_experiment_happened"] is False
    assert ccif["confirmatory_test_tuning_happened"] is False
    assert ccif["first_serious_comparison_includes_closest_prior"] is True
    assert ccif["tsc_rescue_allowed"] is False
    assert ccif["proposal"] == "reports/ccif_vla/researcher_proposal.md"
    assert ccif["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif["proposal_hash_file"] == "reports/ccif_vla/proposal_hash.txt"
    assert ccif["proposal_decision"] == "CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert ccif["reviewer_attack"] == "reports/ccif_vla/reviewer_attack.md"
    assert ccif["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert ccif["researcher_rebuttal"] == "reports/ccif_vla/researcher_rebuttal.md"
    assert ccif["rebuttal_decision"] == "CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert ccif["mathematical_audit"] == "reports/ccif_vla/mathematical_mechanism_audit.md"
    assert ccif["math_audit_decision"] == "CCIF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert ccif["preregistration"] == "reports/ccif_vla/preregistration.md"
    assert ccif["preregistration_decision"] == "CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    ccif_proposal = state["epoch_4_cycle_29_ccif_researcher_proposal"]
    assert ccif_proposal["final_decision"] == "CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert ccif_proposal["proposal"] == "reports/ccif_vla/researcher_proposal.md"
    assert ccif_proposal["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_proposal["closest_prior"] == "Coarse-to-Control"
    assert ccif_proposal["training_happened"] is False
    assert ccif_proposal["validation_search_happened"] is False
    assert ccif_proposal["closed_loop_experiment_happened"] is False
    assert ccif_proposal["confirmatory_test_tuning_happened"] is False
    assert ccif_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    ccif_review = state["epoch_4_cycle_29_ccif_reviewer_attack"]
    assert ccif_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert ccif_review["reviewer_attack"] == "reports/ccif_vla/reviewer_attack.md"
    assert ccif_review["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_review["closest_prior"] == "Coarse-to-Control"
    assert "CAC-VLA" in ccif_review["independent_closest_current_papers"]
    assert "ccif_no_coarse_intent_ablation remains the key ablation" in ccif_review["conditions"]
    assert "matched standard_lora remains the mandatory simple reviewer-killer" in ccif_review["conditions"]
    assert ccif_review["training_happened"] is False
    assert ccif_review["validation_search_happened"] is False
    assert ccif_review["closed_loop_experiment_happened"] is False
    assert ccif_review["confirmatory_test_tuning_happened"] is False
    assert ccif_review["rebuttal_decision"] == "CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    ccif_rebuttal = state["epoch_4_cycle_29_ccif_rebuttal"]
    assert ccif_rebuttal["final_decision"] == "CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert ccif_rebuttal["researcher_rebuttal"] == "reports/ccif_vla/researcher_rebuttal.md"
    assert ccif_rebuttal["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_rebuttal["accepted_reviewer_conditions"] is True
    assert ccif_rebuttal["accepted_key_ablation"] == "ccif_no_coarse_intent_ablation"
    assert ccif_rebuttal["accepted_simple_baseline"] == "standard_lora"
    assert ccif_rebuttal["accepted_task_phase_mean_intent_diagnostic"] is True
    assert ccif_rebuttal["accepted_endpoint_only_intent_diagnostic"] is True
    assert ccif_rebuttal["accepted_no_privileged_inference_inputs"] is True
    assert ccif_rebuttal["math_audit_decision"] == "CCIF_MATHEMATICAL_AUDIT_PREREGISTERED"
    ccif_math = state["epoch_4_cycle_29_ccif_mathematical_audit"]
    assert ccif_math["final_decision"] == "CCIF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert ccif_math["mathematical_audit"] == "reports/ccif_vla/mathematical_mechanism_audit.md"
    assert ccif_math["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_math["intent_dimension"] == 31
    assert ccif_math["waypoint_indices"] == [9, 19, 34, 49]
    assert ccif_math["kl_between_deterministic_actions_used"] is False
    assert ccif_math["deterministic_action_kl_forbidden"] is True
    assert ccif_math["coarse_to_control_proxy_policy_2_required"] is True
    assert ccif_math["identity_preserving_integration_required"] is True
    assert ccif_math["task_phase_mean_intent_diagnostic_required"] is True
    assert ccif_math["endpoint_only_intent_diagnostic_required"] is True
    assert "CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in ccif_math["stage_0_stop_classes"]
    assert ccif_math["preregistration_decision"] == "CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    ccif_prereg = state["epoch_4_cycle_29_ccif_preregistration"]
    assert ccif_prereg["final_decision"] == "CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert ccif_prereg["preregistration"] == "reports/ccif_vla/preregistration.md"
    assert ccif_prereg["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_prereg["stage_0_allowed_next"] is True
    assert ccif_prereg["bounded_validation_search_max_configs"] == 6
    assert ccif_prereg["intent_dimension"] == 31
    assert ccif_prereg["waypoint_indices"] == [9, 19, 34, 49]
    assert ccif_prereg["prototype_protocol"] == "reports/ccif_vla/prototype_protocol.md"
    assert ccif_prereg["prototype_protocol_decision"] == "CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    ccif_protocol = state["epoch_4_cycle_29_ccif_prototype_protocol"]
    assert ccif_protocol["final_decision"] == "CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert ccif_protocol["prototype_protocol"] == "reports/ccif_vla/prototype_protocol.md"
    assert ccif_protocol["proposal_hash"] == CCIF_PROPOSAL_HASH
    assert ccif_protocol["stage_0_allowed_next"] is True
    assert ccif_protocol["runner"] == "scripts/run_ccif_vla_stage0.py"
    assert ccif_protocol["helper_module"] == "tca_map/smolvla/ccif_vla.py"
    assert ccif_protocol["unit_tests"] == "tests/test_ccif_vla.py"
    assert ccif_protocol["stage_0_result"] == "reports/ccif_vla/stage_0_result.json"
    assert ccif_protocol["stage_0_partial"] == "reports/ccif_vla/stage_0_partial.json"
    ccif_implementation = state["epoch_4_cycle_29_ccif_stage_0_implementation"]
    assert ccif_implementation["final_decision"] == "CCIF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert ccif_implementation["compile_passed"] is True
    assert ccif_implementation["focused_test_result"] == "9 passed"
    assert ccif_implementation["serializer_preflight_passed"] is True
    assert ccif_implementation["stage_0_launch_allowed_next"] is True
    assert ccif_implementation["stage_0_final_decision"] == "CCIF_STAGE_0_DESIGN_FAILURE"
    assert ccif_implementation["training_happened"] is False
    assert ccif_implementation["closed_loop_experiment_happened"] is False
    ccif_outcome = state["epoch_4_cycle_29_ccif_stage_0_outcome"]
    assert ccif_outcome["final_decision"] == "CCIF_STAGE_0_DESIGN_FAILURE"
    assert ccif_outcome["completed_model_row_count"] == ccif_outcome["planned_model_row_count"] == 4480
    assert ccif_outcome["unique_observation_row_count"] == 640
    assert ccif_outcome["exception_count"] == 0
    assert ccif_outcome["resume_exception_count"] == 2
    assert ccif_outcome["duplicate_partial_key_count"] == 0
    assert ccif_outcome["key_sets_equal"] is True
    assert ccif_outcome["intent_probe_beats_task_phase_mean"] is False
    assert ccif_outcome["endpoint_only_explains_ccif"] is True
    assert ccif_outcome["bounded_validation_allowed"] is False
    assert ccif_outcome["valid_scientific_result"] is False
    assert ccif_outcome["closed_loop_experiment_happened"] is False
    assert ccif_outcome["ccif_rescue_allowed"] is False
    cycle30 = state["epoch_4_cycle_30_candidate_search"]
    assert cycle30["candidate_search_pending"] is False
    assert cycle30["candidate_count_required"] == 3
    assert cycle30["candidate_count_generated"] == 3
    assert cycle30["previous_method"] == "CCIF-VLA"
    assert cycle30["previous_decision"] == "CCIF_STAGE_0_DESIGN_FAILURE"
    assert cycle30["ccif_repair_allowed"] is False
    assert cycle30["ccif_rescue_allowed"] is False
    assert cycle30["selected_method"] == "URF-VLA"
    assert cycle30["selected_score"] == 92
    assert cycle30["selection_decision"] == "URF_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    urf = state["epoch_4_cycle_30_candidate_selection"]
    assert urf["candidate_count"] == 3
    assert urf["selected_score"] == 92
    assert urf["method"] == "URF-VLA"
    assert urf["closest_prior"] == "SUREFlow"
    assert urf["closest_prior_primary_source"] == "https://arxiv.org/abs/2607.10504"
    assert urf["closest_prior_official_repository"] == "https://github.com/tanvirnwu/SUREFlow"
    assert urf["policy_order"] == [
        "smolvla_base",
        "sureflow_uncertainty_residual_proxy",
        "urf_full",
        "urf_no_uncertainty_route_ablation",
        "standard_lora",
    ]
    assert urf["standard_lora_required"] is True
    assert urf["training_happened"] is False
    assert urf["validation_search_happened"] is False
    assert urf["closed_loop_experiment_happened"] is False
    assert urf["confirmatory_test_tuning_happened"] is False
    assert urf["first_serious_comparison_includes_closest_prior"] is True
    assert urf["proposal"] == "reports/urf_vla/researcher_proposal.md"
    assert urf["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf["proposal_hash_file"] == "reports/urf_vla/proposal_hash.txt"
    assert urf["proposal_decision"] == "URF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert urf["reviewer_attack"] == "reports/urf_vla/reviewer_attack.md"
    assert urf["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert urf["researcher_rebuttal"] == "reports/urf_vla/researcher_rebuttal.md"
    assert urf["rebuttal_decision"] == "URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert urf["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf["math_audit_decision"] == "URF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert urf["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf["preregistration_decision"] == "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert urf["prototype_protocol"] == "reports/urf_vla/prototype_protocol.md"
    assert urf["prototype_protocol_decision"] == "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    urf_proposal = state["epoch_4_cycle_30_urf_researcher_proposal"]
    assert urf_proposal["final_decision"] == "URF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert urf_proposal["proposal"] == "reports/urf_vla/researcher_proposal.md"
    assert urf_proposal["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_proposal["closest_prior"] == "SUREFlow"
    assert urf_proposal["standard_lora_required"] is True
    assert urf_proposal["training_happened"] is False
    assert urf_proposal["validation_search_happened"] is False
    assert urf_proposal["closed_loop_experiment_happened"] is False
    assert urf_proposal["confirmatory_test_tuning_happened"] is False
    assert urf_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert urf_proposal["researcher_rebuttal"] == "reports/urf_vla/researcher_rebuttal.md"
    assert urf_proposal["rebuttal_decision"] == "URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert urf_proposal["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf_proposal["math_audit_decision"] == "URF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert urf_proposal["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf_proposal["preregistration_decision"] == "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert urf_proposal["prototype_protocol"] == "reports/urf_vla/prototype_protocol.md"
    assert urf_proposal["prototype_protocol_decision"] == "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    urf_review = state["epoch_4_cycle_30_urf_reviewer_attack"]
    assert urf_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert urf_review["reviewer_attack"] == "reports/urf_vla/reviewer_attack.md"
    assert urf_review["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_review["closest_prior"] == "SUREFlow"
    assert "Guided Action Flow" in urf_review["independent_closest_current_papers"]
    assert "urf_no_uncertainty_route_ablation remains the key ablation" in urf_review["conditions"]
    assert "standard_lora remains the first simple reviewer-killer" in urf_review["conditions"]
    assert "No deterministic-action KL is allowed" in urf_review["conditions"]
    assert urf_review["training_happened"] is False
    assert urf_review["validation_search_happened"] is False
    assert urf_review["closed_loop_experiment_happened"] is False
    assert urf_review["confirmatory_test_tuning_happened"] is False
    assert urf_review["researcher_rebuttal"] == "reports/urf_vla/researcher_rebuttal.md"
    assert urf_review["rebuttal_decision"] == "URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert urf_review["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf_review["math_audit_decision"] == "URF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert urf_review["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf_review["preregistration_decision"] == "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert urf_review["prototype_protocol"] == "reports/urf_vla/prototype_protocol.md"
    assert urf_review["prototype_protocol_decision"] == "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    urf_rebuttal = state["epoch_4_cycle_30_urf_rebuttal"]
    assert urf_rebuttal["final_decision"] == "URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert urf_rebuttal["researcher_rebuttal"] == "reports/urf_vla/researcher_rebuttal.md"
    assert urf_rebuttal["reviewer_attack"] == "reports/urf_vla/reviewer_attack.md"
    assert urf_rebuttal["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_rebuttal["accepted_reviewer_conditions"] is True
    assert urf_rebuttal["accepted_closest_prior"] == "SUREFlow"
    assert urf_rebuttal["accepted_closest_prior_proxy"] == "sureflow_uncertainty_residual_proxy"
    assert urf_rebuttal["accepted_closest_frozen_smolvla_intervention_prior"] == "Guided Action Flow"
    assert urf_rebuttal["accepted_key_ablation"] == "urf_no_uncertainty_route_ablation"
    assert urf_rebuttal["accepted_simple_baseline"] == "standard_lora"
    assert urf_rebuttal["deterministic_action_kl_forbidden"] is True
    assert "CCIF-VLA" in urf_rebuttal["closed_methods_remain_closed"]
    assert urf_rebuttal["training_happened"] is False
    assert urf_rebuttal["validation_search_happened"] is False
    assert urf_rebuttal["closed_loop_experiment_happened"] is False
    assert urf_rebuttal["confirmatory_test_tuning_happened"] is False
    assert urf_rebuttal["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf_rebuttal["math_audit_decision"] == "URF_MATHEMATICAL_AUDIT_PREREGISTERED"
    urf_math = state["epoch_4_cycle_30_urf_mathematical_audit"]
    assert urf_math["final_decision"] == "URF_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert urf_math["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf_math["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_math["residual_horizon"] == 50
    assert urf_math["action_dimension"] == 7
    assert urf_math["uncertainty_enters_route_gate"] is True
    assert urf_math["deterministic_action_kl_forbidden"] is True
    assert urf_math["first_serious_comparison"] == [
        "smolvla_base",
        "sureflow_uncertainty_residual_proxy",
        "urf_full",
        "urf_no_uncertainty_route_ablation",
        "standard_lora",
    ]
    assert "URF_STAGE_0_DESIGN_FAILURE" in urf_math["stage_0_stop_classes"]
    assert urf_math["requires_uncertainty_strata_monotonicity"] is True
    assert urf_math["requires_no_global_route_gate"] is True
    assert urf_math["requires_identity_reload_error_max"] == 1e-06
    assert urf_math["requires_official_action_validity_semantics"] is True
    assert urf_math["training_happened"] is False
    assert urf_math["validation_search_happened"] is False
    assert urf_math["closed_loop_experiment_happened"] is False
    assert urf_math["confirmatory_test_tuning_happened"] is False
    assert urf_math["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf_math["preregistration_decision"] == "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    urf_prereg = state["epoch_4_cycle_30_urf_preregistration"]
    assert urf_prereg["final_decision"] == "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert urf_prereg["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf_prereg["mathematical_audit"] == "reports/urf_vla/mathematical_mechanism_audit.md"
    assert urf_prereg["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_prereg["stage_0_allowed_next"] is True
    assert urf_prereg["bounded_validation_search_max_configs"] == 6
    assert urf_prereg["development_tasks"] == [
        "libero_spatial/task_3",
        "libero_object/task_3",
        "libero_goal/task_5",
        "libero_10/task_5",
    ]
    assert urf_prereg["route_positive_fraction_min"] == 0.02
    assert urf_prereg["route_positive_fraction_max"] == 0.8
    assert urf_prereg["uncertainty_monotonicity_spearman_min"] == 0.2
    assert urf_prereg["first_serious_comparison"] == [
        "smolvla_base",
        "sureflow_uncertainty_residual_proxy",
        "urf_full",
        "urf_no_uncertainty_route_ablation",
        "standard_lora",
    ]
    assert "URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in urf_prereg["stage_0_stop_classes"]
    assert urf_prereg["training_happened"] is False
    assert urf_prereg["validation_search_happened"] is False
    assert urf_prereg["closed_loop_experiment_happened"] is False
    assert urf_prereg["confirmatory_test_tuning_happened"] is False
    assert urf_prereg["prototype_protocol"] == "reports/urf_vla/prototype_protocol.md"
    assert urf_prereg["prototype_protocol_decision"] == "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    urf_protocol = state["epoch_4_cycle_30_urf_prototype_protocol"]
    assert urf_protocol["final_decision"] == "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert urf_protocol["prototype_protocol"] == "reports/urf_vla/prototype_protocol.md"
    assert urf_protocol["preregistration"] == "reports/urf_vla/preregistration.md"
    assert urf_protocol["proposal_hash"] == URF_PROPOSAL_HASH
    assert urf_protocol["stage_0_allowed_next"] is True
    assert urf_protocol["helper_module"] == "tca_map/smolvla/urf_vla.py"
    assert urf_protocol["runner"] == "scripts/run_urf_vla_stage0.py"
    assert urf_protocol["unit_tests"] == "tests/test_urf_vla.py"
    assert urf_protocol["stage_0_result"] == "reports/urf_vla/stage_0_result.json"
    assert urf_protocol["stage_0_partial"] == "reports/urf_vla/stage_0_partial.json"
    assert urf_protocol["stage_0_manifest"] == "reports/urf_vla/stage_0_manifest.json"
    assert urf_protocol["stage_0_action_semantics"] == "reports/urf_vla/stage_0_action_semantics.json"
    assert urf_protocol["stage_0_serializer_preflight"] == "reports/urf_vla/stage_0_serializer_preflight.json"
    assert urf_protocol["training_happened"] is False
    assert urf_protocol["validation_search_happened"] is False
    assert urf_protocol["closed_loop_experiment_happened"] is False
    assert urf_protocol["confirmatory_test_tuning_happened"] is False
    urf_implementation = state["epoch_4_cycle_30_urf_stage_0_implementation"]
    assert urf_implementation["final_decision"] == "URF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert urf_implementation["compile_passed"] is True
    assert urf_implementation["focused_test_result"] == "8 passed"
    assert urf_implementation["serializer_preflight_passed"] is True
    assert urf_implementation["stage_0_launch_allowed_next"] is True
    assert urf_implementation["stage_0_final_decision"] == "URF_STAGE_0_NO_USABLE_HEADROOM"
    assert urf_implementation["training_happened"] is False
    assert urf_implementation["closed_loop_experiment_happened"] is False
    urf_outcome = state["epoch_4_cycle_30_urf_stage_0_outcome"]
    assert urf_outcome["final_decision"] == "URF_STAGE_0_NO_USABLE_HEADROOM"
    assert urf_outcome["failure_class"] == "NO_USABLE_HEADROOM"
    assert urf_outcome["completed_model_row_count"] == urf_outcome["planned_model_row_count"] == 5120
    assert urf_outcome["exception_count"] == 0
    assert urf_outcome["duplicate_manifest_key_count"] == 0
    assert urf_outcome["duplicate_partial_key_count"] == 0
    assert urf_outcome["missing_manifest_key_count"] == 0
    assert urf_outcome["extra_partial_key_count"] == 0
    assert urf_outcome["split_overlap_key_count"] == 0
    assert urf_outcome["key_sets_equal"] is True
    assert urf_outcome["base_residual_headroom_ok"] is False
    assert urf_outcome["hetero_beats_homoscedastic_relative"] < 0
    assert urf_outcome["hetero_beats_task_phase_relative"] < 0
    assert urf_outcome["bounded_validation_allowed"] is False
    assert urf_outcome["valid_scientific_result"] is False
    assert urf_outcome["closed_loop_experiment_happened"] is False
    assert urf_outcome["urf_rescue_allowed"] is False
    cycle31 = state["epoch_4_cycle_31_candidate_search"]
    assert cycle31["candidate_search_pending"] is False
    assert cycle31["candidate_count_required"] == 3
    assert cycle31["candidate_count_generated"] == 3
    assert cycle31["previous_method"] == "URF-VLA"
    assert cycle31["previous_decision"] == "URF_STAGE_0_NO_USABLE_HEADROOM"
    assert cycle31["selected_method"] == "S2C-VLA"
    assert cycle31["selected_score"] == 95
    assert cycle31["selection_decision"] == "S2C_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    s2c = state["epoch_4_cycle_31_candidate_selection"]
    assert s2c["candidate_count"] == 3
    assert s2c["selected_score"] == 95
    assert s2c["method"] == "S2C-VLA"
    assert s2c["closest_prior"] == "ChunkFlow"
    assert s2c["closest_prior_primary_source"] == "https://arxiv.org/html/2607.12992v1"
    assert s2c["policy_order"] == [
        "smolvla_base",
        "chunkflow_overlap_proxy",
        "s2c_full",
        "s2c_no_learned_overlap_mask_ablation",
        "standard_lora",
    ]
    assert s2c["standard_lora_required"] is True
    assert s2c["training_happened"] is False
    assert s2c["validation_search_happened"] is False
    assert s2c["closed_loop_experiment_happened"] is False
    assert s2c["confirmatory_test_tuning_happened"] is False
    assert s2c["first_serious_comparison_includes_closest_prior"] is True
    assert s2c["proposal"] == "reports/s2c_vla/researcher_proposal.md"
    assert s2c["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c["proposal_hash_file"] == "reports/s2c_vla/proposal_hash.txt"
    assert s2c["selection_decision"] == "S2C_CANDIDATE_SELECTED_STAGE_0_LAUNCH_PENDING"
    assert s2c["proposal_decision"] == "S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED"
    assert s2c["reviewer_attack"] == "reports/s2c_vla/reviewer_attack.md"
    assert s2c["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert "already decoded SmolVLA chunks" in s2c["accepted_novelty_boundary_required"]
    s2c_proposal = state["epoch_4_cycle_31_s2c_researcher_proposal"]
    assert s2c_proposal["final_decision"] == "S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED"
    assert s2c_proposal["proposal"] == "reports/s2c_vla/researcher_proposal.md"
    assert s2c_proposal["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_proposal["closest_prior"] == "ChunkFlow"
    assert s2c_proposal["reviewer_attack"] == "reports/s2c_vla/reviewer_attack.md"
    assert s2c_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert s2c_proposal["standard_lora_required"] is True
    assert s2c_proposal["training_happened"] is False
    assert s2c_proposal["validation_search_happened"] is False
    assert s2c_proposal["closed_loop_experiment_happened"] is False
    assert s2c_proposal["confirmatory_test_tuning_happened"] is False
    assert s2c_proposal["researcher_rebuttal"] == "reports/s2c_vla/researcher_rebuttal.md"
    assert s2c_proposal["rebuttal_decision"] == "S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    s2c_review = state["epoch_4_cycle_31_s2c_reviewer_attack"]
    assert s2c_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert s2c_review["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_review["closest_prior"] == "ChunkFlow"
    assert "ChunkFlow remains the closest prior and policy 2" in s2c_review["conditions"]
    assert s2c_review["researcher_rebuttal"] == "reports/s2c_vla/researcher_rebuttal.md"
    assert s2c_review["rebuttal_decision"] == "S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert s2c_review["training_happened"] is False
    assert s2c_review["validation_search_happened"] is False
    assert s2c_review["closed_loop_experiment_happened"] is False
    assert s2c_review["confirmatory_test_tuning_happened"] is False
    s2c_rebuttal = state["epoch_4_cycle_31_s2c_rebuttal"]
    assert s2c_rebuttal["final_decision"] == "S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert s2c_rebuttal["researcher_rebuttal"] == "reports/s2c_vla/researcher_rebuttal.md"
    assert s2c_rebuttal["reviewer_attack"] == "reports/s2c_vla/reviewer_attack.md"
    assert s2c_rebuttal["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_rebuttal["accepted_reviewer_conditions"] is True
    assert s2c_rebuttal["accepted_closest_prior"] == "ChunkFlow"
    assert s2c_rebuttal["accepted_secondary_prior"] == "SEAM"
    assert s2c_rebuttal["accepted_key_ablation"] == "s2c_no_learned_overlap_mask_ablation"
    assert s2c_rebuttal["accepted_simple_baseline"] == "standard_lora"
    assert s2c_rebuttal["deterministic_action_kl_forbidden"] is True
    assert s2c_rebuttal["accepted_stage_0_headroom_gate"] is True
    assert s2c_rebuttal["accepted_gripper_event_protection"] is True
    assert "URF-VLA" in s2c_rebuttal["closed_methods_remain_closed"]
    assert s2c_rebuttal["training_happened"] is False
    assert s2c_rebuttal["validation_search_happened"] is False
    assert s2c_rebuttal["closed_loop_experiment_happened"] is False
    assert s2c_rebuttal["confirmatory_test_tuning_happened"] is False
    assert s2c_rebuttal["mathematical_audit"] == "reports/s2c_vla/mathematical_mechanism_audit.md"
    assert s2c_rebuttal["math_audit_decision"] == "S2C_MATHEMATICAL_AUDIT_PREREGISTERED"
    s2c_math = state["epoch_4_cycle_31_s2c_mathematical_audit"]
    assert s2c_math["final_decision"] == "S2C_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert s2c_math["mathematical_audit"] == "reports/s2c_vla/mathematical_mechanism_audit.md"
    assert s2c_math["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_math["chunk_horizon"] == 50
    assert s2c_math["replanning_stride"] == 10
    assert s2c_math["overlap_length"] == 10
    assert s2c_math["action_dimension"] == 7
    assert s2c_math["deterministic_action_kl_forbidden"] is True
    assert s2c_math["first_serious_comparison"] == [
        "smolvla_base",
        "chunkflow_overlap_proxy",
        "s2c_full",
        "s2c_no_learned_overlap_mask_ablation",
        "standard_lora",
    ]
    assert "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in s2c_math["stage_0_stop_classes"]
    assert s2c_math["future_zone_drift_max"] == 0.0
    assert s2c_math["gripper_event_destruction_max"] == 0
    assert s2c_math["training_happened"] is False
    assert s2c_math["validation_search_happened"] is False
    assert s2c_math["closed_loop_experiment_happened"] is False
    assert s2c_math["confirmatory_test_tuning_happened"] is False
    assert s2c_math["preregistration"] == "reports/s2c_vla/preregistration.md"
    assert s2c_math["preregistration_decision"] == "S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    s2c_prereg = state["epoch_4_cycle_31_s2c_preregistration"]
    assert s2c_prereg["final_decision"] == "S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert s2c_prereg["preregistration"] == "reports/s2c_vla/preregistration.md"
    assert s2c_prereg["mathematical_audit"] == "reports/s2c_vla/mathematical_mechanism_audit.md"
    assert s2c_prereg["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_prereg["stage_0_allowed_next"] is True
    assert s2c_prereg["bounded_validation_search_max_configs"] == 6
    assert s2c_prereg["development_tasks"] == [
        "libero_spatial/task_3",
        "libero_object/task_3",
        "libero_goal/task_5",
        "libero_10/task_5",
    ]
    assert s2c_prereg["discovery_demo_ids"] == "0..7"
    assert s2c_prereg["validation_demo_ids"] == "8..9"
    assert s2c_prereg["resume_key_fields"] == [
        "split",
        "task_suite",
        "task_id",
        "demo_id",
        "window_start",
        "stride",
        "previous_policy_source",
        "policy",
    ]
    assert "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in s2c_prereg["stage_0_stop_classes"]
    assert s2c_prereg["training_happened"] is False
    assert s2c_prereg["validation_search_happened"] is False
    assert s2c_prereg["closed_loop_experiment_happened"] is False
    assert s2c_prereg["confirmatory_test_tuning_happened"] is False
    assert s2c_prereg["prototype_protocol"] == "reports/s2c_vla/prototype_protocol.md"
    assert s2c_prereg["prototype_protocol_decision"] == "S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    s2c_protocol = state["epoch_4_cycle_31_s2c_prototype_protocol"]
    assert s2c_protocol["final_decision"] == "S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert s2c_protocol["prototype_protocol"] == "reports/s2c_vla/prototype_protocol.md"
    assert s2c_protocol["preregistration"] == "reports/s2c_vla/preregistration.md"
    assert s2c_protocol["proposal_hash"] == S2C_PROPOSAL_HASH
    assert s2c_protocol["stage_0_allowed_next"] is True
    assert s2c_protocol["helper_module"] == "tca_map/smolvla/s2c_vla.py"
    assert s2c_protocol["runner"] == "scripts/run_s2c_vla_stage0.py"
    assert s2c_protocol["unit_tests"] == "tests/test_s2c_vla.py"
    assert s2c_protocol["stage_0_result"] == "reports/s2c_vla/stage_0_result.json"
    assert s2c_protocol["stage_0_partial"] == "reports/s2c_vla/stage_0_partial.json"
    assert s2c_protocol["stage_0_manifest"] == "reports/s2c_vla/stage_0_manifest.json"
    assert s2c_protocol["stage_0_action_semantics"] == "reports/s2c_vla/stage_0_action_semantics.json"
    assert s2c_protocol["stage_0_serializer_preflight"] == "reports/s2c_vla/stage_0_serializer_preflight.json"
    assert s2c_protocol["training_happened"] is False
    assert s2c_protocol["validation_search_happened"] is False
    assert s2c_protocol["closed_loop_experiment_happened"] is False
    assert s2c_protocol["confirmatory_test_tuning_happened"] is False
    assert s2c_protocol["implementation_decision"] == "S2C_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert s2c_protocol["stage_0_launch_allowed_next"] is True
    s2c_implementation = state["epoch_4_cycle_31_s2c_stage_0_implementation"]
    assert s2c_implementation["final_decision"] == "S2C_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert s2c_implementation["helper_module"] == "tca_map/smolvla/s2c_vla.py"
    assert s2c_implementation["runner"] == "scripts/run_s2c_vla_stage0.py"
    assert s2c_implementation["unit_tests"] == "tests/test_s2c_vla.py"
    assert s2c_implementation["compile_passed"] is True
    assert s2c_implementation["focused_test_result"] == "7 passed"
    assert s2c_implementation["serializer_preflight"] == "reports/s2c_vla/stage_0_serializer_preflight.json"
    assert s2c_implementation["serializer_preflight_passed"] is True
    assert s2c_implementation["serializer_preflight_fixture_hash"] == s2c_implementation["serializer_preflight_reproduced_hash"]
    assert s2c_implementation["stage_0_launch_allowed_next"] is True
    assert s2c_implementation["training_happened"] is False
    assert s2c_implementation["validation_search_happened"] is False
    assert s2c_implementation["closed_loop_experiment_happened"] is False
    assert s2c_implementation["confirmatory_test_tuning_happened"] is False
    s2c_outcome = state["epoch_4_cycle_31_s2c_stage_0_outcome"]
    assert s2c_outcome["final_decision"] == "S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    assert s2c_outcome["completed_model_row_count"] == s2c_outcome["planned_model_row_count"] == 885
    assert s2c_outcome["exception_count"] == 0
    assert s2c_outcome["duplicate_manifest_key_count"] == 0
    assert s2c_outcome["duplicate_partial_key_count"] == 0
    assert s2c_outcome["missing_manifest_key_count"] == 0
    assert s2c_outcome["extra_partial_key_count"] == 0
    assert s2c_outcome["split_overlap_key_count"] == 0
    assert s2c_outcome["key_sets_equal"] is True
    assert s2c_outcome["adjacent_pair_count"] == 177
    assert s2c_outcome["base_boundary_headroom_ok"] is False
    assert s2c_outcome["bounded_validation_allowed"] is False
    assert s2c_outcome["valid_scientific_result"] is False
    assert s2c_outcome["closed_loop_experiment_happened"] is False
    assert s2c_outcome["s2c_rescue_allowed"] is False
    cycle32 = state["epoch_4_cycle_32_candidate_search"]
    assert cycle32["candidate_search_pending"] is False
    assert cycle32["candidate_count_required"] == 3
    assert cycle32["candidate_count_generated"] == 3
    assert cycle32["previous_method"] == "S2C-VLA"
    assert cycle32["previous_decision"] == "S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    assert cycle32["prior_mechanism_map"] == "reports/epoch_4_cycle_32_prior_mechanism_map.md"
    assert cycle32["candidate_generation"] == "reports/epoch_4_cycle_32_candidate_generation.md"
    assert cycle32["candidate_ids"] == ["LCG-VLA", "TAGR-VLA", "PGP-VLA"]
    assert cycle32["selected_method"] == "LCG-VLA"
    assert cycle32["selected_score"] == 93
    assert cycle32["selection_decision"] == "LCG_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    assert cycle32["s2c_repair_allowed"] is False
    assert cycle32["s2c_rescue_allowed"] is False
    lcg = state["epoch_4_cycle_32_candidate_selection"]
    assert lcg["method"] == "LCG-VLA"
    assert lcg["candidate_count"] == 3
    assert lcg["selected_score"] == 93
    assert lcg["closest_prior"] == "Counterfactual Action Guidance"
    assert lcg["closest_prior_primary_source"] == "https://arxiv.org/abs/2602.17659"
    assert lcg["contribution_type"] == "PRIOR_EXTENSION"
    assert lcg["proposal"] == "reports/lcg_vla/researcher_proposal.md"
    assert lcg["proposal_hash_file"] == "reports/lcg_vla/proposal_hash.txt"
    assert lcg["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg["proposal_decision"] == "LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert lcg["reviewer_attack_pending"] is False
    assert lcg["reviewer_attack_completed"] is True
    assert lcg["reviewer_attack"] == "reports/lcg_vla/reviewer_attack.md"
    assert lcg["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert lcg["researcher_rebuttal"] == "reports/lcg_vla/researcher_rebuttal.md"
    assert lcg["rebuttal_decision"] == "LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lcg["accepted_reviewer_conditions"] is True
    assert lcg["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg["math_audit_decision"] == "LCG_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lcg["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg["preregistration_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg["stage_0_implementation_pending"] is False
    assert lcg["stage_0_implementation_validated"] is True
    assert lcg["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg["stage_0_launch_allowed_next"] is True
    assert lcg["policy_order"] == [
        "smolvla_base",
        "counterfactual_action_guidance_proxy",
        "lcg_full",
        "lcg_no_language_contrast_ablation",
        "standard_lora",
    ]
    assert lcg["standard_lora_required"] is True
    assert lcg["first_serious_comparison_includes_closest_prior"] is True
    assert lcg["training_happened"] is False
    assert lcg["validation_search_happened"] is False
    assert lcg["closed_loop_experiment_happened"] is False
    assert lcg["confirmatory_test_tuning_happened"] is False
    lcg_proposal = state["epoch_4_cycle_32_lcg_researcher_proposal"]
    assert lcg_proposal["final_decision"] == "LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED"
    assert lcg_proposal["proposal"] == "reports/lcg_vla/researcher_proposal.md"
    assert lcg_proposal["proposal_hash_file"] == "reports/lcg_vla/proposal_hash.txt"
    assert lcg_proposal["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_proposal["closest_prior"] == "Counterfactual Action Guidance"
    assert lcg_proposal["reviewer_attack"] == "reports/lcg_vla/reviewer_attack.md"
    assert lcg_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert lcg_proposal["researcher_rebuttal"] == "reports/lcg_vla/researcher_rebuttal.md"
    assert lcg_proposal["rebuttal_decision"] == "LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lcg_proposal["accepted_reviewer_conditions"] is True
    assert lcg_proposal["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg_proposal["math_audit_decision"] == "LCG_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lcg_proposal["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg_proposal["preregistration_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg_proposal["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_proposal["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_proposal["stage_0_implementation_pending"] is False
    assert lcg_proposal["stage_0_implementation_validated"] is True
    assert lcg_proposal["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_proposal["stage_0_launch_allowed_next"] is True
    assert lcg_proposal["policy_order"] == lcg["policy_order"]
    assert lcg_proposal["training_happened"] is False
    assert lcg_proposal["validation_search_happened"] is False
    assert lcg_proposal["closed_loop_experiment_happened"] is False
    assert lcg_proposal["confirmatory_test_tuning_happened"] is False
    assert lcg_proposal["reviewer_attack_pending"] is False
    assert lcg_proposal["reviewer_attack_completed"] is True
    lcg_review = state["epoch_4_cycle_32_lcg_reviewer_attack"]
    assert lcg_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert lcg_review["reviewer_attack"] == "reports/lcg_vla/reviewer_attack.md"
    assert lcg_review["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_review["closest_prior"] == "Counterfactual Action Guidance"
    assert len(lcg_review["conditions"]) == 10
    assert lcg_review["training_happened"] is False
    assert lcg_review["validation_search_happened"] is False
    assert lcg_review["closed_loop_experiment_happened"] is False
    assert lcg_review["confirmatory_test_tuning_happened"] is False
    assert lcg_review["researcher_rebuttal"] == "reports/lcg_vla/researcher_rebuttal.md"
    assert lcg_review["rebuttal_decision"] == "LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lcg_review["accepted_reviewer_conditions"] is True
    assert lcg_review["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg_review["math_audit_decision"] == "LCG_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lcg_review["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg_review["preregistration_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg_review["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_review["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_review["stage_0_implementation_pending"] is False
    assert lcg_review["stage_0_implementation_validated"] is True
    assert lcg_review["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_review["stage_0_launch_allowed_next"] is True
    lcg_rebuttal = state["epoch_4_cycle_32_lcg_rebuttal"]
    assert lcg_rebuttal["final_decision"] == "LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert lcg_rebuttal["researcher_rebuttal"] == "reports/lcg_vla/researcher_rebuttal.md"
    assert lcg_rebuttal["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_rebuttal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert lcg_rebuttal["accepted_reviewer_conditions"] is True
    assert lcg_rebuttal["closest_prior"] == "Counterfactual Action Guidance"
    assert lcg_rebuttal["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg_rebuttal["math_audit_decision"] == "LCG_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lcg_rebuttal["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg_rebuttal["preregistration_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg_rebuttal["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_rebuttal["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_rebuttal["stage_0_implementation_pending"] is False
    assert lcg_rebuttal["stage_0_implementation_validated"] is True
    assert lcg_rebuttal["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_rebuttal["stage_0_launch_allowed_next"] is True
    assert lcg_rebuttal["training_happened"] is False
    assert lcg_rebuttal["validation_search_happened"] is False
    assert lcg_rebuttal["closed_loop_experiment_happened"] is False
    assert lcg_rebuttal["confirmatory_test_tuning_happened"] is False
    lcg_audit = state["epoch_4_cycle_32_lcg_mathematical_audit"]
    assert lcg_audit["final_decision"] == "LCG_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert lcg_audit["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg_audit["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_audit["null_instruction"] == ""
    assert lcg_audit["horizon"] == 50
    assert lcg_audit["action_dim"] == 7
    assert lcg_audit["tau_lang"] == 0.25
    assert lcg_audit["kl_between_deterministic_actions_used"] is False
    assert "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in lcg_audit["stage_0_stop_classes"]
    assert lcg_audit["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg_audit["preregistration_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg_audit["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_audit["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_audit["stage_0_implementation_pending"] is False
    assert lcg_audit["stage_0_implementation_validated"] is True
    assert lcg_audit["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_audit["stage_0_launch_allowed_next"] is True
    lcg_prereg = state["epoch_4_cycle_32_lcg_preregistration"]
    assert lcg_prereg["final_decision"] == "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert lcg_prereg["preregistration"] == "reports/lcg_vla/preregistration.md"
    assert lcg_prereg["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_prereg["mathematical_audit"] == "reports/lcg_vla/mathematical_mechanism_audit.md"
    assert lcg_prereg["development_tasks"] == [
        "libero_spatial/task_3",
        "libero_object/task_3",
        "libero_goal/task_5",
        "libero_10/task_5",
    ]
    assert lcg_prereg["discovery_demo_ids"] == "0..7"
    assert lcg_prereg["validation_demo_ids"] == "8..9"
    assert lcg_prereg["confirmatory_identities_touched"] is False
    assert "reports/lcg_vla/stage_0_result.json" in lcg_prereg["stage_0_artifacts"]
    assert "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in lcg_prereg["stage_0_stop_classes"]
    assert lcg_prereg["bounded_validation_search_max_configs"] == 6
    assert lcg_prereg["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_prereg["prototype_protocol_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_prereg["stage_0_implementation_pending"] is False
    assert lcg_prereg["stage_0_implementation_validated"] is True
    assert lcg_prereg["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_prereg["stage_0_launch_allowed_next"] is True
    lcg_protocol = state["epoch_4_cycle_32_lcg_prototype_protocol"]
    assert lcg_protocol["final_decision"] == "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert lcg_protocol["prototype_protocol"] == "reports/lcg_vla/prototype_protocol.md"
    assert lcg_protocol["proposal_hash"] == LCG_PROPOSAL_HASH
    assert lcg_protocol["helper_module"] == "tca_map/smolvla/lcg_vla.py"
    assert lcg_protocol["runner"] == "scripts/run_lcg_vla_stage0.py"
    assert lcg_protocol["unit_tests"] == "tests/test_lcg_vla.py"
    assert lcg_protocol["serializer_preflight"] == "reports/lcg_vla/stage_0_serializer_preflight.json"
    assert lcg_protocol["stage_0_implementation_pending"] is False
    assert lcg_protocol["stage_0_implementation_validated"] is True
    assert lcg_protocol["implementation_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_protocol["stage_0_launch_allowed_next"] is True
    assert lcg_protocol["serializer_preflight_passed"] is True
    assert lcg_protocol["serializer_preflight_fixture_hash"] == lcg_protocol["serializer_preflight_reproduced_hash"]
    lcg_implementation = state["epoch_4_cycle_32_lcg_stage_0_implementation"]
    assert lcg_implementation["final_decision"] == "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY"
    assert lcg_implementation["helper_module"] == "tca_map/smolvla/lcg_vla.py"
    assert lcg_implementation["runner"] == "scripts/run_lcg_vla_stage0.py"
    assert lcg_implementation["unit_tests"] == "tests/test_lcg_vla.py"
    assert lcg_implementation["compile_passed"] is True
    assert lcg_implementation["focused_test_result"] == "6 passed"
    assert lcg_implementation["serializer_preflight"] == "reports/lcg_vla/stage_0_serializer_preflight.json"
    assert lcg_implementation["serializer_preflight_passed"] is True
    assert lcg_implementation["serializer_preflight_fixture_hash"] == lcg_implementation["serializer_preflight_reproduced_hash"]
    assert lcg_implementation["stage_0_launch_allowed_next"] is True
    assert lcg_implementation["training_happened"] is False
    assert lcg_implementation["validation_search_happened"] is False
    assert lcg_implementation["closed_loop_experiment_happened"] is False
    assert lcg_implementation["confirmatory_test_tuning_happened"] is False
    lcg_outcome = state["epoch_4_cycle_32_lcg_stage_0_outcome"]
    assert lcg_outcome["final_decision"] == "LCG_STAGE_0_DESIGN_FAILURE"
    assert lcg_outcome["completed_model_row_count"] == lcg_outcome["planned_model_row_count"] == 5120
    assert lcg_outcome["exception_count"] == 0
    assert lcg_outcome["duplicate_manifest_key_count"] == 0
    assert lcg_outcome["duplicate_partial_key_count"] == 0
    assert lcg_outcome["missing_manifest_key_count"] == 0
    assert lcg_outcome["extra_partial_key_count"] == 0
    assert lcg_outcome["split_overlap_key_count"] == 0
    assert lcg_outcome["key_sets_equal"] is True
    assert lcg_outcome["gate_activation_fraction"] == 0.99978125
    assert lcg_outcome["lora_explains"] is True
    assert lcg_outcome["bounded_validation_allowed"] is False
    assert lcg_outcome["valid_scientific_result"] is False
    assert lcg_outcome["closed_loop_experiment_happened"] is False
    assert lcg_outcome["lcg_rescue_allowed"] is False
    cycle33 = state["epoch_4_cycle_33_candidate_search"]
    assert cycle33["candidate_search_pending"] is False
    assert cycle33["candidate_count_required"] == 3
    assert cycle33["candidate_count_generated"] == 3
    assert cycle33["previous_method"] == "LCG-VLA"
    assert cycle33["previous_decision"] == "LCG_STAGE_0_DESIGN_FAILURE"
    assert cycle33["previous_stage_0_result"] == "reports/lcg_vla/stage_0_result.json"
    assert cycle33["prior_mechanism_map"] == "reports/epoch_4_cycle_33_prior_mechanism_map.md"
    assert cycle33["candidate_generation"] == "reports/epoch_4_cycle_33_candidate_generation.md"
    assert cycle33["candidate_ids"] == ["AFID-VLA", "ACR-VLA", "GCF-VLA"]
    assert cycle33["selected_method"] == "AFID-VLA"
    assert cycle33["selected_score"] == 90
    assert cycle33["selection_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert cycle33["lcg_repair_allowed"] is False
    assert cycle33["lcg_rescue_allowed"] is False
    afid = state["epoch_4_cycle_33_candidate_selection"]
    assert afid["final_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid["method"] == "AFID-VLA"
    assert afid["candidate_count"] == 3
    assert afid["selected_score"] == 90
    assert afid["closest_prior"] == "FineVLA"
    assert afid["closest_prior_primary_source"] == "https://arxiv.org/html/2605.27284v1"
    assert afid["researcher_proposal"] == "reports/afid_vla/researcher_proposal.md"
    assert afid["proposal_hash_file"] == "reports/afid_vla/proposal_hash.txt"
    assert afid["proposal_hash"] == AFID_PROPOSAL_HASH
    assert afid["proposal_frozen"] is True
    assert afid["reviewer_attack"] == "reports/afid_vla/reviewer_attack.md"
    assert afid["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert afid["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert afid["rebuttal_decision"] == "AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert afid["rebuttal_completed"] is True
    assert afid["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert afid["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid["preregistration_pending"] is True
    assert afid["prototype_protocol_pending"] is True
    assert afid["policy_order"] == [
        "smolvla_base",
        "finevla_action_factor_proxy",
        "afid_full",
        "afid_no_factor_ablation",
        "standard_lora",
    ]
    assert afid["first_serious_comparison_includes_closest_prior"] is True
    assert afid["training_happened"] is False
    assert afid["confirmatory_test_tuning_happened"] is False
    afid_proposal = state["epoch_4_cycle_33_afid_researcher_proposal"]
    assert afid_proposal["final_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_proposal["researcher_proposal"] == "reports/afid_vla/researcher_proposal.md"
    assert afid_proposal["proposal_hash"] == AFID_PROPOSAL_HASH
    assert afid_proposal["reviewer_attack"] == "reports/afid_vla/reviewer_attack.md"
    assert afid_proposal["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert afid_proposal["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert afid_proposal["rebuttal_decision"] == "AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert afid_proposal["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_proposal["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert afid_proposal["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_proposal["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_proposal["closest_prior"] == "FineVLA"
    assert afid_proposal["policy_order"] == [
        "smolvla_base",
        "finevla_action_factor_proxy",
        "afid_full",
        "afid_no_factor_ablation",
        "standard_lora",
    ]
    assert afid_proposal["reviewer_attack_completed"] is True
    assert afid_proposal["rebuttal_completed"] is True
    assert afid_proposal["preregistration_pending"] is True
    assert afid_proposal["prototype_protocol_pending"] is True
    assert afid_proposal["training_happened"] is False
    afid_review = state["epoch_4_cycle_33_afid_reviewer_attack"]
    assert afid_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert afid_review["reviewer_attack"] == "reports/afid_vla/reviewer_attack.md"
    assert afid_review["proposal_hash"] == AFID_PROPOSAL_HASH
    assert afid_review["closest_prior"] == "FineVLA"
    assert afid_review["policy_order"] == [
        "smolvla_base",
        "finevla_action_factor_proxy",
        "afid_full",
        "afid_no_factor_ablation",
        "standard_lora",
    ]
    assert afid_review["rebuttal_pending"] is True
    assert afid_review["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert afid_review["rebuttal_decision"] == "AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert afid_review["rebuttal_completed"] is True
    assert afid_review["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_review["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert afid_review["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_review["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_review["preregistration_pending"] is True
    assert afid_review["prototype_protocol_pending"] is True
    assert afid_review["training_happened"] is False
    afid_rebuttal = state["epoch_4_cycle_33_afid_rebuttal"]
    assert afid_rebuttal["final_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_rebuttal["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert afid_rebuttal["accepted_reviewer_conditions"] is True
    assert afid_rebuttal["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_rebuttal["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert afid_rebuttal["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_rebuttal["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_rebuttal["preregistration_pending"] is True
    assert afid_rebuttal["prototype_protocol_pending"] is True
    assert afid_rebuttal["training_happened"] is False
    afid_audit = state["epoch_4_cycle_33_afid_mathematical_audit"]
    assert afid_audit["final_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_audit["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_audit["proposal_hash"] == AFID_PROPOSAL_HASH
    assert afid_audit["horizon"] == 50
    assert afid_audit["action_dim"] == 7
    assert afid_audit["tau_conf"] == 0.60
    assert afid_audit["kl_between_deterministic_actions_used"] is False
    assert "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in afid_audit["stage_0_stop_classes"]
    assert afid_audit["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_audit["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_audit["preregistration_pending"] is True
    assert afid_audit["prototype_protocol_pending"] is True
    assert afid_audit["training_happened"] is False
    afid_prereg = state["epoch_4_cycle_33_afid_preregistration"]
    assert afid_prereg["final_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_prereg["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_prereg["development_tasks"] == [
        "libero_spatial/task_3",
        "libero_object/task_3",
        "libero_goal/task_5",
        "libero_10/task_5",
    ]
    assert afid_prereg["discovery_demo_ids"] == "0..7"
    assert afid_prereg["validation_demo_ids"] == "8..9"
    assert afid_prereg["confirmatory_identities_touched"] is False
    assert afid_prereg["bounded_validation_search_max_configs"] == 6
    assert afid_prereg["resume_key"] == "(split, task_suite, task_id, demo_id, window_start, factor_key, policy)"
    assert afid_prereg["prototype_protocol_pending"] is True
    assert "epoch_4_cycle_33_afid_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_prototype_protocol_pending" in state["completed_stages"]
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
