import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
PESA_PROPOSAL_HASH = "B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63"
EAC_PROPOSAL_HASH = "A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E"
G3P_PROPOSAL_HASH = "BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71"
CALA_PROPOSAL_HASH = "5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76"
RAR_PROPOSAL_HASH = "723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56"
COVI_PROPOSAL_HASH = "338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621"
LIFT_PROPOSAL_HASH = "3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F"
FAMR_PROPOSAL_HASH = "96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69"
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
BRID_PROPOSAL_HASH = "2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2"


def test_active_campaign_final_decision_is_nonterminal_pivot() -> None:
    final = (REPORTS / "autonomous_until_paper_final_decision.md").read_text(encoding="utf-8")

    assert "BRID_REVIEWER_ATTACK_COMPLETED_REBUTTAL_PENDING" in final
    assert "BRID-VLA" in final
    assert "Base-Residual Implicit Diffusion" in final
    assert "Diffusion Policy" in final
    assert "reports/epoch_4_cycle_34_prior_mechanism_map.md" in final
    assert "reports/epoch_4_cycle_34_candidate_generation.md" in final
    assert "reports/brid_vla/researcher_proposal.md" in final
    assert "reports/brid_vla/reviewer_attack.md" in final
    assert BRID_PROPOSAL_HASH in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "diffusion_policy_action_chunk_proxy" in final
    assert "brid_no_base_residual_ablation" in final
    assert "epoch_4_cycle_34_brid_rebuttal_pending" in final
    assert "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE" in final
    assert "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING" in final
    assert "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "AFID_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "AFID_REVIEWER_ATTACK_COMPLETED_REBUTTAL_PENDING" not in final
    assert "AFID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING" in final
    assert "AFID-VLA" in final
    assert "Action-Factor Instruction Densification" in final
    assert "FineVLA" in final
    assert "reports/epoch_4_cycle_33_prior_mechanism_map.md" in final
    assert "reports/epoch_4_cycle_33_candidate_generation.md" in final
    assert "reports/afid_vla/researcher_proposal.md" in final
    assert "reports/afid_vla/reviewer_attack.md" in final
    assert "reports/afid_vla/researcher_rebuttal.md" in final
    assert "reports/afid_vla/mathematical_mechanism_audit.md" in final
    assert "reports/afid_vla/preregistration.md" in final
    assert "reports/afid_vla/prototype_protocol.md" in final
    assert AFID_PROPOSAL_HASH in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "finevla_action_factor_proxy" in final
    assert "afid_no_factor_ablation" in final
    assert "reports/afid_vla/stage_0_serializer_preflight.json" in final
    assert "reports/afid_vla/stage_0_result.json" in final
    assert "reports/afid_vla/stage_0_adjudication.md" in final
    assert "LCG_STAGE_0_DESIGN_FAILURE" in final
    assert "LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY" in final
    assert "LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING" in final
    assert "LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "LCG_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING" in final
    assert "LCG-VLA" in final
    assert "Language-Contrastive Guidance" in final
    assert "Counterfactual Action" in final
    assert "Guidance" in final
    assert "counterfactual_action_guidance_proxy" in final
    assert "lcg_no_language_contrast_ablation" in final
    assert "reports/lcg_vla/researcher_proposal.md" in final
    assert "reports/lcg_vla/reviewer_attack.md" in final
    assert "reports/lcg_vla/researcher_rebuttal.md" in final
    assert "reports/lcg_vla/mathematical_mechanism_audit.md" in final
    assert "reports/lcg_vla/preregistration.md" in final
    assert "reports/lcg_vla/prototype_protocol.md" in final
    assert LCG_PROPOSAL_HASH in final
    assert "S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE" in final
    assert "S2C_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY" in final
    assert "S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING" in final
    assert "S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "S2C_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED" in final
    assert "S2C-VLA" in final
    assert "ChunkFlow" in final
    assert "reports/s2c_vla/reviewer_attack.md" in final
    assert "reports/s2c_vla/researcher_rebuttal.md" in final
    assert "reports/s2c_vla/mathematical_mechanism_audit.md" in final
    assert "reports/s2c_vla/preregistration.md" in final
    assert "reports/s2c_vla/prototype_protocol.md" in final
    assert S2C_PROPOSAL_HASH in final
    assert "URF_STAGE_0_NO_USABLE_HEADROOM" in final
    assert "URF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY" in final
    assert "URF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING" in final
    assert "URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "URF_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING" in final
    assert "URF-VLA" in final
    assert "SUREFlow" in final
    assert "sureflow_uncertainty_residual_proxy" in final
    assert URF_PROPOSAL_HASH in final
    assert "epoch_4_cycle_31_candidate_search_pending" in final
    assert "reports/urf_vla/reviewer_attack.md" in final
    assert "reports/urf_vla/researcher_rebuttal.md" in final
    assert "reports/urf_vla/mathematical_mechanism_audit.md" in final
    assert "reports/urf_vla/preregistration.md" in final
    assert "reports/urf_vla/prototype_protocol.md" in final
    assert "Guided Action Flow" in final
    assert "CCIF_STAGE_0_DESIGN_FAILURE" in final
    assert "CCIF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY" in final
    assert "CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING" in final
    assert "CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "CCIF_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING" in final
    assert "CCIF-VLA" in final
    assert "Coarse-to-Control" in final
    assert "coarse_to_control_continuous_proxy" in final
    assert CCIF_PROPOSAL_HASH in final
    assert "TSC_STAGE_0_NO_USABLE_HEADROOM" in final
    assert "TSC-VLA" in final
    assert "TS-Mask VLA" in final
    assert "ts_mask_continuous_proxy" in final
    assert TSC_PROPOSAL_HASH in final
    assert "epoch_4_cycle_29_candidate_search_pending" in final
    assert "SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL" in final
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
    assert "CFR_STAGE_0_NO_USABLE_HEADROOM" in final
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
    assert "reports/rar_vla/preregistration.md" in final
    assert "reports/rar_vla/prototype_protocol.md" in final
    assert "RAR Stage 0 is complete" in final
    assert "-0.03837609884238533" in final
    assert "zero_residual" in final
    assert "Validation search, training, Stage A manifest freeze, and rollout are disallowed for this RAR formulation" in final
    assert "Epoch 4 Cycle 14 generated exactly three post-RAR candidates" in final
    assert "COVI-VLA" in final
    assert "LIBERO-Occ / Viewpoint Imagination" in final
    assert "random_cutout_clean_retention_baseline" in final
    assert "reports/covi_vla/researcher_proposal.md" in final
    assert COVI_PROPOSAL_HASH in final
    assert "reports/covi_vla/reviewer_attack.md" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "reports/covi_vla/researcher_rebuttal.md" in final
    assert "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "reports/covi_vla/mathematical_mechanism_audit.md" in final
    assert "COVI_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "RAP-VLA" in final
    assert "AMP-VLA" in final
    assert "ABot-M0" in final
    assert "AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT" in final
    assert "AMP_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "reports/amp_vla/mathematical_mechanism_audit.md" in final
    assert "AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "reports/amp_vla/preregistration.md" in final
    assert "AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING" in final
    assert "reports/amp_vla/prototype_protocol.md" in final
    assert AMP_PROPOSAL_HASH in final
    assert "RAP_MATHEMATICAL_AUDIT_PREREGISTERED" in final
    assert "RAP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING" in final
    assert "RAP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING" in final
    assert "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE" in final
    assert "OptimusVLA" in final
    assert "AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE" in final
    assert "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED" in final
    assert "CFR-VLA" in final
    assert "DFM-VLA" in final
    assert "Continuous Full-Chunk Refinement" in final
    assert "dfm_vla_continuous_refinement_proxy" in final
    assert CFR_PROPOSAL_HASH in final
    assert "epoch_4_cycle_29_candidate_search_pending" in final
    assert "1280 /" in final
    assert "base_action_in_bounds = false" in final
    assert "640 / 640" in final
    assert "-6.04941221711208" in final
    assert "-6.068176722319228" in final
    assert "optimusvla_memory_prior_proxy" in final
    assert RAP_PROPOSAL_HASH in final
    assert "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE" in final
    assert "1536 / 1536" in final
    assert "VDR-VLA" in final
    assert VDR_PROPOSAL_HASH in final
    assert "FutureVLA" in final
    assert "KITE-VLA" in final
    assert KITE_PROPOSAL_HASH in final
    assert "GeoPredict" in final
    assert "cumulative_action_target" in final
    assert "LIFT-VLA" in final
    assert LIFT_PROPOSAL_HASH in final
    assert "training-free CAG" in final
    assert "last-step-only LIFT" in final
    assert "reports/covi_vla/preregistration.md" in final
    assert "reports/covi_vla/prototype_protocol.md" in final
    assert "APPROVE_WITH_FIXED_EMPIRICAL_RISKS" in final
    assert "runs/marc_vla_stage_a/20260714T171356Z" in final


def test_active_campaign_state_records_governance_v2() -> None:
    state = json.loads((REPORTS / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))

    assert state["governance_file"] == "reports/current_research_governance.md"
    assert state["current_decision"] == "BRID_REVIEWER_ATTACK_COMPLETED_REBUTTAL_PENDING"
    assert state["current_epoch"] == 4
    assert state["current_cycle"] == 34
    assert state["current_stage"] == "epoch_4_cycle_34_brid_rebuttal_pending"
    assert state["method"] == "BRID-VLA"
    assert state["method_identity"] == "BRID-VLA"
    assert state["next_action"] == "Write the BRID-VLA Researcher A rebuttal before mathematical audit."
    assert state["proposal_hash"] == BRID_PROPOSAL_HASH
    assert state["proposal_hash_file"] == "reports/brid_vla/proposal_hash.txt"
    assert state["researcher_proposal"] == "reports/brid_vla/researcher_proposal.md"
    assert state["reviewer_attack"] == "reports/brid_vla/reviewer_attack.md"
    assert state["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert state["researcher_rebuttal"] == "reports/brid_vla/researcher_rebuttal.md"
    assert state["rebuttal_decision"] == "BRID_REBUTTAL_PENDING"
    assert state["mathematical_audit"] is None
    assert state["math_audit_decision"] is None
    assert state["preregistration"] is None
    assert state["preregistration_decision"] is None
    assert state["prototype_protocol"] is None
    assert state["prototype_protocol_decision"] is None
    assert state["stage_0_serializer_preflight"] == "reports/afid_vla/stage_0_serializer_preflight.json"
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
    assert (
        state["next_action"]
        == "Write the BRID-VLA Researcher A rebuttal before mathematical audit."
    )
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
    assert rap_proposal["proposal_hash"] == RAP_PROPOSAL_HASH
    rap_review = state["epoch_4_cycle_25_rap_reviewer_attack"]
    assert rap_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert rap_review["proposal_hash"] == RAP_PROPOSAL_HASH
    rap_rebuttal = state["epoch_4_cycle_25_rap_rebuttal"]
    assert rap_rebuttal["final_decision"] == "RAP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert rap_rebuttal["accepted_reviewer_conditions"] is True
    rap_math = state["epoch_4_cycle_25_rap_mathematical_audit"]
    assert rap_math["final_decision"] == "RAP_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert rap_math["kl_between_deterministic_actions_used"] is False
    rap_preregistration = state["epoch_4_cycle_25_rap_preregistration"]
    assert rap_preregistration["final_decision"] == "RAP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert rap_preregistration["preregistration"] == "reports/rap_vla/preregistration.md"
    assert rap_preregistration["stage_0_allowed_next"] is True
    rap_protocol = state["epoch_4_cycle_25_rap_prototype_protocol"]
    assert rap_protocol["final_decision"] == "RAP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert rap_protocol["prototype_protocol"] == "reports/rap_vla/prototype_protocol.md"
    assert rap_protocol["stage_0_allowed_next"] is True
    assert rap_protocol["stage_0_completed"] is True
    assert rap_protocol["stage_0_decision"] == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    rap_outcome = state["epoch_4_cycle_25_rap_stage_0_outcome"]
    assert rap_outcome["final_decision"] == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert rap_outcome["completed_model_row_count"] == 640
    assert rap_outcome["planned_model_row_count"] == 640
    assert rap_outcome["exception_count"] == 0
    assert rap_outcome["duplicate_partial_key_count"] == 0
    assert rap_outcome["missing_manifest_key_count"] == 0
    assert rap_outcome["extra_partial_key_count"] == 0
    assert rap_outcome["split_overlap_key_count"] == 0
    assert rap_outcome["action_validity_ok"] is False
    assert rap_outcome["base_action_in_bounds"] is False
    assert rap_outcome["official_prior_policy_2_label"] == "optimusvla_memory_prior_proxy"
    cycle26 = state["epoch_4_cycle_26_candidate_search"]
    assert cycle26["candidate_search_pending"] is False
    assert cycle26["candidate_count_required"] == 3
    assert cycle26["candidate_count_generated"] == 3
    assert cycle26["selected_method"] == "AMP-VLA"
    assert cycle26["selection_decision"] == "AMP_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    amp = state["epoch_4_cycle_26_candidate_selection"]
    assert amp["candidate_count"] == 3
    assert amp["selected_score"] == 95
    assert amp["method"] == "AMP-VLA"
    assert amp["closest_prior"] == "ABot-M0"
    assert amp["policy_order"] == [
        "smolvla_base",
        "abot_m0_action_manifold_proxy",
        "amp_full",
        "amp_no_manifold_projection",
        "standard_lora",
    ]
    assert amp["standard_lora_required"] is True
    assert amp["proposal_hash"] == AMP_PROPOSAL_HASH
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
    assert amp_proposal["proposal"] == "reports/amp_vla/researcher_proposal.md"
    amp_review = state["epoch_4_cycle_26_amp_reviewer_attack"]
    assert amp_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert amp_review["proposal_hash"] == AMP_PROPOSAL_HASH
    amp_rebuttal = state["epoch_4_cycle_26_amp_rebuttal"]
    assert amp_rebuttal["final_decision"] == "AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert amp_rebuttal["accepted_reviewer_conditions"] is True
    amp_math = state["epoch_4_cycle_26_amp_mathematical_audit"]
    assert amp_math["final_decision"] == "AMP_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert amp_math["kl_between_deterministic_actions_used"] is False
    amp_prereg = state["epoch_4_cycle_26_amp_preregistration"]
    assert amp_prereg["final_decision"] == "AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert amp_prereg["stage_0_allowed_next"] is True
    amp_protocol = state["epoch_4_cycle_26_amp_prototype_protocol"]
    assert amp_protocol["final_decision"] == "AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
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
    assert amp_outcome["action_validity_ok"] is False
    assert amp_outcome["base_action_in_bounds"] is False
    assert amp_outcome["bounded_validation_allowed"] is False
    cycle27 = state["epoch_4_cycle_27_candidate_search"]
    assert cycle27["candidate_search_pending"] is False
    assert cycle27["candidate_count_required"] == 3
    assert cycle27["candidate_count_generated"] == 3
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
    assert "official action-validity semantics must be defined before Stage 0 and shared across policies" in cfr_review["conditions"]
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
    assert cfr_outcome["duplicate_manifest_key_count"] == 0
    assert cfr_outcome["duplicate_partial_key_count"] == 0
    assert cfr_outcome["missing_manifest_key_count"] == 0
    assert cfr_outcome["extra_partial_key_count"] == 0
    assert cfr_outcome["split_overlap_key_count"] == 0
    assert cfr_outcome["key_sets_equal"] is True
    assert cfr_outcome["official_prior_policy_2_label"] == "dfm_vla_continuous_refinement_proxy"
    assert cfr_outcome["closed_loop_experiment_happened"] is False
    assert cfr_outcome["simulator_load_count"] == 0
    assert cfr_outcome["confirmatory_records_read"] == 0
    assert cfr_outcome["training_happened"] is False
    assert cfr_outcome["validation_search_happened"] is False
    assert cfr_outcome["residual_probe_relative_improvement"] == -6.04941221711208
    assert cfr_outcome["dfm_proxy_headroom_relative_improvement"] == -6.068176722319228
    assert cfr_outcome["action_validity_ok"] is True
    assert cfr_outcome["base_action_valid_under_official_semantics"] is True
    assert cfr_outcome["checkpoint_reload_ok"] is True
    assert cfr_outcome["finite_objectives_and_gradients"] is True
    assert cfr_outcome["cfr_gradient_nonzero"] is True
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
    assert cycle28["selection_decision"] == "TSC_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    tsc = state["epoch_4_cycle_28_candidate_selection"]
    assert tsc["candidate_count"] == 3
    assert tsc["selected_score"] == 91
    assert tsc["method"] == "TSC-VLA"
    assert tsc["closest_prior"] == "TS-Mask VLA"
    assert tsc["closest_prior_primary_source"] == "https://arxiv.org/abs/2607.09818"
    assert tsc["policy_order"] == [
        "smolvla_base",
        "ts_mask_continuous_proxy_or_official_ts_mask_vla_if_installed",
        "tsc_full",
        "tsc_no_targeted_mask_ablation",
        "standard_lora",
    ]
    assert tsc["standard_lora_required"] is True
    assert tsc["training_happened"] is False
    assert tsc["validation_search_happened"] is False
    assert tsc["closed_loop_experiment_happened"] is False
    assert tsc["confirmatory_test_tuning_happened"] is False
    assert tsc["first_serious_comparison_includes_closest_prior"] is True
    assert tsc["cfr_rescue_allowed"] is False
    assert tsc["proposal"] == "reports/tsc_vla/researcher_proposal.md"
    assert tsc["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc["proposal_hash_file"] == "reports/tsc_vla/proposal_hash.txt"
    assert tsc["proposal_decision"] == "TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    tsc_proposal = state["epoch_4_cycle_28_tsc_researcher_proposal"]
    assert tsc_proposal["final_decision"] == "TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert tsc_proposal["proposal"] == "reports/tsc_vla/researcher_proposal.md"
    assert tsc_proposal["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_proposal["closest_prior"] == "TS-Mask VLA"
    tsc_review = state["epoch_4_cycle_28_tsc_reviewer_attack"]
    assert tsc_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert tsc_review["reviewer_attack"] == "reports/tsc_vla/reviewer_attack.md"
    assert tsc_review["proposal_hash"] == TSC_PROPOSAL_HASH
    tsc_rebuttal = state["epoch_4_cycle_28_tsc_rebuttal"]
    assert tsc_rebuttal["final_decision"] == "TSC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert tsc_rebuttal["researcher_rebuttal"] == "reports/tsc_vla/researcher_rebuttal.md"
    assert tsc_rebuttal["accepted_reviewer_conditions"] is True
    assert tsc_rebuttal["accepted_key_ablation"] == "tsc_no_targeted_mask_ablation"
    tsc_math = state["epoch_4_cycle_28_tsc_mathematical_audit"]
    assert tsc_math["final_decision"] == "TSC_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert tsc_math["mathematical_audit"] == "reports/tsc_vla/mathematical_mechanism_audit.md"
    assert tsc_math["proposal_hash"] == TSC_PROPOSAL_HASH
    assert tsc_math["kl_between_deterministic_actions_used"] is False
    tsc_prereg = state["epoch_4_cycle_28_tsc_preregistration"]
    assert tsc_prereg["final_decision"] == "TSC_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert tsc_prereg["preregistration"] == "reports/tsc_vla/preregistration.md"
    assert tsc_prereg["stage_0_allowed_next"] is True
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
    assert cycle33["selection_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert cycle33["lcg_repair_allowed"] is False
    assert cycle33["lcg_rescue_allowed"] is False
    afid = state["epoch_4_cycle_33_candidate_selection"]
    assert afid["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
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
    assert afid["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert afid["prototype_protocol_decision"] == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    assert afid["preregistration_pending"] is True
    assert afid["prototype_protocol_pending"] is True
    assert afid["stage_0_implementation_pending"] is False
    assert afid["stage_0_implementation_validated"] is True
    assert afid["stage_0_serializer_preflight"] == "reports/afid_vla/stage_0_serializer_preflight.json"
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
    assert (
        afid_proposal["final_decision"]
        == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    )
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
    assert afid_proposal["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert (
        afid_proposal["prototype_protocol_decision"]
        == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    )
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
    assert afid_proposal["stage_0_implementation_pending"] is False
    assert afid_proposal["stage_0_implementation_validated"] is True
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
    assert afid_review["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert (
        afid_review["prototype_protocol_decision"]
        == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    )
    assert afid_review["preregistration_pending"] is True
    assert afid_review["prototype_protocol_pending"] is True
    assert afid_review["stage_0_implementation_pending"] is False
    assert afid_review["stage_0_implementation_validated"] is True
    assert afid_review["training_happened"] is False
    afid_rebuttal = state["epoch_4_cycle_33_afid_rebuttal"]
    assert afid_rebuttal["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert afid_rebuttal["researcher_rebuttal"] == "reports/afid_vla/researcher_rebuttal.md"
    assert afid_rebuttal["accepted_reviewer_conditions"] is True
    assert afid_rebuttal["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_rebuttal["math_audit_decision"] == "AFID_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert afid_rebuttal["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_rebuttal["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_rebuttal["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert (
        afid_rebuttal["prototype_protocol_decision"]
        == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    )
    assert afid_rebuttal["preregistration_pending"] is True
    assert afid_rebuttal["prototype_protocol_pending"] is True
    assert afid_rebuttal["stage_0_implementation_pending"] is False
    assert afid_rebuttal["stage_0_implementation_validated"] is True
    assert afid_rebuttal["training_happened"] is False
    afid_audit = state["epoch_4_cycle_33_afid_mathematical_audit"]
    assert afid_audit["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert afid_audit["mathematical_audit"] == "reports/afid_vla/mathematical_mechanism_audit.md"
    assert afid_audit["proposal_hash"] == AFID_PROPOSAL_HASH
    assert afid_audit["horizon"] == 50
    assert afid_audit["action_dim"] == 7
    assert afid_audit["tau_conf"] == 0.60
    assert afid_audit["kl_between_deterministic_actions_used"] is False
    assert "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION" in afid_audit["stage_0_stop_classes"]
    assert afid_audit["preregistration"] == "reports/afid_vla/preregistration.md"
    assert afid_audit["preregistration_decision"] == "AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING"
    assert afid_audit["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert (
        afid_audit["prototype_protocol_decision"]
        == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    )
    assert afid_audit["preregistration_pending"] is True
    assert afid_audit["prototype_protocol_pending"] is True
    assert afid_audit["stage_0_implementation_pending"] is False
    assert afid_audit["stage_0_implementation_validated"] is True
    assert afid_audit["training_happened"] is False
    afid_prereg = state["epoch_4_cycle_33_afid_preregistration"]
    assert afid_prereg["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
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
    assert afid_prereg["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert (
        afid_prereg["prototype_protocol_decision"]
        == "AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING"
    )
    assert afid_prereg["prototype_protocol_pending"] is True
    assert afid_prereg["stage_0_implementation_pending"] is False
    assert afid_prereg["stage_0_implementation_validated"] is True
    afid_protocol = state["epoch_4_cycle_33_afid_prototype_protocol"]
    assert afid_protocol["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert afid_protocol["prototype_protocol"] == "reports/afid_vla/prototype_protocol.md"
    assert afid_protocol["helper_module"] == "tca_map/smolvla/afid_vla.py"
    assert afid_protocol["stage_0_runner"] == "scripts/run_afid_vla_stage0.py"
    assert afid_protocol["focused_tests"] == "tests/test_afid_vla.py"
    assert afid_protocol["serializer_preflight"] == "reports/afid_vla/stage_0_serializer_preflight.json"
    assert afid_protocol["stage_0_result"] == "reports/afid_vla/stage_0_result.json"
    assert afid_protocol["stage_0_implementation_pending"] is False
    assert afid_protocol["stage_0_implementation_validated"] is True
    assert afid_protocol["training_happened"] is False
    assert afid_protocol["closed_loop_experiment_happened"] is False
    afid_implementation = state["epoch_4_cycle_33_afid_stage_0_implementation"]
    assert afid_implementation["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert afid_implementation["helper_module"] == "tca_map/smolvla/afid_vla.py"
    assert afid_implementation["stage_0_runner"] == "scripts/run_afid_vla_stage0.py"
    assert afid_implementation["focused_tests"] == "tests/test_afid_vla.py"
    assert afid_implementation["serializer_preflight"] == "reports/afid_vla/stage_0_serializer_preflight.json"
    assert afid_implementation["py_compile_passed"] is True
    assert afid_implementation["focused_tests_passed"] is True
    assert afid_implementation["combined_regression_tests_passed"] is True
    assert afid_implementation["governance_checker_passed"] is True
    assert afid_implementation["stage_0_launch_happened"] is True
    assert afid_implementation["stage_0_completed"] is True
    assert afid_implementation["stage_0_adjudicated"] is True
    assert afid_implementation["stage_0_result"] == "reports/afid_vla/stage_0_result.json"
    assert afid_implementation["stage_0_adjudication"] == "reports/afid_vla/stage_0_adjudication.md"
    assert afid_implementation["stage_0_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    afid_outcome = state["epoch_4_cycle_33_afid_stage_0_outcome"]
    assert afid_outcome["final_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert afid_outcome["completed_model_row_count"] == 5120
    assert afid_outcome["planned_model_row_count"] == 5120
    assert afid_outcome["exception_count"] == 0
    assert afid_outcome["duplicate_partial_key_count"] == 0
    assert afid_outcome["missing_manifest_key_count"] == 0
    assert afid_outcome["key_sets_equal"] is True
    assert afid_outcome["action_deltas_bounded"] is False
    assert afid_outcome["valid_scientific_result"] is False
    assert afid_outcome["closed_loop_experiment_happened"] is False
    cycle34 = state["epoch_4_cycle_34_candidate_search"]
    assert cycle34["candidate_search_pending"] is False
    assert cycle34["candidate_count_required"] == 3
    assert cycle34["candidate_count_generated"] == 3
    assert cycle34["previous_method"] == "AFID-VLA"
    assert cycle34["previous_decision"] == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    assert cycle34["previous_stage_0_result"] == "reports/afid_vla/stage_0_result.json"
    assert cycle34["prior_mechanism_map"] == "reports/epoch_4_cycle_34_prior_mechanism_map.md"
    assert cycle34["candidate_generation"] == "reports/epoch_4_cycle_34_candidate_generation.md"
    assert cycle34["candidate_ids"] == ["BRID-VLA", "FART-VLA", "RACT-VLA"]
    assert cycle34["selected_method"] == "BRID-VLA"
    assert cycle34["selected_score"] == 94
    assert cycle34["selection_decision"] == "BRID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    assert cycle34["afid_repair_allowed"] is False
    assert cycle34["afid_rescue_allowed"] is False
    selection34 = state["epoch_4_cycle_34_candidate_selection"]
    assert selection34["method"] == "BRID-VLA"
    assert selection34["final_decision"] == "BRID_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING"
    assert selection34["candidate_count"] == 3
    assert selection34["selected_score"] == 94
    assert selection34["closest_prior"] == "Diffusion Policy"
    assert selection34["proposal"] == "reports/brid_vla/researcher_proposal.md"
    assert selection34["proposal_hash_file"] == "reports/brid_vla/proposal_hash.txt"
    assert selection34["proposal_hash"] == BRID_PROPOSAL_HASH
    assert selection34["proposal_decision"] == "BRID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert selection34["reviewer_attack"] == "reports/brid_vla/reviewer_attack.md"
    assert selection34["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert selection34["rebuttal_pending"] is True
    assert selection34["policy_order"] == [
        "smolvla_base",
        "diffusion_policy_action_chunk_proxy",
        "brid_full",
        "brid_no_base_residual_ablation",
        "standard_lora",
    ]
    assert selection34["first_serious_comparison_includes_closest_prior"] is True
    assert selection34["researcher_proposal_pending"] is False
    assert selection34["researcher_proposal_frozen"] is True
    proposal34 = state["epoch_4_cycle_34_brid_researcher_proposal"]
    assert proposal34["final_decision"] == "BRID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert proposal34["proposal"] == "reports/brid_vla/researcher_proposal.md"
    assert proposal34["proposal_hash"] == BRID_PROPOSAL_HASH
    assert proposal34["closest_prior"] == "Diffusion Policy"
    assert proposal34["reviewer_attack_pending"] is False
    assert proposal34["reviewer_attack_completed"] is True
    assert proposal34["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert proposal34["researcher_proposal_frozen"] is True
    review34 = state["epoch_4_cycle_34_brid_reviewer_attack"]
    assert review34["final_decision"] == "BRID_REVIEWER_ATTACK_COMPLETED_REBUTTAL_PENDING"
    assert review34["reviewer_attack"] == "reports/brid_vla/reviewer_attack.md"
    assert review34["proposal_hash"] == BRID_PROPOSAL_HASH
    assert review34["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert review34["researcher_rebuttal_pending"] is True
    assert "epoch_4_cycle_33_afid_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_prototype_protocol_pending" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_prototype_protocol_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_stage_0_implementation_pending" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_stage_0_implementation_validated" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_stage_0_completed" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_stage_0_adjudicated" in state["completed_stages"]
    assert "epoch_4_cycle_33_afid_implementation_failure_recorded" in state["completed_stages"]
    assert "epoch_4_cycle_34_prior_mechanism_map_completed" in state["completed_stages"]
    assert "epoch_4_cycle_34_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_candidate_selected" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_researcher_proposal_pending" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_researcher_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_reviewer_attack_pending" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_34_brid_rebuttal_pending" in state["completed_stages"]
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
    assert vdr["confirmatory_test_tuning_happened"] is False
    vdr_pre = state["epoch_4_cycle_24_vdr_pre_stage_0a"]
    assert vdr_pre["final_decision"] == "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert vdr_pre["stage_0a_pending"] is False
    assert vdr_pre["horizons"] == [4, 12]
    assert vdr_pre["projection_dimension"] == 32
    assert vdr_pre["runner"] == "scripts/run_vdr_vla_stage0a.py"
    assert vdr_pre["runner_validation_decision"] == "VDR_STAGE_0A_RUNNER_VALIDATED_READY_TO_RUN"
    assert vdr_pre["runner_unit_tests_passed"] == 10
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
    hest = state["epoch_4_cycle_21_hest_stage_0a_outcome"]
    assert hest["final_decision"] == "HEST_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert hest["completed_window_count"] == 160
    assert hest["exception_count"] == 0
    assert hest["all_variant_support_valid"] is False
    assert hest["stage_0b_allowed"] is False
    haste = state["epoch_4_cycle_22_haste_pre_stage_0a"]
    assert haste["candidate_count"] == 3
    assert haste["selected_score"] == 95
    assert haste["proposal_hash"] == HASTE_PROPOSAL_HASH
    assert haste["stage_0a_pending"] is False
    outcome = state["epoch_4_cycle_22_haste_stage_0a_outcome"]
    assert outcome["final_decision"] == "HASTE_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert outcome["persisted_row_count"] == 0
    assert outcome["stage_0b_allowed"] is False
    assert outcome["rerun_allowed"] is False
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
    assert kite["runner_validation"] == "reports/kite_vla/stage_0a_runner_validation.json"
    assert kite["stage_0a_pending"] is False
    kite_outcome = state["epoch_4_cycle_23_kite_stage_0a_outcome"]
    assert kite_outcome["final_decision"] == "KITE_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert kite_outcome["completed_model_row_count"] == 128
    assert kite_outcome["resumed_model_row_count"] == 115
    assert kite_outcome["exception_count"] == 1
    assert kite_outcome["missing_manifest_key_count"] == 0
    assert kite_outcome["action_validity_ok"] is False
    assert kite_outcome["invalid_action_row_count"] == 128
    assert kite_outcome["stage_0b_allowed"] is False
    assert state["epoch_4_cycle_16_candidate_selection"]["candidate_count"] == 3
    assert state["epoch_4_cycle_16_candidate_selection"]["selected_score"] == 95
    assert state["epoch_4_cycle_16_iarc_pre_stage_0a"]["confirmatory_rows_decoded_max"] == 0
    famr = state["epoch_4_cycle_17_candidate_selection"]
    assert famr["candidate_count"] == 3
    assert famr["selected_score"] == 93
    assert famr["proposal_hash"] == FAMR_PROPOSAL_HASH
    assert famr["bounded_validation_search_max_configs"] == 6
    assert famr["policy_order"] == [
        "smolvla_base",
        "retain_scalar_proxy",
        "famr_full",
        "famr_target_only",
        "standard_lora_new_task",
    ]
    sparc = state["epoch_4_cycle_19_candidate_selection"]
    assert sparc["candidate_count"] == 3
    assert sparc["selected_score"] == 96
    assert sparc["proposal_hash"] == SPARC_PROPOSAL_HASH
    sparc_outcome = state["epoch_4_cycle_19_sparc_stage_0a_outcome"]
    assert sparc_outcome["valid_scientific_kill"] is False
    assert sparc_outcome["completed_observation_count"] == 2
    assert sparc_outcome["exception_count"] == 0
    assert sparc_outcome["duplicate_key_count"] == 0
    assert sparc_outcome["stage_0b_allowed"] is False
    assert state["resource_contention_audit_20260715"]["duplicate_key_count"] == 0
    iarc_stage_0a = state["epoch_4_cycle_16_iarc_stage_0a_outcome"]
    assert iarc_stage_0a["final_decision"] == "IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    assert iarc_stage_0a["valid_scientific_kill"] is False
    assert iarc_stage_0a["conflict_count"] == 18
    assert iarc_stage_0a["dataset_range_valid_fraction"] == 0.3
    assert iarc_stage_0a["confirmatory_observations_decoded"] == 0
    assert iarc_stage_0a["confirmatory_actions_computed"] == 0
    lift_stage_0 = state["epoch_4_cycle_15_lift_stage_0"]
    assert lift_stage_0["final_decision"] == "LIFT_COMPUTE_INFEASIBLE"
    assert lift_stage_0["manifest_rows_valid"] == 20
    assert lift_stage_0["identity_native_max_abs_error"] == 0.0
    assert lift_stage_0["action_range_valid_fraction"] == 0.8023809523809524
    assert lift_stage_0["confirmatory_policy_observations_decoded"] == 0
    assert lift_stage_0["confirmatory_policy_actions_computed"] == 0
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
    assert "epoch_4_cycle_15_candidate_generation_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_proposal_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_reviewer_attack_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_rebuttal_completed" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_mathematical_audit_preregistered" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_preregistration_frozen" in state["completed_stages"]
    assert "epoch_4_cycle_15_lift_prototype_protocol_frozen" in state["completed_stages"]
    covi = state["epoch_4_cycle_14_pre_proposal"]
    assert covi["selection_decision"] == "SELECT_COVI_VLA"
    assert covi["candidate_count"] == 3
    assert covi["closest_prior"] == "LIBERO-Occ / Viewpoint Imagination"
    assert covi["first_comparison_policies"] == [
        "frozen_smolvla_occluded",
        "vim_view_imagination_proxy",
        "covi_full",
        "covi_no_imagined_view_ablation",
        "random_cutout_clean_retention_baseline",
    ]
    assert covi["proposal"] == "reports/covi_vla/researcher_proposal.md"
    assert covi["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi["proposal_hash_file"] == "reports/covi_vla/proposal_hash.txt"
    assert covi["proposal_decision"] == "COVI_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert covi["reviewer_attack"] == "reports/covi_vla/reviewer_attack.md"
    assert covi["reviewer_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert covi["rebuttal"] == "reports/covi_vla/researcher_rebuttal.md"
    assert covi["rebuttal_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi["mathematical_audit_completed"] is True
    assert covi["preregistration_completed"] is True
    assert covi["prototype_protocol_completed"] is True
    assert covi["stage_0_pending"] is True
    covi_proposal = state["epoch_4_cycle_14_covi_proposal"]
    assert covi_proposal["final_decision"] == "COVI_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING"
    assert covi_proposal["proposal_hash"] == COVI_PROPOSAL_HASH
    assert covi_proposal["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    covi_review = state["epoch_4_cycle_14_covi_review"]
    assert covi_review["final_decision"] == "REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED"
    assert covi_review["vim_proxy_must_remain_transparent"] is True
    assert covi_review["direct_two_camera_fusion_diagnostic_required"] is True
    assert covi_review["random_cutout_simple_killer_must_remain_live"] is True
    assert covi_review["mathematical_audit_completed"] is True
    covi_rebuttal = state["epoch_4_cycle_14_covi_rebuttal"]
    assert covi_rebuttal["final_decision"] == "COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT"
    assert covi_rebuttal["accepted_no_privileged_inference"] is True
    assert covi_rebuttal["mathematical_audit_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    covi_audit = state["epoch_4_cycle_14_covi_mathematical_audit"]
    assert covi_audit["final_decision"] == "COVI_MATHEMATICAL_AUDIT_PREREGISTERED"
    assert covi_audit["mathematical_audit"] == "reports/covi_vla/mathematical_mechanism_audit.md"
    assert covi_audit["closed_loop_experiment_happened"] is False
    assert covi_audit["training_happened"] is False
    assert covi_audit["validation_search_happened"] is False
    assert covi_audit["confirmatory_test_tuning_happened"] is False
    assert covi_audit["transparent_vim_proxy_required"] is True
    assert covi_audit["direct_two_camera_fusion_diagnostic_required"] is True
    assert covi_audit["random_cutout_simple_killer_required"] is True
    assert covi_audit["bounded_validation_search_max_configs"] == 6
    assert covi_audit["preregistration_completed"] is True
    assert covi_audit["prototype_protocol_completed"] is True
    covi_prereg = state["epoch_4_cycle_14_covi_preregistration"]
    assert covi_prereg["final_decision"] == "COVI_PREREGISTRATION_FROZEN_STAGE_0_PENDING"
    assert covi_prereg["visual_token_shape_per_stream"] == [64, 960]
    assert covi_prereg["one_unresolved_check_max"] == 1
    covi_proto = state["epoch_4_cycle_14_covi_prototype_protocol"]
    assert covi_proto["final_decision"] == "COVI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING"
    assert covi_proto["stage_0_pending"] is True
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
