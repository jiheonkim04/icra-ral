# Autonomous Until Paper Decision

Date: 2026-07-15 KST

Current campaign decision: `SELECT_CALA_VLA`

This is not a terminal decision.

Active governance: `reports/current_research_governance.md`

Allowed terminal decisions:

- `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
- `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
- `HARD_EXTERNAL_BLOCKER`
- `SAFETY_RESOURCE_STOP`

## Corrected Epoch 1 Status

Cycle 1 `DICD-VLA` is archived as `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`: full `1 / 10`, direct chunk-index delay `2 / 10`, no-history ablation `1 / 10`. This is a non-GO archive and the current formulation should not be rescued, but a one-episode difference at 10 episodes per policy is not a permanent scientific family kill.

Cycle 2 `FEDO-VLA` is archived as `VALID_CURRENT_FORMULATION_KILL`: faulted full `1 / 10`, static inverse gain `2 / 10`, APEX-style proxy `2 / 10`, no-feedback ablation `2 / 10`, clean frozen `4 / 10`, clean FEDO `0 / 10`. Do not revive the current formulation.

Cycle 3 `GCAP-VLA` is archived as `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`: occluded full `3 / 10`, occluded frozen `4 / 10`, Sobel edge boost `5 / 10`, no-temporal ablation `4 / 10`, clean frozen `1 / 10`, clean GCAP `5 / 10`. Do not rescue the current formulation, but do not call the broader perception-repair family dead.

## Next Action

Cycle 1 of Epoch 2, `PTC-VLA`, is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full PTC reached `0 / 10`, frozen SmolVLA reached `3 / 10`, the task-balanced gap was `0.30`, and the mechanism was active. Do not rescue this formulation.

Cycle 2 of Epoch 2, `SACF-VLA`, is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full SACF reached `0 / 10`, frozen SmolVLA reached `7 / 10`, the task-balanced gap was `0.70`, and the semantic component was active. Do not rescue this formulation.

Cycle 3 of Epoch 2, `OCFN-VLA`, is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` total episodes with zero exceptions, `80` paired episodes per key policy, active mechanism, OCFN full `26 / 80`, zero-noise SmolVLA `27 / 80`, and paired upper confidence bound `0.0625` versus the strongest baseline. Do not rescue this formulation.

The related Epoch 2 failures have been synthesized in `reports/epoch_2_failure_synthesis.md`.

Epoch 3 Cycle 1, `CBFD-VLA`, is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: teacher acquisition passed, student training passed, Stage A completed `50 / 50` held-out episodes with zero exceptions, frozen SmolVLA reached `7 / 10`, and full CBFD reached `0 / 10` with active mechanism. Do not rescue this formulation.

Epoch 3 Cycle 2, `SCVC-VLA`, is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and paired evidence versus shifted frozen was negative.

Epoch 3 Cycle 3, `PSE-VLA`, is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full PSE reached `50 / 80`, while the strongest baseline, `bright_single`, reached `51 / 80`, and the paired bootstrap CI versus bright-single was `[-0.1000, 0.0750]`.

The related Epoch 3 failures have been synthesized in `reports/epoch_3_failure_synthesis.md`.

Epoch 4 Cycle 1, `RCV-VLA`, is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions. Full RCV reached `20 / 40`, while the no-context ablation and stateless first-action baseline each reached `24 / 40`. Full-minus-ablation paired delta was `-0.10` with CI `[-0.250, 0.025]`; full-minus-stateless paired delta was `-0.10` with CI `[-0.225, 0.025]`.

Epoch 4 Cycle 2, `CAVM-VLA`, is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions and a valid shared task/reset manifest. Full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`. Full-minus-nearest paired delta was `0.017241` with CI `[-0.068966, 0.103448]`, which is positive but below the preregistered useful-improvement bar after the only allowed expansion.

Epoch 4 Cycle 3 selected and preregistered `FANG-VLA`, an AFIL-anchored identity-preserving failure-aware action-field guidance method for frozen SmolVLA. Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`.

The development audit passed. The uncalibrated gate validation failure is archived, and the calibrated six-config validation search selected `fang_c01` with clean action validity and bounded activation. Stage A completed `50 / 50` episodes with all five policies tied at `3 / 10`, so the preregistered decision was non-catastrophic advance to Stage B.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. Full-minus-base paired delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

Epoch 4 Cycle 4 selected and preregistered `EvoState-VLA`, an EvoScene/DREAM-anchored action-evolved state guidance method. Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`.

Stage 0 stopped before rollout as `AUDIT_STOP_DESIGN_FAILURE`: the full transition model improved only `0.024689` over an actionless model, below the preregistered `0.05` threshold.

Epoch 4 Cycle 5 selected and preregistered `RAC-VLA`, a Reflective VLA-anchored frozen-policy action-consequence calibration method. Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`.

RAC Stage 0 passed without rollout: full action-consequence validation accuracy `0.585745` beat action-only `0.368496` and no-consequence `0.374483`, with margin `0.211262`; clean action delta p95 was `0.0`. The six-config validation search selected `rac_h4_a0.05` with score `0.508926`.

Stage A completed `50 / 50` episodes with zero exceptions. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. This was `STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`, not a valid Stage A kill.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared manifest. RAC full reached `1 / 40`, shifted Base reached `1 / 40`, Reflective-history proxy reached `1 / 40`, no-consequence ablation reached `2 / 40`, and online diagonal inverse-gain reached `2 / 40`. Full-minus-ablation paired delta was `-0.025` with CI `[-0.125, 0.050]`; full-minus-simple-baseline paired delta was also `-0.025` with CI `[-0.125, 0.050]`.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue or retune RAC.

The post-RAC governance update is installed and active. It requires future methods to maximize the probability of an honest paper-worthy positive result through stronger positive-prior-anchored design, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded validation search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

Epoch 4 Cycle 6 generated exactly three post-RAC candidates and selected `MTF-VLA`, Milestone-Transition Focused VLA Adaptation. Proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`.

The selected method is a FrameSkip and StructVLA anchored cross-paper synthesis. It tests whether structured milestone-transition frame selection plus frozen-base retention can improve SmolVLA adapter training beyond Base, a FrameSkip proxy, a no-retention ablation, and uniform retained-ratio LoRA. The proposal, reviewer attack, rebuttal, mathematical audit, preregistration, and prototype protocol are frozen under `reports/mtf_vla/`.

MTF Stage 0 development audit passed without training or closed-loop rollout using the official stable train/val/test prediction artifact: `1600` development records, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config and training manifest are frozen under `reports/mtf_vla/`.

Next action: train disk-reloadable selected-config adapter checkpoints for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA before any Stage A rollout.


The MTF adapter-training runner is now implemented and dry-run validated. The frozen selected manifest produces four trainable jobs: MTF full `567` events (`176` milestone + `391` retention), no-retention ablation `176`, FrameSkip proxy `176`, and uniform retained-ratio LoRA `240`, with zero train/validation/test frame overlap. This is not adapter training yet; it is the validated checkpoint-production contract.

Next action: run the MTF adapter trainer to produce and disk-reload verify all four selected-config checkpoints before any Stage A rollout.

MTF adapter training is now complete for all four trainable Stage A policies after repairing the development-only FrameSkip proxy collapse. The checkpoints are saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, disk-reloaded successfully, and summarized in `reports/mtf_vla/adapter_checkpoint_manifest.json`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for no-retention, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. No rollout or confirmatory-test tuning occurred.

The MTF Stage A manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` and has now completed as `reports/mtf_vla/stage_a_result.json`. It used exactly `frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, and `mtf_full`; `frameskip_proxy_lora` is a faithful local proxy rather than an official FrameSkip reproduction. Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; no-retention and MTF full each reached `7 / 10`. The frozen decision is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`, so Stage B is required.

The MTF Stage B manifest `reports/mtf_vla/stage_b_manifest.json` completed as `reports/mtf_vla/stage_b_result.json`: `200 / 200` official LIBERO episodes, zero exceptions, all `20` official tasks, reset seeds `20261203` and `20261204`, and the unchanged five-policy comparison. Frozen SmolVLA reached `28 / 40`, the FrameSkip proxy reached `27 / 40`, uniform retained-ratio LoRA reached `29 / 40`, the no-retention ablation reached `32 / 40`, and MTF full reached `26 / 40`.

Final MTF decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Full-minus-no-retention paired delta was `-0.15` with CI `[-0.275, -0.025]`, so the simpler ablation explains or exceeds the full method. Do not rescue or retune MTF.

Epoch 4 Cycle 7 generated exactly three post-MTF candidates in `reports/epoch_4_cycle_7_candidate_generation.md` and selected `DAGR-VLA`, a DAM-VLA anchored dynamic arm/gripper routing method. Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`.

Reviewer B attack completed with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`: DAGR is not killed before implementation, but novelty is narrowed to frozen SmolVLA identity-preserving route-gated residual adaptation, `dam_static_component_proxy` must remain a transparent local proxy, and Stage 0 must reject collapsed or unobservable route supervision before rollout.

Researcher A rebuttal completed with decision `DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The DAGR mathematical audit, preregistration, and prototype protocol are now frozen under `reports/dagr_vla/`.

DAGR Stage 0 passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `1600` development records, zero duplicate sample/frame keys, zero train/validation/test overlap, validation any-route fraction `0.865`, route-probe margins `0.0375`, `0.0725`, and `0.26`, and no hard stops.

The bounded six-config validation search selected `dagr_a020_route_mlp` as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`: residual alpha `0.20`, route architecture `mlp`, validation score `0.8571740870493018`, delta L2 p95 `0.008609326556324959`, clean delta L2 p95 `0.00672802422195673`, and action validity `1.0`.

DAGR policy identity training completed as `DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The checkpoint root is `runs/dagr_vla_checkpoints/dagr_a020_route_mlp`; `dagr_full`, `dam_static_component_proxy`, and `dagr_no_dynamic_route_ablation` all disk-reload and keep validation action validity `1.0`, while `gripper_transition_heuristic` is a saved nontrainable identity.

The DAGR Stage A manifest is frozen as `DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`: `50` planned episodes, reset seeds `20261205` and `20261206`, canonical hash `8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B`, and the unchanged five-policy comparison.

DAGR Stage A policy preflight passed as `DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D action wrappers were produced. At preflight time, no DAGR closed-loop rollout or confirmatory-test tuning had happened.

DAGR Stage A completed as `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`: `50 / 50` official LIBERO episodes, zero exceptions, frozen SmolVLA `8 / 10`, gripper-transition heuristic `7 / 10`, DAGR full `6 / 10`, no-dynamic-route ablation `5 / 10`, and DAM static proxy `2 / 10`. This is not a valid Stage A kill; freeze the DAGR Stage B matched manifest next without retuning.

The DAGR Stage B manifest froze all `20` official tasks, reset seeds `20261207` and `20261208`, `40` paired cases per policy, `200` total episodes, canonical hash `2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B`, and the unchanged five-policy comparison.

DAGR Stage B completed `200 / 200` official LIBERO episodes with zero exceptions and no confirmatory-test tuning. Frozen SmolVLA reached `28 / 40`; the DAM-style static component proxy reached `5 / 40`; DAGR full reached `18 / 40`; the no-dynamic-route ablation reached `16 / 40`; and the gripper-transition heuristic reached `24 / 40`. Full-minus-Base paired delta was `-0.25` with CI `[-0.4, -0.1]`; full-minus-gripper paired delta was `-0.15` with CI `[-0.3, 0.0]`.

Final DAGR decision: `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. This is a valid current-formulation kill because the simple gripper-transition heuristic and Base explain or exceed the full method under the frozen protocol. Do not rescue DAGR by retuning `dagr_a020_route_mlp`, changing route thresholds, changing task/reset identities, changing the policy list, or reinterpreting partial results.

Epoch 4 Cycle 8 generated exactly three post-DAGR candidates in `reports/epoch_4_cycle_8_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_8_prior_mechanism_map.md`, and selected `MARC-VLA`, Median-Anchored Regression Correction for frozen SmolVLA flow actions. Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`.

Reviewer B attack completed with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`: MARC is not killed before implementation, but novelty is narrowed against OpenVLA-OFT to frozen SmolVLA identity-preserving median-anchor correction. Researcher A rebuttal completed with decision `MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical audit, preregistration, and prototype protocol are frozen under `reports/marc_vla/`.

MARC Stage 0 passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `1600` development records, `1200` train records, `400` validation records, `1200` reserved test records not used, zero duplicate sample/frame keys, zero train/validation/test overlap, train disagreement positive fraction `0.4`, validation disagreement positive fraction `0.44`, gate-probe margin `0.0475`, initial action delta p95 `0.0`, and base action validity `1.0`.

The bounded six-config validation search selected `marc_a020_gate_mlp` as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`: correction alpha `0.20`, gate architecture `mlp`, validation score `0.5457964262366295`, gate accuracy margin `0.0525`, gate predicted-positive fraction `0.3325`, delta L2 p95 `0.011818917468190193`, clean delta L2 p95 `0.010853752493858337`, and action validity `1.0`. The linear configs stopped for collapsed gates, preserving the negative validation results.

MARC full validation action L2 was `0.08665236806523112`, the OpenVLA-OFT-style L1 proxy action L2 was `0.08763420091414227`, and full-versus-L1 mean L2 was `0.007010325323790312`. Full-versus-static mixture mean L2 was only `0.0019475044682621956`, so `static_l1_mixture_baseline` remains a live reviewer-killer for the five-policy comparison.

MARC policy identity training completed as `MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. Checkpoints are saved under `runs\marc_vla_checkpoints\marc_a020_gate_mlp`; all four trainable identities disk-reload, have validation action validity `1.0`, and preserve initial base passthrough with initial delta p95 `0.0`. MARC full delta L2 p95 is `0.010693175718188286`; the OpenVLA-OFT-style L1 proxy delta L2 p95 is `0.2307613492012024`; the static mixture delta L2 p95 is `0.07999999821186066`.

The disk-reloaded policy distinctions are no longer trivially identical: full-versus-L1 mean L2 is `0.08430124074220657`, full-versus-no-gate is `0.04372206702828407`, and full-versus-static mixture is `0.032826922833919525`. No closed-loop rollout or confirmatory-test tuning happened during policy identity training.

The MARC Stage A manifest is now frozen as `MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`: `50` planned episodes, reset seeds `20261209` and `20261210`, canonical hash `3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E`, and the unchanged five-policy comparison. `openvla_oft_l1_proxy` is explicitly a faithful transparent local proxy, not an official OpenVLA-OFT reproduction. No closed-loop rollout or confirmatory-test tuning happened during manifest freeze.

MARC Stage A preflight passed as `MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D MARC actions were produced.

MARC Stage A completed as `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE`: `50 / 50` official LIBERO episodes from `runs/marc_vla_stage_a/20260714T171356Z`, zero exceptions, frozen SmolVLA `8 / 10`, OpenVLA-OFT-style L1 proxy `0 / 10`, MARC full `0 / 10`, no-disagreement-gate ablation `7 / 10`, and static L1 mixture `7 / 10`. Full-minus-Base paired delta was `-0.8`; full-minus-no-gate was `-0.7`; full-minus-static was `-0.7`.

Final MARC decision: valid current-formulation kill. Do not rescue MARC by retuning checkpoints, changing thresholds, changing policies, changing task/reset identities, or reinterpreting Stage A outcomes.

Epoch 4 Cycle 9 generated exactly three post-MARC candidates in `reports/epoch_4_cycle_9_candidate_generation.md` and selected `PESA-VLA`, Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies. PESA is anchored to PriorVLA, LoRA-SP, and VLA-GSE. It is not a MARC rescue, action residual wrapper, milestone-frame rescue, or ActionMap mini-proxy revival.

The design-level five-policy comparison is Base, PriorVLA-style proxy, PESA full, no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline. No rollout, training, or confirmatory-test tuning has happened for PESA.

The PESA Researcher A proposal is frozen in `reports/pesa_vla/researcher_proposal.md` with proposal hash `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`.

Reviewer B attack is complete in `reports/pesa_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. PESA is not killed before implementation, but novelty is narrowed to a frozen-SmolVLA, identity-preserving prior-expert spectral adaptation combination that must beat the PriorVLA-style proxy, no-spectral/no-prior-query ablation, and one strong standard-LoRA or clean-retention simple killer.

Researcher A rebuttal is complete in `reports/pesa_vla/researcher_rebuttal.md` with decision `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. PESA proceeds only to mathematical mechanism audit, with the simple killer, PriorVLA-style proxy, no deterministic-action KL rule, and no confirmatory-test tuning commitments preserved.

The PESA mathematical mechanism audit is frozen in `reports/pesa_vla/mathematical_mechanism_audit.md` with decision `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`. The audit preserves exact Base passthrough, bounded 7D action deltas, spectral-energy masking, objective-scale checks, gradient-path checks, and the first five-policy comparison.

The PESA preregistration and prototype protocol are frozen in `reports/pesa_vla/preregistration.md` and `reports/pesa_vla/prototype_protocol.md`. The bounded validation search is capped at six named configurations, and Stage A/B must use the exact five-policy comparison.

PESA Stage 0 completed without rollout, training, manifest freeze, or confirmatory-test tuning. The development audit is saved in `reports/pesa_vla/development_audit.json`.

Final PESA Stage 0 decision: `DESIGN_FAILURE`. The hard stop was `query probe accuracy margin below minimum: -0.077500`; validation accuracy was `0.5225` versus majority `0.6`. Do not rescue PESA by retuning the query labels or thresholds.

Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`. This remains a pre-rollout design stop, not a closed-loop kill.

Epoch 4 Cycle 10 generated exactly three post-PESA candidates in `reports/epoch_4_cycle_10_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_10_prior_mechanism_map.md`, and selected `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

EAC is anchored to Adaptive Action Chunking. It preserves frozen SmolVLA weights and emitted 7D action values, changing only action-queue commitment length from deployment-observable uncertainty and queue-boundary risk. The frozen design-level five-policy comparison is Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and fixed short-replan simple killer.

The EAC Researcher A proposal is frozen in `reports/eac_vla/researcher_proposal.md` with proposal hash `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`.

Reviewer B attack is complete in `reports/eac_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It requires Researcher A to accept narrow AAC-extension novelty, keep the AAC proxy and fixed short-replan simple killer live, audit uncertainty/dispersion validity, and treat action-value modification as implementation failure.

Researcher A rebuttal is complete in `reports/eac_vla/researcher_rebuttal.md` with decision `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts the review constraints and passes only to mathematical mechanism audit, not implementation.

The EAC mathematical mechanism audit is frozen in `reports/eac_vla/mathematical_mechanism_audit.md` with decision `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines exact variables, shapes, dispersion/entropy rules, action-value passthrough, validation search limits, required ablation, and Stage 0 hard stops.

The EAC preregistration and prototype protocol are frozen in `reports/eac_vla/preregistration.md` and `reports/eac_vla/prototype_protocol.md`.

EAC Stage 0 completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The audit is saved in `reports/eac_vla/stage_0_audit.json` and passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `2000` validation records, `400` validation frames, `6000` reserved confirmatory records untouched, zero validation/test overlap, first-two dispersion p95 `0.0007983036317792467`, commitment counts `2:136`, `8:132`, `50:132`, and passthrough max error `5.07000000038449e-07`.

Because the canonical artifact stores first-two previews rather than all `50` postprocessed chunk actions, the runtime full-chunk equality and queue-prefix execution check was run before validation search.

EAC runtime queue check completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. It loaded frozen SmolVLA on `NVIDIA GeForce RTX 5080`, produced a full postprocessed chunk shape `[50, 7]`, verified `select_action` matched `chunk[0]` with max absolute diff `0.0`, observed queue length `0 -> 49`, and verified every commitment prefix in `{1, 2, 4, 8, 16, 50}` preserved action values exactly.

EAC bounded validation search completed with exactly six configurations and no confirmatory records used for tuning. The selected frozen config is `eac_q33_aggressive_1_4_50`, with validation score `0.7530415186081504`, commitment counts `1:132`, `4:136`, `50:132`, policy-calls-per-step proxy `0.4216`, and risk-exposure-reduction proxy `0.9032794643799159`.

The EAC Stage A matched manifest is frozen in `reports/eac_vla/stage_a_manifest.json` with canonical payload hash `63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C`, reset seeds `20261211` and `20261212`, `10` paired cases per policy, and `50` total planned episodes. EAC Stage A policy preflight passed in `reports/eac_vla/stage_a_preflight.json`: CUDA was available on `NVIDIA GeForce RTX 5080`, output shape was `[50, 7]`, and all policy prefixes preserved action values exactly.

EAC Stage A runner validation passed in `reports/eac_vla/stage_a_runner_validation.json`: the runner preserves action values, reconstructs frozen validation-only thresholds, and authorizes the frozen Stage A rollout without training or confirmatory-test tuning.

EAC Stage A completed `50 / 50` episodes with zero exceptions. EAC full reached `8 / 10`; Base fixed queue, no-calibration ablation, and fixed short-replan each reached `7 / 10`; AAC entropy proxy reached `9 / 10`.

The EAC Stage B matched manifest is frozen in `reports/eac_vla/stage_b_manifest.json` with canonical payload hash `31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D`, all `20` official tasks, fresh reset seeds `20261213` and `20261214`, `40` paired cases per policy, and `200` planned episodes.

EAC Stage B completed from the detached run `runs/eac_vla_stage_b/20260714T202334Z` with wrapper exit code `0`, `200 / 200` official LIBERO episodes, zero exceptions, and no confirmatory-test tuning. The final result is saved in `reports/eac_vla/stage_b_result.json` and summarized in `reports/eac_vla/stage_b_result.md`.

Stage B decision: `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Frozen Base fixed queue reached `30 / 40`, AAC entropy proxy reached `30 / 40`, EAC full reached `29 / 40`, the no-calibration/no-hysteresis ablation reached `30 / 40`, and fixed short-replan reached `29 / 40`. EAC preserved action values, kept finite valid `[50, 7]` action chunks, and activated the scheduler with commitment counts `{'1': 807, '4': 199, '50': 148}`.

EAC full-minus-Base paired delta was `-0.025` with CI `[-0.175, 0.125]`; full-minus-AAC proxy was `-0.025` with CI `[-0.15, 0.1]`; full-minus-ablation was `-0.025` with CI `[-0.175, 0.125]`; and full-minus-fixed-short-replan was `0.0` with CI `[-0.15, 0.15]`.

Final EAC decision: valid current-formulation kill. Do not rescue EAC by retuning `eac_q33_aggressive_1_4_50`, changing thresholds, changing tasks or resets, changing the five-policy list, reinterpreting partial results, or applying any post-hoc expansion.

Epoch 4 Cycle 11 generated exactly three post-EAC candidates in `reports/epoch_4_cycle_11_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_11_prior_mechanism_map.md`, and selected `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

G3P is anchored to Direct Action-Head Injection of A Grounded 3D Point, with RoboPoint, RoboGround, and AffordanceVLA as secondary spatial-grounding priors. The selected design changes the mechanism axis from queue scheduling to source-gated gripper-relative spatial grounding at the action interface. It must use only deployment-observable RGB, proprioception, language, and Base features at inference; oracle object state may be used only for discovery/validation diagnostics and training labels, never as hidden confirmatory-test input.

The design-level five-policy comparison is Base, a closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic.

The G3P-VLA Researcher A proposal is frozen in `reports/g3p_vla/researcher_proposal.md` with proposal hash `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`. Reviewer B attack is complete in `reports/g3p_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A rebuttal is complete in `reports/g3p_vla/researcher_rebuttal.md` with decision `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is frozen in `reports/g3p_vla/mathematical_mechanism_audit.md` with decision `G3P_MATHEMATICAL_AUDIT_PREREGISTERED`. The preregistration and prototype protocol are frozen in `reports/g3p_vla/preregistration.md` and `reports/g3p_vla/prototype_protocol.md`.

G3P Stage 0 completed without training, validation search, rollout, or confirmatory-test tuning. The development audit is saved in `reports/g3p_vla/development_audit.json` and summarized in `reports/g3p_vla/development_audit.md`.

Final G3P Stage 0 decision: `DATA_OR_SUPERVISION_FAILURE`. The source gate passed, split overlap was zero, point predictability beat the strongest trivial baseline by margin `0.2136890612067978`, and oracle action headroom was `0.08630366897708504`; however, the future-waypoint material point label collapsed with train material fraction `0.9982142857142857` and validation material fraction `1.0`, violating the frozen label-health gate.

This is a pre-rollout data/supervision stop, not a closed-loop scientific kill. Do not rescue G3P by changing the material threshold, label construction, source gate, validation search, or Stage 0 criteria.

Epoch 4 Cycle 12 generated exactly three post-G3P candidates in `reports/epoch_4_cycle_12_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_12_prior_mechanism_map.md`, and selected `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

CALA is anchored to CAC-VLA, with VLS and World Pilot as secondary action-interface priors. The selected design changes the mechanism axis from source-gated point labels to action-structured latent conditioning. Future 7D action segments may be used only as discovery/validation supervision; inference must use only deployment-observable current RGB, proprioception, language, and Base features.

The design-level five-policy comparison is Base, a CAC-style latent-action proxy, CALA full, no-context-gate ablation, and one simple task-mean latent-action baseline.

Current stage: `epoch_4_cycle_12_candidate_generation_completed`. Next action: freeze and hash the `CALA-VLA` Researcher A proposal before Reviewer B attack.
