# Autonomous RA-L Campaign State

Date: 2026-07-15 KST

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EAC_STAGE_0_PASS_RUNTIME_QUEUE_CHECK_REQUIRED`

Current epoch: `4`

Current cycle: `10`

Current stage: `epoch_4_cycle_10_eac_runtime_queue_check_pending`

## Corrected Epoch 1 Result

Cycle 1 `DICD-VLA`:

- corrected status: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- full: `1 / 10`
- direct chunk-index delay: `2 / 10`
- no-history ablation: `1 / 10`
- ruling: do not rerun or rescue the current formulation; do not treat a one-episode difference at 10 episodes per policy as a permanent scientific family kill.

Cycle 2 `FEDO-VLA`:

- corrected status: `VALID_CURRENT_FORMULATION_KILL`
- faulted full: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen: `4 / 10`
- clean FEDO: `0 / 10`
- ruling: do not revive the current formulation.

Cycle 3 `GCAP-VLA`:

- corrected status: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`
- occluded full: `3 / 10`
- occluded frozen: `4 / 10`
- Sobel edge boost: `5 / 10`
- no-temporal ablation: `4 / 10`
- clean frozen: `1 / 10`
- clean GCAP: `5 / 10`
- ruling: do not rerun or rescue the current formulation; do not call the whole perception-repair family dead.

## Epoch 2 Result

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` total episodes with zero exceptions, `80` paired episodes per key policy, active mechanism, OCFN full `26 / 80`, zero-noise SmolVLA `27 / 80`, and paired upper confidence bound versus the strongest baseline `0.0625`.

These three related failures are synthesized in `reports/epoch_2_failure_synthesis.md`.

## Next Action

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: Stage A completed `50 / 50` held-out episodes with zero exceptions, frozen SmolVLA reached `7 / 10`, and full CBFD reached `0 / 10` with active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

The related Epoch 3 failures are synthesized in `reports/epoch_3_failure_synthesis.md`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

## Epoch 4 Cycle 3

`FANG-VLA` is selected and preregistered as an AFIL-anchored identity-preserving failure-aware action-field guidance method.

Artifacts:

- `reports/epoch_4_cycle_3_prior_mechanism_map.md`
- `reports/epoch_4_cycle_3_candidate_generation.md`
- `reports/fang_vla/researcher_proposal.md`
- `reports/fang_vla/proposal_hash.txt`
- `reports/fang_vla/reviewer_attack.md`
- `reports/fang_vla/researcher_rebuttal.md`
- `reports/fang_vla/mathematical_mechanism_audit.md`
- `reports/fang_vla/preregistration.md`
- `reports/fang_vla/prototype_protocol.md`

Development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. The paired full-minus-base delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

## Epoch 4 Cycle 4

`EvoState-VLA` is selected and preregistered as an EvoScene/DREAM-anchored action-evolved state guidance method.

Artifacts:

- `reports/epoch_4_cycle_4_prior_mechanism_map.md`
- `reports/epoch_4_cycle_4_candidate_generation.md`
- `reports/evostate_vla/researcher_proposal.md`
- `reports/evostate_vla/proposal_hash.txt`
- `reports/evostate_vla/reviewer_attack.md`
- `reports/evostate_vla/researcher_rebuttal.md`
- `reports/evostate_vla/mathematical_mechanism_audit.md`
- `reports/evostate_vla/preregistration.md`
- `reports/evostate_vla/prototype_protocol.md`

Stage 0 development audit completed without closed-loop rollout and stopped as `AUDIT_STOP_DESIGN_FAILURE`. The full transition model improved over a constant predictor by `0.715309`, but improved only `0.024689` over an actionless model, below the preregistered `0.05` action-input improvement threshold. This means the proposed action-conditioned state mechanism is not sufficiently supported for rollout.

## Epoch 4 Cycle 5

`RAC-VLA` is selected and preregistered as a Reflective VLA-anchored action-consequence calibration method for frozen SmolVLA under controlled deployment action-channel shift.

Stage 0 development audit passed with `10769` consequence pairs, `53685` labeled examples, zero duplicate perturbation keys, full validation accuracy `0.585745`, action-only validation accuracy `0.368496`, no-consequence validation accuracy `0.374483`, and full-vs-best-baseline margin `0.211262`.

The bounded six-config validation search selected `rac_h4_a0.05`: history horizon `4`, residual alpha `0.05`, score `0.508926`, full validation accuracy `0.603250`, and full-vs-best-baseline margin `0.244397`.

Stage A completed `50 / 50` episodes with zero exceptions under the frozen hidden `x_attenuate` action-channel shift. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. RAC full tied Base and the key ablation, lost by only `1 / 10` to the strongest baseline and simple baseline, and did not satisfy any permanent Stage A kill criterion.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared task/reset manifest: `200` unique `(variant, task, identity)` keys, duplicate keys `0`, `40` episodes per variant, and identical paired manifests. RAC full reached `1 / 40`; shifted Base reached `1 / 40`; the Reflective-history proxy reached `1 / 40`; the no-consequence ablation reached `2 / 40`; and the online diagonal inverse-gain simple baseline reached `2 / 40`. RAC full tied Base and the closest-prior proxy, but lost to the key ablation and simple baseline.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue RAC, retune `rac_h4_a0.05`, change the hidden shift, or reinterpret the closed result.

## Post-RAC Governance

The post-RAC performance-oriented governance is installed in `reports/current_research_governance.md`, `AGENTS.md`, and `reports/codex_delegation_manual.md`. Future method cycles must maximize the probability of an honest paper-worthy positive result by using positive-prior anchors, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded development search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

## Epoch 4 Cycle 6

`MTF-VLA` is selected and preregistered as a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training.

Stage 0 development audit passed without training or closed-loop rollout using `reports/official_smolvla_stable_prediction_artifact.json`: `1600` development records, `1200` reserved test records not used, `40` task keys, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config and training manifest are frozen under `reports/mtf_vla/`.

Current stage: adapter training pending. Stage A must not start until disk-reloadable checkpoints exist for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA.


The MTF adapter-training runner is implemented and dry-run validated in `scripts/run_mtf_vla_adapter_training.py`. The real selected-training manifest joins cleanly with the official split and stable prediction artifact: MTF full has `567` training events (`176` milestone, `391` frozen-base retention), no-retention ablation has `176`, the FrameSkip proxy has `176`, and uniform retained-ratio LoRA has `240`. Train/validation/test frame overlap is `0 / 0 / 0`; validation remains `400` frames; no training or closed-loop rollout happened in this dry run.

Current stage: adapter-training runner validated. Stage A must still not start until disk-reloadable checkpoints are trained and disk-reload verified for all four trainable policies.

Adapter training completed after the runner dry-run and a development-only FrameSkip proxy repair: all four trainable Stage A policies were trained with seed `101`, saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, reloaded from disk, and evaluated on the `400` validation frames. Final decision: `MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for the no-retention ablation, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. The corrected FrameSkip proxy uses `240` action-variation-selected train events and is distinct from the no-retention ablation. No closed-loop rollout happened and no confirmatory-test identities were used.

The MTF Stage A matched manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` with canonical payload hash `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`. It contains exactly five policies (`frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`), five deterministic task keys selected from the official 20-task manifest, fresh reset seeds `20261201` and `20261202`, `10` paired cases per policy, and `50` total planned episodes. `frameskip_proxy_lora` is labeled as a faithful local proxy, not an official FrameSkip reproduction.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; the no-retention ablation and MTF full each reached `7 / 10`. Full MTF tied the key ablation and was only one episode behind the strongest baselines, so this is not a valid Stage A kill. The frozen adjudication is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`.

The MTF Stage B matched manifest was frozen in `reports/mtf_vla/stage_b_manifest.json` with canonical payload hash `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`. It used all `20` official tasks, fresh reset seeds `20261203` and `20261204`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities were unchanged from Stage A.

Stage B completed `200 / 200` official LIBERO episodes with zero exceptions. Frozen SmolVLA reached `28 / 40`, the FrameSkip proxy reached `27 / 40`, uniform retained-ratio LoRA reached `29 / 40`, the no-retention ablation reached `32 / 40`, and MTF full reached `26 / 40`. Full-minus-no-retention paired delta was `-0.15` with CI `[-0.275, -0.025]`.

Final MTF decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Do not rescue MTF by retuning `mtf_r20_ret100`, changing retention, changing task/reset identities, or reinterpreting Stage B outcomes.

## Epoch 4 Cycle 7

Exactly three post-MTF candidates were generated and scored in `reports/epoch_4_cycle_7_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_7_prior_mechanism_map.md`. MTF remains archived and may not be rescued.

`DAGR-VLA` is selected as a DAM-VLA anchored dynamic arm/gripper routing method for frozen SmolVLA adaptation. Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`.

The selected first comparison is frozen at the design level to five policies: Base, a DAM-style static component proxy, DAGR full, a no-dynamic-route shared residual ablation, and one gripper-transition heuristic simple killer. No closed-loop rollout, training, or confirmatory-test tuning has happened for DAGR.

Reviewer B attack is complete in `reports/dagr_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against DAM-VLA, requires `dam_static_component_proxy` to be labeled as a faithful transparent local proxy rather than an official DAM-VLA reproduction, forbids KL over deterministic 7D actions, and makes noncollapsed route-label health, route observability, bounded action deltas, and identity-preserving integration mandatory before rollout.

Researcher A rebuttal is complete in `reports/dagr_vla/researcher_rebuttal.md` with decision `DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. DAGR will not claim broad dynamic arm/gripper routing novelty; its local claim is frozen SmolVLA identity-preserving route-gated residual adaptation. No training, rollout, or confirmatory-test tuning has happened.

The DAGR mathematical audit, preregistration, and prototype protocol are frozen in `reports/dagr_vla/mathematical_mechanism_audit.md`, `reports/dagr_vla/preregistration.md`, and `reports/dagr_vla/prototype_protocol.md`.

Stage 0 development audit passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` in `reports/dagr_vla/development_audit.json`: `1600` development records, `1200` train, `400` validation, `1200` reserved test records not used, duplicate sample keys `0`, duplicate frame keys `0`, split overlap `0 / 0 / 0`, base action validity `1.0`, validation any-route fraction `0.865`, and no hard stops. Route-probe accuracy margins over validation majority were translation `0.0375`, rotation `0.0725`, and gripper `0.26`.

The bounded six-config validation search completed as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING` in `reports/dagr_vla/validation_search.json`. Selected config: `dagr_a020_route_mlp`, residual alpha `0.20`, route architecture `mlp`, validation score `0.8571740870493018`, delta L2 p95 `0.008609326556324959`, clean delta L2 p95 `0.00672802422195673`, and action validity `1.0`.

DAGR policy identity training is complete in `reports/dagr_vla/policy_checkpoint_manifest.json`. Final decision: `DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The trainable identities `dagr_full`, `dam_static_component_proxy`, and `dagr_no_dynamic_route_ablation` all disk-reload, preserve initial base passthrough, and have validation action validity `1.0`; the nontrainable `gripper_transition_heuristic` config is saved under the same checkpoint root.

Checkpoint root: `runs/dagr_vla_checkpoints/dagr_a020_route_mlp`. DAGR full validation delta L2 p95 is `0.008576558902859688`; DAM static proxy p95 is `0.016259152442216873`; no-dynamic-route ablation p95 is `0.006147781852632761`.

The DAGR Stage A matched manifest is frozen in `reports/dagr_vla/stage_a_manifest.json` with canonical payload hash `8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B`. It contains exactly five policies (`frozen_smolvla`, `dam_static_component_proxy`, `dagr_full`, `dagr_no_dynamic_route_ablation`, `gripper_transition_heuristic`), five evenly spaced official tasks, fresh reset seeds `20261205` and `20261206`, `10` paired cases per policy, and `50` total planned episodes. `dam_static_component_proxy` remains labeled as a faithful transparent local proxy, not an official DAM-VLA reproduction.

DAGR Stage A policy preflight passed as `DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT` in `reports/dagr_vla/stage_a_preflight.json`. Five policies loaded through the official SmolVLA/LIBERO path, four checkpoint identities checksum-verified, no accidental checkpoint reuse was detected, the base policy and learned DAGR heads ran on CUDA, and the wrappers produced finite 7D actions. No rollout, training, or confirmatory-test tuning happened during preflight. Next action: launch the official DAGR Stage A rollout.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA reached `8 / 10`, the gripper-transition heuristic reached `7 / 10`, DAGR full reached `6 / 10`, the no-dynamic-route ablation reached `5 / 10`, and the DAM-style static component proxy reached `2 / 10`. DAGR full beat the closest-prior proxy and key ablation but trailed Base by two episodes, which is noncatastrophic under Stage A governance. Final Stage A decision: `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`. Next action: freeze the DAGR Stage B matched manifest without retuning.

The DAGR Stage B matched manifest froze all `20` official tasks, fresh reset seeds `20261207` and `20261208`, `40` paired cases per policy, `200` total planned episodes, and the unchanged five-policy comparison. No checkpoint, threshold, task, or reset was selected from Stage B outcomes.

DAGR Stage B completed `200 / 200` official LIBERO episodes with zero exceptions and no confirmatory-test tuning. Frozen SmolVLA reached `28 / 40`, the DAM-style static component proxy reached `5 / 40`, DAGR full reached `18 / 40`, the no-dynamic-route ablation reached `16 / 40`, and the gripper-transition heuristic reached `24 / 40`. Full-minus-Base paired delta was `-0.25` with CI `[-0.4, -0.1]`; full-minus-gripper paired delta was `-0.15` with CI `[-0.3, 0.0]`.

Final DAGR decision: `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. This is a valid current-formulation kill because the simple gripper-transition heuristic and Base explain or exceed the full method. Do not rescue DAGR by retuning `dagr_a020_route_mlp`, changing route thresholds, changing task/reset identities, changing the policy list, or reinterpreting partial results.

## Epoch 4 Cycle 8

Exactly three post-DAGR candidates were generated and scored in `reports/epoch_4_cycle_8_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_8_prior_mechanism_map.md`. DAGR remains archived and may not be rescued.

`MARC-VLA` is selected as an OpenVLA-OFT anchored median-anchor correction method for frozen SmolVLA flow actions. Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`.

The selected first comparison is frozen at the design level to five policies: Base, OpenVLA-OFT-style L1 proxy, MARC full, no-disagreement-gate ablation, and one static L1 mixture simple killer. No closed-loop rollout or confirmatory-test tuning has happened for MARC.

Reviewer B attack is complete in `reports/marc_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against OpenVLA-OFT and requires noncollapsed disagreement labels, observable gates, bounded action deltas, identity-preserving integration, and a static-mixture simple killer before rollout.

Researcher A rebuttal is complete in `reports/marc_vla/researcher_rebuttal.md` with decision `MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. MARC will not claim that continuous L1 action prediction is novel; its local claim is frozen SmolVLA median-anchor correction. No training rollout or confirmatory-test tuning has happened.

The MARC mathematical audit, preregistration, and prototype protocol are frozen in `reports/marc_vla/mathematical_mechanism_audit.md`, `reports/marc_vla/preregistration.md`, and `reports/marc_vla/prototype_protocol.md`.

Stage 0 development audit passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` in `reports/marc_vla/development_audit.json`: `1600` development records, `1200` train, `400` validation, `1200` reserved test records not used, duplicate sample keys `0`, duplicate frame keys `0`, split overlap `0 / 0 / 0`, train disagreement fraction `0.4`, validation disagreement fraction `0.44`, gate-probe margin `0.0475`, initial action delta p95 `0.0`, and base action validity `1.0`.

The bounded six-config validation search completed as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING` in `reports/marc_vla/validation_search.json`. Selected config: `marc_a020_gate_mlp`, correction alpha `0.20`, gate architecture `mlp`, validation score `0.5457964262366295`, gate accuracy margin `0.0525`, gate predicted-positive fraction `0.3325`, delta L2 p95 `0.011818917468190193`, clean delta L2 p95 `0.010853752493858337`, and action validity `1.0`. Linear configs were stopped for collapsed gates.

MARC full validation action L2 is `0.08665236806523112`; the L1 proxy action L2 is `0.08763420091414227`; full-versus-L1 proxy mean L2 is `0.007010325323790312`; full-versus-no-gate mean L2 is `0.007010325323790312`; full-versus-static mixture mean L2 is `0.0019475044682621956`. The static mixture remains a live reviewer-killer.

MARC policy identity training is complete in `reports/marc_vla/policy_checkpoint_manifest.json`. Final decision: `MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The trainable identities `openvla_oft_l1_proxy`, `marc_full`, `marc_no_disagreement_gate_ablation`, and `static_l1_mixture_baseline` all disk-reload, preserve initial base passthrough, and have validation action validity `1.0`.

Checkpoint root: `runs\marc_vla_checkpoints\marc_a020_gate_mlp`. MARC full validation delta L2 p95 is `0.010693175718188286`; OpenVLA-OFT-style L1 proxy p95 is `0.2307613492012024`; no-disagreement-gate p95 is `0.12246084958314896`; static L1 mixture p95 is `0.07999999821186066`. Full-versus-L1 mean L2 is `0.08430124074220657`, full-versus-no-gate is `0.04372206702828407`, and full-versus-static is `0.032826922833919525`.

The MARC Stage A matched manifest is frozen in `reports/marc_vla/stage_a_manifest.json` with canonical payload hash `3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E`. It contains exactly five policies (`frozen_smolvla`, `openvla_oft_l1_proxy`, `marc_full`, `marc_no_disagreement_gate_ablation`, `static_l1_mixture_baseline`), five evenly spaced official tasks, fresh reset seeds `20261209` and `20261210`, `10` paired cases per policy, and `50` total planned episodes. `openvla_oft_l1_proxy` remains labeled as a faithful transparent local proxy, not an official OpenVLA-OFT reproduction.

MARC Stage A policy preflight passed as `MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT` in `reports/marc_vla/stage_a_preflight.json`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D MARC actions were produced. No rollout result, training, or confirmatory-test tuning happened during preflight.

The official MARC Stage A rollout completed from `runs/marc_vla_stage_a/20260714T171356Z` with exit code `0`, `50 / 50` episodes, zero exceptions, and no confirmatory-test tuning. The result is saved in `reports/marc_vla/stage_a_result.json` and summarized in `reports/marc_vla/stage_a_result.md`.

Stage A decision: `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE`. Frozen SmolVLA reached `8 / 10`, OpenVLA-OFT-style L1 proxy reached `0 / 10`, MARC full reached `0 / 10`, no-disagreement-gate ablation reached `7 / 10`, and static L1 mixture reached `7 / 10`. MARC full-minus-Base paired delta was `-0.8`, full-minus-no-gate was `-0.7`, and full-minus-static was `-0.7`.

Final MARC decision: valid current-formulation kill. MARC full was catastrophically worse than Base and dominated by both the key ablation and simple static-mixture baseline. Do not rescue MARC by retuning `marc_a020_gate_mlp`, changing thresholds, changing policies, changing task/reset identities, or reinterpreting Stage A outcomes.

## Epoch 4 Cycle 9

Exactly three post-MARC candidates were generated and scored in `reports/epoch_4_cycle_9_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_9_prior_mechanism_map.md`. MARC remains archived and may not be rescued.

`PESA-VLA` is selected as a PriorVLA, LoRA-SP, and VLA-GSE anchored prior-expert spectral adaptation method for frozen SmolVLA 7D policies. Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`.

The selected first comparison is frozen at the design level to five policies: Base, a PriorVLA-style proxy, PESA full, a no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline.

The Researcher A proposal is frozen in `reports/pesa_vla/researcher_proposal.md`.

Reviewer B attack is complete in `reports/pesa_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack requires the closest-prior proxy to remain transparent, forbids deterministic-action KL, and makes label health, mechanism observability, bounded deltas, identity preservation, and a strong simple killer mandatory before rollout.

Researcher A rebuttal is complete in `reports/pesa_vla/researcher_rebuttal.md` with decision `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts the narrow claim and keeps the PriorVLA-style proxy, no-spectral/no-prior-query ablation, and standard-LoRA or clean-retention simple killer live.

The PESA mathematical mechanism audit is frozen in `reports/pesa_vla/mathematical_mechanism_audit.md` with decision `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`. The audit defines variables, shapes, spectral-energy masking, objectives, gradient paths, small-batch scale checks, required ablations, and the no deterministic-action KL rule.

The PESA preregistration and prototype protocol are frozen in `reports/pesa_vla/preregistration.md` and `reports/pesa_vla/prototype_protocol.md`. Stage 0 must pass the frozen label, split, headroom, spectral activation, gradient, action-validity, action-distinction, and identity-preservation gates before validation search or training.

PESA Stage 0 completed without closed-loop rollout, training, manifest freeze, or confirmatory-test tuning. The development audit is saved in `reports/pesa_vla/development_audit.json` and summarized in `reports/pesa_vla/development_audit.md`.

Final PESA Stage 0 decision: `DESIGN_FAILURE`. The only hard stop was query observability: validation query-probe accuracy `0.5225`, majority `0.6`, margin `-0.07750000000000001`, below the frozen `+0.02` requirement. Other Stage 0 checks passed or remained healthy, including split integrity, label balance, standard LoRA headroom, noncollapsed spectral activation, action distinctions, finite gradients, initial Base equality, and Base action validity.

This is a pre-rollout design failure, not a closed-loop kill. Do not rescue PESA by changing query-label construction, thresholds, features, or criteria.

Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`. This remains a pre-rollout design stop, not a closed-loop kill.

## Epoch 4 Cycle 10

Epoch 4 Cycle 10 generated exactly three post-PESA candidates in `reports/epoch_4_cycle_10_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_10_prior_mechanism_map.md`, and selected `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

EAC is anchored to Adaptive Action Chunking, with AR-VLA and AC2-VLA as secondary temporal/action-context priors. The method preserves frozen SmolVLA weights and emitted 7D action values, and changes only action-queue commitment length from deployment-observable uncertainty and queue-boundary risk.

The design-level five-policy comparison is Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and fixed short-replan simple killer. No rollout, training, validation search, or confirmatory-test tuning has happened for EAC.

The EAC Researcher A proposal is frozen in `reports/eac_vla/researcher_proposal.md` with proposal hash `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`.

Reviewer B attack is complete in `reports/eac_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against AAC, forces `aac_entropy_proxy` and `fixed_short_replan_baseline` to remain live, requires uncertainty/dispersion validity before rollout, and treats action-value modification as implementation failure.

Researcher A rebuttal is complete in `reports/eac_vla/researcher_rebuttal.md` with decision `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts narrow AAC-extension novelty, exact action-value passthrough, live AAC proxy and fixed-replan killer baselines, uncertainty/dispersion terminology, and Stage 0 hard stops before rollout.

The EAC mathematical mechanism audit is frozen in `reports/eac_vla/mathematical_mechanism_audit.md` with decision `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines the `50 x 7` chunk variables, dispersion/entropy rules, queue-risk formula, commitment map, action-value equality gate, validation-search score, required ablation, and Stage 0 hard stops.

The EAC preregistration and prototype protocol are frozen in `reports/eac_vla/preregistration.md` and `reports/eac_vla/prototype_protocol.md`.

EAC Stage 0 completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The audit is saved in `reports/eac_vla/stage_0_audit.json` and summarized in `reports/eac_vla/stage_0_audit.md`.

Final EAC Stage 0 decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`. The audit used `2000` validation records and `400` unique validation frames, reserved `6000` confirmatory records unused, found zero validation/test frame or sample overlap, confirmed queue helpers and the canonical `50 x 7` chunk shape, and found noncollapsed first-two chunk dispersion with p95 `0.0007983036317792467` and nonzero fraction `1.0`. The preregistered commitment map was noncollapsed (`2`: `136`, `8`: `132`, `50`: `132`), max commitment share was `0.34`, and first-action passthrough max error was `5.07000000038449e-07`. There were no hard stops.

The canonical artifact stores first-two chunk previews and chunk hashes, not all `50` postprocessed actions. Therefore validation search, Stage A, and rollout remain blocked until the runtime full-chunk equality and queue-prefix execution check is implemented and passed.

Current decision: `EAC_STAGE_0_PASS_RUNTIME_QUEUE_CHECK_REQUIRED`. Current stage: `epoch_4_cycle_10_eac_runtime_queue_check_pending`. Next action: implement the EAC runtime full-chunk equality and queue-prefix execution check before validation search.
