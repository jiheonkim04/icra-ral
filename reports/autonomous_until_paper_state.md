# Autonomous Until Paper State

Date: 2026-07-15 KST

Active governance: `reports/current_research_governance.md`

Branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `SELECT_EAC_VLA`

Current epoch: `4`

Current cycle: `10`

Current stage: `epoch_4_cycle_10_eac_proposal_pending`

Allowed final states:

- `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
- `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
- `HARD_EXTERNAL_BLOCKER`
- `SAFETY_RESOURCE_STOP`

There is no finite global method-cycle limit.

## Corrected Epoch 1

Cycle 1 `DICD-VLA`: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`.

Cycle 2 `FEDO-VLA`: `VALID_CURRENT_FORMULATION_KILL`.

Cycle 3 `GCAP-VLA`: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`.

Epoch 2 must change at least two core dimensions relative to DICD, FEDO, and GCAP, and must not use cosmetic variants of post-hoc delay adapters, residual feedback correction, hold-last/edge image repair, selector/ranker/verifier routes, barrier/filter/damping, generic confidence/progress/value heads, generic DPO, or simple action reweighting.

## Epoch 2 Cycle 1

`PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Stage A completed `50 / 50` episodes with zero exceptions. Full PTC reached `0 / 10`, frozen SmolVLA reached `3 / 10`, and the full method was exactly `0.30` task-balanced success below the strongest baseline. The mechanism was active, so this is a valid current-formulation kill.

## Epoch 2 Cycle 2

`SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Stage A completed `50 / 50` episodes with zero exceptions. Full SACF reached `0 / 10`, frozen SmolVLA reached `7 / 10`, and the full method was `0.70` task-balanced success below the strongest baseline. The semantic component was active, so this is a valid current-formulation kill.

## Epoch 2 Cycle 3

`OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`.

Train acquisition passed `16 / 16` closed-loop acquisition episodes with zero exceptions. Stage A completed `50 / 50` episodes with zero exceptions and required Stage B rather than a permanent Stage A kill.

Expanded Stage B completed `400 / 400` total episodes: `80` paired episodes for each key policy. OCFN full reached `26 / 80` with task-balanced success `0.325`; the strongest baseline, zero-noise SmolVLA, reached `27 / 80` with task-balanced success `0.3375`. The OCFN mechanism was active, with mean initial-noise deltas `0.020219` versus global prior and `0.032354` versus task-shuffled prior.

The paired bootstrap upper confidence bound for `ocfn_full - zero_noise_smolvla` was `0.0625`, excluding the preregistered useful `+0.10` prototype improvement. This is a valid current-formulation kill, not a terminal campaign decision.

## Epoch 2 Failure Synthesis

Epoch 2 produced three related non-GO action-surface methods: `PTC-VLA`, `SACF-VLA`, and `OCFN-VLA`. All three mechanisms acted, but all were harmful or explained by simple baselines.

The synthesized decision is `EPOCH_2_SYNTHESIZED_KILLS_EPOCH_3_PIVOT_REQUIRED`. Epoch 3 must change at least two core dimensions relative to Epoch 2 and should avoid direct small action heads, semantic or phase prefixes, action residual correction, fixed or selected flow-noise priors, ranker/verifier/barrier/filter/damping routes, and simple action-statistic baselines as the main novelty.

## Resume

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/autonomous-until-paper-governance-v2
type reports\current_research_governance.md
```

## Epoch 3 Cycle 1

`CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`.

Teacher acquisition completed `10 / 10` successful Quantized OpenVLA-OFT INT4 episodes and produced `1765` teacher trace rows. Student training passed with `192` retention rows. Stage A completed `50 / 50` held-out episodes with zero exceptions. Frozen SmolVLA reached `7 / 10`; direct distillation, teacher trace memory, no-retention CBFD, and full CBFD each reached `0 / 10`. The CBFD mechanism was active, with full action deltas `1.244676` versus direct distillation and `1.652989` versus teacher memory.

This satisfies the Stage A permanent kill rule: full method `0 / 10` while a paired baseline has at least `4 / 10`.

## Epoch 3 Cycle 2

`SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`.

Stage B completed `200 / 200` episodes with zero exceptions. Full SCVC reached `11 / 40`, while the strongest baseline, shifted frozen SmolVLA, reached `20 / 40`. The paired bootstrap confidence interval for full minus shifted frozen was `[-0.425, -0.025]`. The image canonicalizer acted, but useful closed-loop improvement was excluded.

## Epoch 3 Cycle 3

`PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`.

Stage A completed `50 / 50` episodes with zero exceptions and required Stage B. Stage B completed `40` paired episodes per policy and was unresolved, so current governance allowed one expansion. The expanded Stage B completed `400 / 400` rows with zero exceptions and a valid shared task/reset manifest. Full PSE reached `50 / 80`, while the strongest baseline, `bright_single`, reached `51 / 80`. The paired bootstrap confidence interval for full minus `bright_single` was `[-0.1000, 0.0750]`, excluding useful `+0.10` improvement after maximum expansion.

## Epoch 3 Failure Synthesis

Epoch 3 produced three related non-GO observation/data-side methods: `CBFD-VLA`, `SCVC-VLA`, and `PSE-VLA`. All three mechanisms acted or changed policy behavior, but each was explained by a simpler baseline.

The synthesized decision is `EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED`.

## Epoch 4 Cycle 1

`RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`.

The method passed Stage 0, Stage 1, and Stage 2A, then completed Stage 2B with `200 / 200` episodes and zero exceptions. Full RCV reached `20 / 40` with task-balanced success `0.50`. It beat queued SmolVLA (`14 / 40`) and the SV-deviation proxy (`16 / 40`) but lost to the no-context ablation (`24 / 40`) and stateless first-action baseline (`24 / 40`).

The paired comparison against the no-context ablation was negative: full-minus-ablation delta `-0.10`, wins `2`, losses `6`, ties `32`, CI `[-0.250, 0.025]`. The paired comparison against stateless was also negative: delta `-0.10`, wins `2`, losses `6`, ties `32`, CI `[-0.225, 0.025]`.

RCV's mechanism acted, with full replan rate `0.557293` and heavy policy calls per step `0.563500`, but the no-context ablation achieved higher success with fewer heavy calls per step (`0.429078`). The result excludes a useful improvement from the claimed current-state queued-vs-fresh validity mechanism.

## Epoch 4 Cycle 2

`CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`.

Stage 0/1 acquired and calibrated a contrastive action-value memory with `10801` records and passed the preregistered gateable calibration checks. Stage 2A completed `50 / 50` episodes and required Stage 2B. Stage 2B completed `200 / 200` episodes and produced a positive but unresolved signal, so the preregistered one-time expansion was run unchanged.

The expanded result completed `290 / 290` rows with zero exceptions: `58` paired episodes for each of five variants and an identical task/reset manifest. Full CAVM reached `24 / 58` with task-balanced success `0.413793`. The strongest baseline, nearest-success replay, reached `23 / 58` with task-balanced success `0.396552`; frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and the no-contrast ablation reached `21 / 58`.

Full CAVM beat every baseline and the key ablation numerically, but the effect remained below the preregistered useful-improvement bar after the only allowed expansion. Full-minus-nearest paired delta was `0.017241`, wins `4`, losses `3`, ties `51`, CI `[-0.068966, 0.103448]`. Full-minus-no-contrast paired delta was `0.051724`, CI `[-0.034483, 0.137931]`. Mechanism activation was nonzero (`0.633522` mean gate activation rate), and there was no privileged inference signal, but the final decision is non-GO with no third expansion.

## Epoch 4 Cycle 3

`FANG-VLA` is selected and preregistered as the first post-CAVM performance-oriented method.

Selection artifacts:

- prior mechanism map: `reports/epoch_4_cycle_3_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_3_candidate_generation.md`
- proposal: `reports/fang_vla/researcher_proposal.md`
- proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`
- reviewer attack: `reports/fang_vla/reviewer_attack.md`
- rebuttal: `reports/fang_vla/researcher_rebuttal.md`
- mathematical audit: `reports/fang_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/fang_vla/preregistration.md`
- prototype protocol: `reports/fang_vla/prototype_protocol.md`

Development audit passed with `10801` records, duplicate keys `0`, validation gateable fraction `1.0`, and median action-field separation `0.124345`. The first uncalibrated gate validation search is preserved as `VALIDATION_SEARCH_STOP_DESIGN_FAILURE` because the gate activated almost everywhere. The calibrated validation search then selected `fang_c01` with score `0.996806`, mean delta L2 `0.002555`, gate activation fraction `0.499882`, action validity `1.0`, and gate tau `2.815790`.

Stage A completed `50 / 50` episodes with zero exceptions. All five policies tied at `3 / 10` task-balanced success `0.30`: `base_smolvla`, `afil_local_proxy`, `fang_full`, `fang_no_failure_ablation`, and `nearest_success_replay`. FANG full acted with mean gate `0.095963`, gate activation `0.513922`, and mean action delta L2 `0.008186`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40` with task-balanced success `0.275`, while frozen SmolVLA reached `16 / 40`, the AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`.

The paired comparison against Base was negative: full-minus-base delta `-0.125`, wins `1`, losses `6`, ties `33`, CI `[-0.250, 0.000]`. Full was also `-0.100` versus the AFIL proxy and exactly tied with the key ablation. The FANG mechanism acted, with mean gate `0.086914`, gate activation `0.500365`, and mean action delta L2 `0.008217`, but the failure-aware component did not produce a closed-loop gain beyond simpler explanations.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue `fang_c01`, retune its threshold, or reinterpret Stage A/Stage B identities.

## Epoch 4 Cycle 4

`EvoState-VLA` is selected and preregistered as an EvoScene/DREAM-anchored action-evolved state guidance method for frozen chunked VLAs.

Selection artifacts:

- prior mechanism map: `reports/epoch_4_cycle_4_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_4_candidate_generation.md`
- proposal: `reports/evostate_vla/researcher_proposal.md`
- proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`
- reviewer attack: `reports/evostate_vla/reviewer_attack.md`
- rebuttal: `reports/evostate_vla/researcher_rebuttal.md`
- mathematical audit: `reports/evostate_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/evostate_vla/preregistration.md`
- prototype protocol: `reports/evostate_vla/prototype_protocol.md`

Stage 0 development audit completed without closed-loop rollout. It produced `AUDIT_STOP_DESIGN_FAILURE`: `10769` transition pairs existed with zero duplicate keys, and the full transition model improved strongly over a constant predictor (`0.715309`), but it improved only `0.024689` over an actionless model, below the preregistered `0.05` action-input improvement threshold. Controllability rank was `7`, gate positive fraction was `0.287610`, validation action delta p95 was `0.041577`, and validation action validity was `1.0`.

This is a valid pre-rollout hard stop, not a closed-loop scientific result. Do not lower the threshold, reinterpret the audit, or launch EvoState Stage A.

## Epoch 4 Cycle 5

`RAC-VLA` is selected and preregistered as a Reflective VLA-anchored action-consequence calibration method for frozen SmolVLA under controlled deployment action-channel shift.

Selection artifacts:

- prior mechanism map: `reports/epoch_4_cycle_5_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_5_candidate_generation.md`
- proposal: `reports/rac_vla/researcher_proposal.md`
- proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`
- reviewer attack: `reports/rac_vla/reviewer_attack.md`
- rebuttal: `reports/rac_vla/researcher_rebuttal.md`
- mathematical audit: `reports/rac_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/rac_vla/preregistration.md`
- prototype protocol: `reports/rac_vla/prototype_protocol.md`

Stage 0 development audit passed without closed-loop rollout. It found `10769` consequence pairs and `53685` labeled synthetic inverse-command examples with zero duplicate perturbation keys. The full action-consequence classifier reached validation accuracy `0.585745`, beating action-only `0.368496` and no-consequence `0.374483`; the full-vs-best-baseline margin was `0.211262`, above the preregistered `0.05` threshold. Gate positive fraction was `0.168306`, clean gate positive fraction was `0.0`, clean action delta p95 was `0.0`, and validation action validity was `1.0`.

The bounded six-config validation search selected `rac_h4_a0.05`: history horizon `4`, residual alpha `0.05`, score `0.508926`, full validation accuracy `0.603250`, and full-vs-best-baseline margin `0.244397`.

Stage A completed `50 / 50` episodes with zero exceptions under the frozen hidden `x_attenuate` action-channel shift. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. RAC full tied Base and the key ablation, lost by only `1 / 10` to the strongest baseline and simple baseline, and did not satisfy any permanent Stage A kill criterion.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared task/reset manifest: `200` unique `(variant, task, identity)` keys, duplicate keys `0`, `40` episodes per variant, and identical paired manifests. RAC full reached `1 / 40`; shifted Base reached `1 / 40`; the Reflective-history proxy reached `1 / 40`; the no-consequence ablation reached `2 / 40`; and the online diagonal inverse-gain simple baseline reached `2 / 40`. RAC full tied Base and the closest-prior proxy, but lost to the key ablation and simple baseline.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue RAC, retune `rac_h4_a0.05`, change the hidden shift, or reinterpret the closed result.

## Post-RAC Governance

The post-RAC performance-oriented governance is installed in `reports/current_research_governance.md`, `AGENTS.md`, and `reports/codex_delegation_manual.md`. Future method cycles must maximize the probability of an honest paper-worthy positive result by using positive-prior anchors, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded development search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

## Epoch 4 Cycle 6

`MTF-VLA` is selected and preregistered as a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training.

Selection artifacts:

- prior mechanism map: `reports/epoch_4_cycle_6_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_6_candidate_generation.md`
- proposal: `reports/mtf_vla/researcher_proposal.md`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- reviewer attack: `reports/mtf_vla/reviewer_attack.md`
- rebuttal: `reports/mtf_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mtf_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/mtf_vla/preregistration.md`
- prototype protocol: `reports/mtf_vla/prototype_protocol.md`

The first serious comparison is frozen to five policies: Base, FrameSkip proxy, MTF full, no-retention ablation, and uniform retained-ratio LoRA.

Stage 0 development audit passed without training or closed-loop rollout using `reports/official_smolvla_stable_prediction_artifact.json`. It found `1600` development records (`1200` train, `400` validation), `1200` reserved test records not used, `40` selected task keys, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, state joined fraction `1.0`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config is frozen in `reports/mtf_vla/selected_config.json`; the training manifest is frozen in `reports/mtf_vla/selected_training_manifest.json`.

Current stage: adapter training pending. Stage A must not start until disk-reloadable checkpoints exist for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA.


The MTF adapter-training runner is implemented and dry-run validated in `scripts/run_mtf_vla_adapter_training.py`. The real selected-training manifest joins cleanly with the official split and stable prediction artifact: MTF full has `567` training events (`176` milestone, `391` frozen-base retention), no-retention ablation has `176`, the FrameSkip proxy has `176`, and uniform retained-ratio LoRA has `240`. Train/validation/test frame overlap is `0 / 0 / 0`; validation remains `400` frames; no training or closed-loop rollout happened in this dry run.

Current stage: adapter-training runner validated. Stage A must still not start until disk-reloadable checkpoints are trained and disk-reload verified for all four trainable policies.

Adapter training completed after the runner dry-run and a development-only FrameSkip proxy repair: all four trainable Stage A policies were trained with seed `101`, saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, reloaded from disk, and evaluated on the `400` validation frames. Final decision: `MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for the no-retention ablation, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. The corrected FrameSkip proxy uses `240` action-variation-selected train events and is distinct from the no-retention ablation. No closed-loop rollout happened and no confirmatory-test identities were used.

The MTF Stage A matched manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` with canonical payload hash `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`. It contains exactly five policies (`frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`), five deterministic task keys selected from the official 20-task manifest, fresh reset seeds `20261201` and `20261202`, `10` paired cases per policy, and `50` total planned episodes. `frameskip_proxy_lora` is labeled as a faithful local proxy, not an official FrameSkip reproduction.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; the no-retention ablation and MTF full each reached `7 / 10`. Full MTF tied the key ablation and was only one episode behind the strongest baselines, so this is not a valid Stage A kill. The frozen adjudication is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`.

The MTF Stage B matched manifest was frozen in `reports/mtf_vla/stage_b_manifest.json` with canonical payload hash `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`. It used all `20` official tasks, fresh reset seeds `20261203` and `20261204`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities were unchanged from Stage A.

Stage B completed `200 / 200` official LIBERO episodes with zero exceptions. Frozen SmolVLA reached `28 / 40`, the FrameSkip proxy reached `27 / 40`, uniform retained-ratio LoRA reached `29 / 40`, the no-retention ablation reached `32 / 40`, and MTF full reached `26 / 40`. Full MTF lost to the key ablation by paired delta `-0.15` with CI `[-0.275, -0.025]`, and also trailed Base, the FrameSkip proxy, and uniform retained-ratio LoRA.

Final MTF decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. This is a valid current-formulation kill because the simpler no-retention ablation explains or exceeds the full method. Do not rescue MTF by retuning `mtf_r20_ret100`, changing retention, changing task/reset identities, or reinterpreting Stage B outcomes.

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

The selected first comparison is frozen at the design level to five policies: Base, a PriorVLA-style proxy, PESA full, a no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline. `priorvla_style_proxy` is a faithful transparent local proxy, not an official PriorVLA reproduction unless exact official equivalence is later established.

The Researcher A proposal is frozen in `reports/pesa_vla/researcher_proposal.md`.

Reviewer B attack is complete in `reports/pesa_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against PriorVLA, LoRA-SP, and VLA-GSE; requires `priorvla_style_proxy` to remain an honest local proxy; forbids KL over deterministic 7D actions; and requires noncollapsed labels, observable spectral/query mechanisms, bounded action deltas, identity-preserving integration, and one strong standard-LoRA or clean-retention simple killer before rollout.

Researcher A rebuttal is complete in `reports/pesa_vla/researcher_rebuttal.md` with decision `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts the narrow claim, keeps the PriorVLA-style proxy and simple killer live, and requires mathematical variable/shape/objective/gradient audits before implementation.

The PESA mathematical mechanism audit is frozen in `reports/pesa_vla/mathematical_mechanism_audit.md` with decision `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`. The audit defines spectral-energy variables, tensor shapes, action formula, Huber/L2/entropy objectives, gradient paths, small-batch scale checks, required ablations, identity-preserving Base passthrough, and the no deterministic-action KL rule.

The PESA preregistration and prototype protocol are frozen in `reports/pesa_vla/preregistration.md` and `reports/pesa_vla/prototype_protocol.md`. The frozen first comparison remains exactly five policies: Base, PriorVLA-style proxy, PESA full, no-spectral/no-prior-query ablation, and one standard-LoRA or clean-retention simple killer.

PESA Stage 0 completed without training, closed-loop rollout, manifest freeze, or confirmatory-test tuning. The development audit is saved in `reports/pesa_vla/development_audit.json` and summarized in `reports/pesa_vla/development_audit.md`.

Final PESA Stage 0 decision: `DESIGN_FAILURE`. The query labels were balanced (`0.3858333333333333` train positive fraction, `0.4` validation positive fraction), standard LoRA had positive L1 headroom (`0.0065395455599999985`), spectral activation was noncollapsed (validation active-rank mean `2.3475`), action distinctions passed, Base validity was `1.0`, and gradients were finite. However, the prior-query probe reached validation accuracy `0.5225` versus majority `0.6`, for an accuracy margin `-0.07750000000000001`, below the preregistered `+0.02` requirement.

This is a valid pre-rollout design stop, not a closed-loop scientific kill. Do not rescue PESA by changing query labels, thresholds, features, validation search, or Stage 0 criteria.

Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`. This remains a pre-rollout design stop, not a closed-loop kill.

## Epoch 4 Cycle 10

Exactly three post-PESA candidates were generated and scored in `reports/epoch_4_cycle_10_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_10_prior_mechanism_map.md`. PESA remains stopped and may not be rescued.

`EAC-VLA` is selected as an Adaptive Action Chunking anchored entropy-calibrated queue-scheduling method for frozen SmolVLA. It preserves frozen SmolVLA weights and emitted 7D action values, changing only how many actions from the current `50 x 7` chunk are committed before refreshing the observation.

The selected first comparison is frozen at the design level to five policies: Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and one fixed short-replan simple killer. `aac_entropy_proxy` is a faithful transparent local proxy, not an official AAC reproduction.

Current decision: `SELECT_EAC_VLA`. Current stage: `epoch_4_cycle_10_eac_proposal_pending`. Next action: freeze and hash the EAC-VLA Researcher A proposal before Reviewer B.
