# Autonomous Until Paper State

Date: 2026-07-15 KST

Active governance: `reports/current_research_governance.md`

Branch: `codex/autonomous-until-paper-governance-v2`

Current decision:
`HASTE_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_IMPLEMENTATION_PENDING`

Current epoch: `4`

Current cycle: `22`

Current stage: `epoch_4_cycle_22_haste_stage_0a_implementation_pending`

Allowed final states:

- `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
- `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
- `HARD_EXTERNAL_BLOCKER`
- `SAFETY_RESOURCE_STOP`

There is no finite global method-cycle limit.

## Epoch 4 Cycle 22 HASTE-VLA

Cycle 22 generated exactly three candidates and selected `HASTE-VLA` at
`95 / 100`, anchored to StaKe. Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

The complete review and preregistration package is frozen. Only Stage 0A
implementation is pending. HASTE may not train an adapter, run a simulator, or
read confirmatory records before its source, label, headroom, observability,
identity, and persistence gates pass.

## Epoch 4 Cycle 21 HEST-VLA

Cycle 21 generated exactly three candidates and selected `HEST-VLA` at
`93 / 100`, anchored to Spline Policy. The proposal is frozen at
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.
The review, rebuttal, mathematical audit, preregistration, and prototype
protocol are complete. Stage 0A completed `160 / 160` windows with zero
exceptions and exact manifest and persistence integrity. The frozen
all-variant support gate failed because one validation Base row and HEST's
required whole-Base fallback were outside discovery-defined support.

Decision: `HEST_STAGE_0A_IMPLEMENTATION_FAILURE`. This is not a scientific
kill. Stage 0B, rerun, repair, and HEST rescue are forbidden. Continue to Cycle
22 exact-three candidate generation.

## Epoch 4 Cycle 19 SPARC-VLA

Cycle 19 generated exactly three candidates and selected `SPARC-VLA` at
`96 / 100`, anchored to COAST. Proposal hash:
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.

The first Stage 0A attempt persisted both observations before a capture-reset
implementation exception. The one allowed implementation repair was consumed.
Final PID `306` completed `2 / 2` observations with exit `0`, zero exceptions,
and zero duplicate, missing, or extra observation indices. Identity, reload,
Base-hash, finite-operator, and acting checks passed, but both synthetic action
rows failed the frozen Base-relative range-safety gate.

Decision:
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`.
This unlabeled smoke is not a fitted scientific SPARC operator. Stage 0B,
validation, rollout, confirmatory testing, and SPARC rescue are forbidden.
Continue to Cycle 20 exact-three candidate generation.

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

The EAC Researcher A proposal is frozen in `reports/eac_vla/researcher_proposal.md` with proposal hash `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`.

Reviewer B attack is complete in `reports/eac_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against AAC, forces `aac_entropy_proxy` and `fixed_short_replan_baseline` to remain live, requires uncertainty/dispersion validity before rollout, and treats action-value modification as implementation failure.

Researcher A rebuttal is complete in `reports/eac_vla/researcher_rebuttal.md` with decision `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts narrow AAC-extension novelty, exact action-value passthrough, live AAC proxy and fixed-replan killer baselines, uncertainty/dispersion terminology, and Stage 0 hard stops before rollout.

The EAC mathematical mechanism audit is frozen in `reports/eac_vla/mathematical_mechanism_audit.md` with decision `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines the `50 x 7` chunk variables, dispersion/entropy rules, queue-risk formula, commitment map, action-value equality gate, validation-search score, required ablation, and Stage 0 hard stops.

The EAC preregistration and prototype protocol are frozen in `reports/eac_vla/preregistration.md` and `reports/eac_vla/prototype_protocol.md`.

EAC Stage 0 completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The audit is saved in `reports/eac_vla/stage_0_audit.json` and summarized in `reports/eac_vla/stage_0_audit.md`.

Final EAC Stage 0 decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`. The audit used `2000` validation records and `400` unique validation frames, reserved `6000` confirmatory records unused, found zero validation/test frame or sample overlap, confirmed queue helpers and the canonical `50 x 7` chunk shape, and found noncollapsed first-two chunk dispersion with p95 `0.0007983036317792467` and nonzero fraction `1.0`. The preregistered commitment map was noncollapsed (`2`: `136`, `8`: `132`, `50`: `132`), max commitment share was `0.34`, and first-action passthrough max error was `5.07000000038449e-07`, below the serialization epsilon. There were no hard stops.

The canonical artifact stores first-two chunk previews and chunk hashes, not all `50` postprocessed actions, so the runtime full-chunk equality and queue-prefix execution check was run before validation search.

EAC runtime queue check completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The check is saved in `reports/eac_vla/runtime_queue_check.json` and summarized in `reports/eac_vla/runtime_queue_check.md`. It loaded frozen SmolVLA on `NVIDIA GeForce RTX 5080`, produced a full postprocessed chunk shape `[50, 7]`, verified `select_action` matched `chunk[0]` with max absolute diff `0.0`, observed the official queue length change from `0` before selection to `49` afterward, and verified every commitment prefix in `{1, 2, 4, 8, 16, 50}` preserved action values exactly with max prefix and queue-pop diffs `0.0`.

EAC bounded validation search completed with exactly six configurations and no confirmatory records used for tuning. The search is saved in `reports/eac_vla/validation_search.json`, summarized in `reports/eac_vla/validation_search.md`, and the selected frozen config is `reports/eac_vla/selected_config.json`.

Final EAC validation decision: `EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY`. The selected config is `eac_q33_aggressive_1_4_50`, with validation score `0.7530415186081504`, commitment counts `1:132`, `4:136`, `50:132`, policy-calls-per-step proxy `0.4216`, oscillation fraction `0.6388888888888888`, risk-exposure-reduction proxy `0.9032794643799159`, mechanism activation `0.6599999999999999`, clean action-value passthrough `1.0`, and runtime action validity `1.0`.

The EAC Stage A matched manifest is frozen in `reports/eac_vla/stage_a_manifest.json` with canonical payload hash `63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C`. It contains exactly five policies (`frozen_smolvla_fixed_queue`, `aac_entropy_proxy`, `eac_full`, `eac_no_calibration_no_hysteresis_ablation`, `fixed_short_replan_baseline`), fresh reset seeds `20261211` and `20261212`, `10` paired cases per policy, and `50` total planned episodes. `aac_entropy_proxy` remains a faithful transparent local proxy, not an official AAC reproduction.

EAC Stage A policy preflight passed in `reports/eac_vla/stage_a_preflight.json` as `EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING`: `5` scheduler identities were checked, `0` checkpoint policies were required, CUDA ran on `NVIDIA GeForce RTX 5080`, the policy output shape was `[50, 7]`, all policy prefixes preserved action values exactly, and no accidental checkpoint reuse was possible. No rollout, training, validation search, or confirmatory-test tuning happened during preflight.

EAC Stage A runner validation passed in `reports/eac_vla/stage_a_runner_validation.json` as `EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT`. The validated runner reconstructs the frozen validation-only EAC thresholds, uses `2` runtime samples for dynamic schedulers, preserves all policy prefixes without action-value modification, and keeps rollout/training/confirmatory-test tuning at `False`.

EAC Stage A ran detached from `runs/eac_vla_stage_a/20260714T194025Z` and completed `50 / 50` episodes with zero exceptions. The result is saved in `reports/eac_vla/stage_a_result.json` and summarized in `reports/eac_vla/stage_a_result.md`.

Stage A decision: `EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`. EAC full reached `8 / 10`, Base fixed queue reached `7 / 10`, AAC entropy proxy reached `9 / 10`, no-calibration ablation reached `7 / 10`, and fixed short-replan reached `7 / 10`. EAC full preserved action values, activated the scheduler with commitment counts `{'1': 150, '4': 25, '50': 33}`, and did not satisfy any valid Stage A kill criterion.

The EAC Stage B matched manifest is frozen in `reports/eac_vla/stage_b_manifest.json` with canonical payload hash `31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D`. It uses all `20` official tasks, fresh reset seeds `20261213` and `20261214`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities remain unchanged from Stage A.

EAC Stage B completed from the detached run `runs/eac_vla_stage_b/20260714T202334Z` with wrapper exit code `0`, `200 / 200` official LIBERO episodes, zero exceptions, and no confirmatory-test tuning. The result is saved in `reports/eac_vla/stage_b_result.json`, summarized in `reports/eac_vla/stage_b_result.md`, and checkpointed in `reports/eac_vla/stage_b_partial_result.json`.

Stage B decision: `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Frozen Base fixed queue reached `30 / 40`, AAC entropy proxy reached `30 / 40`, EAC full reached `29 / 40`, the no-calibration/no-hysteresis ablation reached `30 / 40`, and fixed short-replan reached `29 / 40`. EAC full preserved action values and activated the scheduler with commitment counts `{'1': 807, '4': 199, '50': 148}`.

EAC full-minus-Base paired delta was `-0.025` with CI `[-0.175, 0.125]`; full-minus-AAC proxy was `-0.025` with CI `[-0.15, 0.1]`; full-minus-ablation was `-0.025` with CI `[-0.175, 0.125]`; and full-minus-fixed-short-replan was `0.0` with CI `[-0.15, 0.15]`.

Final EAC decision: valid current-formulation kill. Do not rescue EAC by retuning `eac_q33_aggressive_1_4_50`, changing thresholds, changing tasks or resets, changing the five-policy list, reinterpreting partial results, or applying any post-hoc expansion.

## Epoch 4 Cycle 11

Exactly three post-EAC candidates were generated and scored in `reports/epoch_4_cycle_11_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_11_prior_mechanism_map.md`. EAC remains archived and may not be rescued.

`G3P-VLA` is selected as a Direct 3D Grounded Point Injection anchored source-gated spatial-conditioning method for frozen SmolVLA. It changes the active method axis from action-queue scheduling to gripper-relative spatial grounding at the action interface.

The selected first comparison is frozen at the design level to five policies: Base, closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic. The closest-prior proxy is a faithful transparent local proxy, not an official reproduction unless exact official equivalence is later established.

The G3P-VLA Researcher A proposal is frozen in `reports/g3p_vla/researcher_proposal.md` with proposal hash `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`. Reviewer B attack is complete in `reports/g3p_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A rebuttal is complete in `reports/g3p_vla/researcher_rebuttal.md` with decision `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is frozen in `reports/g3p_vla/mathematical_mechanism_audit.md` with decision `G3P_MATHEMATICAL_AUDIT_PREREGISTERED`. The preregistration and prototype protocol are frozen in `reports/g3p_vla/preregistration.md` and `reports/g3p_vla/prototype_protocol.md`.

G3P Stage 0 stopped as `DATA_OR_SUPERVISION_FAILURE` in `reports/g3p_vla/development_audit.json`: the material point label collapsed with train fraction `0.9982142857142857` and validation fraction `1.0`. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 12 generated exactly three candidates in `reports/epoch_4_cycle_12_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_12_prior_mechanism_map.md` and selected `CALA-VLA`.

The CALA-VLA Researcher A proposal is frozen in `reports/cala_vla/researcher_proposal.md` with proposal hash `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`.

Reviewer B attack is complete in `reports/cala_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/cala_vla/researcher_rebuttal.md` with decision `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in `reports/cala_vla/mathematical_mechanism_audit.md` with decision `CALA_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/cala_vla/preregistration.md` and `reports/cala_vla/prototype_protocol.md`.

CALA Stage 0 stopped as `DESIGN_FAILURE` in `reports/cala_vla/development_audit.json`: the latent predictability margin was `-0.01171824382857035`, with `action_history_only` beating the full deployment-observable probe. Source legality, split health, label variance, headroom, gradients, Base action validity, and identity-preserving zero delta passed. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 13 generated exactly three candidates in `reports/epoch_4_cycle_13_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_13_prior_mechanism_map.md` and selected `RAR-VLA`.

The RAR-VLA Researcher A proposal is frozen in `reports/rar_vla/researcher_proposal.md` with proposal hash `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`.

Reviewer B attack is complete in `reports/rar_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/rar_vla/researcher_rebuttal.md` with decision `RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in `reports/rar_vla/mathematical_mechanism_audit.md` with decision `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/rar_vla/preregistration.md` and `reports/rar_vla/prototype_protocol.md`.

RAR Stage 0 stopped as `DESIGN_FAILURE` in `reports/rar_vla/development_audit.json`: the residual predictability margin was `-0.03837609884238533`, with `zero_residual` beating the full legal causal probe. Source legality, split health, residual headroom, gradients, Base action validity, and identity-preserving zero delta passed. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 14 generated exactly three candidates in `reports/epoch_4_cycle_14_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_14_prior_mechanism_map.md` and selected `COVI-VLA`.

COVI is anchored to LIBERO-Occ / Viewpoint Imagination, with CamVLA and STRONG-VLA as secondary priors. The selected first comparison is frozen at the design level to five policies: Base under occlusion, VIM-style proxy, COVI full, no-imagined-view ablation, and `random_cutout_clean_retention_baseline`.

The COVI-VLA Researcher A proposal is frozen in `reports/covi_vla/researcher_proposal.md` with proposal hash `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`.

Reviewer B attack is complete in `reports/covi_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The novelty is narrowed to a frozen-SmolVLA identity-preserving complementary-feature adapter for scene-induced occlusion; the VIM proxy must remain transparent, direct two-camera fusion diagnostics are required, and `random_cutout_clean_retention_baseline` remains live.

Researcher A rebuttal is complete in `reports/covi_vla/researcher_rebuttal.md` with decision `COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. Researcher A accepted narrowed novelty, transparent VIM proxy status, direct two-camera fusion diagnostics, the random-cutout simple killer, physical occlusion validation, identity-preserving integration, and no privileged inference.

The COVI mathematical mechanism audit is frozen in `reports/covi_vla/mathematical_mechanism_audit.md` with decision `COVI_MATHEMATICAL_AUDIT_PREREGISTERED`. It freezes variables and tensor shapes, the legal source gate, complementary-target construction, adapter/gate formulas, objective terms, gradient checks, direct two-camera fusion diagnostics, random-cutout simple killer, transparent VIM proxy status, and the six-configuration validation budget. No implementation, validation search, training, manifest freeze, or rollout has happened.

The COVI preregistration and prototype protocol are frozen in `reports/covi_vla/preregistration.md` and `reports/covi_vla/prototype_protocol.md`. They freeze the measured `[64, 960]` visual-token hook, `600 / 600 / 400 / 1200` fit/one-check/validation/confirmatory record partitions, one fixed Stage 0 configuration, one unresolved-result check, and episode-cluster bootstrap false-negative safeguard. Synthetic Stage 0 occlusion is a development proxy only and cannot establish the physical-occlusion claim.

Previous decision: `COVI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. COVI Stage 0 then completed under its frozen protocol. The repaired admissible result is `IMPLEMENTATION_OR_DATA_FAILURE`: the full weighted objective-gradient ratio was `1345.9529990435792:1` against a frozen `100:1` maximum, only two objectives had nonzero pretraining gradients, and output validity was `0.2`. The no-imagined-view target also had no preregistered headroom. This is `COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`; the one-check set stayed sealed and no validation search or rollout ran.

Epoch 4 Cycle 15 generated exactly three prior-anchored candidates in `reports/epoch_4_cycle_15_candidate_generation.md` after the source audit in `reports/epoch_4_cycle_15_prior_mechanism_map.md`. It selected `LIFT-VLA`, Language-Induced Flow Transport, with score `90 / 100`.

LIFT is a frozen, inference-only cross-domain mechanism transfer. It applies conditional-minus-unconditional language guidance at every SmolVLA action-flow step and compares against CAG final-action mixing under the same two-branch budget. The first comparison contains exactly Base, transparent training-free CAG, LIFT full, and last-step-only LIFT. Standard LoRA and a fifth policy are omitted because they do not test the claimed mechanism.

The Researcher A proposal is frozen in `reports/lift_vla/researcher_proposal.md` with hash `3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`.

Reviewer B attack is complete in `reports/lift_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Essential evidence now includes narrow VLA-flow novelty, a scoreable feasible counterfactual manifest, native-flow-space same-noise CAG, a matched-compute last-step ablation, practical-equivalence thresholds, Base-and-CAG headroom, and one-chunk memory/latency feasibility.

Researcher A accepted every constraint in `reports/lift_vla/researcher_rebuttal.md`.

The mathematical audit, preregistration, and prototype protocol are frozen in `reports/lift_vla/mathematical_mechanism_audit.md`, `reports/lift_vla/preregistration.md`, and `reports/lift_vla/prototype_protocol.md`. They freeze native `[1,50,32]` flow tensors, ten steps, native-space same-noise CAG, `20` evaluations for Prior/Ours/ablation, practical-equivalence threshold construction, target-task partitions `[0-3] / [4-6] / [7-9]`, three guidance scales, and zero confirmatory policy decodes in Stage 0.

LIFT Stage 0 is implemented and adjudicated in `reports/lift_vla/stage_0_adjudication.md` as `LIFT_COMPUTE_INFEASIBLE`. The same-scene target-BDDL manifest retained `20 / 20` valid rows, exact identity passed at `0.0`, mechanism activation and practical separation passed, peak CUDA allocation was `0.9200425148010254 GiB`, and LIFT latency was `2.013133036365988` times Base. Executed-action bound validity was only `0.8023809523809524` against the frozen `1.0` requirement. No clipping rescue, headroom rollout, validation search, training, Stage A, or confirmatory evaluation ran; confirmatory policy observations and actions remained zero.

Archived Cycle 15 decision: `LIFT_COMPUTE_INFEASIBLE`. Cycle 15 is closed without rescue. The campaign then advanced through Cycle 16 candidate generation under the active governance.

## Epoch 4 Cycle 16

Exactly three candidates were generated in
`reports/epoch_4_cycle_16_candidate_generation.md` after the primary-source and
closed-axis audit in `reports/epoch_4_cycle_16_prior_mechanism_map.md`.
`IARC-VLA`, Interference-Aware Robustness Consolidation, was selected with
`95 / 100`.

IARC is anchored to STRONG-VLA and Gradient Episodic Memory. The scientific
method is an asymmetric Stage II projected SGD update that protects a paired
perturbation-replay SmolVLA action gradient while clean fidelity is restored.
The low-compute parameterization is the official rank-4 SmolVLA LoRA wrapper;
LoRA is not the contribution.

The proposal is frozen at `reports/iarc_vla/researcher_proposal.md` with hash
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.
Reviewer B required an actual-step mathematical repair, shared flow noise/time,
exact perturbation semantics, a development-only closed-loop headroom screen,
transparent STRONG proxy status, and resource-contention quarantine. Researcher
A accepted all conditions in `reports/iarc_vla/researcher_rebuttal.md`.

The mathematical audit, preregistration, and prototype protocol are frozen in
`reports/iarc_vla/mathematical_mechanism_audit.md`,
`reports/iarc_vla/preregistration.md`, and
`reports/iarc_vla/prototype_protocol.md`. Stage II now uses explicit SGD with
zero momentum and weight decay, so the realized conflict projection has the
claimed first-order geometry.

The first comparison is exactly Base, transparent STRONG proxy, IARC full,
unprojected joint replay, and matched standard clean-only LoRA. Stage 0A is the
only authorized next step: `20` micro-fit steps, `40` independent gradient
pairs, `40` validation diagnostics, and zero confirmatory decodes.

The user-reported Windows Efficiency Mode interval is recorded in
`reports/resource_contention_intervals.json`. No active Linux worker was found.
The already completed EAC Stage B result passed PID, exit-code, JSON,
`200 / 200` row, exception, synchrony, duplicate-key, and manifest checks and is
accepted without rerun. Timing/resource evidence with unknown overlap is not
final paper evidence.

IARC Stage 0A completed with child and wrapper exit code `0`, `40 / 40`
gradient pairs, `40 / 40` validation rows, zero exceptions, zero duplicate or
manifest mismatches, and zero confirmatory decodes/actions. Conflict activation
was `18 / 40` across all four families; projection constraints passed `18 /
18`, agreeing rows were unchanged `22 / 22`, identity and reload error were
`0.0`, and the Base hash was unchanged.

The frozen hard stop is dataset-range action validity: `12 / 40 = 0.30`
against required `1.0`, with `28` invalid pair rows. The adjudication is
`reports/iarc_vla/stage_0a_adjudication.md`. This is an implementation/action-
validity failure, not a scientific kill. Do not clip, widen bounds, run the
one-check, run Stage 0B, or start validation search.

IARC remains closed unchanged as `IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
Cycle 17 generated exactly three candidates and selected `FAMR-VLA` with
`93 / 100`. Proposal hash:
`96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69`.
The frozen package is under `reports/famr_vla/`. Stage 0A completed as
`FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED`: all `24` discovery rows and
`150 / 150` source-success demonstrations passed provenance and semantic
checks with zero overlap or duplicate keys. The rank-4 adapter completed `20 /
20` optimizer steps and reduced fixed-subset loss from `0.7321685557253659` to
`0.6487047945459684`, a relative reduction of `0.11399528227036353`.

Identity, Base hash, group coverage, coefficient scaling, and disk reload all
passed. Peak CUDA allocation was `1.0808053016662598 GiB`; exceptions,
validation/test decodes, and confirmatory observations/actions were zero.

The frozen 300-step endpoint then completed `300 / 300` optimizer steps and
`2400 / 2400` task-balanced discovery microbatches with zero exceptions or
duplicates. Fixed-subset loss fell by `0.7775824820789773`, action effect was
active on `24 / 24` rows, and gradient, Base-hash, checkpoint-reload, manifest,
and memory gates passed.

The Base-relative action-validity gate failed. Outside-`[-1,1]` frequency was
`0.1130952380952381` versus permitted `0.08738095238095238`; p99 exceedance
was `0.09376012921333322` versus permitted `0.04096377015113834`. This is
`FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, an
`IMPLEMENTATION_OR_DATA_FAILURE`, not a scientific kill. No clipping,
headroom, validation search, rollout, or confirmatory rescue is allowed.

The FAMR endpoint remains closed unchanged as
`FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

Cycle 18 generated exactly three candidates and selected `PCAV-VLA` with
`95 / 100`. TACO is the closest positive prior and ProgressVLA supplies the
progress-consequence extension. Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.
The proposal, review, rebuttal, mathematical audit, preregistration, and
prototype protocol are frozen under `reports/pcav_vla/`.

PCAV Stage 0A completed `96 / 96` rows after preserving the first 24 rows and
resuming only 72 missing keys. Exceptions, duplicates, and manifest mismatches
were zero. Base identity, mapping, source health, partition, reload, and Base
hash checks passed; confirmatory observations/actions remained `0 / 0`.

Only `7 / 96 = 0.0729167` rows had a valid alternative at least 5% better
than Base, and median relative reduction over improvable rows was `0.0166833`.
Both miss the frozen 25% and 5% headroom requirements. The decision is
`PCAV_STAGE_0A_NO_USABLE_HEADROOM`; Stage 0B and PCAV rescue are forbidden.

Historical decision: `PCAV_STAGE_0A_NO_USABLE_HEADROOM`.
Current cycle: `19`. Current stage:
`epoch_4_cycle_19_candidate_search_pending`.

## Epoch 4 Cycle 19 Completion

SPARC Stage 0A is closed unchanged as
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`.
The final worker completed `2 / 2` planned synthetic action rows with exit code
zero, zero exceptions, and zero duplicate, missing-manifest, or extra keys.
Hook identity, serialization, and mechanism activation passed, but both action
rows failed the frozen Base-relative action-validity gates. The one mechanical
repair was consumed. No labeled activation fit, Stage 0B, validation search,
rollout, or confirmatory access occurred. SPARC rescue is forbidden.

## Epoch 4 Cycle 20

Exactly three candidates were generated in
`reports/epoch_4_cycle_20_candidate_generation.md` after the primary-source,
historical-overlap, and local-data audit in
`reports/epoch_4_cycle_20_prior_mechanism_map.md`.

`NICE-VLA`, Normalized-Innovation Corrective Execution for VLAs, was selected
with `96 / 100`. Its closest positive prior is VLA-Corrector at official source
commit `9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`. NICE preserves the prior's
frozen mean, queue, truncation, recovery, OGG, and action semantics, and changes
only the monitor through an action-conditioned heteroscedastic covariance,
normalized innovation, and episode-cluster split-conformal threshold.

The Researcher A proposal is frozen with hash
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.
Reviewer B attack, Researcher A rebuttal, mathematical mechanism audit,
preregistration, and prototype protocol are frozen under `reports/nice_vla/`.

The first comparison is exactly Base, the official-code-derived VLA-Corrector
proxy, NICE full, a shared-mean global-error ablation, and fixed short-horizon
replanning. Validation search is capped at six configurations. Stage 0A is the
only authorized next step: `128` discovery-only latent/action pairs, tiny model
and algebra/calibration smoke, exact Base passthrough, zero validation or
confirmatory reads, and one mechanical implementation repair at most.

Current cycle: `20`. Current stage:
`epoch_4_cycle_20_nice_stage_0b1_pending`.

NICE Stage 0A completed `128 / 128` discovery-only latent/action pairs with
exit code zero, zero exceptions, and exact manifest/partial key equality. The
official source commit/license, `[128,960]` latent mapping, 7D actions,
gradients, diagonal and rank-8 algebra, conformal fixtures, checkpoint reload,
and monitor-disabled Base passthrough all passed. Validation, confirmatory,
outcome, and rollout reads were zero. This is an implementation/data pass, not
a scientific performance result.

The separate Stage 0B1 offline development protocol is frozen at
`reports/nice_vla/stage_0b1_execution_protocol.md`. Only its fixed 1792-pair
observability audit is authorized next.

NICE Stage 0B1 then completed all `1792 / 1792` pair keys with exact manifest
equality but stopped during the action-regime diagnostic. Under the frozen
deadband `2.0`, two validation tasks have counts `[80,0]`, so required
action-regime supervision is collapsed. The adjudicated decision is
`NICE_STAGE_0B1_DATA_FAILURE_COLLAPSED_ACTION_REGIME_CONTRAST`, not a
scientific kill. No Stage 0B2, deadband change, resampling, task replacement,
or NICE rescue is allowed.

Current cycle: `21`. Current stage:
`epoch_4_cycle_21_candidate_search_pending`.
