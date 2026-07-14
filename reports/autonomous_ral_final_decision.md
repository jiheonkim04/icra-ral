# Autonomous RA-L Decision

Date: 2026-07-14 KST

Current decision: `DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

This is not a terminal state under the active governance.

Active governance: `reports/current_research_governance.md`

The prior fixed-cycle terminal stop is procedurally invalid under the current Goal. Epoch 1 is corrected as a completed related-method set that requires an Epoch 2 pivot.

Corrected adjudication:

- Cycle 1 `DICD-VLA`: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- Cycle 2 `FEDO-VLA`: `VALID_CURRENT_FORMULATION_KILL`
- Cycle 3 `GCAP-VLA`: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full PTC reached `0 / 10` versus frozen SmolVLA `3 / 10`, with zero exceptions and active transition mechanism.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full SACF reached `0 / 10` versus frozen SmolVLA `7 / 10`, with zero exceptions and active semantic mechanism.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `80` paired episodes per key policy with zero exceptions and active mechanism. OCFN full reached `26 / 80`, zero-noise SmolVLA reached `27 / 80`, and the paired upper confidence bound for full minus zero-noise was `0.0625`.

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: full CBFD reached `0 / 10` while frozen SmolVLA reached `7 / 10`, with zero exceptions and active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

Epoch 4 Cycle 3 selected and preregistered `FANG-VLA`. Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`.

The development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all five policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. Full-minus-base paired delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

Epoch 4 Cycle 4 selected and preregistered `EvoState-VLA`. Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`.

Stage 0 stopped before rollout as `AUDIT_STOP_DESIGN_FAILURE`: the full transition model improved only `0.024689` over an actionless model, below the preregistered `0.05` threshold.

Epoch 4 Cycle 5 selected and preregistered `RAC-VLA`, a Reflective VLA-anchored frozen-policy action-consequence calibration method. Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`.

RAC Stage 0 passed without rollout: full action-consequence validation accuracy `0.585745` beat action-only `0.368496` and no-consequence `0.374483`, with margin `0.211262`; clean action delta p95 was `0.0`. The six-config validation search selected `rac_h4_a0.05` with score `0.508926`.

Stage A completed `50 / 50` episodes with zero exceptions. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. This was `STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`, not a valid Stage A kill.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared manifest. RAC full reached `1 / 40`, shifted Base reached `1 / 40`, Reflective-history proxy reached `1 / 40`, no-consequence ablation reached `2 / 40`, and online diagonal inverse-gain reached `2 / 40`. Full-minus-ablation paired delta was `-0.025` with CI `[-0.125, 0.050]`; full-minus-simple-baseline paired delta was also `-0.025` with CI `[-0.125, 0.050]`.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue or retune RAC.

The post-RAC governance update is installed and active. It requires future methods to maximize the probability of an honest paper-worthy positive result through stronger positive-prior-anchored design, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded validation search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

Epoch 4 Cycle 6 generated exactly three post-RAC candidates and selected `MTF-VLA`. Proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`.

MTF-VLA is a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training. The first comparison is frozen to Base, FrameSkip proxy, MTF full, no-retention ablation, and uniform retained-ratio LoRA.

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

The DAGR Stage B manifest is now frozen as `DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`: all `20` official tasks, reset seeds `20261207` and `20261208`, `40` paired cases per policy, `200` total episodes, canonical hash `2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B`, and the unchanged five-policy comparison. Next action: launch DAGR Stage B official WSL rollout.
