# Autonomous Until Paper Decision

Date: 2026-07-14 KST

Current campaign decision: `EPOCH_4_CYCLE_6_MTF_ADAPTER_TRAINING_PENDING`

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
