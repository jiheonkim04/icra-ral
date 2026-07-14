# Autonomous Until Paper Decision

Date: 2026-07-14 KST

Current campaign decision: `EPOCH_4_CYCLE_3_FANG_VALIDATION_SELECTED_STAGE_A_PENDING`

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

The development audit passed. The uncalibrated gate validation failure is archived, and the calibrated six-config validation search selected `fang_c01` with clean action validity and bounded activation.

Next action: implement and run the preregistered FANG-VLA Stage A closed-loop comparison without changing the validation-selected configuration.
