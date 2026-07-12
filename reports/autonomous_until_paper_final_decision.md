# Autonomous Until Paper Decision

Date: 2026-07-12 KST

Current campaign decision: `EPOCH_2_CYCLE_2_SACF_KILLED_PIVOT_REQUIRED`

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

Start Epoch 2 Cycle 3 candidate generation under `reports/current_research_governance.md`.
