# Autonomous RA-L Campaign State

Date: 2026-07-12 KST

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED`

Current epoch: `4`

Current cycle: `1`

Current stage: `epoch_4_cycle_1_selection_pending`

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

Next action: apply the post-PSE research-design governance update, add validation tests, and begin Epoch 4 Cycle 1 candidate generation.
