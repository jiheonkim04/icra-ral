# Autonomous RA-L Campaign State

Date: 2026-07-12 KST

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_2_CYCLE_2_SACF_KILLED_PIVOT_REQUIRED`

Current epoch: `2`

Current cycle: `3`

Current stage: `epoch_2_cycle_3_selection_pending`

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

## Next Action

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Start Epoch 2 Cycle 2 candidate generation and select one technically distinct method under the current governance.
