# Autonomous Until Paper State

Date: 2026-07-12 KST

Active governance: `reports/current_research_governance.md`

Branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_2_CYCLE_1_PTC_KILLED_PIVOT_REQUIRED`

Current epoch: `2`

Current cycle: `2`

Current stage: `epoch_2_cycle_2_selection_pending`

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

## Resume

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/autonomous-until-paper-governance-v2
type reports\current_research_governance.md
```
