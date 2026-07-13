# Autonomous Until Paper State

Date: 2026-07-12 KST

Active governance: `reports/current_research_governance.md`

Branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED`

Current epoch: `4`

Current cycle: `1`

Current stage: `epoch_4_cycle_1_selection_pending`

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

Next action: begin Epoch 4 Cycle 1 candidate generation under the post-PSE problem-first, external-prior-early, mathematically justified research-design gate.
