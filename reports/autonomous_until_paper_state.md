# Autonomous Until Paper State

Date: 2026-07-14 KST

Active governance: `reports/current_research_governance.md`

Branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_4_CYCLE_3_FANG_STAGE_A_TIE_TO_STAGE_B`

Current epoch: `4`

Current cycle: `3`

Current stage: `fang_vla_stage_b_pending`

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

Current stage: Stage B pending. Do not alter `fang_c01`, gate tau, task list, identity partitions, baselines, or thresholds after the Stage A tie.
