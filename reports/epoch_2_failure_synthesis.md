# Epoch 2 Failure Synthesis

Date: 2026-07-12 KST

Decision: `EPOCH_2_SYNTHESIZED_KILLS_EPOCH_3_PIVOT_REQUIRED`

This is not a terminal campaign decision.

## Epoch 2 Outcomes

`PTC-VLA` was killed at Stage A:

- final decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`
- full: `0 / 10`
- strongest baseline: frozen SmolVLA, `3 / 10`
- active mechanism: transition-context action generation

`SACF-VLA` was killed at Stage A:

- final decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`
- full: `0 / 10`
- strongest baseline: frozen SmolVLA, `7 / 10`
- active mechanism: semantic action-factor prefix

`OCFN-VLA` was killed at expanded Stage B:

- final decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`
- full: `26 / 80`
- strongest baseline: zero-noise SmolVLA, `27 / 80`
- paired upper confidence bound versus strongest baseline: `0.0625`
- active mechanism: task-conditioned flow-noise prior

## Shared Failure Pattern

Epoch 2 repeatedly tried to improve the frozen SmolVLA action-generation surface without changing the environment evidence source or task-success supervision enough:

- `PTC-VLA` changed temporal transition representation and action generation.
- `SACF-VLA` changed semantic task-prefix representation and a learned action generator.
- `OCFN-VLA` changed latent flow-noise initialization and outcome-conditioned selection.

All three mechanisms acted, but none delivered a useful closed-loop improvement over simple baselines. The common failure is not a null implementation. It is that lightweight action-surface interventions on the same SmolVLA/LIBERO evidence stream were either harmful or explained by simple baselines.

## Epoch 3 Constraints

Epoch 3 must change at least two core dimensions relative to Epoch 2. At minimum, it should avoid:

- direct small action heads;
- semantic or phase prefixes;
- action residual correction;
- fixed or selected flow-noise priors;
- ranker, verifier, barrier, filter, damping, or simple action-statistic baselines as the main novelty.

Epoch 3 should change at least two of:

- core problem;
- representation;
- supervision;
- objective;
- policy generation;
- inference-time intervention;
- data source;
- claim.

Preferred direction: use a different evidence source or claim target before touching action generation again. Examples include demonstration-structure diagnostics, cross-backbone residual analysis, or a failure-taxonomy claim that can be tested without inventing another thin action modifier.

Next action: generate exactly three Epoch 3 Cycle 1 candidates, select exactly one, freeze and hash the proposal, run Reviewer B attack, and continue under `reports/current_research_governance.md`.
