# PTC-VLA Preregistration

Date: 2026-07-12 KST

Governance: `reports/current_research_governance.md`

## Frozen Proposal

Proposal hash: `15A3027E02DFE46EF2B56461A245307E9588F13431A1C92952DDD76683964CC7`

## Variants

1. `frozen_smolvla`
2. `global_mean_action`
3. `phase_mean_action`
4. `ptc_no_transition_ablation`
5. `ptc_full`

## Stage A

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Evaluation identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

This gives `10` paired episodes per policy.

Metrics:

- successes/counts;
- task-balanced success rate;
- per-task success;
- paired task/reset allocation;
- exception count;
- mechanism activation: mean transition-context norm and mean action delta versus the no-transition ablation;
- runtime and CUDA memory.

## Stage A Decision Rules

Permanent kill at Stage A only if one of these holds:

- implementation or data mechanism invalid;
- `ptc_full` is at least `0.30` absolute task-balanced success below the strongest baseline or key ablation;
- `ptc_full` has `0 / 10` success while a paired baseline has at least `4 / 10`;
- an oracle/upper-bound check proves no usable headroom;
- exact trivial equivalence is demonstrated.

Otherwise:

- positive, tied, mixed, or one/two episode negative result advances to Stage B;
- Stage B must use at least 40 paired episodes per key policy.

## No Rescue

If killed, do not rescue this formulation through hidden-dimension tuning, alternate phase bins, different mean-action smoothing, or a renamed state-only MLP.
