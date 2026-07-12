# PhaseBarrier-VLA Power And Sample Plan

Date: 2026-07-12 KST

## Sample Choice

The repair uses the minimum requested sample: `20` episodes per policy across the two original targeted tasks.

Allocation:

| Task | Eval identities | Episodes per policy |
| --- | ---: | ---: |
| `libero_spatial/task_4` | `10` | `10` |
| `libero_10/task_4` | `10` | `10` |
| Total | `20` task/reset cases | `20` |

With five variants, total planned episodes are `100`, below the maximum `300`.

## Why Not Larger

The original average episode runtime was roughly `25` to `27` seconds, with official max horizons of `280` and `520` steps. A `100` episode repeat is already a material increase over the original `10` total episodes and remains locally bounded. A larger `200` to `300` episode repeat would be more expensive while still centered on the same two hard tasks and one frozen method.

## Statistical Interpretation

This sample can adjudicate large prototype-level effects and obvious baseline or ablation domination. It cannot prove a small effect precisely.

The fixed GO gate remains:

- at least `5` percentage-point task-balanced gain over strongest baseline;
- full beats the no-phase ablation;
- or paired Route B evidence plus at least `10%` relative failure-rate reduction.

If full improves numerically but confidence remains weak and it does not clearly beat the ablation, the decision must be `PHASEBARRIER_RESULT_STILL_INCONCLUSIVE` or an appropriate baseline/ablation kill. No second PhaseBarrier repeat is allowed.

## Paired Accounting

Each variant is evaluated on the identical set of task/reset cases:

- `libero_spatial/task_4`: identities `20260712` through `20260721`;
- `libero_10/task_4`: identities `20260712` through `20260721`.

Paired win/loss/tie will be computed per task/reset case for:

- full versus frozen;
- full versus simple global damping;
- full versus no-phase ablation.

Primary metric remains task-balanced official closed-loop success.
