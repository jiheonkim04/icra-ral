# FAMR-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Decision: `FAMR_MATHEMATICAL_AUDIT_PREREGISTERED`

## 1. Variables, Shapes, Units, And Sources

| Variable | Shape | Units | Source |
| --- | --- | --- | --- |
| `x_i` | policy batch | mixed | legal RGB, proprio, task text |
| `a_0i` | `[7]` | LIBERO environment action | frozen Base, fixed flow draw |
| `a_i*` | `[7]` | LIBERO environment action | target demonstration |
| `Delta_m` | parameter group | parameter units | trained endpoint minus Base |
| `d_im` | `[7]` | environment action | one-group direct response |
| `D_i` | `[7,M]` | environment action | stacked group responses |
| `c` | `[M]` | dimensionless | bounded merge coefficients |
| `s` | `[7]` | environment action | discovery action scale |
| `a_hat_i(c)` | `[7]` | environment action | linear response prediction |

Postprocessed action dimensions are translation `0:3`, rotation `3:6`, and
gripper `6`. The method never treats deterministic 7D actions as probability
distributions.

## 2. Frozen Action Scale

For target discovery demonstrations, compute per dimension

`s_j = max(IQR({a_ij*}), 0.05)`.

The floor avoids exploding normalized gripper or low-motion dimensions. `s` is
computed once from training episodes `0-34`, saved, and never updated from
validation or test rows.

## 3. Group Response Construction

For a shared native SmolVLA noise tensor and time schedule:

`d_im = postprocess(f(x_i; theta_0 + Delta_m))`
`       - postprocess(f(x_i; theta_0))`.

The shared draw removes avoidable flow-sampling variance. Every one-group
checkpoint is direct, disk reloadable, and has all non-group coefficients zero.

Coarse `M=3` and fine `M=5` partitions are exactly those in the proposal. A
parameter-assignment manifest must prove one-to-one coverage.

## 4. Primary Target Objective

Define normalized residual

`r_i(c) = (a_0i + D_i c - a_i*) / s`.

With Huber threshold `delta = 1`:

`L_target(c) = (1 / (7 |T|)) sum_{i,j} huber_1(r_ij(c))`.

Scale is dimensionless and approximately order one. Gradients flow only to
`c`; policy tensors and actions are cached and stop-gradient.

Intended effect: preserve task-vector groups whose action response moves Base
toward held-out new-task demonstration actions.

Simpler alternative: standard LoRA, `c = 1`, and scalar RETAIN, `c_m = alpha`.

## 5. Necessary Retention Objective

For original-task discovery rows `R`:

`L_retain(c) = (1 / (7 |R|)) sum_i ||D_i c / s||_2^2`.

This is dimensionless. It penalizes predicted action drift from the frozen
generalist policy without requiring privileged labels or a probability model.

Intended effect: attenuate task-vector groups whose functional response is
large on original tasks.

Required ablation: `lambda = 0` with every other construction unchanged.

## 6. Full Objective And Solver

`L_FAMR(c; lambda) = L_target(c) + lambda L_retain(c)`,

subject to `0 <= c_m <= 1`.

Use deterministic projected Adam for `500` coefficient steps, learning rate
`0.05`, initialized at `c_m=0.5`. Clamp only coefficients after every step;
never clamp policy actions. Run float64 coefficient optimization on CPU.

The six selectable configurations are fixed in the proposal. No broad grid or
seed search is allowed.

## 7. Term Magnitude And Gradient Audit

Before solving each FAMR configuration, report at `c=0.5`:

- `L_target`, `L_retain`, and `lambda L_retain`;
- `||grad_c L_target||_2`;
- `||grad_c lambda L_retain||_2`;
- gradient cosine when both norms exceed `1e-12`;
- ratio of larger to smaller nonzero gradient norm.

A ratio above `100` is not an automatic failure, but it must be reported and
the affected configuration cannot be selected unless direct validation shows
both target action and retention effects. No coefficient is changed after this
audit outside the frozen six-config search.

## 8. Materialized Policy

For every layer in group `m`, write `B'_m = c_m B_m` and save a new adapter.

Identity requirements under identical flow draws:

- all-zero coefficients versus Base max absolute error `<= 1e-6` after
  postprocessing;
- all-one coefficients versus standard-LoRA max absolute error `<= 1e-6`;
- save/reload action max absolute error `<= 1e-6`;
- frozen Base parameter hash unchanged.

## 9. Response Fidelity

On validation rows, compare

`a_hat_i(c) = a_0i + D_i c`

with direct materialized `a_i(c)`. Report normalized RMSE, relative errors,
per-dimension errors, norm correlation, and configuration ordering agreement.
Pass thresholds are frozen in the rebuttal.

The response predictor selects candidates; all scientific conclusions use
direct policy outputs and closed-loop results.

## 10. Practical Equivalence

Repeated Base inference with identical observations and independently frozen
flow seeds estimates stochastic action variation on discovery only. Define the
practical action threshold as the larger of:

- `1e-4` postprocessed action L2;
- the discovery p95 repeated-Base same-observation L2.

FAMR must differ from scalar RETAIN and target-only above this threshold on at
least `25%` of relevant validation rows. Otherwise exact or practical
equivalence blocks a novelty claim.

## 11. Action Validity And Clean Retention

Use the absolute and Base-relative validity rules in the rebuttal. Report
translation, rotation, gripper, total action delta, outside-`[-1,1]` frequency,
exceedance magnitude, and simulator acceptance.

Closed-loop original-task retention is required because action drift alone is
not task success.

## 12. No KL

No KL divergence is used. SmolVLA flow vectors and deterministic postprocessed
7D actions are not normalized probability distributions. Huber and squared
functional drift directly match the quantities and units needed by the claim.

## 13. Compute And Inference

FAMR adds no inference input, branch, gate, memory, or policy call. Once merged,
its architecture, parameter count, number of flow steps, and inference budget
match the standard-LoRA endpoint. Merge fitting is CPU-side cached-action
optimization.

Timing, throughput, and utilization are diagnostic only until resource-overlap
metadata proves eligibility.

## Audit Decision

The objective is mathematically valid, minimum-sufficient, directly ablatable,
and separated from the low-compute LoRA realization. It may proceed to a frozen
preregistration and executable Stage 0 protocol.
