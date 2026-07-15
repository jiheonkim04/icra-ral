# FAMR-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Decision: `FAMR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## 1. Construction And Selection Separation

For every fixed configuration, all action scales, group responses, solver
coefficients, and response-model statistics are constructed from discovery
rows only. Validation rows only score already materialized configurations.
No coefficient is refit on validation. The six-configuration list is frozen.

Offline test rows and confirmatory reset identities remain unread until one
configuration, one prior, and one ablation are frozen.

## 2. Exact LoRA Group Scaling

For each targeted linear layer the effective update is

`Delta W = (alpha_lora / r) B A`.

FAMR leaves `A`, `alpha_lora`, `r`, and the frozen base unchanged and writes
`B' = c_m B` for every layer in group `m`. Therefore

`Delta W' = c_m Delta W`.

The executable audit will verify:

- `c=0` equals Base under an identical flow draw;
- all `c=1` equals the standard-LoRA endpoint;
- one-group `c` has output linearity error below numerical tolerance at the
  layer output;
- checkpoint save/reload preserves coefficients and actions exactly within
  floating-point tolerance.

Any failure is implementation failure and blocks policy rollout.

## 3. Nonlinear Response Fidelity

On independent validation rows, compare the linear prediction
`a_hat_i = a_0i + D_i c` with the directly decoded materialized action `a_i(c)`.

Report normalized RMSE, median and p95 relative error, per-dimension error, and
Spearman correlation between predicted and direct Base-relative action norms.

The response model is considered informative only if:

- normalized RMSE is no larger than `0.50`;
- median relative error is no larger than `0.50`;
- norm correlation is at least `0.50` when norm variance exceeds `1e-8`;
- direct policy ordering on target fit and retention agrees with the predicted
  ordering for at least `4 / 6` configurations.

If variance is too small to define correlation, the mechanism is nonacting.
One repeated-draw fidelity check is allowed only when stochasticity, not mean
response, dominates the uncertainty.

## 4. Task Provenance

Stage 0 writes the official 40 task strings from `lerobot/libero`, the three
selected LIBERO-90 task strings, their set intersection, source paths, file
hashes, and checkpoint dataset metadata. Intersection must be zero.

Semantic similarity to a pretrained skill is allowed and desirable transfer;
exact task-identity overlap is not.

## 5. HDF5 Semantic Audit

Before optimization, the runner records for every target task:

- demonstration and frame counts;
- camera keys, shapes, dtypes, orientation, and value ranges;
- state keys, dimensions, and mapping to official policy state;
- action shape, finite fraction, per-dimension min/max/IQR, and gripper values;
- task-language string and BDDL identity;
- duplicate episode/frame hashes;
- train/validation/test episode intersection;
- expert terminal success metadata or synchronous replay success.

The exact official environment preprocessor and policy pre/postprocessors are
used. No ad hoc action clipping, camera replacement, or state imputation is
allowed.

## 6. Headroom

Discovery closed loop uses the three frozen tasks and reset seeds
`20261701-20261704`, for `12` paired cases per policy.

Base headroom passes when Base fails at least `3 / 12`. The standard-LoRA
endpoint must pass subset fit, act nontrivially, remain action-valid, and be no
more than `4 / 12` paired cases worse than Base. The endpoint need not already
beat Base; FAMR's question includes whether retention-aware merging extracts a
better point from a useful but disruptive endpoint.

Base failure below `3 / 12` is no headroom. An endpoint that cannot fit or act
is parameterization/implementation failure, not a scientific FAMR kill.

## 7. Action Validity

This Cycle 17 gate is new and does not alter IARC.

For every directly decoded validation action:

- finite fraction must be `1.0`;
- absolute max must be at most `max(1.25, base_max + 0.05)`;
- fraction outside `[-1,1]` may not exceed matched Base by more than `0.01`;
- p99 exceedance beyond `[-1,1]` may not exceed Base by more than `0.02`;
- the official synchronous simulator must accept actions without timeout,
  exception, or semantic conversion change.

No clipping or postprocessor change is permitted after results.

## 8. RETAIN And Shrinkage Controls

`retain_scalar_proxy` is always labeled a transparent local core proxy. It uses
the same effective endpoint and scalar `alpha in {0.5,0.8}` selected on
validation.

After FAMR selection, an offline equal-mean scalar diagnostic uses
`alpha = mean(c_m)` on the same validation rows. If it matches FAMR within the
frozen practical-equivalence threshold on target fit, retention, and action
validity, the method is `SIMPLE_BASELINE_EXPLAINS_METHOD` before rollout. If it
does not, it remains a diagnostic and does not become a sixth policy.

## 9. Rollout Partitions

- discovery/headroom resets: `20261701-20261704` on each target task;
- validation resets: `20261711-20261713` on each target task;
- Stage A confirmatory resets: `20261721-20261724` on each target task;
- Stage B confirmatory resets: `20261731-20261744` on each target task.

Original-task retention tasks and reset identities are frozen in separate
manifests before their first use. No target or retention reset identity appears
in more than one partition.

Every result key is `(policy, suite, task, reset_identity)`. Acceptance requires
zero duplicates, zero missing/extra manifest keys, synchronous simulator mode,
zero exceptions, unchanged action semantics, and valid checkpoint hashes.

## 10. False-Negative Decisions

Before rollout, every stop is classified as exactly one of
`FATAL_PREIMPLEMENTATION`, `ROBUST_EMPIRICAL_DESIGN_FAILURE`,
`UNDERPOWERED_OR_UNRESOLVED`, or `IMPLEMENTATION_OR_DATA_FAILURE`.

Only the first two can permanently kill the current formulation. One cheap
repeated-draw response check is allowed for unresolved stochastic fidelity. No
method, task, coefficient, threshold, or identity changes are allowed after
confirmatory results.

## Rebuttal Decision

The method remains one action-function merge objective with one necessary
retention term and one key ablation. Reviewer B's blocking requirements are
accepted and move to mathematical audit.
