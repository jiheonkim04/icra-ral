# TSC-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Method: `TSC-VLA`

Proposal hash:
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`

Decision: `TSC_MATHEMATICAL_AUDIT_PREREGISTERED`

This audit is frozen before preregistration, implementation, validation search,
or rollout.

## Variables And Shapes

Constants:

- `H = 50`: SmolVLA action horizon.
- `D = 7`: LIBERO action dimension.
- `B`: batch size.
- `epsilon = 1e-6`.

Inputs:

- `x`: deployment-observable policy input, containing current RGB streams,
  proprioception, and language instruction.
- `z = f_frozen(x) in R^[B,F]`: frozen feature vector or cached feature bundle
  derived only from deployment inputs.
- `A_B in R^[B,H,D]`: frozen Base SmolVLA decoded action chunk after the
  official postprocessor.
- `A_E in R^[B,H,D]`: aligned demonstration expert action chunk for training or
  validation only.
- `V in {0,1}^[B,H,1]`: valid future-step mask. Invalid padded steps are
  excluded from losses and metrics.

Derived discovery-only statistics:

- `S_d in R_+^[D]`: robust per-dimension residual scale, computed as
  `median(|A_E - A_B|)` on discovery records plus `epsilon`.
- `Tau_d in R_+^[D]`: per-dimension positive-label threshold, computed from
  discovery records only as the `0.80` quantile of
  `|A_E - A_B| / S_d` over valid steps.
- `rho in [0,1]`: discovery positive-cell fraction after applying `Tau_d`.

Labels:

- `R = A_E - stopgrad(A_B) in R^[B,H,D]`.
- `Y in {0,1}^[B,H,D]`, where
  `Y_b,h,d = 1[ V_b,h,0 = 1 and |R_b,h,d| / S_d >= Tau_d ]`.

Model outputs:

- `L_theta(z, A_B) in R^[B,H,D]`: mask logits.
- `P_theta = sigmoid(L_theta) in [0,1]^[B,H,D]`: soft mask probabilities.
- `M_theta = 1[P_theta >= eta] in {0,1}^[B,H,D]`: hard inference mask, with
  `eta = 0.5` for Stage 0 unless a later bounded validation search freezes a
  different threshold before closed-loop testing.
- `Delta_phi(z, A_B, P_theta) in R^[B,H,D]`: continuous completion field.

Action construction:

- Training soft action:
  `A_soft = stopgrad(A_B) + alpha * P_theta * Delta_phi`.
- Inference action:
  `A_TSC = stopgrad(A_B) + alpha * M_theta * Delta_phi`.

`alpha in [0,1]` is the correction scale. Stage 0 may evaluate a fixed
diagnostic `alpha = 0.1` for action-delta/validity smoke. Any nonzero final
`alpha` for rollout must be selected only by bounded validation search.

No ad hoc clipping is allowed. If the official SmolVLA/LIBERO stack exposes an
official projection or validity transform, it may be used and documented. If
not, action validity is a measured pass/fail property of `A_TSC`.

## Objective Terms

All Huber terms use normalized residuals divided by `S_d`. Default Huber
threshold is `delta = 1.0` in normalized units.

### Mask Objective

`L_mask = mean_valid BCEWithLogits(L_theta, Y; pos_weight=w_pos)`.

`w_pos = N_neg / max(N_pos, 1)` is computed from the current training split and
logged. If `N_pos = 0` or `N_neg = 0`, Stage 0 must stop as
`TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

Gradient path: updates mask predictor parameters `theta`; no gradient flows
into `A_B`, frozen SmolVLA, labels, thresholds, or discovery statistics.

Intended effect: make the mask identify the sparse time-dimension cells where
Base is likely wrong.

Simpler alternative: magnitude-only mask from `|A_B|`, residual-size oracle
diagnostic on development data, per-timestep gate, and per-dimension gate.

Required ablation: `tsc_no_targeted_mask_ablation`.

### Completion Objective

`L_comp = sum_valid Y * Huber((A_soft - A_E) / S_d) / max(sum_valid Y, 1)`.

Gradient path: updates `theta` through `P_theta` and updates completion
parameters `phi`; no gradient into frozen Base or expert actions.

Intended effect: train the completion field to repair selected action cells,
not globally rewrite the chunk.

Simpler alternative: direct dense residual regression, per-dimension residual
MLP, low-rank residual adapter, and smoothing baseline.

Required ablation: completion with non-targeted masks under comparable capacity.

### Clean Retention Objective

`L_ret = sum_valid (1 - Y) * Huber((A_soft - stopgrad(A_B)) / S_d) / max(sum_valid (1 - Y), 1)`.

Gradient path: updates `theta` and `phi`; `A_B` is stop-gradient.

Intended effect: preserve Base behavior on cells not labeled as likely errors.

Simpler alternative: global action-delta penalty.

Required audit: report unmasked-cell mean/p95/max delta and clean validation
behavior.

### Sparsity Calibration Objective

`L_sparse = |mean_valid(P_theta) - rho|`.

Gradient path: updates `theta`.

Intended effect: prevent a degenerate mask that activates everywhere or
nowhere while matching the discovery positive rate.

Simpler alternative: fixed top-k mask. If `L_sparse` dominates gradients or is
unnecessary, Stage 0 may classify this as design/optimization failure or defer
coefficient choice to bounded validation. It may not be tuned on confirmatory
outcomes.

## Total Objective

`L_total = lambda_mask L_mask + lambda_comp L_comp + lambda_ret L_ret + lambda_sparse L_sparse`.

Initial diagnostic coefficients for Stage 0 small-batch gradient audit:

- `lambda_mask = 1.0`
- `lambda_comp = 1.0`
- `lambda_ret = 1.0`
- `lambda_sparse = 0.1`

Before any training, Stage 0 must report:

- each term magnitude on a small batch;
- gradient norm for `theta`;
- gradient norm for `phi`;
- max finite gradient;
- ratio of largest to smallest nonzero objective gradient norm;
- frozen-parameter gradient count.

If one objective overwhelms another by more than an order of magnitude and
causes nonacting or destructive behavior, stop or move the coefficient choice
to the bounded validation budget. Do not silently tune after results.

## KL And Distribution Distances

No KL divergence is used. TSC actions are deterministic continuous chunks, not
probability distributions. SmolVLA flow vectors are not automatically normalized
action distributions. If a later probabilistic mask distribution is introduced,
the mathematical audit must be revised before confirmatory testing with valid
`p`, `q`, support, estimator, KL direction, and gradient flow.

## Closest-Prior Proxy

Policy 2 is:

`ts_mask_continuous_proxy_or_official_ts_mask_vla_if_installed`.

Until official TS-Mask VLA code/checkpoints are locally integrated, the proxy
must be a transparent continuous proxy:

- same discovery/validation split as TSC;
- same deployment-observable inputs;
- same Base action chunk access if required for continuous compatibility;
- same action validity checks;
- temporal-spatial masked action modeling without TSC's Base-error-targeted
  mask labels;
- comparable lightweight capacity and inference budget.

The proxy may not be a random weak baseline. If no faithful proxy can be built,
TSC cannot proceed to a serious paper comparison.

## Required Ablations And Simple Baselines

Required policies for the first serious comparison:

1. `smolvla_base`
2. `ts_mask_continuous_proxy_or_official_ts_mask_vla_if_installed`
3. `tsc_full`
4. `tsc_no_targeted_mask_ablation`
5. `standard_lora`

Required Stage 0 cheap diagnostics:

- magnitude-only mask baseline;
- global residual gate baseline;
- per-timestep or per-dimension gate when cheap;
- smoothing or dense residual diagnostic if it can expose trivial equivalence.

## Stage 0 Pass And Stop Conditions

Stage 0 may pass to bounded validation only if all are true:

- discovery and validation records are separated from confirmatory identities;
- `Y` has nonzero positives and negatives on discovery and validation;
- all tasks/phases used by the claim are represented;
- `M_theta` predicts validation labels above trivial-majority and
  magnitude-only baselines;
- `tsc_full` masked completion beats `ts_mask_continuous_proxy` and
  `tsc_no_targeted_mask_ablation` on the preregistered development score;
- action deltas are sparse and bounded, not globally destructive;
- unselected cells remain Base-clamped within numerical tolerance;
- official action validity passes;
- checkpoint save/reload passes;
- finite nonzero gradients reach expected trainable parameters;
- no frozen Base parameters receive gradients;
- no privileged inference inputs are present.

Stop classes:

- `TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `TSC_STAGE_0_NO_USABLE_HEADROOM`
- `TSC_STAGE_0_DESIGN_FAILURE`
- `TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

These are development-only decisions before closed-loop rollout. A Stage 0 stop
is not a closed-loop scientific kill unless the protocol explicitly advances to
closed-loop evaluation.
