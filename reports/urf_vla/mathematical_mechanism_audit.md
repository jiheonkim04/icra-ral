# URF-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `URF_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/urf_vla/researcher_proposal.md`

Proposal SHA-256:
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`

Reviewer attack: `reports/urf_vla/reviewer_attack.md`

Researcher rebuttal: `reports/urf_vla/researcher_rebuttal.md`

This audit freezes URF-VLA's mathematical mechanism before preregistration,
implementation, training, validation search, rollout, simulator access, or
confirmatory-test access.

## Scope

URF's allowed novelty is only:

`Base-preserving uncertainty-routed bounded residual transport around an
already trained SmolVLA action chunk, initialized to exact Base passthrough and
trained from existing demonstrations without rollout-success labels.`

URF is not SUREFlow, not Guided Action Flow, not generic residual LoRA, not a
state-of-the-art uncertainty estimator, and not a rescue or reinterpretation of
CCIF, TSC, CFR, AMP, RAP, or VDR.

The mathematical claim is narrow: a heteroscedastic residual model estimates a
candidate Base-to-expert residual, and an explicit uncertainty-dependent gate
permits bounded action-cell transport only when the predicted residual is both
large enough and sufficiently reliable.

## Variables And Tensor Shapes

Constants:

- `H = 50`: SmolVLA action chunk horizon.
- `D = 7`: official LIBERO action dimension.
- `B`: batch size.
- `eps = 1e-6`: numerical floor.
- `d_trans = {0,1,2}`: translation action coordinates.
- `d_rot = {3,4,5}`: rotation action coordinates.
- `d_grip = {6}`: gripper coordinate.

Per legal demonstration row:

| Symbol | Shape | Source | Gradient path | Meaning |
| --- | --- | --- | --- | --- |
| `o_t` | tuple | legal current observation | frozen Base only unless adapter targets declared hooks | current RGB/proprio/language inputs |
| `B_t` | `[B,H,D]` | frozen Base SmolVLA decoded normalized chunk | stopgrad | Base action chunk |
| `A_t` | `[B,H,D]` | demonstration action chunk | target only | expert action chunk |
| `M_t` | `[B,H,1]` | row/chunk validity mask | no gradient | valid future-step mask |
| `x_t` | implementation-defined | legal deployment-observable features | trainable only through declared URF adapter/head | frozen visual/policy hooks, proprioception, task text or identity, phase proxy, and `B_t` |
| `s_d` | `[D]` | discovery-only residual scale | no gradient | per-coordinate residual normalization scale |
| `Y_t` | `[B,H,D]` | `(A_t - B_t) / s_d` | target only | normalized expert residual |
| `mu_theta` | `[B,H,D]` | URF residual head | trainable | normalized residual mean |
| `ell_theta` | `[B,H,D]` | URF uncertainty head | trainable | log variance in normalized residual units |
| `v_theta` | `[B,H,D]` | clipped `exp(ell_theta)` | trainable through clipping interior | normalized residual variance |
| `q_base_theta` | `[B,H,D]` | URF route head | trainable | route logit before explicit uncertainty terms |
| `Q_theta` | `[B,H,D]` | route logit | trainable | uncertainty-dependent route score |
| `G_theta` | `[B,H,D]` | bounded route gate | trainable | action-cell intervention gate |
| `R_theta` | `[B,H,D]` | bounded residual transport | trainable | normalized residual to apply |
| `A_URF` | `[B,H,D]` | `B_t + s_d * G_theta * R_theta` | trainable through URF only | URF output chunk |

Units:

- `A_t`, `B_t`, and `A_URF` are in normalized SmolVLA/LIBERO action units.
- `Y_t`, `mu_theta`, `R_theta`, and `sqrt(v_theta)` are unitless normalized
  residual units.
- `ell_theta` is log variance of normalized residual units.
- `G_theta` is unitless and constrained to `[0, g_max]`.
- Translation, rotation, and gripper coordinates may have different discovery
  residual scales. Stage 0 must report metrics by these three groups.

Residual scale construction:

- Fit `s_d` on discovery/training rows only as
  `s_d = clamp(p95(|A_t - B_t| by coordinate), s_min, s_max)`.
- Default `s_min = 1e-4`; default `s_max = 10.0` normalized action units.
- If any coordinate has collapsed discovery residual scale before clamping,
  Stage 0 must report it and may stop as
  `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.
- Validation and later partitions use frozen discovery scales.

## Heteroscedastic Residual Model

The residual head predicts:

`mu_theta, ell_theta = F_theta(x_t, stopgrad(B_t))`.

Variance is:

`v_theta = clamp(exp(ell_theta), v_min, v_max)`,

with default audit constants:

- `log(v_min) = -8`;
- `log(v_max) = 4`.

The uncertainty head parameterizes log variance, not log scale or log
precision. Lower `ell_theta` means higher confidence in the predicted residual
mean. Higher `ell_theta` means the model predicts that the residual target is
less reliable or more variable for that state/action cell.

The bounded residual transport is:

`R_theta = r_max * tanh(mu_theta / r_max)`,

where `r_max` is a normalized residual cap fixed before validation search.
Default pre-search value: `r_max = 2.0` normalized residual units.

## Uncertainty-Routed Gate

URF's gate must depend explicitly on uncertainty. A route head that ignores
`ell_theta` is not URF and may only be the key ablation.

Define a lower-confidence residual score per action cell:

`LCB_theta = abs(mu_theta) - kappa * sqrt(v_theta)`.

Default `kappa = 1.0`.

Define route logits:

`Q_theta = q_base_theta
         + alpha_m * abs(mu_theta)
         - alpha_u * sqrt(v_theta)
         - tau_g`,

where:

- `alpha_m = softplus(a_m)` is nonnegative;
- `alpha_u = softplus(a_u)` is nonnegative;
- `tau_g` is a frozen or validation-selected threshold;
- `q_base_theta` is allowed context-dependent evidence from legal inputs.

Equivalent implementations may use `LCB_theta` directly:

`Q_theta = q_base_theta + alpha_lcb * LCB_theta - tau_g`,

with `alpha_lcb >= 0`. They must preserve the same semantics: higher residual
mean supports routing, higher residual variance suppresses routing unless
context evidence justifies otherwise under the same frozen formula.

The action gate is:

`G_theta = eta * g_max * sigmoid(Q_theta)`,

where:

- `g_max <= 1.0`;
- `eta in [0, 1]` is a trainable scalar or vector gate multiplier initialized
  exactly to `0`;
- `eta` may be parameterized as `sigmoid(e_eta)` only if initialization
  reproduces exact Base within `1e-6`; otherwise it must be a zero-initialized
  bounded multiplier.

Output action chunk:

`A_URF = B_t + s_d * G_theta * R_theta`.

Identity-preserving initialization:

- initialize `mu_theta = 0`;
- initialize `eta = 0`;
- initialize route contribution so `G_theta = 0`;
- after disk reload, `max_abs(A_URF - B_t) <= 1e-6`;
- no frozen Base parameter may change.

Decorative-uncertainty failure:

- If `ell_theta` affects only a residual NLL and has no path into `G_theta`,
  classify as `URF_STAGE_0_DESIGN_FAILURE`.
- If removing uncertainty from `Q_theta` is equivalent to URF full, classify as
  `URF_STAGE_0_DESIGN_FAILURE` or stop at validation if the ablation explains
  the result.

## Objective Terms

All objectives use only discovery/training rows before validation selection.
All reductions are masked by `M_t` and are coordinate means unless stated.

### Heteroscedastic Residual Loss

Let:

`Z_theta = (Y_t - mu_theta) / sqrt(v_theta + eps)`.

Use Huber pseudo-negative log-likelihood:

`L_het = mean M_t * [rho_delta(Z_theta) + 0.5 * log(v_theta)]`.

Default `delta = 1.0` in normalized residual units.

Scale and units:

- unitless normalized residual pseudo-NLL;
- expected healthy initial magnitude is `O(1)` if `s_d` is sane.

Gradient path:

- updates residual mean head, uncertainty head, and declared URF adapter
  parameters;
- does not update `A_t`, `B_t`, frozen Base parameters, discovery scales, or
  any confirmatory data.

Failure risk:

- `ell_theta` can inflate variance to reduce residual pressure. Stage 0 must
  report mean, p5, p50, p95, and max `ell_theta` and the fraction clipped at
  `v_min` or `v_max`.

Required ablation:

- homoscedastic residual regression with fixed `v_theta = 1`.

### Route Label Loss

Construct a development-only material-residual target:

`Z_route = 1[abs(Y_t) >= tau_route_d]`.

`tau_route_d` is fitted on discovery/training rows only by coordinate group.
Default:

`tau_route_d = max(p50(|Y_t| for coordinate d), 0.25)`.

This target supervises route observability; it is not an inference input.

Route probability before the identity multiplier is:

`P_route = sigmoid(Q_theta)`.

Route loss:

`L_route = BCE(P_route, Z_route)`.

Scale and units:

- unitless cross entropy;
- Stage 0 must report positive/negative route class balance by task, phase,
  timestep, and action group.

Gradient path:

- updates `q_base_theta`, uncertainty-to-route coefficients, residual/variance
  features if the implementation shares heads, and declared URF adapters;
- does not require `eta` to be nonzero and therefore can train route logits
  while action output remains exact Base at initialization.

Required diagnostics:

- all-zero and all-one `Z_route` are data failures;
- all-zero and all-one predicted routes after small fit are design failures;
- route activation must be concentrated in higher residual-error strata, not
  globally active.

### Bounded Action Reconstruction Loss

`L_action = mean M_t * Huber_delta(A_URF - A_t)`.

Default `delta = 0.05` normalized action units.

Scale and units:

- normalized action units;
- report translation, rotation, and gripper components separately.

Gradient path:

- updates `eta`, `q_base_theta`, residual mean, uncertainty-to-route
  coefficients, and declared adapters through `A_URF`;
- does not update frozen Base parameters.

Required ablation:

- `urf_no_uncertainty_route_ablation`: same residual capacity, residual cap,
  optimizer budget, clean-retention policy, and route-label loss where
  applicable, but `Q_theta` replaces `sqrt(v_theta)` with a learned constant or
  removes the uncertainty term entirely.

### Clean Retention Loss

For clean-retention rows and cells:

`L_clean = mean M_t * Huber_delta(A_URF - B_t)`.

Clean-retention rows are selected using discovery-only or validation-only Base
residual statistics, never confirmatory outcomes. Default clean cells are those
with `abs(Y_t) < tau_route_d`.

Scale and units:

- normalized action units.

Gradient path:

- updates URF residual, route, and uncertainty modules toward Base passthrough;
- does not update Base.

Intended effect:

- prevent the residual branch from globally replacing strong pretrained Base
  behavior.

### Optional Ordinary Flow Retention

If implementation modifies SmolVLA trainable flow/adapter modules rather than
only post-decode heads, include the repository-consistent ordinary flow loss:

`L_flow = existing SmolVLA imitation/flow objective`.

If URF is implemented as post-decode residual/uncertainty heads only, set
`L_flow = 0` and document that no Base-flow gradient exists.

### Total Objective

`L = L_flow
   + lambda_het * L_het
   + lambda_route * L_route
   + lambda_action * L_action
   + lambda_clean * L_clean`.

Default pre-search coefficients:

- `lambda_het = 1.0`;
- `lambda_route = 0.5`;
- `lambda_action = 1.0`;
- `lambda_clean = 0.2`;
- `L_flow = 0` unless declared otherwise.

Bounded validation search may vary at most six total configurations. The
preregistration must freeze the exact search factors. Allowed factors are:

- `g_max in {0.05, 0.10}`;
- `lambda_clean in {0.2, 1.0}`;
- one threshold family for `tau_g`;
- no combinatorial expansion beyond six total configurations.

No coefficient, threshold, route target, residual scale, or policy identity may
be tuned on confirmatory-test outcomes.

## Pre-Training Magnitude And Gradient Audit

Before any expensive training or rollout, Stage 0 must compute on a small
discovery batch:

- `mean/p95(|A_t - B_t|)` by action group;
- `s_d` by coordinate;
- `L_het`, `L_route`, `L_action`, `L_clean`, and optional `L_flow`;
- weighted loss magnitudes;
- per-term gradient norms on URF trainable parameters;
- count of frozen Base parameters with nonzero gradients;
- `mean/p95(abs(mu_theta))`;
- `mean/p95(sqrt(v_theta))`;
- `mean/p95(G_theta)`;
- `mean/p95(abs(A_URF - B_t))`;
- fraction of variance values clipped at floor and ceiling;
- route positive/negative balance;
- postprocessed action finite and validity fractions.

If any expected trainable parameter receives zero or nonfinite gradients after
the smoke fit, classify as `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
If any weighted objective gradient norm exceeds another required objective term
by more than `100:1`, stop as
`URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` unless the repair is made
before expensive training and recorded as a pre-training scale fix.

## Uncertainty Calibration Diagnostics

Stage 0 must bin validation cells by predicted `sqrt(v_theta)` and report:

- actual absolute residual error `abs(Y_t - mu_theta)`;
- actual Base residual magnitude `abs(Y_t)`;
- route activation rate `P_route`;
- action delta magnitude `abs(A_URF - B_t)`;
- clean-retention error.

Required gate:

- uncertainty strata must be noncollapsed;
- at least three populated bins are required unless validation data is too
  small, in which case stop as data/supervision failure;
- predicted variance must be monotonically nondecreasing with actual residual
  prediction error within tolerance;
- if the monotonicity diagnostic fails, URF cannot proceed to bounded
  validation as an uncertainty-routed method.

## Cheap Reviewer-Killer Diagnostics

Stage 0 must save development-only diagnostics for:

- task/phase residual baseline;
- residual-magnitude routing baseline;
- homoscedastic residual regression;
- stochastic-sampling disagreement route where cheap;
- perturbation-disagreement route where cheap;
- SUREFlow proxy residual headroom.

These diagnostics are not additional closed-loop policies unless the
preregistration or a later paper package freezes them as comparisons before
confirmatory testing.

## KL And Distance Policy

URF uses no KL divergence.

Reason:

- deterministic 7D actions are not probability distributions;
- SmolVLA flow vectors are not normalized action densities;
- URF's claims are residual-vector accuracy, uncertainty calibration, route
  behavior, and clean retention, for which Huber pseudo-NLL, BCE route
  supervision, calibration bins, and action validity are the correct first
  diagnostics.

Rejected alternatives for this audit:

- JS: same distributional-support problem as KL;
- Wasserstein: underidentified without calibrated action distributions;
- MMD: useful for sample sets but not per-state residual transport;
- Mahalanobis: covariance estimates are unstable for action cells and can hide
  gripper/rotation failures;
- plain L2: too outlier-sensitive for residual and gripper spikes;
- trajectory discrepancy alone: useful as a diagnostic but does not enforce
  uncertainty-dependent route semantics.

## Required Policy Identities

The first serious comparison remains exactly:

1. `smolvla_base`;
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
   and verified;
3. `urf_full`;
4. `urf_no_uncertainty_route_ablation`;
5. `standard_lora`.

SUREFlow proxy requirements:

- same legal inputs, splits, Base chunks, expert residuals, and postprocessor
  as URF;
- heteroscedastic residual-flow objective, not plain residual MSE;
- comparable optimizer and parameter budget where technically valid;
- transparent local proxy label unless official SUREFlow assets are installed
  and verified.

Standard LoRA remains live because URF trains on the same demonstrations.

## Inference Legality

Allowed inference inputs:

- current images;
- current proprioception/state already exposed to SmolVLA;
- task text or task identity available to the policy;
- frozen Base decoded chunk;
- training statistics and learned checkpoints.

Forbidden inference inputs:

- simulator reward, success, done, or hidden state;
- object pose;
- future observation;
- expert future action;
- confirmatory reset identity;
- test-set residual statistics;
- failed rollout labels collected from confirmatory evaluation.

## Stage 0 Stop Classes

Stage 0 may end only as one of:

- `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `URF_STAGE_0_NO_USABLE_HEADROOM`;
- `URF_STAGE_0_DESIGN_FAILURE`;
- `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Return `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE` when:

- source paths, features, action chunks, or masks are missing or unaligned;
- duplicate, missing, extra, or split-overlap keys are nonzero;
- residual targets, residual scales, route labels, or uncertainty strata are
  collapsed;
- task, phase, timestep, or action-group coverage is inadequate.

Return `URF_STAGE_0_NO_USABLE_HEADROOM` when:

- Base residuals are too small or mostly postprocessor noise;
- SUREFlow proxy leaves no plausible residual failure for URF;
- heteroscedastic residual prediction does not beat homoscedastic and
  task/phase residual baselines on validation.

Return `URF_STAGE_0_DESIGN_FAILURE` when:

- uncertainty does not affect the route gate;
- uncertainty strata fail monotonicity;
- route gate is all-zero, all-one, or globally active;
- `urf_no_uncertainty_route_ablation` is equivalent to URF full;
- stochastic or perturbation disagreement trivially explains the route signal;
- deployment inputs cannot infer the intended residual/uncertainty mechanism.

Return `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` when:

- proposal/source hashes mismatch;
- checkpoint persistence or disk reload fails;
- initialized Base passthrough exceeds `1e-6`;
- expected trainable gradients are zero or nonfinite;
- frozen Base parameters receive gradients;
- action validity fails under frozen official semantics;
- action deltas are globally destructive;
- any exception occurs.

Return `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION` only if all data, headroom,
mechanism, identity, gradient, action-validity, and leakage gates pass.

No Stage 0 stop is a closed-loop scientific kill. Bounded validation, training
search, rollout, or confirmatory testing are disallowed unless Stage 0 returns
`URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

## Audit Decision

URF passes to preregistration only under this mathematical audit. The
preregistration must freeze data partitions, artifact paths, proxy definition,
residual scale construction, route target construction, loss coefficients,
bounded validation-search budget, pass/stop gates, policy identities, resume
rules, duplicate-key checks, and no-confirmatory-access rules before any
implementation or training.
