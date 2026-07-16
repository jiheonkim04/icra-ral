# CCIF-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `CCIF_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/ccif_vla/researcher_proposal.md`

Proposal SHA-256:
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1`

Reviewer attack: `reports/ccif_vla/reviewer_attack.md`

Researcher rebuttal: `reports/ccif_vla/researcher_rebuttal.md`

## Scope

This audit freezes the mathematical mechanism for CCIF-VLA before
preregistration, implementation, training, validation search, rollout, or
confirmatory testing.

CCIF's allowed novelty is only:

`Base-preserving continuous coarse motor-intent residual constraint around an
already trained continuous SmolVLA chunk`.

CCIF is not generic coarse-to-fine generation, not CAC-VLA, not
Coarse-to-Control, not CF-VLA, not standard LoRA, and not a TSC repair.

## Variables And Tensor Shapes

Constants:

- `H = 50`: SmolVLA action chunk horizon.
- `D = 7`: official LIBERO action dimension.
- `W = 4`: coarse waypoint count.
- `K = [9, 19, 34, 49]`: fixed zero-indexed waypoint steps.
- `m = 31`: normalized coarse-intent dimension.

Per legal demonstration timestep `t`:

- `o_t`: legal current observation tuple used by frozen SmolVLA.
- `B_t in R^(H,D)`: frozen Base SmolVLA postprocessed action chunk.
- `A_t in R^(H,D)`: aligned expert demonstration action chunk.
- `x_t`: deployment-observable feature tuple consisting of frozen visual
  features, proprioception, task/language identity, phase proxy, and `B_t`.
- `c_raw(A_t) in R^m`: raw coarse motor-intent vector derived from `A_t`.
- `mu_c, sigma_c in R^m`: discovery-only intent normalization statistics.
- `c_t = (c_raw(A_t) - mu_c) / max(sigma_c, eps_c) in R^m`: normalized intent
  label.
- `c_hat_theta(x_t) in R^m`: predicted normalized intent.
- `R_phi(x_t, c_hat_theta, B_t) in R^(H,D)`: raw residual field.
- `g_phi(x_t, c_hat_theta, B_t) in [0, g_max]^(H,1)`: time-varying residual
  gate broadcast over action dimensions.
- `T(c_hat_theta) in R^(H,D)`: differentiable intent template chunk.
- `P_int(R_phi, c_hat_theta) in R^(H,D)`: intent-constrained residual.
- `A_CCIF in R^(H,D)`: CCIF output chunk before official action-validity audit.

Units:

- all action chunks are in the same normalized postprocessed action units used
  by the official SmolVLA/LIBERO stack;
- intent translation and rotation entries are cumulative sums of normalized
  per-step action components;
- gripper entries use normalized gripper action units;
- normalized intent entries are unitless after discovery-only standardization.

## Coarse Intent Construction

Let:

- `p_h = A_t[h, 0:3]`: translation action at step `h`;
- `r_h = A_t[h, 3:6]`: rotation action at step `h`;
- `q_h = A_t[h, 6]`: gripper action at step `h`;
- `P_k = sum_{h=0}^{k} p_h in R^3`: cumulative translation at waypoint `k`;
- `R_k = sum_{h=0}^{k} r_h in R^3`: cumulative rotation at waypoint `k`.

The raw coarse intent vector is:

`c_raw(A_t) =
 [mean_h p_h,
  P_49,
  mean_h r_h,
  R_49,
  mean_{h=45..49} q_h,
  P_9, P_19, P_34, P_49,
  R_9, R_19, R_34, R_49]`.

Shape check:

- mean translation: `3`;
- terminal translation: `3`;
- mean rotation: `3`;
- terminal rotation: `3`;
- terminal gripper: `1`;
- translation waypoints: `4 * 3 = 12`;
- rotation waypoints: `4 * 3 = 12`;
- total `3 + 3 + 3 + 3 + 1 + 12 + 12 = 37`.

Correction: to keep the proposal's `m = 31` dimension fixed, the terminal
translation and terminal rotation entries are not duplicated outside the
waypoint blocks. The implemented vector is:

`c_raw(A_t) =
 [mean_h p_h,
  mean_h r_h,
  mean_{h=45..49} q_h,
  P_9, P_19, P_34, P_49,
  R_9, R_19, R_34, R_49]`.

Shape:

- mean translation: `3`;
- mean rotation: `3`;
- terminal gripper: `1`;
- translation waypoints: `12`;
- rotation waypoints: `12`;
- total `31`.

Only this corrected `m = 31` vector is valid for CCIF. No implementation may
use the duplicate-terminal `m = 37` draft.

Normalization:

- fit `mu_c` and `sigma_c` on discovery/training rows only;
- use `eps_c = 1e-6`;
- components with discovery `sigma_c < 1e-6` are classified as collapsed and
  trigger `CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- validation and later partitions use frozen discovery statistics.

## Intent Template

The intent template `T(c_hat_theta)` is derived from the denormalized predicted
intent:

`c_denorm = c_hat_theta * sigma_c + mu_c`.

Extract:

- `v_p_mean in R^3`;
- `v_r_mean in R^3`;
- `q_terminal in R`;
- translation cumulative waypoints `P_hat_K in R^(W,3)`;
- rotation cumulative waypoints `R_hat_K in R^(W,3)`.

For translation and rotation, linearly interpolate cumulative waypoint curves
through `(0, 0)`, `(10, P_hat_9)`, `(20, P_hat_19)`, `(35, P_hat_34)`,
`(50, P_hat_49)`, then take first differences to produce per-step template
velocities `T[:,0:3]` and `T[:,3:6]`.

For gripper:

`T[h,6] = q_terminal / 5` for `h in 45..49`, otherwise `0`.

The mean translation and mean rotation entries supervise `c_hat_theta` but do
not directly overwrite the waypoint-derived template. Stage 0 must report
whether mean entries and waypoint entries are mutually consistent; gross
inconsistency triggers design review before validation search.

## Intent-Constrained Residual

The residual head emits `R_phi`. The intent-constrained residual is:

`P_int(R_phi, c_hat_theta) = clip_l2(R_phi, rho)
                            + beta * T(c_hat_theta)`,

where:

- `rho` is a frozen residual norm cap selected only on validation;
- `beta in [0, beta_max]` is a learned scalar gate initialized to `0`;
- `beta_max = 0.25` before validation search unless Reviewer B changes it
  before any run;
- `clip_l2` clips each time-step 7D residual to norm `rho` without changing
  zero residuals.

The action chunk is:

`A_CCIF = B_t + g_phi(x_t, c_hat_theta, B_t) * P_int(R_phi, c_hat_theta)`.

Initialization requirements:

- `R_phi = 0`;
- `g_phi = 0`;
- `beta = 0`;
- disk-reloaded initialized `A_CCIF` equals `B_t` within `1e-6`.

## Objectives

All losses are coordinate means unless otherwise stated.

### Intent Prediction Loss

`L_intent = Huber(c_hat_theta(x_t), c_t; delta=1.0)`.

Scale and units:

- unitless normalized intent units;
- expected initial magnitude should be near `O(1)` per component if
  normalization is healthy.

Gradient path:

- updates intent predictor and any shared lightweight adapter parameters;
- does not update frozen Base SmolVLA parameters.

Simpler alternative:

- task/phase mean intent;
- endpoint-only intent.

Required diagnostic:

- `c_hat_theta` must beat task/phase mean intent by at least `5%` relative
  validation Huber or `0.005` absolute normalized Huber.

### Action Residual Loss

`L_action = Huber(A_CCIF, A_t; delta=0.05)`.

Scale and units:

- normalized 7D action units;
- Stage 0 must report Base Huber, prior-proxy Huber, no-intent Huber, and CCIF
  Huber on validation rows.

Gradient path:

- updates `R_phi`, `g_phi`, `beta`, intent-conditioned adapter parameters, and
  intent predictor through `T(c_hat_theta)`;
- does not update frozen Base parameters.

Required ablation:

- `ccif_no_coarse_intent_ablation`, with the same residual/gate scaffold but no
  `c_hat_theta` input and `beta = 0`.

### Intent Alignment Loss

Let `C_R(P_int)` be the `m = 31` coarse-intent vector computed from the
residual-induced chunk `B_t + P_int(R_phi, c_hat_theta)`.

`L_align = Huber(norm_c(C_R(P_int)), c_hat_theta; delta=1.0)`.

Scale and units:

- normalized intent units;
- same frozen `mu_c`, `sigma_c`, and `eps_c` as intent labels.

Gradient path:

- updates `R_phi`, `beta`, and intent predictor through `c_hat_theta`;
- does not update Base.

Simpler alternative:

- action-only residual with no coarse-intent consistency.

Required ablation:

- `ccif_no_coarse_intent_ablation` and endpoint-only intent diagnostic.

### Clean Retention Loss

For rows classified as clean-retention rows:

`L_clean = Huber(A_CCIF, B_t; delta=0.05)`.

Scale and units:

- normalized action units.

Gradient path:

- updates residual/gate/intent modules toward Base passthrough;
- does not update Base.

Clean-retention rows:

- validation rows whose Base-to-expert Huber is at or below the discovery
  median for the same task family;
- no confirmatory outcomes or reset identities are used.

### Total Objective

`L = L_flow_proxy
   + lambda_c * L_intent
   + lambda_a * L_action
   + lambda_align * L_align
   + lambda_clean * L_clean`.

`L_flow_proxy` may be zero if implementation attaches only post-decoder heads.
If nonzero, it must be the existing repository-consistent SmolVLA lightweight
flow objective and its scale must be audited.

Default pre-audit coefficients:

- `lambda_c = 0.3`;
- `lambda_a = 1.0`;
- `lambda_align = 0.2`;
- `lambda_clean = 1.0`.

Before training, Stage 0 must estimate each term magnitude and gradient norm on
a small discovery batch. If any weighted term's gradient norm exceeds another
trainable objective term by more than `100:1`, classify as
`CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` unless coefficients are
changed before any expensive run and the change is documented as pre-training
scale repair.

## KL And Distribution Distances

No KL divergence is used.

Reason:

- deterministic 7D action chunks are not probability distributions;
- SmolVLA flow vectors are not automatically normalized action densities;
- CCIF's objective is vector-field and coarse-intent consistency, for which
  Huber losses and intent-space diagnostics are the appropriate first audit.

If a later method proposes KL, it must define `p`, `q`, support, estimator,
direction, gradient flow, and why KL is preferred over JS, Wasserstein, MMD,
Mahalanobis, Huber/L2, vector-field consistency, or trajectory discrepancy.

## Required Policy Identities

The first serious comparison remains exactly:

1. `smolvla_base`;
2. `coarse_to_control_continuous_proxy`;
3. `ccif_full`;
4. `ccif_no_coarse_intent_ablation`;
5. `standard_lora`.

Cheap Stage 0 diagnostics that do not become policy identities unless they
explain the signal:

- `task_phase_mean_intent`;
- `endpoint_only_intent`;
- optional `waypoint_only_intent` if cheaper than rollout.

## Stage 0 Stop Classes

Allowed Stage 0 decisions:

- `CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `CCIF_STAGE_0_NO_USABLE_HEADROOM`
- `CCIF_STAGE_0_DESIGN_FAILURE`
- `CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

Stage 0 must stop before validation search or rollout if:

- intent labels are collapsed;
- deployment-input intent prediction does not beat task/phase mean;
- Base or the prior proxy leaves no usable residual headroom;
- CCIF does not beat the no-intent ablation on the frozen validation proxy;
- initialized/disk-reloaded identity fails;
- gradients are nonfinite, zero where expected nonzero, or reach frozen Base;
- action validity fails;
- the residual globally changes nearly every action dimension;
- any confirmatory identity or outcome is read.

Stage 0 stops are development outcomes, not closed-loop scientific kills.

## Frozen Next Step

Proceed to CCIF preregistration. The preregistration must freeze data
partitions, artifact paths, runner/resume behavior, exact pass gates,
validation-search budget, and first comparison policy identities before
implementation.
