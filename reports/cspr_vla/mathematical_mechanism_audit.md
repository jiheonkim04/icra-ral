# CSPR-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `CSPR_MATHEMATICAL_AUDIT_PREREGISTERED`

Method: `CSPR-VLA`

Proposal SHA-256:
`CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`

Researcher rebuttal: `reports/cspr_vla/researcher_rebuttal.md`

## Fixed Variables And Shapes

- batch size: `N`;
- horizon: `H = 50`;
- action dimension: `D = 7`;
- Base action chunk: `B in R^[N, 50, 7]`;
- demonstration action chunk: `Y in R^[N, 50, 7]`, training/audit only;
- legal visual feature: `V in R^[N, 960]` when cached or extracted from
  current observation;
- proprioception: `P in R^[N, 8]`;
- task/language feature: `L`;
- Base chunk summary: `S(B)`, containing groupwise velocity, acceleration,
  curvature, gripper-transition indicators, and action magnitudes;
- criticality target: `C* in {0,1}^[N, 50, 7]`;
- criticality prediction: `C = c_phi(V, P, L, S(B)) in [0,1]^[N, 50, 7]`;
- residual proposal: `R = r_theta(V, P, L, S(B)) in R^[N, 50, 7]`;
- gate: `G = gate_tau(C) in [0,1]^[N, 50, 7]`;
- group cap vector: `Delta in R^7_+`;
- emitted action: `A in R^[N, 50, 7]`.

No confirmatory identities, rewards, success flags, done flags, simulator
state, object pose, future observation, or demonstration time index may enter
`c_phi`, `r_theta`, `G`, or inference.

## Criticality Target

For development rows only, define per-cell raw criticality score:

`q_t,d = w_e * norm_err_t,d + w_k * norm_curv_t,d + w_a * norm_acc_t,d + w_g * grip_event_t,d`.

Where:

- `norm_err_t,d` is a robustly normalized Base-vs-demonstration absolute
  action error `|B_t,d - Y_t,d|`;
- `norm_curv_t,d` is robustly normalized demonstration curvature;
- `norm_acc_t,d` is robustly normalized demonstration acceleration;
- `grip_event_t,d` is nonzero only for gripper-event cells and immediate
  protected neighbors;
- all normalizers are fit on discovery data and applied to validation.

`C*_t,d = 1[q_t,d >= q_tau]`.

`q_tau` is chosen on discovery/validation only and must produce noncollapsed
labels before training. If labels collapse globally or per required task
coverage, stop as `DATA_FAILURE`.

## Forward Formula

Residual cap:

`E = Delta * tanh(R)`.

Gate:

`G = sigmoid((C - tau) / T)` for differentiable training diagnostics, with
`T > 0`; hard thresholding may be used only after the mathematical audit
defines a straight-through or detached inference rule.

Action:

`A_raw = B + G * E`.

`A = postprocess(A_raw)`.

At identity initialization:

- residual head final weights and bias are zero;
- gate bias or threshold defaults to Base passthrough;
- therefore `E = 0` and `A_raw = B`.

## Objective Terms

Criticality loss:

`L_crit = BCE(C, C*)`, optionally replaced by focal BCE only if label imbalance
is healthy and the replacement is selected inside the bounded validation
search.

Residual fit loss on critical cells:

`L_fit = mean(Huber(A - Y; beta_fit) * C*)`.

Clean retention loss on noncritical cells:

`L_keep = mean(Huber(A - B; beta_keep) * (1 - C*))`.

Action validity soft penalty:

`L_bound = mean(relu(|A_raw| - 1)^2)`.

Total:

`L = lambda_crit L_crit + lambda_fit L_fit + lambda_keep L_keep + lambda_bound L_bound`.

Default initial coefficients for small-batch audit:

- `lambda_crit = 1.0`;
- `lambda_fit = 1.0`;
- `lambda_keep = 1.0`;
- `lambda_bound = 1.0`.

These coefficients are not final training hyperparameters until term
magnitudes and gradient norms are inspected on a small batch and any bounded
validation-only coefficient choice is frozen.

## Gradient Paths

- `L_crit` updates `phi` through `C = c_phi(...)`.
- `L_fit` updates `theta` through `R`, `E`, `A_raw`, and `A`; it may update
  `phi` through `G` only if the soft gate is used.
- `L_keep` updates `theta` and, if soft gate is used, `phi`; it prevents
  noncritical drift.
- `L_bound` updates `theta` and, if soft gate is used, `phi`; it discourages
  out-of-bound raw actions before postprocessing.
- Base policy weights are frozen and receive no gradients.

Expected nonzero gradients before training:

- criticality head parameters from `L_crit`;
- residual head parameters from `L_fit` when at least one critical label is
  positive;
- retention and bound paths when residual/gate are nonzero after identity
  smoke.

If expected parameters receive zero, NaN, or infinite gradients after the
appropriate smoke setup, stop as `IMPLEMENTATION_FAILURE`.

## Units And Scale

Actions are normalized 7D SmolVLA action units. Translation, rotation, and
gripper components must be audited separately. Before any nontrivial training,
report:

- `L_crit`, `L_fit`, `L_keep`, `L_bound`;
- gradient norm per objective and per parameter group;
- max/mean action delta for translation, rotation, and gripper;
- gate activation fraction globally and per task;
- postprocessed action validity.

No objective may dominate another by more than `100:1` in weighted gradient
norm without documented normalization or a validation-only coefficient change.

## Bounded Validation Search

Maximum `6` configurations:

- residual cap: at most `3` values;
- criticality gate threshold or soft-gate temperature: at most `2` values.

No combinatorial expansion beyond these factors. No confirmatory-test tuning.

The validation score must combine:

- validation success or closest legal proxy;
- clean retention;
- criticality activation localization;
- action validity;
- separation from `dysl_action_importance_proxy`;
- separation from `critical_step_threshold_simple_killer`;
- compute overhead.

Offline action L2 alone cannot select a final configuration.

## Required Ablations And Baselines

First serious comparison remains exactly:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

The DySL proxy must be transparent and may not use CSPR's learned residual
action correction. The simple killer must remain live.

## Forbidden Mathematics

No KL divergence is used between deterministic 7D action vectors. SmolVLA flow
vectors are not treated as normalized action probability distributions.

If any later objective proposes KL, it must separately define valid
probability distributions, support, estimator, KL direction, gradient flow, and
why KL is preferable to Huber/L2, Mahalanobis, MMD, Wasserstein, JS, or
trajectory discrepancy. That is not part of this audit.

## Stage 0 Stop Classes

- `DATA_FAILURE`: cache mismatch, label collapse, duplicate/overlap failure,
  or insufficient task/phase coverage.
- `DESIGN_FAILURE`: legal criticality predictor fails to beat trivial
  baselines or simple killer explains the method.
- `IMPLEMENTATION_FAILURE`: checkpoint reload failure, zero/NaN gradients,
  Base identity failure, global destructive action changes, or action-validity
  failure.
- `NO_HEADROOM`: Base and DySL proxy leave no measurable development headroom.

None of these pre-rollout stops is a closed-loop scientific kill.

## Current Status

This audit freezes the mathematical mechanism only. No CSPR implementation,
training, validation search, rollout, simulator access, or confirmatory-test
access has happened.
