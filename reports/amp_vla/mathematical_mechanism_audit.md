# AMP-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `AMP_MATHEMATICAL_AUDIT_PREREGISTERED`

This audit freezes AMP-VLA's variables, tensor shapes, objective terms,
gradient paths, projection diagnostics, ablations, and Stage 0 stop classes. It
authorizes preregistration writing only. It does not authorize training,
validation search, rollout, simulator access, or confirmatory-test access.

## Variables And Shapes

For batch size `B`, action horizon `H = 50`, action dimension `D_a = 7`,
manifold coordinate dimension `D_m`, and deployment feature dimension `D_x`
fixed by implementation:

- `o_t`: legal current observation containing RGB streams, proprioception, and
  language/task instruction.
- `A_t in R^[B,50,7]`: normalized expert action chunk.
- `A_base_t in R^[B,50,7]`: frozen Base SmolVLA decoded normalized action chunk
  from unchanged processor, flow path, and action postprocessor.
- `x_t in R^[B,D_x]`: frozen deployment-observable feature built from current
  SmolVLA visual tokens, proprioception, language/task identity, and phase.
- `Phi(A_t) = z_t in R^[B,D_m]`: discovery-fitted action-manifold coordinate.
- `Decode(z_t) = A_man_t in R^[B,50,7]`: discovery-fitted manifold decoder.
- `P(A) in R^[B,50,7]`: frozen projection or reconstruction operator mapping a
  candidate chunk to the nearest local point on the discovered action support.
- `z_theta(x_t) in R^[B,D_m]`: trainable predicted manifold coordinate.
- `A_theta_man = Decode(z_theta(x_t)) in R^[B,50,7]`: predicted manifold action
  chunk.
- `Delta_theta(o_t) in R^[B,50,7]`: zero-initialized residual direction.
- `g_theta(o_t) in R^[B,1,1]`: zero-initialized gate bounded by
  `0 <= g_theta <= g_max`.
- `C_theta = A_base_t + g_theta * Delta_theta(o_t) in R^[B,50,7]`:
  residual candidate.
- `A_amp_t = P_mix(C_theta, A_theta_man, alpha) in R^[B,50,7]`: AMP action
  path used for objective and policy identity checks.

All tensors must be finite. Padding and invalid future timesteps must be masked
before loss aggregation. The hard deployment validity gate is postprocessed 7D
LIBERO action validity. Raw normalized action validity remains a scale,
serializer, and clipping diagnostic.

## Action-Manifold Construction

`Phi` and `Decode` are fitted on discovery/training demonstrations only.
Allowed fitting inputs are legal expert action chunks, legal phase/task labels,
and legal current proprioception summary statistics when explicitly declared in
the executable protocol.

Forbidden inputs:

- validation target values for fitting `Phi` or `Decode`;
- confirmatory task/reset identities;
- rewards, success flags, done flags;
- simulator object poses or hidden state;
- future observations at inference;
- post-hoc failed or successful rollout outcomes.

The manifold family is fixed before Stage 0. The default local implementation
may be PCA/ridge or an equivalent deterministic low-dimensional coordinate
model if it satisfies:

- retained coordinate variance is positive for every dimension;
- reconstruction is finite;
- source rows are recoverable by row key;
- discovery/validation/test overlap is zero;
- projection does not use validation or confirmatory rows to refit support.

## Projection And Mixing Operator

`P(A)` is a deterministic support projection under normalized action Huber
distance:

`P(A) = Decode(argmin_z Huber_delta(Decode(z), A))`.

If exact optimization is too expensive, the executable protocol may use a
frozen approximation such as closed-form PCA projection, nearest local
manifold-coordinate projection, or ridge decoder projection. The approximation
must be declared before Stage 0 and cannot be changed after validation outcomes.

The mixed action path is:

`P_mix(C_theta, A_theta_man, alpha)
 = (1 - alpha) * C_theta + alpha * A_theta_man`

followed by the frozen support projection only when the selected configuration
requires it:

`A_amp_t = P(P_mix(C_theta, A_theta_man, alpha))`.

`alpha` and `g_max` are validation-search factors or fixed preregistration
constants, never confirmatory-test-tuned values.

## Objective Terms

### Ordinary Flow Term

`L_flow` is the repository's existing SmolVLA imitation/flow objective with
unchanged action normalization, horizon, solver, and processor.

Scale: native repository scale. Units: normalized action/flow units.

Gradient path: declared trainable adapter/LoRA modules only; frozen Base
weights receive no gradients.

Required ablation: matched `standard_lora` with the same demonstrations,
optimizer steps, rank, target modules, and ordinary flow objective but no
manifold coordinate prediction or projection.

### Manifold Coordinate Term

`L_coord = mean Huber_delta(z_theta(x_t), stopgrad(Phi(A_t)))`.

Default `delta = 1.0` in coordinate units after coordinate standardization.

Scale: coordinate-mean standardized manifold units.

Gradient path:

- gradients flow into `z_theta` and its declared adapter/LoRA parameters;
- gradients do not flow into `Phi`, discovery statistics, source actions, task
  labels, or frozen Base parameters.

Intended effect: make the deployment-observable current state predict where the
legal expert chunk lies on the action manifold.

Simpler alternative: task/phase mean coordinate predictor.

Required diagnostic: coordinate probe must beat task/phase coordinate baseline.

### Projection-Support Term

`L_proj = mean_masked Huber_delta(A_amp_t, A_t)`.

Scale: coordinate-mean normalized action units. Units: normalized 7D action
coordinates.

Gradient path:

- gradients flow into `z_theta`, `Delta_theta`, and `g_theta` through
  differentiable pieces of `P_mix`;
- if `P` is nondifferentiable, gradients stop at `P` and the executable
  protocol must include a differentiable surrogate, normally `Huber(A_theta_man,
  A_t)` plus `Huber(C_theta, A_t)`;
- gradients do not flow into frozen Base parameters or discovery support rows.

Intended effect: align the trainable path with demonstrated action support
rather than only reducing unconstrained action error.

Required ablation: `amp_no_manifold_projection`.

### Clean Retention Term

`L_clean = mean_masked Huber_delta(A_amp_t, stopgrad(A_base_t))` on clean
development states where Base behavior must be retained.

Scale: coordinate-mean normalized action units.

Gradient path: into AMP adapter/gate only; `A_base_t` is stop-gradient.

Intended effect: preserve strong pretrained behavior and prevent global action
replacement.

### Manifold Consistency Diagnostic

`D_manifold(A) = mean_masked Huber_delta(A, P(A))`.

This is a diagnostic unless explicitly used as a differentiable surrogate. It
must be reported for Base, clipped Base, ABot proxy, AMP no-projection, and AMP
full. It separates "inside action bounds" from "near demonstrated support."

### Total Objective

`L = L_flow
   + lambda_m L_coord
   + lambda_p L_proj
   + lambda_clean L_clean`.

Allowed development search:

- `D_m in {8,16}`;
- `lambda_p in {0.3,1.0}`;
- `g_max = 0.20`;
- no more than six total configurations.

If `lambda_m`, `lambda_clean`, `alpha`, or projection approximation choices are
not searched, they must be fixed before Stage 0 execution. All tried
configurations and negative results must be saved.

## Pre-Training Magnitude Audit

Before any training, compute on one small discovery batch:

- `L_flow`, `L_coord`, `L_proj`, and `L_clean`;
- `D_manifold(A_base_t)`, `D_manifold(clip(A_base_t))`,
  `D_manifold(A_theta_man)`, and `D_manifold(A_amp_t)`;
- gradient norm for AMP trainable parameters;
- gradient norm for ordinary flow adapter parameters;
- count of frozen Base parameters with nonzero gradients;
- `||A_amp_t - A_base_t||`, `||A_theta_man - A_base_t||`,
  `||Delta_theta||`, and `g_theta`;
- normalized action validity and postprocessed 7D action validity.

Proceed to training only if all objectives and gradients are finite, frozen
Base gradient count is zero, expected AMP parameters receive nonzero gradients,
and no loss term overwhelms the others without preregistered normalization.

## KL And Probability Distances

AMP does not use KL divergence between deterministic 7D actions, action chunks,
manifold coordinates, or SmolVLA flow vectors. These are real-valued vectors,
not validated probability distributions. If a future method wants a
distributional action-manifold objective, it must define `p`, `q`, support,
estimator, gradient flow, KL direction, and why KL is preferred over JS,
Wasserstein, MMD, Mahalanobis distance, Huber/L2, vector-field consistency, or
trajectory discrepancy. That is not part of AMP.

## Clipping Diagnostic

A bound-only diagnostic must be saved when Stage 0 evaluates projection:

`A_clip_t = clip(A_base_t, lower=-1, upper=1)` in normalized action units and,
separately, the postprocessed 7D action-bound projection if the postprocessor
defines one.

AMP's projection is not considered a mechanism if:

- `D_manifold(A_clip_t)` matches or beats `D_manifold(A_amp_t)`;
- clipped Base matches or beats the ABot proxy and AMP on the selected
  validation proxy;
- the only improvement is coordinate range validity with no support,
  headroom, or closed-loop evidence.

Clipping remains a diagnostic, not a policy in the first serious comparison
unless Reviewer B reopens the protocol before confirmatory testing.

## Mechanism Smoke

Before rollout, report for representative rows:

- Base action;
- clipped Base diagnostic action;
- ABot proxy action;
- AMP no-projection action;
- AMP full action;
- projection delta;
- residual norm;
- gate value;
- manifold coordinate norm;
- changed dimensions;
- normalized action validity;
- postprocessed 7D action validity;
- clean-versus-shift context.

Expected behavior: AMP differs from Base and no-projection only where manifold
support is useful, and differences are bounded rather than global.

## Stage 0 Decision Rules

Return `AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE` when:

- source paths, action rows, features, proprioception, or timestamps are
  missing or unaligned;
- duplicate, missing, extra, or split-overlap keys are nonzero;
- action manifold coordinates collapse, are nonfinite, all-zero, all-one, or
  duplicate;
- validation tasks or phases are uncovered.

Return `AMP_STAGE_0_NO_USABLE_HEADROOM` when:

- Base or the ABot proxy leaves no plausible residual failure for AMP;
- manifold reconstruction does not beat task/phase means by the frozen margin;
- projection is fully explained by clipping or bound-only validity.

Return `AMP_STAGE_0_DESIGN_FAILURE` when:

- manifold coordinates are not predictable from deployment inputs;
- `amp_no_manifold_projection` is equivalent to AMP;
- the residual/gate branch is nonacting;
- the mechanism activates everywhere rather than in relevant states.

Return `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` when:

- proposal/source hashes mismatch;
- checkpoint persistence or reload fails;
- Base passthrough at initialization fails;
- expected trainable gradients are zero or nonfinite;
- frozen Base parameters receive gradients;
- postprocessed 7D action validity fails;
- action deltas are globally destructive;
- any exception occurs.

Return `AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION` only if all Stage 0 gates pass.

These outcomes are development-only audit decisions, not closed-loop
scientific kills.

## Frozen Baselines And Ablations

The first serious comparison remains exactly five policies:

1. `smolvla_base`;
2. `abot_m0_action_manifold_proxy`;
3. `amp_full`;
4. `amp_no_manifold_projection`;
5. `standard_lora`.

No additional simple baseline may replace matched standard LoRA unless
Reviewer B explicitly reopens the protocol before confirmatory testing. No
prior proxy may be weakened after validation performance is seen.
