# HEST-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.

Decision: `HEST_MATHEMATICAL_AUDIT_PREREGISTERED`

## Variables And Shapes

- Base chunk `A in R^(H x d)`, with `H = 50`, `d = 7`.
- Arm increments `X = A[:, 0:6] in R^(50 x 6)`.
- Gripper stream `g = A[:, 6] in R^50`.
- Cumulative arm path `P = C X in R^(50 x 6)`.
- `C in R^(50 x 50)` is lower triangular with ones on and below its diagonal.
- Second-difference matrix `D2 in R^(48 x 50)` has row stencil `[1,-2,1]`.
- Smoothed cumulative path `Q in R^(50 x 6)`.
- Decoded arm increments `S = C^(-1) Q in R^(50 x 6)`.

All computations use float64 in the reference implementation. Runtime float32
must agree within the frozen tolerance before rollout.

## Units

Each arm column retains its controller-facing units:

- columns `0:3`: relative translation-controller units;
- columns `3:6`: relative rotation-controller units.

Cumulative values have the corresponding accumulated controller units. They
are not asserted to be Cartesian poses or `SE(3)` elements. The gripper stream
retains its original controller convention exactly.

## Constrained Objective

For each arm dimension, solve:

`min_q (q-p)^T(q-p) + lambda (D2 q)^T(D2 q)`

subject to `E q = b`, where:

- `E in R^(2 x 50)` selects indices `0` and `49`;
- `b = [p_0, p_49]^T`;
- `lambda = 4.0`.

The KKT system is:

`[I + lambda D2^T D2, E^T; E, 0] [q; nu] = [p; b]`.

`I + lambda D2^T D2` is positive definite for `lambda >= 0`; adding two
independent endpoint constraints yields a unique feasible minimizer. The same
factorization is reused for all six dimensions.

## Decode And Invariants

Let `B = C^(-1)` be the first-difference operator. Decode `S = BQ` and blend:

`Y = (1-alpha)X + alpha S`.

Since `sum_i S_i = Q_49 = P_49 = sum_i X_i`, then:

`sum_i Y_i = sum_i X_i`

for every `alpha`. The first increment is also preserved because
`Q_0 = P_0 = X_0`.

The final output is:

`H = concat(Y, g)`.

Therefore:

- shape is exactly `50 x 7`;
- first arm action is preserved analytically;
- cumulative arm endpoint is preserved analytically;
- every gripper value is copied exactly.

These statements do not imply physical trajectory or task equivalence.

## Comparator Definitions

### SplineProxy

Apply the same cumulative constrained solve and decode independently to all
seven coordinates. This intentionally tests a homogeneous continuous action
object. It is a transparent proxy, not official Spline Policy.

### NoEndpoint

Solve the unconstrained arm objective:

`Q = (I + lambda D2^T D2)^(-1) P`,

decode it, and copy gripper exactly. This isolates endpoint constraints.

### MovingAverage

Apply one fixed edge-replicated three-tap kernel `[0.25, 0.50, 0.25]` to the
six raw arm-increment sequences and copy gripper exactly. No coefficient is
searched.

## Scale And Support

The objective is scale-equivariant within each dimension: multiplying one
input column by a constant multiplies its output by the same constant. No loss
term combines translation and rotation units.

Discovery action support is computed per arm dimension as `[min_j,max_j]` over
fixed discovery chunks. Runtime validity permits only the closed interval
expanded by `0.01 * max(max_j-min_j, 1e-12)`. Any violation returns Base for the
whole chunk. Clipping is forbidden.

## Gradient Path

There is no training objective and no gradient path into SmolVLA. The quadratic
objective is solved analytically at inference. Consequently:

- loss-weight and gradient-norm audits are not applicable;
- no adapter parameter may receive a gradient;
- no checkpoint selection occurs;
- alpha is selected only by the frozen validation score.

## Numerical Audit

Before simulator replay, unit and real-chunk tests must report:

- KKT residual;
- endpoint residual;
- first-action residual;
- gripper maximum absolute and bitwise difference;
- float64 repeat determinism;
- float32-versus-float64 maximum difference;
- per-dimension support validity;
- second-difference energy ratio;
- Base/output action deltas;
- fallback reason counts.

Required finite fractions are `1.0`. Endpoint residual must be at most `1e-8`
in float64 and `1e-6` in float32. Gripper difference must be exactly zero.

## Alternative Distances And KL

The objective uses squared Euclidean distance separately within each
controller coordinate because it is a deterministic regularized trajectory
fit. KL divergence is not used: deterministic `7D` actions and SmolVLA flow
vectors are not normalized probability distributions.

JS, Wasserstein, MMD, and Mahalanobis distances are unnecessary for the stated
deterministic smoothing problem. A group-aware rotation metric would express a
different scientific method and is not introduced after freezing.

## Required Ablations

- all-channel SplineProxy: tests hybrid event factorization;
- NoEndpoint: tests cumulative endpoint preservation;
- MovingAverage: tests ordinary smoothing;
- Base: tests whether any transformation improves the backbone.

Failure of an invariant is implementation failure. Failure to beat a control
under valid closed-loop evidence is a scientific decision only at the frozen
Stage A/B gate.
