# SPARC-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal hash:
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.

Decision: `SPARC_MATHEMATICAL_AUDIT_PREREGISTERED`

This audit freezes the executable mathematics before labeled activation
extraction. It incorporates the independent Reviewer B attack and Researcher A
rebuttal without changing the proposal.

## Variables, Shapes, Units, And Sources

| Symbol | Shape | Units | Source |
| --- | --- | --- | --- |
| `H_e,r,d,l` | `50 x 720` | model activation units | frozen action-expert post-residual at episode `e`, replan `r`, denoising step `d`, site `l` |
| `x_e,r,d,l` | `720` | model activation units | token mean of `H` |
| `mu_j,l,d` | `720` | model activation units | equal-episode class/task mean |
| `R_j,l,d` | `720 x 720` | squared activation units | equal-episode covariance |
| `C_j,l,d` | `720 x 720` | dimensionless | conceptor operator |
| `M_T,l,d` | `720 x 720` | dimensionless | Base-interpolated inference gate |
| `a_base` | `50 x 7` | postprocessed LIBERO action units | unmodified SmolVLA chunk |
| `a_ours` | `50 x 7` | postprocessed LIBERO action units | steered SmolVLA chunk |
| `beta` | scalar | dimensionless | steering strength |
| `alpha` | scalar | inverse activation scale by convention | conceptor aperture |

Global steering drops denoising index `d` from the fitted covariance by giving
all ten steps equal weight. Per-step steering constructs ten separate matrices.

## Equal-Episode Covariance

For one class in task `j`, retain at most `16` uniformly spaced replans per
episode and all ten denoising steps. Let `I_e` be the retained vectors for
episode `e` and `n_e = |I_e|`.

The weighted mean is:

`mu_j = (1/E_j) sum_e (1/n_e) sum_(i in I_e) x_ei`.

The covariance is:

`R_j = (1/E_j) sum_e (1/n_e) sum_(i in I_e)
       (x_ei - mu_j)(x_ei - mu_j)^T`.

Every episode has weight `1/E_j`; every source task later has weight `1/J`.
This prevents episode length, replanning frequency, or task frame count from
defining the method.

## Conceptor Objective

For centered activation row vector `z in R^720`, a conceptor solves the
regularized reconstruction objective:

`L_C(C) = E[||z - z C||_2^2] + alpha^-2 ||C||_F^2`.

The closed-form minimizer is:

`C(R, alpha) = R (R + alpha^-2 I)^-1`.

Scale and effect:

- the reconstruction term has squared activation units;
- `alpha^-2` controls regularization in the same closed-form convention as
  COAST;
- eigen-direction variance `lambda` is retained by
  `lambda / (lambda + alpha^-2)`;
- no gradient path exists because the solve is closed-form;
- the intended effect is a soft low-rank subspace filter, not action-space
  probability modeling.

Simpler alternative: mean-difference CAA. It is rejected as the method because
the positive prior shows outcome geometry is low-rank but not rank-one. Pooled
CAA remains a diagnostic only.

## Source Failure Mixture

Within-source means are removed before aggregation. For `J=4` source tasks:

`R_f^src = (1/J) sum_(j=1)^J R_f^j`.

Then:

`C_f^src(alpha) = C(R_f^src, alpha)`.

This is the covariance of an equal-task mixture after within-task centering.
It preserves variance shared or recurring across source failures while
excluding between-task mean offsets.

Required diagnostic alternative:

`C_f_mean = (1/J) sum_j C(R_f^j, alpha)`.

It cannot select the method. Report normalized Frobenius similarity,
eigenvalues, quota, effective rank, and leave-one-source-out changes.

## SPARC Boolean Operator

For target success conceptor `C_s^T`:

`NOT(C_f^src) = I - C_f^src`.

The canonical conceptor AND is:

`C_sparc^T = pinv(pinv(C_s^T) + pinv(I-C_f^src) - I)`.

Interpretation:

- `C_s^T` retains target-success variance;
- `I-C_f^src` attenuates source-failure variance;
- AND retains directions supported by both requirements;
- no additive source-success direction enters the operator.

Required ablation:

`C_failure_only = I - C_f^src`.

Required target-success-only diagnostic:

`C_success_only = C_s^T`.

Closest-prior operator for source `j`:

`C_coast^j = pinv(pinv(C_s^j) + pinv(I-C_f^j) - I)`.

## Inference Gate

For each token row `h in R^720`:

`M = (1-beta)I + beta C`,

`h_ours = h M^T`.

Because the conceptor is symmetric up to numerical error, transposition is
explicit only to preserve row-vector shape. `beta=0` gives identity exactly.

Expected internal effect: suppress high-energy reusable failure directions
while retaining target-success directions. Expected action effect: bounded
changes to the frozen `50 x 7` chunk. Expected closed-loop effect: fewer target
failures without importing source-task motor programs.

The post-residual tensor at layer `l` is captured and transformed before the
next layer's action-expert `input_layernorm`. Applying the operator only to the
MLP branch or only to a pooled token is mathematically a different method and
is forbidden.

## Layer And Aperture Selection

Candidate residual sites: `{0, 5, 11, 14}`.

Apertures: `{0.1, 0.5, 1.0, 2.0, 10.0}`.

Quota:

`q(C) = trace(C) / 720`.

At `alpha=10.0`, select the site with maximum mean SPARC quota over targets.
Tie tolerance is `1e-12`; lower site wins.

Normalized success/failure overlap:

`o(C_s,C_f) = trace(C_s C_f) / (||C_s||_F ||C_f||_F)`.

At the selected site, choose the aperture whose mean target overlap is inside
`[0.80,0.90]` and nearest `0.85`. If none is inside, choose the global nearest.
Tie tolerance is `1e-12`; smaller aperture wins.

Only target successes and source failures enter these calculations. Target
failures are forbidden.

## Numerical Linear Algebra

- compute mean/covariance/inverse/pseudoinverse in float64;
- regularized solve condition number must be `<=1e12`;
- `pinv` uses `rcond=1e-12`;
- symmetrize final matrices with `(C+C^T)/2`;
- no eigenvalue truncation or scientific PSD projection;
- eigenvalues must lie in `[-1e-8,1+1e-8]`;
- store float32 canonical row-major bytes and SHA256;
- load to the model device and cast at multiplication time;
- finite fraction must be `1.0`.

A violation is an implementation failure. It cannot be repaired after seeing
validation or confirmatory outcomes.

## Geometry Metrics

For operator `A` and covariance `R`:

`ret(A,R) = trace(A R A^T) / max(trace(R),1e-12)`.

Target separation:

`m_T = ret(C_sparc,R_s^T) - ret(C_sparc,R_f^T)`.

Failure containment:

`k_T = ret(C_f^src,R_f^T)`.

The diagnostic random null conjugates the same `C_f^src` by `256`
deterministic Haar orthogonal matrices generated by QR decompositions of
Gaussian matrices with seed `1919`; QR signs are fixed by positive diagonal of
`R`. This preserves the conceptor spectrum while randomizing orientation.

Headroom requires the exact margins in the rebuttal: `m_T>=0.05` on two of
three and in mean, `k_T` above null p95 by `0.02` on two of three, and SPARC
target-success retention above selected COAST by `0.02` on two of three with
no decline worse than `0.02`.

These metrics are diagnostic. They are not substitutes for closed-loop
success.

## LoRA Control Objective

The standard LoRA control uses the existing SmolVLA flow-matching loss. With
clean action `a`, Gaussian noise `epsilon`, and sampled `t`:

`x_t = t epsilon + (1-t) a`,

`u_t = epsilon - a`,

`L_flow = mean ||v_theta(x_t,t,o) - u_t||_2^2` over the valid `50 x 7`
action entries.

Variables are in normalized native training-action units inside the policy;
postprocessing is used only for action validity and rollout. Gradients flow
only through the rank-4 LoRA/default SmolVLA PEFT targets. SPARC matrices and
Base weights receive no gradient.

The initial batch audit reports loss scale, finite fraction, total and per-
module gradient norms, and predicted-flow variance. No coefficient is added;
there is one objective. KL is neither necessary nor valid here.

## Why No KL Divergence

SPARC acts on deterministic hidden vectors and deterministic `7D` actions.
Neither is a normalized probability distribution. Direct KL between actions
or flow vectors is prohibited. Frobenius geometry, covariance retained energy,
and component-wise action differences answer the actual questions without a
fabricated density approximation.

## Validation Configurations

Exactly six:

| Config | Strategy | Beta |
| --- | --- | ---: |
| `sparc_global_b010` | global | `0.1` |
| `sparc_global_b030` | global | `0.3` |
| `sparc_global_b050` | global | `0.5` |
| `sparc_step_b010` | per-step | `0.1` |
| `sparc_step_b030` | per-step | `0.3` |
| `sparc_step_b050` | per-step | `0.5` |

No layer, aperture, aggregation, task, label, LoRA, or action-threshold search
is hidden inside these configurations.

## Action Consequence And Safety

For the first ten actions and full chunk, report Base/Ours translation,
rotation, gripper, per-step `7D` L2, changed dimensions, and bound exceedance.
The hard thresholds are exactly those in the rebuttal.

The operator must act at `beta=0.1` with mean full-chunk delta `>1e-6`, yet
remain below all Base-relative and component-wise limits. Exact equality after
geometry passes is design failure; nonfinite or wrongly wired equality is
implementation failure.

## Required Ablations And Simpler Alternatives

- complete single-source COAST transfer: closest prior;
- source-failure-only gate: key policy ablation;
- target-success-only conceptor: offline mechanism diagnostic;
- mean-conceptor failure aggregate: offline aggregation diagnostic;
- pooled CAA: offline rank-one diagnostic;
- target-success filtered-BC LoRA: simple data/adaptation control.

Only the first, key ablation, and LoRA join Base and SPARC in the first serious
five-policy comparison.

## Audit Decision

The objective terms, variables, shapes, units, scales, gradient paths,
alternatives, ablations, numerical tolerances, and selection rules are frozen.
The method may proceed to executable preregistration and Stage 0A hook/math
implementation. It may not collect labeled activations until Stage 0A passes.
