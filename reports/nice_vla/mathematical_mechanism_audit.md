# NICE-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `NICE_MATHEMATICAL_AUDIT_PREREGISTERED`.

This audit freezes the variables, tensor shapes, objectives, numerical
implementation, gradients, calibration estimator, search utility, and required
ablations before labeled latent extraction or validation search.

## Source-Anchored Mean Model

The closest-prior source is VLA-Corrector commit
`9d23a0ba6fad562d3ed1a68fc52c8a12459abb41` under Apache-2.0.

The official source defines:

- `SiglipDynamicsDataset` pairs `(t,t+k)` within one episode;
- `k=10` by default;
- `Delta z_t = z_(t+k)-z_t`;
- normalized action `a_t in R^7`;
- `SiglipResidualMLP`, which broadcasts an action embedding to every visual
  token and predicts one delta per token;
- cosine prediction error for the circuit breaker;
- rolling `median + 3*MAD`, a ten-score warmup, and five-step cooldown in the
  default `CircuitBreaker`.

The local prior arm is a transparent official-code-derived proxy. Full
development uses the official residual MLP topology with measured token width
`D`, action embedding width `256`, four residual hidden blocks of width `2048`,
SiLU, zero dropout, and one-frame history. Its checkpoint is shared by prior,
NICE, and the key ablation and is frozen before covariance fitting.

Stage 0A uses a tiny topology only to test shapes, gradients, serialization,
and algebra: action embedding `16`, two residual hidden blocks of width `64`,
zero dropout. A Stage 0A tiny checkpoint is not a policy or scientific prior.

## Variables And Shapes

For batch size `B`, measured visual token count `L`, measured token width `D`,
flattened residual width `n=L*D`, and low rank `R=8`:

- frozen visual tokens: `z_t in R^(B x L x D)`;
- future frozen tokens: `z_(t+10) in R^(B x L x D)`;
- normalized current action: `a_t in R^(B x 7)`;
- normalized previous action: `a_prev in R^(B x 7)`;
- target delta: `y_t=z_(t+10)-z_t in R^(B x L x D)`;
- mean prediction: `mu_phi(z_t,a_t) in R^(B x L x D)`;
- flattened residual: `r=vec(y_t-mu_phi) in R^(B x n)`;
- action-regime condition: `c_t in R^(B x 18)`;
- diagonal variance: `v_theta in R^(B x n)`;
- fixed PCA basis: `B_8 in R^(n x 8)` with orthonormal columns;
- low-rank variance: `lambda_theta in R^(B x 8)`;
- normalized innovation: `q_t in R^B`.

The 18 conditioning values are current action 7, previous action 7,
translation norm 1, rotation norm 1, absolute gripper 1, and gripper-transition
indicator 1. All actions are dimensionless after the official action
normalizer. Latents and residuals use frozen encoder latent units. Variances
use squared latent units. `q_t`, cosine error, NLL after normalization, and all
coverage scores are dimensionless.

Stage 0A measures and persists `L` and `D`. Any shape other than action width 7,
or any shape change within a manifest, is `IMPLEMENTATION_OR_DATA_FAILURE`.

## Fixed Condition Construction

Let `g_t=a_t[6]`. From the frozen discovery extraction only, compute:

`delta_g = median({|g_t-g_(t-1)| : |g_t-g_(t-1)| > 0})`.

If the set is empty, the data gate fails; no fallback is invented. Otherwise:

`u_t = 1[|g_t-g_(t-1)| >= delta_g]`.

Then:

`c_t = [a_t, a_(t-1), ||a_t[0:3]||_2, ||a_t[3:6]||_2,
         |a_t[6]|, u_t]`.

`delta_g` is persisted before validation and never searched. The first valid
frame in an episode uses `a_prev=a_t` and `u_t=0`.

## Mean Objective

Flatten predictions and targets to `[B,n]`. With `eps_cos=1e-8`:

`L_mean = 1 - mean_b cosine(mu_phi_b, y_b; eps_cos)`.

MSE is diagnostic:

`MSE_mean = mean_(b,j) (mu_phi[b,j]-y[b,j])^2`.

Gradient path: `L_mean -> phi` only. The frozen SmolVLA encoder receives no
gradient. Once the mean checkpoint is selected on discovery/validation, `phi`
is set `requires_grad=False` before covariance training.

The zero-change baseline predicts `0`. The task-mean baseline predicts a
discovery-only equal-episode mean delta for its task. Full mean must beat both
on the preregistered validation aggregate before covariance evidence is used.

## Covariance Parameterization

Both families use a token-wise scale network. A 32-dimensional projection of
`c_t` is broadcast across `L` tokens and concatenated with detached `z_t`.
Two width-128 SiLU layers output one log-variance value per token dimension.
The only difference between families is the rank-8 global head.

Raw diagonal output `s in R^(B x n)` becomes:

`v_raw = softplus(s) + v_floor`,

`v = clamp(v_raw, max=v_ceiling)`.

Frozen constants:

- `v_floor=1e-6` squared latent units;
- `v_ceiling=1e2` squared latent units;
- `jitter=1e-8` for small-matrix Cholesky only;
- `R=8`.

For the low-rank family, equal-episode discovery residuals are centered and a
randomized-SVD-free deterministic eigendecomposition of their `m x m` Gram
matrix produces the top eight nonzero left singular directions in flattened
residual space. Basis signs are canonicalized by making each largest-absolute
component positive. Fewer than eight positive singular values is a data gate
failure for the low-rank family, not permission to change rank.

The global head mean-pools detached visual tokens, concatenates `c_t`, applies
two width-128 SiLU layers, and emits `rho in R^(B x 8)`:

`lambda = clamp(softplus(rho)+v_floor, max=v_ceiling)`.

Covariance families:

`Sigma_diag = diag(v)`;

`Sigma_lr = diag(v) + B_8 diag(lambda) B_8^T`.

Every covariance is symmetric positive definite because `v>=v_floor>0` and
`lambda>0`. No dense `[n,n]` matrix is constructed in actual scoring.

## Innovation And NLL Algebra

For diagonal covariance:

`mahal = sum_j r_j^2/v_j`,

`logdet = sum_j log(v_j)`.

For low-rank covariance, let `D_v=diag(v)`, `U=B_8`, `C=diag(lambda)`,
`K=C^(-1)+U^T D_v^(-1) U`. Then:

`r^T Sigma^(-1) r = r^T D_v^(-1)r
 - (U^T D_v^(-1)r)^T K^(-1)(U^T D_v^(-1)r)`;

`logdet(Sigma) = sum_j log(v_j) + sum_i log(lambda_i)
                 + logdet(K)`.

`K+jitter*I_8` is solved by Cholesky in float64 during diagnostics and float32
or float64 during training. A failed Cholesky, nonfinite term, negative
Mahalanobis value below numerical tolerance `-1e-5`, or final normalized
innovation below `-1e-7` is an implementation failure. Values within tolerance
are clamped to zero only after recording the raw value.

Normalized innovation:

`q_b = (r_b^T Sigma_b^(-1) r_b)/n`.

Covariance objective, omitting the constant shared `log(2*pi)`:

`L_cov = 0.5 * mean_b((mahal_b+logdet_b)/n)`.

Gradient path: `L_cov -> theta` and, for low rank, its scale head only. It does
not update SmolVLA, mean parameters, PCA basis, targets, actions, or condition
construction. Before training, assert Base gradient norm and mean gradient norm
are exactly zero and covariance gradient norm is finite and greater than zero.

## Objective Scale Audit

On the first fixed Stage 0A batch, persist:

- `L_mean`, MSE, and `L_cov` before optimization;
- total and per-module gradient L2 norms;
- covariance-to-mean gradient-norm ratio from separate backward passes;
- min, median, p99, and max `v` and `lambda`;
- clamped fraction;
- finite fractions for predictions, losses, scores, and gradients.

Mean and covariance are optimized sequentially, so their gradients are never
summed and no loss coefficient balances them. The gradient ratio is diagnostic
only. Any nonfinite value, zero intended gradient, or nonzero frozen gradient
fails Stage 0A.

No KL divergence is used. Neither deterministic 7D actions nor flow vectors
are represented as probability distributions. NLL is preferred here because
the method requires a conditional scale and an innovation score; MSE/cosine
cannot represent heteroscedastic scale, while JS, Wasserstein, MMD, and KL
would require distributions or samples not supplied by this model.

## Episode-Cluster Split Conformal Estimator

For calibration episode `e`, let `Q_e` be all finite natural-pair innovation
scores after fixed censoring. Require at least 16 scores. Define its cluster
score as the deterministic nearest-rank 90th percentile:

`s_e = Q_e[ceil(0.90*|Q_e|)]` in one-indexed sorted order.

Use exactly the same number of calibration episodes per validation task. Let
the resulting task-balanced multiset contain `m` episode scores. For requested
coverage `c in {0.90,0.95,0.975}`, set:

`j_c = min(m, ceil((m+1)*c))`,

`tau_c = sorted(s)[j_c]` in one-indexed order.

This is a finite-sample split-conformal upper quantile under exchangeability
of episode clusters. The campaign does not claim frame-IID coverage. Empirical
coverage is the equal-task mean fraction of held-out validation evaluation
episodes with `s_e <= tau_c`. Pass tolerance is absolute error `<=0.03`.

Calibration uses validation demos `30..34`; utility evaluation uses validation
demos `35..39`. Neither group trains mean or covariance. Confirmatory data is
never calibration data.

## Monitor And Identity Semantics

The prior monitor computes flattened `1-cosine(mu,y)`, appends scores to its
official rolling history, has a ten-score warmup, uses `median+3*MAD`, and uses
the official five-step cooldown. Any additional persistence observed in the
official execution wrapper is copied exactly and documented in Stage 0A.

NICE replaces only the score and threshold with `q_t` and `tau_c`. It uses the
same warmup, persistence, cooldown, truncation, recovery, OGG, action queue,
and postprocessing as the prior. Monitor-disabled integration must produce
bitwise-identical Base queue length, selected action, and postprocessed 7D
action. Nontriggered prior/Ours integration must also preserve the queued Base
action exactly.

## Required A/B Comparisons

The key ablation uses the shared frozen mean, flattened cosine error, one
validation-frozen global threshold, and the matched recovery path. It removes
conditional covariance and conformal coverage selection.

The simple killer uses fixed short-horizon replanning, the same Base policy
call and action semantics, no learned monitor, and no OGG.

The diagnostic oracle may inspect the future only to estimate attainable
validation headroom. It is never an inference policy and never enters the five
policies.

## Validation Utility

For each of the exactly six configurations and both fixed seeds, compute terms
in `[0,1]`:

- `success_proxy`: legal validation closed-loop success if available,
  otherwise preregistered natural-versus-mismatch balanced accuracy;
- `clean_retention`: clipped ratio to Base clean validation success;
- `interrupt_F1`: episode-task-balanced F1 on development-only mismatch labels;
- `coverage_score = max(0,1-|coverage-c|/0.03)`;
- `action_validity`: fraction of postprocessed actions within dataset bounds;
- `normalized_overhead`: valid uncontaminated compute overhead clipped to
  `[0,1]`, or fixed to `0` for all configurations when timing is quarantined.

`S_val = 0.45*success_proxy + 0.20*clean_retention
         + 0.15*interrupt_F1 + 0.10*coverage_score
         + 0.10*action_validity - 0.05*normalized_overhead`.

Select the highest arithmetic mean across seeds `20262011` and `20262012`.
Ties within `1e-12` prefer higher clean retention, lower trigger frequency,
diagonal covariance, higher requested coverage, then lexicographic config ID.
All trial rows, seeds, checkpoints, and negative results are retained.

## Frozen Numerical And Mechanism Gates

Stage 0A requires:

- measured stable `(L,D)` and action width 7;
- exact within-episode `k=10` keys;
- mean and covariance checkpoint reload max error `<=1e-6`;
- finite nonzero intended gradients and zero frozen gradients;
- direct dense versus diagonal/low-rank score and logdet error `<=1e-5` on
  fixed small matrices;
- minimum covariance eigenvalue `>=v_floor-1e-8`;
- conformal order-statistic tests including ties;
- exact monitor-disabled Base passthrough;
- zero duplicate, split-overlap, privileged-input, validation-read, and
  confirmatory-read counts.

Stage 0B adds the proposal's mean-headroom, empirical-coverage, AUROC,
nonconstant-score, task-balance, safety, and prior-relative closed-loop
headroom gates. Synthetic diagnostics cannot override failed natural or
closed-loop gates.

## Audit Decision

`NICE_MATHEMATICAL_AUDIT_PREREGISTERED`.

The formulas are dimensionally valid, gradients are explicitly owned, no
dense high-dimensional covariance is required, and no decorative KL is used.
Proceed only to frozen preregistration and prototype protocol.
