# PCAV-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Decision: `PCAV_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.

## 1. Variables, Shapes, Units, And Sources

For one row at timestep `t`:

- two RGB images: `I_t^1, I_t^2`, each `3 x 256 x 256` after dataset
  decoding;
- frozen image tokens: `E(I_t^k) in R^(64 x 960)`;
- pooled two-camera visual vector:
  `v_t = [mean E(I_t^1); mean E(I_t^2)] in R^1920`;
- episode-initial visual vector: `v_0 in R^1920`;
- frozen masked-mean language embedding: `e_l in R^960`;
- normalized proprioception: `s_t in R^8`;
- postprocessed action chunk: `a_i in R^(50 x 7)`;
- consequence prefix: `a_i^10 = a_i[0:10] in R^(10 x 7)`;
- future target visual vector: `v_t10 in R^1920` from frame `t+10`;
- normalized progress label: `u_t = t / (T-1) in [0,1]` after terminal
  padding removal;
- final action-expert feature: `h_i in R^720`, mean pooled from
  `R^(50 x 720)` immediately before `action_out_proj` at the final denoising
  step.

Visual and language features are dimensionless frozen neural features.
Actions are postprocessed LIBERO 7D control units: translation dimensions
`0:3`, rotation `3:6`, and gripper `6`. Time is dataset frames at `10 Hz`.

Only discovery data estimates means and standard deviations. Every standard
deviation is floored at `1e-6` and serialized before training.

## 2. Frozen Feature Projections

Standardize visual and language vectors with discovery statistics, then use
fixed Gaussian projections:

- `R_v in R^(128 x 1920)`;
- `R_l in R^(128 x 960)`.

Entries are sampled once with seed `1801` and scaled by `1/sqrt(128)`. The
projected vectors are:

`x_t = R_v standardize(v_t) in R^128`,

`x_0 = R_v standardize(v_0) in R^128`,

`g_l = R_l standardize(e_l) in R^128`.

The matrices and statistics are frozen, disk-persistent, and receive no
gradient. This prevents a trainable projection from collapsing the future
prediction target.

## 3. Candidate-Oracle Headroom

Let `D(a_i, a*)` be the first-10-step demonstration error:

`D = mean_h [ ||dt_h||_2 / sigma_trans`
`             + ||dr_h||_2 / sigma_rot`
`             + |dg_h| / sigma_grip ] / 3`,

where each difference is between postprocessed candidate and demonstration
actions and each `sigma` is the discovery RMS scale for that group, floored at
`1e-6`.

For row `n`, oracle relative reduction is:

`rho_n = (D(a_0,a*) - min_i D(a_i,a*)) / max(D(a_0,a*),1e-6)`.

This oracle is a diagnostic upper bound only. It is never available to the
deployed policy and never used for confirmatory selection.

## 4. TACO-Style Support Head

For discovery demonstration action `a*`, fixed noising level
`tau in {0.25,0.50,0.75,1.00}`, and fixed Gaussian noise `eps`:

`x_tau = tau eps + (1-tau) a*`.

SmolVLA predicts flow vector `w_theta(x_tau,tau)`. The clean endpoint estimate
is:

`a_hat_tau = x_tau - tau w_theta(x_tau,tau)`.

Retain the corresponding pooled expert feature for the `tau` minimizing
`D(a_hat_tau,a*)`. This implements the local high-fidelity feature search.

For retained feature `h_n in R^720`, deterministic Rademacher target
`r_n in {-1,+1}^32`, and CFN `C_psi: R^720 -> R^32`:

`L_CFN = mean_n mean_d (C_psi(h_n)_d - r_n,d)^2`.

CFN architecture:

`Linear(720,256) -> GELU -> Linear(256,32)`.

The pseudo-count score for candidate feature `h_i` is:

`c_i = 1 / (||C_psi(h_i)||_2^2 + 1e-6)`.

Scale: `L_CFN` is dimensionless and expected near `1` at zero output.
Gradient path: `L_CFN -> psi` only; frozen SmolVLA and retained features receive
no gradient.

## 5. Consequence Model

Standardize and flatten `a_i^10` to `b_i in R^70`. Define consequence input:

`z_i = [x_t; s_t; b_i; g_l] in R^334`.

Consequence network:

`F_omega: R^334 -> R^128`,

`Linear(334,256) -> GELU -> Linear(256,256) -> GELU -> Linear(256,128)`.

The target is the directly projected future visual feature
`y = R_v standardize(v_t10) in R^128`.

With elementwise Huber loss of transition `delta=1`:

`L_F = mean Huber(F_omega(z_i) - y)`.

Scale: standardized projected-feature units. Gradient path:
`L_F -> omega` only. No Base parameter or projection receives gradient.

Persistence baseline:

`y_persist = x_t`.

The consequence model must improve over persistence and an action-shuffled
diagnostic after full development training before rollout.

## 6. Progress Model

For real or predicted projected visual feature `x in R^128`, progress input is:

`q = [x_0; x; g_l; s_t] in R^392`.

Progress network:

`P_phi: R^392 -> [0,1]`,

`Linear(392,256) -> GELU -> Linear(256,128) -> GELU -> Linear(128,1) -> Sigmoid`.

Pointwise progress loss:

`L_point = mean Huber(P_phi(q_t) - u_t)` with `delta=0.1`.

For a symmetrized within-episode pair `(i,j)` separated by at least 10 valid
frames, let `y_ij=+1` when `j` is later and `-1` for the reversed input. The
ordering loss is:

`L_rank = mean softplus(-y_ij * (p_j - p_i) / 0.1)`.

Combine:

`L_P = L_point + 0.25 L_rank`.

Scale: dimensionless progress units. Gradient path: `L_P -> phi` only.

## 7. Joint Consequence-Progress Objective

For transition ending at `t+10`, predicted future progress is:

`p_hat_t10 = P_phi([x_0; F_omega(z_i); g_l; s_t])`.

Joint consistency loss:

`L_J = mean Huber(p_hat_t10 - u_t10)` with `delta=0.1`.

Gradient path: `L_J -> phi` and `L_J -> omega`. Frozen embeddings,
projections, Base, and actions receive no gradient.

On one fixed discovery calibration batch before optimization, compute detached
term magnitudes:

`m_k = max(L_k(calibration_batch), 1e-4)` for `k in {F,P,J}`.

The full head objective is:

`L_heads = L_F/m_F + L_P/m_P + L_J/m_J`.

The calibration batch identity, raw magnitudes, normalized magnitudes, per-term
gradient norms, total gradient norm, and pairwise gradient cosine similarities
are persisted. This normalization is deterministic objective engineering, not
a validation search. Nonfinite terms, an expected zero gradient, or a gradient
norm ratio above `1000` is an implementation/scale stop before full training.

## 8. Inference Decision

For candidate `i`, support is `c_i` and predicted progress is:

`p_i = P_phi([x_0; F_omega(z_i); g_l; s_t])`.

Candidate zero is direct Base. Define eligible set:

`E = {0} union {i>0: valid(a_i) and c_i >= gamma}`,

where `gamma` is one of the validation-frozen support percentiles.

Let:

`i* = argmax_(i in E) p_i`, with stable lowest-index ties.

Selected index is:

`i_sel = i* if p_i* - p_0 > m else 0`,

where `m` is the validation-frozen abstention margin. There is no gradient at
inference and no weighted sum of support and progress.

## 9. Exact Base Identity

Before heads are trained, `heads_trained=false` bypasses all scoring and returns
candidate zero. After training, strict inequality and stable ties return Base
for equal evidence.

Required pretraining and disk-reload identity:

`max_abs(a_selected - a_direct_base) = 0.0`.

No candidate is averaged with Base, no residual is added, and no action is
clipped after selection.

## 10. Action Validity

For every postprocessed candidate and selected chunk:

- finite fraction `=1.0`;
- absolute maximum `<=1.25`;
- outside-`[-1,1]` fraction `<= Base + 0.01`;
- p99 exceedance beyond `[-1,1]` `<= Base + 0.02`;
- simulator action-space acceptance `=1.0` before rollout.

Translation, rotation, and gripper deltas from Base are reported separately.
These gates are frozen before trained-head evaluation.

## 11. Simpler Alternatives And Controls

Closest-prior proxy: maximum `c_i` under identical candidates,
`taco_support_proxy`.

Key ablation: progress selection and Base abstention without support threshold,
`pcav_progress_only`.

Simple reviewer-killer: matched `standard_lora_new_task`.

Persistence is the primary consequence baseline. Proprioception-only, task-only,
frame-difference, and path-length are progress diagnostics. They do not become
extra first-stage policies.

## 12. Required Ablations

The paper mechanism requires:

1. support-only versus full PCAV;
2. progress-only versus full PCAV;
3. initial-state anchor present versus removed in offline mechanism metrics;
4. real action prefix versus shuffled action prefix for consequence prediction;
5. Base abstention activation and fallback reason distribution.

Only the first two enter policy-level scientific comparison. The others are
offline mechanism audits.

## 13. Why No KL Is Used

PCAV does not compute KL between deterministic 7D actions, flow vectors, hidden
features, or pseudo-counts. None is automatically a normalized probability
distribution with specified support.

Feature prediction uses Huber distance. Progress uses Huber and pairwise
logistic ordering. Support uses the CFN squared regression defined by the
positive prior. These distances match the actual deterministic variables and
avoid decorative probability notation.

## 14. Known Failure Modes And Classification

- no unique better candidate: `NO_USABLE_HEADROOM`;
- collapsed or shortcut progress labels: `DATA_OR_SUPERVISION_FAILURE`;
- visual future not predictable from legal inputs: `DESIGN_FAILURE` only after
  adequate capacity and optimization, otherwise `UNDERPOWERED_OR_UNRESOLVED`;
- CFN hook or endpoint mismatch: `IMPLEMENTATION_OR_DATA_FAILURE`;
- all alternatives invalid: `IMPLEMENTATION_OR_DATA_FAILURE` or
  `NO_USABLE_HEADROOM`, depending on source;
- zero intervention with weak training: `UNDERPOWERED_OR_UNRESOLVED`;
- full PCAV beaten by support-only or progress-only in adequate matched
  closed-loop evaluation: valid component/formulation failure.

## Audit Decision

Variables, shapes, objectives, scales, gradients, inference behavior, simple
alternatives, ablations, and failure classes are sufficiently specified for an
executable preregistration. No training or confirmatory use is authorized by
this audit alone.
