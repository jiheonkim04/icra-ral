# BRID-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `BRID_MATHEMATICAL_AUDIT_PREREGISTERED`

Method: `BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action
chunks.

Proposal: `reports/brid_vla/researcher_proposal.md`

Reviewer attack: `reports/brid_vla/reviewer_attack.md`

Researcher rebuttal: `reports/brid_vla/researcher_rebuttal.md`

Proposal SHA-256:
`2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2`

## Audit Scope

This audit freezes the mathematical object that may be implemented in Stage 0.
No implementation, training, validation search, rollout, simulator access, or
confirmatory-test tuning has happened before this audit.

AFID and all previous methods remain closed. BRID may not repair or reinterpret
any previous result.

## Fixed Variables And Shapes

For each development row at replanning time `t`:

- `H = 50`: SmolVLA action-chunk horizon.
- `D = 7`: official LIBERO continuous action dimension.
- `x_t = (o_t, q_t, l_t)`: legal deployment input.
- `o_t`: legal RGB observation streams used by SmolVLA.
- `q_t in R^8`: official SmolVLA proprio/state vector.
- `l_t`: task instruction string from the allowed split.
- `B_t in R^{H x D}`: frozen SmolVLA Base action chunk.
- `E_t in R^{H x D}`: aligned demonstration action chunk, training only.
- `R_t = E_t - B_t in R^{H x D}`: Base residual target, training only.
- `k in {1, ..., K}`: diffusion step.
- `epsilon in R^{H x D}`: deterministic standard-normal noise for the row and
  step identity.
- `r_k in R^{H x D}`: noisy residual.
- `epsilon_theta(r_k, k, x_t, B_t) in R^{H x D}`: predicted noise.
- `D_theta(x_t, B_t) in R^{H x D}`: denoised residual proposal after the fixed
  inference schedule.
- `Delta_t in R^{H x D}`: group-clipped residual proposal.
- `g_theta(x_t, B_t, D_theta) in [0, 1]` or `[0, 1]^{H x D}`: intervention
  gate.
- `A_t = B_t + g_theta * Delta_t in R^{H x D}`: deployed BRID chunk.

At inference, BRID may use only `x_t`, `B_t`, model parameters, the frozen
noise/step rule, and validation-frozen thresholds. It may not use `E_t`,
`R_t`, future observations, reward, success, done, object poses, simulator
state, confirmatory-test labels, or confirmatory-test outcomes.

## Diffusion Residual Construction

The development diffusion process is:

`r_k = sqrt(alpha_bar_k) R_t + sqrt(1 - alpha_bar_k) epsilon`.

`alpha_bar_k` is a fixed monotonically decreasing schedule in `(0, 1]` frozen
before Stage 0. The default Stage 0 diagnostic schedule is linear in variance
with `K = 8` candidate denoising steps for lightweight diagnostics; bounded
validation may later select a step count from the preregistered search budget.

Noise identity:

`epsilon = Normal(0, I; seed = SHA256(method, split, task, demo_id,
frame_index, k, noise_identity))`.

Noise identities are deterministic, partition-safe, and independent of
confirmatory-test identities.

## Residual Caps

Group clipping is applied before deployment:

- translation dimensions: `rho_translate`;
- rotation dimensions: `rho_rotate`;
- gripper dimension: `rho_gripper`.

The mathematical operator is:

`Delta_t = clip_group(D_theta, rho_translate, rho_rotate, rho_gripper)`.

Each group is clipped by per-element absolute value in the official 7D action
units. Stage 0 must report p50/p95/max deltas by group. Any cap values used in
Stage 0 must be fixed in preregistration before training or rollout.

## Identity Initialization

Before training and after disk reload:

- the residual output head is initialized to zero;
- the gate logit bias is initialized so `g_theta = 0` within identity
  tolerance;
- `Delta_t = 0`;
- `A_t = B_t`.

Stage 0 identity tolerance:

`max_abs(A_t - B_t) <= 1e-7`

for initialized and disk-reloaded BRID on the serializer-preflight rows.

Any failure here is `IMPLEMENTATION_FAILURE`, not a scientific result.

## Intervention Rule

BRID may intervene only if all validation-frozen checks pass:

1. residual score confidence is above threshold;
2. denoising consistency is within threshold;
3. group-clipped deltas are within caps;
4. action postprocessing remains valid;
5. the row is not classified as a clean-retention row.

If any check fails:

`A_t = B_t`.

The confidence and consistency rules must be frozen in preregistration and may
be selected only on validation data.

## Objective

All objectives operate on discovery/validation partitions only.

Default objective:

`L = L_score + lambda_rec L_rec + lambda_clean L_clean + lambda_gate L_gate +
lambda_valid L_valid`.

Terms:

1. Score-matching noise prediction:

   `L_score = mean(Huber(epsilon_theta(r_k, k, x_t, B_t) - epsilon))`.

2. Residual reconstruction on residual-active development rows:

   `L_rec = mean(Huber(B_t + Delta_t - E_t))`.

3. Clean retention on rows where the frozen rule says not to intervene:

   `L_clean = mean(Huber(A_t - B_t))`.

4. Gate sparsity:

   `L_gate = mean(g_theta)`.

5. Validity penalty:

   `L_valid = mean(bound_violation(official_postprocess(A_t)))`.

No objective term may read confirmatory-test identities or outcomes.

## Units And Scale

- `L_score`: normalized action-residual noise units.
- `L_rec`: official continuous 7D action units after residual clipping.
- `L_clean`: official continuous 7D action units relative to Base.
- `L_gate`: unitless intervention frequency.
- `L_valid`: official action-bound violation units.

Before nontrivial training, Stage 0 must estimate each loss magnitude and the
gradient norm entering the trainable BRID parameters on a small development
batch. No term may be allowed to dominate merely because of scale. Coefficients
may be chosen only through the bounded validation search.

## Gradient Paths

- `L_score` updates the residual denoiser and conditioning adapter.
- `L_rec` updates the residual output head and gate through `Delta_t` and
  `A_t`.
- `L_clean` updates the residual output head and gate toward exact Base
  passthrough.
- `L_gate` updates the gate parameters only.
- `L_valid` updates trainable BRID parameters through the postprocessed action
  validity surrogate.
- Frozen SmolVLA Base receives no gradients.

Stage 0 must explicitly verify finite nonzero gradients on expected trainable
parameters and zero gradients on frozen SmolVLA parameters.

## No KL Between Deterministic Actions

BRID does not compute KL divergence between deterministic `7D` action vectors.

If a later implementation proposes KL, it is rejected unless it defines valid
probability distributions, support, direction, estimator, gradient flow, and
why KL is preferable to Huber/L1, JS, Wasserstein, MMD, Mahalanobis distance,
vector-field consistency, or trajectory discrepancy.

The current BRID objective uses Huber/L1-style distances and valid diffusion
noise prediction, not deterministic-action KL.

## Required Comparisons

First serious comparison:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

Policy 2 is the closest prior. It must be a raw action-chunk diffusion proxy
matched on data, split, legal inputs, action semantics, and compute. It may not
use Base-residual conditioning or exact Base passthrough.

Policy 4 removes Base-residual conditioning and zero-residual identity
integration while preserving denoising objective, budget, and caps.

Policy 5 tests whether ordinary demonstration adaptation explains the gain.

## Stage 0 Required Diagnostics

Stage 0 is development-only and must report:

- proposal hash verification;
- discovery/validation/test identity separation;
- no privileged input access;
- Base/demonstration action alignment under official 7D semantics;
- residual noncollapse by task, phase, time index, and action group;
- residual headroom relative to Base and raw diffusion proxy;
- deterministic noise identity replay;
- score prediction versus zero-noise, mean-noise, and task/phase baselines;
- initialized and disk-reloaded exact Base identity;
- finite nonzero trainable gradients and zero frozen-Base gradients;
- BRID distinctness from Base, raw diffusion proxy, and no-Base-residual
  ablation after a small development fit;
- action-delta p50/p95/max by group;
- intervention frequency by task/phase/action group;
- clean-retention exact Base diagnostics;
- action postprocessing validity.

## Stage 0 Stop Classes

Stage 0 must classify failures as one of:

- `BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`
- `BRID_STAGE_0_DESIGN_FAILURE`
- `BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

Do not proceed to bounded validation if:

- residual targets are collapsed;
- score prediction fails trivial baselines;
- residual coverage is insufficient;
- no residual headroom exists;
- raw Diffusion Policy proxy dominates;
- no-Base-residual ablation explains the effect;
- standard LoRA explains the effect;
- intervention is global or absent;
- action deltas are unbounded;
- clean retention fails;
- action postprocessing validity fails;
- checkpoint reload identity fails;
- any privileged inference input or confirmatory-test identity is used.

## Bounded Validation Search

If Stage 0 passes, the maximum validation-only search is `6` total
configurations. Allowed factors:

- residual cap scale;
- denoising step count;
- clean-retention coefficient;
- score/loss coefficient;
- adapter or LoRA rank;
- deterministic versus validation-frozen stochastic residual seed rule.

Selection must combine validation closed-loop success when available or the
closest feasible proxy, clean retention, residual mechanism activation, action
validity, and compute overhead. Do not select purely by offline action L2.

## Audit Decision

`BRID_MATHEMATICAL_AUDIT_PREREGISTERED`

Immediate next stage: freeze BRID preregistration before implementation,
validation search, rollout, or confirmatory-test access.
