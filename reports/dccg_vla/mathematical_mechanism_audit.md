# DCCG-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `DCCG_MATHEMATICAL_AUDIT_PREREGISTERED`

Method: `DCCG-VLA`, Demonstration-Calibrated Coherence Guidance for SmolVLA.

Proposal: `reports/dccg_vla/researcher_proposal.md`

Reviewer attack: `reports/dccg_vla/reviewer_attack.md`

Researcher rebuttal: `reports/dccg_vla/researcher_rebuttal.md`

Proposal SHA-256:
`AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`

## Audit Scope

This audit freezes the mathematical object that may be implemented in Stage 0.
No implementation, validation search, rollout, simulator evaluation, or
confirmatory-test tuning has happened before this audit.

DCCG may proceed only as a narrow ACG extension: demonstration-calibrated
coherence guidance for frozen SmolVLA flow action chunks. It is not ordinary
smoothing, temporal ensembling, a LoRA contribution, or a rescue of any closed
method.

## Fixed Variables And Shapes

For a batch of development rows:

- `B`: batch size.
- `H = 50`: SmolVLA action-chunk horizon.
- `D = 7`: official continuous LIBERO action dimension.
- `A in R^{B x H x D}`: current normalized SmolVLA action chunk or flow sample.
- `A_base in R^{B x H x D}`: frozen Base SmolVLA action chunk.
- `F_base(A, x, u) in R^{B x H x D}`: frozen SmolVLA flow vector at solver step
  `u`.
- `x = (o, q, l)`: legal SmolVLA deployment input: current observations,
  proprioception, and instruction.
- `task_family`: nonprivileged task-family label available before evaluation.
- `queue_index`: deployed chunk or queue index if the policy runtime exposes it.
- `h`: nonprivileged action history available to the policy, if any.
- `b`: deployment bin chosen only from legal inputs and current generated action
  features.
- `s(A) in R^{B x K_s}`: differentiable coherence feature vector.
- `m_b in R^{K_s}`: demonstration-fitted robust center for bin `b`.
- `q_b in R^{K_s}`: demonstration-fitted robust scale for bin `b`.
- `E(A, b) in R^B`: DCCG coherence energy.
- `G(A, b) in R^{B x H x D}`: group-clipped guidance direction.
- `g(A, b) in {0, 1}`: validation-frozen coherence gate, no gradient through
  the gate.
- `F_dccg(A, x, u) in R^{B x H x D}`: guided flow vector.

At inference, DCCG may use only `x`, the current generated chunk or flow sample,
legal action-derived features, `task_family`, legal `queue_index`, legal
nonprivileged action history, frozen demonstration statistics, and
validation-frozen thresholds. It may not use demonstration time index, reset
identity, future observations, future expert actions, reward, success, done,
simulator state, object pose, privileged phase labels, confirmatory identities,
or confirmatory outcomes.

## Deployment Bin Function

Training diagnostics may stratify demonstration statistics by demonstration time
index, but inference bin selection may not read that value.

The deployment bin is:

`b = b(task_family, legal_queue_bin, r_stop(A), h_legal)`.

Here:

- `legal_queue_bin` is derived only from queue or chunk index available at
  deployment. If unavailable in the runtime, it is dropped.
- `r_stop(A)` contains stop-gradient action-regime descriptors: translation
  magnitude bin, rotation magnitude bin, gripper magnitude bin, and gripper
  sign-change indicator computed from the current generated action chunk.
- `h_legal` is included only when the runtime already exposes nonprivileged
  previous-action history.

The bin index is treated as a discrete stop-gradient choice. Gradients flow
through `E(A, b)` within the selected bin, not through the bin-selection rule.
If a required bin has fewer than the preregistered minimum number of
demonstration chunks, Stage 0 must merge it according to a frozen fallback
order or stop as `DATA_FAILURE`.

## Differentiable Coherence Features

Hard p95, median, IQR, pause counts, transition counts, and reversal counts are
diagnostics unless replaced by the differentiable definitions below.

Let:

- `T = A[:, :, 0:3]`: translation group.
- `R = A[:, :, 3:6]`: rotation group.
- `Y = A[:, :, 6]`: gripper group.
- `D1 Z_h = Z_h - Z_{h-1}`.
- `D2 Z_h = D1 Z_h - D1 Z_{h-1}`.
- `D3 Z_h = D2 Z_h - D2 Z_{h-1}`.
- `tail_tau(z) = tau * log(mean(exp(z / tau)))`, with `tau > 0` fixed before
  Stage 0.
- `soft_pause(v) = sigmoid((epsilon_pause - v) / tau_pause)`.
- `soft_sign(y) = tanh(y / tau_grip)`.

The guided feature vector is:

`s(A) = [
tail_tau(||D1 T||_2),
tail_tau(||D2 T||_2),
tail_tau(||D3 T||_2),
tail_tau(||D1 R||_2),
tail_tau(||D2 R||_2),
tail_tau(||D3 R||_2),
mean(soft_pause(||D1 T||_2)),
mean(||C_high T||_2^2),
sum_h (1 - soft_sign(Y_h) soft_sign(Y_{h-1})) / 2,
sum_h |D2 soft_sign(Y)_h|
]`.

`C_high` is a fixed high-frequency DCT projection matrix on the horizon axis.
All features above are differentiable or subgradient-safe in PyTorch.

Hard p95, hard pause count, hard gripper transition count, and hard reversal
count remain required reports and gates, but they do not supply unverified flow
gradients.

## Demonstration Statistics

Discovery demonstrations fit frozen robust centers and scales for each legal
bin:

`m_b = median(s(A_demo))`

`q_b = max(IQR(s(A_demo)), epsilon_scale)`.

`m_b` and `q_b` are constants during Stage 0 guidance. They are computed only
from discovery data unless the preregistration explicitly allows validation
statistics for configuration selection. Confirmatory demonstrations, reset
identities, and outcomes may not alter them.

Label and contrast health must report:

- bin counts by task family and action regime;
- per-feature variance;
- noncollapsed `E(A_demo, b)`;
- gripper transition coverage;
- duplicate row keys;
- split overlap;
- source hashes;
- no all-zero or all-one gates on validation.

Collapsed bins or features are `DATA_FAILURE`, not closed-loop evidence.

## Energy And Guidance

For each batch item:

`z_i(A, b) = (s_i(A) - m_{b,i}) / q_{b,i}`.

`E_coh(A, b) = mean_i Huber(z_i(A, b), delta = 1)`.

The gripper-preservation term uses the differentiable gripper features already
inside `s(A)` plus hard diagnostic gates. Its coefficient is fixed for Stage 0:

`lambda_grip = 1.0`.

The total energy is:

`E(A, b) = E_coh(A, b)`.

The guidance direction is:

`G_raw(A, b) = grad_A E(A, b)`.

`G(A, b) = clip_group(G_raw, c_trans, c_rot, c_grip)`.

The guided flow vector is:

`F_dccg(A, x, u) = F_base(A, x, u) - gamma * alpha_u * g(A, b) * G(A, b)`.

`gamma = 0` must return exact Base flow. `alpha_u` is a frozen solver-step
schedule in `[0, 1]`. The default Stage 0 schedule is constant `1.0` for the
diagnostic one-step flow hook unless the implementation audit shows that the
local solver exposes a valid step index, in which case the preregistration must
freeze the exact schedule before rollout.

If any feature, energy, gradient, guided flow vector, action, or postprocessed
action is nonfinite, DCCG falls back to `F_base` and records
`IMPLEMENTATION_FAILURE` for that row. It may not silently drop the row.

## Gate And Validity Rule

The gate is:

`g(A, b) = 1[E(A, b) >= theta_b] * 1[hard_validity_precheck(A)]`.

`theta_b` is selected only on validation data through the preregistered bounded
search. No gradient flows through `g`.

Before any rollout, Stage 0 must report:

- gate activation by task, bin, and gripper-transition context;
- Base action chunk;
- DCCG action chunk;
- residual/guidance norm;
- changed dimensions;
- translation, rotation, and gripper delta p50/p95/max from Base;
- normalized action validity;
- official postprocessed action validity;
- hard gripper transition count, reversal count, and sign-change timing;
- exact Base passthrough at `gamma = 0`.

If DCCG activates everywhere, nowhere, or destroys gripper events, it stops as
`DESIGN_FAILURE` or `IMPLEMENTATION_FAILURE` according to the observed cause.

## Objective Terms And Scale

DCCG Stage 0 is not large model training. It fits frozen statistics and audits
energy/guidance behavior. If a lightweight scorer is later introduced, it may
only approximate this frozen energy and must be separately preregistered within
the six-configuration validation budget.

Required Stage 0 scalar diagnostics:

1. `E_coh`: unitless robust normalized feature energy.
2. `||G_raw||_2`: normalized action-gradient units.
3. `||G||_2`: clipped normalized action-gradient units.
4. `||A_dccg - A_base||_2`: normalized and postprocessed action units.
5. `hard_gripper_transition_delta`: count difference from Base and from hard
   demonstration-bin center.
6. `validity_violation`: official postprocessor violation units.

Gradient-norm audit:

- compute `grad_A E` on a small discovery batch and a small validation batch;
- require finite nonzero gradient where the gate is active;
- require zero gradient to frozen SmolVLA parameters when statistics are fitted;
- report gradient norm ratios by action group.

No term may dominate because of scale. If `tail_tau`, `tau_pause`,
`tau_grip`, or clipping constants cause all gradients to vanish or explode,
Stage 0 stops as `IMPLEMENTATION_FAILURE` before any scientific decision.

## No KL Between Deterministic Actions

DCCG does not compute KL divergence between deterministic `7D` actions, action
chunks, or SmolVLA flow vectors.

Any future KL proposal is invalid unless it defines valid probability
distributions, support, KL direction, estimator, gradient flow, and why KL is
preferred over Huber/L2, JS, Wasserstein, MMD, Mahalanobis distance,
vector-field consistency, or trajectory discrepancy. The frozen DCCG protocol
uses Huber-normalized feature energy and vector-field guidance, not
deterministic-action KL.

## Required Comparisons

The first serious comparison remains exactly:

1. `smolvla_base`
2. `acg_official_proxy`
3. `dccg_full`
4. `dccg_no_demo_calibration_ablation`
5. `action_smoothing_simple_killer`

Policy 2 must first attempt official ACG code/assets. If exact local execution
is unavailable, the proxy must document every mismatch and faithfully implement
ACG's published perturbation-guidance mechanism under the same SmolVLA action
interface. It may not be a smoothing-only stand-in.

Policy 4 removes demonstration calibration while preserving DCCG integration,
feature families, action caps, and compute budget.

Policy 5 is the strongest simple action-smoothing baseline under the same
action shape, postprocessor, and gripper-event constraints. A
gripper-event-preserving smoother is required if it is the strongest simple
killer.

## Bounded Validation Search

The only validation-search factors are:

- `gamma in {0.05, 0.10, 0.20}`;
- gate quantile `theta_b in {0.90, 0.95}`.

This is a maximum of six configurations. No feature set, binning method,
solver schedule, clipping cap, task split, identity split, comparator, or stop
threshold may be searched outside this budget.

The validation score remains:

`S_val = 0.40 * closed_loop_success_or_proxy
       + 0.20 * clean_retention
       + 0.15 * coherence_separation
       + 0.15 * action_validity
       + 0.10 * acg_and_smoothing_margin`.

All terms are scaled to `[0, 1]`. Ties break by clean retention, then lower
activation rate, then smaller `gamma`, then lower gate quantile.

## Stage 0 Required Diagnostics

Stage 0A must prove:

- proposal hash verification;
- discovery/validation/test separation;
- no privileged inference input;
- action shape `[50, 7]`;
- finite normalized and postprocessed actions;
- noncollapsed coherence features;
- finite nonzero `grad_A E` on real chunks;
- exact Base passthrough at `gamma = 0`;
- bounded group-clipped guidance;
- DCCG differs from Base, ACG, no-demo-calibration, and smoothing on diagnostic
  incoherent chunks;
- hard gripper-event preservation;
- duplicate-key and split-overlap checks.

Stage 0B must prove:

- Base and ACG leave meaningful headroom;
- ACG, DCCG, ablation, and smoothing use matched rows and action semantics;
- DCCG has validation separation beyond no-demo-calibration and smoothing;
- clean validation behavior is retained;
- action validity is preserved;
- gate activation is neither always on nor always off;
- ACG proxy provenance is transparent.

Do not proceed to rollout when labels or features collapse, no headroom exists,
DCCG is identical to smoothing or ablation, ACG dominates, action validity
fails, gripper events are destroyed, or any privileged inference input appears.

## Failure Classification

- `DATA_FAILURE`: missing legal inputs, collapsed features or bins, inadequate
  gripper coverage, duplicate keys, or split overlap.
- `NO_HEADROOM`: Base and ACG leave no plausible coherence or task-proxy room.
- `IMPLEMENTATION_FAILURE`: source, hook, shape, gradient, action validity,
  postprocessor, serialization, or reload failure.
- `DESIGN_FAILURE`: valid implementation but DCCG is nonacting, globally
  destructive, redundant with ACG, explained by no-demo-calibration, or
  explained by smoothing.
- `VALID_SCIENTIFIC_KILL`: only after frozen confirmatory evaluation loses
  under the preregistered decision rule.

Closed-loop scientific claims cannot be made from Stage 0 data, implementation
failure, or data failure.

## Immediate Next Stage

Proceed to DCCG preregistration and prototype protocol before implementation,
validation search, rollout, or confirmatory-test access.
