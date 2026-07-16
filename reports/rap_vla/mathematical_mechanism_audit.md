# RAP-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `RAP_MATHEMATICAL_AUDIT_PREREGISTERED`

This audit freezes RAP-VLA's variables, tensor shapes, objective terms,
gradient paths, ablations, and Stage 0 stop classes. It authorizes
preregistration writing only. It does not authorize training, validation
search, rollout, simulator access, or confirmatory-test access.

## Variables And Shapes

For batch size `B`, action horizon `H = 50`, action dimension `D_a = 7`, and
retrieval feature dimension `D_f` fixed by the implementation:

- `o_t`: legal current observation containing RGB streams, proprioception, and
  language/task instruction.
- `A_t in R^[B,50,7]`: normalized expert action chunk.
- `A_base_t in R^[B,50,7]`: Base SmolVLA decoded normalized action chunk from
  the unchanged flow path.
- `phi(o_t) = f_t in R^[B,D_f]`: frozen retrieval feature computed only from
  deployment-observable current inputs.
- `M = {(f_i, A_i, meta_i)}`: discovery/training memory entries with
  `f_i in R^[D_f]` and `A_i in R^[50,7]`.
- `idx_t in N^[B,k]`: top-k memory indices under the frozen retrieval metric.
- `w_t in R^[B,k]`: nonnegative retrieval weights satisfying
  `sum_j w_t[b,j] = 1`.
- `A_mem_t in R^[B,k,50,7]`: retrieved memory action chunks.
- `A_anchor_t = sum_j w_t[:,j] A_mem_t[:,j,:,:] in R^[B,50,7]`.
- `R_target_t = A_t - A_anchor_t in R^[B,50,7]`.
- `h_t`: SmolVLA policy representation used by the residual adapter.
- `R_theta(h_t,A_anchor_t,f_t) in R^[B,50,7]`: trainable residual prediction.
- `g_theta(h_t,f_t) in R^[B,1,1]`: trainable nonnegative gate bounded by
  `0 <= g_theta <= g_max`.
- `A_rap_t = A_base_t + g_theta * R_theta in R^[B,50,7]`.

All tensors are finite. Padding and invalid future timesteps must be masked
before loss aggregation. The hard deployment validity gate is evaluated after
the existing SmolVLA 7D LIBERO postprocessor; normalized action validity is a
diagnostic scale check.

## Retrieval Construction

The retrieval metric is frozen before Stage 0:

`d(i,t) = || normalize(f_t) - normalize(f_i) ||_2^2 + beta_task I[task_i != task_t] + beta_phase |phase_i - phase_t|`.

`k`, `beta_task`, `beta_phase`, feature normalization statistics, task filters,
and phase features are fixed in preregistration. Retrieval may use only
discovery/training memory entries for the candidate memory. Validation entries
may be queried only for scoring. Confirmatory entries may not be embedded,
indexed, inspected, or used for tuning before final freeze.

Weights:

`w_j = softmax(-d_j / tau)` over the top-k neighbors.

`tau` is fixed before Stage 0. If `tau` is not used, uniform top-k averaging
must be declared before Stage 0. Changing `tau` after validation outcomes is
forbidden.

## Objective Terms

### Ordinary Flow Term

`L_flow` is the repository's existing SmolVLA imitation/flow objective with
unchanged action normalization, horizon, solver, and processor.

Scale: native repository scale. Units: normalized action/flow units.

Gradient path: ordinary trainable adapter or declared LoRA modules only; frozen
Base weights receive no gradients.

Required ablation: matched `standard_lora` with the same demonstrations,
optimizer steps, rank, target modules, and flow objective but no retrieval
anchor or residual target.

### Residual Anchor Term

`L_res = mean_masked Huber_delta(R_theta(h_t,A_anchor_t,f_t), R_target_t)`.

Default `delta = 1.0` in normalized action units unless Stage 0 scale audit
shows a preregistered smaller delta is necessary before training. No
confirmatory outcome may change `delta`.

Scale: coordinate-mean normalized action residual. Units: normalized 7D action
coordinates.

Gradient path:

- gradients flow into `R_theta` and its declared adapter/LoRA parameters;
- gradients may flow into `g_theta` only through the final RAP action term;
- gradients do not flow into memory actions, retrieval indices, task labels,
  source data, or frozen Base parameters.

Intended effect: make the trainable path explain the current-state residual
around a legal retrieved action anchor rather than relearning the whole action
chunk from scratch.

Simpler alternative: direct retrieved action replay or anchor-only/no-residual.

Required ablation: `rap_anchor_only_no_residual`.

### Action Integration Term

`L_rap = mean_masked Huber_delta(A_rap_t, A_t)`.

This term verifies that the gated residualized action path improves expert
action reconstruction without replacing Base globally.

Scale: coordinate-mean normalized action units. Units: normalized 7D action
coordinates.

Gradient path:

- gradients flow into `R_theta` and `g_theta`;
- gradients may flow through declared adapter/LoRA modules that produce
  `A_base_t` only if those modules are explicitly trainable under the selected
  configuration;
- frozen Base weights remain frozen.

Intended effect: couple the residual prediction to the actual policy action
path and prevent a decorative auxiliary head.

Required ablation: residual head trained but detached from action path is not a
valid RAP policy; it may be a diagnostic only.

### Clean Retention Term

`L_clean = mean_masked Huber_delta(A_rap_t, stopgrad(A_base_t))` on clean
development states where Base behavior must be retained.

Scale: coordinate-mean normalized action units.

Gradient path: into RAP adapter/gate only; `A_base_t` is stop-gradient.

Intended effect: bound disruption of clean behavior and prevent memory anchors
from globally replacing strong Base actions.

Required ablation: matched standard LoRA and anchor-only/no-residual must use
the same clean-retention policy where applicable.

### Total Objective

`L = L_flow + lambda_r L_res + lambda_a L_rap + lambda_clean L_clean`.

Allowed development search:

- `lambda_r in {0.1, 0.3, 1.0}`;
- `g_max = 0.25`;
- `lambda_a` and `lambda_clean` fixed in preregistration before training.

No combinatorial grid is allowed. If a coefficient is not searched, it must be
fixed before Stage 0 execution. All tried configurations and negative results
must be saved.

## Pre-Training Magnitude Audit

Before any training, compute on one small discovery batch:

- `L_flow`, `L_res`, `L_rap`, and `L_clean`;
- gradient norm for RAP trainable parameters;
- gradient norm for ordinary flow adapter parameters;
- count of frozen Base parameters with nonzero gradients;
- `||A_anchor - A_t||`, `||R_target||`, `||R_theta||`, and `g_theta`;
- normalized and postprocessed action validity.

Proceed to training only if all objectives and gradients are finite, frozen
Base gradient count is zero, expected RAP parameters receive nonzero gradients,
and no loss term overwhelms the others without preregistered normalization.

## KL And Probability Distances

RAP does not use KL divergence between deterministic 7D actions. Retrieved
action anchors and residual chunks are real-valued action vectors, not
validated probability distributions. If a later method wants a distributional
memory objective, it must define `p`, `q`, support, estimator, gradient flow,
and why KL is preferred over JS, Wasserstein, MMD, Mahalanobis distance, Huber,
or trajectory discrepancy. That is not part of RAP.

## Mechanism Smoke

Before rollout, report for representative states:

- Base action;
- retrieved anchor action;
- RAP action;
- residual target norm;
- predicted residual norm;
- gate value;
- retrieval confidence and top-k diversity;
- changed dimensions;
- normalized action validity;
- postprocessed 7D action validity;
- clean-versus-shift context.

Expected behavior: RAP differs from Base and anchor-only only where retrieval
and residual targets are active, and differences are bounded rather than
global.

## Stage 0 Decision Rules

Return `RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE` when:

- source paths, memory rows, features, actions, or timestamps are missing or
  unaligned;
- duplicate, missing, extra, or split-overlap keys are nonzero;
- retrieval neighborhoods collapse;
- residual targets have all-zero or all-one structure;
- validation tasks or phases are uncovered.

Return `RAP_STAGE_0_NO_USABLE_HEADROOM` when:

- retrieved anchors do not beat task/phase means by the frozen margin;
- Base or the OptimusVLA proxy leaves no plausible residual failure for RAP;
- anchor distribution is too weak to support the claim.

Return `RAP_STAGE_0_DESIGN_FAILURE` when:

- residual targets are not predictable from deployment inputs;
- anchor-only/no-residual is equivalent to RAP;
- RAP's residual branch is nonacting;
- the mechanism activates everywhere rather than in relevant states.

Return `RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` when:

- proposal/source hashes mismatch;
- checkpoint persistence or reload fails;
- Base passthrough at initialization fails;
- expected trainable gradients are zero or nonfinite;
- frozen Base parameters receive gradients;
- postprocessed 7D action validity fails;
- action deltas are globally destructive;
- any exception occurs.

Return `RAP_STAGE_0_PASS_TO_BOUNDED_VALIDATION` only if all Stage 0 gates pass.

These outcomes are development-only audit decisions, not closed-loop
scientific kills.

## Frozen Baselines And Ablations

The first serious comparison remains exactly five policies:

1. `smolvla_base`;
2. `optimusvla_memory_prior_proxy`;
3. `rap_full`;
4. `rap_anchor_only_no_residual`;
5. `standard_lora`.

No additional simple baseline may replace matched standard LoRA unless
Reviewer B explicitly reopens the protocol before confirmatory testing. No
prior proxy may be weakened after validation performance is seen.
