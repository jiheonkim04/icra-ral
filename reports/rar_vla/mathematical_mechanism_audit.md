# RAR-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `RAR-VLA`

Proposal hash: `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`

Researcher rebuttal: `reports/rar_vla/researcher_rebuttal.md`

Decision: `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`

## Variables And Shapes

Per record or rollout step:

- `I_t`: RGB observation streams available to SmolVLA. Shape implementation
  dependent, normally two image tensors.
- `s_t`: proprioceptive state. Shape `[8]`.
- `u`: language/task instruction or task proxy. Shape implementation
  dependent; local Stage 0 may use task one-hot only as a language proxy.
- `b_t`: frozen SmolVLA Base 7D action at frame `t`. Shape `[7]`.
- `y_t`: demonstration target 7D action at frame `t`, development labels only.
  Shape `[7]`.
- `B_t`: frozen SmolVLA Base action chunk if available. Shape `[H_b, 7]`.
- `e_{t-k}`: previously emitted RAR/Base action. Shape `[7]`.
- `m_t`: causal memory feature built from previous emitted actions, previous
  Base chunks, proprioception deltas, and task proxy. Shape `[d_m]`.
- `x_t`: legal residual-predictor feature. Shape `[d_x]`.
- `rhat_t`: predicted residual action. Shape `[7]`.
- `g_t`: scalar or 7D gate in `[0, g_max]`. Shape `[]` or `[7]`.
- `delta_t = g_t * rhat_t`: applied residual. Shape `[7]`.
- `a_t = b_t + delta_t`: emitted RAR action. Shape `[7]`.
- `rho_t = y_t - b_t`: development-only target residual. Shape `[7]`.

Legal inference features:

`x_t = concat(s_t, b_t, task_proxy(u), m_t)`.

If hidden Base features are later exposed through the official implementation,
they may be appended only after the source gate confirms they are functions of
current deployment inputs and Base computation.

Forbidden inference features:

- future actions or future action segments;
- `y_t` or future labels at inference;
- CALA latent labels;
- future observations;
- success, reward, or failure labels;
- reset identity, manifest key, or held-out outcome;
- simulator object pose, object state, target placement, or oracle phase.

## Causal Memory Construction

For a history horizon `K`, define:

`M_t = [e_{t-1}, ..., e_{t-K}, b_{t-1}, ..., b_{t-K}, s_t - s_{t-1}, ..., s_{t-K+1} - s_{t-K}]`.

Missing early-history values are padded with the first available legal value
and a binary mask. The mask is legal because it is a function of rollout time
since reset, not of future outcome. Reset identity itself is forbidden.

The re-anchoring feature is:

`q_t = b_t - b_{t-1}` when a new Base chunk or Base action refresh is observed,
else zeros.

Stage 0 must report whether the implementation can distinguish inter-chunk
refreshes without using hidden reset or confirmatory metadata. If not, use a
frame-local Base-difference refresh proxy and label it as such.

## Action Formula

The preferred adapter is a post-Base residual wrapper:

`rhat_t = f_theta(x_t)`

`g_t = g_max * sigmoid(alpha_t)`, with `alpha_t` initialized so `g_t = 0` or
numerically zero at initialization.

`a_t = clip_bounds(b_t + g_t * rhat_t)`.

The residual scale is bounded by config:

`||g_t * rhat_t||_2 <= residual_scale * action_scale`.

Translation, rotation, and gripper components are bounded separately.

Fallback hidden-state adapter:

If and only if the SmolVLA action-expert hidden-state interface is safely
available, a zero-initialized hidden adapter may predict `z_delta_t` from
`x_t`. The emitted action must still be audited as above, and Base passthrough
must be exact at initialization.

## Objective Terms

Development training objective, validation only after Stage 0 pass:

`L = lambda_res * L_res + lambda_clean * L_clean + lambda_delta * L_delta + lambda_gate * L_gate + lambda_smooth * L_smooth`

Residual prediction:

`L_res = mean Huber(f_theta(x_t), rho_t)`.

Clean retention:

`L_clean = mean ||a_t - b_t||_2^2` on records where the residual gate is not
expected to activate.

Bounded delta:

`L_delta = mean max(0, ||delta_t||_2 - tau_delta)^2`.

Gate sparsity/localization:

`L_gate = mean |g_t|`.

Action continuity:

`L_smooth = mean ||(a_t - a_{t-1}) - (y_t - y_{t-1})||_Huber`.

All loss magnitudes and gradient norms must be estimated on a small batch
before validation training. No term may dominate merely by units. If gradient
norm ratio exceeds `100:1`, normalize or stop for mathematical audit repair
before training.

## Gradient Paths

Allowed gradients:

- through `f_theta` residual head;
- through gate parameters;
- through optional memory encoder parameters;
- through optional zero-initialized hidden adapter parameters.

Frozen paths:

- SmolVLA Base weights stay frozen in the first prototype;
- Base action `b_t` is treated as a detached input for residual-head training;
- target action labels are used only in development/validation losses.

Forbidden gradients:

- no gradient through confirmatory-test outcomes;
- no reward/success optimization on confirmatory identities;
- no hidden object-pose or simulator-state supervision.

## Mathematical Distance Rules

Do not compute KL divergence between deterministic 7D actions, Base flow
vectors, or residual vectors. They are not normalized probability
distributions.

Permitted distances:

- Huber or L2 residual error;
- Mahalanobis or normalized L2 for component-scaled action deltas;
- action jerk/discontinuity metrics;
- trajectory discrepancy over legal development records;
- paired closed-loop success deltas after frozen rollout.

## Simpler Alternatives And Required Ablations

Required alternatives:

- `ema_action_history_baseline`: exponential moving average or causal smoothing
  over recent Base/emitted actions;
- linear-history residual baseline;
- `rar_no_reanchor_memory_ablation`: removes re-anchor feature or freezes memory
  update to a simple history vector;
- `ar_vla_reanchored_expert_proxy`: transparent closest-prior proxy until
  official equivalence is established.

RAR is killed or stopped if:

- EMA/history baseline matches or beats it;
- the no-reanchor-memory ablation matches or beats it;
- the AR proxy dominates under the matched claim axis;
- the learned residual is global smoothing rather than localized activation.

## Stage 0 Mathematical Audit Requirements

Before validation search:

1. Source gate:
   - all inference features are legal current or past deployment values;
   - no future actions, CALA latents, success labels, reset identities, object
     pose, or confirmatory outcomes.

2. Headroom:
   - Base has measurable inter-chunk or intra-chunk discontinuity/residual
     headroom;
   - diagnostic oracle residual is useful but not an inference method.

3. Observability:
   - legal RAR features beat EMA and linear-history predictors by the
     preregistered margin.

4. Identity:
   - initial action delta p95 at most `1e-6`;
   - Base action validity `1.0`;
   - component deltas bounded.

5. Mechanism:
   - expected parameters receive finite nonzero gradients;
   - residual activation is localized to high-headroom contexts;
   - checkpoint save/reload succeeds before rollout.

Allowed Stage 0 outcomes:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

## Validation Search Budget

Run only if Stage 0 passes.

Maximum six configs:

1. `rar_h4_s003_linear`
2. `rar_h8_s003_linear`
3. `rar_h16_s003_linear`
4. `rar_h4_s006_mlp`
5. `rar_h8_s006_mlp`
6. `rar_h16_s006_mlp`

No other search dimension may be added before confirmatory testing.

## First Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `ar_vla_reanchored_expert_proxy`
3. `rar_full`
4. `rar_no_reanchor_memory_ablation`
5. `ema_action_history_baseline`

## Audit Decision

RAR's mathematical form is valid only under the narrowed claim and source gates.
Proceed to preregistration and prototype protocol. Do not implement, train, or
roll out before those documents freeze Stage 0.
