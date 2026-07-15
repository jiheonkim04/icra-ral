# CALA-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

Proposal hash: `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Audit decision: `CALA_MATHEMATICAL_AUDIT_PREREGISTERED`

## Core Mechanism

CALA adds a deployment-observable latent-action pathway to a frozen SmolVLA
policy. Future 7D demonstration action segments are encoded into latent-action
labels on discovery/validation records only. A predictor estimates those
latents from current RGB/proprioception/language/Base features, then injects
the predicted latent through a zero-initialized context-gated adapter around the
SmolVLA action interface.

The frozen SmolVLA Base remains unchanged. CALA may train only the latent
predictor, gate, latent projection, and small adapter. Any implementation that
uses future actions, latent labels, reset identity, reward, success flag,
simulator state, future observation, or confirmatory metadata at inference is
invalid.

## Variables And Shapes

- `B`: batch size.
- `V`: number of RGB views exposed by the official runner.
- `I_t`: deployment RGB observation tensor after official preprocessing. Shape
  is processor-defined, normally equivalent to `B x V x C x H x W`.
- `u_t`: language instruction or official task-language input.
- `s_t in R^S`: proprioceptive state available to Base through the official
  runner.
- `h_base_t in R^D` or `B x T_h x D`: frozen Base feature or hidden tensor if
  exposed by the local runner.
- `A_base_t in R^{50 x 7}`: frozen SmolVLA postprocessed action chunk.
- `A_future_t in R^{H x 7}`: train-only future demonstration action segment
  from legal discovery/validation records. Forbidden at inference.
- `H`: latent horizon in action steps.
- `z_t in R^K`: deterministic train-only latent-action label
  `z_t = E(A_future_t)`.
- `zhat_t = P_eta(I_t, u_t, s_t, h_base_t, A_base_t) in R^K`: predicted
  deployment-observable latent action.
- `q_t in [0,1]`: predicted latent confidence or quality score.
- `e_t = E_phi(zhat_t, q_t) in R^M`: latent-action conditioning embedding.
- `g_t in [0,1]` or `R^G`: scalar or group context gate.
- `r_psi_t`: hidden residual or action-interface residual. Shape depends on
  hook; emitted action-equivalent residual must map to `R^{50 x 7}`.
- `Delta_t in R^{50 x 7}`: bounded emitted action delta used only for audit
  and, when the hook is post-action, for emission.
- `A_cala_t in R^{50 x 7}`: emitted postprocessed action chunk after CALA and
  official action bounds.

Action dimension order is fixed to the local LIBERO 7D convention:

`a = (dx, dy, dz, droll, dpitch, dyaw, grip)`.

## Source Legality

Legal inference inputs:

- `I_t`;
- `u_t`;
- `s_t`;
- `A_base_t` and `h_base_t` if produced online by the frozen Base runner from
  the current observation.

Forbidden inference inputs:

- `A_future_t`;
- `z_t`;
- future observations or actions;
- HDF5 future action windows;
- simulator object pose or state;
- reset identity;
- hidden episode progress unavailable at deployment;
- task success;
- reward;
- confirmatory manifest metadata;
- any label or threshold derived from confirmatory outcomes.

Stage 0 must produce a source inventory proving that every runtime field used
by `P_eta`, `E_phi`, `g_t`, and adapter `psi` is legal.

## Latent Encoder

Default deterministic encoder for Stage 0 and initial prototype:

1. Normalize each action dimension using train-split robust scale
   `scale_a in R^7_+`.
2. Flatten the future action segment:
   `x_t = vec(A_future_t / scale_a) in R^{7H}`.
3. Apply a fixed OAT-lite transform `E` chosen before validation search.

Allowed OAT-lite encoders:

- DCT coefficients over each action dimension;
- PCA fit on discovery/train identities only;
- fixed summary statistics: mean, endpoint displacement, first difference,
  and low-frequency coefficients.

Forbidden encoders before a later audit:

- any encoder fit on confirmatory identities;
- any encoder that stores per-episode or per-reset lookup tables;
- any encoder whose latent is a hidden copy of future actions at inference;
- any encoder that requires reward, success, object state, or future frames at
  inference.

Stage 0 must freeze:

- `H`;
- latent dimension `K`;
- action scale `scale_a`;
- encoder type;
- whether encoder parameters are fixed or train-split fit;
- reconstruction or variance diagnostics.

## Action Formula

Preferred hidden-state adapter when a valid SmolVLA hook is exposed:

`zhat_t = P_eta(I_t, u_t, s_t, stopgrad(h_base_t), stopgrad(A_base_t))`

`e_t = E_phi([zhat_t, q_t])`

`g_t = sigmoid(b_g + G_psi(stopgrad(h_base_t), e_t))`

`h_cala_t = h_base_t + g_t * R_psi(stopgrad(h_base_t), e_t)`

`A_cala_t = SmolVLA_action_head(h_cala_t)`

Audit delta:

`Delta_t = A_cala_t - A_base_t`

Fallback post-action adapter if no faithful hidden hook exists:

`r_psi_t = R_psi(stopgrad(h_base_t), e_t, stopgrad(A_base_t)) in R^{50 x 7}`

`Delta_t = group_clip(g_t * r_psi_t, alpha_trans, alpha_rot, alpha_grip)`

`A_cala_t = clip_action(A_base_t + Delta_t)`

The fallback is allowed only as a transparent local proxy path if Reviewer B
constraints are preserved. If the implementation is merely an unconditional
final-action residual, stop as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

Identity initialization:

- `R_psi` final projection is zero-initialized, or `b_g` is initialized so
  `g_t` is near zero;
- therefore `A_cala_t = A_base_t` at initialization up to numerical tolerance;
- Stage 0 must report initial action delta p95 by translation, rotation, and
  gripper.

## Latent Prediction Objective

Latent prediction:

`L_lat = mean_t Huber_delta((zhat_t - z_t) / scale_z)`

where `scale_z` is computed on discovery/train identities only.

Latent confidence or quality:

`L_q = mean_t BCEWithLogits(l_q_t, y_q_t)`

where `y_q_t` is a train-only useful/valid latent label if noncollapsed. If no
noncollapsed `y_q_t` exists, omit `L_q` and use validation-selected gate
thresholding without confidence supervision.

Optional latent reconstruction diagnostic, not a rollout objective:

`L_rec = mean_t ||D(zhat_t) - A_future_t||_Huber`

`D` may be used only for diagnostics on development identities. Decoded future
actions are forbidden at inference.

## Adapter Objective

Expert action labels:

- `A_exp_t in R^{50 x 7}` when full demonstration chunks are available;
- otherwise the loss may apply to first action `a_exp_t in R^7`.

Action imitation:

`L_act = mean_t Huber_delta((A_cala_t - A_exp_t) / scale_a)`

Base retention:

`L_ret = mean_t w_clean_t * ||A_cala_t - stopgrad(A_base_t)||_2^2`

Delta regularizer:

`L_delta = mean_t ||Delta_t / scale_a||_2^2`

Gate sparsity/activation target:

`L_gate = mean_t BCEWithLogits(l_g_t, y_gate_t)`

only if a noncollapsed train-only gate target exists. Otherwise use a bounded
activation-rate penalty:

`L_gate_rate = (mean_t g_t - rho_g)^2`

with `rho_g` selected on validation only.

Full default objective:

`L = lambda_lat L_lat + lambda_q L_q + lambda_act L_act + lambda_ret L_ret + lambda_delta L_delta + lambda_gate L_gate_or_rate`

Default development coefficients before validation:

- `lambda_lat = 1.0`
- `lambda_q = 1.0` only if `y_q_t` exists and is noncollapsed
- `lambda_act = 1.0`
- `lambda_ret = 0.10`
- `lambda_delta = 0.10`
- `lambda_gate = 0.10` for rate penalty or `1.0` for noncollapsed labels

Any coefficient change must occur only inside the bounded validation search and
be frozen before confirmatory testing.

## Gradient Path

Gradients may flow into:

- latent predictor parameters `eta`;
- latent confidence/quality head;
- latent projection `E_phi`;
- gate and adapter parameters `psi`;
- optional small projection layers around exposed Base features.

No gradients may flow into:

- frozen SmolVLA Base weights;
- future action labels;
- latent-label construction;
- confirmatory identities or outcomes;
- simulator state or reward sources.

The closest-prior proxy may use a different transparent local conditioning path
only if its gradient path, source use, and missing official components are
documented.

## Small-Batch Magnitude Audit

Before expensive training, validation search, or rollout, report on a
development-only batch:

- `L_lat`;
- `L_q` when applicable;
- `L_act` when action labels exist;
- `L_ret`;
- `L_delta`;
- `L_gate` or `L_gate_rate`;
- gradient norm for latent predictor;
- gradient norm for confidence/quality head;
- gradient norm for adapter/gate;
- ratio of largest to smallest finite nonzero intended gradient norm;
- latent variance by dimension;
- task/phase coverage;
- task-mean baseline latent error;
- action-history baseline latent error;
- initial action delta p95 by translation, rotation, and gripper.

Hard stop when any intended trainable component has zero, nonfinite, or
catastrophically dominant gradients and the issue cannot be isolated as a
narrow implementation defect.

## Validation Search

At most six configurations may be tried.

Allowed factors:

- latent horizon `H`: at most three values;
- gate/residual scale: at most three values;
- adapter architecture: at most two choices.

No combinatorial grid beyond six named configurations is allowed.

Validation score:

`S = 0.25 * latent_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.15 * simple_baseline_margin + 0.05 * efficiency`

Do not select the final configuration by offline action L2 alone.

## Simpler Alternatives

Closest-prior proxy:

- `cac_vla_latent_action_proxy`: closest transparent local approximation of
  CAC-style latent-action conditioning. It is not an official reproduction
  unless official code/checkpoint/protocol equivalence is later proven.

Key ablation:

- `cala_no_context_gate_ablation`: same latent labels and training budget, but
  the context-dependent gate is removed or disabled.

Simple reviewer-killer:

- `task_mean_latent_action_baseline`: task- or instruction-conditioned latent
  prototype without current observation-specific context.

## Why Not KL

CALA does not compute KL between deterministic 7D actions. `A_base_t`,
`A_cala_t`, `A_exp_t`, `Delta_t`, `z_t`, and `zhat_t` are deterministic vectors
in action, latent, or normalized units, not normalized probability
distributions.

Appropriate discrepancies are Huber, L1, L2, BCE for explicit Bernoulli labels,
activation-rate penalties, and paired closed-loop task-success deltas.

If a future implementation proposes KL, it must define `p`, `q`, support,
normalization, estimator, gradient flow, and why KL is preferred over JS,
Wasserstein, MMD, Mahalanobis, Huber/L2, vector-field consistency, or trajectory
discrepancy. That is not part of this audit.

## Stage 0 Audit Requirements

Stage 0 must verify:

- discovery/validation/confirmatory identity separation;
- legal source inventory;
- no future-action or latent-label inference access;
- latent label variance, balance, and task/phase coverage;
- duplicate sample/frame count;
- train/validation/test overlap count;
- latent predictability above task-mean, phase-only, action-history, and
  majority/trivial baselines;
- diagnostic headroom for latent-action conditioning;
- initial Base passthrough;
- finite nonzero intended gradients;
- action validity;
- clean validation behavior;
- full/proxy/ablation/simple-baseline distinction.

Stage 0 hard-stop classes:

- `DATA_OR_SUPERVISION_FAILURE`;
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `DESIGN_FAILURE`.

These stops are not closed-loop scientific kills.

## First Comparison

The first serious comparison remains exactly:

1. `frozen_smolvla`
2. `cac_vla_latent_action_proxy`
3. `cala_full`
4. `cala_no_context_gate_ablation`
5. `task_mean_latent_action_baseline`

No additional mandatory policy baseline may precede this comparison unless a
concrete Stage 0 ambiguity would otherwise invalidate the five-policy test.
