# G3P-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Audit decision: `G3P_MATHEMATICAL_AUDIT_PREREGISTERED`

## Core Mechanism

G3P adds a legal, deployment-observable spatial point pathway to a frozen SmolVLA policy. A point predictor estimates a task target point from RGB/language/proprioception, converts it to a gripper-relative 3D displacement, and feeds that displacement through an identity-preserving adapter around the action interface.

The frozen SmolVLA Base remains unchanged. G3P may train only the point predictor, point-confidence head, and small action-conditioning adapter unless a later preregistration explicitly freezes a nontrainable point source. Any implementation that uses simulator object pose, placement coordinate, reset identity, reward, success flag, future observation, or confirmatory metadata at inference is invalid.

## Variables And Shapes

- `B`: batch size.
- `V`: number of RGB views exposed by the official runner.
- `I_t`: deployment RGB observation tensor after official preprocessing. Shape is processor-defined, normally equivalent to `B x V x C x H x W`.
- `u_t`: language instruction or official task-language input.
- `s_t in R^{S}`: proprioceptive state available to Base through the official runner.
- `p_g_t in R^3`: gripper position in the declared robot or camera frame, derived only from legal proprioception or official deployment observations.
- `p_star_t in R^3`: train-only target point label from discovery/validation geometry or pseudo-labels. Forbidden at inference.
- `m_t in {0,1}`: train-only point-validity label.
- `p_hat_t = f_eta(I_t, u_t, s_t) in R^3`: deployment-observable predicted target point.
- `c_hat_t = f_eta^c(I_t, u_t, s_t) in [0,1]`: predicted point confidence.
- `d_hat_t = p_hat_t - p_g_t in R^3`: predicted gripper-relative displacement.
- `z_t in {0,1}`: unknown/no-point indicator, where `z_t = 1[c_hat_t < tau_c]`.
- `h_base_t`: frozen Base hidden feature if exposed by the local runner. Shape is implementation-defined.
- `A_base_t in R^{50 x 7}`: frozen SmolVLA postprocessed action chunk.
- `e_t = E_phi(norm(d_hat_t), c_hat_t, z_t) in R^M`: point embedding.
- `g_t in [0,1]`: scalar or group gate for adapter activation.
- `Delta_t in R^{50 x 7}`: bounded action-conditioning delta.
- `A_g3p_t in R^{50 x 7}`: emitted postprocessed action chunk after G3P adapter and official action bounds.

Action dimension order is fixed to the local LIBERO 7D convention:

`a = (dx, dy, dz, droll, dpitch, dyaw, grip)`.

## Source Legality

Legal inference inputs:

- `I_t`;
- `u_t`;
- `s_t`;
- `p_g_t` if derivable from deployment proprioception or official observations;
- `A_base_t` and `h_base_t` if available through the frozen Base runner.

Forbidden inference inputs:

- `p_star_t`;
- simulator object pose;
- target placement coordinate;
- reset identity;
- task success;
- reward;
- future observation or action;
- confirmatory manifest metadata;
- any label generated from confirmatory outcomes.

Stage 0 must produce a source inventory proving that every runtime field used by `f_eta`, `E_phi`, gates, and adapters is legal.

## Coordinate Frame And Units

The implementation must freeze exactly one point frame before validation search:

- robot base frame;
- end-effector frame;
- or camera frame with a declared camera-to-robot transform.

The default preferred representation is gripper-relative:

`d_hat_t = p_hat_t - p_g_t`

with `d_hat_t` in meters before normalization.

Normalization:

`d_norm_t[j] = clip(d_hat_t[j] / sigma_p[j], -d_max, d_max)`

where `sigma_p in R^3_+` is a train-split robust scale and `d_max` is a fixed clipping bound. Both are frozen before validation search.

Stage 0 must sanity-check example signs and scales. A frame/sign/unit failure is `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

## Action Formula

Default identity-preserving adapter:

`e_t = E_phi([d_norm_t, c_hat_t, z_t])`

`r_psi_t = R_psi(h_base_t, e_t, A_base_t) in R^{50 x 7}`

`g_t = sigmoid(b_g + G_psi(h_base_t, e_t))`

Group-clipped delta:

`Delta_t = group_clip(g_t * r_psi_t, alpha_trans, alpha_rot, alpha_grip)`

Emission:

`A_g3p_t = clip_action(A_base_t + Delta_t)`

where:

- `group_clip` clips translation dimensions `0:3`, rotation dimensions `3:6`, and gripper dimension `6` separately;
- `alpha_trans`, `alpha_rot`, and `alpha_grip` are validation-selected or preregistered caps;
- `R_psi` output projection is zero-initialized or `b_g` is initialized closed so `A_g3p_t = A_base_t` at initialization up to numerical tolerance.

Alternative hidden-state injection is allowed only if it proves the same initial Base equality and the same bounded emitted action deltas. If hidden injection cannot expose those checks, use the default post-action adapter.

## Point Prediction Objective

Point regression loss:

`L_point = mean_t m_t * Huber_delta((p_hat_t - p_star_t) / sigma_p)`

Confidence loss:

`L_conf = mean_t BCEWithLogits(l_c_t, m_t)`

Optional point-consistency loss over augmentations:

`L_cons = mean_t ||p_hat_t - p_hat_aug_t||_1`

All point labels must come from discovery/training/validation data only. Confirmatory identities may not be used to fit `f_eta`, `sigma_p`, `tau_c`, or any pseudo-label rule.

## Action Adapter Objective

If expert actions are available on legal training identities:

- `A_exp_t in R^{50 x 7}` is the expert or demonstration action chunk in the same postprocessed action units where available;
- if only current action is available, the loss applies to the current action `a_exp_t in R^7` and the corresponding first action of the chunk.

Action imitation:

`L_act = mean_t Huber_delta((A_g3p_t - A_exp_t) / scale_a)`

Base retention:

`L_ret = mean_t w_clean_t * ||A_g3p_t - stopgrad(A_base_t)||_2^2`

Delta regularizer:

`L_delta = mean_t ||Delta_t / scale_a||_2^2`

Confidence calibration:

`L_gate = mean_t BCEWithLogits(l_g_t, y_gate_t)`

where `y_gate_t` is a train-only intervention-useful label, if such labels are noncollapsed and defined before validation search. If no noncollapsed gate labels exist, omit `L_gate` and use confidence thresholding only.

Full default objective:

`L = lambda_point L_point + lambda_conf L_conf + lambda_act L_act + lambda_ret L_ret + lambda_delta L_delta + lambda_gate L_gate`

Default development coefficients before validation:

- `lambda_point = 1.0`
- `lambda_conf = 1.0`
- `lambda_act = 1.0`
- `lambda_ret = 0.10`
- `lambda_delta = 0.10`
- `lambda_gate = 1.0` only if `y_gate_t` exists and is noncollapsed.

Any coefficient change must occur only inside the bounded validation search and be frozen before confirmatory testing.

## Gradient Path

Gradients may flow into:

- point predictor parameters `eta`;
- point confidence head;
- point embedding `E_phi`;
- gate and adapter parameters `psi`;
- optional small projection layers around exposed Base features.

No gradients may flow into:

- frozen SmolVLA Base weights;
- label construction;
- oracle geometry labels;
- confirmatory identities or outcomes;
- simulator object-state sources.

If the closest-prior proxy uses a nontrainable point source, that source has no gradient path by design and must be labeled separately.

## Small-Batch Magnitude Audit

Before expensive training, validation search, or rollout, report on a development-only batch:

- `L_point`;
- `L_conf`;
- `L_act` when action labels exist;
- `L_ret`;
- `L_delta`;
- `L_gate` when gate labels exist;
- gradient norm for point predictor;
- gradient norm for confidence head;
- gradient norm for adapter/gate;
- ratio of largest to smallest finite nonzero intended gradient norm;
- point-label positive/negative counts;
- point-label coordinate variance;
- confidence positive fraction;
- initial action delta p95 by translation, rotation, and gripper.

Hard stop when any intended trainable component has zero, nonfinite, or catastrophically dominant gradients and the issue cannot be isolated as a narrow implementation defect.

## Validation Search

At most six configurations may be tried.

Allowed factors:

- point confidence threshold `tau_c`: at most three values;
- adapter scale or group caps: at most three values;
- point encoder architecture: at most two choices.

No combinatorial grid beyond six named configurations is allowed.

Validation score:

`S = 0.30 * point_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.10 * simple_baseline_margin + 0.05 * efficiency`

Do not select the final configuration by offline action L2 alone.

## Simpler Alternatives

Closest-prior proxy:

- `g3p_3d_point_proxy`: same legal point source and gripper-relative 3D displacement when possible, with the closest transparent local approximation of direct 3D point injection. It is not an official reproduction unless official code/checkpoint/protocol equivalence is later proven.

Key ablation:

- `g3p_no_3d_no_injection_ablation`: removes gripper-relative 3D displacement and action-head/action-interface injection while matching data, Base policy, and parameter budget as closely as possible.

Simple reviewer-killer:

- `simple_2d_phase_or_nearest_object_heuristic`: strongest validation-selected 2D point, task-phase, or nearest-object heuristic before confirmatory testing.

## Why Not KL

G3P does not compute KL between deterministic 7D actions. `A_base_t`, `A_g3p_t`, `A_exp_t`, `Delta_t`, and `d_hat_t` are deterministic vectors in action or metric spatial units, not normalized probability distributions.

Appropriate discrepancies are Huber, L1, L2, BCE for explicit Bernoulli confidence/gate labels, and task-success deltas under paired closed-loop evaluation.

If a future implementation proposes KL, it must define `p`, `q`, support, estimator, gradient flow, and why KL is preferred over JS, Wasserstein, MMD, Mahalanobis, Huber/L2, vector-field consistency, or trajectory discrepancy. That is not part of this audit.

## Stage 0 Audit Requirements

Stage 0 must verify:

- discovery/validation/confirmatory identity separation;
- legal source inventory;
- no privileged inference fields;
- point-label counts, balance, coordinate variance, task coverage, and phase coverage;
- duplicate frame/sample/reset counts;
- oracle headroom diagnostic;
- point predictability above majority, task/language-only, phase, 2D, or nearest-object baselines;
- confidence noncollapse;
- initial Base passthrough;
- action validity and group delta bounds;
- checkpoint save and reload for any trainable source or adapter;
- intended parameters receive finite nonzero gradients;
- full, proxy, ablation, and simple heuristic are not trivially action-equivalent unless that equivalence is the recorded stop.

Stage 0 hard stops:

- `DATA_OR_SUPERVISION_FAILURE`: legal deployable point labels or predictors unavailable, collapsed, duplicated, or split-leaky.
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`: Base, closest-prior proxy, and oracle diagnostics show no plausible spatial improvement surface.
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`: frame/unit/sign failure, invalid action shape, no gradients, checkpoint reload failure, or failure to preserve Base behavior.
- `DESIGN_FAILURE`: point source is not observable from deployment inputs, is explained by trivial heuristics, or activates globally without context.

These are pre-rollout stops, not closed-loop scientific kills.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `g3p_3d_point_proxy`
3. `g3p_full`
4. `g3p_no_3d_no_injection_ablation`
5. `simple_2d_phase_or_nearest_object_heuristic`

Primary metric:

- task-balanced official closed-loop success.

Secondary metrics:

- paired wins/losses/ties;
- paired bootstrap confidence intervals;
- per-task success;
- point predictability;
- point confidence and activation distribution;
- translation, rotation, and gripper deltas;
- clean retention;
- action validity;
- latency and VRAM.

## Mathematical Decision

`G3P_MATHEMATICAL_AUDIT_PREREGISTERED`

G3P is mathematically coherent only as a source-gated, identity-preserving, bounded 3D point-conditioning method. The audit permits preregistration and prototype protocol drafting. It does not permit implementation, Stage 0 data construction, validation search, training, manifest freeze, or rollout until the preregistration and prototype protocol are frozen.
