# DAGR-VLA Mathematical Mechanism Audit

Date: 2026-07-14 KST

Method: `DAGR-VLA`, Dynamic Arm-Gripper Routing for frozen SmolVLA adaptation.

Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`

## Variables And Shapes

- `B`: batch size.
- `x_t`: deployment observation processed by the official SmolVLA input path. Shape is processor-defined.
- `s_t in R^8`: robot proprioceptive state when available from the local stable prediction artifact.
- `u_t`: language instruction or a transparent task-instruction proxy in development diagnostics.
- `rho_t in R^1`: normalized episode phase.
- `a_base_t in R^7`: frozen SmolVLA current action.
- `a_exp_t in R^7`: expert demonstration action.
- `r_t = a_exp_t - stopgrad(a_base_t) in R^7`: expert-minus-base residual target.
- `M_trans in {0,1}^7`: translation mask, dimensions `0:3`.
- `M_rot in {0,1}^7`: rotation mask, dimensions `3:6`.
- `M_grip in {0,1}^7`: gripper mask, dimension `6`.
- `y_t in {0,1}^3`: route labels for translation, rotation, and gripper.
- `g_theta(x_t, s_t, u_t, rho_t) in [0,1]^3`: route gates.
- `d_theta(x_t, s_t, u_t, rho_t) in R^7`: residual proposal.
- `alpha = (alpha_trans, alpha_rot, alpha_grip)`: group residual caps selected only on validation.

Forbidden inference inputs:

- terminal success;
- reward;
- reset identity;
- future action;
- future state;
- object pose unless it is part of the deployment observation available to Base;
- confirmatory-test labels or identities.

## Route-Label Construction

Route labels are generated only from discovery/training/validation identities before confirmatory testing.

For training records, compute group residual magnitudes:

- `m_trans_t = ||r_t[0:3]||_2`
- `m_rot_t = ||r_t[3:6]||_2`
- `m_grip_t = |r_t[6]|`

Train-only thresholds:

- `tau_trans = median_train(m_trans)`
- `tau_rot = median_train(m_rot)`

Gripper transition indicator:

- `z_grip_t = 1` when the expert gripper sign changes between the current frame and either neighboring frame within the same episode;
- otherwise `z_grip_t = 0`.

Material gripper residual:

- `q_grip_t = 1[m_grip_t > 0.02]`.

Labels:

- `y_trans_t = 1[m_trans_t > tau_trans]`
- `y_rot_t = 1[m_rot_t > tau_rot]`
- `y_grip_t = z_grip_t OR q_grip_t`

The fixed `0.02` gripper residual threshold is a development-scale materiality threshold in normalized action units. It is not selected on confirmatory outcomes.

## Action Formula

Group-clipped residual:

`clip_group(v, alpha_g) = v * min(1, alpha_g / (||v||_2 + eps))`

For the scalar gripper group, `||v||_2` is `abs(v)`.

Emission:

`Delta_t = clip_group(g_trans_t * (M_trans * d_theta_t), alpha_trans) + clip_group(g_rot_t * (M_rot * d_theta_t), alpha_rot) + clip_group(g_grip_t * (M_grip * d_theta_t), alpha_grip)`

`a_dagr_t = clip_action(a_base_t + Delta_t)`

Initial condition:

- residual output projection is zero-initialized;
- gate bias is initialized to base passthrough or closed;
- therefore the initial emitted action equals Base up to numerical tolerance.

## Objective

Use group-normalized Huber residual loss, not KL:

`L_res = mean_t sum_g y_g_t * Huber_delta((M_g * d_theta_t - M_g * r_t) / scale_g)`

Group scales are fixed from train residual magnitudes:

- `scale_trans = median_train(m_trans) + eps`
- `scale_rot = median_train(m_rot) + eps`
- `scale_grip = median_train(m_grip) + eps`

Route loss:

`L_route = mean_t BCEWithLogits(logit_g_t, y_g_t)`

Action-delta regularizer:

`L_delta = mean_t ||Delta_t||_2^2`

Clean passthrough regularizer:

`L_clean = mean_t (1 - max_g y_g_t) * ||Delta_t||_2^2`

Full objective:

`L = L_res + lambda_route * L_route + lambda_delta * L_delta + lambda_clean * L_clean`

Default development coefficients before validation:

- `lambda_route = 1.0`
- `lambda_delta = 0.10`
- `lambda_clean = 0.10`

Any coefficient change must occur only inside the bounded validation search and must be frozen before confirmatory testing.

## Gradient Path

Gradients flow into:

- router parameters;
- residual adapter parameters;
- optional small feature projection parameters.

No gradients flow into:

- frozen SmolVLA Base;
- persisted base actions;
- route-label construction;
- confirmatory-test identities.

## Small-Batch Magnitude Audit

Before expensive training, report on a development-only batch:

- `L_res`;
- `L_route`;
- `L_delta`;
- `L_clean`;
- gradient norm of router parameters;
- gradient norm of residual parameters;
- ratio of largest to smallest finite nonzero gradient norm;
- translation, rotation, and gripper residual target magnitudes;
- route positive/negative counts.

Hard stop when expected parameters receive zero, nonfinite, or catastrophically imbalanced gradients and the issue cannot be attributed to a narrow implementation defect.

## Simpler Alternatives

Closest-prior proxy:

- `dam_static_component_proxy`: static component-weighted arm/gripper residual adapter with no dynamic route gates. It is a faithful transparent local proxy, not an official DAM-VLA reproduction.

Key ablation:

- `dagr_no_dynamic_route_ablation`: shared residual adapter with no group-specific dynamic gates.

Simple killer:

- `gripper_transition_heuristic`: bounded gripper timing bias near predicted gripper transitions and no learned group residual.

## Why Not KL

DAGR does not compute KL between deterministic 7D actions. `a_base_t`, `a_exp_t`, `r_t`, and `d_theta_t` are deterministic vectors in normalized action units, not normalized probability distributions. Huber/L2 objectives are the appropriate local discrepancy terms.

The only Bernoulli distributions are the route-label predictions. Their supervised objective is binary cross-entropy over explicit labels, not action-space KL.

