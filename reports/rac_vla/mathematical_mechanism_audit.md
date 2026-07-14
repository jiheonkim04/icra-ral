# RAC-VLA Mathematical Mechanism Audit

Date: 2026-07-14 KST

Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

## Variables And Shapes

- `B`: batch size.
- `H`: history horizon, selected from `{2, 4}` during validation.
- `s_t in R^8`: robot state.
- `a_t in R^7`: frozen base action.
- `p_t in R^7`: previous action.
- `d_t = s_{t+1} - s_t in R^8`: observed action consequence.
- `rho_t in R^1`: chunk phase.
- `u_t in {0, 1}^2`: task one-hot for the two development tasks.
- `x_t = [s_t, a_t, p_t, d_t, rho_t, u_t] in R^33`: one consequence tuple.
- `h_t in R^{H x 33}`: recent consequence history.
- `k in {0, ..., K-1}`: predeclared synthetic calibration class.
- `z_t in R^m`: calibration context.
- `q_theta(k | h_t) in Delta^{K-1}`: explicit class distribution.
- `r_theta(s_t, a_t, z_t) in R^7`: residual calibration.
- `g_t in {0, 1}` or `[0, 1]`: validation-calibrated gate.
- `a'_t in R^7`: calibrated action.

Forbidden inference inputs:

- terminal success;
- reward;
- reset identity;
- object pose;
- future state beyond the already observed previous consequence;
- future action;
- confirmatory-test labels.

## Perturbation Classes

The first implementation uses a finite predeclared set of synthetic action-channel transforms for Stage 0 only:

- `identity`;
- translation scale down;
- translation scale up;
- x-axis attenuation;
- y-axis attenuation;
- small gripper bias.

Each hidden transform `S_k: R^7 -> R^7` maps a command into the action actually applied by the environment. For a clean trace action `a_t`, Stage 0 constructs the synthetic command `c_t = S_k^{-1}(a_t)` and pairs that command with the observed trace consequence `d_t`. The label `k` is therefore an explicit class label, not a hidden probability distribution.

## Context Encoder

The minimal Stage 0 encoder may be linear or a small MLP:

`z_t = f_theta(mean_{i=t-H}^{t-1} phi(x_i))`

where `phi` is either the identity on standardized features or a one-hidden-layer MLP. Standardization parameters are computed on train identities only.

## Classification Objective

`q_theta(k | h_t) = softmax(W_q z_t + b_q)`

`L_cal = - mean_b log q_theta(k_b | h_b)`

Shape:

- logits: `[B, K]`;
- labels: `[B]`;
- loss: scalar.

Gradient path:

- `W_q`, `b_q`, and any encoder parameters receive gradients.
- Frozen SmolVLA receives no gradients.

Units:

- dimensionless cross-entropy over explicit perturbation classes.

Required ablations:

- action-only classifier using `[a_t, p_t, rho_t, u_t]` without `d_t`;
- no-consequence history classifier using `[s_t, a_t, p_t, rho_t, u_t]` without `d_t`;
- full action-consequence classifier.

Stage 0 hard stop:

- full must beat both action-only and no-consequence validation metrics by at least `5` accuracy points or `5%` relative validation error, whichever metric is used in the implementation.

## Residual Calibration

Residual:

`r_t = alpha tanh(W_r [standardize(s_t), standardize(a_t), z_t, rho_t, u_t] + b_r)`

Action:

`a'_t = clip_group(a_t + g_t r_t)`

The deterministic template residual for class `k` is `S_k^{-1}(a_t) - a_t`; the learned or selected residual must obey the same caps below.

Caps:

- `||a'_t - a_t||_2 <= 0.20`;
- translation residual L2 cap `0.10`;
- rotation residual L2 cap `0.10`;
- gripper residual absolute cap `0.05`;
- finite action validity must be `1.0`.

Identity preservation:

- `alpha = 0` or `W_r = 0` at initialization;
- gate threshold initialized closed;
- selected config must prove bounded nonzero action change only when mechanism activates.

## Gate

Let `c_t = max_k q_theta(k | h_t)`.

Let `stab_t` be the fraction of the last `H` predicted classes equal to the modal class.

`g_t = 1[c_t >= tau and stab_t >= eta and predicted_class != identity]`

Validation selects `tau` and `eta` only through the bounded configuration search.

Hard stops:

- mean gate positive fraction `< 0.02` or `> 0.98`;
- p95 clean action delta `> 0.20`;
- clean validation action validity `< 1.0`.

## Loss Scale And Gradient Audit

Before any training beyond Stage 0, report on a small batch:

- `L_cal`;
- optional `L_ret = Huber(a'_t - a_t)`;
- optional `L_bound`;
- gradient norm of encoder parameters;
- gradient norm of residual head parameters;
- ratio of largest to smallest nonzero gradient norm.

If multiple losses are used:

- normalize action groups before computing Huber terms;
- choose coefficients on validation data only;
- freeze coefficients before confirmatory test.

## Why Not KL

No KL divergence is used. SmolVLA deterministic 7D actions and flow vectors are not treated as normalized probability distributions. The only probability distribution in Stage 0 is `q_theta(k | h_t)`, an explicit softmax over synthetic perturbation classes; cross-entropy is the correct supervised objective for that quantity.

## Simpler Alternatives

- static inverse gain;
- online diagonal inverse-gain;
- action-only perturbation classifier;
- no-consequence history classifier;
- Reflective-history template proxy.

RAC must beat the online inverse-gain simple killer and no-consequence ablation before it can become a paper candidate.
