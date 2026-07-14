# MARC-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Method: `MARC-VLA`, Median-Anchored Regression Correction for frozen SmolVLA flow actions.

Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`

## Variables And Shapes

- `B`: batch size.
- `x_t`: deployment observation processed by the official SmolVLA input path. Shape is processor-defined.
- `s_t in R^8`: robot proprioceptive state when available from the stable artifact.
- `u_t`: language instruction or transparent task-language proxy in development diagnostics.
- `rho_t in R^1`: normalized phase or chunk-position feature when available.
- `a_base_t in R^7`: frozen SmolVLA current action in normalized LIBERO 7D units.
- `a_exp_t in R^7`: expert demonstration action in the same units.
- `m_theta(x_t, s_t, u_t, rho_t, a_base_t) in R^7`: robust median anchor.
- `z_t in {0,1}`: train-only disagreement label.
- `g_phi(x_t, s_t, u_t, rho_t, a_base_t) in [0,1]`: learned disagreement gate.
- `alpha in R_+`: correction cap selected on validation only.
- `c_t in R^7`: clipped correction toward the anchor.
- `a_marc_t in R^7`: emitted action.

Forbidden inference inputs:

- reward or terminal success;
- reset identity;
- future action or future state;
- simulator object pose unless it is part of Base's deployment observation;
- confirmatory-test labels or identities.

## Disagreement-Label Construction

Disagreement labels are generated only from discovery/training/validation identities before confirmatory testing.

For training records:

`d_t = ||a_exp_t - a_base_t||_2`

Train-only threshold:

`tau_disagree = quantile_train(d_t, 0.60)`

Label:

`z_t = 1[d_t > tau_disagree]`

The `0.60` quantile is fixed before Stage 0 to avoid tuning labels from validation or confirmatory outcomes. Stage 0 must report the resulting positive fraction on train and validation.

## Action Formula

Raw anchor displacement:

`q_t = m_theta(x_t, s_t, u_t, rho_t, a_base_t) - stopgrad(a_base_t)`

Clipped correction:

`c_t = q_t * min(1, alpha / (||q_t||_2 + eps))`

Emission:

`a_marc_t = clip_action(a_base_t + g_phi(x_t, s_t, u_t, rho_t, a_base_t) * c_t)`

Initial condition:

- anchor correction output is zero-initialized or gate bias is initialized closed;
- therefore `a_marc_t = a_base_t` up to numerical tolerance before training.

## Objective

Anchor loss:

`L_anchor = mean_t Huber_delta((m_theta_t - a_exp_t) / scale_action)`

Action scale:

`scale_action_j = median_train(|a_exp_t[j] - median_train(a_exp[j])|) + eps`

Gate loss:

`L_gate = mean_t BCEWithLogits(logit_phi_t, z_t)`

Delta regularizer:

`L_delta = mean_t ||g_phi_t * c_t||_2^2`

Clean-retention regularizer:

`L_clean = mean_t (1 - z_t) * ||g_phi_t * c_t||_2^2`

Full objective:

`L = L_anchor + lambda_gate L_gate + lambda_delta L_delta + lambda_clean L_clean`

Default development coefficients before validation:

- `lambda_gate = 1.0`
- `lambda_delta = 0.10`
- `lambda_clean = 0.10`

Any coefficient change must occur only inside the bounded validation search and must be frozen before confirmatory testing.

## Gradient Path

Gradients flow into:

- median-anchor parameters;
- gate parameters;
- optional small feature projection parameters.

No gradients flow into:

- frozen SmolVLA Base;
- persisted base actions;
- label construction;
- confirmatory-test identities.

## Small-Batch Magnitude Audit

Before expensive training, report on a development-only batch:

- `L_anchor`;
- `L_gate`;
- `L_delta`;
- `L_clean`;
- gradient norm of anchor parameters;
- gradient norm of gate parameters;
- ratio of largest to smallest finite nonzero gradient norm;
- base/expert disagreement magnitudes;
- disagreement positive/negative counts;
- initial action delta p95.

Hard stop when expected parameters receive zero, nonfinite, or catastrophically imbalanced gradients and the issue cannot be attributed to a narrow implementation defect.

## Simpler Alternatives

Closest-prior proxy:

- `openvla_oft_l1_proxy`: continuous L1/Huber action adapter trained on the same records without MARC's learned disagreement gate. It is a faithful transparent local proxy, not an official OpenVLA-OFT reproduction.

Key ablation:

- `marc_no_disagreement_gate_ablation`: same anchor and cap, but uses a fixed always-on or validation-selected constant gate.

Simple killer:

- `static_l1_mixture_baseline`: static convex mixture of Base and the L1 proxy selected on validation only.

## Why Not KL

MARC does not compute KL between deterministic 7D actions. `a_base_t`, `a_exp_t`, `m_theta_t`, `c_t`, and `a_marc_t` are deterministic vectors in normalized action units, not normalized probability distributions. L1/Huber and L2-style regularization are the appropriate local discrepancy terms.

The only Bernoulli variable is the disagreement gate label. Its supervised objective is binary cross-entropy over explicit labels, not action-space KL.
