# KITE-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.

Decision: `KITE_MATHEMATICAL_AUDIT_PREREGISTERED`.

## Variables And Shapes

- batch size: `B`;
- action chunk: `A in R^(B,50,7)`;
- normalized noisy flow state: `X_u in R^(B,50,32)`;
- predicted velocity: `V_theta in R^(B,50,32)`;
- valid action slice: first `7` dimensions;
- end-effector state: `s in R^(B,6)`;
- horizons: `H in {5,20}`;
- cumulative raw arm command: `c_H in R^(B,6)`;
- target state displacement: `d_H in R^(B,6)`;
- realization operator: `F_H(c)=W_H c+b_H`, with
  `W_H in R^(6,6)`, `b_H in R^6`.

The checkpoint's fixed affine action processor converts reconstructed
normalized actions to raw LIBERO action units before cumulative integration.

## Discovery Operator

For each horizon, task-balance discovery rows, then compute discovery means and
standard deviations with floor `1e-6` for `c_H` and `d_H`. Fit standardized
ridge regression with intercept and coefficient `1e-4`:

`B_H = argmin_B ||X_H B - Y_H||_F^2 + 1e-4 ||B_no_intercept||_F^2`.

Persist source hashes, split keys, normalization, `B_H`, rank, singular values,
and validation metrics. Gradients never enter `B_H`.

Units:

- raw command sums use checkpoint postprocessor action units;
- `ee_states` displacement uses source state coordinates;
- the Huber objective is dimensionless after discovery standardization.

## Clean Action Reconstruction

SmolVLA constructs

`X_u = u E + (1-u) A`

and targets velocity `U=E-A`. Therefore the predicted clean action is

`A_hat = X_u - u V_theta(X_u,u,o)`.

The `32`-dimensional result is sliced to the valid seven action dimensions and
unnormalized through the fixed affine processor. No postprocessor clipping or
non-differentiable transform is allowed in the gradient path.

## Objectives

For `H in {5,20}`:

`c_hat_H = sum_(j=0)^(H-1) A_hat_(j,1:6)`

`d_hat_H = F_H(c_hat_H)`.

With discovery target normalization `N_H` and coordinate-mean Huber
`rho_1`:

`L_kite = (1/2) sum_H mean rho_1(N_H(d_hat_H)-N_H(d_H))`.

`L_total = L_flow + lambda_k L_kite`,

where `lambda_k in {0.1,0.3,1.0}` is selected on validation only.

Gradient path:

`L_kite -> d_hat -> c_hat -> unnormalized A_hat -> V_theta -> rank-4 LoRA`.

The state target, normalization, and `F_H` receive no gradient.

## Prior Proxy

The transparent GeoPredict-style proxy uses declared pooled SmolVLA
representation `z in R^(B,960)` and one MLP
`960 -> 128 -> 12` to predict both normalized displacements. Its loss is the
same Huber target loss. The head is training-only; gradients pass through `z`
to the same LoRA targets. It does not touch generated actions.

## Key Ablation

The cumulative-action-target ablation replaces `d_hat_H,d_H` with standardized
`c_hat_H,c_H` at both horizons. It uses the same Huber delta, coefficient,
LoRA, data, noise/time draws, optimizer, and steps. This tests whether KITE is
only cumulative-action supervision.

## Scale And Gradient Audit

Before optimization on a fixed discovery batch report:

- `L_flow`, `L_kite`, and total magnitude;
- LoRA gradient norm from each objective separately;
- `||grad_kite|| / max(||grad_flow||,1e-12)`;
- finite fraction;
- expected target-module coverage;
- gradient cosine when both are nonzero.

Reject nonfinite values, zero KITE gradient, unexplained gradient ratio above
`100`, wrong target modules, or any gradient into Base-frozen parameters.
Coefficients may be selected only by the preregistered validation score.

## Simpler Alternatives

- ordinary flow-only standard LoRA;
- GeoPredict-style hidden-state future-kinematics proxy;
- direct cumulative-action-target supervision.

All three enter the first comparison. No KL divergence is used. A deterministic
action chunk is not treated as a probability distribution.

## Identity And Persistence

Rank-4 LoRA B matrices initialize to zero. On identical input, noisy action,
and solver state, Base, initialized KITE, and disk-reloaded initialized KITE
must match native flow vectors and decoded actions within `1e-6`. Hash all
non-LoRA Base parameters before and after. The training-only operator must be
absent from the inference call graph.
