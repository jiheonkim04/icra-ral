# EvoState-VLA Mathematical Mechanism Audit

Date: 2026-07-14 KST

Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Decision: `MATH_AUDIT_PASS_TO_PREREGISTRATION_WITH_HARD_STOPS`

## Variables And Shapes

Per control step `t`:

- `s_t in R^8`: deployment-observable robot state used by SmolVLA.
- `a_t^0 in R^7`: frozen SmolVLA base action before any fault wrapper.
- `a_{t-1} in R^7`: previous executed policy action before fault wrapper.
- `rho_t in R^1`: chunk index fraction in `[0, 1]`.
- `q in {0,1}^2`: task one-hot for `libero_spatial/task_4` and `libero_10/task_4`.
- `x_t = concat(s_t, a_t^0, a_{t-1}, rho_t, q) in R^25`.
- `s_{t+1} in R^8`: next observed robot state in development traces.
- `Delta s_t = s_{t+1} - s_t in R^8`.

Learned components:

- `F_theta: R^25 -> R^8`: predicts `Delta s_t`.
- `B_phi: R^25 -> R^{8 x 7}` or a ridge proxy `B_task_phase in R^{8 x 7}`: local controllability map.
- `G_psi: R^{25 + 8} -> R`: mismatch reliability logit.

Inference state:

- `hat{s}_t in R^8`: action-evolved predicted state.
- `e_t = s_t - hat{s}_t in R^8`: observed minus predicted mismatch.
- `g_t = sigmoid(G_psi([x_t, e_t]) - tau) in [0, 1]`: calibrated gate.
- `delta a_t in R^7`: damped inverse-dynamics correction.
- `a_t = clip(a_t^0 + alpha g_t clip_norm(delta a_t, delta_max)) in R^7`: final policy action before the shared fault wrapper.

## Transition Objective

Formula:

```text
L_dyn(theta) = mean_t Huber(F_theta(x_t), Delta s_t; beta_dyn)
```

Scale and units:

- state units are the native official SmolVLA/LIBERO 8D state units;
- `Delta s_t` is one-step state difference;
- Huber beta is fixed in preregistration.

Gradient path:

- gradients flow only into `theta`;
- frozen SmolVLA receives no gradients;
- no rollout/test identities are used.

Intended effect:

- learn how the deployment-observable robot state should evolve under the base action and chunk phase.

Simpler alternatives:

- constant zero-delta predictor;
- previous mean delta by task;
- actionless model `F(s_t, a_{t-1}, rho_t, q)`;
- per-task linear dynamics.

Required ablation:

- actionless and linear baselines in the development audit.

## Controllability Objective

For a learned `B_phi`, use one-step local linearization:

```text
Delta s_t ~= B_phi(x_t) a_t^0
```

Loss:

```text
L_ctrl(phi) = mean_t Huber(B_phi(x_t) a_t^0, Delta s_t; beta_ctrl)
              + lambda_B ||B_phi(x_t)||_F^2
```

For the ridge proxy:

```text
B = argmin_B ||A B^T - Delta S||_F^2 + lambda_ridge ||B||_F^2
```

where `A in R^{N x 7}` contains actions and `Delta S in R^{N x 8}` contains next-state deltas. The resulting `B in R^{8 x 7}` is estimated separately by task and coarse chunk phase when validation data support it.

Gradient path:

- learned `B_phi`: gradients flow into `phi`;
- ridge proxy: closed-form fit on discovery data only.

Intended effect:

- estimate which state mismatch directions can be corrected by action changes.

Simpler alternatives:

- scalar static inverse gain;
- per-dimension diagonal inverse;
- no correction.

Required ablation:

- `static_inverse_dynamics` simple killer baseline;
- `evostate_no_state_prior_ablation`.

## Reliability Gate Objective

Targets are generated on discovery/validation only.

For validation tuple `t`, define:

- prediction error `r_t = ||F_theta(x_t) - Delta s_t||_2`;
- actionless prediction error `r_t^0`;
- controllable projection ratio:

```text
c_t = ||P_B e_t||_2 / (||e_t||_2 + eps)
```

where `P_B = B^T (B B^T + lambda I)^{-1} B` mapped consistently onto the controllable state subspace. If the projection is numerically invalid, `c_t = 0`.

Gate target:

```text
y_t = 1[ r_t < r_t^0 and c_t >= c_min and ||e_t||_2 in [e_min, e_max] ]
```

Gate loss:

```text
L_gate(psi) = BCEWithLogits(G_psi([x_t, e_t]), y_t)
```

Scale:

- binary target, unitless.

Gradient path:

- gradients flow only into `psi`;
- target generation uses frozen fitted dynamics and validation data only.

Intended effect:

- activate only when mismatch is predictable and likely controllable.

Simpler alternatives:

- fixed mismatch threshold;
- always-on correction.

Required ablation:

- fixed-threshold static inverse baseline;
- no-state-prior ablation.

## Inference Correction

Damped correction:

```text
delta a_t = - B_t^T (B_t B_t^T + lambda I)^{-1} e_t
```

Final action:

```text
a_t = clip_action(a_t^0 + alpha g_t clip_norm(delta a_t, delta_max))
```

Units:

- 7D action native official LIBERO action units.

Gradient path:

- none at inference.

Identity preservation:

- if `g_t = 0`, `a_t = a_t^0` exactly before numeric clipping;
- `alpha` and `delta_max` are fixed before Stage A;
- action bounds are checked before rollout.

Failure modes:

- `B_t B_t^T` ill-conditioned;
- mismatch is object/contact state not visible in `s_t`;
- correction tracks a stale expected state;
- static inverse dynamics explains all gains.

## Objective Scale Audit

Before any training beyond smoke:

- report `L_dyn`, actionless loss, linear baseline loss;
- report gradient norms for `F_theta`, `B_phi`, and `G_psi`;
- report `||delta a||_2` distribution on validation;
- report gate activation fraction;
- report invalid-action fraction;
- report clean passthrough max absolute difference when gate is closed.

Hard stop if:

- any loss is nonfinite;
- any gradient norm is nonfinite;
- learned components receive zero gradients when expected to train;
- one objective overwhelms another by more than `100x` without normalization;
- gate targets are all zero or all one;
- correction is globally active.

## Divergence Policy

No KL divergence is used.

No deterministic 7D action vector is treated as a probability distribution.

Distances are Huber/L2 in state space and bounded L2 in action space. If a future version introduces action distributions, it must define support, normalization, estimator, direction, and gradient flow before use.
