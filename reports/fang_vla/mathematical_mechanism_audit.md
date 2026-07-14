# FANG-VLA Mathematical Mechanism Audit

Date: 2026-07-14 KST

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

This audit is preregistered before FANG training or confirmatory rollout. It documents the exact mathematical object being tested and the diagnostics required before expensive evaluation.

## Variables And Shapes

For each trace row `t`:

- `q_t in R^8`: official proprioceptive state.
- `a_t in R^7`: frozen SmolVLA base action for the current step.
- `a_prev_t in R^7`: previous executed action, zero at episode start.
- `rho_t in R^1`: chunk-index fraction in `[0, 1]`.
- `e_task in R^2`: one-hot task key for the two-task prototype.
- `x_t in R^25`: concatenated feature `[q_t, a_t, a_prev_t, rho_t, e_task]`.
- `y_t in {0, 1}`: terminal episode success label copied to each row after the episode finishes for training only.
- `m_plus(x_t) in R^7`: success action-field head.
- `m_minus(x_t) in R^7`: failure action-field head.
- `s_gate(x_t) in R`: gate logit.
- `G_t = sigmoid(s_gate(x_t) - tau) in [0, 1]`: calibrated gate.
- `c_t in [0, 1]`: discovery-only reliability target from same-task success/failure action-field separation.
- `Delta_t in R^7`: bounded residual applied to the base action.
- `a'_t in R^7`: executed FANG action.

Inference features exclude simulator object poses, rewards, BDDL predicates, terminal success of the current episode, future actions, and held-out identity membership.

## Model

One small MLP trunk:

`h_t = MLP_phi(x_t)`.

Three heads:

`m_plus_t = W_plus h_t + b_plus`.

`m_minus_t = W_minus h_t + b_minus`.

`s_gate_t = W_gate h_t + b_gate`.

Identity-preserving initialization:

- gate bias initialized so `G_t` is near zero;
- therefore `a'_t = a_t` at initialization regardless of untrained action-field outputs.

## Inference Formula

Raw guidance:

`u_t = (m_plus_t - a_t) + beta * (m_plus_t - m_minus_t)`.

Bounded residual:

`Delta_t = alpha * G_t * clip_l2(u_t, delta_max)`.

Action:

`a'_t = clip_action(a_t + Delta_t)`.

Fixed constants before validation:

- `beta = 0.50`;
- `delta_max` defined per configuration/protocol before confirmatory testing;
- `alpha` selected from the bounded validation list only.

Gate calibration:

- `tau` is fit once on validation logits after each candidate model is trained;
- the deterministic target is `50%` validation activation under the rule `G_t > 0.05`;
- this is an identity-preserving integration calibration, not a confirmatory-test tuning knob;
- the selected `tau` is saved in the checkpoint metadata and frozen before any rollout.

## Training Objective

Because local traces contain actions produced by the frozen policy, the heads are trained as class-conditional action-field predictors, not as oracle corrective residual labels.

Let `a_obs_t` be the logged frozen action in the training row.

Let `c_t` be computed from discovery records only:

`c_t = density_t * clip((||mu_plus(x_t) - mu_minus(x_t)||_2 - eta) / gamma, 0, 1)`.

`mu_plus` and `mu_minus` are same-task neighbor action-field means from success and failure discovery records, `eta = 0.05`, and `gamma` is the interquartile range of positive discovery separations clipped below by `0.05`.

For success-labeled rows:

`L_success = mean_{y=1} Huber(m_plus(x_t), stopgrad(a_obs_t))`.

For failure-labeled rows:

`L_failure = mean_{y=0} Huber(m_minus(x_t), stopgrad(a_obs_t))`.

Action-disruption penalty:

`L_delta = mean(||Delta_t||_2^2)`.

Gate sparsity penalty:

`L_gate = mean(G_t)`.

Gate reliability loss:

`L_gate_fit = BCEWithLogits(s_gate(x_t), stopgrad(c_t))`.

Total:

`L = L_success + L_failure + lambda_delta L_delta + lambda_gate_fit L_gate_fit + lambda_gate_sparse L_gate`.

Class imbalance handling:

- compute `L_success` and `L_failure` as class means, not raw-row sums;
- if either class has fewer than the preregistered minimum rows per task, classify `DATA_FAILURE`.

## Scale And Gradient Audit

Before training beyond the smoke batch, record:

- `L_success`, `L_failure`, `L_delta`, `L_gate_fit`, `L_gate`;
- gradient norm for trunk parameters;
- gradient norm for `m_plus` head;
- gradient norm for `m_minus` head;
- gradient norm for gate head;
- ratio of largest objective gradient norm to smallest nonzero objective gradient norm;
- count of NaN or Inf values.

Proceed only if:

- expected trainable parameters receive finite nonzero gradients;
- no objective term is more than `100x` the others without documented normalization;
- gate gradients are finite;
- residual-head outputs are finite;
- validation action deltas remain within the preregistered bound.

## Alternatives And Ablations

Simpler alternative:

- `nearest_success_replay`, non-parametric success-action blending.

Closest external-prior proxy:

- `afil_local_proxy`, same dual heads and guidance but no validation-calibrated identity gate.

Key ablation:

- `fang_no_failure_ablation`, removes `m_minus` from inference and training effect.

Required baseline:

- `base_smolvla`, unmodified frozen policy.

## Why No KL

FANG does not compute KL divergence. The 7D actions and residuals are deterministic vectors in this implementation. No normalized probability distribution, support, density estimator, or KL direction is defined. Vector distances and Huber losses are sufficient for the intended action-field objective.

## Intended Representation And Action Effect

Representation:

- class-conditional local action fields around the frozen policy state/action context.

Intended action effect:

- small residual shift toward success-conditioned action field and away from failure-conditioned action field only when the gate is reliable.

Expected closed-loop consequence:

- fewer repeated local failure patterns on held-out hard tasks without losing clean/base competence.

## Identity-Preserving Integration Audit

Before rollout, report:

- base action sample;
- FANG action sample;
- residual norm;
- gate value;
- dimensions changed;
- translation delta norm;
- rotation delta norm;
- gripper delta;
- action-bound validity;
- fraction of validation rows with nonzero intervention;
- mean and 95th percentile `||Delta_t||_2`;
- clean validation retention proxy.

Hard stop conditions:

- all gates zero after training: `IMPLEMENTATION_FAILURE` or `DATA_FAILURE`;
- gate active almost everywhere with large deltas: `DESIGN_FAILURE`;
- labels collapsed: `DATA_FAILURE`;
- no action-field separation or target variance: `NO_HEADROOM` or `DATA_FAILURE`;
- checkpoint cannot reload: `IMPLEMENTATION_FAILURE`;
- any hidden test identity used in training/validation: invalid result.
