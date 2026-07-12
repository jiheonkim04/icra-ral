# PTC-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `PTC-VLA`, Posterior-Transition Conservative VLA

Role: Researcher A

## Hypothesis

Official frozen VLA rollouts expose short-horizon policy-input state transitions that are useful for direct policy generation. A small stochastic head conditioned on current policy-input state, a recent/phase transition latent, and task code can generate actions that are not explained by a phase/task mean-action baseline or a state-only ablation.

## Distinctness

PTC-VLA changes at least two core dimensions relative to Epoch 1:

- representation changes from delay history, action-realization feedback, and image repair to policy-input state transition latents;
- policy generation changes from frozen VLA wrapper/intervention to a direct stochastic action head;
- training signal changes to paired state/action transition supervision from official frozen-policy traces.

It does not use post-hoc action delay adapters, low-level feedback residual correction, image hold-last/edge repair, selector/ranker/verifier routing, barrier/filter/damping, generic progress/value/confidence heads, generic DPO, or simple action reweighting.

## Model

For each trace step:

- `s_t`: policy-input proprioceptive state available to the VLA input processor;
- `dz_t = s_{t+1} - s_t`: observed short-horizon transition;
- `p_t`: episode phase code from normalized step index;
- `c_task`: task code;
- `a_t`: executed frozen-policy action.

Features:

`x_t = concat(s_t, dz_context_t, p_t, c_task)`

Stochastic head:

`mu_t, log_sigma_t = f_theta(x_t)`

Training loss:

`L = ||mu_t - a_t||_2^2 + beta * mean(log_sigma_t^2) + lambda * ||mu_t - mean_action(p_t, c_task)||_2^2`

The conservative term prevents the head from producing unconstrained high-variance actions when transition supervision is sparse.

Inference:

- `dz_context_t` is computed from recent observed policy-input state change plus a phase/task transition prior from training traces;
- action is `clip(mu_t, -1, 1)`;
- no simulator state, reward, success, future observation, object pose, reset identity, or BDDL predicate is used.

## Baselines And Ablation

Required Stage A policies:

1. `frozen_smolvla`: unmodified official backbone.
2. `global_mean_action`: simple global mean-action killer baseline.
3. `phase_mean_action`: phase/task mean-action direct baseline.
4. `ptc_no_transition_ablation`: same architecture with transition latent zeroed.
5. `ptc_full`: full method.

## Stage A Governance

Stage A uses approximately 10 paired episodes per policy over the same task/reset identities.

Stage A may permanently kill only if:

- implementation or data mechanism is invalid;
- `ptc_full` is at least 30 absolute percentage points below the strongest baseline or ablation;
- `ptc_full` has `0 / 10` success while a paired baseline has at least `4 / 10`;
- an oracle or upper bound proves no usable headroom;
- exact trivial equivalence is demonstrated.

Otherwise the method must advance to Stage B.

## Prototype Tasks

Use the same two official SmolVLA/LIBERO tasks for comparability with Epoch 1:

- `libero_spatial/task_4`
- `libero_10/task_4`

Training identities:

- `20260711`
- `20260712`
- `20260713`

Stage A identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

## Expected Outputs

- implementation: `tca_map/smolvla/ptc_vla.py`
- runner: `scripts/run_ptc_vla_prototype.py`
- tests: `tests/test_ptc_vla.py`
- synthetic result: `reports/ptc_vla/synthetic_result.json`
- real trace training result: `reports/ptc_vla/real_trace_train_result.json`
- Stage A result: `reports/ptc_vla/stage_a_result.json`

## Kill Risks

- State-only action prediction may match or beat transition-conditioned generation.
- Phase/task mean action may explain most closed-loop behavior.
- Direct head may be too weak to retain frozen SmolVLA competence.
- Training traces may be too narrow to support a useful transition prior.
