# RCV-VLA Mathematical Mechanism Audit

Date: 2026-07-13 KST

Decision: `MECHANISM_AUDIT_DRAFTED_BEFORE_IMPLEMENTATION`

## Variables And Shapes

- observation: `o_t`, including two RGB images with shape `[1, 3, 256, 256]`
- proprioception/state: `q_t in R^8`
- instruction/task text: `l`
- frozen SmolVLA policy: `pi`
- action chunk: `A_t = pi(o_t, q_t, l) in R^{H x 7}` after official postprocessing
- queued action: `a_t^queue in R^7`
- stateless fresh first action: `a_t^fresh in R^7`
- previous action: `a_{t-1} in R^7`
- chunk index fraction: `rho_t = j_t / H in R`
- task one-hot: `u in {0,1}^{2}` for the first prototype
- verifier input: `z_t = [q_t, a_t^queue, a_{t-1}, rho_t, u] in R^{25}`
- verifier output: `h_phi(z_t) in [0,1]`

## Representation Learned

The verifier learns a binary chunk-validity proxy: whether the queued suffix action has drifted away from the frozen policy's current first-action preference.

It does not learn success, reward, object pose, simulator state, or a value function.

## Exact Policy Component Affected

Only the action-chunk execution schedule changes.

The frozen SmolVLA parameters are unchanged. RCV either:

- continues executing the current queued action; or
- resets/replans the frozen policy and executes the newly generated first action.

## Objective

Training label:

`d_t = ||a_t^queue - a_t^fresh||_1 / 7`

`y_t = 1[d_t > tau_train]`

where `tau_train` is computed only from training acquisition rows.

Verifier:

`h_phi(z_t) = sigmoid(w^T z_t + b)` for the first prototype.

Loss:

`L(phi) = - mean_t [ y_t log h_phi(z_t) + (1 - y_t) log(1 - h_phi(z_t)) ] + lambda ||w||_2^2`

## Objective-Term Justification

Binary cross-entropy compares the Bernoulli target `y_t` with the verifier probability `h_phi(z_t)`.

Why appropriate:

- the inference decision is binary: continue or replan;
- the label is a frozen-policy disagreement event, not a continuous action target;
- the first prototype needs calibrated ranking of disagreement likelihood, not action imitation.

Stage used:

- training only.

Parameters receiving gradients:

- verifier weights `phi = (w, b)` only.

Behavior induced:

- high verifier score on steps where the queued suffix is likely stale relative to the current observation-conditioned frozen policy.

Simpler alternative:

- threshold raw queued-vs-fresh disagreement directly by calling the frozen policy every step. This is the `sv_deviation_proxy`.

Required ablation:

- `rcv_no_context_ablation`, which removes current proprioception/action-history context and tests whether the learned verifier adds anything beyond chunk index and action magnitude.

## Inference Algorithm

At planning boundary:

1. generate a frozen SmolVLA action chunk;
2. set chunk index `j = 0`.

At each step:

1. build `z_t`;
2. compute `p_t = h_phi(z_t)`;
3. if `p_t > theta_train`, reset/replan and execute the fresh first action;
4. else execute the queued action and increment the chunk index.

`theta_train` is selected from training acquisition data only.

## Data And Supervision Source

Training acquisition uses official LIBERO closed-loop rollouts with frozen SmolVLA. Supervision comes from comparing two frozen-policy action-generation modes on the same current observation:

- normal queued execution action;
- stateless fresh first action.

No success labels, reward, object poses, simulator state, or future observations are used at inference.

## Gradient Path

Gradients flow only through the lightweight verifier. No gradient flows into SmolVLA, image encoders, action normalizers, environment state, or simulator.

## Expected Behavioral Effect

RCV should reduce execution of stale queued suffix actions while avoiding full heavy-policy inference at every control step.

## Expected Closed-Loop Consequence

If chunk staleness contributes to failure, RCV should improve closed-loop success over normal queued SmolVLA and approach stateless first-action replanning with fewer heavy-policy calls.

## Closest Mathematical Alternative

SV-VLA-style deviation replanning:

`replan if ||a_t^queue - a_t^fresh||_1 / 7 > tau`

This directly uses the fresh action at inference and is therefore a high-cost closest-prior proxy.

## Simplest Equivalent Baseline

Always replan and execute the stateless first action.

## Key Ablation

`rcv_no_context_ablation`: same label and training split, but remove current robot state and previous-action context. If it matches full RCV, the claimed current-state validity mechanism is not useful.

## Known Failure Modes

- queued-vs-fresh disagreement may not correlate with task success;
- a tiny low-dimensional verifier may not predict disagreement from images or object state;
- stateless first-action replanning may dominate all cheap verifier variants;
- SV-VLA proxy may explain all improvement;
- frequent replanning may change gripper timing in harmful ways.
