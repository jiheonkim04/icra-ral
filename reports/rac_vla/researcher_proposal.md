# RAC-VLA Researcher A Proposal

Date: 2026-07-14 KST

Method: `RAC-VLA`, Reflective Action-Consequence Calibration for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Reflective VLA, https://arxiv.org/abs/2606.25215.

## Claim

Frozen reactive VLAs can fail under deployment-specific action-channel calibration or actuation shift because the current observation alone does not reveal how commanded actions map to realized state changes. A compact action-consequence history can infer that hidden calibration context online and apply a bounded identity-preserving action calibration that improves shifted-condition closed-loop success while retaining clean behavior.

## Positive External Prior

Reflective VLA reports that observation-action-consequence triplets improve cross-environment generalization. Its key evidence is that action consequences, not only extra history length, help a VLA adapt to hidden deployment factors such as camera-to-robot geometry, calibration, or systematic actuation bias.

RAC-VLA does not claim to reproduce Reflective VLA officially. It tests a local frozen-policy extension of the same mechanism:

- Reflective VLA: train an in-context multimodal VLA architecture.
- RAC-VLA: learn a compact deployment-observable action-consequence calibration context and wrap a frozen SmolVLA action with a zero-initialized bounded calibration residual.

## Problem Condition

The primary condition is a controlled action-channel deployment shift. The first implementation may use small deterministic translation/rotation/gripper calibration transforms applied to the executed action in the simulator. These transforms are predeclared before closed-loop testing and are not tuned on confirmatory outcomes.

The clean condition is the same task/reset manifest without the action-channel shift.

## Mechanism Hypothesis

Observed failure chain:

1. deployment action-channel calibration differs from the training environment;
2. current observation alone does not identify the hidden calibration;
3. the frozen reactive VLA emits an action in the nominal action space;
4. realized state changes differ systematically from intended state changes;
5. repeated mismatches compound into closed-loop failure.

Proposed method chain:

1. maintain a short history of deployment-observable `(state, action, delta_state)` triplets;
2. encode the history into a calibration context that predicts stable action-effect mismatch;
3. gate a small residual calibration only when the context is stable and validation-calibrated;
4. emit the base action exactly when the gate is closed;
5. reduce shifted-condition failures while preserving clean behavior.

## Variables

- `s_t in R^8`: deployment-observable robot state at step `t`.
- `a_t in R^7`: frozen SmolVLA action before calibration.
- `d_t = s_{t+1} - s_t in R^8`: realized state consequence.
- `h_t = {(s_i, a_i, d_i)}_{i=t-H}^{t-1}`: recent consequence history.
- `z_t in R^m`: learned calibration context.
- `g_t in [0, 1]`: calibration gate.
- `r_t in R^7`: bounded residual calibration.
- `a'_t = clip(a_t + g_t r_t)`: executed action before any predeclared deployment-shift transform in the simulator.

No object pose, reward, terminal success, future state, future action, or reset identity is available at inference.

## Model Sketch

Stage 0 and validation use only discovery/validation identities from development traces.

History encoder:

`z_t = f_theta(mean_i psi(s_i, a_i, d_i, task_i, rho_i))`

where `rho_i` is chunk phase and `psi` is a small MLP or fixed feature map. The default implementation may begin with a linear/ridge or small MLP classifier to avoid unnecessary complexity.

Calibration residual:

`r_t = alpha * tanh(W_r [z_t, s_t, a_t, rho_t, task_t])`

Gate:

`g_t = 1[p_theta(z_t) >= tau and stability(h_t) >= eta]`

Initial behavior:

- `alpha = 0` or `W_r = 0` at initialization;
- `tau` initialized so the gate is closed;
- selected validation config must prove nonzero but bounded action changes.

## Training And Development Objective

Stage 0 uses controlled synthetic action-channel labels derived from development traces:

- define hidden transforms `S_k` that map a command into the action actually applied by the environment;
- create synthetic command features `c = S_k^{-1}(a)` so that the hidden transform would produce the trace action `a`;
- pair the synthetic command with the observed consequence;
- train the context model to identify `k` or recover an inverse calibration vector from `(state, transformed_action, observed_delta_state)` histories.

Primary development loss:

`L_cal = CE(q_theta(h_t), k)` for perturbation classification, or Huber regression to the inverse calibration vector when regression is used.

Clean retention loss:

`L_ret = ||a'_t - a_t||_Huber` on clean validation histories.

Action bound penalty:

`L_bound = max(0, ||a'_t - a_t||_2 - delta_max)^2`.

The first implementation should prefer classification plus deterministic inverse templates because it is auditable and avoids decorative probability assumptions. No KL divergence is used.

## Data Partitions

Discovery:

- inspect development traces;
- define synthetic perturbation label set;
- debug feature construction.

Validation:

- train identities `20260901..20260910`;
- validation identities `20260911..20260916`;
- select one config from at most six total configurations.

Confirmatory test:

- identities must be frozen after Stage 0 and validation;
- no identity `>= 20260917` may be used before freezing the final Stage A/B protocol.

## Stage 0 Hard Stops

Stop before rollout if any occurs:

- fewer than `5000` usable consequence-history examples;
- any duplicate `(task, identity, step, perturbation)` key;
- perturbation labels collapsed below `10%` or above `90%` for any binary target used;
- full consequence model fails to beat a history-only/no-consequence baseline by at least `5%` relative validation error or `5` accuracy points;
- actionless or no-consequence features explain the target equally well;
- gate positive fraction is below `2%` or above `98%`;
- clean p95 action delta exceeds `0.20`;
- action validity is below `1.0`;
- any privileged inference field is required.

Classify failures as `DATA_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`, not as closed-loop scientific results.

## Validation Search

Maximum six configurations:

- history horizon `H in {2, 4}`;
- residual coefficient `alpha in {0.05, 0.10, 0.20}`.

Select one configuration by:

`score = validation_shift_proxy + clean_retention + mechanism_activation + action_validity - compute_penalty`

where the exact normalized terms are written before running the search.

## First Paper Comparison

Five policies:

1. `base_smolvla_shifted`
2. `reflective_history_proxy`
3. `rac_full`
4. `rac_no_consequence_ablation`
5. `online_diagonal_inverse_gain`

The Reflective proxy is a local transparent proxy, not an official reproduction. It uses the same consequence history to choose among predeclared inverse templates but does not learn the RAC residual adapter.

The simple killer is an online diagonal inverse-gain or affine calibration baseline estimated from recent state/action deltas without a learned consequence context.

## GO And Kill Criteria

Stage A uses approximately `10` paired episodes per policy and may only kill for mechanism invalidity, no headroom, catastrophic harm, exact trivial equivalence, or clear baseline/ablation dominance under current governance.

Stage B uses at least `40` paired episodes per key policy. RAC reaches prototype GO only if:

- `rac_full` beats Base, Reflective proxy, no-consequence ablation, and online inverse-gain;
- absolute gain is at least `10` task-balanced points at prototype scale, or paired evidence is consistently positive with meaningful failure-rate reduction;
- mechanism activation is nonzero and bounded;
- no privileged inference signal is used;
- clean behavior is retained.

Permanent kill if Stage B is complete and a baseline or ablation explains the result, or useful improvement is excluded by the preregistered paired confidence interval.

## Expected Failure Modes

- The synthetic perturbation label is predictable for trivial reasons that do not transfer to closed-loop shift.
- The online inverse-gain killer matches RAC, meaning the learned consequence context adds no value.
- Clean behavior drops because the gate activates too often.
- The action-channel shift is too easy or too hard, creating no meaningful headroom.
- The low-dimensional LIBERO state omits the visual/physical consequence information that Reflective VLA uses.
