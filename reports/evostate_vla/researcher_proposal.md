# EvoState-VLA Researcher Proposal

Date: 2026-07-14 KST

Proposed method: `EvoState-VLA`, Action-Evolved State Guidance for Frozen Chunked VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: EvoScene-VLA, https://arxiv.org/abs/2605.21862.

Secondary prior for comparison: DREAM-Chunk, https://arxiv.org/abs/2606.18589.

## Summary

`EvoState-VLA` tests whether the core positive mechanism from action-updated scene/world-state priors can be reduced to a locally feasible, identity-preserving controller around frozen SmolVLA.

The method does not use terminal success/failure action fields. It does not tune FANG. It does not change SmolVLA weights. It learns from non-confirmatory frozen-policy trace transitions:

```text
current observed state + base action + previous action + chunk phase + task
-> next observed state
```

At inference it maintains an action-evolved predicted state during chunk execution. If the observed state deviates from that predicted state in a validation-calibrated and controllable direction, a small inverse-dynamics correction is added to the base action. Otherwise the base action is emitted exactly.

## Problem

Chunked VLA policies commit several high-rate actions after one low-frequency policy call. If execution proceeds differently than expected because of contact, timing, partial observability, or controlled action-realization mismatch, later chunk actions can continue as if the expected state were true.

Previous local methods showed the boundaries:

- RCV: simply replanning or dropping context was explained by no-context/stateless baselines.
- CAVM: outcome-contrast memory had only a `+1 / 58` margin over the strongest baseline after expansion.
- FANG: learned failure-aware residuals acted but were worse than Base and tied with the no-failure ablation.

The next method must target a different mechanism: action-updated state mismatch, not action memory or failure contrast.

## Prior Anchor

EvoScene-VLA demonstrates a positive action-updated scene-belief mechanism: maintain a scene prior across chunks, update it with actions, and correct it with fresh observations. DREAM-Chunk demonstrates that latent world-model predictions can improve robust chunk execution under stochasticity and action noise.

The local extension is intentionally smaller:

- no scene-token architecture modification;
- no 3D/depth teacher;
- no VLA fine-tuning;
- no object pose at inference;
- use only deployment-observable proprioceptive state and base actions.

## Method

Training data:

- source: `reports/cavm_vla/acquisition_records.jsonl`;
- partitions: discovery/train identities `20260901..20260910`, validation identities `20260911..20260916`;
- forbidden for development: all identities `20260917` and above, including CAVM and FANG confirmatory identities.

For consecutive records from the same `(task_key, identity)` episode, construct transition tuples:

```text
s_t in R^8
a_t in R^7
a_{t-1} in R^7
rho_t in R
task_one_hot in R^2
s_{t+1} in R^8
```

Learn:

1. a transition model `F_theta(x_t) -> Delta s_t`, where `x_t = [s_t, a_t, a_{t-1}, rho_t, task]`;
2. a local controllability map `B_phi(x_t) in R^{8 x 7}` or a calibrated ridge proxy;
3. a reliability gate `g_theta(x_t, e_t)` that predicts when the observed mismatch is both predictable and controllable.

Inference:

1. Start each new chunk with predicted state `s_hat_t = s_t`.
2. For the frozen base action `a_base_t`, predict the next state under the action-evolved model.
3. At the next control step, compute mismatch `e_t = s_t - s_hat_t`.
4. If validation-calibrated reliability is low, use `a_base_t`.
5. If reliable, compute a damped inverse-dynamics correction:

```text
delta_a_t = - B_t^T (B_t B_t^T + lambda I)^{-1} e_t
a_ours_t = clip(a_base_t + alpha * g_t * clip_norm(delta_a_t, delta_max))
```

The correction is bounded, starts at zero, and never uses success labels or object state at inference.

## Expected Behavioral Chain

Problem condition:

```text
chunk execution mismatch
-> actual 8D state diverges from action-evolved expected state
-> queued/base action continues as if expected state were true
-> contact or release timing error compounds
-> task failure
```

Proposed method:

```text
learn action-evolved state prior
-> observe validation-calibrated controllable mismatch
-> add small inverse-dynamics correction
-> reduce execution-state mismatch while preserving base action
-> improved closed-loop robustness with clean retention
```

## Baselines For First Serious Comparison

Exactly five policies:

1. `faulted_base_smolvla`
2. `dream_lite_proxy`
3. `evostate_full`
4. `evostate_no_state_prior_ablation`
5. `static_inverse_dynamics`

The simple killer is `static_inverse_dynamics`: a nonlearned state-action ridge inverse or fixed gain using the same development partition. If it matches or beats EvoState, the claimed action-evolved state mechanism is not useful.

## Audit Before Rollout

Do not launch Stage A until all pass:

- transition tuples exist and are unique;
- no confirmatory/test identity appears in development;
- one-step next-state prediction beats constant, actionless, and per-task linear baselines on validation;
- state dimensions have noncollapsed variance;
- the learned controllability map has usable rank and finite condition numbers;
- mismatch reliability is predictable above a trivial baseline;
- base action passthrough is exact when gate is closed;
- mean action delta and p95 action delta are bounded on validation;
- action bounds remain valid;
- clean validation disruption is below the preregistered cap.

Hard stops are classified as `DATA_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or `DESIGN_FAILURE`, not as closed-loop scientific results.

## Validation Search

Maximum six configurations:

- two transition model choices: ridge-linear and small MLP;
- three correction gains: `alpha in {0.10, 0.25, 0.40}`.

No other hyperparameter grid is allowed. Selection score combines:

- validation next-state prediction;
- controllability reliability;
- bounded action delta;
- clean retention proxy;
- mechanism activation in mismatch states;
- action validity.

The selected configuration is frozen before any confirmatory rollout.

## Confirmatory Identity Plan

If audit and validation pass:

- Stage A identities: `20261101..20261105`, two tasks, five policies, `50` episodes total.
- Stage B identities: `20261106..20261125`, two tasks, five policies, `200` episodes total.
- One expansion to `80` paired cases per policy is allowed only if Stage B is genuinely unresolved by preregistered criteria.

No FANG/CAVM Stage A/B identities may be reused.

## Kill Criteria

Kill the current formulation if:

- audit labels or transition tuples collapse;
- no headroom under the controlled mismatch condition;
- static inverse dynamics matches or beats EvoState;
- the no-state-prior ablation matches EvoState;
- the DREAM-lite proxy dominates EvoState;
- EvoState degrades clean behavior materially;
- action deltas become globally destructive;
- the mechanism does not activate in mismatch states;
- any privileged inference input is required.

Unknown performance before rollout is not a rejection reason.
