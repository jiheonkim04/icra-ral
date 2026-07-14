# EvoState-VLA Preregistration

Date: 2026-07-14 KST

Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Decision: `EVOSTATE_PREREGISTERED_STAGE_0_PENDING`

## Frozen Method

`EvoState-VLA` learns an action-conditioned transition model and validation-calibrated controllability gate from development-only frozen SmolVLA traces. At inference it maintains an action-evolved predicted state and applies a bounded damped inverse-dynamics correction only when observed-vs-predicted mismatch is reliable and controllable.

Frozen source data for development:

- `reports/cavm_vla/acquisition_records.jsonl`

Development partitions:

- discovery/train identities: `20260901..20260910`
- validation identities: `20260911..20260916`

Forbidden identities for development:

- all identities `>= 20260917`

Confirmatory identities:

- Stage A: `20261101..20261105`
- Stage B: `20261106..20261125`
- optional expansion: `20261126..20261145`

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

## Controlled Mismatch Condition

Primary condition: `translation_lag_scale_fault`.

Execution wrapper applied identically to all five Stage A/B policies:

```text
u_t[0:3] = 0.70 * a_t[0:3] + 0.30 * a_{t-1}[0:3]
u_t[3:7] = a_t[3:7]
```

where `a_t` is the policy action before the fault wrapper and `u_t` is the environment action. The wrapper state resets at episode start.

Clean retention is checked before Stage A through validation action-delta diagnostics and may be measured in additional clean rollouts only after a non-killed Stage B signal.

## Stage 0 Development Audit

Inputs:

- development transition tuples from consecutive records with matching `(task_key, identity)` and consecutive `step`.

Required checks:

1. transition-pair count at least `5000`;
2. each task has at least `1000` transition pairs;
3. duplicate `(task_key, identity, step)` transition keys equal `0`;
4. no forbidden identity appears;
5. every state dimension has validation variance above `1e-6`;
6. every action dimension has validation variance above `1e-6`;
7. transition model validation loss beats the constant predictor and actionless predictor by at least `5%`;
8. action input improvement over actionless is positive on both tasks;
9. controllability effective rank at least `3`;
10. damped inverse condition numbers finite;
11. gate targets are not all zero or all one;
12. closed-gate passthrough max absolute action diff is `0.0`;
13. validation action validity is `1.0`;
14. p95 validation action delta under selected config is at most `0.20` in 7D L2.

Hard stop labels:

- `DATA_FAILURE`
- `NO_HEADROOM`
- `IMPLEMENTATION_FAILURE`
- `DESIGN_FAILURE`

Stage 0 is not a closed-loop scientific result.

## Validation Search

Maximum six configurations:

1. ridge-linear dynamics, `alpha = 0.10`
2. ridge-linear dynamics, `alpha = 0.25`
3. ridge-linear dynamics, `alpha = 0.40`
4. small-MLP dynamics, `alpha = 0.10`
5. small-MLP dynamics, `alpha = 0.25`
6. small-MLP dynamics, `alpha = 0.40`

Fixed values:

- `delta_max = 0.20`
- damped inverse `lambda = 1e-2`
- Huber beta `1.0`
- max training epochs `80`
- learning rate `1e-3`
- random seed `271828`

Validation score:

```text
score = 0.30 * transition_improvement
      + 0.20 * controllability_score
      + 0.20 * bounded_delta_score
      + 0.15 * gate_activation_score
      + 0.15 * action_validity_score
```

where each component is clipped to `[0, 1]`. Select the highest score. If tied, choose the lower `alpha`; if still tied, choose ridge-linear.

After selection:

- save config;
- save checkpoint or fitted coefficients;
- save all tried configs and negative results;
- do not tune on Stage A/B.

## First Serious Closed-Loop Comparison

Exactly five policies:

1. `faulted_base_smolvla`
2. `dream_lite_proxy`
3. `evostate_full`
4. `evostate_no_state_prior_ablation`
5. `static_inverse_dynamics`

Shared manifest:

- same tasks;
- same reset identities;
- same controlled mismatch wrapper;
- same success metric;
- same action bounds;
- same maximum episode steps as the official SmolVLA runner.

Stage A:

- `5` identities x `2` tasks x `5` policies = `50` episodes.
- Permanent kill allowed only for mechanism invalidity, no headroom, catastrophic degradation, clear ablation/prior/simple-baseline dominance, or exact trivial equivalence.
- Small differences advance to Stage B.

Stage B:

- `20` identities x `2` tasks x `5` policies = `200` episodes.
- Compute paired wins/losses/ties, bootstrap CI, per-task breakdown, gate activation, action delta, and action validity.
- One expansion to `80` paired episodes per policy is allowed only if Stage B is genuinely unresolved.

## GO Criteria

EvoState becomes a serious paper candidate only if:

- `evostate_full` beats `faulted_base_smolvla`;
- `evostate_full` beats `dream_lite_proxy`;
- `evostate_full` beats `evostate_no_state_prior_ablation`;
- `evostate_full` beats `static_inverse_dynamics`;
- the mechanism activates in mismatch states but not everywhere;
- clean validation behavior is retained;
- no privileged inference input is used.

Prototype useful improvement target:

- at least `+0.05` task-balanced success over the strongest baseline, or
- saturated-baseline route with paired evidence favoring EvoState and at least `10%` relative failure-rate reduction.

## Kill Criteria

Kill without rescue when:

- any Stage 0 hard stop fires;
- Stage A shows catastrophic degradation or exact trivial equivalence;
- Stage B full is weaker than Base, DREAM-lite, no-state-prior ablation, or static inverse dynamics;
- paired evidence excludes useful improvement after allowed expansion;
- clean retention fails;
- mechanism does not act;
- action validity fails;
- privileged inference input is required.

Confirmatory outcomes may not be used to retune EvoState.
