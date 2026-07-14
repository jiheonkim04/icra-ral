# EvoState-VLA Researcher Rebuttal

Date: 2026-07-14 KST

Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Reviewer decision addressed: `CONDITIONALLY_ALLOW_ONLY_WITH_STRICT_AUDIT`

## Rebuttal Summary

Reviewer B's objections are accepted. The method will be narrowed and preregistered as a deployment-observable action-evolved state controller, not as a full EvoScene reproduction.

The paper claim, if any, would be:

> A compact action-evolved state prior can improve frozen chunked VLA robustness under controlled execution mismatch while retaining clean behavior.

It will not claim:

- full scene-token modeling;
- object-state recovery;
- superiority to EvoScene in its original setting;
- general VLA world modeling;
- success/failure negative guidance.

## Changes Accepted

### Controlled Mismatch Condition

The primary Stage A/B condition will be declared before rollout as a deterministic execution mismatch applied identically to all five policies.

The default condition is:

```text
translation_lag_scale_fault
```

It attenuates the first three delta-translation action dimensions during execution and introduces a one-step lag in the attenuated component. Rotation and gripper dimensions remain unchanged. Exact scale/lag constants must be frozen in the preregistration before rollout.

This condition is aligned with DREAM-Chunk-style action-noise/stochasticity and Health-conditioned/A2C2-style execution mismatch, but it is locally controllable and cheap.

### Baselines

The first serious comparison will use exactly five policies:

1. `faulted_base_smolvla`
2. `dream_lite_proxy`
3. `evostate_full`
4. `evostate_no_state_prior_ablation`
5. `static_inverse_dynamics`

`static_inverse_dynamics` is the strongest simple killer. If it ties or wins, EvoState is killed.

### DREAM-Lite Proxy

The DREAM-lite proxy will use the same learned transition model but no inverse-dynamics correction. It selects between the base queued action and a fresh-policy action using validation-calibrated predicted mismatch reduction. This is not official DREAM-Chunk and will be labeled as a faithful local proxy only.

### No-State-Prior Ablation

The no-state-prior ablation has the same correction machinery but resets the predicted state to the observed state at every step. Therefore it tests whether persistent action-evolved state, not merely a corrective residual, matters.

### Audit Stops

The audit may stop the method before rollout as:

- `DATA_FAILURE`: insufficient transition pairs, duplicates, collapsed state/action variance, or identity leakage;
- `NO_HEADROOM`: controlled mismatch does not lower Base or static inverse completely solves it in validation;
- `IMPLEMENTATION_FAILURE`: nonfinite predictions, checkpoint reload mismatch, action bound invalidity, or inactive parameters;
- `DESIGN_FAILURE`: action input fails to improve transition prediction over actionless baselines or controllability rank is unusable.

### Mathematical Safety

The inverse correction will use damped least squares:

```text
delta_a = -B^T (B B^T + lambda I)^{-1} e
```

with fixed damping, norm clipping, and a validation-calibrated gate. No KL or action-distribution claim is used.

## Why The Candidate Still Merits Audit

EvoState is not a rescue of FANG:

- FANG used success/failure labels and action-field residuals.
- EvoState uses next-state dynamics and controllability.

EvoState is not a rescue of CAVM:

- CAVM used non-parametric success/failure action memory.
- EvoState uses transition prediction and model-based mismatch correction.

EvoState is not RCV:

- RCV decided when to use fresh/current actions.
- EvoState maintains a predicted state across chunk execution and applies a bounded correction only when the predicted-vs-observed mismatch is controllable.

EvoState is not FEDO:

- FEDO attempted feedback correction and clean behavior collapsed.
- EvoState must beat `static_inverse_dynamics`, preserve clean validation behavior, and default to base passthrough outside reliable mismatch states.

## Rebuttal Decision

Proceed to mathematical audit and preregistration.

Do not run closed-loop Stage A until the development audit and bounded validation search pass without hard stops.
