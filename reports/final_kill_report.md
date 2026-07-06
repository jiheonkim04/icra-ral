# Final Kill Report

Date: 2026-07-06

## Decision

This project is not RA-L-stable in its current low-compute form.

The current route should not be submitted to RA-L as a main robotics paper. The evidence package is useful, and several engineering components are reusable, but the core robotics-control claim is not established.

## Summary

The project produced valid offline fixed-prior TCA evidence, a validated 7D action bridge, and a validated expert-replay path. It also produced a non-leaking online 7D diagnostic head. The critical failure is that the online 7D action head does not generate actions strong enough for closed-loop support: the best redesigned head still fails to beat a simple mean-action baseline on the required action-quality gate.

This means the current low-compute TCA route is not a stable RA-L robotics-control contribution.

## Key Evidence

- Offline fixed-prior TCA evidence is valid but insufficient.
- The prior-source audit passed and found no inference-time leakage.
- TCA-Select has no meaningful headroom in the current diagnostics and is killed/de-emphasized as a core contribution.
- The representation-collapse claim is unsupported.
- The 7D bridge and expert replay are validated.
- Closed-loop rollout support is not established.
- The online 7D diagnostic head fails to beat the mean-action baseline.
- The rollout gate is red.

## Final Gate Result

Latest bounded 7D action-head redesign gate:

- Best redesigned head: `small_cpu_mlp_fixed_prior_tca_7d`
- Mean-action baseline eval 7D L2: `0.57299313`
- Best redesigned eval 7D L2: `0.669078005`
- Best ActionMap eval 7D L2: `0.992624014`
- Best fixed-prior TCA eval 7D L2: `0.669078005`
- Teacher-forced mean baseline 7D L2: `1.091252901`
- Teacher-forced best non-mean 7D L2: `1.114676933`
- Fixed-prior TCA beats ActionMap on 7D metrics.
- The best head does not beat the mean-action baseline.
- Fixed-prior TCA has no valid rollout-level support.

## Interpretation

Fixed-prior TCA improves over ActionMap in the offline proxy and some bounded action-quality comparisons, but that is not enough. A deployable robotics paper needs a method action source that works in closed loop or at least clears a credible pre-rollout action-quality gate. Here, the online 7D head remains weaker than a train-split mean-action baseline.

This is a hard negative result for the current low-compute RA-L route, not a minor tuning problem.

## Research Integrity Statement

This archive package intentionally does not tune metrics, change splits, hide weak results, revive TCA-Select, or convert offline proxy evidence into rollout claims. The correct conclusion is that the current project should be killed for RA-L-stable submission and either pivoted or preserved as an internal negative result.

