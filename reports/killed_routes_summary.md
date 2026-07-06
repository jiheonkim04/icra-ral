# Killed Routes Summary

## Target-Prior TCA-Map

- original hypothesis: target-prior conditioned action decoding can improve counterfactual robustness and wrong-target behavior under low compute.
- strongest positive evidence: fixed-prior TCA improved offline proxy metrics; prior-source audit passed; fixed-prior TCA beat ActionMap in some 7D diagnostics.
- decisive negative evidence: the online 7D action-quality gate failed because the best TCA head did not beat the mean-action baseline.
- exact kill criterion triggered: no credible rollout-level support because the action decoder failed the pre-rollout quality gate.
- reusable artifacts: target-prior split/audit code, offline ActionMap/TCA comparisons, 7D bridge, expert replay, online 7D diagnostic heads.
- why it should not continue as RA-L-stable: the core action source is not strong enough for closed-loop claims.

## CSS-Shield

- original hypothesis: a counterfactual semantic/safety shield can reduce wrong-target and unsafe VLA actions at inference time.
- strongest positive evidence: controlled and randomized proposal diagnostics showed semantic wrong-target reductions beyond clipping-only and safety-only.
- decisive negative evidence: native-action Phase 2 diagnostic showed no wrong-target gain beyond safety-only and full intervention rate `1.0`.
- exact kill criterion triggered: full shield failed the native-action semantic novelty gate.
- reusable artifacts: WSL simulator path, native SmolVLA CPU inference, shield variants, object/safety metrics, bounded autopilot pattern.
- why it should not continue as RA-L-stable: native-action evidence supports safety damping against clipping-only, not a semantic RA-L contribution.

## Shared Lesson

Future routes must beat simple baselines early. A method that only beats no-method or synthetic proposals should be killed or reframed before scaling.

