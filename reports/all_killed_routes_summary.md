# All Killed Routes Summary

Current archived routes:

- Target-Prior TCA-Map
- CSS-Shield
- ExecSpec-Repair

All three are killed or reframed away from the main RA-L route.

## Target-Prior TCA-Map

- original hypothesis: target-prior conditioned action decoding can improve counterfactual robustness and wrong-target behavior under low compute.
- strongest positive evidence: fixed-prior TCA improved offline proxy metrics; prior-source audit passed; fixed-prior TCA beat ActionMap in some 7D diagnostics.
- decisive negative evidence: the online 7D action-quality gate failed because the best TCA head did not beat the mean-action baseline.
- exact kill criterion triggered: no credible rollout-level support because the action decoder failed the pre-rollout quality gate.
- why not RA-L-stable: the route relied on offline/diagnostic evidence and the action source was not strong enough for closed-loop claims.
- reusable artifacts: target-prior split/audit code, offline ActionMap/TCA comparisons, 7D bridge, expert replay, online 7D diagnostic heads.

## CSS-Shield

- original hypothesis: a counterfactual semantic/safety shield can reduce wrong-target and unsafe VLA actions at inference time.
- strongest positive evidence: controlled and randomized proposal diagnostics showed semantic wrong-target reductions beyond clipping-only and safety-only.
- decisive negative evidence: native-action Phase 2 diagnostic showed no wrong-target gain beyond safety-only and full intervention rate `1.0`.
- exact kill criterion triggered: full shield failed the native-action semantic novelty gate.
- why not RA-L-stable: native-action evidence supports safety damping against clipping-only, not a semantic contribution beyond safety-only.
- reusable artifacts: WSL simulator path, native SmolVLA CPU inference, shield variants, object/safety metrics, bounded autopilot pattern.

## ExecSpec-Repair

- original hypothesis: mismatch-aware executable-spec repair can recover robot policy execution failures beyond simple action-space baselines.
- strongest positive evidence: STATE 3 full repair recovered `17 / 19` degraded exact-init replay cases.
- decisive negative evidence: STATE 3.5 found diagonal affine calibration also recovered `17 / 19`, with full-minus-best-simple gain `0.0`.
- exact kill criterion triggered: best single simple baseline matched full repair within the predeclared tolerance.
- why not RA-L-stable: the broad contribution collapses to per-dimension affine calibration under current evidence.
- reusable artifacts: executable mismatch diagnostics, exact-init replay, calibration baselines, replay validation, baseline dominance audit.

## Common Failure Pattern

- The method looked good until compared to a trivial or simple baseline, or until it reached an online rollout/control gate.
- Offline or diagnostic evidence was not enough.
- Simple baselines had to be tested earlier, not after the route gathered momentum.
- The RA-L topic must beat a simple baseline early.
- Rollout-first and baseline-first execution is mandatory.
- Future topics must define kill criteria before implementation, then obey them.

## Current Rule For New Topics

A new route cannot become the main project unless it:

- produces a rollout, replay, or direct control metric within 48 hours,
- beats a strong simple baseline within 72 hours,
- does not rely on offline-only proxy evidence,
- does not rely on native VLA competence unless that competence is verified first,
- survives clipping-only, safety-only, mean-action, and diagonal-affine baselines where applicable.

