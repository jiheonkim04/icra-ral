# All Killed Routes Summary

Current killed or reframed main-route candidates:
- Target-Prior TCA-Map
- CSS-Shield
- ExecSpec-Repair
- AMP-GD
- ResetSpec-Retarget
- Phase-Locked Action Chunk Retiming

## Target-Prior TCA-Map

- original hypothesis: target-prior conditioned action decoding can improve counterfactual robustness and wrong-target behavior under low compute.
- strongest positive evidence: fixed-prior TCA improved offline proxy metrics; prior-source audit passed; fixed-prior TCA beat ActionMap in some 7D diagnostics.
- decisive negative evidence: the online 7D action-quality gate failed because the best TCA head did not beat the mean-action baseline.
- exact kill criterion triggered: no credible rollout-level support because the action decoder failed the pre-rollout quality gate.
- strongest trivial baseline that killed it: mean-action baseline.
- why not RA-L-stable: the route relied on offline/diagnostic evidence and the action source was not strong enough for closed-loop claims.
- reusable artifacts: target-prior split/audit code, offline ActionMap/TCA comparisons, 7D bridge, expert replay, online 7D diagnostic heads.

## CSS-Shield

- original hypothesis: a counterfactual semantic/safety shield can reduce wrong-target and unsafe VLA actions at inference time.
- strongest positive evidence: controlled and randomized proposal diagnostics showed semantic wrong-target reductions beyond clipping-only and safety-only.
- decisive negative evidence: native-action Phase 2 diagnostic showed no wrong-target gain beyond safety-only and full intervention rate `1.0`.
- exact kill criterion triggered: full shield failed the native-action semantic novelty gate.
- strongest trivial baseline that killed it: safety-only.
- why not RA-L-stable: native-action evidence supports safety damping against clipping-only, not a semantic contribution beyond safety-only.
- reusable artifacts: WSL simulator path, native SmolVLA CPU inference, shield variants, object/safety metrics, bounded autopilot pattern.

## ExecSpec-Repair

- original hypothesis: mismatch-aware executable-spec repair can recover robot policy execution failures beyond simple action-space baselines.
- strongest positive evidence: STATE 3 full repair recovered `17 / 19` degraded exact-init replay cases.
- decisive negative evidence: STATE 3.5 found diagonal affine calibration also recovered `17 / 19`, with full-minus-best-simple gain `0.0`.
- exact kill criterion triggered: best single simple baseline matched full repair within the predeclared tolerance.
- strongest trivial baseline that killed it: diagonal affine calibration.
- why not RA-L-stable: the broad contribution collapsed to per-dimension affine calibration under current evidence.
- reusable artifacts: executable mismatch diagnostics, exact-init replay, calibration baselines, replay validation, baseline dominance audit.

## AMP-GD

- original hypothesis: active micro-probes can disambiguate intended targets before commitment and reduce wrong-target behavior.
- strongest positive evidence: toy point-world AMP-GD reached wrong-target rate `0.0` and success `1.0`.
- decisive negative evidence: deterministic informative-probe and entropy-greedy heuristics matched toy AMP-GD, and the LIBERO port did not beat safety-only or random-probe on the tiny diagnostic.
- exact kill criterion triggered: simple probe/safety baselines matched or beat the method outside the toy claim.
- strongest trivial baselines that killed it: informative-probe heuristic, safety-only, and random-probe.
- why not RA-L-stable: the method did not show active-probe value beyond simple heuristics in the real simulator port.
- reusable artifacts: object/EEF observability inventory, instruction-to-visible-object resolver, tiny LIBERO micro-probe harness.

## ResetSpec-Retarget

- original hypothesis: object-relative and EEF-relative replay retargeting can recover execution under reset/object-pose mismatch better than raw replay and action-only calibration baselines.
- strongest positive evidence: exact-init expert replay succeeded, default-reset raw replay failed, and object-relative retargeting improved EEF-object progress and shifted-trajectory drift.
- decisive negative evidence: fixed global-scale replay from default reset succeeded with reward/success `1.0 / true`, while object-relative retargeting stayed `0.0 / false`.
- exact kill criterion triggered: object-relative retargeting did not beat the strongest simple baseline.
- strongest trivial baseline that killed it: fixed global scale.
- why not RA-L-stable: the novelty is not separable from a trivial action-scale change on the tested task.
- reusable artifacts: reset-mismatch replay runner, object/EEF state capture, object-shifted trajectory drift metric, baseline-first retarget report.

## Phase-Locked Action Chunk Retiming

- original hypothesis: event-locked timing should recover replay/control success when action chunks are temporally out of phase.
- strongest positive evidence: exact-init expert replay succeeded, and all nine synthetic phase perturbations degraded replay.
- decisive negative evidence: event-locked retiming recovered over raw on `0 / 9` perturbations and beat the best simple baseline on `0 / 9`.
- exact kill criterion triggered: event-locked retiming improved neither replay/progress over raw perturbed replay nor best-simple-baseline performance.
- strongest trivial baselines that exposed the failure: gripper-only timing correction, fixed time shift, repeat-last/hold, linear time warp, and diagonal affine depending on perturbation.
- why not RA-L-stable: the method did not produce a positive replay/control recovery signal, and simple timing/action baselines already recovered or matched several perturbation families.
- reusable artifacts: phase perturbation generator, event-anchor extraction, exact-init phase replay runner, baseline table, and result report.

## Common Failure Pattern

The method must not merely beat no-method. It must beat the strongest trivial baseline available for the failure mode:
- mean-action,
- clipping-only,
- safety-only,
- nearest-target,
- random-probe,
- informative-probe heuristic,
- diagonal affine,
- global scale,
- oracle/replay leakage,
- exact-init expert replay upper bound.

Any future route that only beats no-method, raw replay, or a weak ablation should be killed or reframed before implementation scale-up.
