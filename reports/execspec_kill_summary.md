# ExecSpec-Repair Kill Summary

Date: 2026-07-07

Final decision: kill ExecSpec-Repair as the main RA-L-stable route.

This archive is report-only. No experiment, training, rollout, replay, download, GPU job, heavy VLA import, OpenVLA-OFT execution, loss computation, or paper-grade claim happened during this archive step.

## Original Hypothesis

ExecSpec-Repair tested whether VLA and robot policies fail because the executable policy is incomplete without action-space metadata, controller conventions, gripper sign/threshold rules, action normalization, clipping assumptions, and robot-specific execution contracts.

The intended claim was that a mismatch-aware executable-spec repair layer could diagnose and repair these failures better than simple action-space controls such as identity, clipping-only, global affine calibration, gripper-only calibration, or per-dimension diagonal affine calibration.

## Strongest Positive Evidence

The route produced real local evidence that executable mismatch can matter:

- STATE 0-1 reproduced action-space mismatch from local LIBERO HDF5 demonstrations and exact-init replay.
- Correct expert replay reached reward/success `1.0 / true` while plausible wrong executable specs degraded to `0.0 / false`.
- STATE 2 showed full ExecSpec-Repair recovered both replayed degraded mismatch cases.
- STATE 3 broadened exact-init replay to `21` cases, found `19` degraded wrong-spec cases, and full repair recovered `17 / 19`.
- The final full-repair success, reward, and done-index recovery rates were all `0.894736842`.

This evidence is scientifically useful: it validates the local exact-init replay path, shows action conventions can break execution, and leaves reusable mismatch/replay infrastructure.

## Decisive Negative Evidence

STATE 3.5 showed that the broad novelty does not survive the strongest simple baseline:

- degraded replay cases analyzed: `19`
- full ExecSpec-Repair recovery: `17 / 19 = 0.894736842`
- best single simple baseline: `diagonal_affine_calibration`
- best single simple baseline recovery: `17 / 19 = 0.894736842`
- full minus best simple baseline: `0.0`
- selector gain over diagonal affine baseline: `0.0`
- simple baselines explain the result: yes
- repair routing meaningful enough to rescue the broad route: no

The four STATE 3 simple-baseline matched recovery cases were also explained by global affine calibration in living-room global-scale and range-clipping mismatches.

## Exact Kill Criterion Triggered

The triggered criterion is:

> Kill or reframe if the best single simple baseline matches full repair within the predeclared tolerance on replay/control recovery, leaving no nontrivial mismatch-aware gain.

STATE 3.5 triggered it exactly: diagonal affine matched full ExecSpec-Repair on success recovery and action recovery, and the full-minus-best-simple margin was `0.0`.

## Why Diagonal Affine Kills Novelty

Diagonal affine calibration is a simple per-dimension scale/offset repair. It does not need a rich executable-spec ontology, mismatch-specific routing, language grounding, semantic repair logic, or a learned policy. Under the current evidence, it recovers every case that full ExecSpec-Repair recovers.

That matters because the route's publishable claim was not "action-space calibration can be useful"; it was "mismatch-aware executable-spec repair provides nontrivial value beyond simple baselines." A per-dimension affine baseline matching full repair collapses the method novelty into calibration. The mismatch-aware selector also matched full repair, but since diagonal affine alone did the same thing, routing did not add a measurable contribution.

## Reusable Artifacts

Reusable artifacts remain valuable as diagnostics and guardrails:

- exact-init LIBERO/RoboSuite HDF5 replay path,
- correct expert, wrong-spec, identity, clipping, global, gripper, diagonal, and full-repair replay/control comparisons,
- action mismatch diagnostics over HDF5 demonstrations,
- non-leaking calibration/eval split discipline,
- baseline dominance audit code and reports,
- tree-check and safe-runner integration,
- route archive pattern for future topics.

These artifacts should be reused to reject weak future topics early, not to revive the broad ExecSpec-Repair claim.

## Why Not Continue As RA-L-Stable Route

Do not continue this route as the main RA-L topic because:

- the best simple baseline already matches the full method,
- exact-init evidence is narrower than paper-grade rollout evidence,
- the default-reset sanity check did not establish robust task execution,
- the method's routing mechanism has zero measured gain over diagonal affine,
- continuing would likely optimize a killed claim rather than test a new robotics idea,
- a future paper would need a harder predeclared benchmark where diagonal affine is not sufficient.

Honest continuations are limited to tooling, diagnostics, or a future harder executable-spec benchmark with stronger baselines declared before implementation.

