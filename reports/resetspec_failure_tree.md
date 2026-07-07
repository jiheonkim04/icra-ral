# ResetSpec-Retarget Failure Tree

Root question: does reset/object-pose mismatch require object-relative executable retargeting?

## Branch A: Replay Bridge

- exact-init expert replay succeeded: yes.
- action convention was viable under matched state: yes.
- conclusion: bridge was good enough for a reset-mismatch diagnostic.

## Branch B: Reset Mismatch

- default-reset raw replay failed: yes.
- exact-init versus default-reset gap existed: yes.
- conclusion: the target failure mode was present.

## Branch C: Object Pose Observability

- target object key resolvable from instruction text plus visible object keys: yes.
- EEF and object poses available: yes.
- target labels, task IDs, filenames, BDDL target fields used at inference: no.
- conclusion: observability was not the blocker.

## Branch D: Object-Relative Retargeting

- EEF-object progress improved: yes.
- shifted trajectory drift improved: yes.
- reward/success recovered: no.
- conclusion: retargeting produced progress but not task completion.

## Branch E: Simple Baselines

- diagonal affine: failed.
- clipping-only: failed.
- fixed global scale: succeeded.
- nearest-demo: skipped because no non-leaking object-pose nearest-demo selector/cache exists.
- conclusion: fixed global scale explains the recoverable part better than object-relative retargeting.

## Failure Classification

`simple_global_scale_baseline_dominates_object_relative_retarget`

The route failed the anti-baseline gate, not the simulator plumbing gate.
