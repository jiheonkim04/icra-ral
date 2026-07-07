# Phase-Locked Retiming Failure Tree

Root question: does temporal phase mismatch require event-locked action-chunk retiming?

## Branch A: Replay Upper Bound

- exact-init expert replay succeeded: yes.
- reward/success/done: `1.0 / true / 260`.
- conclusion: replay bridge was viable.

## Branch B: Phase Perturbation Validity

- phase perturbations degraded replay: `9 / 9`.
- perturbation families included gripper timing, lift timing, full chunk shift, linear time scaling, and chunk-boundary offset.
- conclusion: the diagnostic created real temporal failures.

## Branch C: Event-Locked Retiming

- recovered over raw perturbed replay: `0 / 9`.
- beat best simple baseline: `0 / 9`.
- conclusion: the method did not produce recovery.

## Branch D: Simple Baselines

- gripper-only correction recovered both gripper timing perturbations.
- fixed time shift recovered chunk-shifted-backward.
- linear time warp recovered time-compression.
- repeat-last/hold and diagonal-affine/raw-equivalent baselines matched or weakened Event-Locked Retiming on several remaining cases.
- conclusion: separate obvious simple baselines explain the recoverable sub-failures.

## Failure Classification

`per_failure_mode_simple_baselines_dominate_event_locked_retiming`

The route failed the simple-baseline gate after producing a real replay/control metric.
