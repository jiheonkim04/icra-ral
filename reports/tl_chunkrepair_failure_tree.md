# TL-ChunkRepair Failure Tree

Root question: does temporal-logic-guided action-chunk repair recover safe robot execution better than simple action baselines?

## Branch A: Replay Infrastructure

- exact-init replay/control metric produced: yes.
- simulator steps: `19803`.
- variants: `73`.
- conclusion: the replay bridge was viable, so the route was not killed for lack of real metrics.

## Branch B: Perturbation Validity

- temporal perturbations tested: `8`.
- perturbations degraded replay: `7 / 8`.
- perturbation families included early release, delayed close, lift before grasp, open-gripper transport, premature release, chunk truncation, phase skip, and inserted unsafe contact action.
- conclusion: the diagnostic created real temporal failures.

## Branch C: Temporal Monitor And Repair

- symbolic temporal violations reduced: `8 / 8`.
- temporal properties were observable enough for the finite-state monitor to fire.
- conclusion: the monitor and repair logic worked as a symbolic constraint mechanism.

## Branch D: Replay/Control Utility

- TL safe-success: `0 / 8`.
- TL reward/success: `0.0 / 0`.
- TL did not improve real reward, success, safe-success, or useful task progress enough to pass the gate.
- conclusion: symbolic repair did not become robot-execution repair.

## Branch E: Simple Baselines

- best single simple baseline: `no_repair`.
- best single simple reward/success: `1.0 / 1`.
- TL beat best single simple baseline: false.
- TL beat best per-failure-mode simple baseline: false.
- conclusion: simple baselines dominated the method on decision-relevant utility.

## Failure Classification

`symbolic_temporal_repair_without_replay_control_utility`

The route failed after producing real replay/control evidence because symbolic safety/property satisfaction did not beat simple baselines on robot execution.
