# ECHO-VLA Candidate Headroom Result

- ran: `True`
- passed: `False`
- group count: `4`
- candidate records: `16`
- tasks: `libero_spatial/task_0`, `libero_object/task_4`
- reset identities: `20260711`, `20260712`
- candidate count per group: `4`
- fixed horizon: `4`
- default success rate: `0.0`
- oracle success rate: `0.0`
- oracle improvement pp: `0.0`
- default failure recoverable rate: `0.0`
- hard kill reason: `oracle improvement <10pp or fewer than 15% of default-failure states contain a successful/materially better candidate`
- same-state group proofs valid: `4/4`
- non-gripper effect labels populated: `eef_delta_norm`, `target_distance_delta`

The oracle is diagnostic only. It uses realized effects after executing all same-state candidates and is not available at deployment.

## Interpretation

This is a valid headroom kill for the bounded first gate, not an ECHO training result. The frozen SmolVLA default chunk and bounded perturbation candidates produced only small end-effector/target-distance changes over the 4-step initial chunks, with no object displacement, retention, contact transition, task success, or material phase-compatible recovery. Because the oracle over realized effects has no meaningful candidate to select, training ECHO would test a selector without recoverable candidate headroom.
