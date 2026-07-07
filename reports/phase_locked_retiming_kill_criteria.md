# Phase-Locked Retiming Kill Criteria

Continue only if:
- exact-init expert replay succeeds or provides a usable upper bound,
- at least one phase perturbation degrades replay success, done index, reward, or meaningful progress,
- event-locked retiming recovers reward, success, done index, or trajectory progress,
- event-locked retiming beats fixed shift, gripper-only correction, linear warp, global scale, diagonal affine, and nearest-progress demo on at least one meaningful replay/control metric.

Kill if:
- phase perturbations do not degrade replay,
- fixed shift or gripper-only correction matches event-locked retiming,
- linear time warp or nearest-progress demo matches event-locked retiming,
- event-locked retiming improves only event/offline timing metrics but not replay/progress,
- no real replay/control metric appears.

This route must not continue merely because it beats raw perturbed replay.

STATE 1 outcome: killed. All nine phase perturbations degraded exact-init replay, but event-locked retiming did not improve over raw perturbed replay and did not beat the best simple baseline on any perturbation.
