# ResetSpec-Retarget Experiment Plan

STATE 1 compares:
- exact-init HDF5 expert replay,
- default-reset raw HDF5 expert replay,
- default-reset diagonal-affine action-only replay,
- default-reset fixed global-scale replay,
- default-reset clipping replay,
- object-relative translation retargeting,
- object-relative translation plus gripper-phase retargeting.

Perturbed-init replay is attempted only if a task-generic safe perturbation helper exists. Nearest-demo replay is attempted only if a non-leaking object-pose nearest-demo selector exists.

Primary metrics:
- reward and success,
- first reward/done/success index,
- EEF-object distance change,
- object movement,
- EEF trajectory drift against the object-shifted demo trajectory,
- translation, rotation, and gripper action error versus raw expert replay,
- gripper timing error,
- controller-valid action rate and clip rate.

Continuation requires exact-init success, default-reset degradation, object-relative improvement, and a win over all simple baselines on success, reward, done index, or meaningful progress.
