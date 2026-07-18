# SGL-XVLA Stage 0 Simple-Control Freeze

Decision: `SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_FROZEN_NO_TRAINING_NO_OURS`

This is a report-only preregistration of exactly one simple-control comparator: `FIXED-LIFT-REGRASP-CONTROL`. It did not train, load a VLA model, run a simulator episode, write a checkpoint, roll out the control, or roll out Ours.

Frozen control:

- Role: strongest simple explanation for any future SGL-XVLA gain.
- Activation: same language-level `ramekin` support condition as SGL-XVLA.
- Template: nonadaptive fixed lift/regrasp bias for chunks `0` and `1`, then zero bias.
- Bounds: lift-axis bias abs `0.20`, gripper-close bias abs `0.25`, lateral/rotation bias `0.0`, post-bias clamp abs `1.0`.
- Axis/sign binding: must come from official LIBERO/X-VLA action-adapter semantics before any rollout; never from outcome tuning.

Comparator calibration: this control blocks the SGL novelty claim only if it explains substantially all future gain at equal or lower cost without clean-retention or generalization loss. It is not a universal beat-everything rule.

Clean-retention identities `20260731` and `20260732` remain mandatory because the support condition would activate there too.

Validation: `py_compile` passed and focused pytest passed with `22 passed`.

Next: freeze the held-out identity/development manifest before any control or Ours rollout. Still no training, checkpoints, or Ours.
