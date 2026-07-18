# SGL-XVLA Stage 0 Action-Bias Bounds Gate

Decision: `SGL_ACTION_BIAS_BOUNDS_FROZEN_POST_CLAMP_NO_OPTIMIZER_NO_TRAINING`

This is a report-only Stage0 gate. It froze conservative SGL-XVLA action-bias bounds from existing expert action statistics and X-VLA failure/retention metadata only. No model was loaded, no simulator episode ran, no optimizer step occurred, no checkpoint was written, and no Ours rollout happened.

Frozen bounds:

- Activation window: at most the first two X-VLA chunks, 60 assumed steps.
- Enabled bias dimensions: one lift-axis translation component and the gripper component.
- Disabled bias dimensions: lateral translation and roll/pitch/yaw rotation.
- Lift-axis translation bias max abs: `0.20` per step.
- Gripper bias max abs: `0.25` per step.
- Post-bias action clamp: `[-1.0, 1.0]`.

Evidence basis:

- Expert task-level headroom records for identities `20260730` and `20260733` selected `demo_9`; selected-demo action shape was `[118, 7]`, finite, with zero env-adapter clip rate.
- Expert translation max abs was `0.9375`; the frozen lift bound is under 25% of that range.
- Expert gripper max abs was `1.0`; the frozen gripper bound is exactly 25% of that range.
- X-VLA first-two-chunk metadata already reached max abs `1.1772905588150024`, so the gate requires post-bias clamping and saturation guards.

Clean-retention consequence: identities `20260731` and `20260732` remain mandatory because X-VLA solved them and the language-level support gate would activate there too. Any future executable must record pre/post-bias component actions and fail closed if clean-retention added clipping is positive.

Validation: `py_compile` passed and focused pytest passed with `16 passed`.

Next: freeze the simple fixed-lift/regrasp control before any SGL-XVLA method execution. Still no training, checkpoints, or Ours rollout.
