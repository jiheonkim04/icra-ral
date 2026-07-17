# Post-Calibration LIBERO-Goal 20260727 Base Gate Result

Decision: `POST_CALIBRATION_LIBERO_GOAL_IDENTITY20260727_SMOLVLA_BASE_CLEAN_FAILURE_TASK9`

The matched SmolVLA frozen-Base gate for `libero_goal/task_9`, reset identity `20260727`, completed cleanly. It failed the same instruction as the X-VLA first-prior residual: `put the wine bottle on the rack`.

Result: 0/1 episodes succeeded, with zero infrastructure failures. The episode ran `300` steps, generated `6` action chunks, reached max reward `0.0`, and terminated as `max_steps_or_truncated_without_success`.

Key artifacts:

- Run directory: `runs/xvla_prior/diagnostic_smolvla_base_libero_goal_task9_id20260727_officialenv_20260718T0312KST`
- Result SHA-256: `107d7cb0ff1b31ba98be98b9230fdc859218172a0d1c260f2f88751b8a2d0d5f`
- Resume command SHA-256: `7d3c8ec92aa908248babdee50290b7db4585398bcf9e0f5f6e940e1970bba341`
- Exit code: `0`
- Windows WSL PID: `25544`
- Started: `2026-07-18T03:12:08.0052655+09:00`
- Initial state SHA-256: `73ecfa5d9d3d2323b0641386784a54abbe1ce25a61ded6c7444158bbcccf0714`
- Manifest canonical payload SHA-256: `99b70045101efef24540088f220711da8662267c9dd662223b2c9e52c1d2da7d`
- Video SHA-256: `2a513ee00761b9e39b9fca429be1625f0decb52a92870681da0928498d41b914`
- Peak VRAM: `926.638` MB
- Runner module SHA-256: `3ca3b09f93ca69f8e779a687b6d00389e2642da6e8c2a08739524c34c3e8fabf`

Execution classification:

- Type: `VLA_INFERENCE`
- Evidence role: `BASE`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Policy: `frozen_base` SmolVLA through LeRobot policy/processors/postprocessors and LIBERO env wrapper
- PEFT used: `false`
- Old custom LIBERO 7D route used: `false`
- Paligemma import stub used: `false`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: `libero_goal/task_9`, reset identity `20260727`, is now a shared X-VLA first-prior and SmolVLA Base clean residual. This still does not authorize Ours. The next gate is expert headroom for the same task/reset; if recoverability is positive, a valid comparable second prior must be checked before any candidate generation or training.
