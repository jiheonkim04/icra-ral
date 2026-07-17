# Post-Calibration LIBERO-Goal 20260727 Headroom Result

Decision: `TASK9_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`

The expert-headroom diagnostic for `libero_goal/task_9`, reset identity `20260727`, completed with exit code `0`.

Result: task-level recoverability is positive, but same-reset HDF5 oracle evidence is unavailable. The residual benchmark initial-state SHA-256 was recomputed as `73ecfa5d9d3d2323b0641386784a54abbe1ce25a61ded6c7444158bbcccf0714`, matching the expected residual identity. No HDF5 demo init-state hash matched it.

Selected demo:

- Demo: `demo_9`
- Selection reason: `nearest_hdf5_demo_init_state_by_l2_no_hash_match`
- Demo init SHA-256: `d466c20e1d5e589c52c1fa92ec33a6b4df06d55e09f76bff3315d6c296d41bfa`
- L2 to benchmark residual init: `0.30316867`
- Demo steps: `169`

Replay variants:

- Zero-action exact selected-demo init: failed, reward sum `0.0`.
- HDF5 expert replay exact selected-demo init: succeeded, reward sum `1.0`, first reward/done/success index `140`, after-set-state L2 `0.0`.
- HDF5 expert replay default reset: succeeded, reward sum `1.0`, first reward/done/success index `140`.

Key artifacts:

- Run directory: `runs/xvla_prior/diagnostic_libero_goal_task9_expert_headroom_20260727_20260718T032238KST`
- Result SHA-256: `df3bbadf2c886144fa0c274e0c9f8f4761674d05294c0dc77ffae22fd043f399`
- Stdout SHA-256: `d1cb1e8325939e28ee522941cd8d5bb3954eb6ba077445bc31193524ce49b254`
- Stderr SHA-256: `6315d77d31503a034100b4316aefca1f6967c73a55e0fad24867520202129a0c`
- Resume command SHA-256: `bfdf082389892d1d81aefc4f38e156a25e8b4495b4a9b6756dff3420bc41e8e4`
- Windows launcher PID: `20468`
- Worker start: `2026-07-18T03:22:38.3208859+09:00`
- Elapsed: `36.891` seconds

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, VLA model load, learned-policy inference, or Ours rollout happened.

Interpretation: this passes the recoverability/headroom gate only at task level. It must not be reported as same-reset expert success. The next required gate is a valid comparable second prior on `libero_goal/task_9`, reset identity `20260727`, before any candidate generation or training.
