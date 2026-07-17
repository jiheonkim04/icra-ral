# Post-R2P Archive Second Prior: LIBERO Goal Task 3 Identity 20260728

Decision: `POST_R2P_ARCHIVE_LIBERO_GOAL_IDENTITY20260728_SECOND_PRIOR_CLEAN_FAILURE_TARGET_REMAINS_DIAGNOSTIC_ONLY`

Quantized OpenVLA-OFT INT4 executed as the comparable second prior for `libero_goal/task_3`, reset identity `20260728`. It failed cleanly: one completed episode, zero successes, zero infrastructure failures, final reward `0.0`.

Runtime root: `runs/openvla_oft_int4/diagnostic_goal_task3_openvla_int4_20260728_openvlaenv_repaired_20260718T0532KST`

Key facts:

- Runtime unnormalization key: `libero_goal_no_noops`
- Dataset statistics support: `libero_goal_no_noops` present; unsupported `libero_90_no_noops` not used
- Initial-state SHA matched manifest: `8e711166d5f2d13c564cb0e1b5ae46c260e5b5c8eb220ff23bd5106b99f8728e`
- Steps: `310`; action chunks/model calls: `38`
- Peak CUDA: `5591.251 MiB`
- Video: `rollouts/2026_07_18/2026_07_18-05_31_02--openvla_oft--episode=110000--success=False--task=open_the_top_drawer_and_put_the_bowl_inside.mp4`

No Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout happened.

Interpretation: task3 now has a single identity where X-VLA, SmolVLA Base, and Quantized OpenVLA-OFT INT4 all failed, with task-level expert headroom positive. Candidate generation is still not authorized because repeated-residual safeguards are not yet satisfied.

Next gate: run preregistered repeated-residual confirmation for `libero_goal/task_3` on additional independent reset identities.
