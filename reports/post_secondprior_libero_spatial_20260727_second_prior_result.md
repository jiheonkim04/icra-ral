# Post-Second-Prior LIBERO-Spatial 20260727 Second-Prior Result

Decision: `POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_SECOND_PRIOR_CLEAN_FAILURE_TARGET_REMAINS`

Quantized OpenVLA-OFT INT4 was a valid comparable second prior for this condition: the runtime used `libero_spatial_no_noops`, and the local spatial/object/goal/10 checkpoint contains that unnormalization key. No unsupported proxy was used.

Result: OpenVLA-OFT INT4 failed cleanly on `libero_spatial/task_5`, reset identity `20260727`, at the same benchmark initial-state SHA-256 `7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76`.

- Completed episodes: `1`
- Successful episodes: `0`
- Infrastructure failures: `0`
- Reward: `0.0`
- Done: `false`
- Steps: `230` including the `10` wait steps, with max task steps `220`
- Action chunks/model forwards: `28`
- Runtime unnormalization key: `libero_spatial_no_noops`
- Peak CUDA memory: `5598.559` MiB
- Result SHA-256: `ac550a1cf3c779495f645c6a9f9cf10d336d99723ddefdc872b803e19a69b0f1`
- Video SHA-256: `83c8db433af3c9dfeeb030b4dbd062980c9ba8e347221019576efc91ebdbd2fb`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: this condition now has a matched X-VLA failure, matched SmolVLA Base failure, positive task-level headroom, and a clean comparable second-prior failure. It remains a potential target, but candidate generation/training should still wait for a local data/supervision audit.
