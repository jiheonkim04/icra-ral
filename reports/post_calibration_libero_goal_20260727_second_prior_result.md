# Post-Calibration LIBERO-Goal 20260727 Second-Prior Result

Decision: `POST_CALIBRATION_LIBERO_GOAL_IDENTITY20260727_SECOND_PRIOR_SOLVED_NO_OURS_TARGET`

Quantized OpenVLA-OFT INT4 was a valid comparable second prior for this condition: the local spatial/object/goal/10 checkpoint contains `libero_goal_no_noops`, and the runtime used that unnormalization key. No unsupported proxy was used.

Result: OpenVLA-OFT INT4 solved `libero_goal/task_9`, reset identity `20260727`, at the same benchmark initial-state SHA-256 `73ecfa5d9d3d2323b0641386784a54abbe1ce25a61ded6c7444158bbcccf0714`.

- Completed episodes: `1`
- Successful episodes: `1`
- Infrastructure failures: `0`
- Reward: `1.0`
- Done: `true`
- Steps: `139` of max `300`
- Action chunks/model forwards: `17`
- Runtime unnormalization key: `libero_goal_no_noops`
- Peak CUDA memory: `5589.641` MiB
- Result SHA-256: `e900a9105643c2d658d8bddba1a31d8797a35c9c0168511438492ba17e21f4b0`
- Video SHA-256: `b8a67445a5e744c51124f29768eff88b1ae773991169f96bc51f8e614804bf0d`

Support metadata:

- `libero_goal_no_noops`: present
- `libero_spatial_no_noops`: present
- `libero_object_no_noops`: present
- `libero_10_no_noops`: present
- `libero_90_no_noops`: absent

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: the second-prior gate solved the shared X-VLA/SmolVLA residual. This condition is not an Ours target. Continue bounded official-prior residual mining on a non-redundant condition.
