# Post-Task5 Exhaustion LIBERO-Spatial Task4 20260731 Second-Prior Result

Decision: `POST_TASK5_EXHAUSTED_LIBERO_SPATIAL_IDENTITY20260731_SECOND_PRIOR_SOLVED_NO_OURS_TARGET`

The matched second-prior gate used quantized OpenVLA-OFT INT4 on `libero_spatial/task_4`, reset identity `20260731`. It executed cleanly and solved the residual that X-VLA and SmolVLA Base both failed.

Result: 1/1 episodes succeeded, with zero infrastructure failures. The episode reached reward `1.0` in `128` steps, generated `15` action chunks, and used runtime unnormalization key `libero_spatial_no_noops`.

Key artifacts:

- Run directory: `runs/openvla_oft_int4/diagnostic_spatial_task4_openvla_int4_20260731_openvlaenv_20260718T125850KST`
- Result SHA-256: `22f42f46ac2faf5580ae70be08f0f81240c994c287c576db4fdfeeb5e9449161`
- Video SHA-256: `78d1a24929ec5e9b51e3acbf2812091f339670c98b82c1eb315c992319d7ee1c`
- Exit code: `0`
- Windows launcher PID: `16484`
- Initial-state runtime SHA-256: `ab224edee886b131234e99a0f368026164c95d0cf66b86e2b8e5d57f4752fd0c`
- Peak VRAM: `5600.467` MiB
- Gate script SHA-256: `c2c20652cccc23afcc0e22a5a727712779a5fc9c824f149741159136b1a0c082`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: this residual is solved by the second prior, so `libero_spatial/task_4`, reset identity `20260731`, is not an Ours target. Resume official-prior-first residual search elsewhere.
