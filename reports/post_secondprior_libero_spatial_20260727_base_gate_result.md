# Post-Second-Prior LIBERO-Spatial 20260727 Base Gate Result

Decision: `POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_SMOLVLA_BASE_CLEAN_FAILURE_TASK5`

The matched SmolVLA frozen-Base gate for `libero_spatial/task_5`, reset identity `20260727`, completed cleanly and failed the same instruction as the X-VLA first-prior residual: `pick up the black bowl on the ramekin and place it on the plate`.

Result: 0/1 episodes succeeded, with zero infrastructure failures. The episode ran `280` steps, generated `6` action chunks, reached max reward `0.0`, and terminated as `max_steps_or_truncated_without_success`.

Key artifacts:

- Run directory: `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task5_id20260727_officialenv_20260718T034354KST`
- Result SHA-256: `353e3d66bd98696f2a5d64e86f3eb72295b61b18091aba56fdda09da0b3e0941`
- Video SHA-256: `b06bed4febfc09e6891e56e677297683e74f25992d29ac1dfe1282d47aa2ff59`
- Exit code: `0`
- Windows launcher PID: `25388`
- Worker start: `2026-07-18T03:43:55.0251338+09:00`
- Initial state SHA-256: `7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76`
- Peak VRAM: `926.638` MB
- Runner module SHA-256: `3ca3b09f93ca69f8e779a687b6d00389e2642da6e8c2a08739524c34c3e8fabf`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: `libero_spatial/task_5`, reset identity `20260727`, is now a shared X-VLA first-prior and SmolVLA Base clean residual. This still does not authorize Ours. The next gate is expert headroom for the same task/reset.
