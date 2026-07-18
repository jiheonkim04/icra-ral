# Post-Task5 Exhaustion LIBERO-Spatial Task4 20260731 Base Gate Result

Decision: `POST_TASK5_EXHAUSTED_LIBERO_SPATIAL_IDENTITY20260731_SMOLVLA_BASE_CLEAN_FAILURE_TASK4`

The matched SmolVLA frozen-Base gate for `libero_spatial/task_4`, reset identity `20260731`, completed cleanly and failed the same instruction as the X-VLA first-prior residual: `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate`.

Result: 0/1 episodes succeeded, with zero infrastructure failures. The episode ran `280` steps, generated `6` action chunks, reached max reward `0.0`, and terminated as `max_steps_or_truncated_without_success`.

Key artifacts:

- Run directory: `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task4_id20260731_officialenv_20260718T125036KST`
- Result SHA-256: `138310144baa433e3b1ea45acbb0a3ba709afcb8d54afcca07f3ee752d906a36`
- Video SHA-256: `3039468a6d4520637ed0daaff71d5b837ef83f985942b40f6bb53e4d4c885c93`
- Exit code: `0`
- Windows launcher PID: `26780`
- Worker start: `2026-07-18T12:50:36.2907876+09:00`
- Initial state SHA-256: `ab224edee886b131234e99a0f368026164c95d0cf66b86e2b8e5d57f4752fd0c`
- Peak VRAM: `926.638` MB
- Runner module SHA-256: `3ca3b09f93ca69f8e779a687b6d00389e2642da6e8c2a08739524c34c3e8fabf`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: `libero_spatial/task_4`, reset identity `20260731`, is now a shared X-VLA first-prior and SmolVLA Base clean residual. This still does not authorize Ours. The next gate is expert headroom for the same task/reset.
