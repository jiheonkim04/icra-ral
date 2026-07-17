# Post-Second-Prior LIBERO-Spatial 20260727 Headroom Result

Decision: `TASK5_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE`

The expert-headroom diagnostic for `libero_spatial/task_5`, reset identity `20260727`, completed with exit code `0`.

Result: task-level recoverability is positive, but same-reset HDF5 oracle evidence is unavailable. The residual benchmark initial-state SHA-256 was recomputed as `7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76`, matching the expected residual identity. No HDF5 demo init-state hash matched it.

Selected demo: `demo_9`, nearest by L2 with no hash match; demo init SHA-256 `0d599c208cb9d95b4e724e2a883c651a720276cd8e15e754cf6f3a7527ae497f`; L2 to benchmark residual init `2.984242906`; demo steps `118`.

Replay variants:

- Zero-action exact selected-demo init: failed, reward sum `0.0`.
- HDF5 expert replay exact selected-demo init: succeeded, reward sum `1.0`, first reward/done/success index `93`, after-set-state L2 `0.0`.
- HDF5 expert replay default reset: succeeded, reward sum `1.0`, first reward/done/success index `95`.

Key artifacts:

- Run directory: `runs/xvla_prior/diagnostic_spatial_task5_expert_headroom_20260727_20260718T034720KST`
- Result SHA-256: `42c0b9e287904a7781cf077397c64578a3a5fb7ab651f30f85f810f18eb44fb9`
- Stdout SHA-256: `921f7f9f1c782c09d1afb6108e08d1e4f2cac6cdb35801e24edc685de9299576`
- Stderr SHA-256: `6315d77d31503a034100b4316aefca1f6967c73a55e0fad24867520202129a0c`
- Windows launcher PID: `18256`
- Worker start: `2026-07-18T03:47:20.3412206+09:00`
- Elapsed: `31.437` seconds

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, VLA model load, learned-policy inference, or Ours rollout happened.

Interpretation: this passes the recoverability/headroom gate only at task level. It must not be reported as same-reset expert success. The next required gate is a valid comparable second prior on `libero_spatial/task_5`, reset identity `20260727`, before any candidate generation or training.
