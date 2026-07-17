# Post-Calibration LIBERO-Goal 20260727 Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_GOAL_IDENTITY20260727_XVLA_PRIOR_RESIDUAL_TASK9`

The official-prior-only X-VLA scan on `libero_goal`, reset identity `20260727`, tasks `0..9`, completed cleanly with zero infrastructure failures.

Result: 9/10 tasks succeeded. `libero_goal/task_9`, “put the wine bottle on the rack,” failed cleanly at the same reset: final reward `0.0`, `900` steps, `30` action chunks.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_goal_identity20260727_post_calibration_20260718T0303+09:00ST`
- Summary SHA-256: `6b08b4bec25019854d5914e28d73f43b8f6b54565122016f2a0a110da4ead6ef`
- Manifest SHA-256: `a3a88ee1dc60a365dc16e4b2cd04387991ee960a149b6775f51e2989007a79ad`
- Task-9 result SHA-256: `78a168d5755e025dea13a5e26b5193be5a12b45293c789cfbc57a2203d1433ec`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T03:07:08+09:00`
- Windows WSL PID: `7096`; WSL worker PID: `310`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `66`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: this is a clean first-prior residual only. It does not authorize Ours. Next gate is matched SmolVLA Base on `libero_goal/task_9`, reset identity `20260727`.
