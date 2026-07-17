# Post-Second-Prior LIBERO-Spatial 20260727 Prior Scan Result

Decision: `POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_XVLA_PRIOR_RESIDUAL_TASK5`

The X-VLA official-prior-only scan on `libero_spatial`, reset identity `20260727`, tasks `0..9`, completed cleanly with zero infrastructure failures.

Result: 9/10 tasks succeeded. `libero_spatial/task_5`, `pick up the black bowl on the ramekin and place it on the plate`, failed cleanly at the same reset: final reward `0.0`, `900` steps, `30` action chunks.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_spatial_identity20260727_post_secondprior_20260718T033637KST`
- Summary SHA-256: `768171a6406a3e15d8c47f3a36a3b20f992721316e234f0cb6d8c5525a242e91`
- Manifest SHA-256: `0380bc27d46d05252be3e2405c8d788a43cea35267d3d7c429f8b3da704150cf`
- Task-5 result SHA-256: `9a6da411db84298748e5a35d23aa5784339f6bc14cdbe24f6842e6a5e6ce40be`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T03:40:41+09:00`
- Windows WSL PID: `25444`; WSL worker PID: `580`

No training, optimizer step, checkpoint write, Ours design, LoRA/QLoRA training, or Ours rollout happened.

Interpretation: this is a clean first-prior residual only. It does not authorize Ours. Next gate is matched SmolVLA Base on `libero_spatial/task_5`, reset identity `20260727`.
