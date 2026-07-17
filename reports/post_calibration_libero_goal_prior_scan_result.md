# Post-Calibration LIBERO-Goal Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_GOAL_IDENTITY20260725_XVLA_PRIOR_SATURATED`

After recording the comparator-role calibration, the campaign continued with an official-prior-only X-VLA scan on `libero_goal`, reset identity `20260725`, tasks `0..9`.

Result: 10/10 tasks succeeded, with zero infrastructure failures. No Base gate, headroom gate, second-prior gate, Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout is authorized from this scan.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_goal_identity20260725_post_calibration_20260718T0232KST`
- Summary SHA-256: `c5054062a8f333d6c7dfda2b5fc77a9c6bcea6d6c2bc06afad5ee84731469979`
- Manifest SHA-256: `604444510fc717736626f5b84e22dcac9237e13b6c3c3429c24161047f0155e8`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T02:36:06+09:00`
- Windows WSL PID: `25388`; WSL worker PID: `313`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `38`

Interpretation: `libero_goal` identity `20260725` is exhausted for X-VLA residual mining. Continue official-prior-first search in another preregistered supported suite/reset; do not generate a method from this saturated scan.
