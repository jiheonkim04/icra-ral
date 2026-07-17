# Post-Calibration LIBERO-Object Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_OBJECT_IDENTITY20260725_XVLA_PRIOR_SATURATED`

The campaign continued with an official-prior-only X-VLA scan on `libero_object`, reset identity `20260725`, tasks `0..9`.

Result: 10/10 tasks succeeded, with zero infrastructure failures. No Base gate, headroom gate, second-prior gate, Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout is authorized from this scan.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_object_identity20260725_post_calibration_20260718T0238KST`
- Summary SHA-256: `3640595f3d4549007d7c80e3546c8575ea9d2b8a5af019db227ec7c2bf4609b7`
- Manifest SHA-256: `0a3b39a1b4b6b66139b26bbd7f817c1b7cf7aa5f23b8bb2dcb08d2e64c824e0e`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T02:41:45+09:00`
- Windows WSL PID: `5480`; WSL worker PID: `313`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `48`

Interpretation: the supported `libero_goal`/`libero_object`/`libero_spatial` identity `20260725` prior sweep is saturated. Continue official-prior-first search at another preregistered supported reset; do not generate a method from this saturated scan.
