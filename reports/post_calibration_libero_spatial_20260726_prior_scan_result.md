# Post-Calibration LIBERO-Spatial 20260726 Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_SPATIAL_IDENTITY20260726_XVLA_PRIOR_SATURATED`

The campaign continued with an official-prior-only X-VLA scan on `libero_spatial`, reset identity `20260726`, tasks `0..9`.

Result: 10/10 tasks succeeded, with zero infrastructure failures. No Base gate, headroom gate, second-prior gate, Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout is authorized from this scan.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_spatial_identity20260726_post_calibration_20260718T0255KST`
- Summary SHA-256: `a21cd76d789a04d4a1befc51d7fa78d7d396a84bbb18a89ba57e1b61eaa45979`
- Manifest SHA-256: `b65c6033ff477c3c6eebc0ea8d6e0f4a64121164b41417f1754bb35e62d4d2bb`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T02:59:26+09:00`
- Windows WSL PID: `16780`; WSL worker PID: `304`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `41`

Interpretation: the supported `libero_goal`/`libero_object`/`libero_spatial` identity `20260726` prior sweep is saturated. Continue official-prior-first search at another preregistered supported reset; do not generate a method from this saturated scan.
