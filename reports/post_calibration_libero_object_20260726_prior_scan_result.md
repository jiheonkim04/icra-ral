# Post-Calibration LIBERO-Object 20260726 Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_OBJECT_IDENTITY20260726_XVLA_PRIOR_SATURATED`

The campaign continued with an official-prior-only X-VLA scan on `libero_object`, reset identity `20260726`, tasks `0..9`.

Result: 10/10 tasks succeeded, with zero infrastructure failures. No Base gate, headroom gate, second-prior gate, Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout is authorized from this scan.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_object_identity20260726_post_calibration_20260718T0250KST`
- Summary SHA-256: `f7c2426b1ae19a8420fed2f5e4dcb7628cdd04157934efbed7b48892473dcf49`
- Manifest SHA-256: `b889bd9ae3002322380d888e2b3a9f35af582e243dfc0d823528cea5df71d09f`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T02:53:31+09:00`
- Windows WSL PID: `26536`; WSL worker PID: `312`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `47`

Interpretation: `libero_object` identity `20260726` is exhausted for X-VLA residual mining. Continue official-prior-first search with `libero_spatial` identity `20260726` or another preregistered supported reset; do not generate a method from this saturated scan.
