# Post-Calibration LIBERO-Goal 20260726 Prior Scan Result

Decision: `POST_CALIBRATION_LIBERO_GOAL_IDENTITY20260726_XVLA_PRIOR_SATURATED`

The campaign continued with an official-prior-only X-VLA scan on `libero_goal`, reset identity `20260726`, tasks `0..9`.

Result: 10/10 tasks succeeded, with zero infrastructure failures. No Base gate, headroom gate, second-prior gate, Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout is authorized from this scan.

Key artifacts:

- Run directory: `runs/xvla_prior/failure_scan_libero_goal_identity20260726_post_calibration_20260718T0244KST`
- Summary SHA-256: `831d1c0565f8a12587fd21d60baa92e499083ecca100a159cf24b6ca50b5c23b`
- Manifest SHA-256: `0a52e99985ab9869b7814e938250ceb42a6ce5203de970d9dc110611c3d9ca8f`
- Exit code: `0`
- Heartbeat/finish: `2026-07-18T02:48:07+09:00`
- Windows WSL PID: `16796`; WSL worker PID: `312`
- Model: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source: `C:/assets/repos/X-VLA`, head `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Peak VRAM: `3518.634` MiB
- Simulator episodes: `10`
- Model forwards/action chunks: `40`

Interpretation: `libero_goal` identity `20260726` is exhausted for X-VLA residual mining. Continue official-prior-first search with the remaining supported suites at identity `20260726` or another preregistered supported reset; do not generate a method from this saturated scan.
