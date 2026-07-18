# Action-Consistent Missing-View Numerical Threshold Freeze

Decision: `STAGE0_NUMERICAL_NOISE_AND_PRACTICAL_THRESHOLDS_FROZEN`

The unchanged frozen calibration completed 12 rows × 3 repeats with 36 real
clean-teacher and 36 real dropout-student forwards, zero optimizer steps, zero
exceptions, and no repeat gripper flips. The official image mask remained
`[true,true,false]` while wrist pixels changed under dropout.

## Frozen normalization denominators

| component | denominator |
|---|---:|
| action hidden MSE | 0.002987779696316769 |
| translation MSE | 0.0015878713347774465 |
| rotation MSE | 0.0018577529408503324 |
| raw gripper-margin MSE | 56.628908475240074 |
| wrist-reconstruction MSE | 0.9934094299872717 |

All four measured repeat-noise values were exactly zero. The practical
absolute minima therefore remain the prespecified floors:

- action-hidden MSE: `1e-5`
- translation RMSE: `1e-4`
- rotation RMSE: `2e-4`
- raw gripper-margin MAE: `0.002`

Full must improve at least one co-primary measure by both 5% relative and its
absolute minimum. The frozen other-measure nonregression, discrete-gripper,
reconstruction, full-versus-no-reconstruction, and full-versus-generic rules
remain unchanged.

The clean-teacher adjacent-step p99 envelopes are `0.011365951299667342` for
translation and `5.268177709579409` for rotation. Peak allocated/reserved VRAM
was `3,698,516,992 / 3,867,148,288` bytes; peak observed system-RAM fraction
was `0.17559117374675567`; swap growth was zero.

No validation or confirmatory outcome, optimizer step, checkpoint, offload, or
physical manipulation occurred. The next authorized stage is the frozen
actual-path microbatch preflight over `1,2,4,8`.
