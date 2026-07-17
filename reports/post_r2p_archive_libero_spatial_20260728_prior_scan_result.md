# Post-R2P Archive LIBERO-Spatial Identity 20260728 X-VLA Prior Scan

Decision: `POST_R2P_ARCHIVE_LIBERO_SPATIAL_IDENTITY20260728_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_spatial` tasks at reset identity `20260728`. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | true | 74 | 1.0 | 3 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | true | 106 | 1.0 | 4 |
| 2 | pick up the black bowl from table center and place it on the plate | true | 95 | 1.0 | 4 |
| 3 | pick up the black bowl on the cookie box and place it on the plate | true | 87 | 1.0 | 3 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | true | 135 | 1.0 | 5 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | true | 92 | 1.0 | 4 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | true | 107 | 1.0 | 4 |
| 7 | pick up the black bowl on the stove and place it on the plate | true | 117 | 1.0 | 4 |
| 8 | pick up the black bowl next to the plate and place it on the plate | true | 104 | 1.0 | 4 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | true | 118 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_spatial_identity20260728_post_r2p_archive_20260718T0553KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `11236`; WSL worker PID: `307`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Summary SHA-256: `96af89488c452054a950ea09b796c3a5bcd7d6bb6b270d6ea9de3bfbbf2d9823`

Scientific interpretation: this is a clean prior saturation result. It closes this suite/identity pair as a residual source and does not authorize candidate generation, training, LoRA/QLoRA updates, or Ours rollout.

