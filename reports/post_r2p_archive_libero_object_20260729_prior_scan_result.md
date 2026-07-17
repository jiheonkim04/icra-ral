# Post-R2P Archive LIBERO-Object Identity 20260729 X-VLA Prior Scan

Decision: `POST_R2P_ARCHIVE_LIBERO_OBJECT_IDENTITY20260729_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_object` tasks at reset identity `20260729`. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | true | 135 | 1.0 | 5 |
| 1 | pick up the cream cheese and place it in the basket | true | 123 | 1.0 | 5 |
| 2 | pick up the salad dressing and place it in the basket | true | 113 | 1.0 | 4 |
| 3 | pick up the bbq sauce and place it in the basket | true | 116 | 1.0 | 4 |
| 4 | pick up the ketchup and place it in the basket | true | 146 | 1.0 | 5 |
| 5 | pick up the tomato sauce and place it in the basket | true | 116 | 1.0 | 4 |
| 6 | pick up the butter and place it in the basket | true | 145 | 1.0 | 5 |
| 7 | pick up the milk and place it in the basket | true | 136 | 1.0 | 5 |
| 8 | pick up the chocolate pudding and place it in the basket | true | 141 | 1.0 | 5 |
| 9 | pick up the orange juice and place it in the basket | true | 116 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_object_identity20260729_post_r2p_archive_20260718T0606KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `4056`; WSL worker PID: `405`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Summary SHA-256: `a761b06b41dd1807d3df2cdacab68d1f3f76a6ad916de698dd36a29cfb4e18c7`

Scientific interpretation: this is another clean prior saturation result. It narrows the residual search space but does not authorize candidate generation, training, LoRA/QLoRA updates, or Ours rollout.

