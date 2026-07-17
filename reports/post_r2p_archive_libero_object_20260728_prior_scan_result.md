# Post-R2P Archive LIBERO-Object Identity 20260728 X-VLA Prior Scan

Decision: `POST_R2P_ARCHIVE_LIBERO_OBJECT_IDENTITY20260728_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_object` tasks at reset identity `20260728`. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | true | 146 | 1.0 | 5 |
| 1 | pick up the cream cheese and place it in the basket | true | 131 | 1.0 | 5 |
| 2 | pick up the salad dressing and place it in the basket | true | 118 | 1.0 | 4 |
| 3 | pick up the bbq sauce and place it in the basket | true | 144 | 1.0 | 5 |
| 4 | pick up the ketchup and place it in the basket | true | 150 | 1.0 | 5 |
| 5 | pick up the tomato sauce and place it in the basket | true | 120 | 1.0 | 4 |
| 6 | pick up the butter and place it in the basket | true | 149 | 1.0 | 5 |
| 7 | pick up the milk and place it in the basket | true | 127 | 1.0 | 5 |
| 8 | pick up the chocolate pudding and place it in the basket | true | 148 | 1.0 | 5 |
| 9 | pick up the orange juice and place it in the basket | true | 121 | 1.0 | 5 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_object_identity20260728_post_r2p_archive_20260718T0549KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `21000`; WSL worker PID: `314`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Summary SHA-256: `2ea8ca13c880a3327b621d22d8f688b99ea30368cb58a2e70a4b785969d51271`

Scientific interpretation: this is a clean prior saturation result. It is useful negative evidence for residual mining, not a method result. Candidate generation, training, LoRA/QLoRA updates, and Ours rollout remain unauthorized.

