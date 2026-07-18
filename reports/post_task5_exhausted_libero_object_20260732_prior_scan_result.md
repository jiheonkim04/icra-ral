# Post-Task5 Exhaustion LIBERO-Object Identity 20260732 X-VLA Prior Scan

Decision: `POST_TASK5_EXHAUSTED_LIBERO_OBJECT_IDENTITY20260732_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_object` tasks at reset identity `20260732`, after `libero_goal` identity `20260732` saturated. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | true | 147 | 1.0 | 5 |
| 1 | pick up the cream cheese and place it in the basket | true | 166 | 1.0 | 6 |
| 2 | pick up the salad dressing and place it in the basket | true | 117 | 1.0 | 4 |
| 3 | pick up the bbq sauce and place it in the basket | true | 130 | 1.0 | 5 |
| 4 | pick up the ketchup and place it in the basket | true | 138 | 1.0 | 5 |
| 5 | pick up the tomato sauce and place it in the basket | true | 134 | 1.0 | 5 |
| 6 | pick up the butter and place it in the basket | true | 146 | 1.0 | 5 |
| 7 | pick up the milk and place it in the basket | true | 138 | 1.0 | 5 |
| 8 | pick up the chocolate pudding and place it in the basket | true | 152 | 1.0 | 6 |
| 9 | pick up the orange juice and place it in the basket | true | 115 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_object_identity20260732_post_goal20260732_saturated_20260718T1311KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `26636`; WSL worker PID: `302`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `50`
- Summary SHA-256: `b65ed5f3fe586858db66c1ea4cef2e0cec50e7d2f20cf4964ac578a3349e19f2`

Scientific interpretation: this is a clean prior saturation result. It narrows the residual search space but does not authorize candidate generation, Base/headroom/second-prior gating, training, LoRA/QLoRA updates, optimizer steps, checkpoint writes, or Ours rollout.

Next: continue official-prior-first residual search at the next unscanned suite/identity.
