# Post-Task5 Exhaustion LIBERO-Object Identity 20260733 X-VLA Prior Scan

Decision: `POST_TASK5_EXHAUSTED_LIBERO_OBJECT_IDENTITY20260733_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_object` tasks at reset identity `20260733`, after `libero_goal` identity `20260733` saturated. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | true | 143 | 1.0 | 5 |
| 1 | pick up the cream cheese and place it in the basket | true | 125 | 1.0 | 5 |
| 2 | pick up the salad dressing and place it in the basket | true | 116 | 1.0 | 4 |
| 3 | pick up the bbq sauce and place it in the basket | true | 120 | 1.0 | 4 |
| 4 | pick up the ketchup and place it in the basket | true | 145 | 1.0 | 5 |
| 5 | pick up the tomato sauce and place it in the basket | true | 132 | 1.0 | 5 |
| 6 | pick up the butter and place it in the basket | true | 146 | 1.0 | 5 |
| 7 | pick up the milk and place it in the basket | true | 179 | 1.0 | 6 |
| 8 | pick up the chocolate pudding and place it in the basket | true | 146 | 1.0 | 5 |
| 9 | pick up the orange juice and place it in the basket | true | 120 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_object_identity20260733_post_goal20260733_saturated_20260718T1332KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `25236`; WSL worker PID: `312`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `48`
- Summary SHA-256: `1a4004322d02312d551cce11cd7fb1153368c2cb0c774496d1dac5b690b66af2`

Scientific interpretation: this is a clean prior saturation result. It narrows the residual search space but does not authorize candidate generation, Base/headroom/second-prior gating, training, LoRA/QLoRA updates, optimizer steps, checkpoint writes, or Ours rollout.

Next: continue official-prior-first residual search with a full `libero_spatial` identity `20260733` scan. The known `libero_spatial/task5` residual evidence remains closed by candidate exhaustion and must not be reopened as a fresh target.
