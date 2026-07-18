# Post-Task5 Exhaustion LIBERO-Goal Identity 20260732 X-VLA Prior Scan

Decision: `POST_TASK5_EXHAUSTED_LIBERO_GOAL_IDENTITY20260732_XVLA_PRIOR_SATURATED_NO_RESIDUAL`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_goal` tasks at reset identity `20260732`, after the `20260731` spatial task4 residual was solved by OpenVLA-OFT INT4. X-VLA solved every task, so this suite/identity pair creates no Ours target and needs no Base, headroom, or second-prior gate.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | open the middle drawer of the cabinet | true | 113 | 1.0 | 4 |
| 1 | put the bowl on the stove | true | 84 | 1.0 | 3 |
| 2 | put the wine bottle on top of the cabinet | true | 85 | 1.0 | 3 |
| 3 | open the top drawer and put the bowl inside | true | 190 | 1.0 | 7 |
| 4 | put the bowl on top of the cabinet | true | 82 | 1.0 | 3 |
| 5 | push the plate to the front of the stove | true | 132 | 1.0 | 5 |
| 6 | put the cream cheese in the bowl | true | 91 | 1.0 | 4 |
| 7 | turn on the stove | true | 77 | 1.0 | 3 |
| 8 | put the bowl on the plate | true | 74 | 1.0 | 3 |
| 9 | put the wine bottle on the rack | true | 146 | 1.0 | 5 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_goal_identity20260732_post_20260731_secondprior_solved_20260718T1304KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `20044`; WSL worker PID: `318`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `40`
- Summary SHA-256: `35740491f4fd8f4ed9c0b49a81a1272700eb4e67be91e2752497beecd2b12f0f`

Scientific interpretation: this is a clean prior saturation result. It narrows the residual search space but does not authorize candidate generation, Base/headroom/second-prior gating, training, LoRA/QLoRA updates, optimizer steps, checkpoint writes, or Ours rollout.

Next: continue official-prior-first residual search at the next unscanned suite/identity.
