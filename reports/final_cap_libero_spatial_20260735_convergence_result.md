# Final Capped LIBERO-Spatial Identity 20260735 X-VLA Prior Scan

Decision: `NATURAL_RESET_SEARCH_SATURATED`

I ran the preregistered final natural-reset scan: official X-VLA first-prior inference across all 10 `libero_spatial` tasks at reset identity `20260735`. X-VLA solved every task, including task 5. This consumes the final natural-reset budget from `reports/residual_search_convergence_cap_20260735_result.json`.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | true | 73 | 1.0 | 3 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | true | 113 | 1.0 | 4 |
| 2 | pick up the black bowl from table center and place it on the plate | true | 102 | 1.0 | 4 |
| 3 | pick up the black bowl on the cookie box and place it on the plate | true | 83 | 1.0 | 3 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | true | 119 | 1.0 | 4 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | true | 90 | 1.0 | 3 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | true | 104 | 1.0 | 4 |
| 7 | pick up the black bowl on the stove and place it on the plate | true | 114 | 1.0 | 4 |
| 8 | pick up the black bowl next to the plate and place it on the plate | true | 100 | 1.0 | 4 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | true | 123 | 1.0 | 5 |

Execution metadata:

- Run dir: `runs/xvla_prior/final_cap_libero_spatial_identity20260735_20260718T1424KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `15184`; WSL worker PID: `309`
- Simulator episodes: `10`; successful tasks: `10`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `38`
- Summary SHA-256: `83206cdc3194eef6103d588f20a369d5a96099782c7583c57401ae9e9085355e`

Convergence decision:

`NATURAL_RESET_SEARCH_SATURATED`

Reason: the final capped sweep saturated. No eligible repeated cross-prior residual survived within the frozen budget. Prior `libero_spatial/task5` failures remain closed-family evidence because the task5 candidate set was already exhausted; they are not fresh Ours targets.

Natural-reset residual mining is now closed. Do not launch another natural-reset identity sweep.

Next: decide whether OCR-XVLA qualifies for exactly one bounded no-training trace-acquisition pass. If it does not, archive it as `OBSERVABILITY_DATA_BLOCKED_ARCHIVED`.
