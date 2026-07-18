# Post-R2P Archive LIBERO-Spatial Task5 Candidate Generation

Decision: `LIBERO_SPATIAL_TASK5_TWO_CANDIDATES_GENERATED_NO_TRAINING_NO_OURS_ROLLOUT`

The repeated residual gate authorizes at most two new candidates for `libero_spatial/task5`. I generated exactly two and did not reopen R2P-XVLA or any other archived/closed method.

| Rank | Candidate | Core idea | Status |
| ---: | --- | --- | --- |
| 1 | `SGL-XVLA` — Support-Gated Lift | Detect bowl-on-support context from allowed observations and gate a bounded lift/regrasp action prior. | Spec only; no training |
| 2 | `OCR-XVLA` — Observation-Consistency Retry | Detect no-progress after first grasp/lift attempt and trigger one bounded re-center/regrasp retry. | Spec only; no training |

Still forbidden:

- Training, optimizer steps, checkpoints, and Ours rollout.
- Reopening or retuning R2P-XVLA.
- Reopening R2R-OFT, BR-XVLA, MPR-XVLA, PRC-XVLA, CR-LightVLA, ATCD, MCI-VLA, or CSPR-VLA.
- Any inference input from simulator state, reward, success flags, HDF5 identity, or reset identity labels.

Next recommended action: freeze a no-training Stage 0 specification/gate for `SGL-XVLA`, including support-condition observability, simple fixed-lift control, key ablation, clean-retention identities `20260731/20260732`, and a held-out identity pool before any Ours result.
