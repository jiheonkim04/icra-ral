# Post-Task5 Exhaustion LIBERO-Spatial Identity 20260734 X-VLA Prior Scan

Decision: `POST_TASK5_EXHAUSTED_LIBERO_SPATIAL_IDENTITY20260734_TASK5_PRIOR_FAILURE_CLOSED_FAMILY_NO_FRESH_TARGET`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_spatial` tasks at reset identity `20260734`, after `libero_goal` and `libero_object` identity `20260734` saturated. X-VLA solved 9/10 tasks and failed only task 5, with no infrastructure failures.

This is a fresh reset but not a fresh target family. The `libero_spatial/task5` residual family already passed the repeated-residual screen, generated exactly two candidates, and exhausted that candidate set. This scan adds closed-family evidence; it does not reopen SGL-XVLA, OCR-XVLA, or task5 method generation.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | true | 77 | 1.0 | 3 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | true | 108 | 1.0 | 4 |
| 2 | pick up the black bowl from table center and place it on the plate | true | 94 | 1.0 | 4 |
| 3 | pick up the black bowl on the cookie box and place it on the plate | true | 85 | 1.0 | 3 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | true | 140 | 1.0 | 5 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | false | 900 | 0.0 | 30 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | true | 112 | 1.0 | 4 |
| 7 | pick up the black bowl on the stove and place it on the plate | true | 116 | 1.0 | 4 |
| 8 | pick up the black bowl next to the plate and place it on the plate | true | 96 | 1.0 | 4 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | true | 118 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_spatial_identity20260734_post_goal_object20260734_saturated_20260718T1359KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `3800`; WSL worker PID: `314`
- Simulator episodes: `10`; successful tasks: `9`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `65`
- Summary SHA-256: `570efa633f09d099ff65a3d2a91e71d16f1c60c1a7cd6c7a48f1908279d2277d`

Closure context:

- Repeated residual confirmation: `reports/post_r2p_archive_libero_spatial_task5_residual_confirmation_result.json`
- Candidate generation: `reports/post_r2p_archive_libero_spatial_task5_candidate_generation_result.json`
- SGL-XVLA adjudication: `reports/post_r2p_archive_libero_spatial_task5_sgl_stage0_adjudication_result.json`
- OCR-XVLA observability block: `reports/post_r2p_archive_libero_spatial_task5_ocr_observability_audit_result.json`

Scientific interpretation: this is a clean first-prior task5 failure on a fresh reset, but it belongs to a closed residual family. It does not authorize another task5 candidate, Base/headroom/second-prior rerun, training, LoRA/QLoRA update, optimizer step, checkpoint write, or Ours rollout.

Next: continue official-prior-first residual search elsewhere, starting with `libero_goal` identity `20260735`.
