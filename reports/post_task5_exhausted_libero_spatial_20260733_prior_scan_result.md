# Post-Task5 Exhaustion LIBERO-Spatial Identity 20260733 X-VLA Prior Scan

Decision: `POST_TASK5_EXHAUSTED_LIBERO_SPATIAL_IDENTITY20260733_KNOWN_TASK5_RESIDUAL_CANDIDATE_SET_EXHAUSTED_NO_FRESH_TARGET`

I ran the official X-VLA first-prior diagnostic across all 10 `libero_spatial` tasks at reset identity `20260733`, after `libero_goal` and `libero_object` identity `20260733` saturated. X-VLA solved 9/10 tasks and failed only task 5, with no infrastructure failures.

This is not a fresh target. `libero_spatial/task5` identity `20260733` was already part of the repeated task5 residual confirmation set. Its Base, headroom, and OpenVLA-OFT INT4 second-prior evidence were already recorded, and the capped task5 candidate set was already exhausted: SGL-XVLA was reduced to the fixed-lift/regrasp simple-control threat, and OCR-XVLA was blocked by missing allowed progress traces.

| Task | Instruction | Success | Steps | Final reward | Action chunks |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | true | 80 | 1.0 | 3 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | true | 107 | 1.0 | 4 |
| 2 | pick up the black bowl from table center and place it on the plate | true | 93 | 1.0 | 4 |
| 3 | pick up the black bowl on the cookie box and place it on the plate | true | 83 | 1.0 | 3 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | true | 132 | 1.0 | 5 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | false | 900 | 0.0 | 30 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | true | 98 | 1.0 | 4 |
| 7 | pick up the black bowl on the stove and place it on the plate | true | 118 | 1.0 | 4 |
| 8 | pick up the black bowl next to the plate and place it on the plate | true | 93 | 1.0 | 4 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | true | 117 | 1.0 | 4 |

Execution metadata:

- Run dir: `runs/xvla_prior/failure_scan_libero_spatial_identity20260733_post_goal_object20260733_saturated_20260718T1338KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `12528`; WSL worker PID: `306`
- Simulator episodes: `10`; successful tasks: `9`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Model forward count: `65`
- Summary SHA-256: `de92a3bbe55906ef4648c50a7b73662b59f744498faf563e2184c72663aee797`

Closure evidence:

- Repeated residual confirmation: `reports/post_r2p_archive_libero_spatial_task5_residual_confirmation_result.json`
- Candidate generation: `reports/post_r2p_archive_libero_spatial_task5_candidate_generation_result.json`
- SGL-XVLA adjudication: `reports/post_r2p_archive_libero_spatial_task5_sgl_stage0_adjudication_result.json`
- OCR-XVLA observability block: `reports/post_r2p_archive_libero_spatial_task5_ocr_observability_audit_result.json`

Scientific interpretation: this scan re-confirms a known clean first-prior task5 failure, but it does not authorize another task5 candidate, Base/headroom/second-prior rerun, training, LoRA/QLoRA update, optimizer step, checkpoint write, or Ours rollout.

Next: continue official-prior-first residual search elsewhere, starting with `libero_goal` identity `20260734`.
