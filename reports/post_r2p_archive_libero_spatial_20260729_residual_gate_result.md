# Post-R2P Archive LIBERO-Spatial Identity 20260729 Residual Gate

Decision: `POST_R2P_ARCHIVE_LIBERO_SPATIAL_TASK4_IDENTITY20260729_SECOND_PRIOR_SOLVES_NO_OURS_TARGET`

The `libero_spatial` identity `20260729` X-VLA scan found one failure: task4, “pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate.” Matched SmolVLA Base also failed, and task-level expert headroom was positive, but Quantized OpenVLA-OFT INT4 solved the exact task/reset. This closes the target as not an Ours condition.

| Gate | Result | Meaning |
| --- | --- | --- |
| X-VLA first prior | 9/10 tasks succeeded; task4 failed after 900 steps | One first-prior residual found |
| SmolVLA Base | task4 failed after 280 steps, reward 0 | Shared Base/X-VLA residual |
| Expert replay | selected `demo_2`; expert exact replay succeeded, same-reset demo unavailable | Task-level recoverability exists |
| OpenVLA INT4 second prior | task4 succeeded in 127 steps, reward 1.0 | Residual solved by matched second prior; no Ours target |

Key artifacts:

- X-VLA scan summary: `runs/xvla_prior/failure_scan_libero_spatial_identity20260729_post_r2p_archive_20260718T0612KST/scan_summary.json`
- Base result: `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task4_id20260729_officialenv_20260718T0616KST/result.json`
- Headroom result: `runs/xvla_prior/diagnostic_libero_spatial_task4_expert_headroom_20260729_20260718T0618KST/result.json`
- OpenVLA INT4 result: `runs/openvla_oft_int4/diagnostic_spatial_task4_openvla_int4_20260729_openvlaenv_20260718T0620KST/result.json`
- OpenVLA video SHA-256: `f42248e5948d3a066130aee0c950d5154cac7ebac65be3f50d74a17ba18f0c83`

No training, LoRA/QLoRA update, checkpoint write, candidate generation, or Ours rollout occurred. The next action is to continue official-prior-first residual search elsewhere.

