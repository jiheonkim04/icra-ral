# Post-R2P Archive LIBERO-Spatial Identity 20260730 Residual Gate

Decision: `POST_R2P_ARCHIVE_LIBERO_SPATIAL_TASK5_IDENTITY20260730_SECOND_PRIOR_CLEAN_FAILURE_REPEATED_RESIDUAL_EVIDENCE_NO_OURS_REOPEN`

I scanned `libero_spatial` identity `20260730` with X-VLA. X-VLA solved 9/10 tasks and failed task5, `pick up the black bowl on the ramekin and place it on the plate`. The matched SmolVLA Base gate also failed task5. Expert replay provided task-level recoverability evidence, with same-reset expert headroom unavailable. Quantized OpenVLA-OFT INT4 then failed the same task/reset cleanly.

| Gate | Result | Evidence |
| --- | --- | --- |
| X-VLA first prior | task5 failure | 900 steps, reward 0.0, 30 action chunks |
| SmolVLA Base | task5 failure | 280 steps, reward 0.0, 6 action chunks |
| Expert headroom | task-level positive; same-reset unavailable | exact replay success at index 93; zero-action failed |
| OpenVLA-OFT INT4 second prior | task5 failure | 230 steps, reward 0.0, 28 action chunks; no offload |

Execution metadata:

- X-VLA run: `runs/xvla_prior/failure_scan_libero_spatial_identity20260730_post_r2p_archive_20260718T0645KST`
- Base gate: `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task5_id20260730_officialenv_20260718T1103KST`
- Expert headroom: `runs/xvla_prior/diagnostic_libero_spatial_task5_expert_headroom_20260730_20260718T1105KST`
- OpenVLA INT4 gate: `runs/openvla_oft_int4/diagnostic_spatial_task5_openvla_int4_20260730_openvlaenv_20260718T1107KST`
- X-VLA summary hash: `c3a64e1f84c485b0cd0c489e96579f9bde04a2857bb87492ba3cdca166cbcb7a`
- Base result hash: `f4d17ee30c482265eabd349c9e30b80761959788864e5cd588ceb1049ad39d03`
- Headroom result hash: `1c102be6dad1a7300ae7ea76ee4fced3f0f8e2a93aaea129b951d18ced653e89`
- OpenVLA result hash: `08e0bf649f83bbbf60f09244f6ec307f090b0f6eeaa056bc3aa306cc1a4afe64`

Comparator-role calibration:

| Comparator | Scientific question | Matched result | Uncertainty | Does it block the claim? | Reason |
| --- | --- | --- | --- | --- | --- |
| SmolVLA Base | Does the frozen backbone also fail the first-prior failure? | Failed task5 at identity `20260730`. | Single matched reset diagnostic. | No. | Supports shared-residual status. |
| X-VLA first prior | Does the closest official first prior solve the task/reset? | Failed task5 at identity `20260730`. | Single reset diagnostic. | No. | Keeps downstream gates open. |
| OpenVLA-OFT INT4 | Does a second official prior already solve the residual? | Failed task5 at identity `20260730`. | Single matched reset diagnostic. | No. | Leaves residual evidence open. |
| R2P-XVLA archived Ours | May the previous task5 method be reopened? | No; archived at frozen offline selection. | Not applicable. | Yes, for R2P. | The frozen archive remains binding. |

Frozen protocol decision: task5 identity `20260730` is a clean Base/X-VLA/OpenVLA-INT4 failure with task-level expert headroom positive and same-reset expert headroom unavailable. This does not reopen the archived R2P-XVLA method.

Calibrated scientific interpretation: this result supports bounded residual confirmation, not a paper-candidate claim. There are now two known independent task5 shared-failure identities, `20260727` and `20260730`; the campaign safeguard still targets at least three clean shared failures when locally available before new method generation.

No candidate generation, training, LoRA/QLoRA update, or Ours rollout is authorized. Next authorized action: bounded residual-confirmation screening for additional independent `libero_spatial/task5` identities.
