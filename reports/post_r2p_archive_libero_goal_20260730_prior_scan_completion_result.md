# Post-R2P Archive LIBERO-Goal Identity 20260730 Prior Scan Completion

Decision: `POST_R2P_ARCHIVE_LIBERO_GOAL_IDENTITY20260730_NO_SHARED_RESIDUAL_TASK2_BASE_SOLVED`

I completed `libero_goal` identity `20260730` by scanning tasks `0..2` and `4..9`, while reusing the already-recorded task3 repeated-residual result. X-VLA solved task3 at this identity, and its only new failure was task2. The matched SmolVLA Base gate solved task2 on the same reset, so identity `20260730` contains no shared Base/Prior residual.

| Task | Instruction | X-VLA result | Base gate | Steps / evidence |
| ---: | --- | --- | --- | --- |
| 0 | open the middle drawer of the cabinet | success | not needed | 117 steps |
| 1 | put the bowl on the stove | success | not needed | 88 steps |
| 2 | put the wine bottle on top of the cabinet | failure | Base success | X-VLA 900 steps; Base 86 steps |
| 3 | open the top drawer and put the bowl inside | success | not needed | 179 steps; referenced repeated-residual report |
| 4 | put the bowl on top of the cabinet | success | not needed | 84 steps |
| 5 | push the plate to the front of the stove | success | not needed | 137 steps |
| 6 | put the cream cheese in the bowl | success | not needed | 81 steps |
| 7 | turn on the stove | success | not needed | 77 steps |
| 8 | put the bowl on the plate | success | not needed | 79 steps |
| 9 | put the wine bottle on the rack | success | not needed | 149 steps |

Execution metadata:

- Segment `0..2`: `runs/xvla_prior/failure_scan_libero_goal_identity20260730_tasks0_2_post_r2p_archive_20260718T0626KST`
- Segment `4..9`: `runs/xvla_prior/failure_scan_libero_goal_identity20260730_tasks4_9_post_r2p_archive_20260718T0628KST`
- Base gate for task2: `runs/xvla_prior/diagnostic_smolvla_base_libero_goal_task2_id20260730_officialenv_20260718T0630KST`
- Task3 reference: `reports/post_r2p_archive_libero_goal_task3_20260729_31_xvla_repeated_residual_result.json`
- X-VLA execution type: `VLA_INFERENCE`; evidence role: `FIRST_PRIOR`; artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Base execution type: `VLA_INFERENCE`; evidence role: `BASE`; artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- New X-VLA simulator episodes: `9`; referenced task3 first-prior episode: `1`; Base gate episodes: `1`
- X-VLA model forwards for the full identity including task3: `65`; Base gate model forwards: `2`
- X-VLA summary hashes: `adb71ca257217864499e7f92d6c8d814b5ae335460d2385821e84afe3d9c051b`, `17c1861e745df9ee8b9407f80553475297733a06b9d397e279a7b8855b4b3e8c`
- Base result hash: `b3f122e14aa7a78b28e506278da32385111b6f18117b48f663b06f1f70809740`
- Base video hash: `3d7bce7727cec86e22e4fc7b8fc5c8a4dec0294008d2047052c392ac1050a9ad`

Comparator-role calibration:

| Comparator | Scientific question | Matched result | Uncertainty | Does it block the claim? | Reason |
| --- | --- | --- | --- | --- | --- |
| SmolVLA Base | Does the first-prior failure also fail on the frozen backbone under the matched reset? | Base solved task2 at identity `20260730` in 86 steps, reward 1.0. | Single matched reset diagnostic; enough for this gate because a Base success closes shared-residual status. | Yes, for this identity. | It blocks treating task2 as a shared Base/Prior residual and therefore blocks method development here. |
| X-VLA first prior | Does the closest official prior already solve the task/reset under the local matched protocol? | Solved tasks 0,1,3,4,5,6,7,8,9; failed task2. | Single reset per task diagnostic; residual search only, not a paper-scale performance claim. | No by itself. | The task2 failure is insufficient because Base solved the same reset; task3 is solved by X-VLA at this identity. |
| OpenVLA-OFT INT4 second prior | Would a second official prior also fail after Base and first prior both fail? | Not run. | Not applicable. | No. | The second-prior gate is not authorized when Base already resolves the only X-VLA failure. |
| Ours | Does a proposed method improve the backbone/prior on a repeated residual? | No Ours method exists and no Ours rollout occurred. | Not applicable. | No. | No claim is opened because no repeated shared residual is present. |

Frozen protocol decision: no shared residual for `libero_goal` identity `20260730`.

Calibrated scientific interpretation: same conclusion. Base is not being used as a generic score competitor here; its role is to test whether the first-prior failure is also a backbone failure. Because Base solved task2, the task2 path closes cleanly.

Paper-candidate status: not applicable. No Ours method or paper candidate was evaluated; this report only decides whether a shared residual exists for possible future method development.

Scientific interpretation: this identity does not support method development. No second-prior gate, candidate generation, training, LoRA/QLoRA update, or Ours rollout is authorized.
