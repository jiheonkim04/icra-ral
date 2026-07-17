# Post-R2P Archive LIBERO-Goal Task3 Identity 20260729 Base Gate

Decision: `TASK3_IDENTITY20260729_BASE_SOLVED_NOT_SHARED_RESIDUAL_REPEATED_RESIDUAL_NOT_CONFIRMED`

The repeated X-VLA screen found one new first-prior failure for `libero_goal/task_3` at reset identity `20260729`. I ran the matched SmolVLA Base gate on the same task and reset identity. Base succeeded cleanly, so this identity is not a shared Base/Prior residual.

| Comparator | Scientific question | Matched result | Uncertainty | Does it block candidate generation? | Reason |
| --- | --- | --- | --- | --- | --- |
| X-VLA first prior | Does the official prior solve this reset? | Failure, 900 steps, final reward 0.0 | Single deterministic reset diagnostic | No by itself | Prior failure alone is not an Ours target |
| SmolVLA Base | Is this a shared Base/Prior residual? | Success, 183 steps, reward 1.0 | Single deterministic reset diagnostic | Yes for this identity | Base solves it, so a method would not address a shared residual |

Execution metadata:

- Run dir: `runs/xvla_prior/diagnostic_smolvla_base_libero_goal_task3_id20260729_officialenv_20260718T0542KST`
- Execution type: `VLA_INFERENCE`
- Evidence role: `BASE`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `19128`; WSL worker PID: `299`
- Simulator episodes: `1`; infrastructure failures: `0`
- Peak VRAM: `926.638 MiB`
- Result SHA-256: `b8d2cc128690e713b89918dfc73ba8ad1772e07d9a8f0d84e123cfa7445bbb13`
- Video: `runs/xvla_prior/smolvla_goal_task3_id20260729_videos/frozen_base/libero_goal/task_3_identity_20260729.mp4`
- Video SHA-256: `4c90414772b06716ba6f6cda7dfdbda769ae2b87bcfe5e02c1ba2c4846cb6240`

Scientific interpretation: task3 still has the earlier identity `20260728` as a genuine Base/X-VLA/OpenVLA-INT4 clean failure with task-level expert headroom, but the required repeated-residual safeguard is not met. Identities `20260730` and `20260731` were solved by X-VLA, and identity `20260729` was solved by Base. No candidate generation, training, LoRA/QLoRA update, or Ours rollout is authorized from this evidence.

