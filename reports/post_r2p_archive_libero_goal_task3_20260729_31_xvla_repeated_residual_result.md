# Post-R2P Archive LIBERO-Goal Task3 X-VLA Repeated-Residual Screen

Decision: `TASK3_REPEATED_RESIDUAL_NOT_CONFIRMED_BY_XVLA_SCREEN_ADDITIONAL_IDENTITIES_MIXED_1_FAIL_2_SUCCESS`

After the single shared task3 residual at reset identity `20260728`, I screened X-VLA on three additional independent reset identities (`20260729..20260731`). This was a first-prior diagnostic only: no training, no LoRA/QLoRA, no Ours design, and no Ours rollout.

| Identity | Initial state index | X-VLA success | Steps | Final reward | Action chunks | Gate interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `20260729` | 18 | false | 900 | 0.0 | 30 | New first-prior failure; requires matched Base gate |
| `20260730` | 19 | true | 179 | 1.0 | 6 | First prior solved; not a residual |
| `20260731` | 20 | true | 176 | 1.0 | 6 | First prior solved; not a residual |

Execution metadata:

- Run dir: `runs/xvla_prior/repeated_residual_goal_task3_id20260729_31_xvla_prior_20260718T0536KST`
- Policy: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`
- Source repo head: `C:\assets\repos\X-VLA` at `6bc2513f5f1cbec715cc668b414392a6cae5c671`
- Execution type: `VLA_INFERENCE`
- Evidence role: `FIRST_PRIOR`
- Artifact status: `OFFICIAL_CODE_WITH_ENVIRONMENT_WORKAROUND`
- Windows launcher PID: `10744`; WSL worker PID: `304`
- Simulator episodes: `3`; infrastructure failures: `0`
- Peak VRAM: `3518.634 MiB`
- Result SHA-256: `da87268101cf4af728137baa97140d57f1e52b55fc6c484c2a3a00e6cf9ef9d3`

Scientific interpretation: the additional X-VLA screen did not confirm a robust repeated residual by itself. Only `20260729` failed; `20260730` and `20260731` were solved by the first prior. The only authorized next step is a matched SmolVLA Base gate on `20260729`, not candidate generation or training.

