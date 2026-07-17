# Post-R2P Archive X-VLA Prior Scan: LIBERO Goal Identity 20260728

Decision: `POST_R2P_ARCHIVE_LIBERO_GOAL_IDENTITY20260728_XVLA_PRIOR_RESIDUAL_TASKS2_3`

This was official-prior inference only: X-VLA-Libero closed-loop evaluation on `libero_goal`, reset identity `20260728`. No Ours design, training, optimizer step, checkpoint write, or Ours rollout happened.

Runtime root: `runs/xvla_prior/failure_scan_libero_goal_identity20260728_post_r2p_archive_launch2_20260718T0517KST`

Summary:

- Completed tasks: `10 / 10`
- Successful tasks: `8 / 10`
- Clean X-VLA failures: `task_2`, `task_3`
- Infrastructure failures: `0`
- Simulator episodes: `10`
- Model action-chunk queries: `90`
- Peak VRAM: `3518.634 MiB`

| Task | Instruction | Success | Steps | Final reward | Result SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| 0 | open the middle drawer of the cabinet | yes | 123 | 1.0 | `935f50bf64deb55481a3eec4822d50cf7929c6f57e157a6e3fc1393e5a1748e7` |
| 1 | put the bowl on the stove | yes | 90 | 1.0 | `9ab6c4f403ba491b1c00247a99f4bbd8862d0f6e7b6479e7229c8e2eac4800f0` |
| 2 | put the wine bottle on top of the cabinet | no | 900 | 0.0 | `de009d7899bde5dfbc776cac18f2af4b61d6c28c9d1dd3b85c93a9e77c079503` |
| 3 | open the top drawer and put the bowl inside | no | 900 | 0.0 | `73204458fc1ce96174ebc9b9ac63b3500a5fedf3b82614e86420a4e4b65e134d` |
| 4 | put the bowl on top of the cabinet | yes | 84 | 1.0 | `bd1838568d8738253f9c5f06da682351e60625169a9536c22b6ca72ac6c8e3f8` |
| 5 | push the plate to the front of the stove | yes | 122 | 1.0 | `7c33b7c76600241a1f58662aa75224773e298efed2b3c928816258edeac80e7f` |
| 6 | put the cream cheese in the bowl | yes | 89 | 1.0 | `0842ec4c1ef82e2ea2a7ee3f25e401cbf98328b476079429407d83672130c131` |
| 7 | turn on the stove | yes | 79 | 1.0 | `5962aace0fe242009edc91b653e696ef9dccf656722903132c632a463434e4be` |
| 8 | put the bowl on the plate | yes | 79 | 1.0 | `bfefb2ce47b14a49f8ae9075c337f12c7f84b220f6a2cac86f6030f9cc6f4657` |
| 9 | put the wine bottle on the rack | yes | 145 | 1.0 | `5ae4691cc18732cc0eb51be21c953ebf0368abdd2bbf5d6c5e75f5e95f7669ea` |

Next gate: run matched SmolVLA Base diagnostics for `libero_goal/task_2` and `libero_goal/task_3` at reset identity `20260728`. Candidate generation and training remain unauthorized.
