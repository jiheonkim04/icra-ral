# Post-R2P Archive SmolVLA Base Gate: LIBERO Goal Identity 20260728

Decision: `POST_R2P_ARCHIVE_LIBERO_GOAL_IDENTITY20260728_SMOLVLA_BASE_MIXED_TASK3_SHARED_RESIDUAL`

This was official SmolVLA Base inference only. No Ours design, LoRA/QLoRA training, optimizer step, checkpoint write, or Ours rollout happened.

Runtime root: `runs/xvla_prior/diagnostic_smolvla_base_libero_goal_tasks2_3_id20260728_officialenv_20260718T0525KST`

| Task | Instruction | X-VLA prior | SmolVLA Base | Decision |
| --- | --- | --- | --- | --- |
| 2 | put the wine bottle on top of the cabinet | failed | succeeded | `BASE_SOLVED_NOT_A_SHARED_RESIDUAL` |
| 3 | open the top drawer and put the bowl inside | failed | failed | `BASE_AND_FIRST_PRIOR_CLEAN_FAILURE_REQUIRES_HEADROOM_THEN_SECOND_PRIOR` |

Task 3 is the only surviving residual from this gate. It is not yet an Ours target: expert headroom, second-prior screening, and repeated-residual safeguards remain required before candidate generation or training.

Execution summary:

- Completed episodes: `2 / 2`
- Base successes: `1 / 2`
- Infrastructure failures: `0`
- Simulator episodes: `2`
- Model action-chunk queries: `8`
- Peak VRAM: `926.638 MB`

Next gate: run expert-headroom diagnostic for `libero_goal/task_3`, reset identity `20260728`.
