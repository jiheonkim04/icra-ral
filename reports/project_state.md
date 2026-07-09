# Project State

Date: 2026-07-09 KST

Branch: `codex/smolvla-7d-standard-replay-baseline`

Current decision: `EXPERT_REPLAY_UNSTABLE`

## Current Route

SmolVLA 7D standard replay baseline reproduction is the active gate before any new method work.

## Standard Replay Baseline

- tasks: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo']`
- expert aggregate: `{'case_count': 2, 'success_count': 1, 'success_rate': 0.5, 'reward_sum_mean': 0.5, 'first_done_indices': [None, 225], 'progress_proxy_mean': 0.21065, 'object_movement_mean': 0.170842, 'runtime_case_steps': [272, 226]}`
- mean aggregate: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.068504, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- ridge aggregate: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.154496, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- MLP aggregate: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.060931, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- SmolVLA adapter aggregate: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': -0.055137, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- action range fix needed: `False`

## Conclusion

`EXPERT_REPLAY_UNSTABLE`

Fix or narrow exact-init replay until expert succeeds on every evaluated replay case.
