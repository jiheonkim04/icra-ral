# Exact-Init Expert Replay Stabilization

Final decision: `OFFLINE_TO_CONTROL_GAP`

This is an evaluation protocol gate, not a new method, not paper novelty, and not OpenVLA-OFT.

## Candidate Sweep

- candidate demos tested: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_7', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_6', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- candidate count: `8`
- expert aggregate: `{'case_count': 8, 'eligible_case_count': 6, 'failed_case_count': 2, 'success_count': 6, 'success_rate': 0.75, 'reward_sum_mean': 0.75, 'first_done_indices': [None, 259, 250, 215, 225, None, 222, 245], 'progress_proxy_mean': 0.239075, 'runtime_case_steps': [272, 260, 251, 216, 226, 241, 223, 246], 'eligible_case_ids': ['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']}`
- expert-success eligible cases: `6`
- expert-failed cases: `2`

## Learned Replay Boundary

- learned policy replay happened: `True`
- learned aggregate: `{'expert': {'case_count': 6, 'success_count': 6, 'success_rate': 1.0, 'reward_sum_mean': 1.0, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'progress_proxy_mean': 0.246324, 'object_movement_mean': 0.272579, 'runtime_case_steps': [260, 251, 216, 226, 223, 246]}, 'mean_action': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.038336, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}, 'ridge': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.040788, 'object_movement_mean': 0.000286, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}, 'small_mlp': {'case_count': 0, 'success_count': 0, 'success_rate': None, 'reward_sum_mean': None, 'first_done_indices': [], 'progress_proxy_mean': None, 'object_movement_mean': None, 'runtime_case_steps': []}, 'smolvla_7d_adapter': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.059671, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}, 'learned_aggregate_uses_only_eligible_cases': True, 'eligible_case_count': 6}`
- MLP replay result: `{'case_count': 0, 'success_count': 0, 'success_rate': None, 'reward_sum_mean': None, 'first_done_indices': [], 'progress_proxy_mean': None, 'object_movement_mean': None, 'runtime_case_steps': []}`
- MLP skip reason: `No persisted executable MLP artifact exists; no MLP retraining was performed.`

## Action Validity

- adapter clip-rate step mean: `0.23312`
- adapter controller-valid proxy mean: `0.76688`
- action validity fix needed: `False`

Exact next step: Stop method work; diagnose the offline-to-control gap in the fixed SmolVLA 7D baseline on the eligible set before proposing any new method.
