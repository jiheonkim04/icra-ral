# SmolVLA 7D Replay Protocol Decision

Final decision: `OFFLINE_TO_CONTROL_GAP`

- experiments happened: `True`
- training happened: `False`
- replay/control happened: `True`
- learned policy replay happened: `True`
- candidate demos tested: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_7', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_6', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- fixed eligibility set: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- mean replay result: `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.038336, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}`
- ridge replay result: `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.040788, 'object_movement_mean': 0.000286, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}`
- MLP replay result: `{'case_count': 0, 'success_count': 0, 'success_rate': None, 'reward_sum_mean': None, 'first_done_indices': [], 'progress_proxy_mean': None, 'object_movement_mean': None, 'runtime_case_steps': []}`
- adapter replay result: `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.059671, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}`

Exact next step: Stop method work; diagnose the offline-to-control gap in the fixed SmolVLA 7D baseline on the eligible set before proposing any new method.
