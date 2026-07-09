# SmolVLA 7D Standard Replay Baseline Result

Final decision: `EXPERT_REPLAY_UNSTABLE`

## Split

- tasks: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo']`
- train/eval demos: `10 / 4`
- sampled train/eval records: `80 / 32`
- raw timesteps: `25732`
- leakage: `{'exact_record_overlap': 0, 'demo_overlap': 0, 'task_overlap': 2, 'has_exact_record_leakage': False, 'has_demo_overlap': False, 'has_task_overlap': True, 'note': 'Task/demo overlap can be intentional for same-task or same-demo time holdout; exact record overlap must remain zero.'}`
- mean-action nontrivial: `True`

## Offline Metrics

- mean_action: action_l2 `0.995239`, translation `0.489242`, rotation `0.099001`, gripper_error `0.814062`
- ridge: action_l2 `0.889572`, translation `0.451487`, rotation `0.095941`, gripper_error `0.682265`
- small_mlp: action_l2 `0.799645`, translation `0.438851`, rotation `0.09357`, gripper_error `0.595578`
- frozen_base_smolvla_7d_adapter: action_l2 `1.105272`, translation `0.646902`, rotation `0.169899`, gripper_error `0.800004`
- smolvla_7d_adapter_no_lora: action_l2 `0.79289`, translation `0.471411`, rotation `0.102925`, gripper_error `0.568126`
- smolvla_state_proj_lora_rank4_7d_adapter: action_l2 `0.769789`, translation `0.460595`, rotation `0.087832`, gripper_error `0.540287`
- smolvla_state_proj_lora_rank8_7d_adapter: action_l2 `0.775132`, translation `0.469288`, rotation `0.089615`, gripper_error `0.539364`

## Replay Aggregate

- expert: `{'case_count': 2, 'success_count': 1, 'success_rate': 0.5, 'reward_sum_mean': 0.5, 'first_done_indices': [None, 225], 'progress_proxy_mean': 0.21065, 'object_movement_mean': 0.170842, 'runtime_case_steps': [272, 226]}`
- mean_action: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.068504, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- ridge: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.154496, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- small_mlp: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': 0.060931, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`
- smolvla_7d_adapter: `{'case_count': 1, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None], 'progress_proxy_mean': -0.055137, 'object_movement_mean': 0.000246, 'runtime_case_steps': [237]}`

## Action Validity

- best LoRA policy: `smolvla_state_proj_lora_rank4_7d_adapter`
- offline eval validity: `{'action_shape': [32, 7], 'expected_action_shape': ['T', 7], 'shape_exactly_7d': True, 'finite': True, 'action_low_high': {'low': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 'high': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'min': [-0.451046, -0.205794, -0.466078, -0.035002, -0.07305, -0.255751, -1.161312], 'max': [0.317361, 0.439485, 0.508266, 0.061366, 0.062319, 0.171873, 1.099052]}, 'clip_rate_element': 0.026786, 'clip_rate_step': 0.1875, 'controller_valid_rate_proxy': 0.8125, 'silent_broadcast_or_truncation_detected': False, 'note': 'Proxy validity uses LIBERO HDF5/controller action convention [-1, 1]; env acceptance is reported separately when replay runs.', 'per_dim_clip_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1875], 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.1875}`
- dimensions clip most: `6`
- gripper dominates clipping: `True`
- action range/normalization fix needed: `False`

Exact next step: Fix or narrow exact-init replay until expert succeeds on every evaluated replay case.
