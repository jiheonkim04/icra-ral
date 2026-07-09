# SmolVLA 7D Split Audit

## same_task_demo_holdout

- task names: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo']`
- train/eval count: `300 / 100`
- sampled records: `400`
- raw timesteps: `13298`
- train demos: `['demo_0', 'demo_1', 'demo_2', 'demo_3', 'demo_4', 'demo_5', 'demo_6', 'demo_7', 'demo_8', 'demo_9', 'demo_10', 'demo_11', 'demo_12', 'demo_13', 'demo_14', 'demo_15', 'demo_16', 'demo_17', 'demo_18', 'demo_19', 'demo_20', 'demo_21', 'demo_22', 'demo_23', 'demo_24', 'demo_25', 'demo_26', 'demo_27', 'demo_28', 'demo_29']`
- eval demos: `['demo_30', 'demo_31', 'demo_32', 'demo_33', 'demo_34', 'demo_35', 'demo_36', 'demo_37', 'demo_38', 'demo_39']`
- train action variance: `[0.059143, 0.083346, 0.119546, 0.00121, 0.003625, 0.01038, 0.939154]`
- eval action variance: `[0.069181, 0.090757, 0.122339, 0.001108, 0.002822, 0.01138, 0.9324]`
- gripper distribution: `{'train': {'-1.0': 187, '1.0': 113}, 'eval': {'-1.0': 63, '1.0': 37}}`
- mean-action strength: `{'mean_action_l2': 1.082453, 'train_first6_std_l2': 0.526545, 'train_gripper_variance': 0.939155, 'mean_action_strong_due_to_low_variance': False}`
- leakage: `{'exact_record_overlap': 0, 'demo_overlap': 0, 'task_overlap': 1, 'has_exact_record_leakage': False, 'has_demo_overlap': False, 'has_task_overlap': True, 'note': 'Task/demo overlap can be intentional for same-task or same-demo time holdout; exact record overlap must remain zero.'}`

## same_task_time_holdout

- task names: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo']`
- train/eval count: `160 / 80`
- sampled records: `240`
- raw timesteps: `13298`
- train demos: `['demo_0', 'demo_1', 'demo_2', 'demo_3', 'demo_4', 'demo_5', 'demo_6', 'demo_7', 'demo_8', 'demo_9', 'demo_10', 'demo_11', 'demo_12', 'demo_13', 'demo_14', 'demo_15', 'demo_16', 'demo_17', 'demo_18', 'demo_19']`
- eval demos: `['demo_0', 'demo_1', 'demo_2', 'demo_3', 'demo_4', 'demo_5', 'demo_6', 'demo_7', 'demo_8', 'demo_9', 'demo_10', 'demo_11', 'demo_12', 'demo_13', 'demo_14', 'demo_15', 'demo_16', 'demo_17', 'demo_18', 'demo_19']`
- train action variance: `[0.074629, 0.084952, 0.148876, 0.001091, 0.002761, 0.01251, 0.711093]`
- eval action variance: `[0.029911, 0.039624, 0.087788, 0.000453, 0.002096, 0.001739, 0.99]`
- gripper distribution: `{'train': {'-1.0': 123, '1.0': 37}, 'eval': {'-1.0': 36, '1.0': 44}}`
- mean-action strength: `{'mean_action_l2': 1.158166, 'train_first6_std_l2': 0.569929, 'train_gripper_variance': 0.711094, 'mean_action_strong_due_to_low_variance': False}`
- leakage: `{'exact_record_overlap': 0, 'demo_overlap': 20, 'task_overlap': 1, 'has_exact_record_leakage': False, 'has_demo_overlap': True, 'has_task_overlap': True, 'note': 'Task/demo overlap can be intentional for same-task or same-demo time holdout; exact record overlap must remain zero.', 'temporal_chunk_overlap_risk': True, 'temporal_chunk_overlap_note': 'Same-demo time holdout has disjoint sampled timesteps but 50-step action chunks can be temporally near each other.'}`

## multi_task_demo_holdout

- task names: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo']`
- train/eval count: `150 / 60`
- sampled records: `210`
- raw timesteps: `40964`
- train demos: `['demo_0', 'demo_1', 'demo_2', 'demo_3', 'demo_4', 'demo_5', 'demo_6', 'demo_7', 'demo_8', 'demo_9']`
- eval demos: `['demo_10', 'demo_11', 'demo_12', 'demo_13']`
- train action variance: `[0.073419, 0.089334, 0.073856, 0.003089, 0.003064, 0.005721, 0.7296]`
- eval action variance: `[0.067762, 0.109399, 0.090329, 0.001764, 0.006132, 0.005957, 0.812222]`
- gripper distribution: `{'train': {'-1.0': 114, '1.0': 36}, 'eval': {'-1.0': 43, '1.0': 17}}`
- mean-action strength: `{'mean_action_l2': 0.959567, 'train_first6_std_l2': 0.498481, 'train_gripper_variance': 0.7296, 'mean_action_strong_due_to_low_variance': False}`
- leakage: `{'exact_record_overlap': 0, 'demo_overlap': 0, 'task_overlap': 3, 'has_exact_record_leakage': False, 'has_demo_overlap': False, 'has_task_overlap': True, 'note': 'Task/demo overlap can be intentional for same-task or same-demo time holdout; exact record overlap must remain zero.'}`
