# SmolVLA 7D Closed-Loop Failure Analysis

- closed-loop divergence executed: `False`
- reason: `Skipped as a model-quality measurement because STATE 1 found live closed-loop feature mismatch.`
- blocked by: `FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP`
- required before rerun: `Provide live env features matching HDF5 ee_states (ee_pos + ee_ori) or retrain/evaluate with the live observation schema.`

Prior eligible-only open-loop replay result:

`{'eligible_case_count': 6, 'expert': {'case_count': 6, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'object_movement_mean': 0.272579, 'progress_proxy_mean': 0.246324, 'reward_sum_mean': 1.0, 'runtime_case_steps': [260, 251, 216, 226, 223, 246], 'success_count': 6, 'success_rate': 1.0}, 'learned_aggregate_uses_only_eligible_cases': True, 'mean_action': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': 0.038336, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'ridge': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000286, 'progress_proxy_mean': 0.040788, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'small_mlp': {'case_count': 0, 'first_done_indices': [], 'object_movement_mean': None, 'progress_proxy_mean': None, 'reward_sum_mean': None, 'runtime_case_steps': [], 'success_count': 0, 'success_rate': None}, 'smolvla_7d_adapter': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': -0.059671, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}}`
