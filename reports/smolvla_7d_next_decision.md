# SmolVLA 7D Next Decision

Final decision: `FEATURE_PATH_MISMATCH`

- failure category: `FEATURE_PATH_MISMATCH`
- feature path audit result: `FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP`
- open-loop action replay result: `{'eligible_case_count': 6, 'expert': {'case_count': 6, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'object_movement_mean': 0.272579, 'progress_proxy_mean': 0.246324, 'reward_sum_mean': 1.0, 'runtime_case_steps': [260, 251, 216, 226, 223, 246], 'success_count': 6, 'success_rate': 1.0}, 'learned_aggregate_uses_only_eligible_cases': True, 'mean_action': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': 0.038336, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'ridge': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000286, 'progress_proxy_mean': 0.040788, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'small_mlp': {'case_count': 0, 'first_done_indices': [], 'object_movement_mean': None, 'progress_proxy_mean': None, 'reward_sum_mean': None, 'runtime_case_steps': [], 'success_count': 0, 'success_rate': None}, 'smolvla_7d_adapter': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': -0.059671, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}}`

Exact next step: Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.

Do not propose a new RA-L method unless the decision is `READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS`.
