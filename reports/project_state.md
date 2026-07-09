# Project State

Date: 2026-07-09 KST

Branch: `codex/smolvla-7d-offline-to-control-gap`

Current decision: `FEATURE_PATH_MISMATCH`

## Current Route

SmolVLA 7D offline-to-control gap diagnosis is the active evaluation gate.

## Diagnosis

- eligible demos used: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- feature path audit result: `FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP`
- teacher-forced result: `{'mean_action': {'case_count': 6, 'action_l2_mean': 1.073821, 'translation_l2_mean': 0.512406, 'rotation_l2_mean': 0.101629, 'gripper_error_mean': 0.878721, 'first20_action_l2_mean': 0.724121, 'phase_critical_error_ratio_mean': 1.378987, 'gripper_timing_error_values': [None, None, None, None, None, None], 'gripper_sign_agreement_mean': 0.642681, 'translation_cosine_mean': 0.306502, 'translation_cosine_negative_rate_mean': 0.217218}, 'ridge': {'case_count': 6, 'action_l2_mean': 0.949596, 'translation_l2_mean': 0.495662, 'rotation_l2_mean': 0.094938, 'gripper_error_mean': 0.704616, 'first20_action_l2_mean': 0.429934, 'phase_critical_error_ratio_mean': 1.344207, 'gripper_timing_error_values': [120, None, None, 2, 48, 24], 'gripper_sign_agreement_mean': 0.721458, 'translation_cosine_mean': 0.280931, 'translation_cosine_negative_rate_mean': 0.311952}, 'smolvla_7d_adapter': {'case_count': 6, 'action_l2_mean': 0.867437, 'translation_l2_mean': 0.494638, 'rotation_l2_mean': 0.089128, 'gripper_error_mean': 0.628795, 'first20_action_l2_mean': 0.417105, 'phase_critical_error_ratio_mean': 1.35718, 'gripper_timing_error_values': [-6, -3, 3, -21, 26, 6], 'gripper_sign_agreement_mean': 0.778529, 'translation_cosine_mean': 0.321581, 'translation_cosine_negative_rate_mean': 0.265838}, 'critical_question': {'low_sparse_offline_l2_hides_full_sequence_or_phase_error': True, 'adapter_teacher_forced_l2_worse_than_ridge': False}}`
- open-loop result: `{'eligible_case_count': 6, 'expert': {'case_count': 6, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'object_movement_mean': 0.272579, 'progress_proxy_mean': 0.246324, 'reward_sum_mean': 1.0, 'runtime_case_steps': [260, 251, 216, 226, 223, 246], 'success_count': 6, 'success_rate': 1.0}, 'learned_aggregate_uses_only_eligible_cases': True, 'mean_action': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': 0.038336, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'ridge': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000286, 'progress_proxy_mean': 0.040788, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'small_mlp': {'case_count': 0, 'first_done_indices': [], 'object_movement_mean': None, 'progress_proxy_mean': None, 'reward_sum_mean': None, 'runtime_case_steps': [], 'success_count': 0, 'success_rate': None}, 'smolvla_7d_adapter': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': -0.059671, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}}`
- closed-loop divergence result: `{'executed': False, 'reason': 'Skipped as a model-quality measurement because STATE 1 found live closed-loop feature mismatch.', 'blocked_by': 'FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP', 'required_before_rerun': 'Provide live env features matching HDF5 ee_states (ee_pos + ee_ori) or retrain/evaluate with the live observation schema.'}`
- failure category: `FEATURE_PATH_MISMATCH`

## Conclusion

`FEATURE_PATH_MISMATCH`

Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.
