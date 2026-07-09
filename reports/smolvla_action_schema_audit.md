# SmolVLA Action Schema Audit

- canonical LIBERO schema: `LIBERO_7D`
- LIBERO action dim: `7`
- labels 7D throughout: `True`
- LIBERO stats: `{'count': 13298, 'action_dim': 7, 'min': [-0.774107, -0.886607, -0.9375, -0.147857, -0.233571, -0.110357, -1.0], 'max': [0.932143, 0.875893, 0.9375, 0.235714, 0.290357, 0.375, 1.0], 'mean': [0.039349, 0.056619, -0.091883, 0.012416, -0.005815, 0.063391, -0.178072], 'std': [0.262207, 0.308574, 0.370461, 0.035068, 0.05897, 0.102793, 0.984038], 'variance': [0.068752, 0.095218, 0.137241, 0.00123, 0.003477, 0.010566, 0.968331], 'translation_dims': [0, 1, 2], 'rotation_dims': [3, 4, 5], 'gripper_dim': 6, 'translation_variance_mean': 0.100404, 'rotation_variance_mean': 0.005091, 'gripper_values': [-1.0, 1.0], 'gripper_std': 0.984017, 'gripper_variance': 0.96829}`
- action semantics audit: `{'mean_l2_action_xyz_to_next_eef_delta': 0.485201, 'mean_l2_action_xyz_to_current_eef_xyz': 1.253746, 'mean_action_xyz_norm': 0.490377, 'mean_next_eef_delta_norm': 0.00547, 'inference': 'controller_delta_like_not_absolute_pose', 'caveat': 'This is an HDF5 numeric audit only; no simulator/controller was instantiated.'}`
- env action spec: `{'available_without_env_instantiation': False, 'env_expected_action_dim': None, 'action_low_high': None, 'gripper_convention': 'not instantiated; HDF5 labels expose signed binary 7th dimension', 'clipping_behavior': 'not audited; no simulator environment was created', 'reason': 'Runner intentionally avoids rollout or simulator creation in this bounded interface fix.'}`
- canonical SmolVLA schema: `SMOLVLA_NATIVE_SO100_6D`
- SmolVLA model action shape: `[6]`
- SmolVLA preprocessor action shape: `[6]`
- SmolVLA postprocessor action shape: `[6]`
- SmolVLA action output modules: `['model.action_in_proj', 'model.action_out_proj', 'model.action_time_mlp_in', 'model.action_time_mlp_out', 'model.state_proj']`
- alignment audit: `{'demo_name': 'demo_0', 'timestep': 3, 'chunk_shape': [50, 7], 'chunk_first_matches_action_t_7d': True, 'chunk_second_matches_action_t_plus_1_7d': True, 'off_by_one_detected': False, 'sampled_records_preserve_temporal_ordering': True, 'action_chunks_reduced_to_6d': False}`
- mismatch table: `[{'axis': 'action_dim', 'libero_7d': 7, 'smolvla_native': [6], 'status': 'mismatch'}, {'axis': 'normalization', 'libero_7d': 'train-split-only 7D mean/std required', 'smolvla_native': {'VISUAL': 'IDENTITY', 'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}, 'status': 'mismatch'}, {'axis': 'gripper', 'libero_7d': 'learned label dimension 6', 'smolvla_native': 'no native 7th output; prior bridge hard-coded gripper', 'status': 'mismatch'}]`
