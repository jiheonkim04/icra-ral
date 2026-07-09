# SmolVLA Action Interface Audit

- audit result: `ACTION_INTERFACE_BUG`
- HDF5 action dim: `7`
- model action shape: `[6]`
- policy preprocessor action shape: `[6]`
- policy postprocessor action shape: `[6]`
- normalization mapping: `{'VISUAL': 'IDENTITY', 'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}`
- local action first6 mean: `[0.039349, 0.056619, -0.091883, 0.012416, -0.005815, 0.063391]`
- local action first6 std: `[0.262207, 0.308574, 0.370461, 0.035068, 0.05897, 0.102793]`
- local action min: `[-0.774107, -0.886607, -0.9375, -0.147857, -0.233571, -0.110357, -1.0]`
- local action max: `[0.932143, 0.875893, 0.9375, 0.235714, 0.290357, 0.375, 1.0]`
- translation scale: `{'local_translation_std': [0.262207, 0.308574, 0.370461], 'local_translation_min': [-0.774107, -0.886607, -0.9375], 'local_translation_max': [0.932143, 0.875893, 0.9375]}`
- rotation scale: `{'local_rotation_std': [0.035068, 0.05897, 0.102793], 'local_rotation_min': [-0.147857, -0.233571, -0.110357], 'local_rotation_max': [0.235714, 0.290357, 0.375]}`
- local/checkpoint action mean abs z: `{'so100-blue.buffer.action.mean': [0.098128, 3.186899, 6.881818, 2.714833, 2.795356, 0.650732], 'so100-red.buffer.action.mean': [0.167797, 2.919682, 6.55052, 3.0935, 3.097478, 0.631502], 'so100.buffer.action.mean': [0.058979, 2.287434, 2.203676, 1.532316, 0.461875, 0.627071]}`
- label reconstruction sanity: `{'demo_name': 'demo_0', 'timestep': 3, 'chunk_first_matches_action_t_first6': True, 'chunk_second_matches_action_t_plus_1_first6': True, 'chunk_shape': [50, 6], 'expert_action_t_shape': [7]}`
- action chunk horizon alignment: `{'chunk_size': 50, 'chunk_starts_at_observation_timestep': True, 'chunk_second_step_is_next_hdf5_action': True, 'off_by_one_detected_in_chunk_builder': False}`
- bug indicators: `{'action_dimension_mismatch_6d_model_7d_hdf5': True, 'checkpoint_action_normalization_mismatch': True, 'gripper_dimension_synthesized_not_learned': True}`
- gripper convention: `{'hdf5_gripper_unique_values': [-1.0, 1.0], 'current_adapter': 'ACTION_STRATEGY_GRIPPER_CLOSE fills the 7th action dimension outside the 6D model head'}`
