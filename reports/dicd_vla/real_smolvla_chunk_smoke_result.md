# DICD-VLA Mechanism Smoke Result

Date: `2026-07-12`

Final decision: `DICD_REAL_SMOLVLA_CHUNK_SMOKE_PASSED`

- smoke type: `real_smolvla_action_chunk`
- mechanism smoke passed: `True`
- training happened: `False`
- real SmolVLA chunk smoke happened: `True`
- closed-loop experiment happened: `False`
- checkpoint: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`
- checkpoint sha256: `a4f66e383283dd710c6938f9a0709ea6d20af5039ce2ad43b2068c81687196af`
- checks: `{'official_policy_loaded': True, 'old_custom_route_not_used': True, 'raw_action_chunk_horizon_exceeds_delay': True, 'postprocessed_chunks_finite': True, 'postprocessed_chunk_horizon_exceeds_delay': True, 'postprocessed_action_dim_is_7': True, 'features_match_config_width': True, 'features_finite': True, 'real_delay_contrast_present': True, 'no_privileged_inference_fields': True}`
- probe: `None`
- records: `[{'step': 0, 'postprocessed_chunk_shape': [8, 7], 'postprocessed_chunk_finite': True, 'direct_delay_delta_norm': 0.44032, 'feature_dim': 74, 'feature_finite': True, 'synthetic_checkpoint_prediction_finite': True, 'synthetic_checkpoint_prediction_delta_from_direct': 0.334214}, {'step': 1, 'postprocessed_chunk_shape': [8, 7], 'postprocessed_chunk_finite': True, 'direct_delay_delta_norm': 0.332649, 'feature_dim': 74, 'feature_finite': True, 'synthetic_checkpoint_prediction_finite': True, 'synthetic_checkpoint_prediction_delta_from_direct': 0.323373}, {'step': 2, 'postprocessed_chunk_shape': [8, 7], 'postprocessed_chunk_finite': True, 'direct_delay_delta_norm': 0.405616, 'feature_dim': 74, 'feature_finite': True, 'synthetic_checkpoint_prediction_finite': True, 'synthetic_checkpoint_prediction_delta_from_direct': 0.367559}]`
- elapsed seconds: `22.618`

Next step: run real trace training before Stage A closed-loop rollout.
