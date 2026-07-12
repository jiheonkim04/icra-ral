# DICD-VLA Mechanism Smoke Result

Date: `2026-07-12`

Final decision: `DICD_SYNTHETIC_MECHANISM_SMOKE_PASSED`

- smoke type: `synthetic_core_mechanism`
- mechanism smoke passed: `True`
- training happened: `True`
- real SmolVLA chunk smoke happened: `False`
- closed-loop experiment happened: `False`
- checkpoint: `reports\dicd_vla\checkpoints\dicd_synthetic_smoke.pt`
- checkpoint sha256: `a4f66e383283dd710c6938f9a0709ea6d20af5039ce2ad43b2068c81687196af`
- checks: `{'chunk_horizon_exceeds_delay': True, 'full_finite_gradients': True, 'full_loss_decreased': True, 'ablation_loss_decreased': True, 'checkpoint_reloaded': True, 'full_changes_direct_chunk_index': True, 'full_differs_from_no_history_ablation': True, 'no_privileged_inference_fields': True}`
- probe: `{'probe_index': 6, 'direct_chunk_index_action': [0.3400000035762787, 0.041005197912454605, 0.022888582199811935, 0.05999999865889549, 0.003599999938160181, -0.019999999552965164, -1.0], 'dicd_full_action': [0.4170002043247223, 0.035284582525491714, 0.0029570981860160828, 0.045875340700149536, 0.004686422646045685, -0.010194070637226105, -1.0154014825820923], 'dicd_reloaded_action': [0.4170002043247223, 0.035284582525491714, 0.0029570981860160828, 0.045875340700149536, 0.004686422646045685, -0.010194070637226105, -1.0154014825820923], 'dicd_no_history_action': [0.49418553709983826, 0.03506306931376457, 0.014230303466320038, 0.053471639752388, 0.005914054811000824, -0.01983019709587097, -0.992210328578949], 'full_vs_direct_delta_norm': 0.083025, 'full_vs_ablation_delta_norm': 0.082308}`
- elapsed seconds: `1.093`

Next step: run real SmolVLA chunk smoke before Stage A closed-loop rollout.
