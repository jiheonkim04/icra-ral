# DICD-VLA Mechanism Smoke Result

Date: `2026-07-12`

Final decision: `DICD_REAL_TRACE_TRAINING_PASSED`

- smoke type: `None`
- mechanism smoke passed: `None`
- training happened: `True`
- real SmolVLA chunk smoke happened: `None`
- closed-loop experiment happened: `False`
- checkpoint: `None`
- checkpoint sha256: `None`
- full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`
- full checkpoint sha256: `36d6c14bacf7bd3992d530fd428557175e626229eafca41b2449302ff5cb4538`
- ablation checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`
- ablation checkpoint sha256: `a0dc6b8a0b5e7db14896549d4bd2f60368751316d7f421671cd45eeab3c364d0`
- checks: `{'training_examples_exist': True, 'labels_have_required_contrast': True, 'full_finite_gradients': True, 'full_loss_decreased': True, 'ablation_loss_decreased': True, 'full_checkpoint_reloaded': True, 'ablation_checkpoint_reloaded': True, 'full_changes_direct_chunk_index': True, 'full_differs_from_no_history_ablation': True, 'no_privileged_inference_fields': True}`
- probe: `{'probe_task': {'suite': 'libero_spatial', 'task_id': 4, 'role': 'stable_grasp_contact_transition', 'instruction': 'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'}, 'probe_index': 0, 'direct_chunk_index_action': [0.5840660333633423, -0.08086876571178436, 0.22452396154403687, -0.0067689064890146255, 0.020327724516391754, -0.03447887301445007, -0.9876165390014648], 'dicd_full_action': [0.433946430683136, -0.10617810487747192, 0.20637957751750946, -0.025242261588573456, 0.06502930074930191, -0.052407048642635345, -0.8977994322776794], 'dicd_full_reloaded_action': [0.433946430683136, -0.10617810487747192, 0.20637957751750946, -0.025242261588573456, 0.06502930074930191, -0.052407048642635345, -0.8977994322776794], 'dicd_no_history_action': [0.5520819425582886, -0.10739916563034058, 0.2472749650478363, -0.008722785860300064, 0.07256665080785751, -0.023675136268138885, -0.9566575884819031], 'dicd_no_history_reloaded_action': [0.5520819425582886, -0.10739916563034058, 0.2472749650478363, -0.008722785860300064, 0.07256665080785751, -0.023675136268138885, -0.9566575884819031], 'full_vs_direct_delta_norm': 0.185024, 'full_vs_ablation_delta_norm': 0.142301}`
- records: `None`
- train traces: `[{'suite': 'libero_spatial', 'task_id': 4, 'identity': 20260711, 'step_count': 80, 'chunk_sha256': '55bbbcbb90a8d52a8c7dfd0019c3d8587c52681e269fcc36224801f245e243cb', 'executed_action_sha256': '913871088d71ff70707f1fbd183e097bca0be84736f23b5ea6377faf26f25113', 'mean_delay_delta_norm': 0.221917, 'max_delay_delta_norm': 2.082289, 'executed_action_std': 0.500675, 'full_example_count': 156, 'ablation_example_count': 156}, {'suite': 'libero_10', 'task_id': 4, 'identity': 20260711, 'step_count': 80, 'chunk_sha256': '14bd8931047ae387ededf18e2ddaf1f1b635a4a5bd850f166f0aaeab875e7129', 'executed_action_sha256': '1231c228d15b8d6f567f0f0b19594803cb024e4f146ec4253c1c2aee4c2e5e05', 'mean_delay_delta_norm': 0.190799, 'max_delay_delta_norm': 2.002287, 'executed_action_std': 0.432327, 'full_example_count': 156, 'ablation_example_count': 156}]`
- elapsed seconds: `76.581`

Next step: run Stage A closed-loop rollout.
