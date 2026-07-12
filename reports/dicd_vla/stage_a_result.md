# DICD-VLA Mechanism Smoke Result

Date: `2026-07-12`

Final decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`

- smoke type: `None`
- mechanism smoke passed: `None`
- training happened: `False`
- real SmolVLA chunk smoke happened: `None`
- closed-loop experiment happened: `True`
- checkpoint: `None`
- checkpoint sha256: `None`
- full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`
- full checkpoint sha256: `36d6c14bacf7bd3992d530fd428557175e626229eafca41b2449302ff5cb4538`
- ablation checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`
- ablation checkpoint sha256: `a0dc6b8a0b5e7db14896549d4bd2f60368751316d7f421671cd45eeab3c364d0`
- checks: `None`
- probe: `None`
- records: `None`
- train traces: `None`
- summary: `{'by_variant': {'frozen_smolvla_clean': {'successes': 5, 'total': 10, 'task_balanced_success_rate': 0.5, 'wilson_95_ci': [0.23659, 0.76341], 'per_task': {'libero_10/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}, 'libero_spatial/task_4': {'successes': 3, 'total': 5, 'rate': 0.6}}, 'mean_action_delta_norm': 0.0, 'mean_shaped_step_count': 0.0}, 'frozen_smolvla_delay': {'successes': 2, 'total': 10, 'task_balanced_success_rate': 0.2, 'wilson_95_ci': [0.056681, 0.509843], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}}, 'mean_action_delta_norm': 0.153489, 'mean_shaped_step_count': 371.9}, 'direct_chunk_index_delay': {'successes': 2, 'total': 10, 'task_balanced_success_rate': 0.2, 'wilson_95_ci': [0.056681, 0.509843], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}}, 'mean_action_delta_norm': 0.174042, 'mean_shaped_step_count': 370.0}, 'dicd_no_history_ablation': {'successes': 1, 'total': 10, 'task_balanced_success_rate': 0.1, 'wilson_95_ci': [0.017876, 0.404156], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_action_delta_norm': 0.308998, 'mean_shaped_step_count': 396.7}, 'dicd_full': {'successes': 1, 'total': 10, 'task_balanced_success_rate': 0.1, 'wilson_95_ci': [0.017876, 0.404156], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_action_delta_norm': 0.291109, 'mean_shaped_step_count': 387.5}}, 'paired_full_vs': {'frozen_smolvla_delay': {'win': 0, 'loss': 1, 'tie': 9}, 'direct_chunk_index_delay': {'win': 1, 'loss': 2, 'tie': 7}, 'dicd_no_history_ablation': {'win': 1, 'loss': 1, 'tie': 8}}, 'strongest_delayed_baseline': 'frozen_smolvla_delay', 'mechanism_active': True, 'passes_prototype_go': False, 'method_decision': 'SIMPLE_BASELINE_EXPLAINS_METHOD'}`
- elapsed seconds: `5637.278`

Next step: follow the Stage A method decision.
