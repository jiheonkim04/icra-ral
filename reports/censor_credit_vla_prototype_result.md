# CensorCredit-VLA Prototype Result

Final decision: `CENSOR_CREDIT_VALID_KILL`

- training happened: `True`
- closed-loop experiment happened: `True`
- training records: `24`
- eval manifest: `{'tasks': [{'suite': 'libero_spatial', 'task_id': 4, 'role': 'stable_grasp_contact_transition', 'instruction': 'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'}, {'suite': 'libero_10', 'task_id': 4, 'role': 'long_horizon_contact_and_release', 'instruction': 'put the white mug on the left plate and put the yellow and white mug on the right plate'}], 'eval_identities': [20260712], 'planned_episodes': 10}`
- summary: `{'by_variant': {'censor_credit_full': {'successes': 1, 'total': 2, 'success_rate': 0.5, 'task_balanced_success_rate': 0.5, 'mean_shaped_steps': 296.5, 'mean_action_delta_norm': 0.119921, 'exceptions': 0}, 'frozen_smolvla': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 0.0, 'mean_action_delta_norm': 0.0, 'exceptions': 0}, 'simple_temporal_ema': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 400.0, 'mean_action_delta_norm': 0.041235, 'exceptions': 0}, 'uncensored_recovery_ablation': {'successes': 1, 'total': 2, 'success_rate': 0.5, 'task_balanced_success_rate': 0.5, 'mean_shaped_steps': 278.5, 'mean_action_delta_norm': 0.11322, 'exceptions': 0}, 'vla_corrector_jump_proxy': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 357.5, 'mean_action_delta_norm': 1.864512, 'exceptions': 0}}, 'strongest_non_ablation_baseline': 'frozen_smolvla', 'full_task_balanced_success_rate': 0.5, 'strongest_baseline_task_balanced_success_rate': 0.0, 'ablation_task_balanced_success_rate': 0.5, 'absolute_gain_over_strongest_baseline_pp': 50.0, 'relative_failure_rate_reduction': 0.5, 'route_a_go': False, 'route_b_go': False, 'passes_prototype_go': False}`
- latency/VRAM: `{'elapsed_seconds': 465.897, 'cuda_memory': {'allocated_bytes': 9568256, 'max_allocated_bytes': 971650560, 'allocated_mb': 9.125, 'max_allocated_mb': 926.638}}`

## Exact Next Step

Archive as second implemented valid kill; campaign may now conclude TWO_IMPLEMENTED_METHODS_KILLED if no GO exists.
