# PhaseBarrier-VLA Prototype Result

Final decision: `PHASE_BARRIER_VALID_KILL`

- training happened: `True`
- closed-loop experiment happened: `True`
- variants: `['frozen_smolvla', 'pre_vla_style_halt_proxy', 'simple_global_damping', 'phase_barrier_no_phase_ablation', 'phase_barrier_full']`
- training records: `20`
- eval manifest: `{'tasks': [{'suite': 'libero_spatial', 'task_id': 4, 'role': 'stable_grasp_contact_transition', 'instruction': 'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'}, {'suite': 'libero_10', 'task_id': 4, 'role': 'long_horizon_contact_and_release', 'instruction': 'put the white mug on the left plate and put the yellow and white mug on the right plate'}], 'eval_identities': [20260712], 'planned_episodes': 10}`
- summary: `{'by_variant': {'frozen_smolvla': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 0.0, 'mean_action_delta_norm': 0.0, 'exceptions': 0}, 'phase_barrier_full': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 357.5, 'mean_action_delta_norm': 0.111434, 'exceptions': 0}, 'phase_barrier_no_phase_ablation': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 175.0, 'mean_action_delta_norm': 0.00476, 'exceptions': 0}, 'pre_vla_style_halt_proxy': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 341.5, 'mean_action_delta_norm': 0.494227, 'exceptions': 0}, 'simple_global_damping': {'successes': 0, 'total': 2, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 400.0, 'mean_action_delta_norm': 0.114796, 'exceptions': 0}}, 'strongest_non_ablation_baseline': 'frozen_smolvla', 'full_task_balanced_success_rate': 0.0, 'strongest_baseline_task_balanced_success_rate': 0.0, 'ablation_task_balanced_success_rate': 0.0, 'absolute_gain_over_strongest_baseline_pp': 0.0, 'relative_failure_rate_reduction': 0.0, 'route_a_go': False, 'route_b_go': False, 'passes_prototype_go': False}`
- latency/VRAM: `{'elapsed_seconds': 487.086, 'cuda_memory': {'allocated_bytes': 9568256, 'max_allocated_bytes': 971650560, 'allocated_mb': 9.125, 'max_allocated_mb': 926.638}}`

## Exact Next Step

Archive PhaseBarrier-VLA as first implemented kill and select a genuinely different second implemented method cycle.
