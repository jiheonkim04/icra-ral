# PhaseBarrier-VLA Prototype Result

Invalidated result: `CHECKPOINT_IDENTITY_MISMATCH`

This 100-episode run is preserved only as invalid evidence. It retrained the PhaseBarrier head and produced a different training set identity from the original prototype (`1` positive label instead of the original `8`). It is not used for the bounded PhaseBarrier decision. The valid adjudication is `reports/phase_barrier_bounded_repair_result.json`.

Final decision: `PHASE_BARRIER_PROTOTYPE_GO`

- training happened: `True`
- closed-loop experiment happened: `True`
- variants: `['frozen_smolvla', 'pre_vla_style_halt_proxy', 'simple_global_damping', 'phase_barrier_no_phase_ablation', 'phase_barrier_full']`
- training records: `20`
- eval manifest: `{'tasks': [{'suite': 'libero_spatial', 'task_id': 4, 'role': 'stable_grasp_contact_transition', 'instruction': 'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'}, {'suite': 'libero_10', 'task_id': 4, 'role': 'long_horizon_contact_and_release', 'instruction': 'put the white mug on the left plate and put the yellow and white mug on the right plate'}], 'eval_identities': [20260712, 20260713, 20260714, 20260715, 20260716, 20260717, 20260718, 20260719, 20260720, 20260721], 'planned_episodes': 100}`
- summary: `{'by_variant': {'frozen_smolvla': {'successes': 7, 'total': 20, 'success_rate': 0.35, 'task_balanced_success_rate': 0.35, 'mean_shaped_steps': 0.0, 'mean_action_delta_norm': 0.0, 'exceptions': 0}, 'phase_barrier_full': {'successes': 8, 'total': 20, 'success_rate': 0.4, 'task_balanced_success_rate': 0.4, 'mean_shaped_steps': 246.5, 'mean_action_delta_norm': 0.047903, 'exceptions': 0}, 'phase_barrier_no_phase_ablation': {'successes': 7, 'total': 20, 'success_rate': 0.35, 'task_balanced_success_rate': 0.35, 'mean_shaped_steps': 254.6, 'mean_action_delta_norm': 0.057307, 'exceptions': 0}, 'pre_vla_style_halt_proxy': {'successes': 2, 'total': 20, 'success_rate': 0.1, 'task_balanced_success_rate': 0.1, 'mean_shaped_steps': 311.2, 'mean_action_delta_norm': 0.183718, 'exceptions': 0}, 'simple_global_damping': {'successes': 0, 'total': 20, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'mean_shaped_steps': 400.0, 'mean_action_delta_norm': 0.115545, 'exceptions': 0}}, 'strongest_non_ablation_baseline': 'frozen_smolvla', 'full_task_balanced_success_rate': 0.4, 'strongest_baseline_task_balanced_success_rate': 0.35, 'ablation_task_balanced_success_rate': 0.35, 'absolute_gain_over_strongest_baseline_pp': 5.0, 'relative_failure_rate_reduction': 0.076923, 'route_a_go': True, 'route_b_go': False, 'passes_prototype_go': True}`
- latency/VRAM: `{'elapsed_seconds': 2678.916, 'cuda_memory': {'allocated_bytes': 9568256, 'max_allocated_bytes': 971650560, 'allocated_mb': 9.125, 'max_allocated_mb': 926.638}}`

## Exact Next Step

Scale PhaseBarrier-VLA with larger rollout count, confidence intervals, second backbone, and second condition.
