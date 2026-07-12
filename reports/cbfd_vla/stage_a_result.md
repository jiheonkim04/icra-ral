# CBFD-VLA Stage A Result

Date: `2026-07-12`

Final decision: `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`

- mode: `stage-a`
- training happened: `False`
- closed-loop experiment happened: `True`
- summary: `{'by_variant': {'frozen_smolvla': {'successes': 7, 'total': 10, 'success_rate': 0.7, 'task_balanced_success_rate': 0.7, 'per_task': {'libero_10/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}, 'libero_spatial/task_4': {'successes': 5, 'total': 5, 'rate': 1.0}}, 'exceptions': 0, 'policy_latency_mean_s': 0.008328, 'peak_cuda_allocated_mb': 926.638, 'mean_action_delta_full_vs_direct': 0.0, 'mean_action_delta_full_vs_memory': 0.0}, 'direct_distill_proxy': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'exceptions': 0, 'policy_latency_mean_s': 0.000212, 'peak_cuda_allocated_mb': 926.638, 'mean_action_delta_full_vs_direct': 0.0, 'mean_action_delta_full_vs_memory': 0.0}, 'teacher_trace_memory': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'exceptions': 0, 'policy_latency_mean_s': 0.009835, 'peak_cuda_allocated_mb': 926.638, 'mean_action_delta_full_vs_direct': 0.0, 'mean_action_delta_full_vs_memory': 0.0}, 'cbfd_no_retention': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'exceptions': 0, 'policy_latency_mean_s': 0.000209, 'peak_cuda_allocated_mb': 926.638, 'mean_action_delta_full_vs_direct': 0.0, 'mean_action_delta_full_vs_memory': 0.0}, 'cbfd_full': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'exceptions': 0, 'policy_latency_mean_s': 0.009857, 'peak_cuda_allocated_mb': 926.638, 'mean_action_delta_full_vs_direct': 1.244676, 'mean_action_delta_full_vs_memory': 1.652989}}, 'strongest_baseline': 'frozen_smolvla', 'mechanism_active': True, 'exception_count': 0}`
- elapsed seconds: `1182.387`

Next step: Archive or repair according to governance.
