# FEDO-VLA Prototype Result

Date: `2026-07-12`

Final decision: `CLEAN_RETENTION_FAILURE`

- mode: `stage-a`
- training happened: `False`
- closed-loop experiment happened: `True`
- full checkpoint: `reports/fedo_vla/checkpoints/fedo_full.pt`
- full checkpoint sha256: `89f6fc614f1bffcd4424416982cb030b5d8678afa7ed9f58e1d4f2e5b92cf99a`
- no-feedback checkpoint: `reports/fedo_vla/checkpoints/fedo_no_feedback.pt`
- no-feedback checkpoint sha256: `439366df6b70063c4ba5502234d6064a9834b7f86b9f4271643132bc3d805797`
- summary: `{'by_variant': {'faulted_frozen_smolvla': {'condition': 'faulted', 'successes': 0, 'total': 10, 'task_balanced_success_rate': 0.0, 'wilson_95_ci': [0.0, 0.27754], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_residual_norm': 0.0, 'mean_realized_error_norm': 0.19977}, 'static_inverse_gain': {'condition': 'faulted', 'successes': 2, 'total': 10, 'task_balanced_success_rate': 0.2, 'wilson_95_ci': [0.056681, 0.509843], 'per_task': {'libero_10/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_residual_norm': 0.161802, 'mean_realized_error_norm': 0.132586}, 'apex_feedback_proxy': {'condition': 'faulted', 'successes': 2, 'total': 10, 'task_balanced_success_rate': 0.2, 'wilson_95_ci': [0.056681, 0.509843], 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_residual_norm': 0.150621, 'mean_realized_error_norm': 0.159061}, 'fedo_no_feedback_ablation': {'condition': 'faulted', 'successes': 2, 'total': 10, 'task_balanced_success_rate': 0.2, 'wilson_95_ci': [0.056681, 0.509843], 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_residual_norm': 0.145709, 'mean_realized_error_norm': 0.146841}, 'fedo_full': {'condition': 'faulted', 'successes': 1, 'total': 10, 'task_balanced_success_rate': 0.1, 'wilson_95_ci': [0.017876, 0.404156], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_residual_norm': 0.150396, 'mean_realized_error_norm': 0.152719}, 'clean_frozen_smolvla': {'condition': 'clean', 'successes': 4, 'total': 10, 'task_balanced_success_rate': 0.4, 'wilson_95_ci': [0.168178, 0.68733], 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 3, 'total': 5, 'rate': 0.6}}, 'mean_residual_norm': 0.0, 'mean_realized_error_norm': 0.0}, 'clean_fedo_full': {'condition': 'clean', 'successes': 0, 'total': 10, 'task_balanced_success_rate': 0.0, 'wilson_95_ci': [0.0, 0.27754], 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_residual_norm': 0.143938, 'mean_realized_error_norm': 0.143938}}, 'strongest_faulted_baseline': 'static_inverse_gain', 'strongest_faulted_baseline_rate': 0.2, 'clean_retention_drop': 0.4, 'exception_count': 0, 'passes_prototype_go': False, 'method_decision': 'CLEAN_RETENTION_FAILURE'}`
- elapsed seconds: `1879.48`

Next step: Follow the Stage A method decision.
