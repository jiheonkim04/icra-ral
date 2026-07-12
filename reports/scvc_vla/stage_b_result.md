# SCVC-VLA Stage B Result

Date: `2026-07-12`

Final decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

- mode: `stage-b`
- training happened: `False`
- closed-loop experiment happened: `True`
- summary: `{'by_variant': {'clean_frozen_smolvla': {'successes': 10, 'total': 40, 'success_rate': 0.25, 'task_balanced_success_rate': 0.25, 'per_task': {'libero_10/task_4': {'successes': 4, 'total': 20, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 6, 'total': 20, 'rate': 0.3}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.0, 'mean_mse_output_vs_clean': 0.0, 'mean_image_delta_vs_shifted': 0.0}, 'shifted_frozen_smolvla': {'successes': 20, 'total': 40, 'success_rate': 0.5, 'task_balanced_success_rate': 0.5, 'per_task': {'libero_10/task_4': {'successes': 7, 'total': 20, 'rate': 0.35}, 'libero_spatial/task_4': {'successes': 13, 'total': 20, 'rate': 0.65}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.01998, 'mean_mse_output_vs_clean': 0.01998, 'mean_image_delta_vs_shifted': 0.0}, 'known_inverse_affine': {'successes': 10, 'total': 40, 'success_rate': 0.25, 'task_balanced_success_rate': 0.25, 'per_task': {'libero_10/task_4': {'successes': 4, 'total': 20, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 6, 'total': 20, 'rate': 0.3}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.019823, 'mean_mse_output_vs_clean': 0.0, 'mean_image_delta_vs_shifted': 0.126194}, 'scvc_no_temporal': {'successes': 10, 'total': 40, 'success_rate': 0.25, 'task_balanced_success_rate': 0.25, 'per_task': {'libero_10/task_4': {'successes': 4, 'total': 20, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 6, 'total': 20, 'rate': 0.3}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.01987, 'mean_mse_output_vs_clean': 0.016889, 'mean_image_delta_vs_shifted': 0.111272}, 'scvc_full': {'successes': 11, 'total': 40, 'success_rate': 0.275, 'task_balanced_success_rate': 0.275, 'per_task': {'libero_10/task_4': {'successes': 5, 'total': 20, 'rate': 0.25}, 'libero_spatial/task_4': {'successes': 6, 'total': 20, 'rate': 0.3}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.01981, 'mean_mse_output_vs_clean': 0.016791, 'mean_image_delta_vs_shifted': 0.111153}}, 'strongest_baseline': 'shifted_frozen_smolvla', 'mechanism_active': True, 'exception_count': 0}`
- elapsed seconds: `4555.17`

Next step: Archive or scale according to governance.
