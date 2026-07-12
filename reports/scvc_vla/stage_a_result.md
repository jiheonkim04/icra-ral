# SCVC-VLA Stage A Result

Date: `2026-07-12`

Final decision: `STAGE_A_NON_GO_TO_STAGE_B_REQUIRED`

- mode: `stage-a`
- training happened: `False`
- closed-loop experiment happened: `True`
- summary: `{'by_variant': {'clean_frozen_smolvla': {'successes': 6, 'total': 10, 'success_rate': 0.6, 'task_balanced_success_rate': 0.6, 'per_task': {'libero_10/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}, 'libero_spatial/task_4': {'successes': 4, 'total': 5, 'rate': 0.8}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.0, 'mean_mse_output_vs_clean': 0.0, 'mean_image_delta_vs_shifted': 0.0}, 'shifted_frozen_smolvla': {'successes': 4, 'total': 10, 'success_rate': 0.4, 'task_balanced_success_rate': 0.4, 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 3, 'total': 5, 'rate': 0.6}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.02, 'mean_mse_output_vs_clean': 0.02, 'mean_image_delta_vs_shifted': 0.0}, 'known_inverse_affine': {'successes': 5, 'total': 10, 'success_rate': 0.5, 'task_balanced_success_rate': 0.5, 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 4, 'total': 5, 'rate': 0.8}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.020244, 'mean_mse_output_vs_clean': 0.0, 'mean_image_delta_vs_shifted': 0.127235}, 'scvc_no_temporal': {'successes': 5, 'total': 10, 'success_rate': 0.5, 'task_balanced_success_rate': 0.5, 'per_task': {'libero_10/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}, 'libero_spatial/task_4': {'successes': 4, 'total': 5, 'rate': 0.8}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.020639, 'mean_mse_output_vs_clean': 0.014225, 'mean_image_delta_vs_shifted': 0.106669}, 'scvc_full': {'successes': 4, 'total': 10, 'success_rate': 0.4, 'task_balanced_success_rate': 0.4, 'per_task': {'libero_10/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 4, 'total': 5, 'rate': 0.8}}, 'exceptions': 0, 'mean_mse_shifted_vs_clean': 0.020781, 'mean_mse_output_vs_clean': 0.01386, 'mean_image_delta_vs_shifted': 0.106163}}, 'strongest_baseline': 'clean_frozen_smolvla', 'mechanism_active': True, 'exception_count': 0}`
- elapsed seconds: `1076.034`

Next step: Run Stage B.
