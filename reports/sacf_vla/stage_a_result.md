# SACF-VLA Stage A Result

Date: `2026-07-12`

Final decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`

- mode: `stage-a`
- training happened: `False`
- closed-loop experiment happened: `True`
- summary: `{'by_variant': {'frozen_smolvla': {'successes': 7, 'total': 10, 'success_rate': 0.7, 'task_balanced_success_rate': 0.7, 'wilson_95_ci': [0.396773, 0.892211], 'per_task': {'libero_object/task_4': {'successes': 5, 'total': 5, 'rate': 1.0}, 'libero_spatial/task_4': {'successes': 2, 'total': 5, 'rate': 0.4}}, 'mean_semantic_component_norm': 0.0, 'mean_full_plain_action_delta': 0.0, 'mean_cag_full_null_delta': 0.0}, 'task_phase_mean_prefix': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'wilson_95_ci': [0.0, 0.27754], 'per_task': {'libero_object/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_semantic_component_norm': 0.0, 'mean_full_plain_action_delta': 0.0, 'mean_cag_full_null_delta': 0.0}, 'plain_bc_prefix': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'wilson_95_ci': [0.0, 0.27754], 'per_task': {'libero_object/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_semantic_component_norm': 0.0, 'mean_full_plain_action_delta': 0.0, 'mean_cag_full_null_delta': 0.0}, 'cag_null_guidance': {'successes': 1, 'total': 10, 'success_rate': 0.1, 'task_balanced_success_rate': 0.1, 'wilson_95_ci': [0.017876, 0.404156], 'per_task': {'libero_object/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 1, 'total': 5, 'rate': 0.2}}, 'mean_semantic_component_norm': 0.0, 'mean_full_plain_action_delta': 0.0, 'mean_cag_full_null_delta': 0.633193}, 'sacf_full': {'successes': 0, 'total': 10, 'success_rate': 0.0, 'task_balanced_success_rate': 0.0, 'wilson_95_ci': [0.0, 0.27754], 'per_task': {'libero_object/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}, 'libero_spatial/task_4': {'successes': 0, 'total': 5, 'rate': 0.0}}, 'mean_semantic_component_norm': 1.709826, 'mean_full_plain_action_delta': 0.429388, 'mean_cag_full_null_delta': 0.0}}, 'strongest_baseline': 'frozen_smolvla', 'strongest_baseline_task_balanced_success_rate': 0.7, 'sacf_full_task_balanced_success_rate': 0.0, 'final_decision': 'STAGE_A_PERMANENT_KILL_CLEARLY_WORSE'}`
- elapsed seconds: `2078.122`

Next step: Archive kill and pivot if permanent kill; otherwise run Stage B under governance.
