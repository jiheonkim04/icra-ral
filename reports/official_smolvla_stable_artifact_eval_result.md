# Official SmolVLA Stable Artifact Eval Result

Date: 2026-07-10 KST

- final decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`
- experiments happened: `True`
- training happened: `True`
- trained components: `['standard rank-4 LoRA baseline']`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- official model/dataset used: `True`
- old custom route used: `False`
- artifact generated: `True`
- artifact path: `reports\official_smolvla_stable_prediction_artifact.json`
- artifact size bytes: `7219361`
- artifact record count: `2800`

## Test Metrics

| baseline | action L2 | task-balanced L2 | translation L2 | rotation L2 | gripper abs | gripper sign acc | range violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_base | 0.085558433 | 0.085558433 | 0.069736605 | 0.01374415 | 0.022632962 | 0.993333333 | 0.550833333 |
| rank4_lora | 0.09123014 | 0.09123014 | 0.070690943 | 0.013045764 | 0.027685609 | 0.990833333 | 0.575 |
| mean_action_prior | 1.197255124 | 1.197255124 | 0.60695913 | 0.077452536 | 0.995449574 | 0.545833333 | 0.0 |
| frame_oracle | 0.068470215 | 0.068470215 | 0.056971588 | 0.012921991 | 0.017395659 | 0.995833333 | 0.560833333 |
| task_oracle | 0.079386015 | 0.079386015 | 0.068160808 | 0.013377581 | 0.017816481 | 0.995833333 | 0.566666667 |
| moira_style_instruction_task_router | 0.092209764 | 0.092209764 | 0.070046466 | 0.013393855 | 0.029344422 | 0.99 | 0.566666667 |
| static_mix_val_selected | 0.08113506 | 0.08113506 | 0.063464903 | 0.011991432 | 0.02435428 | 0.994166667 | 0.561666667 |

## Static Selection

`{'selected_weight': 0.5, 'selection_split': 'val', 'grid': {'0.0': {'train': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.079121107, 'action_l2_max': 2.013029337, 'translation_l2_mean': 0.069664788, 'rotation_l2_mean': 0.013586589, 'gripper_abs_mean': 0.01593317, 'gripper_sign_accuracy': 0.996666667, 'finite_all': True, 'range_violation_rate': 0.558333333, 'per_dim_abs_mean': [0.034372809, 0.032172607, 0.037262108, 0.0043032, 0.008611317, 0.006546818, 0.015933172]}, 'val': {'sample_count': 400, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.086303669, 'action_l2_max': 1.99358952, 'translation_l2_mean': 0.073316542, 'rotation_l2_mean': 0.015014106, 'gripper_abs_mean': 0.019755242, 'gripper_sign_accuracy': 0.995, 'finite_all': True, 'range_violation_rate': 0.5275, 'per_dim_abs_mean': [0.037189508, 0.032078638, 0.038984202, 0.004848023, 0.00913735, 0.007177847, 0.01975524]}, 'test': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.085558433, 'action_l2_max': 2.023066044, 'translation_l2_mean': 0.069736605, 'rotation_l2_mean': 0.01374415, 'gripper_abs_mean': 0.022632962, 'gripper_sign_accuracy': 0.993333333, 'finite_all': True, 'range_violation_rate': 0.550833333, 'per_dim_abs_mean': [0.034386059, 0.031320687, 0.037702447, 0.004402419, 0.00898412, 0.006308454, 0.022632962]}}, '0.25': {'train': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.074494647, 'action_l2_max': 2.013196707, 'translation_l2_mean': 0.065325362, 'rotation_l2_mean': 0.012634682, 'gripper_abs_mean': 0.015306255, 'gripper_sign_accuracy': 0.996666667, 'finite_all': True, 'range_violation_rate': 0.555, 'per_dim_abs_mean': [0.032577397, 0.029901872, 0.034756702, 0.00399642, 0.007998523, 0.006126558, 0.015306263]}, 'val': {'sample_count': 400, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.0798407, 'action_l2_max': 1.993033886, 'translation_l2_mean': 0.068302949, 'rotation_l2_mean': 0.013812091, 'gripper_abs_mean': 0.017818192, 'gripper_sign_accuracy': 0.995, 'finite_all': True, 'range_violation_rate': 0.525, 'per_dim_abs_mean': [0.035167145, 0.030226048, 0.035819547, 0.004536373, 0.00823855, 0.006757542, 0.017818187]}, 'test': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.08146935, 'action_l2_max': 2.022952318, 'translation_l2_mean': 0.065012532, 'rotation_l2_mean': 0.012549425, 'gripper_abs_mean': 0.023251239, 'gripper_sign_accuracy': 0.993333333, 'finite_all': True, 'range_violation_rate': 0.553333333, 'per_dim_abs_mean': [0.03207376, 0.028944218, 0.035133188, 0.004067552, 0.008148471, 0.005780653, 0.023251235]}}, '0.5': {'train': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.073324852, 'action_l2_max': 2.013827085, 'translation_l2_mean': 0.064223235, 'rotation_l2_mean': 0.012321731, 'gripper_abs_mean': 0.015135102, 'gripper_sign_accuracy': 0.995833333, 'finite_all': True, 'range_violation_rate': 0.55, 'per_dim_abs_mean': [0.032350713, 0.029047071, 0.034265368, 0.003943153, 0.007721735, 0.006046548, 0.015135093]}, 'val': {'sample_count': 400, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.076538267, 'action_l2_max': 1.992480159, 'translation_l2_mean': 0.066399748, 'rotation_l2_mean': 0.013248657, 'gripper_abs_mean': 0.016311025, 'gripper_sign_accuracy': 0.9975, 'finite_all': True, 'range_violation_rate': 0.5275, 'per_dim_abs_mean': [0.034514755, 0.02972744, 0.034235478, 0.00439091, 0.007780105, 0.006560298, 0.016311005]}, 'test': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.08113506, 'action_l2_max': 2.022903442, 'translation_l2_mean': 0.063464903, 'rotation_l2_mean': 0.011991432, 'gripper_abs_mean': 0.02435428, 'gripper_sign_accuracy': 0.994166667, 'finite_all': True, 'range_violation_rate': 0.561666667, 'per_dim_abs_mean': [0.031301774, 0.027938484, 0.034275681, 0.003910589, 0.007716025, 0.005535343, 0.024354286]}}, '0.75': {'train': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.075716783, 'action_l2_max': 2.014920235, 'translation_l2_mean': 0.066560223, 'rotation_l2_mean': 0.01269989, 'gripper_abs_mean': 0.01544431, 'gripper_sign_accuracy': 0.996666667, 'finite_all': True, 'range_violation_rate': 0.575833333, 'per_dim_abs_mean': [0.033824671, 0.029632643, 0.035568689, 0.00407671, 0.007902917, 0.006247587, 0.015444293]}, 'val': {'sample_count': 400, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.076758803, 'action_l2_max': 1.991928577, 'translation_l2_mean': 0.067901097, 'rotation_l2_mean': 0.013378104, 'gripper_abs_mean': 0.015256697, 'gripper_sign_accuracy': 0.9975, 'finite_all': True, 'range_violation_rate': 0.5125, 'per_dim_abs_mean': [0.035493058, 0.030654072, 0.03480403, 0.004504397, 0.007979627, 0.006569263, 0.015256705]}, 'test': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.084424623, 'action_l2_max': 2.022919416, 'translation_l2_mean': 0.065430309, 'rotation_l2_mean': 0.012184771, 'gripper_abs_mean': 0.025777241, 'gripper_sign_accuracy': 0.990833333, 'finite_all': True, 'range_violation_rate': 0.576666667, 'per_dim_abs_mean': [0.032356116, 0.028589129, 0.035546897, 0.003966086, 0.007784926, 0.005692541, 0.025777238]}}, '1.0': {'train': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.081342388, 'action_l2_max': 2.034900904, 'translation_l2_mean': 0.071874185, 'rotation_l2_mean': 0.013668235, 'gripper_abs_mean': 0.01621326, 'gripper_sign_accuracy': 0.996666667, 'finite_all': True, 'range_violation_rate': 0.5775, 'per_dim_abs_mean': [0.036644216, 0.031991497, 0.038160052, 0.004426388, 0.008439672, 0.006744173, 0.01621326]}, 'val': {'sample_count': 400, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.080697991, 'action_l2_max': 1.991379499, 'translation_l2_mean': 0.072590412, 'rotation_l2_mean': 0.014139794, 'gripper_abs_mean': 0.014707825, 'gripper_sign_accuracy': 0.9975, 'finite_all': True, 'range_violation_rate': 0.525, 'per_dim_abs_mean': [0.038072957, 0.032717555, 0.036925742, 0.00482351, 0.008533545, 0.00685013, 0.014707825]}, 'test': {'sample_count': 1200, 'eval_loss_mean': None, 'eval_loss_max': None, 'action_l2_mean': 0.09123014, 'action_l2_max': 2.032616377, 'translation_l2_mean': 0.070690943, 'rotation_l2_mean': 0.013045764, 'gripper_abs_mean': 0.027685609, 'gripper_sign_accuracy': 0.990833333, 'finite_all': True, 'range_violation_rate': 0.575, 'per_dim_abs_mean': [0.034900582, 0.030644929, 0.038510715, 0.004224912, 0.008305197, 0.006188717, 0.027685608]}}}, 'test_tuning_allowed': False}`

## Stability Analysis

- is_frozen_base_still_competitive: `True`
- is_rank4_lora_robustly_better_than_frozen_base: `False`
- is_rank4_lora_robustly_worse_than_frozen_base: `True`
- rank4_lora_task_wins_over_base: `16`
- rank4_lora_task_count: `40`
- does_static_merge_beat_both_base_and_lora: `True`
- does_frame_oracle_headroom_remain_meaningful: `True`
- does_frame_oracle_remain_after_static: `True`
- does_task_oracle_remain_weak: `False`
- does_moira_style_router_remain_weak: `True`
- are_metrics_stable_enough_for_method_design_later: `True`
- is_method_worthy_gap_left_after_simple_static_baselines: `True`
- does_larger_artifact_resolve_previous_split_instability: `True`
- best_realistic_action_l2: `0.08113506`
- action_l2: `{'frozen_base': 0.085558433, 'rank4_lora': 0.09123014, 'mean_action_prior': 1.197255124, 'frame_oracle': 0.068470215, 'task_oracle': 0.079386015, 'moira_style_instruction_task_router': 0.092209764, 'static_mix_val_selected': 0.08113506}`
- base_minus_lora: `-0.005671707`
- static_gain_over_best_single: `0.004423373`
- frame_oracle_headroom_over_base: `0.017088218`
- frame_oracle_headroom_after_static: `0.012664845`
- task_oracle_headroom_over_base: `0.006172418`
- frozen_base_task_bootstrap_ci_width: `0.021387461`
- realistic_win_counts_by_task: `{'frozen_base': 7, 'rank4_lora': 4, 'static_mix_val_selected': 29}`

## Exact Next Step

Run independent standard rank-4 LoRA seeds under the fixed manifest; do not design a new method yet.
