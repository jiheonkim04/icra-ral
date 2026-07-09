# FCAR Tiny Gate Result

Date: 2026-07-10 KST

- final decision: `FCAR_KILLED_BY_STATIC_BASELINE`
- status: `completed`
- experiments happened: `True`
- training happened: `True`
- trained components: `['fixed rank-4 LoRA baseline regenerated for predictions', 'FCAR tiny CPU gate']`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- official model/dataset used: `True`
- old custom route used: `False`

## Split

- prediction artifact: `regenerated`
- seed: `0`
- counts: `{'train': {'frame_count': 120, 'episode_distribution': {'1': 20, '2': 20, '4': 20, '7': 20, '8': 20, '14': 20}, 'task_distribution': {'1': 40, '2': 20, '4': 20, '5': 20, '8': 20}}, 'val': {'frame_count': 40, 'episode_distribution': {'3': 20, '9': 20}, 'task_distribution': {'2': 20, '4': 20}}, 'test': {'frame_count': 40, 'episode_distribution': {'13': 20, '15': 20}, 'task_distribution': {'5': 20, '8': 20}}}`
- leakage checks: `{'episode_disjoint_train_val': True, 'episode_disjoint_train_test': True, 'episode_disjoint_val_test': True, 'no_ground_truth_or_oracle_in_inference_features': True, 'old_custom_route_used': False}`

## Test Metrics

| variant | action L2 | translation L2 | rotation L2 | gripper abs | gripper sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen_base | 0.123998278 | 0.077212497 | 0.014495393 | 0.058697024 | 0.975 |
| rank4_lora | 0.076191123 | 0.072429293 | 0.015669167 | 0.009318775 | 1.0 |
| mean_action_prior | 1.148631734 | 0.528918376 | 0.079824928 | 0.990071797 | 0.6 |
| frame_oracle | 0.066124022 | 0.062313526 | 0.013812679 | 0.010473554 | 1.0 |
| task_oracle | 0.076191123 | 0.072429293 | 0.015669167 | 0.009318775 | 1.0 |
| moira_style_instruction_task_router | 0.123998278 | 0.077212497 | 0.014495393 | 0.058697024 | 0.975 |
| adapter_soup_static_merge | 0.091179973 | 0.066174662 | 0.013442673 | 0.032959476 | 1.0 |
| fcar_tiny_gate | 0.100144625 | 0.067082693 | 0.013609166 | 0.041781114 | 0.975 |

## FCAR

- gain over frozen/base: `{'absolute': 0.023853653, 'relative': 0.192370841}`
- recovered fraction of frame-oracle headroom: `0.41216345`
- alpha stats: `{'mean': 0.443432957, 'std': 0.02648654, 'min': 0.320281953, 'max': 0.493465692, 'fraction_routed_to_lora_alpha_ge_0_5': 0.0, 'fraction_routed_to_base_alpha_lt_0_5': 1.0}`
- train/eval gap: `{'train_action_l2': 0.114790271, 'test_action_l2': 0.100144625, 'absolute_gap_test_minus_train': -0.014645646, 'relative_gap': -0.127586126}`
- static selected weight: `0.5`
- MoIRA-style routing: `{'1': 'frozen_base', '2': 'frozen_base', '4': 'frozen_base', '5': 'frozen_base', '8': 'frozen_base'}`

## Failure Cases

- top helps vs base: `[{'episode_index': 15, 'frame_index': 139, 'task_index': 8, 'task': 'put the black bowl in the bottom drawer of the cabinet and close it', 'phase': 'mid', 'fcar_action_l2': 1.33133173, 'base_action_l2': 1.960363984, 'base_minus_fcar_gain': 0.629032254}, {'episode_index': 15, 'frame_index': 202, 'task_index': 8, 'task': 'put the black bowl in the bottom drawer of the cabinet and close it', 'phase': 'late', 'fcar_action_l2': 0.301154971, 'base_action_l2': 0.384019345, 'base_minus_fcar_gain': 0.082864374}, {'episode_index': 13, 'frame_index': 0, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'early', 'fcar_action_l2': 0.03754228, 'base_action_l2': 0.066861071, 'base_minus_fcar_gain': 0.029318791}, {'episode_index': 13, 'frame_index': 179, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'mid', 'fcar_action_l2': 0.078602716, 'base_action_l2': 0.104345769, 'base_minus_fcar_gain': 0.025743053}, {'episode_index': 13, 'frame_index': 254, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'late', 'fcar_action_l2': 0.166973844, 'base_action_l2': 0.19118534, 'base_minus_fcar_gain': 0.024211496}]`
- top hurts vs base: `[{'episode_index': 13, 'frame_index': 239, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'late', 'fcar_action_l2': 0.066717923, 'base_action_l2': 0.043397024, 'base_minus_fcar_gain': -0.023320899}, {'episode_index': 13, 'frame_index': 14, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'early', 'fcar_action_l2': 0.069995537, 'base_action_l2': 0.053059462, 'base_minus_fcar_gain': -0.016936075}, {'episode_index': 13, 'frame_index': 224, 'task_index': 5, 'task': 'put both the alphabet soup and the tomato sauce in the basket', 'phase': 'late', 'fcar_action_l2': 0.069581583, 'base_action_l2': 0.055783395, 'base_minus_fcar_gain': -0.013798188}, {'episode_index': 15, 'frame_index': 190, 'task_index': 8, 'task': 'put the black bowl in the bottom drawer of the cabinet and close it', 'phase': 'late', 'fcar_action_l2': 0.066717722, 'base_action_l2': 0.054197349, 'base_minus_fcar_gain': -0.012520373}, {'episode_index': 15, 'frame_index': 126, 'task_index': 8, 'task': 'put the black bowl in the bottom drawer of the cabinet and close it', 'phase': 'mid', 'fcar_action_l2': 0.075910181, 'base_action_l2': 0.068704993, 'base_minus_fcar_gain': -0.007205188}]`

Exact next prompt: None