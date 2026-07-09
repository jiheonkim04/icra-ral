# Official SmolVLA Robust Baseline Sweep Result

Date: 2026-07-10 KST

- final decision: `METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`
- experiments happened: `True`
- training happened: `False`
- trained components: `[]`
- GPU/download/OpenVLA-OFT happened: `False` / `False` / `False`
- official model/dataset used: `True`
- old custom route used: `False`
- splits/seeds: `5`

## Split Mean/Std Action L2

| baseline | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| frozen_base | 0.106514933 | 0.030256808 | 0.068254242 | 0.140706452 |
| rank4_lora | 0.118024225 | 0.023707422 | 0.072736632 | 0.142809041 |
| mean_action_prior | 1.144859705 | 0.018515874 | 1.126037973 | 1.173568796 |
| frame_oracle | 0.084582167 | 0.027591676 | 0.059649888 | 0.124051527 |
| task_oracle | 0.106079936 | 0.029986441 | 0.068254242 | 0.140706452 |
| moira_style_instruction_task_router | 0.106514933 | 0.030256808 | 0.068254242 | 0.140706452 |
| static_mix_val_selected | 0.105142674 | 0.026514373 | 0.063468234 | 0.140706452 |

## Win Counts

- realistic: `{'frozen_base': 2, 'static_mix_val_selected': 3}`
- with oracles: `{'frame_oracle': 5}`

## Surviving Gap Answers

- is_standard_lora_robustly_better_than_frozen_base: `False`
- is_standard_lora_split_dependent: `True`
- lora_wins_over_base_count: `2`
- is_static_merge_consistently_better_than_fcar_like_gating: `True`
- does_frame_oracle_headroom_remain_large: `True`
- is_task_oracle_still_weak: `True`
- is_method_worthy_frame_gap_left_after_static_merge: `True`
- are_simple_baselines_enough: `False`
- mean_action_l2: `{'frozen_base': 0.106514933, 'rank4_lora': 0.118024225, 'static_mix_val_selected': 0.105142674, 'frame_oracle': 0.084582167, 'task_oracle': 0.106079936, 'moira_style_router': 0.106514933}`
- frame_oracle_headroom_mean: `0.021932766`
- task_oracle_headroom_mean: `0.000434997`
- static_gap_to_frame_oracle_mean: `0.020560507`

## Fold Rank Orderings

- fold `0` test episodes `[1, 4]` realistic order: `[{'baseline': 'frozen_base', 'action_l2': 0.140706452}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.140706452}, {'baseline': 'static_mix_val_selected', 'action_l2': 0.140706452}, {'baseline': 'rank4_lora', 'action_l2': 0.142809041}, {'baseline': 'mean_action_prior', 'action_l2': 1.136228429}]`
- fold `1` test episodes `[2, 3]` realistic order: `[{'baseline': 'frozen_base', 'action_l2': 0.071951753}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.071951753}, {'baseline': 'static_mix_val_selected', 'action_l2': 0.089300645}, {'baseline': 'rank4_lora', 'action_l2': 0.125088216}, {'baseline': 'mean_action_prior', 'action_l2': 1.129058814}]`
- fold `2` test episodes `[7, 9]` realistic order: `[{'baseline': 'static_mix_val_selected', 'action_l2': 0.063468234}, {'baseline': 'frozen_base', 'action_l2': 0.068254242}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.068254242}, {'baseline': 'rank4_lora', 'action_l2': 0.072736632}, {'baseline': 'mean_action_prior', 'action_l2': 1.159404512}]`
- fold `3` test episodes `[8, 13]` realistic order: `[{'baseline': 'static_mix_val_selected', 'action_l2': 0.11940447}, {'baseline': 'rank4_lora', 'action_l2': 0.126285636}, {'baseline': 'frozen_base', 'action_l2': 0.127170723}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.127170723}, {'baseline': 'mean_action_prior', 'action_l2': 1.173568796}]`
- fold `4` test episodes `[14, 15]` realistic order: `[{'baseline': 'static_mix_val_selected', 'action_l2': 0.112833571}, {'baseline': 'rank4_lora', 'action_l2': 0.123201599}, {'baseline': 'frozen_base', 'action_l2': 0.124491494}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.124491494}, {'baseline': 'mean_action_prior', 'action_l2': 1.126037973}]`
