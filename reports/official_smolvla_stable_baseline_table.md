# Official SmolVLA Stable Baseline Table

Date: 2026-07-10 KST

Final decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`

| baseline | action L2 | task-balanced L2 | episode-balanced L2 | task CI95 | episode CI95 | help/hurt/tie vs base |
| --- | ---: | ---: | ---: | --- | --- | --- |
| frozen_base | 0.085558433 | 0.085558433 | 0.085558433 | [0.075639184, 0.097026645] | [0.076220559, 0.095960753] | 0/0/1200 |
| rank4_lora | 0.09123014 | 0.09123014 | 0.09123014 | [0.07986336, 0.101656345] | [0.080742141, 0.102306624] | 599/601/0 |
| mean_action_prior | 1.197255124 | 1.197255124 | 1.197255124 | [1.181136375, 1.216539622] | [1.184961778, 1.209060703] | 8/1192/0 |
| frame_oracle | 0.068470215 | 0.068470215 | 0.068470215 | [0.060309178, 0.077742673] | [0.061091498, 0.077260171] | 599/0/601 |
| task_oracle | 0.079386015 | 0.079386015 | 0.079386015 | [0.071240998, 0.089715596] | [0.071824237, 0.088244789] | 266/214/720 |
| moira_style_instruction_task_router | 0.092209764 | 0.092209764 | 0.092209764 | [0.080982548, 0.104633344] | [0.081892039, 0.103806089] | 230/250/720 |
| static_mix_val_selected | 0.08113506 | 0.08113506 | 0.08113506 | [0.071302319, 0.091902178] | [0.072296478, 0.091217641] | 769/431/0 |

Realistic rank order: `[{'baseline': 'static_mix_val_selected', 'action_l2': 0.08113506}, {'baseline': 'frozen_base', 'action_l2': 0.085558433}, {'baseline': 'rank4_lora', 'action_l2': 0.09123014}, {'baseline': 'moira_style_instruction_task_router', 'action_l2': 0.092209764}, {'baseline': 'mean_action_prior', 'action_l2': 1.197255124}]`
