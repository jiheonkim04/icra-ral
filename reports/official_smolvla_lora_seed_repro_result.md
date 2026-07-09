# Official SmolVLA Rank-4 LoRA Seed Reproduction Result

Date: 2026-07-10 KST

- final decision: `STATIC_MERGE_ROBUST_BASELINE_READY`
- experiments happened: `True`
- training happened: `True`
- trained components: `['standard rank-4 LoRA baseline seeds']`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- official model/dataset used: `True`
- old custom route used: `False`
- seeds: `[11, 22, 33]`

## Mean/Std Across Seeds

| baseline | action L2 mean | action L2 std | task-balanced mean | translation mean | rotation mean | gripper abs mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_base | 0.085558433 | 0.0 | 0.085558433 | 0.069736605 | 0.01374415 | 0.022632962 |
| rank4_lora | 0.088239344 | 0.00290867 | 0.088239344 | 0.070671738 | 0.01334547 | 0.024954815 |
| mean_action_prior | 1.197255124 | 0.0 | 1.197255124 | 0.60695913 | 0.077452536 | 0.995449574 |
| frame_oracle | 0.069117204 | 0.002049401 | 0.069117204 | 0.056471108 | 0.012979944 | 0.018754088 |
| task_oracle | 0.081138309 | 0.001955707 | 0.081138309 | 0.067793179 | 0.01341679 | 0.020043934 |
| moira_style_instruction_task_router | 0.088145305 | 0.001719703 | 0.088145305 | 0.070067681 | 0.013620942 | 0.025253157 |
| static_mix_val_selected | 0.080616431 | 0.002595356 | 0.080616431 | 0.064095123 | 0.012227972 | 0.023262987 |

## Seed Robustness Answers

- rank4_lora_robustly_beats_frozen_base: `False`
- rank4_lora_robustly_beats_static_merge: `False`
- static_merge_remains_strongest_realistic_baseline: `True`
- lora_seed_variance_action_l2_std: `0.00290867`
- frame_oracle_headroom_remains_after_static: `True`
- task_oracle_remains_meaningful: `False`
- moira_style_task_router_remains_weak: `True`
- method_worthy_gap_left_after_static: `True`
- lora_instability_confirmed: `True`

- seed win counts realistic: `{'static_mix_val_selected': 3}`
- task win counts realistic sum: `{'frozen_base': 20, 'rank4_lora': 7, 'static_mix_val_selected': 93}`
- LoRA seed variance: `{'action_l2': {'mean': 0.088239344, 'std': 0.00290867, 'min': 0.084128699, 'max': 0.090426934}, 'range': 0.006298235, 'relative_std': 0.032963417}`
- frame oracle after static: `{'values': [0.011120454, 0.009974197, 0.01340303], 'mean': 0.011499227, 'std': 0.001425208, 'min': 0.009974197, 'max': 0.01340303}`
- task oracle headroom: `{'values': [0.00718575, 0.003063135, 0.003011486], 'mean': 0.004420124, 'std': 0.001955707, 'min': 0.003011486, 'max': 0.00718575}`

## Exact Next Step

Treat validation-selected static merge as the main realistic baseline for any later planning gate.
