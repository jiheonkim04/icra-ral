# Official SmolVLA Rank-4 LoRA Seed Reproduction Result

Date: 2026-07-10 KST

- final decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`
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
| rank4_lora | 0.087287222 | 0.001135689 | 0.087287222 | 0.070958647 | 0.013453266 | 0.023613221 |
| mean_action_prior | 1.197255124 | 0.0 | 1.197255124 | 0.60695913 | 0.077452536 | 0.995449574 |
| frame_oracle_upper_bound | 0.069590253 | 0.001577148 | 0.069590253 | 0.056519656 | 0.013076851 | 0.019226536 |
| task_oracle_upper_bound | 0.080959383 | 0.001502216 | 0.080959383 | 0.068138193 | 0.013452324 | 0.0196042 |
| task_or_instruction_router_proxy | 0.087401693 | 0.000190795 | 0.087401693 | 0.070005683 | 0.013634458 | 0.024551312 |
| validation_selected_action_space_static_mix | 0.079536743 | 0.0010252 | 0.079536743 | 0.063983205 | 0.012353563 | 0.022123452 |

## Seed Robustness Answers

- rank4_lora_robustly_beats_frozen_base: `False`
- rank4_lora_robustly_beats_static_merge: `False`
- static_merge_remains_strongest_realistic_baseline: `True`
- lora_seed_variance_action_l2_std: `0.001135689`
- frame_oracle_headroom_remains_after_static: `True`
- task_oracle_remains_meaningful: `False`
- moira_style_task_router_remains_weak: `True`
- method_worthy_gap_left_after_static: `True`
- lora_instability_confirmed: `False`

- seed win counts realistic: `{'static_mix_val_selected': 3}`
- task win counts realistic sum: `{'frozen_base': 8, 'rank4_lora': 11, 'static_mix_val_selected': 101}`
- LoRA seed variance: `{'action_l2': {'mean': 0.087287222, 'std': 0.001135689, 'min': 0.085934428, 'max': 0.088713382}, 'range': 0.002778954, 'relative_std': 0.013010939}`
- frame oracle after static: `{'values': [0.009448451, 0.009399861, 0.010991158], 'mean': 0.00994649, 'std': 0.000738958, 'min': 0.009399861, 'max': 0.010991158}`
- task oracle headroom: `{'values': [0.004072504, 0.003079897, 0.006644748], 'mean': 0.00459905, 'std': 0.001502216, 'min': 0.003079897, 'max': 0.006644748}`

## Exact Next Step

Do not proceed toward rollout; diagnose configuration drift against the frozen regeneration plan.
