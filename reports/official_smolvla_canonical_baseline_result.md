# Official SmolVLA Canonical Baseline Result

Date: 2026-07-10 KST

- intermediate decision: `CANONICAL_BASELINES_READY_FOR_ROLLOUT`
- final decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`
- training happened: `False`
- checkpoint regeneration happened: `False`
- GPU inference happened: `True`
- downloads happened: `False`
- rollouts happened: `False`
- historical status: `SUPERSEDED_NONCANONICAL_PROTOCOL`

## Canonical Metrics

| baseline | action L2 mean | action L2 std | task-balanced L2 | task-balanced std | gripper abs | range violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_base | 0.085579125 | 0.001891792 | 0.085579125 | 0.001891792 | 0.021789329 | 0.555833333 |
| mean_action_prior | 1.197255124 | 0.0 | 1.197255124 | 0.0 | 0.995449574 | 0.0 |
| rank4_lora_seed_11 | 0.086743582 | 0.002208762 | 0.086743583 | 0.002208762 | 0.022823576 | 0.491 |
| validation_selected_action_space_static_mix_seed_11 | 0.085579125 | 0.001891792 | 0.085579125 | 0.001891792 | 0.021789329 | 0.555833333 |
| task_or_instruction_router_proxy_seed_11 | 0.085588454 | 0.001879102 | 0.085588454 | 0.001879102 | 0.021830886 | 0.540666667 |
| task_oracle_upper_bound_seed_11 | 0.085476523 | 0.001928636 | 0.085476523 | 0.001928636 | 0.021777214 | 0.5465 |
| frame_oracle_upper_bound_seed_11 | 0.084224881 | 0.00190461 | 0.084224881 | 0.00190461 | 0.021510728 | 0.525833333 |
| rank4_lora_seed_22 | 0.086474081 | 0.002680732 | 0.086474081 | 0.002680732 | 0.022580491 | 0.565166667 |
| validation_selected_action_space_static_mix_seed_22 | 0.085579125 | 0.001891792 | 0.085579125 | 0.001891792 | 0.021789329 | 0.555833333 |
| task_or_instruction_router_proxy_seed_22 | 0.086136031 | 0.001604338 | 0.086136031 | 0.001604338 | 0.022289969 | 0.564166667 |
| task_oracle_upper_bound_seed_22 | 0.084464263 | 0.001426185 | 0.084464263 | 0.001426185 | 0.020687639 | 0.560166666 |
| frame_oracle_upper_bound_seed_22 | 0.082615809 | 0.001405709 | 0.082615809 | 0.001405709 | 0.020611899 | 0.559333333 |
| rank4_lora_seed_33 | 0.086918872 | 0.002855345 | 0.086918872 | 0.002855345 | 0.023345334 | 0.578166667 |
| validation_selected_action_space_static_mix_seed_33 | 0.085579125 | 0.001891792 | 0.085579125 | 0.001891792 | 0.021789329 | 0.555833333 |
| task_or_instruction_router_proxy_seed_33 | 0.08684683 | 0.002549027 | 0.08684683 | 0.002549027 | 0.023286924 | 0.566833333 |
| task_oracle_upper_bound_seed_33 | 0.084499333 | 0.001897528 | 0.084499333 | 0.001897528 | 0.020915644 | 0.564166667 |
| frame_oracle_upper_bound_seed_33 | 0.083219926 | 0.00187481 | 0.083219926 | 0.00187481 | 0.020673581 | 0.568833333 |

## Static Selection
- seed 11: alpha `0.0`, split `val`
- seed 22: alpha `0.0`, split `val`
- seed 33: alpha `0.0`, split `val`

## Exact Next Step

Move the same canonical artifacts/checkpoints into the verified WSL/Linux LeRobot LIBERO environment, install only official `lerobot[libero]` dependencies, then run official smoke before any pilot.
