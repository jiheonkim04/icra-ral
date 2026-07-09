# TG-7D Adapter Method Gate Results

- dataset/split used: `local_libero_goal_libero_para_group_holdout`
- LoRA rank: `4`
- runtime sec: `14.093`
- VRAM peak MB: `0.0`
- mean-action held-out paraphrase L2: `0.903848`
- MLP held-out paraphrase L2: `0.619985`
- MLP clean L2: `0.619985`
- standard LoRA held-out paraphrase L2: `0.600887`
- canonicalization held-out paraphrase L2: `0.587661`
- TG-7D held-out paraphrase L2: `0.740922`
- TG-7D object lexical L2: `0.744749`
- TG-7D clean L2: `0.735738`
- TG-7D counterfactual sensitivity: `{'available': True, 'pair_count': 30, 'prediction_delta_l2': 0.06286, 'expert_counterfactual_delta_l2': 0.819498, 'cf_prediction_to_counter_expert_l2': 0.899556, 'collapse_rate_delta_lt_0_05': 0.5}`
- TG-7D target consistency: `{'available': True, 'same_target_prediction_l2': 0.017795, 'pair_count': 360}`
- TG-7D trainable params: `295623`
- oracle target upper bound held-out paraphrase L2: `0.724674`
