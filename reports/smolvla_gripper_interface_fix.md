# SmolVLA Gripper Interface Fix

- before: `{'mode': 'hard_coded_bridge_fill', 'value_source': 'ACTION_STRATEGY_GRIPPER_CLOSE or related 6D-to-7D bridge strategy', 'learned': False}`
- after: `{'mode': 'learned_output_dimension', 'dim': 6, 'learned': True, 'loss': 'separate normalized gripper MSE added to pose regression loss', 'observed_label_values': [-1.0, 1.0]}`
- one-sample gripper accuracy: `1.0`
- one-demo gripper accuracy: `1.0`
- larger held-out gripper accuracy: `0.9`
