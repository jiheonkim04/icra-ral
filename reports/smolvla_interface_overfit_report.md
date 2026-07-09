# SmolVLA Interface Overfit Report

- one-sample passed: `True`
- one-sample metrics: `{'sample_count': 1, 'action_l2': 0.0, 'action_l2_first6': 0.0, 'translation_l2': 0.0, 'rotation_l2': 0.0, 'gripper_error': 0.0, 'gripper_accuracy': 1.0, 'per_dim_mae': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'worst_action_dimensions': [{'dim': 0, 'mae': 0.0}, {'dim': 1, 'mae': 0.0}, {'dim': 2, 'mae': 0.0}]}`
- one-demo passed: `True`
- one-demo metrics: `{'sample_count': 3, 'action_l2': 0.002593, 'action_l2_first6': 0.002593, 'translation_l2': 0.002589, 'rotation_l2': 0.000118, 'gripper_error': 0.0, 'gripper_accuracy': 1.0, 'per_dim_mae': [0.002045, 0.001046, 0.001186, 0.0, 3.1e-05, 0.000113, 0.0], 'worst_action_dimensions': [{'dim': 0, 'mae': 0.002045}, {'dim': 2, 'mae': 0.001186}, {'dim': 1, 'mae': 0.001046}]}`
- previous split mean-action: `0.486561`
- previous split fixed adapter: `0.353069`
- larger split mean-action: `1.082453`
- larger split fixed adapter: `0.573503`
- larger split best MLP/ridge: `0.518738`
- per-dimension fixed adapter MAE: `[0.128575, 0.099236, 0.171897, 0.026936, 0.055568, 0.038649, 0.435579]`
