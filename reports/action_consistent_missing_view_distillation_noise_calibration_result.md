# Action-Consistent Missing-View Distillation: Numerical-Noise Calibration

Decision: `NUMERICAL_NOISE_CALIBRATION_VALID`

- Frozen forward repetitions per row: `3`
- Fixed discovery calibration rows: `12`
- Optimizer steps: `0`
- Confirmatory outcomes accessed: `False`
- Condition image mask preserved: `True`

## Frozen normalization denominators

- `hidden_mse`: `0.002987779696316769`
- `translation_mse`: `0.0015878713347774465`
- `rotation_mse`: `0.0018577529408503324`
- `raw_gripper_margin_mse`: `56.628908475240074`
- `wrist_reconstruction_mse`: `0.9934094299872717`

## Repeated-forward numerical noise

- `translation_RMSE`: `0.0`
- `rotation_RMSE`: `0.0`
- `raw_gripper_margin_MAE`: `0.0`
- `action_hidden_MSE`: `0.0`

## Practical absolute thresholds

- `translation_RMSE`: `0.0001`
- `rotation_RMSE`: `0.0002`
- `raw_gripper_margin_MAE`: `0.002`
- `action_hidden_MSE`: `1e-05`

## Execution

- CUDA PID: `369`
- Elapsed seconds: `32.39`
- Exceptions: `0`
