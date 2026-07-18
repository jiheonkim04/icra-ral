# Action-Consistent Missing-View Distillation Resumed Stage 0

Decision: `STAGE0_MECHANISM_NOT_SUPPORTED`

- Execution valid: `True`
- CUDA PID / device: `360 / NVIDIA GeForce RTX 5080`
- Training records / validation records: `480 / 12`
- Optimizer steps per arm: `[128, 128, 128, 128]`
- Teacher / student forward counts: `572 / 5452`
- Peak allocated / reserved VRAM bytes: `4593820672 / 4762632192`
- Peak system RAM fraction: `0.28530330796455144`
- Swap growth bytes: `0`
- Confirmatory outcomes accessed: `False`

## Teacher-agreement metrics

| policy | translation RMSE | rotation RMSE | raw gripper MAE | hidden MSE | reconstruction MSE |
|---|---:|---:|---:|---:|---:|
| BASE | 0.04733279142157762 | 0.04216545840152274 | 5.252569556236267 | 0.003265815059421584 | 0.9941876629988352 |
| OURS_FULL | 0.04699494321848803 | 0.03668702658188988 | 5.00265630086263 | 0.0031993202186034373 | 0.9079925566911697 |
| NO_RECONSTRUCTION | 0.047000837050223385 | 0.03665962870301436 | 5.014096220334371 | 0.0032020214615234486 | 0.9941876629988352 |
| NO_RAW_GRIPPER_MARGIN | 0.0470392515832138 | 0.03670080475981615 | 5.241029520829518 | 0.0032491753421102962 | 0.9237938672304153 |
| GENERIC_WRIST_DROPOUT_ADAPTER | 0.051431347702488604 | 0.24254405906817206 | 3.516813804705938 | 0.003962486827125152 | 0.9941876629988352 |

## Frozen gates

- `execution_valid`: `True`
- `real_clean_teacher_forwards`: `True`
- `real_dropout_student_forwards`: `True`
- `cuda_execution`: `True`
- `trainable_parameter_count_exact`: `True`
- `optimizer_steps_exact`: `True`
- `finite_nonzero_gradients`: `True`
- `weights_changed`: `True`
- `checkpoint_write_and_exact_reload`: `True`
- `frozen_xvla_unchanged`: `True`
- `exact_clean_bypass`: `True`
- `action_outputs_finite`: `True`
- `official_action_ranges`: `True`
- `translation_smoothness`: `True`
- `rotation_smoothness`: `True`
- `no_privileged_deployment_inputs`: `True`
- `no_direct_reconstructed_input`: `True`
- `reconstruction_gate`: `True`
- `base_directional_gate`: `True`
- `action_legality_and_smoothness`: `True`

The result uses discovery demos 0..39 for optimization and validation demo 40 only. Demos 41..49 and all confirmatory simulator outcomes remain untouched. No physical robot manipulation occurred.
