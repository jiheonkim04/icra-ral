# R2P-XVLA Optimizer-Step Gate

Decision: `R2P_XVLA_OPTIMIZER_GATE_FROZEN_TRAINING_NOT_LAUNCHED`

This is a report-only pre-training contract. It freezes the exact first optimizer-step gate for `R2P-XVLA`, but it does not launch training, create an optimizer, call `optimizer.step`, write checkpoints, or run closed-loop Ours.

## Gate status

- Frozen: `true`
- Armed for training launch: `false`
- Reason not armed: task5 spec-locked `train_lora` and offline-validation runners are not yet implemented or tested.
- Training launched now: `false`
- Optimizer step authorized now: `false`

## Frozen arms

| Arm | Role | Output dir | Steps | Phase weights source / transit / target |
| --- | --- | --- | ---: | --- |
| `r2p_xvla_rank8_phase_weights_lr1e4_steps64` | primary selected method | `runs/xvla_prior/epoch5_r2p_xvla_task5_training/r2p_xvla_rank8_phase_weights_lr1e4_steps64` | 64 | `1.0 / 2.0 / 1.5` |
| `uniform_task5_xvla_rank8_lambda0_lr1e4_steps64` | uniform ablation | `runs/xvla_prior/epoch5_r2p_xvla_task5_training/uniform_task5_xvla_rank8_lambda0_lr1e4_steps64` | 64 | `1.0 / 1.0 / 1.0` |

Shared limits: local files only, no downloads, device index `0`, batch size `1`, learning rate `0.0001`, max CUDA peak `14500` MiB, max wall clock `90` minutes per arm.

## Required before first optimizer step

- Implement and test a task5 spec-locked `train_lora` runner.
- Reject unknown `arm_id` and any third task5 configuration.
- Write `worker.pid`, `training_status.json`, `heartbeat.json`, and `frozen_spec_snapshot.json` before the first optimizer step.
- Enforce `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Refuse output directories outside `runs/xvla_prior/epoch5_r2p_xvla_task5_training`.
- Refuse `max_steps_override > 64`.
- Record `stdout.log`, `stderr.log`, `exit_code.txt`, `result.json`, and git commit.

## Stop conditions

- Any attempted network download.
- Optimizer step before required status/heartbeat/spec snapshot writes.
- Unknown arm or third configuration.
- Nonfinite loss or gradient.
- CUDA peak above `14500` MiB.
- Checkpoint outside frozen save steps `[16, 32, 64]`.
- Closed-loop Ours rollout during training.
- Residual reward used for checkpoint selection.
- Privileged object state or phase labels used as inference input.

## Next action

Implement and validate the spec-locked `R2P-XVLA` `train_lora` runner under this frozen gate. Do not launch training until runner tests pass and this gate can be armed.
