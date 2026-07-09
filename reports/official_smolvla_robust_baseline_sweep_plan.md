# Official SmolVLA Robust Baseline Sweep Plan

Date: 2026-07-10 KST

Purpose: determine whether the LoRA/static-merge behavior exposed by the FCAR kill is split-dependent, without tuning FCAR or implementing a new method.

Boundary:

- no FCAR tuning or FCAR v2
- no new method training
- no simulator rollout or full benchmark
- no OpenVLA-OFT
- no old custom LIBERO_7D route
- no new downloads
- no test-set tuning

Data source:

- prediction artifact: `reports\fcar_prediction_artifact.json`
- official checkpoint: `C:\assets\checkpoints\smolvla_libero`
- official dataset: `C:\assets\datasets\lerobot_libero`

Sweep:

- folds: `5`
- frames per test fold: `40`
- static alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- validation-selected static alpha uses val split only, then evaluates on test

Baselines:

- frozen_base
- rank4_lora
- mean_action_prior
- frame_oracle
- task_oracle
- moira_style_instruction_task_router
- static_mix_val_selected
