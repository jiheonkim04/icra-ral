# Missing Assets

This file is updated by preflight when local assets are missing.

The scaffold policy is local paths only:

- Do not download OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, checkpoints, or datasets automatically.
- Missing assets do not block scaffold, dummy smoke, or interface validation.
- Real adapter GPU work and simulator rollouts must stay skipped until these paths are configured and pass checks.

## Setup instructions

1. Copy `configs/paths.local.yaml.example` to `configs/paths.local.yaml`.
2. Fill in local paths for the assets you have.
3. Alternatively set environment variables:
   - `OPENVLA_OFT_CKPT`
   - `SMOLVLA_CKPT`
   - `LIBERO_ROOT`
   - `LIBERO_DATA_ROOT`
   - `ROBOSUITE_ROOT`
   - `DATA_ROOT`
   - `CHECKPOINT_ROOT`
   - `HF_HOME`
   - `WANDB_API_KEY`
4. Re-run preflight.

## Current status

Not checked yet. Run `scripts/00_preflight.ps1` locally.
