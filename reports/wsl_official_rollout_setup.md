# WSL Official Rollout Setup

Date: 2026-07-10 KST

## Objective

Set up a verified WSL/Linux official LeRobot LIBERO evaluation path for the locked SmolVLA base and rank-4 LoRA seeds, then run bounded official closed-loop smoke and pilot rollouts.

## Host And WSL

- Windows: Windows 10 Education 2009, build `26200`
- WSL distribution: `Ubuntu-22.04`, WSL2
- Linux: `Ubuntu 22.04.5 LTS`
- Kernel: `6.6.114.1-microsoft-standard-WSL2`
- WSL root disk: `/dev/sdd`, `1007G` total, `933G` available at setup check
- Windows mount: `/mnt/c`, `931G` total, `350G` available at setup check
- RAM at setup check: `11Gi` total, `10Gi` available
- Swap at setup check: `3.0Gi`

## CUDA Gate

- Windows GPU: `NVIDIA GeForce RTX 5080`
- WSL GPU: `NVIDIA GeForce RTX 5080`
- Driver reported in WSL: `596.21`
- GPU memory: `16303 MiB`
- WSL torch check:
  - `torch.cuda.is_available()`: `True`
  - device name: `NVIDIA GeForce RTX 5080`
  - torch: `2.10.0+cu128`
  - torch CUDA runtime: `12.8`

The hard gate passed: RTX 5080 was visible inside WSL before rollout.

## Isolated Environment

- Miniconda root: `/home/jiheon/miniconda3-official`
- Conda env: `/home/jiheon/miniconda3-official/envs/official-smolvla-libero`
- Python: `3.10.20`
- Runtime env:
  - `MUJOCO_GL=egl`
  - `LIBERO_CONFIG_PATH=/home/jiheon/.libero`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
  - `TOKENIZERS_PARALLELISM=false`

## Execution Path

- Locked assets were copied to WSL ext4 under `/home/jiheon/assets`.
- A WSL repo copy exists at `/home/jiheon/tca_map`.
- The actual smoke and pilot ran from `/mnt/c/Users/jiheo/tca_map` so the current Windows working-tree edits were used without a second sync. This has a performance risk versus ext4, but the bounded pilot still completed.

## Smoke And Pilot

- Smoke: official LIBERO `libero_spatial`, task `0`, one episode per policy, all four policies executed.
- Pilot: official LIBERO suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, task `0`, three episodes per task per policy, total `48` planned and `48` completed.
- No old custom `LIBERO_7D` replay/bridge route was imported or used.
