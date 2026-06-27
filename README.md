# TCA-Map

Target-Conditioned ActionMap for counterfactual grounding in vision-language-action policies.

This repository starts with a conservative scaffold for a two-week kill-or-continue pilot. The immediate milestone is not a full paper run. It is:

1. scaffold,
2. preflight,
3. dummy smoke test,
4. one real adapter smoke test only when local paths exist,
5. one tiny offline pilot later,
6. go/no-go report.

## Safety policy

- No automatic downloads of OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, checkpoints, or datasets.
- No GPU training in the scaffold step.
- No real rollouts until simulator paths pass preflight.
- Missing assets should not block dummy smoke or interface validation.
- Offline proxy metrics are engineering validation only and must not be described as final standard success.

Heavy actions require explicit environment gates:

- `ALLOW_DOWNLOADS=1`
- `ALLOW_HEAVY_IMPORT=1`
- `ALLOW_GPU_TRAINING=1`
- `ALLOW_ROLLOUTS=1`
- `ALLOW_CLOUD_HANDOFF=1`

## Local-first execution

On Windows PowerShell from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts/00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts/04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/05_eval_smoke.ps1
```

The shell script `scripts/00_preflight.sh` is included for Linux/WSL users, but PowerShell preflight is the supported first path on Windows.

## Real asset readiness

The current recommendation is **SmolVLA-first** for the first real-adapter smoke on an RTX 5080 16GB local machine. OpenVLA-OFT remains the primary paper-grade baseline target, but full OpenVLA-OFT fine-tuning should not be attempted locally.

Read the plan:

```powershell
Get-Content reports/real_asset_setup_plan.md
```

Check local paths without downloads, heavy model imports, GPU jobs, or rollouts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/11_check_real_assets.ps1
```

Linux/WSL equivalent:

```bash
bash scripts/11_check_real_assets.sh
```

## Path to Paper-Grade Experiments Without Leaving Home

1. Local Windows scaffold validation: run tree check, preflight, dummy train/eval smoke, pytest, asset checks, and system readiness checks.
2. WSL2/Linux rollout setup: use `scripts/24_wsl2_setup_check.ps1`, then install Ubuntu manually if needed and validate GPU visibility from WSL2.
3. SmolVLA-first local smoke: configure `SMOLVLA_CKPT`, `HF_HOME`, and `CHECKPOINT_ROOT`, then run readiness checks. Model execution remains a later approved task.
4. Small local rollout: after WSL2/Linux, LIBERO, RoboSuite, and data paths pass checks, run only a separately approved tiny rollout task.
5. OpenVLA-OFT frozen smoke: keep OpenVLA-OFT as the paper-grade baseline target, but attempt only frozen/load smoke locally and only after memory checks pass.
6. Cloud/remote GPU handoff: use `scripts/23_cloud_handoff_manifest.*` to prepare a manifest for 24GB, 48GB, or 80GB GPU classes depending on baseline scale.

Planner scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/20_system_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts/21_make_asset_dirs.ps1
powershell -ExecutionPolicy Bypass -File scripts/22_plan_local_experiment_matrix.ps1
powershell -ExecutionPolicy Bypass -File scripts/23_cloud_handoff_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts/24_wsl2_setup_check.ps1
```

## Asset configuration

Copy `configs/paths.local.yaml.example` to `configs/paths.local.yaml` and fill in local paths if available. The local file is ignored by git.

Environment variables with equivalent meaning are also supported:

- `OPENVLA_OFT_CKPT`
- `SMOLVLA_CKPT`
- `LIBERO_ROOT`
- `LIBERO_DATA_ROOT`
- `ROBOSUITE_ROOT`
- `DATA_ROOT`
- `CHECKPOINT_ROOT`
- `HF_HOME`
- `WANDB_API_KEY`

## Current scaffold contents

The Python package contains lightweight, dependency-minimal dummy components for interface validation:

- dummy LIBERO-style dataset samples,
- counterfactual target-swap generation,
- dummy VLA adapter,
- ActionMap-style heatmap head,
- target-conditioned TCA-Map head,
- offline proxy metrics,
- preflight and smoke entrypoints.
