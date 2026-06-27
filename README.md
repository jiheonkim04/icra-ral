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

## Local-first execution

On Windows PowerShell from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts/00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts/04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/05_eval_smoke.ps1
```

The shell script `scripts/00_preflight.sh` is included for Linux/WSL users, but PowerShell preflight is the supported first path on Windows.

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
