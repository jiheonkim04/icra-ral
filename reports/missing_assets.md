# Missing Assets Setup Guide

This tracked file is a static setup guide. Runtime preflight status is written to `reports/missing_assets_runtime.json`, which is ignored by git.

The scaffold policy is local paths only:

- Do not download OpenVLA-OFT, SmolVLA, LIBERO, RoboCasa, checkpoints, or datasets automatically.
- Missing assets do not block scaffold, dummy smoke, or interface validation.
- Real adapter GPU work and simulator rollouts stay skipped until paths are configured and checks pass.

## Environment variables

Set only the paths you have locally. PowerShell examples:

```powershell
$env:OPENVLA_OFT_CKPT="C:\path\to\openvla-oft\checkpoint"
$env:SMOLVLA_CKPT="C:\path\to\smolvla\checkpoint"
$env:LIBERO_ROOT="C:\path\to\LIBERO"
$env:LIBERO_DATA_ROOT="C:\path\to\libero\data"
$env:ROBOSUITE_ROOT="C:\path\to\robosuite"
$env:DATA_ROOT="C:\path\to\data"
$env:CHECKPOINT_ROOT="C:\path\to\checkpoints"
$env:HF_HOME="C:\path\to\hf_home"
```

Command Prompt examples:

```bat
set OPENVLA_OFT_CKPT=C:\path\to\openvla-oft\checkpoint
set SMOLVLA_CKPT=C:\path\to\smolvla\checkpoint
set LIBERO_ROOT=C:\path\to\LIBERO
set LIBERO_DATA_ROOT=C:\path\to\libero\data
set ROBOSUITE_ROOT=C:\path\to\robosuite
set DATA_ROOT=C:\path\to\data
set CHECKPOINT_ROOT=C:\path\to\checkpoints
set HF_HOME=C:\path\to\hf_home
```

## Local YAML config

Alternatively, copy the template and edit it:

```powershell
Copy-Item configs\paths.local.yaml.example configs\paths.local.yaml
notepad configs\paths.local.yaml
```

Example contents:

```yaml
assets:
  openvla_oft_ckpt: "C:/path/to/openvla-oft/checkpoint"
  smolvla_ckpt: "C:/path/to/smolvla/checkpoint"
  libero_root: "C:/path/to/LIBERO"
  libero_data_root: "C:/path/to/libero/data"
  robosuite_root: "C:/path/to/robosuite"
  data_root: "C:/path/to/data"
  checkpoint_root: "C:/path/to/checkpoints"
  hf_home: "C:/path/to/hf_home"
  wandb_api_key: null
```

`configs/paths.local.yaml` is ignored by git because it can contain local machine paths or tokens.

## After configuration

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
```

Preflight will still refuse real GPU pilot work unless required assets and memory checks pass.
