# SmolVLA Checkpoint Acquisition Plan

## Purpose

This document plans safe SmolVLA checkpoint acquisition for the SmolVLA-first low-compute path. It is a planning and guard document only. It is not a download log, not an adapter smoke result, not a training result, and not paper-grade evidence.

No repository script in this stage downloads checkpoints or datasets. Any future download-capable script must require `ALLOW_DOWNLOADS=1`, must avoid committing tokens, and must still keep OpenVLA-OFT large local execution forbidden.

## Required Local Layout

Expected local checkpoint path:

```text
C:\assets\checkpoints\smolvla
```

Expected cache path:

```text
C:\assets\hf_home
```

Expected checkpoint root:

```text
C:\assets\checkpoints
```

Configure these paths with environment variables or `configs/paths.local.yaml`. Do not commit `configs/paths.local.yaml`.

PowerShell environment example:

```powershell
$env:SMOLVLA_CKPT="C:\assets\checkpoints\smolvla"
$env:CHECKPOINT_ROOT="C:\assets\checkpoints"
$env:HF_HOME="C:\assets\hf_home"
```

`configs/paths.local.yaml` example:

```yaml
assets:
  smolvla_ckpt: "C:/assets/checkpoints/smolvla"
  checkpoint_root: "C:/assets/checkpoints"
  hf_home: "C:/assets/hf_home"
```

## Required Checkpoint Files

Adapter-smoke readiness requires all three groups below:

- config: `config.json`
- tokenizer: `tokenizer.json` or `tokenizer_config.json` or `vocab.json` or `merges.txt` or `tokenizer.model` or `sentencepiece.bpe.model`
- weights: `model.safetensors` or `pytorch_model.bin` or sharded `*.safetensors` or sharded `*.bin`

An empty `C:\assets\checkpoints\smolvla` directory is only path-ready. It is not adapter-smoke-ready.

## Manual Acquisition Options

Use a manual, authenticated workflow outside this repository to place a SmolVLA-compatible checkpoint under `SMOLVLA_CKPT`. Acceptable options include:

- copying a checkpoint directory from another local disk,
- using an organization-approved Hugging Face CLI workflow outside this script,
- restoring a checkpoint directory from an internal artifact store,
- mounting a read-only checkpoint directory and pointing `SMOLVLA_CKPT` at it.

Never paste tokens into committed files. Keep Hugging Face tokens, WANDB keys, and private credentials outside the repository.

## Dry-Run Guard

Run the planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
```

Linux/WSL:

```bash
bash scripts/14_plan_smolvla_download.sh
```

The planner prints intended `SMOLVLA_CKPT`, `CHECKPOINT_ROOT`, and `HF_HOME` values, required file groups, and whether `ALLOW_DOWNLOADS` is set. It performs no downloads, creates no directories, imports no heavy VLA models, runs no GPU jobs, runs no training, runs no rollouts, and does not execute OpenVLA-OFT.

If `ALLOW_DOWNLOADS=1` is set, the current planner still performs no downloads. That gate is documented only for a later explicitly approved download-capable task.

## Verification After Manual Placement

After files are manually placed, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

Go for a later separately approved load-only adapter smoke only if:

- `ready_for_smolvla_path_check=true`,
- `smolvla_checkpoint_files_present=true`,
- `ready_for_smolvla_adapter_smoke=true`,
- `HF_HOME` or `CHECKPOINT_ROOT` exists,
- the RTX 5080 16GB memory estimate leaves headroom,
- the later task explicitly authorizes the load-only smoke.

No-go if any required file group is missing, if a runtime download would be needed, if a heavy import is not explicitly authorized, or if the task drifts into training, rollouts, LIBERO evaluation, or OpenVLA-OFT execution.
