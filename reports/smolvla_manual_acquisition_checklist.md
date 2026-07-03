# SmolVLA Manual Acquisition Checklist

## Purpose

This checklist covers manual placement and verification of SmolVLA checkpoint files before any model loading. It is a local file-readiness checklist only. It does not authorize downloads, GPU jobs, training, rollouts, heavy VLA imports, or OpenVLA-OFT execution.

Actual download support still requires explicit approval in a later task. Bounded SmolVLA load-only adapter smoke is now covered by the SmolVLA autonomous pilot standing approval, and `ALLOW_HEAVY_IMPORT=1` may be set only inside that bounded load-only task.

## Target Directory

Place the SmolVLA checkpoint directory at:

```text
C:\assets\checkpoints\smolvla
```

The expected local paths are:

```text
SMOLVLA_CKPT=C:\assets\checkpoints\smolvla
CHECKPOINT_ROOT=C:\assets\checkpoints
HF_HOME=C:\assets\hf_home
```

These may be configured through environment variables or `configs/paths.local.yaml`. Do not commit `configs/paths.local.yaml`.

## Required Files

The directory under `C:\assets\checkpoints\smolvla` must contain all three groups:

- config: `config.json`
- tokenizer: one or more of `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.model`, or `sentencepiece.bpe.model`
- weights: one or more of `model.safetensors`, `pytorch_model.bin`, sharded `*.safetensors`, or sharded `*.bin`

The readiness checkers also accept `special_tokens_map.json` as tokenizer-related metadata, but a complete tokenizer normally includes one of the tokenizer files listed above.

Do not create fake marker files to pass the checker. The next real adapter smoke will require an actual compatible checkpoint.

## Verify Without Loading The Model

Use file checks only:

```powershell
Test-Path C:\assets\checkpoints\smolvla
Get-ChildItem C:\assets\checkpoints\smolvla
Test-Path C:\assets\checkpoints\smolvla\config.json
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.safetensors
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.bin
```

Then run the repository readiness checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

These checks do not download assets, train, run rollouts, import SmolVLA, or execute OpenVLA-OFT. `scripts\13_check_smolvla_adapter_smoke.ps1` only performs a lightweight local adapter-guard import and file/memory readiness checks.

## Readiness States

Path-ready means:

- `SMOLVLA_CKPT` is configured,
- `C:\assets\checkpoints\smolvla` exists,
- the checkpoint may still be empty or incomplete.

Checkpoint-complete means:

- path-ready is true,
- `config.json` exists,
- at least one tokenizer file exists,
- at least one weights file exists.

Adapter-smoke-ready means:

- checkpoint-complete is true,
- `HF_HOME` or `CHECKPOINT_ROOT` exists,
- the lightweight adapter guard import succeeds,
- the memory estimate fits the RTX 5080 16GB budget,
- a later task explicitly authorizes a load-only adapter smoke.

Current expected status before files are placed:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=false
ready_for_smolvla_adapter_smoke=false
```

Expected status after valid files are placed:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Safety Gates

Keep these unset for this checklist:

```powershell
$env:ALLOW_DOWNLOADS
$env:ALLOW_HEAVY_IMPORT
$env:ALLOW_GPU_TRAINING
$env:ALLOW_ROLLOUTS
```

Do not run:

- automatic checkpoint or dataset downloads,
- GPU jobs,
- adapter training,
- simulator rollouts,
- heavy VLA model imports,
- OpenVLA-OFT local execution.
