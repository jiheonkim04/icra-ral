# SmolVLA Asset Setup And Adapter Smoke Plan

## Purpose

SmolVLA is the first real-adapter smoke target for the local low-compute path. The smoke is interface validation only. It is not a training result, not a rollout result, and not paper-grade evidence.

Paper-grade evidence still requires real benchmark data, clear baselines, and simulator rollouts later. Offline proxy and load/interface checks must stay labeled as engineering checks.

OpenVLA-OFT large local execution remains forbidden. Do not run OpenVLA-OFT locally except for a separately approved frozen/load smoke task.

## Safety Policy

Default behavior is dry-run and local-only. No downloads happen by default.

- no downloads unless `ALLOW_DOWNLOADS=1` is explicitly set,
- no heavy VLA imports unless `ALLOW_HEAVY_IMPORT=1` is explicitly set,
- no GPU training,
- no rollouts,
- no LIBERO dependency for SmolVLA adapter smoke readiness,
- no tokens in committed files,
- no OpenVLA-OFT execution.

The current scripts do not implement automatic downloading. If future download support is added, it must require `ALLOW_DOWNLOADS=1`. If future heavy model import support is added, it must require `ALLOW_HEAVY_IMPORT=1`. The current scripts prepare directories, print setup instructions, and check local checkpoint readiness.

## Required Local Paths

Use environment variables or `configs/paths.local.yaml`:

```powershell
$env:SMOLVLA_CKPT="C:\assets\checkpoints\smolvla"
$env:CHECKPOINT_ROOT="C:\assets\checkpoints"
$env:HF_HOME="C:\assets\hf_home"
```

Equivalent YAML:

```yaml
assets:
  smolvla_ckpt: "C:/assets/checkpoints/smolvla"
  checkpoint_root: "C:/assets/checkpoints"
  hf_home: "C:/assets/hf_home"
```

## Expected Checkpoint Contents

An empty SmolVLA checkpoint directory is only **path-ready**. It is not **adapter-smoke-ready**. The real-asset checker reports these states separately so a placeholder directory cannot be mistaken for a usable checkpoint.

The smoke checker looks for:

- `config.json`,
- at least one tokenizer file such as `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.model`, or `sentencepiece.bpe.model`,
- at least one weights file such as `model.safetensors`, `pytorch_model.bin`, sharded `*.safetensors`, or sharded `*.bin`.

Actual config, tokenizer, and weights files are required before any later load-only adapter smoke. Creating `C:\assets\checkpoints\smolvla` by itself is not enough.

Exact SmolVLA file names may differ by release. If the checker reports missing files but the local checkpoint is valid, update the expected-file list in `configs/smolvla_smoke.yaml` and the checker scripts.

## Commands

Dry-run asset plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\12_prepare_smolvla_assets.ps1
```

Create local asset directories only when you explicitly want directories created:

```powershell
$env:ALLOW_CREATE_DIRS="1"
powershell -ExecutionPolicy Bypass -File scripts\12_prepare_smolvla_assets.ps1
```

Check SmolVLA adapter smoke readiness without downloads, training, rollouts, LIBERO, heavy VLA imports, or OpenVLA-OFT:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

Linux/WSL equivalents:

```bash
bash scripts/12_prepare_smolvla_assets.sh
bash scripts/13_check_smolvla_adapter_smoke.sh
```

## Go / No-Go For Later Real Adapter Smoke

Go only if:

- `SMOLVLA_CKPT` resolves to a local directory,
- `HF_HOME` or `CHECKPOINT_ROOT` is configured,
- config/tokenizer/weights files are present,
- estimated memory fits RTX 5080 16GB with headroom,
- a later task explicitly authorizes the actual load-only adapter smoke.

No-go if:

- any checkpoint file is missing,
- a download would be required at runtime,
- the memory estimate exceeds the local budget,
- the task would require training, rollouts, LIBERO, OpenVLA-OFT, or a heavy import without an explicit gate.
