# Next Actions

## Current Next Action

Manually place a valid SmolVLA-compatible checkpoint under:

```text
C:\assets\checkpoints\smolvla
```

Required minimum file groups:

- `config.json`
- one tokenizer file such as `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.model`, or `sentencepiece.bpe.model`
- one weights file such as `model.safetensors`, `pytorch_model.bin`, sharded `*.safetensors`, or sharded `*.bin`

## After Files Are Placed

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Still Forbidden

Until explicitly approved later:

- no downloads,
- no GPU jobs,
- no training,
- no rollouts,
- no heavy VLA imports,
- no OpenVLA-OFT execution,
- no token or secret handling in committed files.

## Later Task

After readiness is true, create a new branch for a SmolVLA load-only adapter smoke. That later branch should remain load/interface-only and must not train, rollout, or run OpenVLA-OFT.
