# Autonomous RA-L Resource Ledger

Current hardware:

- NVIDIA GeForce RTX 5080
- approximately 16 GiB VRAM
- Windows 11 with WSL2 official SmolVLA/LIBERO environment

Current runtime:

- WSL Python: `/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python`
- repository path in WSL: `/mnt/c/Users/jiheo/tca_map`
- no new downloads in this reopened DICD cycle
- no CPU or disk offload planned

Stage A command:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode stage-a
```

Stage A partial checkpoint after repair:

- `reports/dicd_vla/stage_a_partial_result.json`

Stage A completed artifacts:

- `reports/dicd_vla/stage_a_result.json`
- `reports/dicd_vla/stage_a_result.md`
- `reports/dicd_vla/stage_a_stdout.log`
- `reports/dicd_vla/stage_a_stderr.log`

Observed resource envelope during checkpointed Stage A:

- GPU memory by `nvidia-smi`: approximately `3.24 GiB / 16.3 GiB`
- CUDA allocation reported by the result JSON: peak approximately `926.638 MiB`
- WSL RAM during polling: approximately `4.1-4.2 GiB / 11 GiB`
- disk free on `C:` during polling: approximately `357 GiB`
- new downloads during this cycle: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed
