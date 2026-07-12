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
wsl -e bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode stage-a"
```
