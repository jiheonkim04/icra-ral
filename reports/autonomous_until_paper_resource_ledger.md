# Autonomous Until Paper Resource Ledger

Date: 2026-07-12 KST

Hardware/resource constraints inherited from the objective:

- RTX 5080, approximately 16GB VRAM;
- 24GB system RAM;
- Windows 11 with verified WSL2/Linux CUDA environment;
- no remote GPU, no paid cloud, no physical robot.

Current measured local state:

- C: free bytes: `382892236800`
- active long-running command: none
- active checkpoint: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`
- epoch-1 new downloads: `0.0 GiB`
- campaign downloads inherited: `14.845 GiB` for quantized OpenVLA-OFT INT4
- GPU time in new campaign: `0.0 h`

Latest completed command:

- `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_dicd_vla_prototype.py`
- result: `DICD_SYNTHETIC_MECHANISM_SMOKE_PASSED`
- elapsed seconds: `1.093`

Next staged command:

- `wsl -e bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode real-smolvla-chunk"`
- purpose: load official SmolVLA, call `predict_action_chunk`, postprocess a real action chunk, and verify delay-index contrast.

Latest completed real SmolVLA smoke:

- result: `DICD_REAL_SMOLVLA_CHUNK_SMOKE_PASSED`
- elapsed seconds: `22.618`
- peak CUDA allocation: `926.638 MB`
- raw action chunk shape: `[1, 50, 7]`
- postprocessed probe chunk shape: `[8, 7]`

Resource policy:

- no uncontrolled CPU or disk offload;
- no simultaneous large VLA model loads;
- stop if disk free falls below `50 GiB`;
- stop after two identical CUDA OOMs;
- max single uncheckpointed command: `4 h`.
