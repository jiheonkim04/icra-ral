# Autonomous Until Paper Resource Ledger

## 2026-07-12 KST Continuity Update

Cycle 2 `FEDO-VLA` Stage A completed `70 / 70` episodes with zero exceptions.

FEDO Stage A artifacts:

- `reports/fedo_vla/stage_a_partial_result.json`
- `reports/fedo_vla/stage_a_result.json`
- `reports/fedo_vla/stage_a_result.md`
- `reports/fedo_vla/stage_a_stdout.log`
- `reports/fedo_vla/stage_a_stderr.log`

Observed FEDO Stage A resource envelope:

- rollout elapsed time: `1879.48 s`
- GPU memory by `nvidia-smi` during polling: approximately `3.1-5.0 GiB / 16.3 GiB`
- WSL RAM during polling: approximately `3.7-4.2 GiB / 11 GiB`
- swap during polling: `0 B`
- new downloads during Cycle 2: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed

Approximate governed campaign GPU time consumed so far: `3.24 h`.

Cycle 3 `GCAP-VLA` pre-Stage-A artifacts:

- `reports/gcap_vla/synthetic_result.json`
- `scripts/run_gcap_vla_prototype.py`
- `tca_map/smolvla/gcap_vla.py`
- `tests/test_gcap_vla.py`

Synthetic result: `SYNTHETIC_MECHANISM_PASS`.

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

Next staged command:

- `wsl -e bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode real-trace-train"`
- purpose: collect real frozen SmolVLA traces on training identity `20260711`, train full/no-history DICD adapters, persist checkpoints, and probe identity `20260712`.

Latest completed real trace training:

- result: `DICD_REAL_TRACE_TRAINING_PASSED`
- elapsed seconds: `76.581`
- full examples: `312`
- no-history examples: `312`
- full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`
- full checkpoint sha256: `36d6c14bacf7bd3992d530fd428557175e626229eafca41b2449302ff5cb4538`
- no-history checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`
- no-history checkpoint sha256: `a0dc6b8a0b5e7db14896549d4bd2f60368751316d7f421671cd45eeab3c364d0`

Resource policy:

- no uncontrolled CPU or disk offload;
- no simultaneous large VLA model loads;
- stop if disk free falls below `50 GiB`;
- stop after two identical CUDA OOMs;
- max single uncheckpointed command: `4 h`.
