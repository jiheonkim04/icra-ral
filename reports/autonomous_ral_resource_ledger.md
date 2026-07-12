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

Cycle 2 FEDO pre-Stage-A artifacts:

- synthetic result: `reports/fedo_vla/synthetic_result.json`
- real trace training result: `reports/fedo_vla/real_trace_train_result.json`
- full checkpoint: `reports/fedo_vla/checkpoints/fedo_full.pt`
- no-feedback checkpoint: `reports/fedo_vla/checkpoints/fedo_no_feedback.pt`

Observed FEDO real-trace training resource envelope:

- CUDA allocation reported by result JSON: peak approximately `926.638 MiB`
- new downloads during Cycle 2: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed

FEDO Stage A completed artifacts:

- `reports/fedo_vla/stage_a_partial_result.json`
- `reports/fedo_vla/stage_a_result.json`
- `reports/fedo_vla/stage_a_result.md`
- `reports/fedo_vla/stage_a_stdout.log`
- `reports/fedo_vla/stage_a_stderr.log`

Observed FEDO Stage A resource envelope:

- rollout elapsed time: `1879.48 s`
- GPU memory by `nvidia-smi` during polling: approximately `3.1-5.0 GiB / 16.3 GiB`
- CUDA allocation reported by the result JSON: peak approximately `926.638 MiB`
- WSL RAM during polling: approximately `3.7-4.2 GiB / 11 GiB`
- swap during polling: `0 B`
- new downloads during Cycle 2: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed

Cycle 2 final result:

- final decision: `CLEAN_RETENTION_FAILURE`
- Stage A episodes: `70 / 70`
- exceptions: `0`

Cycle 3 GCAP pre-Stage-A artifacts:

- proposal: `reports/gcap_vla/researcher_proposal.md`
- reviewer attack: `reports/gcap_vla/reviewer_attack.md`
- preregistration: `reports/gcap_vla/preregistration.md`
- synthetic result: `reports/gcap_vla/synthetic_result.json`
- runner: `scripts/run_gcap_vla_prototype.py`
- module: `tca_map/smolvla/gcap_vla.py`
- tests: `tests/test_gcap_vla.py`

Observed GCAP synthetic resource envelope:

- elapsed time: `0.01 s`
- new downloads during Cycle 3: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed

GCAP Stage A completed artifacts:

- `reports/gcap_vla/stage_a_partial_result.json`
- `reports/gcap_vla/stage_a_result.json`
- `reports/gcap_vla/stage_a_result.md`
- local ignored logs: `reports/gcap_vla/stage_a_stdout.log`, `reports/gcap_vla/stage_a_stderr.log`

Observed GCAP Stage A resource envelope:

- rollout elapsed time: `1748.648 s`
- CUDA allocation reported by the result JSON: peak approximately `929.638 MiB`
- new downloads during Cycle 3: `0 GiB`
- repeated CUDA OOM: none
- uncontrolled CPU or disk offload: not observed

Cycle 3 final result:

- final decision: `NO_OCCLUSION_ROBUSTNESS_GAIN`
- Stage A episodes: `70 / 70`
- exceptions: `0`
- terminal campaign decision: `NO_METHOD_AFTER_3_VALID_CYCLES`
