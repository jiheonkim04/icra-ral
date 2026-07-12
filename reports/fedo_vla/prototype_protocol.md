# FEDO-VLA Prototype Protocol

Date: `2026-07-12 KST`

Command sequence:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode synthetic
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode real-trace-train
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode stage-a
```

Required artifacts:

- `reports/fedo_vla/synthetic_result.json`
- `reports/fedo_vla/real_trace_train_result.json`
- `reports/fedo_vla/stage_a_result.json`
- `reports/fedo_vla/stage_a_partial_result.json`
- `reports/fedo_vla/checkpoints/fedo_full.pt`
- `reports/fedo_vla/checkpoints/fedo_no_feedback.pt`

No thresholds may be changed after seeing results.
