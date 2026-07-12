# SACF-VLA Prototype Protocol

Date: 2026-07-12 KST

Run order:

1. synthetic factorization smoke;
2. real-demo training;
3. Stage A closed-loop rollout only if synthetic and real-demo training pass.

Commands:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_sacf_vla.py
```

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_sacf_vla_prototype.py --mode synthetic
```

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_sacf_vla_prototype.py --mode real-demo-train
```

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_sacf_vla_prototype.py --mode stage-a
```

Expected outputs:

- `reports/sacf_vla/synthetic_result.json`
- `reports/sacf_vla/synthetic_result.md`
- `reports/sacf_vla/real_demo_train_result.json`
- `reports/sacf_vla/real_demo_train_result.md`
- `reports/sacf_vla/checkpoints/sacf_full.pt`
- `reports/sacf_vla/checkpoints/plain_bc_prefix.pt`
- `reports/sacf_vla/stage_a_partial_result.json`
- `reports/sacf_vla/stage_a_result.json`
- `reports/sacf_vla/stage_a_result.md`
