# GCAP-VLA Prototype Protocol

Date: 2026-07-12 KST

Command:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_gcap_vla_prototype.py --mode stage-a
```

Partial checkpoint:

`reports/gcap_vla/stage_a_partial_result.json`

Final artifacts:

- `reports/gcap_vla/stage_a_result.json`
- `reports/gcap_vla/stage_a_result.md`

The runner checkpoints after every episode and preserves official SmolVLA queue semantics.
