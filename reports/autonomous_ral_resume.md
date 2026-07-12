# Autonomous RA-L Resume

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode stage-a
```

Current stage: `cycle_2_real_trace_training_passed_stage_a_ready`

Completed Stage A artifacts:

- `reports/dicd_vla/stage_a_result.json`
- `reports/dicd_vla/stage_a_result.md`
- `reports/dicd_vla/stage_a_partial_result.json`

Cycle 1 DICD-VLA decision: `SIMPLE_BASELINE_EXPLAINS_METHOD`.

Reason: the direct chunk-index delay baseline reached `2 / 10`, while full DICD reached `1 / 10`; the no-history ablation also reached `1 / 10`. The mechanism was active but did not improve closed-loop task success.

Cycle 2 FEDO-VLA status:

- synthetic mechanism smoke: `SYNTHETIC_MECHANISM_PASS`
- real SmolVLA trace training: `REAL_TRACE_TRAIN_PASS`
- focused tests: `tests/test_fedo_vla.py` and `tests/test_dicd_vla.py` pass

Next automatic stage: run FEDO-VLA Stage A with checkpointed partial result at `reports/fedo_vla/stage_a_partial_result.json`.

No paper-ready terminal decision has been reached.
