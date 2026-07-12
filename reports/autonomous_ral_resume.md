# Autonomous RA-L Resume

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
type reports\autonomous_ral_campaign_state.json
```

Updated resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-03-gcap-vla
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_gcap_vla_prototype.py --mode stage-a
```

Current stage: `epoch_1_completed_pivot_required`

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
- Stage A closed-loop rollout: completed `70 / 70`
- Stage A decision: `CLEAN_RETENTION_FAILURE`
- faulted full FEDO: `1 / 10`
- strongest faulted baseline: `static_inverse_gain`, `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen SmolVLA: `4 / 10`
- clean FEDO full: `0 / 10`
- exceptions: `0`

Cycle 3 GCAP-VLA status:

- proposal hash: `C5A9BA15A608A5EAA93C49409C56B0F6F8EE0A59D103F646E720FD514238F655`
- synthetic mechanism smoke: `SYNTHETIC_MECHANISM_PASS`
- focused tests: `tests/test_gcap_vla.py`, `tests/test_fedo_vla.py`, and `tests/test_dicd_vla.py` pass
- Stage A closed-loop rollout: completed `70 / 70`
- Stage A decision: `NO_OCCLUSION_ROBUSTNESS_GAIN`
- occluded frozen SmolVLA: `4 / 10`
- Sobel edge boost: `5 / 10`
- GCAP no-temporal ablation: `4 / 10`
- GCAP full under occlusion: `3 / 10`
- clean frozen SmolVLA: `1 / 10`
- clean GCAP full: `5 / 10`
- exceptions: `0`

Corrected decision: `EPOCH_1_COMPLETED_PIVOT_REQUIRED`.

No paper-ready package or RA-L acceptance claim is made. Continue to Epoch 2 under `reports/current_research_governance.md`.
