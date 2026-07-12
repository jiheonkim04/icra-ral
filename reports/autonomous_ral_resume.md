# Autonomous RA-L Resume

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
type reports\autonomous_ral_campaign_state.json
```

Current stage: `cycle_2_valid_kill_recorded_cycle_3_selection_pending`

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

Next automatic stage: commit and push the Cycle 2 archive if not already done, then start Cycle 3, the final permitted distinct method cycle.

No paper-ready terminal decision has been reached.
