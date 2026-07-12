# Autonomous Until Paper Resume

## 2026-07-12 KST Continuity Update

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
type reports\autonomous_ral_campaign_state.json
```

Current stage: `cycle_2_valid_kill_recorded_cycle_3_selection_pending`

Cycle 2 `FEDO-VLA` Stage A completed `70 / 70` episodes with zero exceptions and final decision `CLEAN_RETENTION_FAILURE`.

Key result:

- faulted full FEDO: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen SmolVLA: `4 / 10`
- clean FEDO full: `0 / 10`

Next automatic stage: commit and push the Cycle 2 archive if not already done, then start Cycle 3, the final permitted distinct method cycle.

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode stage-a
```

Current stage: `cycle_2_real_trace_training_passed_stage_a_ready`

Next automatic stage:

1. Run FEDO-VLA Stage A.
2. Adjudicate against static inverse-gain, APEX-style feedback, and no-feedback ablation.
3. If killed, preserve the valid result and pivot to Cycle 3 without rescuing FEDO.

Governance correction: the active target is now `PAPER_READY_EXPERIMENTAL_PACKAGE`, with at most three distinct method cycles and a 24 hour total GPU-time cap.

Checkpoint from synthetic smoke: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`

Real full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`

Real no-history checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`

DICD-VLA Stage A completed `50 / 50` episodes with zero exceptions and final decision `SIMPLE_BASELINE_EXPLAINS_METHOD`.

FEDO-VLA has passed synthetic smoke and real trace training; the Stage A command is the next resumable action.
