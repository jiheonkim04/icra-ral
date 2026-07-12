# Autonomous Until Paper Resume

## 2026-07-12 KST Continuity Update

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
type reports\autonomous_ral_campaign_state.json
```

Current stage: `epoch_1_completed_pivot_required`

Cycle 2 `FEDO-VLA` Stage A completed `70 / 70` episodes with zero exceptions and final decision `CLEAN_RETENTION_FAILURE`.

Key result:

- faulted full FEDO: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen SmolVLA: `4 / 10`
- clean FEDO full: `0 / 10`

Cycle 3 `GCAP-VLA` Stage A completed `70 / 70` episodes with zero exceptions and final decision `NO_OCCLUSION_ROBUSTNESS_GAIN`.

Key Cycle 3 result:

- occluded frozen SmolVLA: `4 / 10`
- Sobel edge boost: `5 / 10`
- GCAP no-temporal ablation: `4 / 10`
- GCAP full under occlusion: `3 / 10`
- clean frozen SmolVLA: `1 / 10`
- clean GCAP full: `5 / 10`

Corrected decision: `EPOCH_1_COMPLETED_PIVOT_REQUIRED`.

Final state inspection command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/autonomous-until-paper-governance-v2
type reports\current_research_governance.md
```

## Archived Prior Resume Block

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/ral-cycle-02-fedo-vla
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_fedo_vla_prototype.py --mode stage-a
```

Archived stage: `cycle_2_real_trace_training_passed_stage_a_ready`

Archived next automatic stage at that time:

1. Run FEDO-VLA Stage A.
2. Adjudicate against static inverse-gain, APEX-style feedback, and no-feedback ablation.
3. If killed, preserve the valid result and pivot to Cycle 3 without rescuing FEDO.

Governance correction: the active target is now `PAPER_READY_EXPERIMENTAL_PACKAGE`, with at most three distinct method cycles and a 24 hour total GPU-time cap.

Checkpoint from synthetic smoke: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`

Real full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`

Real no-history checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`

DICD-VLA Stage A completed `50 / 50` episodes with zero exceptions and final decision `SIMPLE_BASELINE_EXPLAINS_METHOD`.

FEDO-VLA has passed synthetic smoke and real trace training; the Stage A command is the next resumable action.
