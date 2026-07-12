# Autonomous Until Paper Resume

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/auto-method-20260712-01-dicd-vla
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode stage-a
```

Current stage: `cycle_1_stage_a_rollout_checkpointed_running`

Next automatic stage:

1. Implement Stage A delayed-rollout variants if not already present.
2. Run Stage A closed-loop evaluation on identities `20260713` through `20260717`.
3. Classify DICD-VLA according to the preregistered criteria.

Governance correction: the active target is now `PAPER_READY_EXPERIMENTAL_PACKAGE`, with at most three distinct method cycles and a 24 hour total GPU-time cap.

Checkpoint from synthetic smoke: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`

Real full checkpoint: `reports/dicd_vla/checkpoints/dicd_real_full.pt`

Real no-history checkpoint: `reports/dicd_vla/checkpoints/dicd_real_no_history.pt`

The Stage A command is the next resumable action.
