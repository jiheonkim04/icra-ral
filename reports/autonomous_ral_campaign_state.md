# Autonomous RA-L Campaign State

Date: 2026-07-12 KST

Target terminal state: `PAPER_READY_EXPERIMENTAL_PACKAGE`

Governance correction applied:

- maximum distinct method cycles: `3`
- maximum total GPU time: `24 h`
- maximum wall time per method cycle: `12 h`
- maximum single uncheckpointed command: `4 h`
- no routine user approvals for bounded local research actions

Current cycle: `1`

Current method: `DICD-VLA`

Current branch: `codex/auto-method-20260712-01-dicd-vla`

Prompt branch alias to create/preserve after commit: `codex/ral-cycle-01-dicd-vla`

Current stage: `cycle_1_stage_a_rollout_ready_to_run`

Next command:

```powershell
wsl -e bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_dicd_vla_prototype.py --mode stage-a"
```

The Stage A runner compiles and `tests/test_dicd_vla.py` passes.
