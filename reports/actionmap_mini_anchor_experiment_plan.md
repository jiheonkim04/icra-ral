# ActionMap Mini-Anchor Experiment Plan

Date: 2026-07-08

## STATE 1 Plan

Run the smallest local LIBERO/HDF5-backed ActionMap-style diagnostic:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\220_actionmap_mini_anchor_diagnostic.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The runner must stop if dangerous gates such as downloads, GPU training, rollouts, heavy imports, OpenVLA, OpenVLA-OFT, runtime install, or simulator execution are set.

## Metrics

- 7D action L2.
- Translation L2.
- Rotation L2.
- Gripper error.
- Candidate top-k accuracy.
- Candidate NLL / heatmap loss.
- Candidate diversity and collapse.
- Oracle nearest candidate upper bound.
- Per-task and per-phase breakdowns when cheap.

## Outputs

- `reports/actionmap_mini_anchor_state1_result.md`
- `reports/actionmap_mini_anchor_state1_result.json`
- `reports/actionmap_mini_anchor_autopilot_state.md`
- `reports/actionmap_mini_anchor_decision_log.md`
- `reports/actionmap_mini_anchor_risk_register.md`

## Hard Stop

Do not proceed to Target-Grounded ActionMap in this run, even if the mini-anchor passes.
