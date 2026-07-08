# Project State

Date: 2026-07-08

Branch:

`codex/actionmap-mini-anchor-gate`

Current decision:

`KILL_ACTIONMAP_ANCHOR`

## Execution Boundary For This Pass

- Experiments happened: yes, bounded ActionMap mini-anchor diagnostic only.
- Training happened: yes, tiny CPU NumPy action heads only.
- Loss computation happened: yes.
- Rollout/replay happened: no.
- Downloads happened: no.
- GPU use happened: no.
- OpenVLA-OFT happened: no.
- Full official ActionMap reproduction happened: no.
- Target-Grounded ActionMap implementation happened: no.
- Large VLA training happened: no.

## Repository Start State

- Starting branch: `main`.
- Starting latest commit: `8500db2 Consolidate research reset direction`.
- Working branch created: `codex/actionmap-mini-anchor-gate`.

## STATE 0

Created concise anchor-gate docs:

- `reports/actionmap_mini_anchor_task_definition.md`
- `reports/actionmap_mini_anchor_experiment_plan.md`
- `reports/actionmap_mini_anchor_kill_criteria.md`
- `reports/actionmap_mini_anchor_autopilot_state.md`

## STATE 1

Ran:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\220_actionmap_mini_anchor_diagnostic.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Generated:

- `reports/actionmap_mini_anchor_state1_result.md`
- `reports/actionmap_mini_anchor_state1_result.json`
- `reports/actionmap_mini_anchor_autopilot_state.md`
- `reports/actionmap_mini_anchor_decision_log.md`
- `reports/actionmap_mini_anchor_risk_register.md`

## Dataset And Split

- Local data root: `C:\assets\data\libero`
- Usable demos: `8`
- Train/eval split: `deterministic_per_demo_time_holdout`
- Train/eval records: `1008 / 432`
- Real LIBERO/HDF5-backed metric appeared: yes.

## Key Metrics

| Variant | Action L2 |
| --- | ---: |
| Mean action | `0.466767673` |
| Linear/L1 action head | `0.812610317` |
| Simple MLP action head | `0.501926707` |
| ActionMap-style heatmap/candidate head | `0.529931357` |
| Oracle nearest candidate upper bound | `0.065653208` |

Additional ActionMap-style diagnostics:

- candidate top1: `0.018519`
- translation top3: `0.111111`
- rotation top3: `0.981481`
- heatmap NLL: `8.41813`
- unique translation/rotation/gripper bins: `5 / 1 / 2`

## Conclusion

The oracle candidate upper bound has strong headroom, but the learned ActionMap-style head does not exploit it. Mean action beats the ActionMap-style head, cheap MLP matches/beats it, and candidate collapse is detected.

Final decision:

`KILL_ACTIONMAP_ANCHOR`
