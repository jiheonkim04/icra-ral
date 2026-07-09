# Next Actions

Date: 2026-07-10 KST

Current decision: `NEEDS_LARGER_PREDICTION_ARTIFACT`

## Immediate Next Action

Generate the larger official SmolVLA prediction artifact under the fixed task-stratified, episode-disjoint manifest.

Exact next command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\official_smolvla_split_manifest.json -Output reports\official_smolvla_stable_prediction_artifact.json
```

## Required Boundaries

- Use official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`.
- Use the fixed manifest at `reports/official_smolvla_split_manifest.json`.
- Do not change the split after seeing prediction metrics.
- Do not tune FCAR, implement FCAR v2, or design a new method before the stable artifact report exists.
- Do not use the archived custom `LIBERO_7D` adapter route.
- Do not run OpenVLA-OFT.
- Do not run simulator rollout or full benchmark as a substitute for the fixed offline artifact report.
- Do not download additional assets unless explicitly approved.

## What Must Be Reported Next

- frozen/base official SmolVLA predictions
- standard rank-4 LoRA predictions or a clearly labeled existing rank-4 LoRA source
- mean-action prior
- MoIRA-style task/instruction router
- validation-selected static mixture, with alpha frozen before test
- frame oracle and task oracle diagnostics
- raw 7D action L2
- translation, rotation, and gripper breakdowns
- task-balanced and frame-weighted aggregates
- episode and task bootstrap intervals
- action-range validity
- help/hurt counts and per-task failure table

## Current Evidence To Preserve

- FCAR is killed by static/LoRA baselines and must not be scaled from the tiny-gate result.
- Post-FCAR robust sweep found split-dependent ranking: frozen/base won `2` folds, val-selected static mix won `3`, rank-4 LoRA beat frozen/base in `2` / `5` folds but won no realistic fold.
- Frame oracle still has headroom: mean action L2 `0.084582167` versus frozen/base `0.106514933`.
- Task oracle has tiny headroom: mean action L2 `0.106079936`.
- The stable manifest now covers all `40` eligible official tasks with train `1200`, validation `400`, and test `1200` frames.
- Final current decision remains `NEEDS_LARGER_PREDICTION_ARTIFACT` until that artifact is generated and scored under the fixed metric protocol.
