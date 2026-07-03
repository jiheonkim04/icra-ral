# Hard-Stop Approval Status

## Purpose

This report explains the current bounded-autopilot stop condition in one place.

The project has completed the safe scaffolds, planners, readiness checks, dummy cache path, cached-feature eval-only smoke, tiny head-only pilot planner, and the explicitly approved SmolVLA runtime package install. The next meaningful steps now require explicit approval.

## Summary Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

It writes an ignored runtime report:

```text
reports\hard_stop_status_report.json
```

The script is summary-only. It does not install packages, download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, access tokens, execute simulators, execute OpenVLA-OFT, or make paper-grade claims.

## Current Approval Choices

The runtime install gate has been used and is currently clear: `torch`, `transformers`, `lerobot`, and `safetensors` are installed in the `tca_map` environment.

Approve at most one remaining gate at a time:

1. SmolVLA load-only heavy-import task with `ALLOW_HEAVY_IMPORT=1`, after runtime dependencies are present.
2. Tiny head-only training task with `ALLOW_TINY_TRAINING=1`, after cached features are real and runtime checks are valid.

Do not combine heavy import/model loading and training in the same approval unless a later task explicitly narrows and justifies that combined scope.

## Current Known Blockers

- Runtime dependencies are present in the current `tca_map` environment.
- Actual SmolVLA load-only construction requires `ALLOW_HEAVY_IMPORT=1`.
- Any tiny head-only training requires explicit training approval.
- LIBERO/RoboSuite/RoboCasa/simulator assets remain missing for rollout work.
- OpenVLA-OFT local large execution remains forbidden.

## Safe Commands Still Allowed

These remain safe routine checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```
