# Tiny Head-Only Pilot Plan

## Purpose

This plan prepares the future tiny head-only ActionMap/TCA-Map offline-proxy pilot. It does not run training.

The tiny smoke version is now covered by the SmolVLA autonomous pilot risk envelope if risk assessment is green and it stays inside the bounded budget:

- dummy or tiny local non-paper data only,
- frozen backbone,
- max 300 steps after stable smaller smoke,
- max 30 minutes,
- max 14GB VRAM,
- no rollout,
- no OpenVLA-OFT,
- no multi-seed,
- no paper claim.

## Planner

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

The planner writes an ignored runtime report:

```text
reports\tiny_head_only_pilot_plan_report.json
```

It verifies that the ActionMap and TCA-Map head-only configs remain within the local compute policy:

- frozen backbone,
- cached features,
- batch size 1 style execution,
- max 1000 initial local pilot steps in config, with the local autonomous smoke capped by risk assessment,
- trainable parameters under the initial 50M limit,
- grid size 8,
- low-resolution heatmaps,
- no full backbone tuning,
- no rollouts,
- no OpenVLA-OFT execution,
- no multi-seed sweep.

## Metric Naming

Offline proxy metrics are not standard success. Reports must use names such as:

```text
offline_standard_proxy
standard_proxy_score
```

Paper-grade standard success requires later simulator rollouts after simulator setup and rollout risk assessments pass.

## Safety Boundary

This planner refuses execution gates such as `ALLOW_GPU_TRAINING=1`, `ALLOW_TINY_TRAINING=1`, `ALLOW_HEAVY_IMPORT=1`, or `ALLOW_DOWNLOADS=1` because it is planning-only.

It does not download assets, run GPU jobs, import heavy VLA models, load models, run inference, train, rollout, or execute OpenVLA-OFT.

## Bounded Smoke Runner

The bounded runner is:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

It writes:

```text
reports\tiny_head_only_smoke_report.json
```

This runner trains tiny CPU NumPy heads on cached/dummy feature records only. It refuses work outside the local risk budget, refuses download/heavy-import/GPU/rollout gates, and never loads SmolVLA or OpenVLA-OFT.

Passing this runner validates only cached-feature head optimization and offline proxy metric plumbing. It is not standard success and not a paper-grade empirical result.
