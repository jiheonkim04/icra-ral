# Tiny Head-Only Pilot Approval Plan

## Purpose

This plan prepares the future tiny head-only ActionMap/TCA-Map offline-proxy pilot. It does not run training.

The pilot is still blocked by explicit approval gates:

- runtime package installation approval,
- real SmolVLA feature extraction heavy-import/load-only approval,
- tiny head-only training approval.

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
- max 1000 initial local steps,
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

Paper-grade standard success requires later simulator rollouts after separate simulator setup and rollout approvals.

## Safety Boundary

This planner refuses dangerous gates such as `ALLOW_GPU_TRAINING=1`, `ALLOW_TINY_TRAINING=1`, `ALLOW_HEAVY_IMPORT=1`, or `ALLOW_DOWNLOADS=1`.

It does not download assets, run GPU jobs, import heavy VLA models, load models, run inference, train, rollout, or execute OpenVLA-OFT.
