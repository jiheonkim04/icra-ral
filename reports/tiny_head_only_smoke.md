# Tiny Head-Only Smoke Runner

## Purpose

This runner performs a bounded engineering smoke only. It trains tiny NumPy ActionMap and TCA-Map heads over cached feature records to validate the head-only training interface.

It is not a paper-grade result, not standard success, and not a simulator rollout.

## Command

Run only inside the bounded SmolVLA autonomous pilot envelope:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The script writes an ignored runtime report:

```text
reports\tiny_head_only_smoke_report.json
```

## Safety Boundary

The runner enforces:

- max 100 training steps,
- max 900 seconds runtime,
- cached/dummy features only,
- frozen backbone by construction,
- CPU NumPy heads only,
- no downloads,
- no GPU jobs,
- no heavy VLA imports,
- no SmolVLA/OpenVLA model loading,
- no VLA inference,
- no rollout or simulator execution,
- no paper-grade claims.

The runner refuses to start unless `ALLOW_TINY_TRAINING=1` is set for that task. It also refuses dangerous gates such as `ALLOW_DOWNLOADS=1`, `ALLOW_HEAVY_IMPORT=1`, `ALLOW_GPU_TRAINING=1`, `ALLOW_ROLLOUTS=1`, or `ALLOW_RUNTIME_INSTALL=1`.

## Interpretation

A passing smoke means only that cached hidden-token records can drive bounded head-only optimization and offline proxy metric plumbing. It does not prove task success. Real paper-grade standard success still requires later simulator rollouts after LIBERO/RoboSuite setup and a green rollout risk assessment.
