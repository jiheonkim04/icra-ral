# Feature Cache Eval Smoke Plan

## Purpose

This eval-only smoke validates that cached feature records can feed the TCA-Map head and offline proxy metric path. It uses dummy cached features only.

It is not training, not model inference, and not a paper-grade result.

## Command

Create a dummy cache if needed and run the eval-only smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache
```

The script writes an ignored report:

```text
reports\feature_cache_eval_report.json
```

The dummy cache lives under ignored `runs\feature_cache\dummy_contract`.

## Safety Boundary

The script refuses dangerous gates such as `ALLOW_HEAVY_IMPORT=1` and does not download assets, run GPU jobs, import heavy VLA models, load models, run VLA inference, train, rollout, run simulators, or execute OpenVLA-OFT.

Real SmolVLA feature extraction remains blocked until runtime packages and the heavy-import/load-only gate are explicitly approved.
