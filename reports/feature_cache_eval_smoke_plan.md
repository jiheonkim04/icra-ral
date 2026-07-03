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

## Latest Local Result

The eval-only cached-feature smoke passed on the dummy cache:

```text
cache_valid=true
cache_record_count=4
offline_standard_proxy=0.157986
target_top1_accuracy=0.25
wrong_target_proxy_rate=0.75
downloads_performed=false
gpu_jobs_performed=false
heavy_model_imports_performed=false
model_inference_performed=false
training_performed=false
rollouts_performed=false
openvla_oft_executed=false
```
