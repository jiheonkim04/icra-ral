# Feature Cache Interface Plan

## Purpose

This plan defines a lightweight feature-cache contract for head-only ActionMap/TCA-Map development before any real SmolVLA feature extraction. It uses dummy hidden tokens only.

The feature-cache scaffold is not a paper-grade result and is not evidence that SmolVLA works locally. It is an interface contract for later frozen-backbone feature extraction after runtime install and within the risk-assessed bounded SmolVLA pilot policy.

## Files

The cache format is:

```text
manifest.json
features.jsonl
```

Each record stores:

- `sample_id`,
- `dataset_version`,
- `instruction`,
- `target`,
- `distractor`,
- `expert_action`,
- `hidden_tokens`,
- `hidden_dim`,
- policy flags proving no downloads, GPU jobs, heavy imports, model loading, inference, training, rollouts, or OpenVLA-OFT execution were performed.

## Safe Planner

Run the planner without writing a cache:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
```

Write a dummy cache for interface validation only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1 -WriteDummyCache
```

The report is ignored by git:

```text
reports\feature_cache_plan_report.json
```

The dummy cache is written under `runs\`, which is also ignored by git.

## Real SmolVLA Boundary

Real SmolVLA feature extraction remains blocked until all of these are explicitly approved and valid:

- runtime packages are installed and checked,
- `ALLOW_HEAVY_IMPORT=1` is set only inside green risk-assessed bounded load-only behavior,
- no inference/training/rollout is performed,
- memory policy remains within the local RTX 5080 16GB budget,
- no OpenVLA-OFT execution occurs.

The dummy cache scaffold is safe to use for downstream head and metric interface tests.

## Latest Local Result

The dummy feature-cache interface validation passed:

```text
cache_valid=true
record_count=4
hidden_dims=[12]
downloads_performed=false
gpu_jobs_performed=false
heavy_model_imports_performed=false
model_inference_performed=false
training_performed=false
rollouts_performed=false
openvla_oft_executed=false
```
