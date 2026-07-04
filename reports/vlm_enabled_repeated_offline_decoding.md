# VLM-Enabled Repeated Offline Decoding

This bounded runner repeats the tiny offline LIBERO HDF5 action-decoding diagnostic after local SmolVLA has been proven to construct with `load_vlm_weights=true`.

Command:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\116_bounded_vlm_enabled_repeated_offline_decoding.ps1
Remove-Item Env:\ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

Scope:

- CPU-only,
- at most three local HDF5 timesteps,
- local SmolVLA with VLM weights enabled,
- offline action-distance metrics only,
- no simulator environment,
- no rollout,
- no training,
- no GPU job,
- no downloads or installs,
- no OpenVLA-OFT,
- no token access,
- no paper-grade claim.

This result must be compared against the previous `load_vlm_weights=false` repeated offline diagnostic before deciding whether any further learned-policy rollout is justified.

Current local result:

- passed,
- decoded timesteps `0`, `136`, and `271`,
- `load_vlm_weights=true`,
- mean action L1/MSE to expert: `0.301665` / `0.216188`,
- previous no-VLM mean action L1/MSE: `0.412322` / `0.286972`,
- delta versus previous no-VLM: `-0.110657` L1 and `-0.070784` MSE,
- offline alignment signal: `weak`,
- clipped values total: `3`,
- CUDA max allocated: `0MB`,
- no simulator, rollout, training, download, GPU job, OpenVLA-OFT, token access, or paper-grade claim.

Interpretation: VLM-enabled loading improves the tiny offline distance metrics, but not enough to unblock rollout scaling. The next safe step is a report-only VLM-on/off summary and action-normalization/provenance analysis.
