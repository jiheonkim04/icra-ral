# Repeated Offline Demonstration Action Decoding

This bounded runner follows the repeated offline decoding plan and compares local SmolVLA actions against at most three expert HDF5 actions from one LIBERO demonstration.

Command:

```powershell
$env:ALLOW_REPEATED_OFFLINE_DEMO_DECODING="1"
powershell -ExecutionPolicy Bypass -File scripts\110_bounded_repeated_offline_demo_action_decoding.ps1
Remove-Item Env:\ALLOW_REPEATED_OFFLINE_DEMO_DECODING -ErrorAction SilentlyContinue
```

Scope:

- loads local SmolVLA on CPU,
- runs at most three `select_action` calls,
- reads local HDF5 observations/actions only,
- records action L1/MSE, clipping, gripper strategy, `load_vlm_weights`, and image aliases,
- does not create simulator environments,
- does not rollout,
- does not train,
- does not download,
- does not use GPU jobs,
- does not execute OpenVLA-OFT,
- does not make paper-grade claims.

Interpretation:

- weak repeated offline alignment keeps learned-policy rollout scaling blocked,
- moderate or strong repeated offline alignment is still diagnostic-only and needs a separate tiny baseline/offline comparison before rollout scaling,
- this report is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.

Current local result:

- execution passed,
- decoded timesteps: `0`, `136`, and `271`,
- sample count: 3,
- mean action L1 to expert: `0.412322`,
- max action L1 to expert: `0.478394`,
- mean action MSE to expert: `0.286972`,
- mean policy-6D L1 to expert first 6 dimensions: `0.608221`,
- clipped action values total: 3,
- `load_vlm_weights=false`,
- offline alignment signal: `weak`,
- rollout scaling ready: false,
- paper-grade claim ready: false.

This result reinforces the no-go decision for learned-policy rollout scaling. The next safe research direction is VLM-enabled loading risk/provenance and action-normalization analysis, not another rollout variant.
