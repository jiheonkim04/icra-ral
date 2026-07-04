# VLM-Enabled Repeated Offline Decoding Plan

This planning stage decides whether to repeat the tiny offline LIBERO HDF5 action-decoding diagnostic after VLM-enabled SmolVLA construction has passed.

Planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\115_plan_vlm_enabled_repeated_offline_decoding.ps1
```

The planner compares:

- prior repeated offline decoding with `load_vlm_weights=false`,
- the bounded VLM-enabled load-only smoke with `load_vlm_weights=true`,
- the existing selected HDF5 timesteps.

It is planning-only. It does not load models, run inference, train, rollout, use GPU jobs, download, install, execute OpenVLA-OFT, access tokens, or make paper claims.

If green, the future runner may decode at most three local HDF5 timesteps on CPU with VLM weights enabled. That future result is still offline diagnostic evidence, not standard success, rollout evidence, or paper-grade evidence.
