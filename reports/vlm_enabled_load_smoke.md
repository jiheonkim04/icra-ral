# VLM-Enabled Load Smoke

This bounded runner tests only whether local SmolVLA can be constructed with `load_vlm_weights=true` using the local SmolVLM2 dependency files.

Command:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_VLM_ENABLED_LOAD_SMOKE="1"
powershell -ExecutionPolicy Bypass -File scripts\114_bounded_vlm_enabled_load_smoke.ps1
Remove-Item Env:\ALLOW_VLM_ENABLED_LOAD_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

Scope:

- CPU-first,
- load-only,
- no `select_action`,
- no training,
- no rollout,
- no simulator execution,
- no GPU job,
- no downloads or installs,
- no OpenVLA-OFT,
- no token access,
- no paper-grade claim.

Passing this runner is engineering evidence only. It does not prove action alignment, manipulation success, benchmark success, or counterfactual robustness. If it passes, the next safe step is to plan a tiny repeated offline action-decoding recheck with `load_vlm_weights=true`, still without rollouts.
