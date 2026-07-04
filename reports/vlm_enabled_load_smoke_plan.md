# VLM-Enabled Load Smoke Plan

This planning stage decides whether a later bounded SmolVLA load-only smoke may enable the local VLM dependency path.

The relevant local dependency is:

```text
C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct
```

The planner is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\113_plan_vlm_enabled_load_smoke.ps1
```

It is planning-only. It does not download files, install packages, import heavy models, load models, run inference, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

If the plan is green, a future runner may be created with both task-local gates:

```text
ALLOW_HEAVY_IMPORT=1
ALLOW_VLM_ENABLED_LOAD_SMOKE=1
```

The future runner must stay CPU-first, load-only, no inference, no rollout, no training, no OpenVLA-OFT, no token access, and no paper claim. It exists only to test whether `load_vlm_weights=true` can be constructed locally before revisiting offline action-decoding diagnostics.
