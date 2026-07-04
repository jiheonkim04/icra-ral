# VLM-Enabled Offline Decoding Summary

This report-only step compares the previous repeated offline action-decoding diagnostic with `load_vlm_weights=false` against the bounded VLM-enabled recheck with `load_vlm_weights=true`.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\117_summarize_vlm_enabled_offline_decoding.ps1
```

Scope:

- reads existing ignored runtime JSON reports,
- reads local SmolVLA config/preprocessor/postprocessor JSON files,
- computes VLM-on versus VLM-off action-distance deltas,
- records action-normalization and adapter-provenance blockers,
- performs no downloads,
- performs no installs,
- imports no heavy VLA models,
- loads no model,
- performs no model inference,
- runs no training,
- runs no simulator or rollout,
- uses no GPU job,
- executes no OpenVLA-OFT,
- accesses no tokens,
- makes no paper-grade claim.

Current interpretation:

- VLM-enabled loading improves the tiny repeated offline action-distance metrics versus the no-VLM diagnostic.
- The alignment signal remains `weak`.
- Adapted actions still clip values.
- The policy uses ACTION `MEAN_STD` normalization and a 6D action head that is adapted to the 7D LIBERO expert-action convention.

Current local result:

- no-VLM mean action L1/MSE: `0.412322` / `0.286972`,
- VLM-enabled mean action L1/MSE: `0.301665` / `0.216188`,
- L1 percent reduction: `26.838`,
- MSE percent reduction: `24.666`,
- no-VLM alignment signal: `weak`,
- VLM-enabled alignment signal: `weak`,
- clipped values total in both runs: `3`,
- rollout scaling ready: false,
- benchmark claim ready: false,
- paper claim ready: false.

This is useful diagnostic evidence that VLM loading changes behavior, but it is not standard success, benchmark success, counterfactual robustness evidence, or paper-grade evidence. The next safe step is action-normalization/provenance diagnosis before learned-policy rollout scaling.
