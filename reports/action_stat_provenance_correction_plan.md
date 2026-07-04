# Action-Stat Provenance Correction Plan

This planning-only step chooses the next safe response to the action-stat/checkpoint-provenance mismatch found by the action normalization provenance audit.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\119_plan_action_stat_provenance_correction.ps1
```

Scope:

- reads `reports/action_normalization_provenance_audit_report.json`,
- selects the next safe audit/correction step,
- performs no downloads,
- performs no installs,
- imports no heavy VLA models,
- loads no model,
- performs no inference,
- runs no training,
- runs no simulator or rollout,
- uses no GPU job,
- executes no OpenVLA-OFT,
- alters no policy behavior,
- accesses no tokens,
- makes no paper-grade claim.

Planned direction:

1. First compute LIBERO action statistics from a bounded local HDF5 subset.
2. Compare those local action stats against checkpoint processor stats.
3. Only then decide whether normalized-action-space probing, postprocessor bypass/replacement, checkpoint provenance resolution, or rollout-path blocking is justified.

Until the action-stat mismatch is resolved, learned-policy rollout scaling remains blocked.

Current local result:

- plan passed,
- decision: `reduce_scope`,
- selected next step: `libero_action_stat_subset_audit`,
- ready for LIBERO action-stat audit: true,
- rollout scaling ready: false,
- benchmark claim ready: false,
- paper claim ready: false.

The selected next step is a report-only LIBERO HDF5 action-stat subset audit. It should compute local action mean/std/range from bounded dataset files before any normalized-action probe, postprocessor bypass/replacement, new checkpoint download, rollout, or policy behavior change.
