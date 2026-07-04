# LIBERO Action-Stat Subset Audit

This report-only step computes action statistics from a bounded subset of local LIBERO HDF5 demonstration files and compares them against the checkpoint processor action statistics.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\120_audit_libero_action_stats.ps1
```

Scope:

- reads bounded local HDF5 `actions` arrays,
- samples at most 5 files by default,
- samples at most 500 actions per file by default,
- compares local LIBERO action scale/dimensionality against checkpoint processor stats,
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

If this confirms scale or dimension mismatch, learned-policy rollout scaling remains blocked and the next safe task is a normalized-action-space probe or checkpoint/task provenance resolution plan.

Current local result:

- audit passed,
- decision: `no_go_rollout_scaling`,
- sampled files: `5`,
- sampled actions: `2500`,
- LIBERO action dim: `7`,
- LIBERO max abs: `1.0`,
- checkpoint action stat prefixes: `so100`, `so100-blue`, `so100-red`,
- checkpoint action mean max abs: `125.720543`,
- checkpoint action std max: `59.359951`,
- scale mismatch confirmed: true,
- dimension mismatch confirmed: true,
- rollout scaling ready: false,
- benchmark claim ready: false,
- paper claim ready: false.

Interpretation: this confirms that the local learned-policy rollout path is not a fair performance test of SmolVLA-on-LIBERO yet. The next safe step is a normalized-action-space probe or checkpoint/task provenance resolution plan before any learned-policy rollout.
