# LIBERO Offline ActionMap vs TCA-Map Comparison Plan

This plan adds a tiny real/offline proxy comparison over local LIBERO HDF5 action snippets.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\52_compare_libero_offline_actionmap_tca.ps1
```

Scope:

- reads `reports\libero_offline_counterfactual_split_report.json`,
- reads only a tiny number of action rows from local HDF5 demos,
- compares deterministic proxy arms for ActionMap, TCA-Map, and TCA-Map + Distributional TCA-Select plumbing,
- reports offline proxy metrics only.

This is not a trained baseline, not standard success, not rollout success, and not paper-grade evidence.

Forbidden behavior:

- no downloads,
- no GPU jobs,
- no training,
- no simulator execution,
- no rollouts,
- no heavy VLA imports,
- no OpenVLA-OFT execution,
- no token access,
- no paper-grade claims.
