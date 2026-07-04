# LIBERO Offline Counterfactual Split Plan

This plan builds a tiny local counterfactual split manifest from already-acquired LIBERO assets.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\51_build_libero_offline_counterfactual_split.ps1
```

Scope:

- reads local BDDL task metadata from `LIBERO_ROOT`,
- reads local HDF5 structure from `LIBERO_DATA_ROOT`,
- matches task ids to `*_demo.hdf5` files,
- creates target/action counterfactual pairs only when both sides have local demo files,
- labels all outputs as offline proxy only.

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

Passing this gate only means the local dataset can provide tiny HDF5-backed counterfactual split plumbing. It is not standard success, not rollout success, and not paper-grade evidence.
