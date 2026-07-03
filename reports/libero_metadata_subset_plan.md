# LIBERO Metadata Subset Plan

This stage builds a metadata-only LIBERO task manifest from the local official LIBERO source checkout.

It reads BDDL files and task lists only. It does not use demonstration trajectories, run simulators, import LIBERO/RoboSuite/MuJoCo, train, rollout, run GPU jobs, import heavy VLA models, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

## Purpose

- Validate that LIBERO task metadata can feed target/counterfactual split plumbing.
- Create a tiny manifest of task instructions, objects of interest, goals, and metadata-only counterfactual pairs.
- Keep real offline dataset smoke blocked until actual documented demo files are present under `LIBERO_DATA_ROOT`.

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File scripts\47_build_libero_metadata_subset.ps1
powershell -ExecutionPolicy Bypass -File scripts\42_plan_libero_dataset_risk.ps1
```

## Readiness Semantics

- `ready_for_metadata_only_subset`: local LIBERO source checkout and BDDL metadata are readable.
- `ready_for_real_dataset_interface_smoke`: metadata is readable and real demo/data files are detected under `LIBERO_DATA_ROOT`.
- `ready_for_rollout`: always false in this stage.

Metadata-only subset construction is useful for interface design, but it is not standard success, not offline proxy success, not rollout success, and not paper-grade evidence.
