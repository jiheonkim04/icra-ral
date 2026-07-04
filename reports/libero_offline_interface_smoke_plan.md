# LIBERO Offline Interface Smoke Gate

This gate checks whether a tiny local LIBERO-style data file is present and structurally readable for offline interface work.

It is not standard success, not rollout success, not training, and not paper-grade evidence. It performs no downloads, GPU jobs, simulator execution, rollouts, heavy VLA imports, token access, OpenVLA-OFT execution, or paper claims.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\48_plan_libero_offline_interface_smoke.ps1
```

## Semantics

- `ready_for_offline_interface_smoke=true`: at least one local tiny file has readable instruction/action-like fields.
- `ready_for_offline_interface_smoke=false`: no usable tiny data file is present, or HDF5 data are present but the `h5py` reader dependency is unavailable; metadata-only work may continue, but real offline dataset smoke remains blocked.
- `ready_for_rollout=false`: this gate never clears rollout readiness.

For official LIBERO HDF5 files, first check the reader dependency:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\50_check_libero_hdf5_reader.ps1
```

The current expected local state is `stop` if `h5py` is missing, even when official HDF5 demo files are present under `LIBERO_DATA_ROOT`.
