# LIBERO HDF5 Reader Dependency Plan

The official LIBERO demonstrations are HDF5 files. The offline interface smoke gate can detect the files without extra packages, but it cannot inspect instruction/action-like fields unless `h5py` is available in the project Python environment.

Check the dependency without installing anything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\50_check_libero_hdf5_reader.ps1
```

This checker is read-only. It does not install packages, download data, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims.

Current semantics:

- `ready_for_libero_hdf5_interface_read=true`: `h5py` is available and `scripts\48_plan_libero_offline_interface_smoke.ps1` can inspect HDF5 files.
- `ready_for_libero_hdf5_interface_read=false`: LIBERO data may be present, but offline dataset interface smoke remains blocked at the reader dependency gate.
- `ready_for_libero_rollout=false`: rollout readiness is never inferred from paths, dataset files, or HDF5 reader availability.

If `h5py` is missing, run a separate dependency risk assessment before any install. Installing `h5py` must not change CUDA/PyTorch versions, must not install simulator stacks, and must not trigger training, rollouts, OpenVLA-OFT, or heavy VLA imports.
