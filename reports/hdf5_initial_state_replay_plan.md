# HDF5 Initial-State Replay Plan

This report defines the planning gate for a bounded HDF5 replay diagnostic.

The previous HDF5-to-rollout alignment audit found that the selected HDF5 demonstration and learned-policy rollout use the same task, but the rollout bridge does not establish the HDF5 demonstration initial state. A replay diagnostic should test the simulator/data convention before another learned-policy rollout.

`scripts/98_plan_hdf5_initial_state_replay.ps1` reads local reports, one local HDF5 demonstration, and local LIBERO/RoboSuite source files. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

If the planner reports `decision=proceed`, the next runner should be separately gated with `ALLOW_HDF5_REPLAY_DIAGNOSTIC=1` and should:

- use one task and one HDF5 demonstration,
- set the HDF5 `init_state` or flattened state if supported,
- replay only the first demonstration action in the first runner,
- avoid learned-policy inference and model loading,
- avoid training, GPU jobs, OpenVLA-OFT, multi-seed evaluation, and paper claims.

This plan is simulator/data compatibility work only. It is not standard success, benchmark success, counterfactual robustness, SOTA evidence, or paper-grade evidence.
