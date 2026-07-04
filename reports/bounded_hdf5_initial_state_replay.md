# Bounded HDF5 Initial-State Replay Diagnostic

This report defines the first executable HDF5 replay diagnostic.

The runner in `scripts/100_bounded_hdf5_initial_state_replay.ps1` is separately gated by `ALLOW_HDF5_REPLAY_DIAGNOSTIC=1`. It uses one local LIBERO HDF5 demonstration, sets the demonstration initial state if supported, and replays only the first demonstration action in the first runner.

Allowed scope:

- one task,
- one HDF5 demonstration,
- one replay step,
- WSL CPU simulator topology,
- no learned-policy loading or inference,
- no training,
- no GPU jobs,
- no downloads or installs,
- no OpenVLA-OFT,
- no benchmark, SOTA, or paper-grade claim.

Passing this runner means the local HDF5 initial-state/action replay convention is operational. It does not establish policy success, benchmark success, standard performance, counterfactual robustness, or paper-grade evidence.
