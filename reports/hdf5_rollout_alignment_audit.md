# HDF5-to-Rollout Alignment Audit

This report defines a report-only audit for the compatibility gap between the local LIBERO HDF5 demonstration and the learned-policy rollout bridge.

The key question is whether the offline first-action reproduction evidence is being applied to the same task and initial-state convention as the rollout diagnostic. If the HDF5 demonstration has an `init_state` and state trajectory but the rollout bridge only calls `env.reset()`, then a successful offline first-action reproduction does not by itself justify rerunning the same learned-policy rollout variant.

`scripts/97_audit_hdf5_rollout_alignment.ps1` reads existing ignored reports, one local HDF5 demonstration, and the rollout bridge source. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected outcomes:

- `decision=reduce_scope`: the task appears aligned but the HDF5 initial-state/replay convention is not established, so the next step should be a planning-only HDF5 initial-state or first-action replay diagnostic.
- `decision=stop`: report inputs are missing, task names are inconsistent, HDF5 cannot be read, or execution gates are set.
- `decision=proceed`: no report-only HDF5/rollout alignment blocker was found, but rollout scaling still remains blocked until a separate positive diagnostic signal exists.

This audit does not establish standard success, benchmark success, counterfactual robustness, SOTA, or paper-grade evidence.
