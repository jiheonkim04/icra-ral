# Init-State Recheck Metric Summary

This report defines a summary-only comparison between prior reset-only learned-policy diagnostics and the HDF5-init-state learned-policy recheck.

The command in `scripts/103_generate_init_state_recheck_metric_summary.ps1` reads existing ignored JSON reports only. It does not load models, infer, create simulator environments, rollout, train, use GPU, download, install, execute OpenVLA-OFT, or make benchmark/SOTA/paper-grade claims.

The expected interpretation is conservative:

- wrapper/execution pass means the bridge ran,
- task success and reward determine whether there is a positive diagnostic signal,
- rollout scaling remains blocked when success is false and reward is zero.
