# Init-State Learned-Policy Recheck Plan

This report defines the planning gate for a future bounded learned-policy LIBERO recheck that starts from the local HDF5 demonstration initial state.

Allowed future scope after a green plan:

- one `libero_10` task,
- one local HDF5 demonstration initial state,
- at most 5 policy-controlled steps in the first runner,
- WSL CPU simulator plus local SmolVLA policy topology,
- task-local `ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1`,
- diagnostic/local-pilot evidence label only.

Forbidden scope:

- no training,
- no GPU job by default,
- no downloads or installs,
- no OpenVLA-OFT,
- no multi-seed evaluation,
- no benchmark, SOTA, or paper-grade claim.

Passing this planner does not establish policy success. It only authorizes a separately gated narrow compatibility recheck that uses the HDF5 initial-state convention already validated by `scripts/100_bounded_hdf5_initial_state_replay.ps1`.
