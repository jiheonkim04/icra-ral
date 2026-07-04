# Bounded Init-State Learned-Policy Recheck

This report defines the separately gated runner for the first learned-policy recheck that uses the validated local HDF5 demonstration initial-state convention.

The runner in `scripts/102_bounded_init_state_learned_policy_recheck.ps1` must run only under task-local `ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1`.

Allowed scope:

- one `libero_10` task,
- one local HDF5 demonstration initial state,
- at most five policy-controlled steps,
- WSL CPU simulator plus local SmolVLA policy topology,
- local files only,
- diagnostic/local-pilot evidence label only.

Forbidden scope:

- no downloads or installs,
- no training,
- no GPU job by default,
- no OpenVLA-OFT,
- no multi-seed evaluation,
- no benchmark, SOTA, or paper-grade claim.

Passing this runner means only that the learned-policy rollout bridge can execute from a documented HDF5 initial-state convention. It does not establish benchmark success or policy quality.
