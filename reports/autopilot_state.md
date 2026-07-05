# Autopilot State

- current main commit at branch start: `5ff7835 add fixed-prior rollout readiness gate`
- branch: `codex/fixed-prior-7d-action-bridge`
- git status at state update: modified branch files pending validation/commit
- attempted: State 1 bridge fix, State 2 readiness gate rerun, and State 3 limited fixed-prior rollout diagnostic
- succeeded: original HDF5 `7D` LIBERO actions are preserved for rollout candidates without silent padding; readiness gate is `green`
- rollout happened: `true`, bounded diagnostic only
- training happened: `false`
- LoRA training happened: `false`
- loss computed: `false`
- GPU/download/heavy import/OpenVLA-OFT happened: `false`
- simulator environment created: `true` during the bounded fixed-prior rollout diagnostic
- rollout result: `1` task, `3` variants, `10` steps each, `30` total steps
- success/reward: all variants had reward `0.0` and success `false`
- action bridge result: same-dim `7D` passthrough, no clipping, gripper dimension preserved from HDF5
- fixed-prior diagnostic result: fixed-prior TCA moved the EEF more than the ActionMap-style mean baseline, but did not improve reward or success
- exact next state decision: treat this as partial action-bridge support only; do not claim rollout success. Next milestone should diagnose whether short-horizon HDF5 replay can produce reward/success or target-directed movement before scaling rollout.
