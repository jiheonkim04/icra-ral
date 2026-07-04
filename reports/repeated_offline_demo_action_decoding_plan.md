# Repeated Offline Demonstration Action Decoding Plan

This report defines the next bounded step after weak one-sample offline SmolVLA action decoding and the VLM/action-normalization audit.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\109_plan_repeated_offline_demo_action_decoding.ps1
```

The planner reads the VLM/action audit, one existing offline decoding report, and local HDF5 metadata. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

Expected interpretation:

- `decision=proceed`: a future separately gated runner may run at most three CPU SmolVLA action decodes on local HDF5 timesteps.
- `ready_for_bounded_repeated_offline_demo_action_decoding_runner=true`: implement the bounded runner only under `ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1`.
- `ready_for_rollout_scaling=false`: repeated offline decoding is still not simulator or benchmark evidence.

The future runner must log:

- `load_vlm_weights`,
- policy/adapted/expert action previews,
- action L1/MSE to expert action,
- policy 6D L1 to expert first 6 dimensions,
- clipped action count,
- gripper strategy,
- image source aliases.

This step is meant to decide whether the weak one-sample action alignment is stable across a few demonstration timesteps. It does not justify rollout scaling by itself.
