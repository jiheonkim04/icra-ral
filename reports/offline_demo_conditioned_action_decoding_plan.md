# Offline Demonstration-Conditioned Action Decoding Plan

This planning gate prepares a one-sample offline check before any additional learned-policy rollout. The goal is to test whether local SmolVLA action decoding from a real LIBERO demonstration observation is closer to the expert action than the zero-reward rollout diagnostics suggest.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\105_plan_offline_demo_conditioned_action_decoding.ps1
```

This planner reads existing report JSON, local checkpoint file presence, and the selected HDF5 path only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

If it reports `decision=proceed`, a future separately gated runner may perform exactly one CPU SmolVLA action decoding call on one local HDF5 demonstration observation under `ALLOW_OFFLINE_DEMO_ACTION_DECODING=1`. That runner must still avoid simulator environments, rollouts, training, downloads, OpenVLA-OFT, and paper-grade claims.
