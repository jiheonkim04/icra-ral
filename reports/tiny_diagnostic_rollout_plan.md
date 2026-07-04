# Tiny Diagnostic Rollout Plan

This report defines the planning-only gate after bounded import, render, and reset/step smoke pass.

Run the planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\62_plan_tiny_diagnostic_rollout.ps1
```

Scope:

- planning only,
- requires a passed bounded reset/step smoke report,
- estimates a future tiny diagnostic rollout envelope,
- does not execute rollout,
- does not create LIBERO/RoboSuite environments,
- does not run policy inference, training, GPU jobs, downloads, heavy VLA imports, OpenVLA-OFT, token access, or paper claims.

The planner may say the risk envelope is internally bounded, but execution remains false. A future rollout task must be a separate branch, task-local gated, and compatible with the current user policy.
