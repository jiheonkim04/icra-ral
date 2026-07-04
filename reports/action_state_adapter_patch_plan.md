# Action/State Adapter Patch Plan

This report defines the next patch plan after the metadata audit and zero-action versus SmolVLA-action comparison.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\80_plan_action_state_adapter_patch.ps1
```

The planner reads existing reports and local SmolVLA metadata only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

The patch plan is motivated by four confirmed diagnostic risks:

- SmolVLA emits 6D actions while the LIBERO/RoboSuite environment expects 7D actions,
- the current bridge pads the missing gripper component with `0.0`,
- the current state builder flattens several observation keys and silently truncates to policy state dim 6,
- camera feature names differ between config and preprocessor metadata.

## Required Patch Direction

1. Replace implicit action padding/truncation with an explicit action adapter.
2. Replace silent state truncation with an explicit state adapter.
3. Add camera feature alias auditing or an explicit image-key adapter.
4. Keep every result labeled diagnostic/local-pilot until benchmark protocol is implemented.

The next implementation should start with pure adapter functions and unit tests, then rerun single-sample and bounded diagnostic checks before any rollout scaling.

## Current Local Result

Latest local planner result: `proceed`.

Required adapters:

- action adapter: required,
- state adapter: required,
- camera alias adapter: required.

Rollout scaling remains blocked. The next safe step is pure action/state/image adapter helpers with unit tests. No simulator, model load, GPU job, training, rollout, OpenVLA-OFT execution, token access, or paper-grade claim is needed for that step.
