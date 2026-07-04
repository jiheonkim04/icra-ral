# Environment-Policy Compatibility Audit

This report records the report-only compatibility audit after the learned-policy diagnostic synthesis returned `no_go_rollout_scaling`.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\93_audit_environment_policy_compatibility.ps1
```

The audit reads local configs, LIBERO task metadata, rollout source, policy-loader source, and existing diagnostic reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Runtime outputs are ignored:

- `reports\environment_policy_compatibility_audit_report.json`,
- `reports\environment_policy_compatibility_audit_report.md`.

## Current Local Result

Latest audit result: `no_go_rollout_scaling`.

High-severity findings:

- Task/checkpoint alignment: the local SmolVLA config does not record LIBERO task-suite provenance or a confirmed match to the selected `libero_10` diagnostic task.
- VLM loading policy: the local learned-policy diagnostic path keeps `load_vlm_weights=false` through the policy loader. This is acceptable for smoke diagnostics but may remove or weaken visual-language grounding needed for task success.
- Action convention: the policy config exposes a 6D action while LIBERO/RoboSuite reports a 7D environment action interface, so current execution relies on a diagnostic gripper adapter whose deployment semantics are not proven.
- Diagnostic ladder result: zero-action comparison, adapter strategy, action scale, prompt format, camera source, and state sufficiency all completed without nonzero reward or diagnostic success.

Medium-severity finding:

- Observation convention: the policy expects a 6D state and three 256x256 images. The diagnostic bridge adapts LIBERO observations into that contract, but the correct state/camera convention remains unproven.

Observed local metadata:

- SmolVLA config type: `smolvla`,
- state shape: `[6]`,
- action shape: `[6]`,
- max action dim: `32`,
- chunk size: `50`,
- action steps: `50`,
- tokenizer max length: `48`,
- LIBERO_10 BDDL files detected: `10`,
- LIBERO_10 HDF5 files detected: `10`,
- first diagnostic task language: `turn on the stove and put the moka pot on it`.

Recommended next step:

Create a bounded offline demonstration interface audit: inspect one LIBERO HDF5 file for action dimensions, action ranges, observation keys, camera shapes, and language/task alignment without model loading or simulator rollout.
