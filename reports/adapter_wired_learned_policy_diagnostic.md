# Adapter-Wired Learned-Policy Diagnostic

This report records the first bounded learned-policy LIBERO diagnostic after the rollout bridge was wired to the explicit action, state, and image adapter helpers.

Command sequence:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1
$env:ALLOW_BOUNDED_LEARNED_POLICY_MATRIX="1"; powershell -ExecutionPolicy Bypass -File scripts\75_bounded_reduced_scope_learned_policy_rollout.ps1; Remove-Item Env:\ALLOW_BOUNDED_LEARNED_POLICY_MATRIX -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File scripts\76_generate_reduced_scope_rollout_metric_summary.ps1
powershell -ExecutionPolicy Bypass -File scripts\79_compare_zero_action_policy_diagnostic.ps1
```

Evidence level: reduced-scope learned-policy diagnostic.

This is not standard success, benchmark success, counterfactual robustness evidence, SOTA evidence, or paper-grade evidence.

## Current Local Result

The bounded adapter-wired diagnostic passed as an execution wrapper:

- task suite: `libero_10`,
- task count: 1,
- steps: 10,
- policy calls: 10,
- diagnostic success count: 0,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- last environment action max abs: about 0.793,
- last environment action L2: about 1.222,
- gripper component: 0.0,
- mean recorded final-step policy latency: about 0.150 seconds.

The runtime policy flags stayed inside the bounded diagnostic envelope:

- downloads performed: false,
- training performed: false,
- GPU jobs performed: false,
- benchmark rollouts performed: false,
- OpenVLA-OFT executed: false,
- token access: false,
- paper-grade claims made: false.

## Adapter Metadata

The updated metric summary confirms explicit adapter metadata is present:

- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- state adapter: `diagnostic_eef_pos_quat_xyz_6d_state_adapter`,
- image source mapping:
  - `observation.images.camera1 -> agentview_image`,
  - `observation.images.camera2 -> robot0_eye_in_hand_image`,
  - `observation.images.camera3 -> agentview_image`,
- implicit action padding performed: false,
- action truncation performed: false,
- state implicit padding performed: false,
- state silent truncation performed: false,
- zero-image fallback performed: false.

## Interpretation

The explicit adapter wiring did not create task success on the selected diagnostic task. It also did not improve reward over the prior zero-action diagnostic.

The next safe research step is not another pure wiring patch. It is adapter-strategy and action-scale diagnosis:

- compare gripper strategies,
- inspect action scale/normalization and step magnitude,
- verify language prompt format,
- verify image/camera mapping,
- inspect whether the diagnostic state adapter is enough for this policy topology.
