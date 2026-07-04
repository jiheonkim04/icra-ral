# Offline Adapter Reproduction Check

This report records the report-only offline adapter reproduction check from the first local LIBERO HDF5 timestep.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\95_check_offline_adapter_reproduction.ps1
```

The check reads one local LIBERO HDF5 demonstration, local SmolVLA config, pure adapter helpers, and existing reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Runtime outputs are ignored:

- `reports\offline_adapter_reproduction_check_report.json`,
- `reports\offline_adapter_reproduction_check_report.md`.

## Current Local Result

Latest check result: `no_go_rollout_scaling`.

Inspected file:

```text
C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
```

Observed first demonstration action:

```text
[0.0, 0.05625, -0.01875, 0.0, 0.0, -0.0, -1.0]
```

Adapter reproduction:

- `policy_6d_delta_pose_plus_gripper_close`: first-action L1 to demo `0.0`,
- `policy_6d_delta_pose_plus_gripper_zero_hold`: first-action L1 to demo about `0.142857`,
- `policy_6d_delta_pose_plus_gripper_open`: larger mismatch for the first action,
- best first-action adapter strategy: `policy_6d_delta_pose_plus_gripper_close`.

State reproduction:

- HDF5 `obs/ee_pos + obs/ee_ori` exactly reproduces HDF5 `obs/ee_states`,
- this gives a 6D state vector matching the SmolVLA policy state shape.

Image reproduction:

- HDF5 provides `agentview_rgb` and `eye_in_hand_rgb`,
- the policy config expects three image inputs,
- camera3 remains an alias/duplication choice unless a documented deployment camera contract is found.

Interpretation:

The current zero-hold gripper default is not demonstration-informed for this first local LIBERO trajectory. A future one-task diagnostic may test `policy_6d_delta_pose_plus_gripper_close` as a specific compatibility hypothesis, but rollout scaling remains blocked.
