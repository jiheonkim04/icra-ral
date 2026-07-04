# LIBERO HDF5 Interface Audit

This report records the report-only LIBERO HDF5 demonstration interface audit.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\94_audit_libero_hdf5_interface.ps1
```

The audit reads one local LIBERO HDF5 demonstration, local SmolVLA config, and existing reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Runtime outputs are ignored:

- `reports\libero_hdf5_interface_audit_report.json`,
- `reports\libero_hdf5_interface_audit_report.md`.

## Current Local Result

Latest audit result: `no_go_rollout_scaling`.

Inspected file:

```text
C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
```

Observed HDF5 interface:

- first demo action shape: `[272, 7]`,
- action dimension: `7`,
- state-like key `obs/ee_states`: shape `[272, 6]`,
- RGB image keys: `obs/agentview_rgb`, `obs/eye_in_hand_rgb`,
- RGB image shape: `[272, 128, 128, 3]`,
- rewards/dones present.

SmolVLA config interface:

- policy action shape: `[6]`,
- policy state shape: `[6]`,
- policy image inputs: three cameras,
- policy image shape: `[3, 256, 256]` per camera.

Findings:

- High: LIBERO demonstration actions are 7D while the SmolVLA policy config action is 6D.
- Low: `obs/ee_states` is 6D and matches the SmolVLA policy state dimension.
- Medium: the HDF5 demonstration exposes fewer RGB camera streams than the policy config image inputs.
- Low: HDF5 images are 128x128 and require resizing to the policy 256x256 input.
- Medium: local task language can be matched, but checkpoint training/task provenance is still not established.
- High: the previous compatibility audit already blocks rollout scaling.

Recommended next step:

Create a report-only offline adapter reproduction check that builds SmolVLA-compatible state/image/action adapter inputs from the first HDF5 timestep and compares dimensions/ranges, without model loading or rollout.
