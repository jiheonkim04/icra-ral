# Pure Action/State/Image Adapter Helpers

This report records the pure helper layer for the SmolVLA-to-LIBERO interface.

The helper module is:

- `tca_map\smolvla\interface_adapters.py`

It defines:

- explicit 6D-policy to 7D-environment action adaptation with named gripper strategies,
- explicit state vector construction from named observation fields and slices,
- explicit image feature to RoboSuite observation key alias selection.

The helpers do not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Rollout wiring remains a later step. The current stage is unit-test-only interface logic.

## Current Local Result

Latest local result: pure adapter unit tests passed.

Confirmed behavior:

- 6D policy actions can be adapted to 7D environment actions only through a named gripper strategy,
- unsupported action dimension mappings raise an error,
- state adaptation uses explicit fields and refuses silent truncation or padding,
- missing state/image keys raise clear errors,
- image aliases report the selected RoboSuite observation key.

## Single-Sample Metadata Wiring

The next safe wiring step records adapter metadata in the synthetic single-sample interface smoke. This still does not run a simulator or rollout.

Current local wiring result: bounded synthetic single-sample smoke passed with `adapter_metadata_recorded=true`, diagnostic adapted action dim `7`, and no simulator, rollout, training, GPU training, OpenVLA-OFT execution, token access, or paper-grade claim.
