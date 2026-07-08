# ActionMap Mini-Anchor Kill Criteria

Date: 2026-07-08

## Continue To Target-Grounded ActionMap Only If

- Oracle candidate upper bound clearly beats mean action, linear/L1, and simple MLP.
- ActionMap-style head beats mean action.
- ActionMap-style head beats linear/L1.
- ActionMap-style head is not matched by the simple MLP.
- ActionMap-style head does not collapse to trivial candidate selection.
- Real LIBERO/HDF5-backed metrics are produced.

## Kill Or Stop If

- Mean action matches or beats ActionMap-style head.
- Linear/L1 matches or beats ActionMap-style head.
- Simple MLP matches or beats ActionMap-style head.
- Oracle candidate upper bound is weak.
- No real LIBERO/HDF5-backed metric appears.
- The run requires full official ActionMap reproduction, large assets, GPU, downloads, OpenVLA-OFT, large VLA training, simulator rollout, or target-grounded method implementation.

## Current Boundary

This gate can support only one conclusion: whether the local ActionMap-style heatmap substrate is useful enough to justify a later Target-Grounded ActionMap feasibility pass.

## Archived Outcome

Final decision: `KILL_ACTIONMAP_ANCHOR`

Triggered criteria:

- mean-action action L2 `0.466767673` beat ActionMap-style action L2 `0.529931357`;
- simple MLP action L2 `0.501926707` matched or beat ActionMap-style action L2 `0.529931357`;
- candidate top1 was `0.018518519`;
- candidate diversity collapsed to unique translation/rotation/gripper bins `5 / 1 / 2`.

Positive but non-decisive evidence: oracle nearest-candidate action L2 was `0.065653208`, which shows candidate-space headroom but is invalid as learned method evidence.

Consequence: Target-Grounded ActionMap cannot proceed from this local mini-anchor result. Revival requires official ActionMap reproduction or a stronger non-collapsed heatmap implementation that first clears the same simple-baseline gate.
