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
