# ActionMap Anchor Kill Criteria

## Continue Criteria

Continue to STATE 2 failure mining only if all are true:

- a real local LIBERO HDF5-backed metric is produced,
- ActionMap-style heatmap/candidate head beats mean-action on held-out 7D action L2,
- ActionMap-style heatmap/candidate head beats the linear/L1 action head on held-out 7D action L2,
- cheap MLP does not match or beat the ActionMap-style head,
- candidate predictions do not collapse to one translation or rotation bin,
- oracle nearest candidate upper bound has clear headroom.
- no official full reproduction, method extension, or failure mining is needed to pass the gate.

## Kill Criteria

Kill or reframe if any are true:

- mean-action baseline matches or beats ActionMap-style head,
- linear/L1 action head matches or beats ActionMap-style head,
- cheap MLP matches or beats ActionMap-style head,
- oracle nearest candidate upper bound is weak, meaning the candidate grid itself lacks headroom,
- no real LIBERO/HDF5-backed metric appears,
- only docs/planning are produced without diagnostic metrics,
- implementation becomes broad planner-only work,
- implementation starts becoming official full ActionMap reproduction,
- reproduction requires full VLA training, OpenVLA-OFT, heavy imports, GPU jobs, or downloads outside the current scope.

## Evidence Boundary

STATE 1 can only support a mini-anchor feasibility decision. It cannot support standard success, SOTA, official ActionMap reproduction, real VLA competence, failure mining, extension design, or paper-grade claims.
