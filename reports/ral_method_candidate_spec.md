# RA-L Method Candidate Spec

Date: 2026-07-11 KST

Selected method: `NONE`

## Gate Result

No method candidate is specified in this pass.

Reason:

- the hard visual mechanism gate did not pass;
- the strongest spatial failure is supported by only two independent rerun-failure reset seeds;
- the `libero_10/task_4` failure is a different long-horizon/multi-object failure;
- recent 2025-2026 VLA work already covers the obvious generic interventions.

## Explicitly Rejected Candidate Families

- LoRA-only or frozen-base adaptation contribution
- confidence or failure detection head
- action-candidate verification
- 3D/spatial candidate verification
- generic corrective replanning
- adaptive chunking or chunk-boundary smoothing
- progress monitor, progress head, rewind, or recovery
- failure rollouts as negative training data
- adapter routing, prior-preserving experts, or PEFT expert specialization
- task routing or best-seed selection

## Strongest Non-Selected Direction

The strongest non-selected direction would be a mechanism-specific drawer/bowl extraction intervention. It is not specified as a method because the evidence is too narrow and too close to recent correction/verification/chunking work without second-task, second-backbone, or second-benchmark support.

## Implementation Status

Implementation approved: `false`

Exact implementation prompt: `NONE`
