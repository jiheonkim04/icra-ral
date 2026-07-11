# Cross-Model Method Readiness Decision

Date: 2026-07-11 KST

Final decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

## Decision Basis

The second backbone and benchmark were selected, but the selected second backbone cannot be run in this pass.

- selected second backbone: `OpenVLA-OFT`
- selected checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- checkpoint size: `14.845` GiB
- selected second benchmark: `LIBERO-PRO`
- large download approval: not provided
- local 16GB inference feasibility: not proven
- lab GPU path: likely feasible but not executed

## State Decisions

- State 1: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`
- State 2: `SECOND_BENCHMARK_READY_AFTER_SECOND_BACKBONE`
- State 3: protocol frozen in `reports/cross_model_failure_manifest.json`
- State 4: not run
- State 5: not run
- State 6: generality not proven
- State 7: focused latest-work comparison recorded, but no method route activated
- State 8: no method specification

## Required Final Choice

Chosen value: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

Rejected values:

- `READY_TO_IMPLEMENT_CROSS_BACKBONE_VLA_METHOD`: no second-backbone or second-benchmark result exists.
- `STABLE_GRASP_ROUTE_KILLED_BY_PRIOR_ART`: not reached; stable-grasp generality is unproven.
- `LONG_HORIZON_ROUTE_KILLED_BY_PRIOR_ART`: not reached; long-horizon generality is unproven.
- `FAILURE_IS_SMOLVLA_SPECIFIC`: not proven because OpenVLA-OFT did not run.
- `FAILURE_NOT_CROSS_BENCHMARK`: not proven because LIBERO-PRO did not run.
- `NO_REVIEW_RESISTANT_METHOD_FOUND`: too strong; the gate is blocked before cross-model evidence.

## Implementation Authorization

Implementation authorized: `false`

Exact next implementation prompt: `NONE`
