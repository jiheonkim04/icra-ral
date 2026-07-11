# Cross-Model Failure Generality

Date: 2026-07-11 KST

Final generality status: `not_proven`

## Mechanisms Kept Separate

Mechanism A: `stable_grasp`

- source: `libero_spatial/task_4`
- physical failure: drawer-contained black bowl extraction fails before transport
- current evidence: SmolVLA-only rerun failures on seeds `20260713` and `20260714`

Mechanism B: `long_horizon_compounding`

- source: `libero_10/task_4`
- physical failure: two-mug, two-plate sequence does not complete
- current evidence: SmolVLA-only rerun failures, strongest on seed `20260715`

These mechanisms remain separate.

## State 6 Gate

| Requirement | Status | Evidence |
| --- | --- | --- |
| Appears in at least two VLA backbones | `not_proven` | OpenVLA-OFT not downloaded or run. |
| Appears in at least two tasks or benchmark conditions | `not_proven` | LIBERO-PRO not run. |
| Repeats across at least three independent reset/evaluation seeds | `not_proven` | SmolVLA visual evidence has at most two clean rerun-failure seeds per mechanism. |
| Materially reduces closed-loop success | `partially_supported` | SmolVLA failures time out, but no second-backbone result exists. |
| Is not an environment or asset bug | `not_proven_cross_model` | SmolVLA env path worked, but OpenVLA-OFT/LIBERO-PRO path not validated. |
| Is not solved by frozen base instead of LoRA | `supported_within_smolvla_only` | Frozen base also failed selected SmolVLA slices. |
| Is not explained only by stochastic outcome flips | `not_proven` | Prior same-identity rerun flipped `8/24` outcomes. |
| Has visually and physically identifiable mechanism | `supported_within_smolvla_only` | Prior video annotations support phase labels. |
| Can be measured without privileged test-time labels | `likely` | Binary success and video phase review are sufficient, but cross-model runner not executed. |
| Has headroom for at least 5 percentage-point success gain | `not_proven` | No method or cross-model baseline result exists. |

## Generality Decision

Neither failure is currently cross-backbone or cross-benchmark. The correct status is blocked before evidence, not killed by negative cross-model results.
