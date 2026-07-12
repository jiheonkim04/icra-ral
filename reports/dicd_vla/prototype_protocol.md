# DICD-VLA Prototype Protocol

Date: 2026-07-12 KST

Protocol status: `FROZEN_BEFORE_IMPLEMENTATION`

## Implementation Steps

1. Add a lightweight DICD adapter module.
2. Add a prototype runner that can call `predict_action_chunk`, postprocess the chunk, train the adapter, run mechanism smoke, and evaluate the preregistered variants.
3. Add unit tests for feature construction, delay-index behavior, gradient flow, checkpoint identity, and no-privileged inference fields.
4. Run mechanism smoke.
5. Only then run Stage A closed-loop evaluation.

## Frozen Variants

- `frozen_smolvla_clean`
- `frozen_smolvla_delay`
- `direct_chunk_index_delay`
- `dicd_no_history_ablation`
- `dicd_full`

## Frozen Tasks And Identities

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Training:

- `20260711`

Smoke:

- `20260712`

Stage A:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

## Required Reports

- `reports/dicd_vla/mechanism_smoke_result.md`
- `reports/dicd_vla/mechanism_smoke_result.json`
- `reports/dicd_vla/stage_a_result.md`
- `reports/dicd_vla/stage_a_result.json`
- `reports/dicd_vla/method_decision.md`

## No-Privileged-Inference Rule

Allowed inference inputs:

- current observation and instruction through official SmolVLA preprocessing;
- current action chunk;
- recent executed actions;
- declared deployment delay;
- current step fraction.

Forbidden inference inputs:

- simulator state;
- task success;
- future observation;
- future action target;
- reset identity;
- held-out outcome labels.
